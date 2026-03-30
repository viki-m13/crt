#!/usr/bin/env python3
"""
Cross-sectional stock selection with sector context.
Uses the full 100-stock universe for diversification.
Monthly rebalancing to keep transaction costs reasonable.
"""
import numpy as np
import pandas as pd
from .engine import BENCHMARK, SECTOR_ETFS

# Full stock universe (from prepare.py)
STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM",
    "CSCO", "VZ", "ADBE", "CRM", "CMCSA", "PFE", "NFLX", "INTC",
    "ABT", "KO", "PEP", "TMO", "MRK", "ABBV", "COST", "AVGO", "ACN",
    "CVX", "LLY", "MCD", "WMT", "DHR", "TXN", "NEE", "BMY", "QCOM",
    "UNP", "HON", "LOW", "AMGN", "LIN", "RTX",
    "ORCL", "PM", "UPS", "CAT", "GS", "MS", "BLK", "ISRG", "MDT",
    "DE", "ADP", "GILD", "BKNG", "SYK", "MMM", "GE", "CB", "CI",
    "SO", "DUK", "MO", "CL", "ITW", "FIS", "USB", "SCHW", "PNC",
    "CME", "AON", "ICE", "NSC", "EMR", "APD", "SHW", "ETN", "ECL",
    "WM", "ROP", "LRCX", "KLAC", "AMAT", "MCHP", "SNPS", "CDNS",
    "FTNT", "PANW", "NOW", "WDAY",
]


def build_stock_dfs(data):
    """Build aligned close/open DataFrames for all stocks."""
    available = [s for s in STOCKS if s in data]
    close_frames = {s: data[s]["Close"] for s in available}
    open_frames = {s: data[s]["Open"] for s in available if "Open" in data[s].columns}
    close_df = pd.DataFrame(close_frames).dropna(how="all")
    open_df = pd.DataFrame(open_frames).dropna(how="all")
    return close_df, open_df, available


def run_stock_momentum(close_df, open_df, spy_close, params=None):
    """
    Monthly cross-sectional momentum: buy top N stocks by 6-month return.
    Rebalance on 1st trading day of each month.
    """
    p = {
        "lookback": 126,   # 6-month momentum
        "n_stocks": 20,    # hold top 20
        "vol_target": 0.10,
        "sma_gate": True,
        "sma_period": 50,
        "skip_recent": 21,  # skip most recent 21 days (avoids reversal)
    }
    if params:
        p.update(params)

    stocks = close_df.columns.tolist()
    spy_sma = spy_close.rolling(p["sma_period"]).mean()
    spy_ret = spy_close.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.0)

    # Compute momentum: return from T-lookback to T-skip_recent
    if p["skip_recent"] > 0:
        mom = close_df.shift(p["skip_recent"]).pct_change(p["lookback"] - p["skip_recent"])
    else:
        mom = close_df.pct_change(p["lookback"])

    weights = pd.DataFrame(0.0, index=close_df.index, columns=stocks)
    prev_month = None

    for date in close_df.index:
        month = date.month
        if prev_month is not None and month == prev_month:
            # Not a new month — hold previous weights
            if date == close_df.index[0]:
                continue
            prev_idx = close_df.index.get_loc(date) - 1
            if prev_idx >= 0:
                weights.loc[date] = weights.iloc[prev_idx]
            continue

        prev_month = month

        # Gate: SPY > SMA
        if p["sma_gate"]:
            if date not in spy_sma.index or pd.isna(spy_sma.loc[date]):
                continue
            if spy_close.loc[date] <= spy_sma.loc[date]:
                continue

        # Score stocks
        if date not in mom.index:
            continue
        row = mom.loc[date].dropna()
        if len(row) < p["n_stocks"]:
            continue

        top = row.nlargest(p["n_stocks"])
        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5
        w = vs / p["n_stocks"]
        for stock in top.index:
            weights.loc[date, stock] = w

    # Cap
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def run_stock_multifactor(close_df, open_df, spy_close, data=None, params=None):
    """
    Multi-factor stock selection:
    1. Momentum (6-month, skip last month)
    2. Low volatility (lower vol = higher score)
    3. Quality (momentum persistence)

    Rebalance monthly, hold top 15-20.
    """
    p = {
        "n_stocks": 15,
        "vol_target": 0.08,
        "sma_gate": True,
        "sma_period": 50,
        "mom_weight": 0.5,
        "lowvol_weight": 0.25,
        "quality_weight": 0.25,
    }
    if params:
        p.update(params)

    stocks = close_df.columns.tolist()
    rets = close_df.pct_change()

    # Factor 1: Momentum (skip recent month)
    mom = close_df.shift(21).pct_change(105)  # 5-month, skip 1 month
    mom_rank = mom.rank(axis=1, pct=True)

    # Factor 2: Low volatility (inverse of 63d vol)
    vol = rets.rolling(63).std() * np.sqrt(252)
    lowvol_rank = (-vol).rank(axis=1, pct=True)

    # Factor 3: Quality (momentum persistence: % of 21d windows with positive return)
    pos_count = (rets.rolling(21).sum() > 0).rolling(126).mean()
    quality_rank = pos_count.rank(axis=1, pct=True)

    # Composite
    composite = (p["mom_weight"] * mom_rank +
                 p["lowvol_weight"] * lowvol_rank +
                 p["quality_weight"] * quality_rank)

    spy_sma = spy_close.rolling(p["sma_period"]).mean()
    spy_ret = spy_close.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.0)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=stocks)
    prev_month = None

    for date in close_df.index:
        month = date.month
        if prev_month is not None and month == prev_month:
            prev_idx = close_df.index.get_loc(date) - 1
            if prev_idx >= 0:
                weights.loc[date] = weights.iloc[prev_idx]
            continue
        prev_month = month

        if p["sma_gate"]:
            if date not in spy_sma.index or pd.isna(spy_sma.loc[date]):
                continue
            if spy_close.loc[date] <= spy_sma.loc[date]:
                continue

        if date not in composite.index:
            continue
        row = composite.loc[date].dropna()
        if len(row) < p["n_stocks"]:
            continue

        top = row.nlargest(p["n_stocks"])
        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5

        # Inverse vol weighting among top stocks
        top_vol = vol.loc[date, top.index] if date in vol.index else pd.Series(0.2, index=top.index)
        inv_vol = 1.0 / top_vol.clip(lower=0.05)
        inv_vol_w = inv_vol / inv_vol.sum()
        for stock in top.index:
            weights.loc[date, stock] = inv_vol_w.get(stock, 1/p["n_stocks"]) * vs

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def run_stock_sector_hybrid(close_df, open_df, spy_close,
                            sector_close_df, data=None, params=None):
    """
    HYBRID: Use sector signals to select sectors, then pick stocks within.

    1. Rank sectors by momentum
    2. Within top 3 sectors, rank stocks by multi-factor score
    3. Hold top 5 stocks per sector = 15 total
    4. Monthly rebalance, vol targeting
    """
    p = {
        "n_sectors": 3,
        "n_stocks_per_sector": 5,
        "vol_target": 0.08,
        "sma_gate": True,
        "sma_period": 50,
        "sector_mom_lookback": 63,
    }
    if params:
        p.update(params)

    # Map stocks to sectors (approximate via correlation)
    # Use predefined mapping
    SECTOR_MAP = {
        "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "INTC",
                 "TXN", "QCOM", "AMAT", "LRCX", "KLAC", "MCHP", "SNPS", "CDNS",
                 "FTNT", "PANW", "NOW", "WDAY", "ORCL", "ACN"],
        "XLF": ["JPM", "BAC", "GS", "MS", "BLK", "SCHW", "USB", "PNC",
                 "CME", "ICE", "AON", "CB", "V", "MA"],
        "XLE": ["XOM", "CVX"],
        "XLV": ["JNJ", "UNH", "PFE", "ABT", "TMO", "MRK", "ABBV", "LLY",
                 "AMGN", "GILD", "ISRG", "SYK", "MDT", "CI", "BMY"],
        "XLI": ["UNP", "HON", "CAT", "DE", "RTX", "GE", "ETN", "EMR",
                 "NSC", "ITW", "WM", "ROP", "MMM", "UPS", "ADP"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "LOW", "NFLX", "BKNG", "DIS"],
        "XLP": ["PG", "KO", "PEP", "COST", "WMT", "CL", "PM", "MO"],
        "XLU": ["NEE", "DUK", "SO"],
        "XLB": ["LIN", "APD", "SHW", "ECL"],
        "XLC": ["GOOGL", "META", "CMCSA", "VZ", "DIS"],
        "XLRE": [],
    }

    stocks = close_df.columns.tolist()
    rets = close_df.pct_change()

    # Sector momentum
    sector_mom = sector_close_df[SECTOR_ETFS].pct_change(p["sector_mom_lookback"])

    # Stock scores (momentum + low vol)
    stock_mom = close_df.shift(21).pct_change(105)
    stock_vol = rets.rolling(63).std()
    stock_score = stock_mom.rank(axis=1, pct=True) + (-stock_vol).rank(axis=1, pct=True)

    spy_sma = spy_close.rolling(p["sma_period"]).mean()
    spy_ret = spy_close.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.0)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=stocks)
    prev_month = None

    for date in close_df.index:
        month = date.month
        if prev_month is not None and month == prev_month:
            prev_idx = close_df.index.get_loc(date) - 1
            if prev_idx >= 0:
                weights.loc[date] = weights.iloc[prev_idx]
            continue
        prev_month = month

        if p["sma_gate"]:
            if date not in spy_sma.index or pd.isna(spy_sma.loc[date]):
                continue
            if spy_close.loc[date] <= spy_sma.loc[date]:
                continue

        if date not in sector_mom.index:
            continue

        # Top sectors
        sec_row = sector_mom.loc[date].dropna()
        if len(sec_row) < p["n_sectors"]:
            continue
        top_sectors = sec_row.nlargest(p["n_sectors"]).index.tolist()

        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5

        # Within each top sector, pick top stocks
        selected = []
        for sector in top_sectors:
            sector_stocks = [s for s in SECTOR_MAP.get(sector, []) if s in stocks]
            if not sector_stocks or date not in stock_score.index:
                continue
            scores = stock_score.loc[date, [s for s in sector_stocks if s in stock_score.columns]].dropna()
            if len(scores) == 0:
                continue
            top_stocks = scores.nlargest(min(p["n_stocks_per_sector"], len(scores)))
            selected.extend(top_stocks.index.tolist())

        if not selected:
            continue

        w = vs / len(selected)
        for stock in selected:
            weights.loc[date, stock] = w

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights
