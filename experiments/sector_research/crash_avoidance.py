#!/usr/bin/env python3
"""
PRISM Strategy: Multi-Asset Crash Avoidance + Sector-Stock Selection
=====================================================================
Novel approach: The path to 3+ Sharpe is AVOIDING DRAWDOWNS, not picking winners.

Key insight: A strategy that avoids the worst 20% of market days while staying
invested otherwise can achieve 2-3x the Sharpe of buy-and-hold.

DANGER SIGNALS (composite crash avoidance):
1. Credit stress: HYG (high yield bonds) momentum negative = credit stress
2. Flight to safety: TLT (long bonds) outperforming SPY = risk-off
3. Correlation spike: all sectors moving together = panic
4. Volatility regime: realized vol > 1.5x its 126d average
5. Breadth collapse: <30% of sectors above SMA(50)

When danger composite > threshold: GO TO CASH
When safe: allocate to top sectors/stocks by momentum + quality
"""
import numpy as np
import pandas as pd
from .engine import SECTOR_ETFS, BENCHMARK


def compute_danger_signals(close_df):
    """Compute multi-factor danger composite. Higher = more dangerous."""
    spy = close_df[BENCHMARK]
    signals = pd.DataFrame(index=close_df.index)

    # 1. Credit stress: HYG 21d momentum negative
    if "HYG" in close_df.columns:
        hyg_mom = close_df["HYG"].pct_change(21)
        hyg_z = (hyg_mom - hyg_mom.rolling(252).mean()) / hyg_mom.rolling(252).std().clip(lower=1e-8)
        signals["credit_stress"] = (-hyg_z).clip(-2, 2) / 2  # normalize to ~[-1, 1]
    else:
        signals["credit_stress"] = 0

    # 2. Flight to safety: TLT outperforming SPY over 21d
    if "TLT" in close_df.columns:
        tlt_mom = close_df["TLT"].pct_change(21)
        spy_mom = spy.pct_change(21)
        flight = tlt_mom - spy_mom
        flight_z = (flight - flight.rolling(252).mean()) / flight.rolling(252).std().clip(lower=1e-8)
        signals["flight_to_safety"] = flight_z.clip(-2, 2) / 2
    else:
        signals["flight_to_safety"] = 0

    # 3. Sector correlation spike
    sector_rets = close_df[SECTOR_ETFS].pct_change()
    # Average pairwise correlation over 21d
    def rolling_avg_corr(rets, window=21):
        result = pd.Series(index=rets.index, dtype=float)
        for i in range(window, len(rets)):
            chunk = rets.iloc[i-window:i]
            corr_mat = chunk.corr()
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            avg_c = corr_mat.values[mask].mean()
            result.iloc[i] = avg_c
        return result

    avg_corr = rolling_avg_corr(sector_rets, 21)
    corr_ma = avg_corr.rolling(126).mean()
    corr_std = avg_corr.rolling(126).std().clip(lower=1e-8)
    signals["corr_spike"] = ((avg_corr - corr_ma) / corr_std).clip(-2, 2) / 2

    # 4. Volatility regime
    spy_ret = spy.pct_change()
    vol_21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_126 = spy_ret.rolling(126).std() * np.sqrt(252)
    vol_ratio = vol_21 / vol_126.clip(lower=0.01)
    signals["vol_regime"] = ((vol_ratio - 1.0) * 2).clip(-2, 2) / 2

    # 5. Breadth collapse
    sector_sma50 = close_df[SECTOR_ETFS].rolling(50).mean()
    breadth = (close_df[SECTOR_ETFS] > sector_sma50).sum(axis=1) / len(SECTOR_ETFS)
    signals["breadth_collapse"] = (-(breadth - 0.5) * 2).clip(-1, 1)  # low breadth = danger

    # 6. SPY drawdown depth
    spy_peak = spy.rolling(252, min_periods=21).max()
    spy_dd = (spy - spy_peak) / spy_peak
    signals["drawdown"] = (-spy_dd * 5).clip(0, 2) / 2  # deeper dd = more danger

    return signals


def run_prism(close_df, open_df, data=None, params=None):
    """
    PRISM: Crash avoidance + sector allocation.

    The key: GO TO CASH when danger is high. Invest in sectors when safe.
    """
    p = {
        "danger_threshold": 0.3,   # composite danger score to trigger cash
        "danger_weights": {
            "credit_stress": 0.25,
            "flight_to_safety": 0.20,
            "corr_spike": 0.15,
            "vol_regime": 0.20,
            "breadth_collapse": 0.10,
            "drawdown": 0.10,
        },
        "vol_target": 0.12,
        "max_sectors": 3,
        "mom_lookback": 63,
    }
    if params:
        p.update(params)

    danger_signals = compute_danger_signals(close_df)

    # Composite danger score
    danger = pd.Series(0.0, index=close_df.index)
    for name, weight in p["danger_weights"].items():
        if name in danger_signals.columns:
            danger += danger_signals[name].fillna(0) * weight

    # Sector momentum for allocation
    mom = close_df[SECTOR_ETFS].pct_change(p["mom_lookback"])

    # Vol targeting
    spy_ret = close_df[BENCHMARK].pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.5)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in danger.index or pd.isna(danger.loc[date]):
            continue

        # CRASH AVOIDANCE: go to cash when danger is high
        if danger.loc[date] > p["danger_threshold"]:
            continue

        # When safe: pick top sectors
        row = mom.loc[date].dropna()
        if len(row) < 3:
            continue

        top = row.nlargest(p["max_sectors"]).index
        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5
        base_w = vs / len(top)
        for etf in top:
            weights.loc[date, etf] = min(base_w, 0.5)

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights, danger


def run_prism_with_bonds(close_df, open_df, data=None, params=None):
    """
    PRISM V2: When danger is high, allocate to bonds instead of cash.
    This generates POSITIVE returns during risk-off periods.
    """
    p = {
        "danger_threshold": 0.3,
        "danger_weights": {
            "credit_stress": 0.25,
            "flight_to_safety": 0.20,
            "corr_spike": 0.15,
            "vol_regime": 0.20,
            "breadth_collapse": 0.10,
            "drawdown": 0.10,
        },
        "vol_target": 0.12,
        "max_sectors": 3,
        "mom_lookback": 63,
        "bond_alloc": 0.5,  # fraction to bonds when danger high
    }
    if params:
        p.update(params)

    danger_signals = compute_danger_signals(close_df)
    danger = pd.Series(0.0, index=close_df.index)
    for name, weight in p["danger_weights"].items():
        if name in danger_signals.columns:
            danger += danger_signals[name].fillna(0) * weight

    mom = close_df[SECTOR_ETFS].pct_change(p["mom_lookback"])
    spy_ret = close_df[BENCHMARK].pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.5)

    # Add TLT and IEF to allocation universe
    all_tickers = SECTOR_ETFS + ["TLT", "IEF"]
    weights = pd.DataFrame(0.0, index=close_df.index, columns=all_tickers)

    for date in close_df.index:
        if date not in danger.index or pd.isna(danger.loc[date]):
            continue

        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5

        if danger.loc[date] > p["danger_threshold"]:
            # DANGER: allocate to bonds
            bond_w = p["bond_alloc"] * vs
            if "TLT" in close_df.columns:
                weights.loc[date, "TLT"] = bond_w * 0.6
            if "IEF" in close_df.columns:
                weights.loc[date, "IEF"] = bond_w * 0.4
        else:
            # SAFE: top sector momentum
            row = mom.loc[date].dropna()
            if len(row) < 3:
                continue
            top = row.nlargest(p["max_sectors"]).index
            base_w = vs / len(top)
            for etf in top:
                weights.loc[date, etf] = min(base_w, 0.5)

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights, danger
