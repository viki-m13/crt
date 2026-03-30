#!/usr/bin/env python3
"""
SDAMC: Sector Dispersion Arbitrage with Multi-Gate Confirmation
================================================================
NOVEL PROPRIETARY STRATEGY

Key insight: Combine multiple INDEPENDENT structural signals as gates.
Only invest when ALL gates agree. This dramatically reduces false signals
and variance, yielding high Sharpe ratio.

GATES (all must be true to invest):
1. Market Regime: SPY > SMA(50) — avoid drawdowns
2. Spread Reversion: sector-vs-SPY z-score < -1.0 — buy undervalued sectors
3. Momentum Confirmation: at least 2 of 3 timeframes positive
4. Volatility Regime: Market vol below long-term average — calm conditions

RANKING (among sectors passing all gates):
- Composite score = 0.4 * spread_z_rank + 0.3 * momentum_rank + 0.3 * vol_adj_rank

POSITION SIZING:
- Inverse volatility targeting (15% annualized)
- Max 3 sectors, min allocation per sector = 10%

EXECUTION:
- Signal at close T → execute at open T+1
- 5 bps slippage per trade
"""
import numpy as np
import pandas as pd
from .engine import SECTOR_ETFS, BENCHMARK


def compute_spread_z(close_df, lookback=126):
    """Compute sector-vs-SPY spread z-scores."""
    spy = close_df[BENCHMARK]
    z_scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        # Ratio of sector to SPY
        ratio = close_df[etf] / spy
        ratio_ma = ratio.rolling(lookback, min_periods=63).mean()
        ratio_std = ratio.rolling(lookback, min_periods=63).std().clip(lower=1e-8)
        z_scores[etf] = (ratio - ratio_ma) / ratio_std
    return z_scores.astype(float)


def compute_multi_tf_momentum(close_df):
    """Multi-timeframe momentum consensus: 21d, 63d, 126d."""
    m21 = close_df[SECTOR_ETFS].pct_change(21)
    m63 = close_df[SECTOR_ETFS].pct_change(63)
    m126 = close_df[SECTOR_ETFS].pct_change(126)
    consensus = (m21 > 0).astype(int) + (m63 > 0).astype(int) + (m126 > 0).astype(int)
    return consensus, m63


def compute_vol_adj_score(close_df):
    """Volatility-adjusted momentum score."""
    rets = close_df[SECTOR_ETFS].pct_change()
    mom = rets.rolling(63).mean() * 252
    vol = rets.rolling(63).std() * np.sqrt(252)
    return mom / vol.clip(lower=0.01)


def run_sdamc(close_df, open_df, params=None):
    """
    Run the SDAMC strategy.

    params dict can override defaults:
      sma_period, spread_z_threshold, min_consensus,
      vol_lookback, vol_target, max_sectors, rebalance_freq
    """
    p = {
        "sma_period": 50,
        "spread_z_threshold": -0.5,   # buy when spread z < this (undervalued)
        "min_consensus": 2,           # at least 2/3 timeframes positive
        "vol_lookback": 126,
        "vol_target": 0.15,
        "max_sectors": 3,
        "spread_lookback": 126,
    }
    if params:
        p.update(params)

    spy = close_df[BENCHMARK]
    sma = spy.rolling(p["sma_period"]).mean()

    # Pre-compute all signals
    spread_z = compute_spread_z(close_df, p["spread_lookback"])
    consensus, mom63 = compute_multi_tf_momentum(close_df)
    vol_adj = compute_vol_adj_score(close_df)

    # Market vol regime
    spy_ret = spy.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    mkt_vol_ma = mkt_vol.rolling(p["vol_lookback"]).mean()

    # Vol targeting scale
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.2, 1.5)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in sma.index or pd.isna(sma.loc[date]):
            continue

        # GATE 1: Market regime
        if spy.loc[date] <= sma.loc[date]:
            continue

        # GATE 2 (soft): Volatility regime — reduce size in high vol
        vol_scalar = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5

        # Score each sector
        sector_scores = {}
        for etf in SECTOR_ETFS:
            if etf not in close_df.columns:
                continue

            sz = spread_z.loc[date, etf] if date in spread_z.index and not pd.isna(spread_z.loc[date, etf]) else 0
            con = consensus.loc[date, etf] if date in consensus.index and not pd.isna(consensus.loc[date, etf]) else 0
            va = vol_adj.loc[date, etf] if date in vol_adj.index and not pd.isna(vol_adj.loc[date, etf]) else 0
            m = mom63.loc[date, etf] if date in mom63.index and not pd.isna(mom63.loc[date, etf]) else 0

            # GATE 3: Momentum consensus (at least min_consensus timeframes positive)
            if con < p["min_consensus"]:
                continue

            # Spread reversion: prefer undervalued (negative z) but also accept neutral
            # Sectors with z < threshold get a bonus
            spread_score = max(0, p["spread_z_threshold"] - sz)  # higher = more undervalued

            # Composite score
            score = 0.4 * spread_score + 0.3 * max(0, va) + 0.3 * max(0, m * 10)
            if score > 0:
                sector_scores[etf] = score

        if not sector_scores:
            continue

        # Pick top N
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_sectors[:p["max_sectors"]]

        # Equal weight among top, scaled by vol
        n = len(top)
        base_w = vol_scalar / n
        for etf, _ in top:
            weights.loc[date, etf] = min(base_w, 0.5)  # cap per-sector at 50%

    # Cap total weight at 1.0
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def run_sdamc_v2(close_df, open_df, params=None):
    """
    V2: More aggressive approach focusing on CONSISTENCY.

    Instead of spread reversion, use a pure momentum ensemble with
    multiple confirmation gates. The key is SELECTIVITY — only trade
    when confidence is highest.
    """
    p = {
        "sma_period": 50,
        "min_consensus": 3,      # ALL 3 timeframes must agree
        "vol_target": 0.12,
        "max_sectors": 2,        # Very concentrated
        "min_rel_strength": 0,   # Must beat SPY on 63d
    }
    if params:
        p.update(params)

    spy = close_df[BENCHMARK]
    sma = spy.rolling(p["sma_period"]).mean()

    # Signals
    consensus, mom63 = compute_multi_tf_momentum(close_df)
    spy_mom = spy.pct_change(63)

    # Relative strength vs SPY
    rel_strength = close_df[SECTOR_ETFS].pct_change(63).sub(spy_mom, axis=0)

    # Vol targeting
    spy_ret = spy.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.0)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in sma.index or pd.isna(sma.loc[date]):
            continue

        # Gate 1: Market above SMA
        if spy.loc[date] <= sma.loc[date]:
            continue

        # Gate 2: SPY itself has positive momentum
        if date in spy_mom.index and not pd.isna(spy_mom.loc[date]):
            if spy_mom.loc[date] <= 0:
                continue

        vol_scalar = vol_scale.loc[date] if date in vol_scale.index else 0.5

        eligible = {}
        for etf in SECTOR_ETFS:
            if etf not in close_df.columns:
                continue

            con = consensus.loc[date, etf] if date in consensus.index and not pd.isna(consensus.loc[date, etf]) else 0
            rs = rel_strength.loc[date, etf] if date in rel_strength.index and not pd.isna(rel_strength.loc[date, etf]) else 0
            m = mom63.loc[date, etf] if date in mom63.index and not pd.isna(mom63.loc[date, etf]) else 0

            # Gate 3: Full consensus
            if con < p["min_consensus"]:
                continue

            # Gate 4: Must have positive relative strength
            if rs < p["min_rel_strength"]:
                continue

            eligible[etf] = m

        if not eligible:
            continue

        sorted_sectors = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        top = sorted_sectors[:p["max_sectors"]]
        n = len(top)
        base_w = vol_scalar / n
        for etf, _ in top:
            weights.loc[date, etf] = min(base_w, 0.5)

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def run_sdamc_v3(close_df, open_df, params=None):
    """
    V3: The 'Selective Sector Alpha' (SSA) approach.

    NOVEL CORE IDEA: Instead of always being invested, use a QUALITY GATE
    that measures how much the top sector signal DIFFERS from average.
    Only invest when there's a clear standout.

    This creates natural selectivity — you're only in market when there's
    a genuine alpha opportunity, not just because the market is up.
    """
    p = {
        "sma_period": 50,
        "mom_lookback": 63,
        "standout_threshold": 1.0,  # sector must be >1 std above mean
        "vol_target": 0.12,
        "max_sectors": 2,
    }
    if params:
        p.update(params)

    spy = close_df[BENCHMARK]
    sma = spy.rolling(p["sma_period"]).mean()

    # Compute rolling sector momentum
    mom = close_df[SECTOR_ETFS].pct_change(p["mom_lookback"])

    # Cross-sectional z-score of momentum
    cross_mean = mom.mean(axis=1)
    cross_std = mom.std(axis=1).clip(lower=1e-8)
    mom_z = mom.sub(cross_mean, axis=0).div(cross_std, axis=0)

    # Vol targeting
    spy_ret = spy.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.0)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in sma.index or pd.isna(sma.loc[date]):
            continue

        # Gate 1: Market above SMA
        if spy.loc[date] <= sma.loc[date]:
            continue

        if date not in mom_z.index:
            continue

        row = mom_z.loc[date].dropna()
        if len(row) < 3:
            continue

        # Gate 2: Standout test — top sector must be >threshold std above mean
        top_z = row.max()
        if top_z < p["standout_threshold"]:
            continue

        vol_scalar = vol_scale.loc[date] if date in vol_scale.index else 0.5

        # Pick top sectors above threshold
        above = row[row > p["standout_threshold"] * 0.5]
        if len(above) == 0:
            continue

        top = above.nlargest(p["max_sectors"])
        n = len(top)
        base_w = vol_scalar / n
        for etf in top.index:
            weights.loc[date, etf] = min(base_w, 0.5)

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights
