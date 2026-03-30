#!/usr/bin/env python3
"""
Sector Signal Ensemble Strategy
=================================
Blend the 4 most CONSISTENT signals (positive across all periods):
1. Spread Reversion (sector-vs-SPY undervaluation)
2. Volume-Price Accumulation
3. Information Ratio (alpha vs SPY)
4. Multi-TF Momentum Consensus

Key innovations:
- Signal blending (not gating) preserves opportunities
- Cross-signal agreement weighting (high agreement = larger position)
- Aggressive volatility targeting (low vol = high Sharpe)
- Continuous allocation (no binary in/out)
"""
import numpy as np
import pandas as pd
from .engine import SECTOR_ETFS, BENCHMARK


def _rank_pct(series):
    """Cross-sectional rank as percentile."""
    return series.rank(pct=True)


def compute_all_scores(close_df, data=None):
    """Compute normalized scores from each signal component."""
    spy = close_df[BENCHMARK]
    spy_ret = spy.pct_change()

    scores = {}

    # 1. SPREAD REVERSION SCORE
    spread_scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        ratio = close_df[etf] / spy
        ratio_ma = ratio.rolling(126, min_periods=63).mean()
        ratio_std = ratio.rolling(126, min_periods=63).std().clip(lower=1e-8)
        z = (ratio - ratio_ma) / ratio_std
        spread_scores[etf] = -z  # negative z = undervalued = good
    scores["spread"] = spread_scores

    # 2. VOLUME-PRICE SCORE
    vp_scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if data and etf in data and "Volume" in data[etf].columns:
            vol = data[etf]["Volume"].reindex(close_df.index)
            vol_ma = vol.rolling(20).mean().clip(lower=1)
            vol_ratio = vol / vol_ma
            ret = close_df[etf].pct_change()
            vp_scores[etf] = (ret * vol_ratio).rolling(21).mean()
        else:
            vp_scores[etf] = close_df[etf].pct_change().rolling(21).mean()
    scores["vp"] = vp_scores

    # 3. INFORMATION RATIO SCORE
    ir_scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        etf_ret = close_df[etf].pct_change()
        cov = etf_ret.rolling(63).cov(spy_ret)
        var = spy_ret.rolling(63).var().clip(lower=1e-10)
        beta = cov / var
        alpha_daily = etf_ret - beta * spy_ret
        ir_scores[etf] = alpha_daily.rolling(63).mean() / alpha_daily.rolling(63).std().clip(lower=1e-8)
    scores["ir"] = ir_scores

    # 4. MULTI-TF MOMENTUM CONSENSUS SCORE
    m21 = close_df[SECTOR_ETFS].pct_change(21)
    m63 = close_df[SECTOR_ETFS].pct_change(63)
    m126 = close_df[SECTOR_ETFS].pct_change(126)
    consensus = (m21 > 0).astype(float) + (m63 > 0).astype(float) + (m126 > 0).astype(float)
    scores["consensus"] = consensus / 3.0  # normalize to [0, 1]

    # 5. MOMENTUM SCORE (raw 63d momentum, cross-sectionally ranked)
    scores["momentum"] = m63

    return scores


def run_ensemble(close_df, open_df, data=None, params=None):
    """
    Signal ensemble with vol targeting.

    params:
        weights: dict of signal weights (must sum to 1)
        vol_target: target annualized volatility
        max_sectors: max number of sectors to hold
        min_score_pct: minimum percentile score to be eligible
    """
    p = {
        "signal_weights": {
            "spread": 0.25,
            "vp": 0.15,
            "ir": 0.25,
            "consensus": 0.20,
            "momentum": 0.15,
        },
        "vol_target": 0.10,
        "max_sectors": 3,
        "min_score_pct": 0.5,  # must be in top 50% composite
        "agreement_bonus": True,  # boost when multiple signals agree
    }
    if params:
        p.update(params)

    scores = compute_all_scores(close_df, data)

    # Cross-sectional rank-normalize each score
    ranked = {}
    for name, df in scores.items():
        ranked[name] = df.rank(axis=1, pct=True)

    # Blend scores
    composite = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for name, weight in p["signal_weights"].items():
        if name in ranked:
            composite += ranked[name].fillna(0.5) * weight

    # Agreement bonus: if a sector ranks in top 3 on 3+ signals, boost it
    if p["agreement_bonus"]:
        top_3_count = pd.DataFrame(0, index=close_df.index, columns=SECTOR_ETFS)
        for name, df in ranked.items():
            top_3_count += (df > 0.7).astype(int)  # top 30% on each signal
        # Boost: multiply by (1 + 0.2 * count) for sectors with multi-signal agreement
        agreement_mult = 1.0 + 0.15 * top_3_count.clip(upper=4)
        composite = composite * agreement_mult

    # Vol targeting
    spy_ret = close_df[BENCHMARK].pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.5)

    # Build weights
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in composite.index:
            continue
        row = composite.loc[date].dropna()
        if len(row) < 3:
            continue

        # Filter by minimum score
        threshold = row.quantile(1.0 - p["min_score_pct"])
        eligible = row[row >= threshold]
        if len(eligible) == 0:
            continue

        top = eligible.nlargest(p["max_sectors"])
        # Weight proportional to score
        w = top / top.sum()
        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5
        w = w * vs

        weights.loc[date, top.index] = w.values

    # Cap
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def run_ensemble_v2(close_df, open_df, data=None, params=None):
    """
    V2: Ensemble with regime-dependent signal weighting.

    In bull market: weight momentum and consensus higher
    In bear/volatile: weight spread reversion and IR higher (defensive alpha)
    """
    p = {
        "vol_target": 0.08,
        "max_sectors": 3,
        "sma_period": 50,
        "min_score_pct": 0.5,
    }
    if params:
        p.update(params)

    spy = close_df[BENCHMARK]
    sma = spy.rolling(p["sma_period"]).mean()
    scores = compute_all_scores(close_df, data)

    ranked = {}
    for name, df in scores.items():
        ranked[name] = df.rank(axis=1, pct=True)

    # Vol targeting
    spy_ret = spy.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.0)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in sma.index or pd.isna(sma.loc[date]):
            continue

        # Regime-dependent weights
        bull = spy.loc[date] > sma.loc[date]
        if bull:
            sw = {"spread": 0.15, "vp": 0.15, "ir": 0.15, "consensus": 0.30, "momentum": 0.25}
        else:
            sw = {"spread": 0.35, "vp": 0.10, "ir": 0.35, "consensus": 0.10, "momentum": 0.10}

        composite_val = pd.Series(0.0, index=SECTOR_ETFS)
        for name, weight in sw.items():
            if name in ranked and date in ranked[name].index:
                composite_val += ranked[name].loc[date].fillna(0.5) * weight

        eligible = composite_val.dropna()
        if len(eligible) < 3:
            continue

        threshold = eligible.quantile(1.0 - p["min_score_pct"])
        above = eligible[eligible >= threshold]
        if len(above) == 0:
            continue

        top = above.nlargest(p["max_sectors"])
        w = top / top.sum()
        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 0.5

        # In bear: reduce overall allocation
        if not bull:
            vs *= 0.5

        w = w * vs
        weights.loc[date, top.index] = w.values

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights
