#!/usr/bin/env python3
"""
Refined multifactor stock strategy — pushing for higher Sharpe.

Key improvements over v1:
1. Weekly rebalancing (captures more information)
2. 5 factors: momentum, low_vol, quality, reversal, relative_strength
3. Regime-adaptive vol target (reduce in danger, increase in calm)
4. Better diversification (25-30 stocks)
5. Turnover penalty in scoring (prefer holding existing positions)
"""
import numpy as np
import pandas as pd
from .engine import BENCHMARK, SECTOR_ETFS
from .stock_strategy import STOCKS


def compute_factors(close_df, spy_close):
    """Compute all alpha factors. All use only past data."""
    rets = close_df.pct_change()
    stocks = close_df.columns.tolist()

    factors = {}

    # F1: Momentum (6-month, skip most recent month — avoids short-term reversal)
    factors["momentum"] = close_df.shift(21).pct_change(105)

    # F2: Low Volatility (inverse 63d realized vol)
    vol63 = rets.rolling(63).std() * np.sqrt(252)
    factors["low_vol"] = -vol63

    # F3: Quality (momentum persistence: fraction of 21d windows with positive ret)
    rolling_21d_ret = rets.rolling(21).sum()
    factors["quality"] = (rolling_21d_ret > 0).rolling(126).mean()

    # F4: Short-term Reversal (buy recent losers — 5-day reversal)
    factors["reversal"] = -close_df.pct_change(5)

    # F5: Relative Strength vs SPY
    stock_ret_63 = close_df.pct_change(63)
    spy_ret_63 = spy_close.pct_change(63)
    factors["rel_strength"] = stock_ret_63.sub(spy_ret_63, axis=0)

    # F6: Volatility compression (low short vol / long vol = squeeze)
    vol5 = rets.rolling(5).std()
    vol63_raw = rets.rolling(63).std()
    factors["vol_compress"] = -(vol5 / vol63_raw.clip(lower=1e-8))

    return factors


def rank_normalize(df):
    """Cross-sectional percentile rank each day."""
    return df.rank(axis=1, pct=True)


def compute_danger_score(spy_close, sector_close_df=None):
    """Quick market danger score for regime detection."""
    spy_ret = spy_close.pct_change()

    # Realized vol vs its average
    vol21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vol126 = spy_ret.rolling(126).std() * np.sqrt(252)
    vol_danger = ((vol21 / vol126.clip(lower=0.01)) - 1).clip(0, 2) / 2

    # SPY drawdown
    peak = spy_close.rolling(252, min_periods=21).max()
    dd = (spy_close - peak) / peak
    dd_danger = (-dd * 5).clip(0, 2) / 2

    # SPY trend (below SMA = danger)
    sma50 = spy_close.rolling(50).mean()
    trend_danger = (spy_close < sma50).astype(float) * 0.5

    # Composite
    danger = 0.4 * vol_danger + 0.3 * dd_danger + 0.3 * trend_danger
    return danger.fillna(0)


def run_refined_multifactor(close_df, open_df, spy_close,
                            sector_close_df=None, params=None):
    """
    Refined multi-factor stock selection.

    Weekly rebalancing on Fridays (or last trading day of week).
    Regime-adaptive vol targeting.
    """
    p = {
        "n_stocks": 25,
        "base_vol_target": 0.12,
        "danger_vol_target": 0.05,
        "danger_threshold": 0.4,
        "rebalance_freq": "weekly",  # "weekly" or "monthly"
        "factor_weights": {
            "momentum": 0.30,
            "low_vol": 0.15,
            "quality": 0.20,
            "reversal": 0.10,
            "rel_strength": 0.15,
            "vol_compress": 0.10,
        },
        "turnover_penalty": 0.0,  # penalty for switching positions
    }
    if params:
        p.update(params)

    stocks = close_df.columns.tolist()
    factors = compute_factors(close_df, spy_close)
    danger = compute_danger_score(spy_close, sector_close_df)

    # Rank-normalize each factor
    ranked = {name: rank_normalize(df) for name, df in factors.items()}

    # Composite score
    composite = pd.DataFrame(0.0, index=close_df.index, columns=stocks)
    for name, weight in p["factor_weights"].items():
        if name in ranked:
            composite += ranked[name].fillna(0.5) * weight

    # Vol targeting
    spy_ret = spy_close.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=stocks)
    prev_rebal_week = None
    prev_rebal_month = None

    for i, date in enumerate(close_df.index):
        # Check if rebalance day
        is_rebal = False
        if p["rebalance_freq"] == "weekly":
            week = date.isocalendar()[1]
            if prev_rebal_week is None or week != prev_rebal_week:
                is_rebal = True
                prev_rebal_week = week
        elif p["rebalance_freq"] == "monthly":
            month = date.month
            if prev_rebal_month is None or month != prev_rebal_month:
                is_rebal = True
                prev_rebal_month = month

        if not is_rebal:
            # Hold previous weights
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        if date not in composite.index:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        # Get composite scores
        scores = composite.loc[date].dropna()
        if len(scores) < p["n_stocks"]:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        # Regime-adaptive vol target
        d = danger.loc[date] if date in danger.index and not pd.isna(danger.loc[date]) else 0
        if d > p["danger_threshold"]:
            vol_target = p["danger_vol_target"]
        else:
            vol_target = p["base_vol_target"]

        mv = mkt_vol.loc[date] if date in mkt_vol.index and not pd.isna(mkt_vol.loc[date]) else 0.15
        vol_scale = min(vol_target / max(mv, 0.05), 1.5)

        # Turnover penalty: boost scores for currently held stocks
        if p["turnover_penalty"] > 0 and i > 0:
            held = weights.iloc[i - 1]
            bonus = (held > 0).astype(float) * p["turnover_penalty"]
            scores = scores.add(bonus, fill_value=0)

        # Select top N stocks
        top = scores.nlargest(p["n_stocks"])

        # Inverse-vol weighting
        rets = close_df.pct_change()
        if date in rets.index:
            stock_vol = rets.loc[:date].tail(63).std()
            top_vol = stock_vol.reindex(top.index).clip(lower=0.005)
            inv_vol = 1.0 / top_vol
            w = inv_vol / inv_vol.sum() * vol_scale
        else:
            w = pd.Series(vol_scale / p["n_stocks"], index=top.index)

        weights.loc[date, top.index] = w.values

    # Cap total at 1.0
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights
