"""Strategies that operate on cached feature DataFrames (fast).

A strategy here is a pure function: pd.DataFrame (indexed by ticker, columns =
features) -> pd.Series (score per ticker). Higher score = more preferred.

We always defensively fillna and reindex to feats.index so that the engine
never crashes on missing fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import Strategy


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.median()
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _safe(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------
def quality_pullback(df: pd.DataFrame) -> pd.Series:
    pull = _safe(df, "pullback_1y")
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.5)
    accel = _safe(df, "accel")
    mom = _safe(df, "mom_12_1")
    rsi = _safe(df, "rsi_14", 50)

    s = (
        _z(-pull) * 1.0
        + _z(trend) * 1.5
        + _z(rec) * 0.7
        + _z(accel) * 0.6
        + _z(mom) * 0.4
    )
    s = s.where((rsi > 30) | (accel > 0))
    return s


def dual_momentum(df: pd.DataFrame) -> pd.Series:
    mom = _safe(df, "mom_12_1")
    dsma = _safe(df, "d_sma200")
    score = _z(mom)
    return score.where((dsma > 0) & (mom > 0))


def low_vol_trend(df: pd.DataFrame) -> pd.Series:
    vol = _safe(df, "vol_1y")
    trend = _safe(df, "trend_health_5y")
    sharpe = _safe(df, "sharpe_1y")
    dsma = _safe(df, "d_sma200")
    score = _z(-vol) + _z(trend) + 0.5 * _z(sharpe)
    return score.where(dsma > 0)


def pullback_in_winner(df: pd.DataFrame) -> pd.Series:
    pull = _safe(df, "pullback_1y")
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.5)
    accel = _safe(df, "accel")
    mom = _safe(df, "mom_12_1")
    rsi = _safe(df, "rsi_14", 50)
    score = _z(-pull) + _z(trend) + 0.6 * _z(rec) + 0.4 * _z(accel) + 0.5 * _z(mom)
    mask = (trend > 0.55) & (pull < -0.15) & (rsi > 30)
    return score.where(mask)


def winner_only(df: pd.DataFrame) -> pd.Series:
    mom = _safe(df, "mom_12_1")
    trend = _safe(df, "trend_health_5y")
    vol = _safe(df, "vol_1y")
    dsma = _safe(df, "d_sma200")
    score = _z(mom) + 0.6 * _z(trend) - 0.5 * _z(vol)
    return score.where((dsma > 0) & (mom > 0))


def explosive_winners(df: pd.DataFrame) -> pd.Series:
    mom = _safe(df, "mom_12_1")
    accel = _safe(df, "accel")
    sharpe = _safe(df, "sharpe_1y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    score = _z(mom) + 0.6 * _z(accel) + 0.4 * _z(sharpe)
    return score.where((dsma > 0) & (mom > 0.10) & (rsi < 80))


def min_dd_compounders(df: pd.DataFrame) -> pd.Series:
    trend = _safe(df, "trend_health_5y")
    vol = _safe(df, "vol_1y")
    rec = _safe(df, "recovery_rate").fillna(0.5)
    dsma = _safe(df, "d_sma200")
    score = _z(trend) * 1.5 + _z(-vol) + 0.5 * _z(rec)
    return score.where((trend >= 0.80) & (dsma > -0.10))


def proprietary_v1(df: pd.DataFrame) -> pd.Series:
    """Resilient compounders on sale: Q * discount * reignition."""
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.5)
    vol = _safe(df, "vol_1y").clip(0.1, 1.0)
    pull = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    accel = _safe(df, "accel")
    mom = _safe(df, "mom_12_1")
    Q = trend * rec * (1 - vol) * (1 + np.tanh(mom))
    discount = np.tanh(-pull / 0.20)
    reign = ((rsi > 35) & (accel > -0.02)).astype(float)
    return (Q * discount * (0.3 + 0.7 * reign)).astype(float)


def proprietary_v2(df: pd.DataFrame) -> pd.Series:
    """Hard quality gate + depth-of-discount blend."""
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.6)
    sma_above = _safe(df, "sma50_above_200")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    pull = _safe(df, "pullback_1y")
    accel = _safe(df, "accel")
    mom = _safe(df, "mom_12_1")
    gate = (
        (trend >= 0.65)
        & (rec >= 0.55)
        & (sma_above > 0)
        & (dsma > -0.05)
        & (rsi > 35)
        & (pull >= -0.45)
        & (pull <= -0.05)
    )
    score = 0.5 * (-pull) + 0.3 * rec + 0.4 * mom + 0.3 * accel
    return score.where(gate)


def proprietary_v3(df: pd.DataFrame) -> pd.Series:
    """Proprietary 'Buy The Strong Dip' v3.

    Idea: combine three orthogonal signals
      A. Long-term trend strength: trend_health_5y >= 0.7 (compounders)
      B. Short-term over-reaction: 21d return < -8% (panic) AND 5d return > 21d return (selling decel)
      C. Regime ok: SPY breadth (we approximate via accel of SPY-like behavior on the candidate)

    Score = depth-of-21d-drop * sqrt(trend_health_5y) * (1 + recovery_rate)
    """
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.5)
    ret21 = _safe(df, "ret_21d")
    ret5 = _safe(df, "ret_5d")
    pull = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    score = (-ret21).clip(lower=0) * np.sqrt(trend.clip(lower=0)) * (1 + rec)
    gate = (
        (trend >= 0.65)
        & (ret21 < -0.05)
        & (ret5 > ret21)
        & (rsi > 25)
        & (pull < -0.10)
        & (dsma > -0.20)
    )
    return score.where(gate)


def proprietary_v4(df: pd.DataFrame) -> pd.Series:
    """Proprietary 'Smooth Compounder Drawdown' v4.

    The idea: we want a stock that almost never gets hurt (low DD history) but
    is currently mildly hurt. These tend to recover fast.
    """
    trend = _safe(df, "trend_health_5y")
    vol = _safe(df, "vol_1y")
    sharpe = _safe(df, "sharpe_1y")
    pull = _safe(df, "pullback_1y")
    rec = _safe(df, "recovery_rate").fillna(0.6)
    rsi = _safe(df, "rsi_14", 50)
    accel = _safe(df, "accel")

    # Rank by Q (composite)
    Q = _z(trend) * 1.0 + _z(-vol) * 0.7 + _z(sharpe) * 0.7 + _z(rec) * 0.5
    # Score depth of pullback
    discount = np.tanh(-pull / 0.15)
    # Don't catch falling knives
    not_freefall = ((rsi > 30) & (accel > -0.02)).astype(float)
    score = Q * discount * (0.3 + 0.7 * not_freefall)
    # Hard gate for very-low quality
    gate = trend >= 0.70
    return score.where(gate)


def proprietary_v5(df: pd.DataFrame) -> pd.Series:
    """Proprietary 'Maximum Tail Asymmetry' v5.

    Use price-only proxies for asymmetry: stocks with frequent multi-year
    new-highs and shallow drawdowns.
    """
    trend = _safe(df, "trend_health_5y")
    pull = _safe(df, "pullback_1y")
    pull_all = _safe(df, "pullback_all")
    rec = _safe(df, "recovery_rate").fillna(0.6)
    sharpe = _safe(df, "sharpe_1y")
    mom = _safe(df, "mom_12_1")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")

    # Tail-asymmetry composite
    asym = trend * rec * (1 + sharpe.clip(-1, 5))
    # Want some discount (5-25% pullback) — not extreme freefall
    sweet = ((-pull >= 0.05) & (-pull <= 0.30)).astype(float)
    # Trend filter
    gate = (trend >= 0.70) & (dsma > -0.15) & (rsi > 35) & (pull_all > -0.5)
    score = asym * sweet
    return score.where(gate)


def proprietary_v6(df: pd.DataFrame) -> pd.Series:
    """Proprietary 'Deep Quality on 30%+ Sale' v6.

    Ultra-strict: stock must have:
      - trend_health_5y >= 0.75
      - recovery_rate >= 0.7 (or NaN if too few priors -> ok)
      - currently 20-50% off 1y high
      - RSI 30-55 (oversold but recovering)
      - 5d return > -2% (selling has stopped)
      - momentum 12-1 still positive (long-term winner)

    Score: sqrt(pullback_depth) * trend_health * recovery_rate * (1 + momentum)
    """
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate")
    rec_filled = rec.fillna(0.7)
    pull = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    ret5 = _safe(df, "ret_5d")
    mom = _safe(df, "mom_12_1")

    gate = (
        (trend >= 0.70)
        & ((rec.isna()) | (rec >= 0.6))
        & (pull >= -0.50)
        & (pull <= -0.15)
        & (rsi >= 30)
        & (rsi <= 55)
        & (ret5 > -0.02)
        & (mom > -0.10)
    )
    score = (
        np.sqrt((-pull).clip(lower=0))
        * trend
        * rec_filled
        * (1 + mom.clip(lower=-0.5))
    )
    return score.where(gate)


def proprietary_v7(df: pd.DataFrame) -> pd.Series:
    """Proprietary 'Compounder Continuation' v7.

    No pullback bias. Just buy the stocks that compound the most cleanly.
    Score: trend_health_5y * sharpe_1y * (1 + mom_12_1)
    Gate: trend_health >= 0.75, dsma200 > 0
    """
    trend = _safe(df, "trend_health_5y")
    sharpe = _safe(df, "sharpe_1y")
    mom = _safe(df, "mom_12_1")
    dsma = _safe(df, "d_sma200")
    rec = _safe(df, "recovery_rate").fillna(0.7)
    score = trend * sharpe.clip(-2, 5) * (1 + mom.clip(lower=-0.5)) * (0.5 + 0.5 * rec)
    gate = (trend >= 0.75) & (dsma > 0)
    return score.where(gate)


def proprietary_v8(df: pd.DataFrame) -> pd.Series:
    """Proprietary 'Slope-Up Discount' v8.

    Combine: long-term trend + recent reversal + winner-on-pullback.
    """
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.65)
    pull = _safe(df, "pullback_1y")
    accel = _safe(df, "accel")
    rsi = _safe(df, "rsi_14", 50)
    mom = _safe(df, "mom_12_1")
    sharpe = _safe(df, "sharpe_1y")
    dsma = _safe(df, "d_sma200")

    score = (
        _z(trend) * 1.2
        + _z(rec) * 0.6
        + _z(-pull) * 0.8     # discount
        + _z(accel) * 0.5
        + _z(sharpe) * 0.6
        + _z(mom) * 0.4
    )
    gate = (
        (trend >= 0.65)
        & (rsi >= 32)
        & (rsi <= 75)
        & (dsma > -0.20)
        & (pull < -0.05)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def all_strategies(top_k: int = 5) -> list[Strategy]:
    return [
        Strategy("quality_pullback", quality_pullback, top_k=top_k),
        Strategy("dual_momentum", dual_momentum, top_k=top_k),
        Strategy("low_vol_trend", low_vol_trend, top_k=top_k),
        Strategy("pullback_in_winner", pullback_in_winner, top_k=top_k),
        Strategy("winner_only", winner_only, top_k=top_k),
        Strategy("explosive_winners", explosive_winners, top_k=top_k),
        Strategy("min_dd_compounders", min_dd_compounders, top_k=top_k),
        Strategy("proprietary_v1", proprietary_v1, top_k=top_k),
        Strategy("proprietary_v2", proprietary_v2, top_k=top_k),
        Strategy("proprietary_v3", proprietary_v3, top_k=top_k),
        Strategy("proprietary_v4", proprietary_v4, top_k=top_k),
        Strategy("proprietary_v5", proprietary_v5, top_k=top_k),
        Strategy("proprietary_v6", proprietary_v6, top_k=top_k),
        Strategy("proprietary_v7", proprietary_v7, top_k=top_k),
        Strategy("proprietary_v8", proprietary_v8, top_k=top_k),
    ]
