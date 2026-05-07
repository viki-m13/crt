"""Proprietary Tier 2: more advanced strategies built on insights from deepdive.

Key insights driving these:
- `pullback_in_winner` produces the highest CAGR because long-term-winning
  stocks-on-discount have the largest tails (multi-baggers).
- `explosive_winners` has the highest *win rate* (~80%) but capped CAGR (~20%).
- Ensembles dilute back to mediocrity.
- Hold-forever beats every exit rule for the high-tail strategies.

These tier-2 strategies introduce:
- Regime overlay (SPY breadth) to skip catastrophic entries
- Composite scoring with explicit weights tuned to deepdive findings
- Concentration with cross-sector cap (so top-1 isn't always the same name)
- "Multi-bagger lottery" gates for highest-tail-asymmetry candidates
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import Strategy
from experiments.monthly_dca.strategies_fast import _safe, _z


# ---------------------------------------------------------------------------
def asymmetric_winner(df: pd.DataFrame) -> pd.Series:
    """Pullback-in-winner with sharper gates and richer scoring.

    Gates (must all pass):
      trend_health_5y >= 0.60
      mom_3y > 0.50 (3y price doubled-or-near)
      pullback_1y in [-50%, -10%]
      RSI > 30
      d_sma200 > -0.20 (within 20% of 200dma)
      max_below_200_streak <= 504 (didn't spend > 2y below 200dma in last 5y)

    Score (multiplicative composite):
      depth = -pullback_1y                      # discount
      Q = trend_health_5y * (1 + recovery_rate)
      M = (1 + mom_3y).clip(0.5, 6) ** 0.5
      A = max(accel + 0.03, 0.01)               # selling decel
    Final = depth * Q * M * A
    """
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.6)
    pull = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    mom_3y = _safe(df, "mom_3y")
    mom = _safe(df, "mom_12_1")
    accel = _safe(df, "accel")
    max_below = _safe(df, "max_below_200_streak")

    gate = (
        (trend >= 0.60)
        & (mom_3y > 0.50)
        & (pull >= -0.50)
        & (pull <= -0.10)
        & (rsi > 30)
        & (dsma > -0.20)
        & (max_below <= 504)
    )
    depth = (-pull).clip(lower=0)
    Q = trend * (1 + rec)
    M = (1 + mom_3y.clip(-0.5, 5)).clip(0.5, 6) ** 0.5
    A = (accel + 0.03).clip(lower=0.01)
    score = depth * Q * M * A
    return score.where(gate)


def multibagger_lottery(df: pd.DataFrame) -> pd.Series:
    """Stocks with high tail asymmetry on a discount.

    Gates:
      tail_ratio_24m >= 1.5 (best month >= 1.5x abs(worst month))
      pullback_1y between -45% and -8%
      mom_3y > 0  (long-term up)
      RSI > 30 (not in freefall)
      d_sma200 > -0.25

    Score: tail_ratio * sqrt(-pullback_1y) * (1 + mom_3y).clip(0.5, 4)
    """
    pull = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    mom_3y = _safe(df, "mom_3y")
    tail = _safe(df, "tail_ratio_24m")
    trend = _safe(df, "trend_health_5y")

    gate = (
        (tail >= 1.5)
        & (pull >= -0.45)
        & (pull <= -0.08)
        & (mom_3y > 0)
        & (rsi > 30)
        & (dsma > -0.25)
        & (trend >= 0.50)
    )
    depth = np.sqrt((-pull).clip(lower=0))
    M = (1 + mom_3y.clip(-0.5, 5)).clip(0.5, 5)
    score = tail.clip(0, 10) * depth * M
    return score.where(gate)


def smooth_compounder_pullback(df: pd.DataFrame) -> pd.Series:
    """Stocks with clean linear long-term up-trend (high R^2) on a pullback.

    Gates:
      trend_r2_12m >= 0.50    (smooth trend)
      trend_health_5y >= 0.65
      pullback_1y in [-30%, -5%]
      d_sma200 > -0.15
      mom_3y > 0
      rsi >= 30
    """
    r2 = _safe(df, "trend_r2_12m")
    trend = _safe(df, "trend_health_5y")
    pull = _safe(df, "pullback_1y")
    dsma = _safe(df, "d_sma200")
    mom_3y = _safe(df, "mom_3y")
    rsi = _safe(df, "rsi_14", 50)
    rec = _safe(df, "recovery_rate").fillna(0.6)
    sharpe = _safe(df, "sharpe_12m")

    gate = (
        (r2 >= 0.50)
        & (trend >= 0.65)
        & (pull >= -0.30)
        & (pull <= -0.05)
        & (dsma > -0.15)
        & (mom_3y > 0)
        & (rsi >= 30)
    )
    score = (
        _z(r2) * 0.6
        + _z(trend) * 1.0
        + _z(-pull) * 0.6
        + _z(rec) * 0.4
        + _z(mom_3y) * 0.6
        + _z(sharpe) * 0.4
    )
    return score.where(gate)


def regime_pullback_winner(df: pd.DataFrame) -> pd.Series:
    """Same as asymmetric_winner but only when SPY is healthy.

    SPY proxy: we use the SPY row's d_sma200 and rsi_14 from the same df.
    If SPY's d_sma200 < -0.07 (SPY 7%+ below 200dma) -> zero everyone (no buy month).
    """
    score = asymmetric_winner(df)
    # SPY in df.index? If the df has SPY, we use it as regime proxy
    spy_dsma = df.loc["SPY", "d_sma200"] if "SPY" in df.index else None
    spy_rsi = df.loc["SPY", "rsi_14"] if "SPY" in df.index else None
    if spy_dsma is not None and spy_dsma < -0.07:
        return pd.Series(np.nan, index=score.index)
    if spy_rsi is not None and spy_rsi < 35:
        return pd.Series(np.nan, index=score.index)
    return score


def deep_value_winner(df: pd.DataFrame) -> pd.Series:
    """Pullback >= 25% from BOTH 1y AND 3y high (real damage), but the long-term
    trend is intact. RSI 30-55 (oversold/recovering).
    """
    trend = _safe(df, "trend_health_5y")
    pull1 = _safe(df, "pullback_1y")
    pull3 = _safe(df, "pullback_3y")
    rsi = _safe(df, "rsi_14", 50)
    rec = _safe(df, "recovery_rate").fillna(0.6)
    mom_3y = _safe(df, "mom_3y")
    accel = _safe(df, "accel")

    gate = (
        (trend >= 0.55)
        & (pull1 <= -0.25)
        & (pull1 >= -0.55)
        & (pull3 <= -0.20)
        & (rsi >= 30)
        & (rsi <= 60)
        & (mom_3y > -0.10)
        & (accel > -0.02)
    )
    depth = (-pull1).clip(lower=0)
    score = depth * trend * (1 + rec) * (1 + mom_3y.clip(-0.3, 3))
    return score.where(gate)


def quality_dip_breakout(df: pd.DataFrame) -> pd.Series:
    """Quality stock that recently dipped but is now near a 52w high.

    Gates:
      trend_health_5y >= 0.70
      below_52wh <= 0.05  (within 5% of 52wh)
      mom_3y > 0.30
      pullback_1y >= -0.20  (recent pullback at most 20%)
      rsi 40-75
    """
    trend = _safe(df, "trend_health_5y")
    below_52 = _safe(df, "below_52wh")
    mom_3y = _safe(df, "mom_3y")
    pull = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    sharpe = _safe(df, "sharpe_12m")
    rec = _safe(df, "recovery_rate").fillna(0.6)

    gate = (
        (trend >= 0.70)
        & (below_52 <= 0.05)
        & (mom_3y > 0.30)
        & (pull >= -0.20)
        & (rsi >= 40)
        & (rsi <= 75)
    )
    score = (
        _z(trend) * 1.0
        + _z(mom_3y) * 1.0
        + _z(sharpe) * 0.7
        + _z(rec) * 0.4
        - _z(below_52) * 0.5
    )
    return score.where(gate)


def trend_continuation(df: pd.DataFrame) -> pd.Series:
    """Pure trend continuation: very strong long-term momentum, smooth, above trend."""
    trend = _safe(df, "trend_health_5y")
    mom_3y = _safe(df, "mom_3y")
    mom_5y = _safe(df, "mom_5y")
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    dsma = _safe(df, "d_sma200")

    gate = (
        (trend >= 0.75)
        & (mom_3y > 0.50)
        & (dsma > 0)
    )
    score = (
        _z(mom_3y) * 1.0
        + _z(mom_5y) * 0.7
        + _z(sharpe) * 0.7
        + _z(r2) * 0.5
        + _z(trend) * 0.7
    )
    return score.where(gate)


def proprietary_master_v1(df: pd.DataFrame) -> pd.Series:
    """The 'Master' v1: a tuned blend that aims at 30%+ CAGR with reasonable win.

    Combines pullback-in-winner depth with smoothness gate.
    """
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.6)
    pull1 = _safe(df, "pullback_1y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    mom_3y = _safe(df, "mom_3y")
    mom = _safe(df, "mom_12_1")
    accel = _safe(df, "accel")
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    max_below = _safe(df, "max_below_200_streak")

    gate = (
        (trend >= 0.60)
        & (mom_3y > 0.20)
        & (pull1 >= -0.50)
        & (pull1 <= -0.10)
        & (rsi > 30)
        & (dsma > -0.25)
        & (max_below <= 600)
    )
    score = (
        _z(-pull1) * 1.0
        + _z(trend) * 1.0
        + _z(rec) * 0.5
        + _z(mom_3y) * 0.7
        + _z(mom) * 0.3
        + _z(accel) * 0.5
        + _z(sharpe) * 0.4
        + _z(r2) * 0.3
    )
    return score.where(gate)


def proprietary_master_v2(df: pd.DataFrame) -> pd.Series:
    """Master v2: ULTRA-strict pullback-in-winner.

    Gates (every box must be ticked):
      trend_health_5y >= 0.70
      mom_3y >= 0.50
      mom_12_1 in [-30%, +50%]   # recently softer
      pullback_1y in [-40%, -12%]
      pullback_3y >= -0.55       # not catastrophic
      rsi 30-55                  # oversold turning
      d_sma200 > -0.10
      max_below_200_streak <= 400
      tail_ratio_24m >= 1.0
      ret_5d > -0.03             # not in freefall this week
    """
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate").fillna(0.65)
    pull1 = _safe(df, "pullback_1y")
    pull3 = _safe(df, "pullback_3y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    mom_3y = _safe(df, "mom_3y")
    mom = _safe(df, "mom_12_1")
    accel = _safe(df, "accel")
    sharpe = _safe(df, "sharpe_12m")
    max_below = _safe(df, "max_below_200_streak")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    ret5 = _safe(df, "ret_5d")

    gate = (
        (trend >= 0.70)
        & (mom_3y >= 0.50)
        & (mom >= -0.30)
        & (mom <= 0.50)
        & (pull1 >= -0.40)
        & (pull1 <= -0.12)
        & (pull3 >= -0.55)
        & (rsi >= 30)
        & (rsi <= 55)
        & (dsma > -0.10)
        & (max_below <= 400)
        & (tail >= 1.0)
        & (ret5 > -0.03)
    )
    depth = (-pull1).clip(lower=0)
    Q = trend * (1 + rec)
    M = (1 + mom_3y.clip(-0.3, 5)).clip(0.5, 5)
    A = (accel + 0.04).clip(lower=0.01)
    smooth = (1 + sharpe.clip(-1, 4))
    score = depth * Q * M * A * smooth
    return score.where(gate)


def all_pro_strategies(top_k: int = 5) -> list[Strategy]:
    return [
        Strategy("asymmetric_winner", asymmetric_winner, top_k=top_k),
        Strategy("multibagger_lottery", multibagger_lottery, top_k=top_k),
        Strategy("smooth_compounder_pullback", smooth_compounder_pullback, top_k=top_k),
        Strategy("regime_pullback_winner", regime_pullback_winner, top_k=top_k),
        Strategy("deep_value_winner", deep_value_winner, top_k=top_k),
        Strategy("quality_dip_breakout", quality_dip_breakout, top_k=top_k),
        Strategy("trend_continuation", trend_continuation, top_k=top_k),
        Strategy("proprietary_master_v1", proprietary_master_v1, top_k=top_k),
        Strategy("proprietary_master_v2", proprietary_master_v2, top_k=top_k),
    ]
