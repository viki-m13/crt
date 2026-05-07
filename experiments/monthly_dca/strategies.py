"""A library of candidate strategies, scored on FeaturePack."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.backtester import FeaturePack, Strategy


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.median()
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - mu) / sd


def score_quality_pullback(p: FeaturePack) -> pd.Series:
    """The "rebound + quality" thesis from PLAN.md, made explicit:

    + deeper pullback from 1y high  (pullback_1y is negative -> we want it more negative)
    + healthy long-term trend (trend_health_5y, frac days above 200dma)
    + recovery track record (recovery_rate)
    + selling deceleration (accel positive)
    - in active freefall (RSI < 30 AND falling) -- we discount it
    """
    f = p.f
    parts = {}
    parts["pullback"] = -f["pullback_1y"]                              # bigger pullback -> bigger score
    parts["trend"] = f["trend_health_5y"]
    parts["recovery"] = f["recovery_rate"].fillna(f["recovery_rate"].median())
    parts["accel"] = f["accel"]
    parts["mom_12_1"] = f.get("mom_12_1", pd.Series(dtype=float))     # winner stocks
    # Penalize active freefall
    rsi = f["rsi_14"]
    not_freefall = ((rsi > 30) | (f["accel"] > 0)).astype(float)
    s = (
        _z(parts["pullback"]) * 1.0
        + _z(parts["trend"]) * 1.5
        + _z(parts["recovery"]) * 0.7
        + _z(parts["accel"]) * 0.6
        + _z(parts["mom_12_1"]) * 0.4 if not parts["mom_12_1"].empty else 0
    )
    if isinstance(s, int):  # mom missing
        s = (
            _z(parts["pullback"]) * 1.0
            + _z(parts["trend"]) * 1.5
            + _z(parts["recovery"]) * 0.7
            + _z(parts["accel"]) * 0.6
        )
    s = s * not_freefall
    return s.dropna()


def score_dual_momentum(p: FeaturePack) -> pd.Series:
    """Classic dual momentum: high 12-1 momentum + above SPY."""
    f = p.f
    mom = f.get("mom_12_1")
    if mom is None or mom.empty:
        return pd.Series(dtype=float)
    excess = f.get("excess_5y_logret", pd.Series(dtype=float))
    score = _z(mom)
    if not excess.empty:
        score = score + 0.5 * _z(excess)
    # Filter: only include those above 200d sma
    mask = (f["d_sma200"] > 0) & (mom > 0)
    return score.where(mask).dropna()


def score_low_vol_trend(p: FeaturePack) -> pd.Series:
    f = p.f
    score = _z(-f["vol_1y"]) + _z(f["trend_health_5y"]) + 0.5 * _z(f["sharpe_1y"])
    mask = f["d_sma200"] > 0
    return score.where(mask).dropna()


def score_pullback_in_winner(p: FeaturePack) -> pd.Series:
    """The "fork in the road": stock has crushing long-term trend AND meaningful pullback now."""
    f = p.f
    # Long-term winner: 5y excess return positive AND trend_health high
    excess = f.get("excess_5y_logret")
    trend = f["trend_health_5y"]
    pullback = f["pullback_1y"]   # negative
    # Want: large pullback, strong long-term trend, recovery history
    score = (
        _z(-pullback) * 1.0
        + _z(trend) * 1.0
        + _z(f["recovery_rate"].fillna(f["recovery_rate"].median())) * 0.6
        + _z(f["accel"]) * 0.4
    )
    if excess is not None:
        score = score + _z(excess) * 0.7
    # Filter: must be a "winner" (trend>0.55, excess>0)
    mask = trend > 0.55
    if excess is not None:
        mask = mask & (excess > 0)
    # Pullback at least 15%
    mask = mask & (pullback < -0.15)
    # Not in freefall
    mask = mask & (f["rsi_14"] > 30)
    return score.where(mask).dropna()


def score_winner_only(p: FeaturePack) -> pd.Series:
    """Pure 'long-term winner' filter: top excess-return stocks above 200dma, no pullback bias."""
    f = p.f
    excess = f.get("excess_5y_logret")
    if excess is None:
        return pd.Series(dtype=float)
    score = _z(excess) + 0.7 * _z(f["trend_health_5y"]) + 0.3 * _z(f["mom_12_1"]) - 0.5 * _z(f["vol_1y"])
    mask = (f["d_sma200"] > 0) & (excess > 0)
    return score.where(mask).dropna()


def score_proprietary_v1(p: FeaturePack) -> pd.Series:
    """Novel composite: 'Resilient Compounders on Sale'

    Build a quality-of-compounding score:
      Q = trend_health * recovery_rate * (1 - clip(vol_1y, 0.1, 1.0))
    Multiply by a 'discount' sigmoid centered at 25% pullback.
    Multiply by 'momentum reignition' = 1 if RSI > 30 and accel>0.
    """
    f = p.f
    trend = f["trend_health_5y"].fillna(0)
    rec = f["recovery_rate"].fillna(f["recovery_rate"].median())
    vol = f["vol_1y"].fillna(f["vol_1y"].median()).clip(0.1, 1.0)
    excess = f.get("excess_5y_logret", pd.Series(0, index=trend.index))
    Q = trend * rec * (1 - vol) * (1 + np.tanh(excess.fillna(0)))
    # Discount: 0 at no pullback, 1 at >=30% pullback
    pull = -f["pullback_1y"].fillna(0)
    discount = np.tanh(pull / 0.20)
    # Reignition: RSI > 35 AND accel > -0.02 (not in freefall)
    rsi = f["rsi_14"].fillna(50)
    reign = ((rsi > 35) & (f["accel"].fillna(0) > -0.02)).astype(float)
    score = Q * discount * (0.3 + 0.7 * reign)
    return score.dropna()


def score_proprietary_v2(p: FeaturePack) -> pd.Series:
    """Hard quality gate + score by depth-of-discount, breadth of recovery, and momentum regime.

    Gate:
      trend_health_5y >= 0.65
      recovery_rate >= 0.6 (or NaN if too few prior DD events -> assume neutral)
      sma50_above_200 AND price > sma200 (regime filter)
      rsi_14 > 35 (not in active freefall)
      pullback_1y between -10% and -45% (meaningful pullback but not catastrophic)

    Score:
      0.5 * (-pullback)           # depth of discount
      + 0.3 * recovery_rate
      + 0.4 * mom_12_1            # winner status
      + 0.3 * accel               # turning around
      + 0.2 * (excess_5y_logret)
    """
    f = p.f
    rec = f["recovery_rate"]
    rec_filled = rec.fillna(rec.median())

    trend = f["trend_health_5y"]
    pull = f["pullback_1y"]
    rsi = f["rsi_14"]
    sma_above = f.get("sma50_above_200", pd.Series(0, index=trend.index))
    dsma200 = f["d_sma200"]
    accel = f["accel"]
    mom = f.get("mom_12_1", pd.Series(0, index=trend.index))
    excess = f.get("excess_5y_logret", pd.Series(0, index=trend.index))

    gate = (
        (trend >= 0.65)
        & (rec_filled >= 0.6)
        & (sma_above > 0)
        & (dsma200 > -0.05)  # within 5% of 200dma
        & (rsi > 35)
        & (pull >= -0.45)
        & (pull <= -0.05)
    )

    score = (
        0.5 * (-pull.fillna(0))
        + 0.3 * rec_filled
        + 0.4 * mom.fillna(0)
        + 0.3 * accel.fillna(0)
        + 0.2 * excess.fillna(0)
    )
    return score.where(gate).dropna()


def score_explosive_winners(p: FeaturePack) -> pd.Series:
    """High momentum + above 200dma + recent acceleration. Aimed at 'rallies of 100s of %'."""
    f = p.f
    mom = f.get("mom_12_1")
    if mom is None or mom.empty:
        return pd.Series(dtype=float)
    sharpe = f.get("sharpe_1y", pd.Series(0, index=mom.index))
    score = _z(mom) * 1.0 + _z(f["accel"]) * 0.6 + _z(sharpe) * 0.4
    # Gates: above 200dma, momentum positive, RSI not extreme overbought (<80)
    mask = (f["d_sma200"] > 0) & (mom > 0.10) & (f["rsi_14"] < 80)
    return score.where(mask).dropna()


def score_minimum_drawdown_compounders(p: FeaturePack) -> pd.Series:
    """Stocks with high % time above 200dma and low volatility. The 'never goes down' set."""
    f = p.f
    score = _z(f["trend_health_5y"]) * 1.5 + _z(-f["vol_1y"]) * 1.0 + _z(f["recovery_rate"].fillna(0.5)) * 0.5
    # Gate: trend_health >= 0.8 (above 200dma 80% of the time)
    mask = (f["trend_health_5y"] >= 0.8) & (f["d_sma200"] > -0.10)
    return score.where(mask).dropna()


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
def all_strategies(top_k: int = 5) -> list[Strategy]:
    return [
        Strategy("quality_pullback", score_quality_pullback, top_k=top_k,
                 description="Pullback-in-quality blend (PLAN.md baseline)"),
        Strategy("dual_momentum", score_dual_momentum, top_k=top_k,
                 description="Classic 12-1 momentum + above 200dma"),
        Strategy("low_vol_trend", score_low_vol_trend, top_k=top_k,
                 description="Low-vol + trend health"),
        Strategy("pullback_in_winner", score_pullback_in_winner, top_k=top_k,
                 description="Long-term winners on pullback"),
        Strategy("winner_only", score_winner_only, top_k=top_k,
                 description="Pure long-term excess returns"),
        Strategy("proprietary_v1", score_proprietary_v1, top_k=top_k,
                 description="Resilient compounders on sale"),
        Strategy("proprietary_v2", score_proprietary_v2, top_k=top_k,
                 description="Hard quality gate + depth-of-discount blend"),
        Strategy("explosive_winners", score_explosive_winners, top_k=top_k,
                 description="High-momentum trend continuation"),
        Strategy("min_dd_compounders", score_minimum_drawdown_compounders, top_k=top_k,
                 description="Stocks that spend >=80% of days above 200dma"),
    ]
