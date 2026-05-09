"""Strategy-rotation-plus: refined regime-switching strategies.

The base strategy_rotation works because:
  * It switches between purpose-built sub-strategies based on SPY regime
  * Each sub-strategy is well-tuned for its market environment
  * Bear-regime → cash, avoiding -50% drawdowns

This module adds variants:

  rotation_plus       — same regime classifier; tighter delist filter on each leg
  rotation_5regimes   — 5 distinct regimes (bull, recovery, sideways, correction, bear)
  rotation_breadth    — uses SPY+breadth (% of stocks above their 200dma) for regime
  rotation_apex       — uses apex_balanced/turbocharged as legs in different regimes
  rotation_adaptive_k — same classifier; varies k from 3-7 by regime intensity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.strategies_fast import (
    _safe, _z, quality_pullback, explosive_winners, pullback_in_winner,
)
from experiments.monthly_dca.strategies_apex import (
    _hard_exclusion_filter, _momentum_leg, _quality_leg, _asymmetry_leg, _penalty_leg,
)


# ---------------------------------------------------------------------------
# 1) Tight-filter rotation: same regime classifier; add hard delist filter
# ---------------------------------------------------------------------------
def rotation_plus(df: pd.DataFrame) -> pd.Series:
    if "SPY" not in df.index:
        return quality_pullback(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    if spy_dsma < -0.10 and spy_rsi < 35:
        return pd.Series(np.nan, index=df.index)
    if -0.05 < spy_dsma < 0.03:
        score = pullback_in_winner(df)
    elif spy_mom > 0.15:
        score = explosive_winners(df)
    else:
        score = quality_pullback(df)

    eligible = _hard_exclusion_filter(df)
    return score.where(eligible)


# ---------------------------------------------------------------------------
# 2) 5-regime classifier
# ---------------------------------------------------------------------------
def rotation_5regimes(df: pd.DataFrame) -> pd.Series:
    if "SPY" not in df.index:
        return quality_pullback(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    # Bear: cash
    if spy_dsma < -0.08 and spy_rsi < 40:
        return pd.Series(np.nan, index=df.index)
    # Correction: mild bear, buy quality only (no momentum)
    if spy_dsma < -0.03 and spy_rsi < 50:
        # Buy compounders that haven't broken
        trend = _safe(df, "trend_health_5y")
        rec = _safe(df, "recovery_rate", 0.6).fillna(0.6)
        sharpe = _safe(df, "sharpe_12m")
        rsi = _safe(df, "rsi_14", 50)
        mom3y = _safe(df, "mom_3y")
        score = _z(trend) * 1.5 + _z(rec) * 0.7 + _z(sharpe) * 0.7 + _z(mom3y) * 0.5
        gate = (trend > 0.70) & (rsi > 32) & (mom3y > 0.05) & (rec >= 0.5)
        eligible = _hard_exclusion_filter(df)
        return score.where(gate & eligible)
    # Recovery: SPY just reclaimed 200dma
    if -0.05 <= spy_dsma <= 0.03:
        score = pullback_in_winner(df)
        eligible = _hard_exclusion_filter(df)
        return score.where(eligible)
    # Strong bull: momentum
    if spy_mom > 0.18 and spy_dsma > 0.05:
        score = explosive_winners(df)
        eligible = _hard_exclusion_filter(df)
        return score.where(eligible)
    # Default uptrend / sideways: quality_pullback
    score = quality_pullback(df)
    eligible = _hard_exclusion_filter(df)
    return score.where(eligible)


# ---------------------------------------------------------------------------
# 3) APEX rotation: use apex strategies as legs
# ---------------------------------------------------------------------------
def rotation_apex(df: pd.DataFrame) -> pd.Series:
    """Use apex strategies as legs in different regimes.

    Use:
      - bear: cash
      - correction: apex_quality_break (smooth quality)
      - recovery: apex_deep_value (winner on dip)
      - strong_bull: apex_rs_leader or explosive_winners
      - default: quality_pullback (proven simple)
    """
    from experiments.monthly_dca.strategies_apex_v2 import (
        apex_deep_value, apex_quality_break, apex_rs_leader,
    )
    if "SPY" not in df.index:
        return quality_pullback(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    if spy_dsma < -0.10 and spy_rsi < 35:
        return pd.Series(np.nan, index=df.index)
    if -0.05 < spy_dsma < 0.03:
        return apex_deep_value(df)
    if spy_mom > 0.15:
        # Mix explosive_winners and apex_rs_leader
        a = explosive_winners(df)
        b = apex_rs_leader(df)
        a_r = a.rank(pct=True, na_option="keep")
        b_r = b.rank(pct=True, na_option="keep")
        return a_r.combine(b_r, lambda x, y: max(x or 0, y or 0))
    return quality_pullback(df)


# ---------------------------------------------------------------------------
# 4) Hardened rotation: same regime, but ULTRA-tight filter for picks
# ---------------------------------------------------------------------------
def rotation_hardened(df: pd.DataFrame) -> pd.Series:
    if "SPY" not in df.index:
        return quality_pullback(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    if spy_dsma < -0.10 and spy_rsi < 35:
        return pd.Series(np.nan, index=df.index)
    if -0.05 < spy_dsma < 0.03:
        score = pullback_in_winner(df)
    elif spy_mom > 0.15:
        score = explosive_winners(df)
    else:
        score = quality_pullback(df)

    # Ultra-tight filter
    vol_exp = _safe(df, "vol_expansion_24m", 1.0)
    dd_52wh = _safe(df, "dd_from_52wh", 0.0)
    mom_3y = _safe(df, "mom_3y", 0.0)
    price = _safe(df, "price", 100.0)
    trend = _safe(df, "trend_health_5y", 0.5)
    rsi = _safe(df, "rsi_14", 50)
    qs5 = _safe(df, "quality_score_5y", 0)
    pull = _safe(df, "pullback_1y", 0)
    sharpe = _safe(df, "sharpe_12m", 0)

    eligible = (
        (vol_exp < 2.0)            # tight: not in vol meltdown
        & (dd_52wh > -0.50)         # tight: not in 50%+ DD
        & (mom_3y > 0.0)            # tight: positive 3y trend
        & (price >= 7.0)            # no penny stocks
        & (trend > 0.45)            # tight: must have decent long-term character
        & (rsi >= 30)
        & (rsi <= 80)
        & (sharpe > -0.5)           # not catastrophic risk-adj
    )
    return score.where(eligible)


# ---------------------------------------------------------------------------
# 5) Rotation-rich: same legs, plus consensus bonus from cross-sectional rank
# ---------------------------------------------------------------------------
def rotation_rich(df: pd.DataFrame) -> pd.Series:
    if "SPY" not in df.index:
        return quality_pullback(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    if spy_dsma < -0.10 and spy_rsi < 35:
        return pd.Series(np.nan, index=df.index)
    if -0.05 < spy_dsma < 0.03:
        score = pullback_in_winner(df)
    elif spy_mom > 0.15:
        score = explosive_winners(df)
    else:
        score = quality_pullback(df)

    # Cross-sectional consensus bonus
    sigs = []
    for col in ["mom_3y", "rs_12m_spy", "trend_r2_12m", "sharpe_12m",
                "frac_above_50dma_1y", "mom_consistency_12m", "quality_score_5y",
                "idio_mom_12_1"]:
        if col in df.columns:
            sigs.append(df[col].rank(pct=True, na_option="keep"))
    if sigs:
        pct = pd.concat(sigs, axis=1)
        bonus = pct.mean(axis=1)  # 0..1
        # Add as boost
        score = score + bonus * 1.5

    eligible = _hard_exclusion_filter(df)
    return score.where(eligible)


# ---------------------------------------------------------------------------
# 6) Pure cross-sectional rank composite — orthogonal approach
# ---------------------------------------------------------------------------
def composite_xrank(df: pd.DataFrame) -> pd.Series:
    """Weighted cross-sectional rank composite, regime-conditional weights."""
    if "SPY" not in df.index:
        regime = "default"
    else:
        spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
        spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
        spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0
        if spy_dsma < -0.10 and spy_rsi < 35:
            return pd.Series(np.nan, index=df.index)
        if -0.05 < spy_dsma < 0.03:
            regime = "recovery"
        elif spy_mom > 0.15:
            regime = "bull"
        else:
            regime = "default"

    # Define feature -> weight per regime
    weights = {
        "bull": {
            "mom_3y": 0.8, "rs_12m_spy": 0.9, "mom_12_1": 1.0, "sharpe_12m": 0.7,
            "trend_r2_12m": 0.6, "frac_above_50dma_1y": 0.5, "near_52wh_60d": 0.4,
            "idio_mom_12_1": 0.6, "tail_ratio_24m": 0.4,
        },
        "recovery": {
            "trend_health_5y": 1.2, "recovery_rate": 0.8, "mom_3y": 0.6,
            "quality_score_5y": 0.8, "sharpe_5y": 0.5, "accel": 0.4,
            "dist_from_low_1y": 0.4, "rs_3m_spy": 0.5,
            # Note: invert pullback (deeper pullback = lower rank = bad here, so we use -pull)
        },
        "default": {
            "trend_health_5y": 1.0, "mom_3y": 0.9, "sharpe_12m": 0.8,
            "trend_r2_12m": 0.7, "rs_12m_spy": 0.8, "quality_score_5y": 0.7,
            "mom_consistency_12m": 0.5, "frac_above_50dma_1y": 0.5,
        },
    }
    cfg = weights[regime]
    score_sum = pd.Series(0.0, index=df.index)
    weight_sum = 0.0
    for col, w in cfg.items():
        if col not in df.columns:
            continue
        rk = df[col].rank(pct=True, na_option="keep").fillna(0.5)
        score_sum += w * rk
        weight_sum += w
    if weight_sum > 0:
        score_sum = score_sum / weight_sum

    # For recovery, ALSO add deep-pullback bonus
    if regime == "recovery":
        pull = _safe(df, "pullback_1y", 0)
        # Best score for pullback in [-0.30, -0.10]
        pull_bonus = (-pull).clip(0.10, 0.30) - 0.10  # 0..0.20
        score_sum = score_sum + pull_bonus * 2.0

    eligible = _hard_exclusion_filter(df)
    return score_sum.where(eligible)


def all_rotation_plus():
    return {
        "rotation_plus": rotation_plus,
        "rotation_5regimes": rotation_5regimes,
        "rotation_apex": rotation_apex,
        "rotation_hardened": rotation_hardened,
        "rotation_rich": rotation_rich,
        "composite_xrank": composite_xrank,
    }
