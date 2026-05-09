"""V3 strategies — designed for the compounding portfolio engine.

Six distinct philosophies:

  A. MOMENTUM LOCOMOTIVE     — pure relentless momentum, lock gains via trail
  B. COMPOUND QUALITY         — smooth compounders, monthly rebalance
  C. ASYMMETRIC REBOUND       — deep value in quality, give room with trail50
  D. DYNAMIC CONCENTRATION    — k varies by regime
  E. MULTI-SIGNAL CONSENSUS   — top-decile across 8 signals
  F. APEX ENGINE              — multi-stage proprietary master (filter+score+rank)

All are pure functions: features DataFrame -> Series of scores (NaN = excluded).
Higher score => more preferred.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from experiments.monthly_dca.strategies_fast import _safe, _z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def _safe_pct(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col].rank(pct=True, na_option="keep")


def _spy_regime(df: pd.DataFrame) -> str:
    """Classify SPY market regime."""
    if "SPY" not in df.index:
        return "default"
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    # Deep bear: total cash
    if spy_dsma < -0.10 and spy_rsi < 35:
        return "bear"
    # Mild bear or correction
    if spy_dsma < -0.05 and spy_rsi < 45:
        return "correction"
    # Recovery: just reclaimed 200dma
    if -0.05 <= spy_dsma <= 0.03 and spy_rsi > 38:
        return "recovery"
    # Strong bull
    if spy_mom > 0.18 and spy_dsma > 0.05:
        return "strong_bull"
    # Healthy uptrend
    if spy_dsma > 0 and spy_mom > 0.08:
        return "uptrend"
    # Default
    return "default"


# ---------------------------------------------------------------------------
# A. MOMENTUM LOCOMOTIVE — relentless momentum
# ---------------------------------------------------------------------------
def momentum_locomotive(df: pd.DataFrame) -> pd.Series:
    """High-octane momentum: stocks running hard with strong RS and clean trends.

    Designed to be paired with trail_25 or trail_35 exit, so we lock gains
    when momentum stalls and recycle into next month's runners.
    """
    rs12 = _safe(df, "rs_12m_spy")
    rs6 = _safe(df, "rs_6m_spy")
    rs3 = _safe(df, "rs_3m_spy")
    mom3y = _safe(df, "mom_3y")
    mom12 = _safe(df, "mom_12_1")
    sharpe12 = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    accel = _safe(df, "mom_accel")
    near = _safe(df, "near_52wh_60d", 0)
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")
    vol_exp = _safe(df, "vol_expansion_24m", 1.0)
    ret21 = _safe(df, "ret_21d", 0)
    pull = _safe(df, "pullback_1y")

    score = (
        _z(rs12) * 1.2
        + _z(rs6) * 0.8
        + _z(rs3) * 0.5
        + _z(mom3y) * 1.0
        + _z(mom12) * 1.0
        + _z(sharpe12) * 0.9
        + _z(r2) * 0.7
        + _z(cons) * 0.6
        + _z(accel) * 0.5
        + near * 0.4
    )

    gate = (
        (rs12 > 0)                  # outperforming SPY trailing 12m
        & (mom12 > 0.10)            # at least 10% trailing 12-1
        & (mom3y > 0.20)            # solid 3y trend
        & (dsma > 0)                # above 200dma
        & (rsi >= 40) & (rsi <= 78)  # not extreme
        & (trend > 0.55)
        & (vol_exp < 2.5)            # not blowing off
        & (pull > -0.30)             # not collapsing
        & (ret21 > -0.08)            # no recent waterfall
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# B. COMPOUND QUALITY — smooth compounders, monthly rebalance
# ---------------------------------------------------------------------------
def compound_quality(df: pd.DataFrame) -> pd.Series:
    """Smooth high-Sharpe compounders. Designed for monthly_rebalance exit:
    each month, rotate into the current best smooth compounders.
    """
    sharpe12 = _safe(df, "sharpe_12m")
    sharpe1y = _safe(df, "sharpe_1y")
    r2 = _safe(df, "trend_r2_12m")
    frac = _safe(df, "frac_above_50dma_1y")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    rs12 = _safe(df, "rs_12m_spy")
    mom3y = _safe(df, "mom_3y")
    trend = _safe(df, "trend_health_5y")
    vol12 = _safe(df, "vol_12m")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    rec = _safe(df, "recovery_rate", 0.6)
    drawdown_age = _safe(df, "drawdown_age_days", 252)

    score = (
        _z(sharpe12) * 1.0
        + _z(sharpe1y) * 0.8
        + _z(r2) * 1.0
        + _z(frac) * 0.8
        + _z(cons) * 0.6
        + _z(rs12) * 0.7
        + _z(mom3y) * 0.7
        + _z(rec) * 0.4
        - _z(vol12) * 0.5
    )
    gate = (
        (trend > 0.65)
        & (dsma > 0)
        & (rsi >= 40) & (rsi <= 75)
        & (sharpe12 > 0.8)
        & (mom3y > 0.10)
        & (frac > 0.6)
        & (cons > 0.5)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# C. ASYMMETRIC REBOUND — deep value in quality, trail50 hold
# ---------------------------------------------------------------------------
def asymmetric_rebound(df: pd.DataFrame) -> pd.Series:
    """Long-term winners on 15-45% pullback with selling decel.

    Pair with trail_50 to give the rebound room to breathe.
    """
    pull = _safe(df, "pullback_1y")
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate", 0.6).fillna(0.6)
    accel = _safe(df, "accel")
    rs12 = _safe(df, "rs_12m_spy")
    rs3 = _safe(df, "rs_3m_spy")
    mom3y = _safe(df, "mom_3y")
    mom12 = _safe(df, "mom_12_1")
    sharpe5 = _safe(df, "sharpe_5y", 0.5)
    rsi = _safe(df, "rsi_14", 50)
    ret5 = _safe(df, "ret_5d", 0)
    dist_low = _safe(df, "dist_from_low_1y", 0.1)
    dsma = _safe(df, "d_sma200")
    qs = _safe(df, "quality_score_5y", 0)

    score = (
        _z(-pull) * 0.9
        + _z(trend) * 1.0
        + _z(rec) * 0.5
        + _z(accel) * 0.6
        + _z(mom3y) * 0.7
        + _z(rs3) * 0.5
        + _z(qs) * 0.6
        + _z(dist_low) * 0.3
    )
    gate = (
        (pull <= -0.15)
        & (pull >= -0.55)
        & (trend > 0.65)
        & (rec >= 0.5)
        & (rsi >= 30)
        & (rsi <= 60)
        & (ret5 > -0.04)            # selling decelerating
        & (mom3y > 0.10)            # was a long-term winner
        & (dsma > -0.20)            # not in death spiral
        & (dist_low > 0.05)         # at least 5% above 1y low
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# D. DYNAMIC CONCENTRATION — same scoring, K varies (called from engine)
# ---------------------------------------------------------------------------
def dyn_conc_score(df: pd.DataFrame) -> pd.Series:
    """Use regime to pick which underlying to score by.

    Returns scores; the *engine* will determine k based on regime.
    """
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)
    if regime == "correction":
        # Buy quality compounders that haven't broken
        return compound_quality(df)
    if regime == "recovery":
        return asymmetric_rebound(df)
    if regime == "strong_bull":
        return momentum_locomotive(df)
    if regime == "uptrend":
        # Both momentum and quality work
        m = momentum_locomotive(df)
        q = compound_quality(df)
        m_r = m.rank(pct=True, na_option="keep")
        q_r = q.rank(pct=True, na_option="keep")
        return m_r.combine(q_r, max, fill_value=0)
    # default
    return compound_quality(df)


def dyn_conc_k(df: pd.DataFrame) -> int:
    """Number of picks for dynamic concentration based on regime."""
    regime = _spy_regime(df)
    return {
        "bear": 0,
        "correction": 4,
        "recovery": 3,
        "strong_bull": 2,
        "uptrend": 3,
        "default": 5,
    }[regime]


# ---------------------------------------------------------------------------
# E. MULTI-SIGNAL CONSENSUS — top-decile across many signals
# ---------------------------------------------------------------------------
_CONSENSUS_FIELDS = [
    "mom_3y", "mom_12_1", "trend_r2_12m", "sharpe_12m",
    "frac_above_50dma_1y", "rs_12m_spy", "tail_ratio_24m",
    "mom_consistency_12m", "near_52wh_60d", "trend_slope_252",
    "quality_score_5y",     # alpha2
    "idio_mom_12_1",        # alpha2
    "mom_per_unit_vol_12",  # alpha2
]


def consensus_engine(df: pd.DataFrame) -> pd.Series:
    sigs = []
    for col in _CONSENSUS_FIELDS:
        if col in df.columns:
            sigs.append(df[col].rank(pct=True, na_option="keep"))
    if not sigs:
        return pd.Series(np.nan, index=df.index)
    pct = pd.concat(sigs, axis=1)
    avg = pct.mean(axis=1)
    in_top = (pct > 0.85).sum(axis=1)
    in_bottom = (pct < 0.30).sum(axis=1)
    score = avg + 0.05 * in_top - 0.03 * in_bottom

    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    mom3y = _safe(df, "mom_3y")
    fence = (
        (trend > 0.55)
        & (dsma > -0.10)
        & (rsi >= 32)
        & (rsi <= 78)
        & (mom3y > 0.0)
    )
    return score.where(fence)


# ---------------------------------------------------------------------------
# F. APEX ENGINE — multi-stage proprietary master
# ---------------------------------------------------------------------------
def apex_engine(df: pd.DataFrame) -> pd.Series:
    """Multi-stage proprietary master.

    Stage 1: Quality filter
      - trend_health_5y > 0.55
      - rsi between 30 and 78
      - dsma200 > -0.15

    Stage 2: Compute composite score blending three legs:
      - Momentum leg: rs_12m_spy + mom_3y + sharpe_12m + trend_r2_12m
      - Quality leg: frac_above_50dma_1y + mom_consistency + recovery_rate + low max_dd_5y
      - Asymmetry leg: tail_ratio_24m + best_month_24m + dist_from_low_1y
      - Penalty: high beta_2y, high vol_expansion_24m

    Stage 3: Regime-conditional weight on the legs.

    Stage 4: Bonus for stocks in top decile across multiple signals.
    """
    regime = _spy_regime(df)

    # Quality filter
    trend = _safe(df, "trend_health_5y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    mom3y = _safe(df, "mom_3y")
    pull = _safe(df, "pullback_1y")
    vol_exp = _safe(df, "vol_expansion_24m", 1.0)

    base_filter = (
        (trend > 0.55)
        & (rsi >= 30)
        & (rsi <= 78)
        & (dsma > -0.15)
        & (vol_exp < 3.0)            # not in chaos
        & (pull > -0.55)             # not collapsing
    )

    # Momentum leg
    rs12 = _z(_safe(df, "rs_12m_spy"))
    rs6 = _z(_safe(df, "rs_6m_spy"))
    momz = _z(mom3y)
    mom12z = _z(_safe(df, "mom_12_1"))
    sharpe = _z(_safe(df, "sharpe_12m"))
    r2 = _z(_safe(df, "trend_r2_12m"))
    accel = _z(_safe(df, "accel"))
    mom_leg = (
        rs12 * 1.0 + rs6 * 0.5 + momz * 0.9 + mom12z * 0.8
        + sharpe * 0.7 + r2 * 0.6 + accel * 0.3
    )

    # Quality leg
    frac = _z(_safe(df, "frac_above_50dma_1y"))
    cons = _z(_safe(df, "mom_consistency_12m", 0.5))
    rec = _z(_safe(df, "recovery_rate", 0.6).fillna(0.6))
    qs5 = _z(_safe(df, "quality_score_5y", 0))   # alpha2
    sma_above = _safe(df, "sma50_above_200", 0)
    qual_leg = frac * 0.7 + cons * 0.6 + rec * 0.4 + qs5 * 0.6 + sma_above.astype(float) * 0.3

    # Asymmetry leg
    tail = _z(_safe(df, "tail_ratio_24m", 1.0))
    best_m = _z(_safe(df, "best_month_24m", 0))
    multi = _z(_safe(df, "multibagger_ratio_24m", 0))
    dist_low = _z(_safe(df, "dist_from_low_1y", 0.1))
    asym_leg = tail * 0.5 + best_m * 0.4 + multi * 0.4 + dist_low * 0.3

    # Penalty leg
    beta = _z(_safe(df, "beta_2y", 1.0))         # alpha2
    vol_exp_z = _z(vol_exp)
    fip = _z(_safe(df, "fip_score", 0))          # alpha2 (lower = smoother)
    pen = beta * 0.4 + vol_exp_z * 0.3 - fip * 0.4   # subtract -fip => prefer smooth

    # Regime-conditional weights
    if regime == "strong_bull":
        score = mom_leg * 1.4 + qual_leg * 0.5 + asym_leg * 0.7 - pen * 0.3
    elif regime == "uptrend":
        score = mom_leg * 1.0 + qual_leg * 1.0 + asym_leg * 0.5 - pen * 0.4
    elif regime == "recovery":
        score = mom_leg * 0.5 + qual_leg * 1.2 + asym_leg * 0.4 - pen * 0.5
        # Boost stocks with deeper pullback
        pull_z = _z(-pull)
        score = score + pull_z * 0.6
    elif regime == "correction":
        score = mom_leg * 0.3 + qual_leg * 1.5 + asym_leg * 0.2 - pen * 0.6
    elif regime == "bear":
        return pd.Series(np.nan, index=df.index)
    else:  # default
        score = mom_leg * 0.9 + qual_leg * 0.9 + asym_leg * 0.5 - pen * 0.4

    # Cross-sectional consensus bonus
    sigs = []
    for col in ["rs_12m_spy", "mom_3y", "sharpe_12m", "trend_r2_12m", "frac_above_50dma_1y"]:
        if col in df.columns:
            sigs.append(df[col].rank(pct=True, na_option="keep"))
    if sigs:
        pct = pd.concat(sigs, axis=1)
        in_top = (pct > 0.85).sum(axis=1)
        score = score + 0.10 * in_top.fillna(0)

    return score.where(base_filter)


# ---------------------------------------------------------------------------
# G. APEX ENGINE V2 — same as APEX but tighter quality bar + adaptive K
# ---------------------------------------------------------------------------
def apex_engine_v2(df: pd.DataFrame) -> pd.Series:
    """Tighter version of APEX with stronger quality bar and harder regime
    filtering.  Also adds:
      - Eligible only if either (mom_3y > 30%) or (deep pullback in winner)
      - Boost for 'tight_consolidation' (alpha2)
    """
    regime = _spy_regime(df)
    base = apex_engine(df)
    if regime == "bear":
        return base

    # Add tight consolidation bonus
    tc = _safe(df, "tight_consolidation_60", 0)   # alpha2
    bo = _safe(df, "breakout_strength_60", 0)     # alpha2
    rsi_zone = _safe(df, "rsi_zone_score", 0.5)   # alpha2
    boost = _z(tc) * 0.3 + _z(bo) * 0.5 + _z(rsi_zone) * 0.4

    # Tighter eligibility
    mom3y = _safe(df, "mom_3y")
    pull = _safe(df, "pullback_1y")
    trend = _safe(df, "trend_health_5y")
    eligibility = (
        ((mom3y > 0.30) & (pull > -0.20))
        | ((mom3y > 0.10) & (pull <= -0.15) & (pull >= -0.55) & (trend > 0.70))
        | ((bo > 0.0) & (mom3y > 0.20) & (trend > 0.60))
    )
    return (base + boost).where(eligibility)


# ---------------------------------------------------------------------------
# H. BREAKOUT MOMENTUM — pre-breakout coil + breakout confirmation
# ---------------------------------------------------------------------------
def breakout_momentum(df: pd.DataFrame) -> pd.Series:
    """Stocks coiling tight then breaking out (VCP-style).

    Uses alpha2 features:
      tight_consolidation_60, breakout_strength_60, vol_contraction
    """
    tc = _safe(df, "tight_consolidation_60", 0)
    bo = _safe(df, "breakout_strength_60", 0)
    vc = _safe(df, "vol_contraction", 1.0)
    rs12 = _safe(df, "rs_12m_spy")
    mom3y = _safe(df, "mom_3y")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")
    near = _safe(df, "near_52wh_60d", 0)
    drawdown_age = _safe(df, "drawdown_age_days", 0)

    score = (
        _z(tc) * 0.6
        + _z(bo) * 1.0
        - _z(vc) * 0.6                   # lower vc = more contraction = better
        + _z(rs12) * 0.8
        + _z(mom3y) * 0.7
        + _z(cons) * 0.6
        + _z(sharpe) * 0.6
        + _z(r2) * 0.5
        + near * 0.5
    )
    gate = (
        (trend > 0.60)
        & (dsma > 0)
        & (rsi >= 40) & (rsi <= 78)
        & (mom3y > 0.10)
        & (cons > 0.45)
        & (vc < 1.0)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# I. PERFECT STORM — multi-leg agreement: momentum AND quality AND breakout AND regime
# ---------------------------------------------------------------------------
def perfect_storm(df: pd.DataFrame) -> pd.Series:
    """Score = product of 4 sub-scores. Stocks must score high in all 4 legs.

    Used for ultra-concentration (k=1-3).
    """
    mom = momentum_locomotive(df)
    qual = compound_quality(df)
    breakout = breakout_momentum(df)
    consensus = consensus_engine(df)

    # Convert to percentile ranks; multiply
    mom_r = mom.rank(pct=True, na_option="keep").fillna(0)
    qual_r = qual.rank(pct=True, na_option="keep").fillna(0)
    bo_r = breakout.rank(pct=True, na_option="keep").fillna(0)
    cons_r = consensus.rank(pct=True, na_option="keep").fillna(0)
    score = mom_r * qual_r * bo_r * cons_r
    # Need at least 3 of 4 to have valid scores (i.e., not 0)
    nonzero = (mom_r > 0).astype(int) + (qual_r > 0).astype(int) + (bo_r > 0).astype(int) + (cons_r > 0).astype(int)
    return score.where(nonzero >= 3)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
def all_v3_strategies():
    from experiments.monthly_dca.compound_engine import Strategy
    return {
        "momentum_locomotive": momentum_locomotive,
        "compound_quality": compound_quality,
        "asymmetric_rebound": asymmetric_rebound,
        "dynamic_concentration": dyn_conc_score,
        "consensus_engine": consensus_engine,
        "apex_engine": apex_engine,
        "apex_engine_v2": apex_engine_v2,
        "breakout_momentum": breakout_momentum,
        "perfect_storm": perfect_storm,
    }
