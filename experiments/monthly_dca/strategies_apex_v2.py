"""Strategy variants V2 — alternative compositions for the compounding engine.

Approaches:
  J. APEX_DEEP_VALUE      — hard pullback bias, asymmetric upside
  K. APEX_RS_LEADER       — pure RS leadership multi-timeframe
  L. APEX_QUALITY_BREAK   — quality compounder + tight consolidation breakout
  M. APEX_MULTIBAGGER     — high tail-asymmetry hunter
  N. APEX_CONSENSUS_HARD  — must score top decile in 5+ signals
  O. APEX_LOW_BETA_MOM    — high momentum + low beta (Frazzini-Pedersen leverage constraint alpha)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.strategies_fast import _safe, _z
from experiments.monthly_dca.strategies_apex import (
    _spy_regime, _hard_exclusion_filter,
    _momentum_leg, _quality_leg, _asymmetry_leg, _penalty_leg,
)


def apex_deep_value(df: pd.DataFrame) -> pd.Series:
    """Deep value: long-term winner on a 20-50% pullback with high recovery rate."""
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    pull = _safe(df, "pullback_1y", 0)
    trend = _safe(df, "trend_health_5y", 0.5)
    rec = _safe(df, "recovery_rate", 0.6).fillna(0.6)
    accel = _safe(df, "accel", 0)
    rs3 = _safe(df, "rs_3m_spy", 0)
    qs5 = _safe(df, "quality_score_5y", 0)
    mom3y = _safe(df, "mom_3y", 0)
    sharpe5 = _safe(df, "sharpe_5y", 0)
    rsi = _safe(df, "rsi_14", 50)
    ret5 = _safe(df, "ret_5d", 0)
    dist_low = _safe(df, "dist_from_low_1y", 0.1)
    dsma = _safe(df, "d_sma200", 0)

    score = (
        _z(-pull) * 1.2
        + _z(trend) * 0.9
        + _z(rec) * 0.6
        + _z(accel) * 0.6
        + _z(rs3) * 0.5
        + _z(qs5) * 0.7
        + _z(mom3y) * 0.6
        + _z(sharpe5) * 0.5
        + _z(dist_low) * 0.4
    )
    deep_value_gate = (
        (pull <= -0.15) & (pull >= -0.55)
        & (trend > 0.65)
        & (rec >= 0.5)
        & (rsi >= 30)
        & (ret5 > -0.04)
        & (mom3y > 0.10)
        & (dsma > -0.20)
    )
    return score.where(eligible & deep_value_gate)


def apex_rs_leader(df: pd.DataFrame) -> pd.Series:
    """Pure relative strength leadership across 3, 6, 12-month windows."""
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)
    rs3 = _safe(df, "rs_3m_spy")
    rs6 = _safe(df, "rs_6m_spy")
    rs12 = _safe(df, "rs_12m_spy")
    excess5 = _safe(df, "excess_5y_logret", 0)
    mom3y = _safe(df, "mom_3y")
    r2 = _safe(df, "trend_r2_12m")
    sharpe = _safe(df, "sharpe_12m")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")

    score = (
        _z(rs3) * 0.9
        + _z(rs6) * 1.0
        + _z(rs12) * 1.2
        + _z(excess5) * 0.7
        + _z(mom3y) * 0.7
        + _z(r2) * 0.5
        + _z(sharpe) * 0.5
        + _z(cons) * 0.5
    )
    gate = (
        (rs12 > 0) & (rs6 > 0) & (rs3 > -0.05)
        & (mom3y > 0.10)
        & (dsma > 0)
        & (rsi >= 35) & (rsi <= 80)
    )
    return score.where(eligible & gate)


def apex_quality_break(df: pd.DataFrame) -> pd.Series:
    """Quality compounder breaking out of tight consolidation."""
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)
    tc = _safe(df, "tight_consolidation_60", 0)
    bo = _safe(df, "breakout_strength_60", 0)
    vc = _safe(df, "vol_contraction", 1.0)
    qs5 = _safe(df, "quality_score_5y", 0)
    rs12 = _safe(df, "rs_12m_spy")
    sharpe = _safe(df, "sharpe_12m")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    near = _safe(df, "near_52wh_60d", 0)
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    mom3y = _safe(df, "mom_3y")
    trend = _safe(df, "trend_health_5y")

    score = (
        _z(tc) * 0.5
        + _z(bo) * 1.0
        - _z(vc) * 0.6
        + _z(qs5) * 0.8
        + _z(rs12) * 0.7
        + _z(sharpe) * 0.6
        + _z(cons) * 0.5
        + near * 0.5
    )
    gate = (
        (qs5 > 0)
        & (trend > 0.60)
        & (dsma > 0)
        & (rsi >= 40) & (rsi <= 78)
        & (mom3y > 0.10)
        & (vc < 1.0)
    )
    return score.where(eligible & gate)


def apex_multibagger(df: pd.DataFrame) -> pd.Series:
    """High tail-asymmetry hunter: tail_ratio + multibagger + best_month + accel."""
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    tail = _safe(df, "tail_ratio_24m", 1.0)
    multi = _safe(df, "multibagger_ratio_24m", 0)
    best_m = _safe(df, "best_month_24m", 0)
    accel = _safe(df, "accel", 0)
    rs12 = _safe(df, "rs_12m_spy")
    mom3y = _safe(df, "mom_3y")
    bo = _safe(df, "breakout_strength_60", 0)
    drift = _safe(df, "earnings_drift_proxy", 0)
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")

    score = (
        _z(tail) * 0.8
        + _z(multi) * 0.7
        + _z(best_m) * 0.6
        + _z(accel) * 0.5
        + _z(rs12) * 0.8
        + _z(mom3y) * 0.7
        + _z(bo) * 0.5
        + _z(drift) * 0.4
    )
    gate = (
        (tail > 1.2)
        & (mom3y > 0.10)
        & (dsma > -0.10)
        & (rsi >= 30) & (rsi <= 78)
        & (trend > 0.50)
    )
    return score.where(eligible & gate)


def apex_consensus_hard(df: pd.DataFrame) -> pd.Series:
    """Top-decile across many signals."""
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    fields = [
        "rs_12m_spy", "mom_3y", "sharpe_12m", "trend_r2_12m",
        "frac_above_50dma_1y", "mom_consistency_12m", "tail_ratio_24m",
        "quality_score_5y", "idio_mom_12_1", "mom_per_unit_vol_12",
    ]
    sigs = []
    for f in fields:
        if f in df.columns:
            sigs.append(df[f].rank(pct=True, na_option="keep"))
    if not sigs:
        return pd.Series(np.nan, index=df.index)
    pct = pd.concat(sigs, axis=1)
    avg = pct.mean(axis=1)
    in_top = (pct > 0.85).sum(axis=1)
    in_bottom = (pct < 0.30).sum(axis=1)
    score = avg + 0.10 * in_top - 0.05 * in_bottom

    fence = (
        _safe(df, "trend_health_5y") > 0.55
    ) & (
        _safe(df, "d_sma200") > -0.10
    ) & (
        _safe(df, "rsi_14", 50) >= 32
    ) & (
        _safe(df, "rsi_14", 50) <= 80
    ) & (
        _safe(df, "mom_3y") > 0.10
    )
    return score.where(eligible & fence)


def apex_low_beta_mom(df: pd.DataFrame) -> pd.Series:
    """High momentum + low beta — Frazzini-Pedersen leverage-constraint alpha.

    Buy stocks that:
      - Have strong relative strength
      - Have low beta (deleverage-constraint alpha effect)
      - Have low residual vol (smooth trend)
    """
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    rs12 = _safe(df, "rs_12m_spy")
    mom3y = _safe(df, "mom_3y")
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    beta = _safe(df, "beta_2y", 1.0)
    cons = _safe(df, "mom_consistency_12m", 0.5)
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")

    score = (
        _z(rs12) * 1.0
        + _z(mom3y) * 1.0
        + _z(sharpe) * 0.9
        + _z(r2) * 0.6
        - _z(beta) * 0.8     # lower beta is better
        + _z(cons) * 0.6
    )
    gate = (
        (rs12 > 0)
        & (mom3y > 0.10)
        & (beta < 1.5)
        & (dsma > 0)
        & (rsi >= 35) & (rsi <= 78)
    )
    return score.where(eligible & gate)


def all_apex_v2_strategies():
    return {
        "apex_deep_value": apex_deep_value,
        "apex_rs_leader": apex_rs_leader,
        "apex_quality_break": apex_quality_break,
        "apex_multibagger": apex_multibagger,
        "apex_consensus_hard": apex_consensus_hard,
        "apex_low_beta_mom": apex_low_beta_mom,
    }
