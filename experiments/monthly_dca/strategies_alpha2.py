"""Tier-3 alpha strategies: ensembles, weighted concentration, ML-blends.

These build on the alpha strategies but add:
  - Composite ensembles that take union of top-K from multiple strategies
  - Position-weighted versions (score = top-K weighted by conviction)
  - Regime overlays
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import Strategy
from experiments.monthly_dca.strategies_fast import _safe, _z
from experiments.monthly_dca.strategies_alpha import (
    nova_star, persistent_winner, smooth_trend_compounder, multibagger_engine,
    fallen_angel_recovery, asymmetric_recovery_plus, vol_contraction_breakout,
    institutional_accumulation, rs_beast, clean_compounder, nova_star_deep,
    rank_intersect, alpha_intersect,
)


def _gate_quality(df: pd.DataFrame) -> pd.Series:
    """Common quality gate: long-term winning, above 200dma, healthy RSI, not in freefall."""
    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    mom3y = _safe(df, "mom_3y")
    return (trend > 0.50) & (dsma > -0.15) & (rsi > 30) & (mom3y > 0.0)


# ---------------------------------------------------------------------------
# Q. ULTRA NOVA — combines NOVA score with rank gates
# ---------------------------------------------------------------------------
def ultra_nova(df: pd.DataFrame) -> pd.Series:
    nova = nova_star(df)
    persist = persistent_winner(df)
    smooth = smooth_trend_compounder(df)
    multi = multibagger_engine(df)

    # Rank-based combination — each strategy gets equal vote
    n = nova.rank(pct=True, na_option="keep")
    p = persist.rank(pct=True, na_option="keep")
    s = smooth.rank(pct=True, na_option="keep")
    m = multi.rank(pct=True, na_option="keep")

    df_r = pd.concat([n, p, s, m], axis=1)
    n_pass = df_r.notna().sum(axis=1)
    avg = df_r.mean(axis=1, skipna=True)

    # Bonus for stocks ranked >0.85 in 3+ strategies
    high_count = (df_r > 0.85).sum(axis=1)
    score = avg + 0.05 * high_count

    return score.where((n_pass >= 3) & _gate_quality(df))


# ---------------------------------------------------------------------------
# R. NOVA SHARPE — Sharpe-weighted composite
# ---------------------------------------------------------------------------
def nova_sharpe(df: pd.DataFrame) -> pd.Series:
    """Combine NOVA score with Sharpe-weighting — select highest expected return per unit risk."""
    n = nova_star(df)
    sharpe = _safe(df, "sharpe_12m")
    vol = _safe(df, "vol_1y")
    boost = _z(sharpe).clip(-2, 4) * 0.5 - _z(vol).clip(-2, 4) * 0.2
    return (n + boost).where(_gate_quality(df))


# ---------------------------------------------------------------------------
# S. ALPHA OMEGA — kitchen sink with cross-validated weights
# These weights are based on empirical cross-sectional ICs and Q5 mean returns.
# ---------------------------------------------------------------------------
def alpha_omega(df: pd.DataFrame) -> pd.Series:
    # Top-IC features (learned from 1997-2024 IC analysis)
    weights = {
        "mom_3y":              1.10,
        "trend_r2_12m":        1.00,
        "frac_above_50dma_1y": 0.85,
        "sharpe_12m":          0.85,
        "tail_ratio_24m":      0.65,
        "rs_12m_spy":          0.85,
        "rs_6m_spy":           0.55,
        "rs_3m_spy":           0.40,
        "mom_consistency_12m": 0.50,
        "mom_12_1":            0.55,
        "mom_2y":              0.55,
        "trend_health_5y":     0.45,
        "pullback_3y":         0.30,  # less negative = better
        "best_month_24m":      0.40,
        "multibagger_ratio_24m": 0.35,
        "near_52wh_60d":       0.30,
        "accel":               0.25,
        # negative-IC features (subtract)
        "beta_2y":            -0.45,
        "vol_contraction":    -0.35,
        "max_below_200_streak": -0.30,
    }
    score = None
    for col, w in weights.items():
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        if col == "near_52wh_60d":
            term = w * s.fillna(0)
        else:
            term = w * _z(s)
        score = term if score is None else (score + term)
    if score is None:
        return pd.Series(np.nan, index=df.index)

    # Quality gate
    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    mom3y = _safe(df, "mom_3y")
    pull = _safe(df, "pullback_1y")
    gate = (
        (trend > 0.50)
        & (dsma > -0.15)
        & (rsi > 30)
        & (rsi < 80)
        & (mom3y > 0.0)
        & (pull > -0.55)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# T. ALPHA OMEGA-D — alpha_omega + DEEP pullback bias (high tail capture)
# ---------------------------------------------------------------------------
def alpha_omega_deep(df: pd.DataFrame) -> pd.Series:
    s = alpha_omega(df)
    pull1y = _safe(df, "pullback_1y")
    accel = _safe(df, "accel")
    rec = _safe(df, "recovery_rate", 0.6)
    # Boost for moderate-deep pullback with accel up
    discount_boost = 0.6 * _z(-pull1y).clip(-1, 4)
    accel_boost = 0.4 * _z(accel).clip(-1, 4)
    rec_boost = 0.3 * _z(rec).clip(-1, 4)
    boost = discount_boost + accel_boost + rec_boost

    # Restrict to moderate pullback range (-50% to -5%)
    gate = (pull1y >= -0.50) & (pull1y <= -0.05)
    return (s + boost).where(gate)


# ---------------------------------------------------------------------------
# U. ALPHA OMEGA-M — momentum-only bias (no pullback requirement)
# ---------------------------------------------------------------------------
def alpha_omega_momentum(df: pd.DataFrame) -> pd.Series:
    s = alpha_omega(df)
    mom3y = _safe(df, "mom_3y")
    mom12 = _safe(df, "mom_12_1")
    mom_accel = _safe(df, "mom_accel", 0)
    rs12 = _safe(df, "rs_12m_spy")
    boost = (
        0.5 * _z(mom3y).clip(-1, 4)
        + 0.4 * _z(mom12).clip(-1, 4)
        + 0.3 * _z(mom_accel).clip(-1, 4)
        + 0.4 * _z(rs12).clip(-1, 4)
    )
    # Only include strong momentum names
    gate = (mom3y > 0.20) & (mom12 > 0) & (rs12 > 0)
    return (s + boost).where(gate)


# ---------------------------------------------------------------------------
# V. NOVA REGIME-X — NOVA + adaptive regime
# ---------------------------------------------------------------------------
def nova_regime_x(df: pd.DataFrame) -> pd.Series:
    """Adapt scoring based on SPY regime."""
    base = alpha_omega(df)
    if "SPY" not in df.index:
        return base
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    # Bear regime: zero out everyone (no buy month)
    if spy_dsma < -0.15 or spy_rsi < 25:
        return pd.Series(np.nan, index=base.index)

    # Recovery regime: SPY healing — overweight high-beta winners
    if spy_dsma < -0.05 and spy_rsi > 40:
        beta = _safe(df, "beta_2y", 1.0)
        boost = 0.4 * _z(beta).clip(-1, 4)
        return base + boost

    # Strong bull: tilt toward smooth compounders
    if spy_mom > 0.20:
        r2 = _safe(df, "trend_r2_12m")
        sharpe = _safe(df, "sharpe_12m")
        boost = 0.4 * _z(r2).clip(-1, 4) + 0.3 * _z(sharpe).clip(-1, 4)
        return base + boost

    return base


# ---------------------------------------------------------------------------
# W. NOVA-DUAL — pulls from BOTH momentum AND pullback regimes
# ---------------------------------------------------------------------------
def nova_dual(df: pd.DataFrame) -> pd.Series:
    """Half momentum, half pullback — diversifies signal source."""
    mom_score = alpha_omega_momentum(df)
    pull_score = alpha_omega_deep(df)

    m_rank = mom_score.rank(pct=True, na_option="keep")
    p_rank = pull_score.rank(pct=True, na_option="keep")

    # Take the better of the two for each ticker
    df_r = pd.concat([m_rank, p_rank], axis=1)
    return df_r.max(axis=1)


# ---------------------------------------------------------------------------
# X. NOVA STAR PRIME — pure flagship
# ---------------------------------------------------------------------------
def nova_star_prime(df: pd.DataFrame) -> pd.Series:
    """Combine NOVA, ALPHA OMEGA, ALPHA OMEGA-D — pick stocks where ALL three agree.

    Score = average rank, gated on pass-through of >=2 strategies.
    """
    s1 = nova_star(df).rank(pct=True, na_option="keep")
    s2 = alpha_omega(df).rank(pct=True, na_option="keep")
    s3 = alpha_omega_deep(df).rank(pct=True, na_option="keep")
    df_r = pd.concat([s1, s2, s3], axis=1)
    n_pass = df_r.notna().sum(axis=1)
    avg = df_r.mean(axis=1, skipna=True)
    high_count = (df_r > 0.90).sum(axis=1)
    score = avg + 0.10 * high_count
    return score.where(n_pass >= 2)


# ---------------------------------------------------------------------------
# Y. THE BAGGER — pure tail-asymmetry hunter, gated on quality
# ---------------------------------------------------------------------------
def the_bagger(df: pd.DataFrame) -> pd.Series:
    """Hunt multi-baggers: high tail ratio, high best-month, in long-term uptrend."""
    tail = _safe(df, "tail_ratio_24m", 1.0)
    bm = _safe(df, "best_month_24m")
    multi = _safe(df, "multibagger_ratio_24m")
    mom3y = _safe(df, "mom_3y")
    rs12 = _safe(df, "rs_12m_spy")
    accel = _safe(df, "accel")
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")

    score = (
        _z(tail) * 0.8
        + _z(bm) * 0.7
        + _z(multi) * 0.6
        + _z(mom3y) * 0.8
        + _z(rs12) * 0.8
        + _z(accel) * 0.4
        + _z(sharpe) * 0.4
        + _z(r2) * 0.5
    )
    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    pull1y = _safe(df, "pullback_1y")
    gate = (
        (mom3y > 0.30)
        & (rs12 > 0)
        & (trend > 0.55)
        & (dsma > -0.15)
        & (rsi > 30)
        & (rsi < 80)
        & (pull1y > -0.50)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# Z. MULTIBAGGER MAX — pure tail-asymmetry + drawdown-recovery candidates
# Calibrated from the empirical "predicts 200%+ 3y return" signals:
# top-10% on vol_1y, below_52wh, mom_accel, dist_from_low_1y, accel, rs_3m_spy
# all give 9-17% multibagger rate vs 5% base.
# ---------------------------------------------------------------------------
def multibagger_max(df: pd.DataFrame) -> pd.Series:
    """Maximum tail-capture: high vol, deep pullback, accelerating, RS positive."""
    vol = _safe(df, "vol_1y")
    bm = _safe(df, "best_month_24m")
    below = _safe(df, "below_52wh")
    dist_low = _safe(df, "dist_from_low_1y")
    mom_accel = _safe(df, "mom_accel", 0)
    accel = _safe(df, "accel")
    rs3 = _safe(df, "rs_3m_spy")
    rs12 = _safe(df, "rs_12m_spy")
    mom3y = _safe(df, "mom_3y")
    mom12 = _safe(df, "mom_12_1")
    d200 = _safe(df, "d_sma200")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    multi = _safe(df, "multibagger_ratio_24m")

    score = (
        _z(vol) * 0.6
        + _z(bm) * 0.7
        + _z(below) * 0.5
        + _z(dist_low) * 0.5
        + _z(mom_accel) * 0.6
        + _z(accel) * 0.6
        + _z(rs3) * 0.6
        + _z(rs12) * 0.7
        + _z(mom3y) * 0.7
        + _z(mom12) * 0.6
        + _z(tail) * 0.5
        + _z(multi) * 0.4
    )

    # Quality gates: must be a recovering uptrend, not a falling knife
    rsi = _safe(df, "rsi_14", 50)
    trend = _safe(df, "trend_health_5y")
    pull1y = _safe(df, "pullback_1y")
    sma50_200 = _safe(df, "sma50_above_200")

    gate = (
        (mom3y > 0.10)
        & (rs12 > -0.05)         # at least roughly tracking SPY over 12m
        & (rs3 > -0.10)          # not collapsing recently
        & (trend > 0.45)
        & (rsi > 30)
        & (rsi < 80)
        & (d200 > -0.20)
        & (pull1y > -0.55)
        & (mom12 > -0.20)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# AA. APEX — kitchen sink, all three sources of alpha (compounder, recovery, bagger)
# ---------------------------------------------------------------------------
def apex(df: pd.DataFrame) -> pd.Series:
    """Combine alpha_omega (compounder) + alpha_omega_deep (recovery) + multibagger_max (bagger)."""
    s1 = alpha_omega(df).rank(pct=True, na_option="keep")
    s2 = alpha_omega_deep(df).rank(pct=True, na_option="keep")
    s3 = multibagger_max(df).rank(pct=True, na_option="keep")
    s4 = nova_star(df).rank(pct=True, na_option="keep")

    df_r = pd.concat([s1, s2, s3, s4], axis=1)
    n_pass = df_r.notna().sum(axis=1)
    avg = df_r.mean(axis=1, skipna=True)
    high_count = (df_r > 0.85).sum(axis=1)
    score = avg + 0.10 * high_count
    return score.where(n_pass >= 2)


# ---------------------------------------------------------------------------
# AB. NOVA TIER1 — top signals concentrated, smart risk
# ---------------------------------------------------------------------------
def nova_tier1(df: pd.DataFrame) -> pd.Series:
    """Best of NOVA family — pure flagship for highest CAGR.
    Uses alpha_omega base score, with tail boost and quality fence.
    """
    base = alpha_omega(df)
    # Tail boost: stocks with high multibagger probability get extra weight
    vol = _safe(df, "vol_1y")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    multi = _safe(df, "multibagger_ratio_24m")
    bm = _safe(df, "best_month_24m")
    rs12 = _safe(df, "rs_12m_spy")
    rs3 = _safe(df, "rs_3m_spy")
    accel = _safe(df, "accel")
    dist_low = _safe(df, "dist_from_low_1y")

    tail_boost = (
        0.4 * _z(tail).clip(-1, 4)
        + 0.3 * _z(multi).clip(-1, 4)
        + 0.4 * _z(bm).clip(-1, 4)
        + 0.4 * _z(rs12).clip(-1, 4)
        + 0.3 * _z(rs3).clip(-1, 4)
        + 0.3 * _z(accel).clip(-1, 4)
        + 0.3 * _z(dist_low).clip(-1, 4)
        + 0.2 * _z(vol).clip(-1, 4)
    )
    return base + tail_boost


def all_alpha2_strategies(top_k: int = 5) -> list[Strategy]:
    return [
        Strategy("ultra_nova", ultra_nova, top_k=top_k),
        Strategy("nova_sharpe", nova_sharpe, top_k=top_k),
        Strategy("alpha_omega", alpha_omega, top_k=top_k),
        Strategy("alpha_omega_deep", alpha_omega_deep, top_k=top_k),
        Strategy("alpha_omega_momentum", alpha_omega_momentum, top_k=top_k),
        Strategy("nova_regime_x", nova_regime_x, top_k=top_k),
        Strategy("nova_dual", nova_dual, top_k=top_k),
        Strategy("nova_star_prime", nova_star_prime, top_k=top_k),
        Strategy("the_bagger", the_bagger, top_k=top_k),
        Strategy("multibagger_max", multibagger_max, top_k=top_k),
        Strategy("apex", apex, top_k=top_k),
        Strategy("nova_tier1", nova_tier1, top_k=top_k),
    ]
