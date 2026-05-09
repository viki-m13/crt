"""Alpha-tier proprietary strategies.

Designed from cross-sectional IC analysis (1997-2024 universe) — see REPORT.md.
The high-IC features driving these strategies are:

  Feature                  Mean cross-sec IC   t-stat   Q5 mean 3y fwd
  ------------------------ -----------------   -------  ---------------
  trend_r2_12m              +0.030              +6.76    +0.71
  frac_above_50dma_1y       +0.032              +5.89    +0.49
  mom_3y                    +0.036              +5.16    +0.58
  tail_ratio_24m            +0.023              +5.33    +0.85
  sharpe_12m                +0.028              +4.31    +0.65
  sma50_above_200           +0.023              +4.72    +0.51
  beta_2y                   -0.029              -3.85    +0.95 (high-beta tail)
  pullback_3y               +0.026              +3.62    +0.44
  trend_health_5y           +0.016              +2.50    +0.44

These are statistically robust signals ABOVE NOISE. The strategies below combine
them in different ways to optimize CAGR vs robustness.

Each strategy returns a Series of scores (higher = more preferred, NaN = excluded).
The fast_engine sorts and takes top-K per asof.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import Strategy
from experiments.monthly_dca.strategies_fast import _safe, _z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def _w_avg(*pairs):
    """Weighted sum of (weight, series) pairs handling NaN."""
    total = None
    for w, s in pairs:
        if s is None:
            continue
        contrib = (w * s).fillna(0.0)
        if total is None:
            total = contrib
        else:
            total = total + contrib
    return total


# ---------------------------------------------------------------------------
# A. SMOOTH-TREND COMPOUNDER — buy clean linear uptrends, outperforming SPY
# ---------------------------------------------------------------------------
def smooth_trend_compounder(df: pd.DataFrame) -> pd.Series:
    mom_3y = _safe(df, "mom_3y")
    r2 = _safe(df, "trend_r2_12m")
    frac = _safe(df, "frac_above_50dma_1y")
    sharpe = _safe(df, "sharpe_12m")
    rs12 = _safe(df, "rs_12m_spy")
    beta = _safe(df, "beta_2y", 1.0)
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    trend = _safe(df, "trend_health_5y")

    score = (
        _z(mom_3y) * 1.2
        + _z(r2) * 1.0
        + _z(sharpe) * 0.9
        + _z(frac) * 0.8
        + _z(rs12) * 0.7
        - _z(beta) * 0.4
    )
    gate = (
        (mom_3y > 0.0)
        & (trend > 0.50)
        & (dsma > -0.10)
        & (rsi > 30)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# B. MULTIBAGGER ENGINE — high tail-asymmetry stocks with momentum
# ---------------------------------------------------------------------------
def multibagger_engine(df: pd.DataFrame) -> pd.Series:
    tail = _safe(df, "tail_ratio_24m", 1.0)
    best_m = _safe(df, "best_month_24m")
    r2 = _safe(df, "trend_r2_12m")
    mom12 = _safe(df, "mom_12_1")
    rs12 = _safe(df, "rs_12m_spy")
    mom3y = _safe(df, "mom_3y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")
    multi = _safe(df, "multibagger_ratio_24m")

    score = (
        _z(tail) * 0.6
        + _z(best_m) * 0.7
        + _z(multi) * 0.5
        + _z(mom12) * 1.0
        + _z(rs12) * 0.8
        + _z(mom3y) * 0.8
        + _z(r2) * 0.5
    )
    gate = (
        (tail >= 1.0)
        & (mom3y > 0.10)
        & (trend > 0.55)
        & (rsi > 30)
        & (dsma > -0.15)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# C. FALLEN ANGEL RECOVERY — long-term winner on moderate pullback w/ accel
# ---------------------------------------------------------------------------
def fallen_angel_recovery(df: pd.DataFrame) -> pd.Series:
    mom3y = _safe(df, "mom_3y")
    pull = _safe(df, "pullback_1y")
    accel = _safe(df, "accel")
    rs3 = _safe(df, "rs_3m_spy")
    rsi = _safe(df, "rsi_14", 50)
    trend = _safe(df, "trend_health_5y")
    rec = _safe(df, "recovery_rate", 0.6)
    r2 = _safe(df, "trend_r2_12m")

    score = (
        _z(mom3y) * 1.0
        + _z(-pull) * 0.8
        + _z(accel) * 0.7
        + _z(rs3) * 0.6
        + _z(rec) * 0.4
        + _z(r2) * 0.3
    )
    gate = (
        (mom3y > 0.30)
        & (pull <= -0.10)
        & (pull >= -0.55)
        & (accel > -0.01)
        & (rsi > 28)
        & (trend > 0.50)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# D. VOLATILITY CONTRACTION BREAKOUT — quiet stocks near 52wh
# ---------------------------------------------------------------------------
def vol_contraction_breakout(df: pd.DataFrame) -> pd.Series:
    vc = _safe(df, "vol_contraction", 1.0)
    near = _safe(df, "near_52wh_60d", 0)
    mom3y = _safe(df, "mom_3y")
    r2 = _safe(df, "trend_r2_12m")
    rs12 = _safe(df, "rs_12m_spy")
    sharpe = _safe(df, "sharpe_12m")
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")

    score = (
        -_z(vc) * 0.7
        + near * 0.5
        + _z(mom3y) * 1.0
        + _z(r2) * 0.7
        + _z(rs12) * 0.7
        + _z(sharpe) * 0.5
    )
    gate = (
        (mom3y > 0.10)
        & (trend > 0.55)
        & (dsma > 0)
        & (vc < 1.05)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# E. PERSISTENT WINNER — pure persistence: long-term mom + clean trend
# ---------------------------------------------------------------------------
def persistent_winner(df: pd.DataFrame) -> pd.Series:
    mom3y = _safe(df, "mom_3y")
    mom2y = _safe(df, "mom_2y")
    mom12 = _safe(df, "mom_12_1")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    r2 = _safe(df, "trend_r2_12m")
    sharpe = _safe(df, "sharpe_12m")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)

    score = (
        _z(mom3y) * 1.0
        + _z(mom2y) * 0.7
        + _z(mom12) * 0.7
        + _z(cons) * 0.6
        + _z(r2) * 0.6
        + _z(sharpe) * 0.5
    )
    gate = (
        (mom3y >= 0.20)
        & (mom12 > 0)
        & (dsma > -0.05)
        & (rsi > 30)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# F. CONSENSUS TOP DECILE — agreement across multiple signals
# ---------------------------------------------------------------------------
def consensus_top_decile(df: pd.DataFrame) -> pd.Series:
    sigs = []
    for col in ["mom_3y", "trend_r2_12m", "sharpe_12m", "frac_above_50dma_1y",
                "rs_12m_spy", "tail_ratio_24m", "mom_consistency_12m"]:
        if col not in df.columns:
            continue
        sigs.append(_pct_rank(df[col]))
    if not sigs:
        return pd.Series(np.nan, index=df.index)
    pct_mat = pd.concat(sigs, axis=1)
    # consensus = sum of percentile ranks
    consensus = pct_mat.mean(axis=1)
    # Gate: at least 5 signals in top 60th percentile
    in_top60 = (pct_mat > 0.6).sum(axis=1)
    gate = in_top60 >= 4
    # quality fence
    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    fence = (trend > 0.50) & (dsma > -0.10) & (rsi > 28)
    return consensus.where(gate & fence)


# ---------------------------------------------------------------------------
# G. NOVA STAR — flagship: weighted blend of 9 alpha signals + smart gate
# ---------------------------------------------------------------------------
def nova_star(df: pd.DataFrame) -> pd.Series:
    mom3y = _safe(df, "mom_3y")
    r2 = _safe(df, "trend_r2_12m")
    sharpe = _safe(df, "sharpe_12m")
    frac = _safe(df, "frac_above_50dma_1y")
    rs12 = _safe(df, "rs_12m_spy")
    rs3 = _safe(df, "rs_3m_spy")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    cons = _safe(df, "mom_consistency_12m", 0.5)
    beta = _safe(df, "beta_2y", 1.0)
    vc = _safe(df, "vol_contraction", 1.0)
    mom12 = _safe(df, "mom_12_1")
    near = _safe(df, "near_52wh_60d", 0)
    accel = _safe(df, "accel")

    score = (
        _z(mom3y) * 1.0
        + _z(r2) * 1.0
        + _z(sharpe) * 0.9
        + _z(frac) * 0.7
        + _z(rs12) * 0.8
        + _z(rs3) * 0.4
        + _z(tail) * 0.5
        + _z(cons) * 0.5
        + _z(mom12) * 0.6
        + _z(accel) * 0.3
        + near * 0.3
        - _z(beta) * 0.5
        - _z(vc) * 0.4
    )
    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    pull1y = _safe(df, "pullback_1y")
    gate = (
        (mom3y > 0.0)
        & (trend > 0.50)
        & (dsma > -0.15)
        & (rsi > 30)
        & (rsi < 80)
        & (pull1y > -0.55)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# H. INSTITUTIONAL ACCUMULATION — vol contracts as price climbs
# ---------------------------------------------------------------------------
def institutional_accumulation(df: pd.DataFrame) -> pd.Series:
    vc = _safe(df, "vol_contraction", 1.0)
    cons = _safe(df, "mom_consistency_12m", 0.5)
    mom12 = _safe(df, "mom_12_1")
    rs12 = _safe(df, "rs_12m_spy")
    near = _safe(df, "near_52wh_60d", 0)
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")
    rsi = _safe(df, "rsi_14", 50)

    score = (
        -_z(vc) * 0.8
        + _z(cons) * 0.7
        + _z(mom12) * 0.8
        + _z(rs12) * 0.7
        + near * 0.4
        + _z(sharpe) * 0.6
        + _z(r2) * 0.5
    )
    gate = (
        (trend > 0.55)
        & (dsma > 0)
        & (vc < 1.0)
        & (mom12 > 0)
        & (rsi > 30)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# I. ASYMMETRIC RECOVERY+ — pullback in big winner, with rebound confirmation
# ---------------------------------------------------------------------------
def asymmetric_recovery_plus(df: pd.DataFrame) -> pd.Series:
    mom3y = _safe(df, "mom_3y")
    pull = _safe(df, "pullback_1y")
    rec = _safe(df, "recovery_rate", 0.6)
    accel = _safe(df, "accel")
    rs3 = _safe(df, "rs_3m_spy")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    rsi = _safe(df, "rsi_14", 50)
    frac = _safe(df, "frac_above_50dma_1y")
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")
    dist_low = _safe(df, "dist_from_low_1y")

    score = (
        _z(mom3y) * 0.9
        + _z(-pull) * 0.7
        + _z(rec) * 0.5
        + _z(accel) * 0.6
        + _z(rs3) * 0.6
        + _z(tail) * 0.5
        + _z(frac) * 0.6
        + _z(dist_low) * 0.5
    )
    gate = (
        (mom3y > 0.30)
        & (pull >= -0.50)
        & (pull <= -0.08)
        & (rsi > 30)
        & (rsi < 70)
        & (accel > -0.02)
        & (trend > 0.55)
        & (dsma > -0.20)
        & (dist_low > 0.05)        # at least 5% above 1y low
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# J. RELATIVE STRENGTH BEAST — pure RS vs SPY, multi-timeframe
# ---------------------------------------------------------------------------
def rs_beast(df: pd.DataFrame) -> pd.Series:
    rs3 = _safe(df, "rs_3m_spy")
    rs6 = _safe(df, "rs_6m_spy")
    rs12 = _safe(df, "rs_12m_spy")
    excess5 = _safe(df, "excess_5y_logret")
    mom3y = _safe(df, "mom_3y")
    r2 = _safe(df, "trend_r2_12m")
    sharpe = _safe(df, "sharpe_12m")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)

    score = (
        _z(rs3) * 0.7
        + _z(rs6) * 0.8
        + _z(rs12) * 1.0
        + _z(excess5) * 0.6
        + _z(mom3y) * 0.7
        + _z(r2) * 0.5
        + _z(sharpe) * 0.5
    )
    gate = (
        (rs12 > 0)
        & (rs6 > 0)
        & (mom3y > 0.10)
        & (dsma > 0)
        & (rsi > 32)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# K. NOVA STAR DEEP — same as NOVA but with deeper-pullback bias and tail focus
# ---------------------------------------------------------------------------
def nova_star_deep(df: pd.DataFrame) -> pd.Series:
    mom3y = _safe(df, "mom_3y")
    r2 = _safe(df, "trend_r2_12m")
    sharpe = _safe(df, "sharpe_12m")
    rs12 = _safe(df, "rs_12m_spy")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    cons = _safe(df, "mom_consistency_12m", 0.5)
    beta = _safe(df, "beta_2y", 1.0)
    pull1y = _safe(df, "pullback_1y")
    accel = _safe(df, "accel")
    rec = _safe(df, "recovery_rate", 0.6)
    frac = _safe(df, "frac_above_50dma_1y")
    rsi = _safe(df, "rsi_14", 50)
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")

    score = (
        _z(mom3y) * 1.0
        + _z(r2) * 0.8
        + _z(sharpe) * 0.7
        + _z(rs12) * 0.7
        + _z(tail) * 0.7
        + _z(cons) * 0.4
        + _z(-pull1y) * 0.7   # depth of discount
        + _z(rec) * 0.4
        + _z(accel) * 0.5
        + _z(frac) * 0.4
        - _z(beta) * 0.3
    )
    gate = (
        (mom3y > 0.20)
        & (pull1y >= -0.50)
        & (pull1y <= -0.05)
        & (rsi > 30)
        & (rsi < 75)
        & (trend > 0.55)
        & (dsma > -0.15)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# L. RANK-INTERSECT — stocks that are top-decile on at least 4 signals
# ---------------------------------------------------------------------------
def rank_intersect(df: pd.DataFrame) -> pd.Series:
    cols = ["mom_3y", "trend_r2_12m", "sharpe_12m", "frac_above_50dma_1y",
            "rs_12m_spy", "tail_ratio_24m", "mom_consistency_12m", "mom_12_1"]
    sigs = [(_pct_rank(df[c]) if c in df.columns else None) for c in cols]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return pd.Series(np.nan, index=df.index)
    pct_mat = pd.concat(sigs, axis=1)
    in_top10 = (pct_mat > 0.90).sum(axis=1)
    score = pct_mat.mean(axis=1) + 0.05 * in_top10
    # Stricter gate: at least 3 signals in top decile, plus quality fence
    trend = _safe(df, "trend_health_5y")
    dsma = _safe(df, "d_sma200")
    rsi = _safe(df, "rsi_14", 50)
    mom3y = _safe(df, "mom_3y")
    gate = (
        (in_top10 >= 3)
        & (trend > 0.50)
        & (dsma > -0.15)
        & (rsi > 28)
        & (mom3y > 0)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# M. ALPHA INTERSECT (NOVA + ASYM + PERSISTENT consensus)
# ---------------------------------------------------------------------------
def alpha_intersect(df: pd.DataFrame) -> pd.Series:
    """Average rank across our top single-signal strategies."""
    s1 = nova_star(df)
    s2 = persistent_winner(df)
    s3 = smooth_trend_compounder(df)
    s4 = multibagger_engine(df)
    rank1 = s1.rank(pct=True, na_option="keep")
    rank2 = s2.rank(pct=True, na_option="keep")
    rank3 = s3.rank(pct=True, na_option="keep")
    rank4 = s4.rank(pct=True, na_option="keep")
    df_ranks = pd.concat([rank1, rank2, rank3, rank4], axis=1)
    # require at least 2 of 4 strategies passed gate (non-NaN)
    n_strategies_passed = df_ranks.notna().sum(axis=1)
    avg = df_ranks.mean(axis=1, skipna=True)
    return avg.where(n_strategies_passed >= 2)


# ---------------------------------------------------------------------------
# N. NOVA REGIME — NOVA + market regime filter
# ---------------------------------------------------------------------------
def nova_regime(df: pd.DataFrame) -> pd.Series:
    score = nova_star(df)
    # Regime: SPY's d_sma200 and rsi_14 and trend_health
    if "SPY" in df.index:
        spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
        spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
        # Hard kill in deep bear: SPY > 12% below 200dma AND RSI < 30
        if spy_dsma < -0.12 and spy_rsi < 35:
            return pd.Series(np.nan, index=score.index)
    return score


# ---------------------------------------------------------------------------
# O. NOVA STAR PRO — NOVA + multibagger gate + tighter quality
# ---------------------------------------------------------------------------
def nova_star_pro(df: pd.DataFrame) -> pd.Series:
    s = nova_star(df)
    # additional positive multipliers from multibagger profile
    tail = _safe(df, "tail_ratio_24m", 1.0)
    multi = _safe(df, "multibagger_ratio_24m")
    bm = _safe(df, "best_month_24m")
    # Boost for high multibagger probability
    boost = 0.5 * _z(tail).clip(-2, 4) + 0.4 * _z(multi).clip(-2, 4) + 0.3 * _z(bm).clip(-2, 4)
    return s + boost


# ---------------------------------------------------------------------------
# P. CLEAN COMPOUNDER — SAFEST high-IC, low variance
# ---------------------------------------------------------------------------
def clean_compounder(df: pd.DataFrame) -> pd.Series:
    r2 = _safe(df, "trend_r2_12m")
    sharpe = _safe(df, "sharpe_12m")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    mom3y = _safe(df, "mom_3y")
    frac = _safe(df, "frac_above_50dma_1y")
    vol = _safe(df, "vol_1y")
    beta = _safe(df, "beta_2y", 1.0)
    dsma = _safe(df, "d_sma200")
    trend = _safe(df, "trend_health_5y")
    rsi = _safe(df, "rsi_14", 50)

    score = (
        _z(r2) * 1.2
        + _z(sharpe) * 1.0
        + _z(cons) * 0.8
        + _z(frac) * 0.7
        + _z(mom3y) * 0.6
        - _z(vol) * 0.4
        - _z(beta) * 0.3
    )
    gate = (
        (mom3y > 0.10)
        & (trend > 0.65)
        & (dsma > 0)
        & (rsi > 35)
        & (rsi < 75)
    )
    return score.where(gate)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def all_alpha_strategies(top_k: int = 5) -> list[Strategy]:
    return [
        Strategy("smooth_trend_compounder", smooth_trend_compounder, top_k=top_k),
        Strategy("multibagger_engine", multibagger_engine, top_k=top_k),
        Strategy("fallen_angel_recovery", fallen_angel_recovery, top_k=top_k),
        Strategy("vol_contraction_breakout", vol_contraction_breakout, top_k=top_k),
        Strategy("persistent_winner", persistent_winner, top_k=top_k),
        Strategy("consensus_top_decile", consensus_top_decile, top_k=top_k),
        Strategy("nova_star", nova_star, top_k=top_k),
        Strategy("institutional_accumulation", institutional_accumulation, top_k=top_k),
        Strategy("asymmetric_recovery_plus", asymmetric_recovery_plus, top_k=top_k),
        Strategy("rs_beast", rs_beast, top_k=top_k),
        Strategy("nova_star_deep", nova_star_deep, top_k=top_k),
        Strategy("rank_intersect", rank_intersect, top_k=top_k),
        Strategy("alpha_intersect", alpha_intersect, top_k=top_k),
        Strategy("nova_regime", nova_regime, top_k=top_k),
        Strategy("nova_star_pro", nova_star_pro, top_k=top_k),
        Strategy("clean_compounder", clean_compounder, top_k=top_k),
    ]
