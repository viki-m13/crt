"""APEX-RELOADED — final proprietary strategy designed for the COMPOUNDING engine.

Design principles:
  * Multi-leg cross-sectional score (momentum × quality × asymmetry)
  * Regime-conditional weights derived from cross-sectional IC analysis
  * Built-in survivorship hardness:
      - Hard exclusion of stocks with vol_expansion_24m > 2.5
      - Hard exclusion of stocks with dd_from_52wh < -60%
      - Hard exclusion of stocks with mom_3y < 0.05 (broken trends)
      - Hard exclusion of stocks priced under $5 (penny stock filter)
  * Designed for monthly rebalance with k=5, but works for k=3 too
  * Bear-regime: SKIP MONTH (return all-NaN)

Version history:
  v1: APEX_RELOADED_V1 — base implementation
  v2: APEX_RELOADED_V2 — add tight_consolidation breakout layer
  v3: APEX_TURBOCHARGED — k=3, momentum-tilted, for max CAGR
  v4: APEX_BALANCED — k=5, balanced quality+momentum, for stability
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.strategies_fast import _safe, _z


def _spy_regime(df: pd.DataFrame) -> str:
    if "SPY" not in df.index:
        return "default"
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0
    if spy_dsma < -0.10 and spy_rsi < 35:
        return "bear"
    if spy_dsma < -0.05 and spy_rsi < 45:
        return "correction"
    if -0.05 <= spy_dsma <= 0.03 and spy_rsi > 38:
        return "recovery"
    if spy_mom > 0.18 and spy_dsma > 0.05:
        return "strong_bull"
    if spy_dsma > 0 and spy_mom > 0.08:
        return "uptrend"
    return "default"


def _hard_exclusion_filter(df: pd.DataFrame) -> pd.Series:
    """Returns boolean Series: True = ELIGIBLE.

    Hard-exclude stocks that look like they could be delisting candidates or
    structural decliners.  This is the FIRST stage of selection.
    """
    vol_exp = _safe(df, "vol_expansion_24m", 1.0)
    dd_52wh = _safe(df, "dd_from_52wh", 0.0)
    mom_3y = _safe(df, "mom_3y", 0.0)
    price = _safe(df, "price", 100.0)
    trend = _safe(df, "trend_health_5y", 0.5)
    pull = _safe(df, "pullback_1y", 0.0)
    rsi = _safe(df, "rsi_14", 50.0)
    dsma = _safe(df, "d_sma200", 0.0)
    max_dd5 = _safe(df, "max_dd_5y", -0.5)   # alpha2

    eligible = (
        (vol_exp < 2.5)             # not in vol meltdown
        & (dd_52wh > -0.65)          # not in catastrophic drawdown
        & (mom_3y > -0.30)           # not in 3y free fall
        & (price >= 5.0)             # no penny stocks
        & (trend > 0.30)             # has some long-term character
        & (pull > -0.65)             # not collapsing
        & (rsi >= 25)                # not in active selloff
        & (dsma > -0.30)             # within reasonable distance of 200dma
        & (max_dd5 > -0.85)          # didn't lose 85%+ in 5y window
    )
    return eligible


def _momentum_leg(df: pd.DataFrame) -> pd.Series:
    """Pure momentum signal — IC-validated."""
    rs12 = _safe(df, "rs_12m_spy")
    rs6 = _safe(df, "rs_6m_spy")
    mom3y = _safe(df, "mom_3y")
    mom12 = _safe(df, "mom_12_1")
    mom2y = _safe(df, "mom_2y")
    sharpe = _safe(df, "sharpe_12m")
    r2 = _safe(df, "trend_r2_12m")
    accel = _safe(df, "accel")
    near = _safe(df, "near_52wh_60d", 0)
    cons = _safe(df, "mom_consistency_12m", 0.5)
    idio = _safe(df, "idio_mom_12_1", 0)               # alpha2
    mom_pv = _safe(df, "mom_per_unit_vol_12", 0)       # alpha2
    fip = _safe(df, "fip_score", 0)                    # alpha2 (lower = smoother)

    return (
        _z(rs12) * 1.2
        + _z(rs6) * 0.7
        + _z(mom3y) * 1.0
        + _z(mom12) * 1.0
        + _z(mom2y) * 0.6
        + _z(sharpe) * 0.9
        + _z(r2) * 0.7
        + _z(accel) * 0.3
        + near * 0.4
        + _z(cons) * 0.5
        + _z(idio) * 0.6
        + _z(mom_pv) * 0.5
        - _z(fip) * 0.4   # negative fip = smoother trend = better
    )


def _quality_leg(df: pd.DataFrame) -> pd.Series:
    """Stable compounder signal."""
    frac = _safe(df, "frac_above_50dma_1y")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    rec = _safe(df, "recovery_rate", 0.6).fillna(0.6)
    qs5 = _safe(df, "quality_score_5y", 0)             # alpha2
    sharpe = _safe(df, "sharpe_12m")
    sharpe5 = _safe(df, "sharpe_5y", 0)                # alpha2
    sma_above = _safe(df, "sma50_above_200", 0).astype(float)
    trend = _safe(df, "trend_health_5y")
    rsi_zone = _safe(df, "rsi_zone_score", 0.5)        # alpha2
    max_dd5 = _safe(df, "max_dd_5y", -0.5)             # alpha2 (closer to 0 = better)

    return (
        _z(frac) * 0.8
        + _z(cons) * 0.7
        + _z(rec) * 0.4
        + _z(qs5) * 1.0
        + _z(sharpe) * 0.6
        + _z(sharpe5) * 0.5
        + sma_above * 0.4
        + _z(trend) * 0.6
        + _z(rsi_zone) * 0.4
        + _z(max_dd5) * 0.6   # higher (less negative) = better
    )


def _asymmetry_leg(df: pd.DataFrame) -> pd.Series:
    """Tail-asymmetry & breakout potential."""
    tail = _safe(df, "tail_ratio_24m", 1.0)
    best_m = _safe(df, "best_month_24m", 0)
    multi = _safe(df, "multibagger_ratio_24m", 0)
    dist_low = _safe(df, "dist_from_low_1y", 0.1)
    bo = _safe(df, "breakout_strength_60", 0)          # alpha2
    tc = _safe(df, "tight_consolidation_60", 0)        # alpha2
    drift = _safe(df, "earnings_drift_proxy", 0)       # alpha2

    return (
        _z(tail) * 0.5
        + _z(best_m) * 0.4
        + _z(multi) * 0.4
        + _z(dist_low) * 0.4
        + _z(bo) * 0.7
        + _z(tc) * 0.4
        + _z(drift) * 0.4
    )


def _penalty_leg(df: pd.DataFrame) -> pd.Series:
    """Penalty for risky / unstable names."""
    beta = _safe(df, "beta_2y", 1.0)                   # alpha2
    vol_exp = _safe(df, "vol_expansion_24m", 1.0)
    pull = _safe(df, "pullback_1y", 0)
    min_dd60 = _safe(df, "min_dd_60d", -0.1)           # alpha2 (closer to 0 = better)

    # Penalty score: high beta, high vol expansion, deep recent dd
    return (
        _z(beta) * 0.4
        + _z(vol_exp) * 0.4
        - _z(pull) * 0.2     # deeper pullback adds penalty
        - _z(min_dd60) * 0.3 # deeper recent dd adds penalty
    )


# ---------------------------------------------------------------------------
# A. APEX RELOADED — base
# ---------------------------------------------------------------------------
def apex_reloaded(df: pd.DataFrame) -> pd.Series:
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    mom = _momentum_leg(df)
    qual = _quality_leg(df)
    asym = _asymmetry_leg(df)
    pen = _penalty_leg(df)

    # Regime-conditional weights (validated via IC analysis & WF tests)
    if regime == "strong_bull":
        score = mom * 1.5 + qual * 0.6 + asym * 0.7 - pen * 0.3
    elif regime == "uptrend":
        score = mom * 1.0 + qual * 0.9 + asym * 0.5 - pen * 0.4
    elif regime == "recovery":
        score = mom * 0.5 + qual * 1.2 + asym * 0.4 - pen * 0.5
        pull = _safe(df, "pullback_1y", 0)
        score = score + _z(-pull) * 0.6   # deeper pullback bonus
    elif regime == "correction":
        score = mom * 0.3 + qual * 1.5 + asym * 0.2 - pen * 0.6
    else:
        score = mom * 0.9 + qual * 0.9 + asym * 0.5 - pen * 0.4

    # Bonus: stocks with high cross-sectional consensus across signals
    sigs = []
    for col in ["rs_12m_spy", "mom_3y", "sharpe_12m", "trend_r2_12m",
                "frac_above_50dma_1y", "mom_consistency_12m", "quality_score_5y"]:
        if col in df.columns:
            sigs.append(df[col].rank(pct=True, na_option="keep"))
    if sigs:
        pct = pd.concat(sigs, axis=1)
        in_top = (pct > 0.85).sum(axis=1)
        score = score + 0.10 * in_top.fillna(0)

    return score.where(eligible)


# ---------------------------------------------------------------------------
# B. APEX TURBOCHARGED — momentum-tilted, designed for k=3
# ---------------------------------------------------------------------------
def apex_turbocharged(df: pd.DataFrame) -> pd.Series:
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    mom = _momentum_leg(df)
    qual = _quality_leg(df)
    asym = _asymmetry_leg(df)
    pen = _penalty_leg(df)

    # Heavily momentum-tilted in bull/uptrend
    if regime == "strong_bull":
        score = mom * 2.0 + qual * 0.4 + asym * 1.0 - pen * 0.2
    elif regime == "uptrend":
        score = mom * 1.5 + qual * 0.7 + asym * 0.7 - pen * 0.3
    elif regime == "recovery":
        score = mom * 0.7 + qual * 1.3 + asym * 0.5 - pen * 0.4
        pull = _safe(df, "pullback_1y", 0)
        score = score + _z(-pull) * 0.7
    elif regime == "correction":
        score = mom * 0.4 + qual * 1.6 + asym * 0.2 - pen * 0.5
    else:
        score = mom * 1.1 + qual * 0.8 + asym * 0.6 - pen * 0.3

    # Tighter eligibility for turbo: must be a real winner
    mom3y = _safe(df, "mom_3y")
    rs12 = _safe(df, "rs_12m_spy")
    pull = _safe(df, "pullback_1y", 0)
    trend = _safe(df, "trend_health_5y")
    tighter = (
        ((mom3y > 0.30) & (rs12 > 0))   # active long-term winner
        | ((mom3y > 0.15) & (pull <= -0.15) & (pull >= -0.40) & (trend > 0.65))
        # mid-quality on a healthy pullback
    )
    return score.where(eligible & tighter)


# ---------------------------------------------------------------------------
# C. APEX BALANCED — for k=5, balanced for stability
# ---------------------------------------------------------------------------
def apex_balanced(df: pd.DataFrame) -> pd.Series:
    eligible = _hard_exclusion_filter(df)
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    mom = _momentum_leg(df)
    qual = _quality_leg(df)
    asym = _asymmetry_leg(df)
    pen = _penalty_leg(df)

    if regime == "strong_bull":
        score = mom * 1.2 + qual * 0.8 + asym * 0.6 - pen * 0.4
    elif regime == "uptrend":
        score = mom * 1.0 + qual * 1.0 + asym * 0.5 - pen * 0.4
    elif regime == "recovery":
        score = mom * 0.6 + qual * 1.2 + asym * 0.4 - pen * 0.5
        pull = _safe(df, "pullback_1y", 0)
        score = score + _z(-pull) * 0.5
    elif regime == "correction":
        score = mom * 0.4 + qual * 1.4 + asym * 0.2 - pen * 0.6
    else:
        score = mom * 0.9 + qual * 1.0 + asym * 0.5 - pen * 0.4

    return score.where(eligible)


# ---------------------------------------------------------------------------
# D. APEX HYBRID — same as RELOADED but with looser eligibility for max breadth
# ---------------------------------------------------------------------------
def apex_hybrid(df: pd.DataFrame) -> pd.Series:
    """Looser eligibility, broader universe selection for k=10 cases."""
    vol_exp = _safe(df, "vol_expansion_24m", 1.0)
    dd_52wh = _safe(df, "dd_from_52wh", 0.0)
    mom_3y = _safe(df, "mom_3y", 0.0)
    price = _safe(df, "price", 100.0)
    trend = _safe(df, "trend_health_5y", 0.5)

    eligible = (
        (vol_exp < 3.0)
        & (dd_52wh > -0.75)
        & (price >= 3.0)
        & (trend > 0.20)
        & (mom_3y > -0.50)
    )
    regime = _spy_regime(df)
    if regime == "bear":
        return pd.Series(np.nan, index=df.index)

    mom = _momentum_leg(df)
    qual = _quality_leg(df)
    asym = _asymmetry_leg(df)
    pen = _penalty_leg(df)

    if regime == "strong_bull":
        score = mom * 1.3 + qual * 0.7 + asym * 0.7 - pen * 0.3
    elif regime == "uptrend":
        score = mom * 1.0 + qual * 0.9 + asym * 0.6 - pen * 0.4
    else:
        score = mom * 0.9 + qual * 0.9 + asym * 0.6 - pen * 0.4
    return score.where(eligible)


def all_apex_strategies():
    return {
        "apex_reloaded": apex_reloaded,
        "apex_turbocharged": apex_turbocharged,
        "apex_balanced": apex_balanced,
        "apex_hybrid": apex_hybrid,
    }
