"""V2 selection logic: integrate novel CRT/RBI/archetype signals into the
existing regime structure (strategy_rotation from REPORT.md §3).

Regimes:
  Deep bear (SPY d_sma200 < -0.10 AND rsi_14 < 35):
    -> ALL CASH (return NaN)
  Recovery (-0.05 < SPY d_sma200 < 0.03):
    -> footprint_pullback: deep-pullback long-term winners with novel signals
       on top.  Replaces the existing pullback_in_winner.
  Strong bull (SPY mom_12_1 > 0.15):
    -> momentum_with_crt: top momentum names re-ranked by CRT.
  Default uptrend / sideways:
    -> quality_pullback_with_crt: existing quality_pullback gates with the
       composite as a tie-break.

Each leg uses the SAME 5-component composite for ranking when possible:
  composite = 0.40 * z(crt_6m)
            + 0.20 * z(rbi_60)
            + 0.20 * (-z(prerunner_dist))
            + 0.10 * z(cst_score)
            + 0.10 * z(trend_health_5y)

This lets the novel signal replace the legacy `score = trend*pullback*rs`
formula in each leg, while keeping the proven regime classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from strategy.selection import zscore, quantile


def composite_score(df: pd.DataFrame) -> pd.Series:
    """Z-scaled 5-component composite — same as the no-gate diagnostic."""
    needed = ["crt_6m", "rbi_60", "prerunner_dist", "cst_score", "trend_health_5y"]
    if not all(c in df.columns for c in needed):
        return pd.Series(np.nan, index=df.index)
    z_crt = zscore(df["crt_6m"])
    z_rbi = zscore(df["rbi_60"])
    z_arch = -zscore(df["prerunner_dist"])
    z_cst = zscore(df["cst_score"])
    z_qual = zscore(df["trend_health_5y"])
    return 0.40 * z_crt + 0.20 * z_rbi + 0.20 * z_arch + 0.10 * z_cst + 0.10 * z_qual


def _legacy_pullback_in_winner(df: pd.DataFrame) -> pd.Series:
    """Replicate the legacy gate: trend_health_5y > 0.55, pullback_1y < -0.15,
    accel > 0.  Score by  trend_health * (-pullback_1y) * accel * dist_from_low_1y."""
    out = pd.Series(np.nan, index=df.index)
    needed = ["trend_health_5y", "pullback_1y", "accel"]
    if not all(c in df.columns for c in needed):
        return out
    gate = (
        (df["trend_health_5y"] > 0.55) &
        (df["pullback_1y"] < -0.15) &
        (df["accel"] > 0)
    )
    out.loc[gate.fillna(False)] = (
        df.loc[gate, "trend_health_5y"]
        * (-df.loc[gate, "pullback_1y"])
        * (df.loc[gate, "accel"].clip(lower=0))
    )
    return out


def _legacy_explosive_winners(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    needed = ["mom_12_1", "d_sma200", "rsi_14"]
    if not all(c in df.columns for c in needed):
        return out
    gate = (
        (df["mom_12_1"] > 0.10) &
        (df["d_sma200"] > 0) &
        (df["rsi_14"] > 50) &
        (df["rsi_14"] < 80)
    )
    out.loc[gate.fillna(False)] = df.loc[gate, "mom_12_1"]
    return out


def _legacy_quality_pullback(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    needed = ["trend_health_5y", "pullback_1y", "accel", "recovery_rate"]
    if not all(c in df.columns for c in needed):
        return out
    gate = (
        (df["trend_health_5y"] > 0.50) &
        (df["pullback_1y"] < -0.05) &
        (df["pullback_1y"] > -0.50) &
        (df["accel"] > 0) &
        (df["recovery_rate"] > 0.50)
    )
    out.loc[gate.fillna(False)] = (
        df.loc[gate, "trend_health_5y"] *
        (-df.loc[gate, "pullback_1y"]) *
        df.loc[gate, "recovery_rate"]
    )
    return out


def regime_v2_score(df: pd.DataFrame) -> pd.Series:
    """V2 score function: regime-conditional with CRT-augmented sub-strategies.

    The hybrid: in each regime, take the legacy gate; on the gated subset,
    rank by the novel composite (CRT + RBI + archetype + CST + quality).
    The gated subset selects the right pool; the composite picks the best
    of that pool.
    """
    out = pd.Series(np.nan, index=df.index)
    if "SPY" not in df.index:
        return out
    spy = df.loc["SPY"]
    spy_dsma = spy.get("d_sma200", np.nan)
    spy_rsi = spy.get("rsi_14", np.nan)
    spy_mom = spy.get("mom_12_1", np.nan)

    # Defensive bear
    if (np.isfinite(spy_dsma) and spy_dsma < -0.10 and
        np.isfinite(spy_rsi) and spy_rsi < 35):
        return out

    # Compute legacy leg gate
    if np.isfinite(spy_dsma) and -0.05 < spy_dsma < 0.03:
        leg_score = _legacy_pullback_in_winner(df)
    elif np.isfinite(spy_mom) and spy_mom > 0.15:
        leg_score = _legacy_explosive_winners(df)
    else:
        leg_score = _legacy_quality_pullback(df)

    # Now augment: where leg_score is finite (i.e. passed the legacy gate),
    # blend with the composite.  composite is z-scored across the whole
    # universe, so we add the value where the gate passes.
    comp = composite_score(df)
    # On gated subset, score = leg_score's rank * 0.5 + composite * 0.5
    gated = leg_score.dropna().index
    if len(gated) == 0:
        return out
    leg_rank = leg_score.rank(pct=True)  # higher = better
    blended = pd.Series(np.nan, index=df.index)
    blended.loc[gated] = (0.5 * leg_rank.loc[gated] + 0.5 * (comp.loc[gated]
                                                              .fillna(0) / 5.0
                                                              + 0.5))

    # Exclude bench
    for ex in ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"):
        if ex in blended.index:
            blended.loc[ex] = np.nan
    return blended


def regime_v2_pure_composite(df: pd.DataFrame) -> pd.Series:
    """Pure composite within the legacy regime gate (no rank blending)."""
    out = pd.Series(np.nan, index=df.index)
    if "SPY" not in df.index:
        return out
    spy = df.loc["SPY"]
    spy_dsma = spy.get("d_sma200", np.nan)
    spy_rsi = spy.get("rsi_14", np.nan)
    spy_mom = spy.get("mom_12_1", np.nan)

    if (np.isfinite(spy_dsma) and spy_dsma < -0.10 and
        np.isfinite(spy_rsi) and spy_rsi < 35):
        return out

    if np.isfinite(spy_dsma) and -0.05 < spy_dsma < 0.03:
        leg_score = _legacy_pullback_in_winner(df)
    elif np.isfinite(spy_mom) and spy_mom > 0.15:
        leg_score = _legacy_explosive_winners(df)
    else:
        leg_score = _legacy_quality_pullback(df)

    comp = composite_score(df)
    gated = leg_score.dropna().index
    if len(gated) == 0:
        return out
    out_score = pd.Series(np.nan, index=df.index)
    out_score.loc[gated] = comp.loc[gated]
    for ex in ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"):
        if ex in out_score.index:
            out_score.loc[ex] = np.nan
    return out_score


def regime_v2_no_gate_composite(df: pd.DataFrame) -> pd.Series:
    """Composite with regime cash defense, but no legacy gate."""
    out = pd.Series(np.nan, index=df.index)
    if "SPY" not in df.index:
        return out
    spy = df.loc["SPY"]
    spy_dsma = spy.get("d_sma200", np.nan)
    spy_rsi = spy.get("rsi_14", np.nan)
    if (np.isfinite(spy_dsma) and spy_dsma < -0.10 and
        np.isfinite(spy_rsi) and spy_rsi < 35):
        return out
    out = composite_score(df)
    for ex in ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"):
        if ex in out.index:
            out.loc[ex] = np.nan
    return out


def make_v2_strategy(top_k: int = 5, variant: str = "blended"):
    from experiments.monthly_dca.compound_engine import Strategy
    if variant == "blended":
        fn = regime_v2_score
        name = "regime_v2_blended"
    elif variant == "pure":
        fn = regime_v2_pure_composite
        name = "regime_v2_pure_composite"
    elif variant == "no_gate":
        fn = regime_v2_no_gate_composite
        name = "regime_v2_no_gate"
    else:
        raise ValueError(variant)
    return Strategy(name=name, score_fn=fn, top_k=top_k,
                    description=f"FHtzX v2 {variant}")
