"""
Regime-conditional scoring:
- normal: ML score (existing v3 model)
- recovery: ML score + value emphasis (the model's strength on rebounds)
- bull: trend-following (mom_per_unit_vol_12 + earnings_drift + breakout)
- crash: cash (handled by engine)

The downstream engine treats "regime" via gating; here we pre-compute multiple
score panels and select based on regime per-asof.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_factory import (
    load_panel, load_mlpreds,
    s_concretum_trend, s_alpha_apex, s_alpha_apex_v2,
)
ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
from lib_engine import REGIMES, load_spy_features  # type: ignore


def build_regime_score_panel(
    bull_strategy: str = "alpha_apex",
    recovery_strategy: str = "ml",
    normal_strategy: str = "ml",
    regime_gate: str = "tight",
) -> pd.DataFrame:
    """For each (asof, ticker), compute a regime-aware score."""
    panel = load_panel()
    ml = load_mlpreds()
    spy = load_spy_features()
    cls_fn = REGIMES[regime_gate]

    # Per-asof regime
    regime_by_asof = {}
    for d in spy.index:
        regime_by_asof[d] = cls_fn(spy.loc[d].to_dict())

    # Pre-compute scores per strategy
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
    p["ml_rank"] = p.groupby("asof")["ml_score"].rank(pct=True)
    p["concretum_rank"] = s_concretum_trend(panel)
    p["apex_rank"] = s_alpha_apex(panel)
    p["apex_v2_rank"] = s_alpha_apex_v2(panel)

    # Map asof to regime
    p["regime"] = p["asof"].map(regime_by_asof).fillna("normal")

    # Build score by regime
    def pick(row, strat):
        if strat == "ml":
            return row["ml_rank"]
        if strat == "concretum":
            return row["concretum_rank"]
        if strat == "alpha_apex":
            return row["apex_rank"]
        if strat == "alpha_apex_v2":
            return row["apex_v2_rank"]
        if strat == "ml_x_apex":
            return (row["ml_rank"] + row["apex_rank"]) / 2
        if strat == "ml_x_concretum":
            return (row["ml_rank"] + row["concretum_rank"]) / 2
        return row["ml_rank"]

    score = np.zeros(len(p))
    for strat, key in [(bull_strategy, "bull"),
                       (recovery_strategy, "recovery"),
                       (normal_strategy, "normal")]:
        mask = p["regime"] == key
        score[mask] = p.loc[mask].apply(lambda r: pick(r, strat), axis=1).values

    p["score"] = score
    out = p[["asof", "ticker", "score", "vol_1y", "mom_12_1", "pullback_1y",
             "trend_health_5y", "d_sma200"]].copy()
    out = out.dropna(subset=["score"])
    out["vol_rank"] = out.groupby("asof")["vol_1y"].rank(pct=True)
    return out


if __name__ == "__main__":
    sp = build_regime_score_panel(
        bull_strategy="alpha_apex",
        recovery_strategy="ml",
        normal_strategy="ml",
    )
    print("Score panel:", sp.shape)
    print(sp.head())
