"""
v8 — Strategy runner.

Wraps the v6 engine and runs many strategy specs in parallel-friendly form,
exporting WF metrics so we can compare and pick winners.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_engine import (  # type: ignore
    V6Config,
    simulate,
    evaluate,
    build_spy_aligned,
    load_spy_features,
    REGIMES,
)
from score_factory import build_score_panel  # noqa: E402


CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
RESULTS = ROOT / "experiments" / "monthly_dca" / "v8b" / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


def load_monthly_returns() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")


@dataclass
class StratSpec:
    name: str
    strategy: str
    weights: dict | None = None
    k_normal: int = 3
    k_recovery: int = 3
    k_bull: int = 3
    weighting: str = "ew"           # ew | invvol | conv | softmax
    regime_gate: str = "tight"
    hold_months: int = 6
    cost_bps: float = 10.0
    cash_yield_yr: float = 0.03
    pullback_filter: float = 0.0
    min_pick_mom: float = 0.0
    quality_blend: float = 0.0
    spy_dd_scale: float = 0.0
    crash_persist: int = 1
    monthly_exposure: bool = False


def run_one(spec: StratSpec, mr: pd.DataFrame, spy_feat: pd.DataFrame) -> dict:
    sp = build_score_panel(spec.strategy, weights=spec.weights)
    cfg = V6Config(
        name=spec.name,
        scorer="custom",
        regime_gate=spec.regime_gate,
        k_normal=spec.k_normal,
        k_recovery=spec.k_recovery,
        k_bull=spec.k_bull,
        weighting=spec.weighting,
        hold_months=spec.hold_months,
        cost_bps=spec.cost_bps,
        cash_yield_yr=spec.cash_yield_yr,
        pullback_filter=spec.pullback_filter,
        min_pick_mom=spec.min_pick_mom,
        quality_blend=spec.quality_blend,
        spy_dd_scale=spec.spy_dd_scale,
        crash_persist=spec.crash_persist,
        monthly_exposure=spec.monthly_exposure,
    )
    eq = simulate(cfg, sp, mr, spy_feat)
    spy_aligned = build_spy_aligned(eq, mr)
    metrics = evaluate(eq, spy_aligned, name=spec.name)
    return metrics, eq


def benchmark(specs: list[StratSpec], save_prefix: str = "bench"):
    mr = load_monthly_returns()
    spy_feat = load_spy_features()
    rows = []
    for spec in specs:
        try:
            m, eq = run_one(spec, mr, spy_feat)
            print(
                f"{spec.name:48s}  WFmean={m['wf_mean_cagr']*100:6.2f}%  "
                f"Full={m['cagr_full']*100:6.2f}%  Sharpe={m['sharpe']:.2f}  "
                f"MaxDD={m['max_dd']*100:6.2f}%  PosSpl={m['wf_n_pos']}/{m['wf_n_splits']}  "
                f"BeatSPY={m['wf_n_beats_spy']}"
            )
            rows.append({**m, **{f"spec_{k}": v for k, v in asdict(spec).items()}})
        except Exception as e:
            print(f"{spec.name:48s}  FAILED: {e}")
            import traceback; traceback.print_exc()
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{save_prefix}_results.csv", index=False)
    return df


if __name__ == "__main__":
    base_specs = [
        # baselines
        StratSpec("v3_baseline_ml", "ml_3plus6"),
        StratSpec("mom_12_1", "mom_12_1"),
        StratSpec("idio_mom", "idio_mom"),
        StratSpec("mom_per_vol", "mom_per_vol"),
        StratSpec("breakout", "breakout"),
        StratSpec("breakout_strength", "breakout_strength"),
        StratSpec("trend_quality", "trend_quality"),
        StratSpec("concretum_trend", "concretum_trend"),
        StratSpec("alpha_apex", "alpha_apex"),
        StratSpec("alpha_apex_v2", "alpha_apex_v2"),
        StratSpec("dual_momentum", "dual_momentum"),
        StratSpec("low_dd_winner", "low_dd_winner"),
        StratSpec("earnings_drift", "earnings_drift"),
        StratSpec("acceleration", "acceleration"),
        StratSpec("quality_score", "quality_score"),
        StratSpec("multibagger", "multibagger"),
        StratSpec("qmom_ml_50_50", "qmom_ml", weights={"ml": 0.5}),
        StratSpec("qmom_ml_70_30", "qmom_ml", weights={"ml": 0.7}),
        StratSpec("qmom_ml_30_70", "qmom_ml", weights={"ml": 0.3}),
    ]
    benchmark(base_specs, save_prefix="atomic_factors")
