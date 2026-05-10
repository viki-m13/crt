"""Reproduce the v3 baseline metrics with the v6 engine and save them to disk
so all subsequent experiments compare against an identical, in-process number.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("[load] score panel + monthly returns + SPY features")
    t0 = time.time()
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    print(f"  panel rows={len(panel)} asofs={panel['asof'].nunique()} ({time.time()-t0:.1f}s)")

    print("[run] v3 baseline")
    cfg = V6Config(
        name="v3_baseline_ml3plus6_k3_tight_h6",
        scorer="ml_3plus6", universe="sp500_pit", regime_gate="tight",
        k_normal=3, k_recovery=3, k_bull=3,
        weighting="ew", hold_months=6, cost_bps=10.0,
    )
    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    metrics = evaluate(eq, spy_aln, cfg.name)

    print("[result] v3 baseline metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    eq.to_csv(OUT / "v3_baseline_equity.csv", index=False)
    (OUT / "v3_baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {OUT}/v3_baseline_*.csv,json")


if __name__ == "__main__":
    main()
