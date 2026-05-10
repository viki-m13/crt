"""Tier 1 sweep — concentration + horizon + scorer + regime, monthly cadence.

Reuses the existing v6 engine (parity with deployed v3) and the existing
ml_preds_v2.parquet (walk-forward HistGBM, 7-month embargo, retrained
annually). No model retraining needed at this stage.

Output:
  experiments/monthly_dca/v8/results/tier1_sweep.csv
  experiments/monthly_dca/v8/results/tier1_sweep_top.csv
  experiments/monthly_dca/v8/results/tier1_sweep_pareto.csv
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (  # noqa: E402
    V2, V6Config, build_spy_aligned, evaluate,
    load_score_panel, load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


SCORERS = ["ml_3plus6", "ml_h3", "ml_h6", "ml_3plus6plus1"]
KS = [1, 2, 3]
HOLDS = [1, 2, 3, 6]
REGIMES = ["tight", "strict_dd", "safer", "combo"]
WEIGHTINGS = ["ew", "invvol"]


def main():
    print("[load] panels and SPY features")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    panels = {}
    for s in SCORERS:
        panels[s] = load_score_panel(s, "sp500_pit")
        print(f"  scorer={s} rows={len(panels[s])}")

    print("[run] sweep")
    t0 = time.time()
    rows = []
    cfgs = list(itertools.product(SCORERS, KS, HOLDS, REGIMES, WEIGHTINGS))
    for i, (sc, k, h, rg, w) in enumerate(cfgs):
        cfg = V6Config(
            name=f"{sc}|k{k}|h{h}|{rg}|{w}",
            scorer=sc, universe="sp500_pit", regime_gate=rg,
            k_normal=k, k_recovery=k, k_bull=k,
            weighting=w, hold_months=h, cost_bps=10.0,
        )
        eq = simulate(cfg, panels[sc], monthly_returns, spy_feats)
        spy_aln = build_spy_aligned(eq, monthly_returns)
        m = evaluate(eq, spy_aln, cfg.name)
        m.update({
            "scorer": sc, "k": k, "hold": h, "regime": rg, "weighting": w,
        })
        rows.append(m)
        if (i + 1) % 25 == 0 or (i + 1) == len(cfgs):
            print(f"  {i+1}/{len(cfgs)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tier1_sweep.csv", index=False)

    # Floors
    floor_min_cagr = 0.0
    floor_sharpe = 1.0
    floor_dd = -0.50
    floor_beats = 8

    df_pass = df[
        (df["wf_min_cagr"] >= floor_min_cagr)
        & (df["wf_mean_sharpe"] >= floor_sharpe)
        & (df["max_dd"] >= floor_dd)
        & (df["wf_n_beats_spy"] >= floor_beats)
    ].copy()

    df_top = df.sort_values("wf_mean_cagr", ascending=False).head(40)
    df_pass_top = df_pass.sort_values("wf_mean_cagr", ascending=False).head(40)

    df_top.to_csv(OUT / "tier1_sweep_top_unconstrained.csv", index=False)
    df_pass_top.to_csv(OUT / "tier1_sweep_top_passing_floors.csv", index=False)

    summary = {
        "n_total": int(len(df)),
        "n_passing_floors": int(len(df_pass)),
        "best_unconstrained": df_top.iloc[0].to_dict() if len(df_top) else None,
        "best_passing": df_pass_top.iloc[0].to_dict() if len(df_pass_top) else None,
        "baseline_v3_wf_mean": 0.42800538003320804,
    }
    (OUT / "tier1_sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"[done] {len(df)} configs in {time.time()-t0:.0f}s")
    print(f"       {len(df_pass)} pass floors")
    print()
    print("Top 10 unconstrained (by WF mean CAGR):")
    cols = ["scorer", "k", "hold", "regime", "weighting", "wf_mean_cagr",
            "wf_min_cagr", "wf_mean_sharpe", "max_dd", "wf_n_beats_spy", "cagr_full"]
    print(df_top.head(10)[cols].to_string(index=False))
    print()
    print("Top 10 passing floors:")
    if len(df_pass_top):
        print(df_pass_top.head(10)[cols].to_string(index=False))
    else:
        print("  NONE PASS — relax floors or accept higher MaxDD")


if __name__ == "__main__":
    main()
