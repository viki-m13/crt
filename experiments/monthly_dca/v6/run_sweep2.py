"""V6 focused sweep #2 — best Pareto candidates.

After sweep #1 the surviving knobs were:
  - tight gate (best of three)
  - invvol weighting (better Sharpe and WF min)
  - cash yield 3% (mild positive, no downside)
  - drawdown de-risk modest

Sweep #2 adds the new knobs introduced in lib_engine.py:
  - SPY DD-from-52w-high continuous scaling (0, 0.10, 0.15, 0.20, 0.30)
  - Sticky cash re-entry (0, 1, 2, 3 normal months required)
  - Trailing-stop with sticky reentry combined
  - Crash persistence 1 vs 2
  - Better-and-finer trailing-stop levels
  - K=3 vs K=5

Goal: find a Pareto-better point (>= v3 CAGR, > v3 Sharpe, > v3 MaxDD).
"""
from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_engine import (
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    rows = []

    # Baseline
    base = V6Config(name="v3_baseline", scorer="ml_3plus6", universe="sp500_pit",
                    regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                    weighting="ew", hold_months=6, cost_bps=10.0)
    eq = simulate(base, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    m = evaluate(eq, spy_aln, base.name)
    m["cfg"] = json.dumps(base.__dict__, default=str)
    rows.append(m)

    # Knob set
    weightings = ["ew", "invvol"]
    cy_grid = [0.0, 0.03]
    spy_dd_scales = [0.0, 0.10, 0.15, 0.20, 0.30]
    cash_stickys = [0, 1, 2, 3]
    trailing_stops = [0.0, 0.20, 0.25, 0.30, 0.35]
    crash_persists = [1, 2]
    ks = [(3, 3, 3), (5, 3, 3), (5, 5, 5)]
    holds = [6]

    n = 0
    t0 = time.time()
    for (w, cy, sds, cs, ts, cp, kk, h) in itertools.product(
        weightings, cy_grid, spy_dd_scales, cash_stickys, trailing_stops,
        crash_persists, ks, holds
    ):
        kN, kR, kB = kk
        cfg = V6Config(
            name=(
                f"w={w}|cy{cy}|sds{sds}|cs{cs}|ts{ts}|cp{cp}|k{kN}_{kR}_{kB}|h{h}"
            ),
            scorer="ml_3plus6", universe="sp500_pit",
            regime_gate="tight",
            k_normal=kN, k_recovery=kR, k_bull=kB,
            weighting=w, hold_months=h, cost_bps=10.0,
            cash_yield_yr=cy,
            spy_dd_scale=sds,
            cash_sticky=cs,
            trailing_stop=ts,
            crash_persist=cp,
        )
        try:
            eq = simulate(cfg, panel, monthly_returns, spy_feats)
            spy_aln = build_spy_aligned(eq, monthly_returns)
            m = evaluate(eq, spy_aln, cfg.name)
            m["cfg"] = json.dumps(cfg.__dict__, default=str)
            rows.append(m)
        except Exception as e:
            print(f"  ! {cfg.name}: {e}")
        n += 1
        if n % 100 == 0:
            print(f"  {n} variants in {time.time()-t0:.1f}s")
    print(f"[done] {len(rows)} in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v6_sweep2_results.csv", index=False)

    # Pareto: better than v3 on >=2 of {CAGR, Sharpe, MaxDD}
    v3 = df[df["name"] == "v3_baseline"].iloc[0]
    df["n_better"] = (
        (df["cagr_full"] >= v3["cagr_full"]).astype(int)
        + (df["sharpe"] > v3["sharpe"]).astype(int)
        + (df["max_dd"] > v3["max_dd"]).astype(int)
    )
    df["all3_better"] = (
        (df["cagr_full"] >= v3["cagr_full"]) & (df["sharpe"] > v3["sharpe"]) & (df["max_dd"] > v3["max_dd"])
    )

    # Strict candidates
    cand = df[(df["wf_n_splits"] == 10) & (df["wf_n_pos"] >= 9) & (df["wf_n_beats_spy"] >= 9)
             & (df["n_better"] >= 2)
             & (df["wf_mean_cagr"] >= 0.40)].copy()
    cand["score"] = (
        cand["cagr_full"] * 100
        + cand["sharpe"] * 10
        + (cand["max_dd"] - v3["max_dd"]) * 100      # absolute pp delta on MaxDD
        + cand["wf_mean_edge_pp"] * 0.3
        + cand["wf_min_cagr"] * 30
    )
    print(f"\n[pareto candidates >=2 of 3 vs v3] {len(cand)}")
    cols = ["name", "cagr_full", "sharpe", "max_dd",
            "wf_mean_cagr", "wf_min_cagr", "wf_min_sharpe", "wf_mean_dd",
            "wf_n_pos", "wf_n_beats_spy", "n_better", "score"]
    print(cand.sort_values("score", ascending=False).head(30)[cols].round(3).to_string(index=False))
    cand.to_csv(OUT / "v6_sweep2_pareto.csv", index=False)

    # All-3 better
    all3 = df[df["all3_better"] & (df["wf_n_splits"] == 10) & (df["wf_n_pos"] >= 9) & (df["wf_n_beats_spy"] >= 9)]
    print(f"\n[all-3 better] {len(all3)} candidates")
    if len(all3) > 0:
        all3["score"] = (
            all3["cagr_full"] * 100
            + all3["sharpe"] * 10
            + (all3["max_dd"] - v3["max_dd"]) * 100
            + all3["wf_mean_edge_pp"] * 0.3
        )
        print(all3.sort_values("score", ascending=False).head(20)[cols + ["score"]].round(3).to_string(index=False))
        all3.to_csv(OUT / "v6_sweep2_all3_better.csv", index=False)


if __name__ == "__main__":
    main()
