"""V6 final big sweep — pulls from sweep #1 and #2 lessons + new knobs.

Key lessons learned:
  - tight gate dominates over alternatives
  - invvol weighting marginally improves Sharpe and DD
  - cash yield 3% is mildly positive
  - SPY DD continuous scaling reduces DD but at large CAGR cost; needs HIGH floor
  - h=6 dominates other holds
  - K=3 dominates K=5 for CAGR, K=5 better for Sharpe but loses too much CAGR
  - Crash fallback to SPY/TLT didn't help

This sweep probes a tighter, more sensible parameter region:
  - tight gate, but try cy 0/3%
  - weighting in {ew, invvol, conv}
  - vol_penalty in {0.0, 0.05}
  - spy_dd_scale x floor combinations with floor>=0.85
  - K in {3, 4, 5}
  - drawdown_de_risk in {0.0, 0.20, 0.25}
  - trailing_stop in {0.0, 0.30, 0.35} with reset_on_reentry
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
    panel = load_score_panel("ml_3plus6", "sp500_pit", attach_pullback=True)
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    rows = []

    # Always include baseline
    base = V6Config(name="v3_baseline")
    eq = simulate(base, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    m = evaluate(eq, spy_aln, base.name)
    m["cfg"] = json.dumps(base.__dict__, default=str)
    rows.append(m)

    weightings = ["ew", "invvol"]
    cy_grid = [0.0, 0.03]
    sds_grid = [0.0, 0.15, 0.20, 0.30]
    floor_grid = [0.85, 0.90, 0.95]
    monthly_exps = [False, True]
    pullback_filters = [0.0, 0.50, 0.40]
    vol_penalties = [0.0, 0.05]
    ks = [(3, 3, 3), (4, 3, 3), (5, 3, 3)]
    drawdown_derisks = [0.0, 0.20]
    trailing_stops = [0.0, 0.30]
    crash_persists = [1, 2]

    n = 0
    t0 = time.time()
    incr_path = OUT / "v6_sweep3_results.csv"
    rows_buf = []
    for (w, cy, sds, fl, me, pb, vp, kk, ddr, ts, cp) in itertools.product(
        weightings, cy_grid, sds_grid, floor_grid, monthly_exps,
        pullback_filters, vol_penalties, ks, drawdown_derisks,
        trailing_stops, crash_persists
    ):
        if me and sds == 0.0:
            continue  # me only meaningful with sds>0
        kN, kR, kB = kk
        cfg = V6Config(
            name=(
                f"w={w}|cy{cy}|sds{sds}|fl{fl}|me{int(me)}|pb{pb}|vp{vp}|"
                f"k{kN}|ddr{ddr}|ts{ts}|cp{cp}"
            ),
            scorer="ml_3plus6", universe="sp500_pit",
            regime_gate="tight",
            k_normal=kN, k_recovery=kR, k_bull=kB,
            weighting=w, hold_months=6, cost_bps=10.0,
            cash_yield_yr=cy, spy_dd_scale=sds, spy_dd_floor=fl,
            monthly_exposure=me, pullback_filter=pb, vol_penalty=vp,
            drawdown_de_risk=ddr, trailing_stop=ts,
            crash_persist=cp, ts_reset_on_reentry=True,
        )
        try:
            eq = simulate(cfg, panel, monthly_returns, spy_feats)
            spy_aln = build_spy_aligned(eq, monthly_returns)
            m = evaluate(eq, spy_aln, cfg.name)
            m["cfg"] = json.dumps(cfg.__dict__, default=str)
            rows.append(m)
            rows_buf.append(m)
        except Exception as e:
            print(f"  ! {cfg.name}: {e}")
        n += 1
        if n % 200 == 0:
            print(f"  {n} variants in {time.time()-t0:.1f}s")
            # incremental save so partial runs aren't lost
            pd.DataFrame(rows).to_csv(incr_path, index=False)
    print(f"[done] {len(rows)} in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v6_sweep3_results.csv", index=False)

    v3 = df[df["name"] == "v3_baseline"].iloc[0]
    print(f"\nv3 baseline: cagr={v3.cagr_full:.3f} sh={v3.sharpe:.3f} dd={v3.max_dd:.3f} wf={v3.wf_mean_cagr:.3f}")

    df["delta_cagr"] = df["cagr_full"] - v3["cagr_full"]
    df["delta_wfcagr"] = df["wf_mean_cagr"] - v3["wf_mean_cagr"]
    df["delta_sharpe"] = df["sharpe"] - v3["sharpe"]
    df["delta_dd"] = df["max_dd"] - v3["max_dd"]

    # Pareto by criteria
    print("\n=== Pareto: WF mean >= v3 - 1pp AND sharpe > v3 AND mdd > v3 ===")
    cand = df[
        (df["wf_n_splits"] == 10)
        & (df["wf_n_pos"] >= 10)
        & (df["wf_n_beats_spy"] >= 9)
        & (df["wf_mean_cagr"] >= v3["wf_mean_cagr"] - 0.01)
        & (df["sharpe"] > v3["sharpe"])
        & (df["max_dd"] > v3["max_dd"])
    ].copy()
    print(f"  candidates: {len(cand)}")
    if len(cand):
        cand["score"] = (
            cand["wf_mean_cagr"] * 100
            + cand["sharpe"] * 12
            + (cand["max_dd"] - v3["max_dd"]) * 100
            + cand["wf_min_cagr"] * 30
        )
        cols = ["name", "cagr_full", "sharpe", "max_dd",
                "wf_mean_cagr", "wf_min_cagr", "wf_min_sharpe", "wf_n_beats_spy",
                "delta_cagr", "delta_wfcagr", "delta_sharpe", "delta_dd", "score"]
        print(cand.sort_values("score", ascending=False).head(30)[cols].round(4).to_string(index=False))
        cand.to_csv(OUT / "v6_sweep3_pareto.csv", index=False)

    print("\n=== Top by CAGR with sharpe>v3 and mdd>v3 ===")
    strict = df[
        (df["wf_n_splits"] == 10)
        & (df["wf_n_pos"] >= 10)
        & (df["wf_n_beats_spy"] >= 9)
        & (df["sharpe"] > v3["sharpe"])
        & (df["max_dd"] > v3["max_dd"])
    ].copy()
    print(f"  candidates: {len(strict)}")
    if len(strict):
        cols = ["name", "cagr_full", "sharpe", "max_dd",
                "wf_mean_cagr", "wf_min_cagr", "wf_min_sharpe", "wf_n_beats_spy"]
        print(strict.sort_values("wf_mean_cagr", ascending=False).head(20)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
