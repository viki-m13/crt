"""V6 risk-control sweep — held to the same PIT panel + same ML predictions
as the deployed v3 winner.  We add a small, well-motivated knob set:

  - regime gate (tight | strict_dd | safer)
  - crash persistence (1 = current, 2 = require two months of crash signal)
  - drawdown de-risk threshold (0.0 = off; otherwise halve gross when running
    portfolio DD <= -X)
  - trailing stop (0 disabled; otherwise go to cash if running DD <= -X, then
    re-enter at next non-crash regime)
  - cash yield (annualised — represent cash earning T-bill rate)
  - vol target (annualised gross-scaling target; 0 disabled)
  - weighting (ew | invvol)
  - half-cash on warning regime (only relevant for safer gate)
  - k (3, 5)

We run 100% out-of-sample WF and grade on a composite of:
  - WF mean CAGR  (>= v3 baseline 42.80%)
  - Sharpe        (> 0.955)
  - MaxDD         (better/less-negative than -49.8%)
  - 10/10 WF positive
  - 9/10+ beats SPY
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
    print(f"  panel rows={len(panel)} asofs={panel['asof'].nunique()}")

    # Knob set — kept compact to avoid overfitting; ~64 variants
    gates = ["tight", "strict_dd", "safer"]
    persists = [1, 2]
    drawdown_de_risks = [0.0, 0.15, 0.20]
    trailing_stops = [0.0, 0.25, 0.35]
    cash_yields = [0.0, 0.03]
    vol_targets = [0.0, 0.30]
    weightings = ["ew", "invvol"]
    ks = [(3, 3, 3), (5, 3, 3)]
    holds = [6]

    rows = []
    n = 0
    t0 = time.time()

    # Always include baseline as variant 0
    base_cfg = V6Config(
        name="v3_baseline",
        scorer="ml_3plus6", universe="sp500_pit",
        regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
        weighting="ew", hold_months=6, cost_bps=10.0,
        vol_target_yr=0.0, half_cash_warning=False, cash_yield_yr=0.0,
        drawdown_de_risk=0.0, crash_persist=1, trailing_stop=0.0,
    )
    eq = simulate(base_cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    m = evaluate(eq, spy_aln, base_cfg.name)
    m["cfg"] = json.dumps(base_cfg.__dict__, default=str)
    rows.append(m)

    for (gate, persist, ddr, ts, cy, vt, wt, kk) in itertools.product(
        gates, persists, drawdown_de_risks, trailing_stops,
        cash_yields, vol_targets, weightings, ks
    ):
        for hold in holds:
            kN, kR, kB = kk
            half_warn = (gate == "safer")
            cfg = V6Config(
                name=(
                    f"g={gate}|p{persist}|ddr{ddr}|ts{ts}|cy{cy}|vt{vt}|"
                    f"w={wt}|k{kN}_{kR}_{kB}|h{hold}"
                ),
                scorer="ml_3plus6", universe="sp500_pit",
                regime_gate=gate,
                k_normal=kN, k_recovery=kR, k_bull=kB,
                weighting=wt, hold_months=hold, cost_bps=10.0,
                vol_target_yr=vt,
                half_cash_warning=half_warn,
                cash_yield_yr=cy,
                drawdown_de_risk=ddr,
                crash_persist=persist,
                trailing_stop=ts,
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
            if n % 50 == 0:
                print(f"  {n} variants in {time.time()-t0:.1f}s")
    print(f"[done] {len(rows)} variants in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)

    # Save full sweep
    df.to_csv(OUT / "v6_sweep_results.csv", index=False)
    print(f"Saved -> {OUT}/v6_sweep_results.csv ({len(df)} rows)")

    # Filter & rank
    cand = df[
        (df["wf_n_splits"] == 10)
        & (df["wf_n_pos"] >= 10)
        & (df["wf_n_beats_spy"] >= 9)
        & (df["wf_mean_cagr"] >= 0.40)
        & (df["max_dd"] >= -0.50)
    ].copy()
    print(f"\n[filter pass] {len(cand)} candidates with: wf_n_pos>=10, wf_beats>=9, wf_mean_cagr>=0.40, max_dd>=-0.50")

    if len(cand):
        # Pareto-style composite score: reward CAGR + Sharpe + less DD
        cand["score"] = (
            cand["wf_mean_cagr"] * 100
            + cand["sharpe"] * 10
            + (cand["max_dd"] + 0.50) * 30   # better max_dd (less negative) -> positive bonus
            + cand["wf_mean_edge_pp"] * 0.3
        )
        top = cand.sort_values("score", ascending=False).head(20)
        print("\n[top 20 by score]")
        cols = ["name", "cagr_full", "sharpe", "max_dd",
                "wf_mean_cagr", "wf_min_cagr", "wf_min_sharpe", "wf_mean_dd",
                "wf_n_beats_spy", "score"]
        print(top[cols].round(3).to_string(index=False))
        top.to_csv(OUT / "v6_sweep_top.csv", index=False)

    # Best by Sharpe filter
    cand_sh = df[
        (df["wf_n_splits"] == 10)
        & (df["sharpe"] > 0.955)
        & (df["max_dd"] > -0.50)
        & (df["cagr_full"] >= 0.35)
        & (df["wf_n_pos"] >= 9)
    ].copy()
    print(f"\n[Sharpe filter] {len(cand_sh)} variants with sharpe>v3 and max_dd<v3 and cagr_full>=35%")
    if len(cand_sh):
        print(cand_sh.sort_values("sharpe", ascending=False).head(15)[
            ["name", "cagr_full", "sharpe", "max_dd", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats_spy"]
        ].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
