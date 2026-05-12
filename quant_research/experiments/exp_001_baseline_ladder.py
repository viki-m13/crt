"""
Experiment 001: Phase 2 Baseline Ladder
Tests momentum -> quality -> regime -> ML composite progression.
Period: 2003-2021 (OOS), last 2 years (2022-2023) reserved.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, make_regime_fn, walk_forward
from features.signals import (
    momentum_12_1, momentum_lowvol, momentum_quality_lowvol,
    composite_v1, smooth_compounder, ml_score_v3, ml_plus_sharpe,
    ml_plus_lowvol, ml_plus_smooth,
)

RESULTS = []
TOTAL_HYP = 0


def run(name, score_fn, top_k, weighting="ew", regime=None,
        start="2003-09-30", end="2021-12-31", cost_bps=5.0, n_hyp=1):
    global TOTAL_HYP
    TOTAL_HYP += n_hyp
    t0 = time.time()
    regime_fn = make_regime_fn(regime) if regime else None
    df, metrics = run_backtest(
        score_fn=score_fn,
        start=start, end=end,
        top_k=top_k, weighting=weighting,
        cost_bps=cost_bps, regime_fn=regime_fn,
    )
    elapsed = time.time() - t0
    if not metrics:
        print(f"  {name}: NO RESULTS")
        return

    cash_m = int((df["n_picks"] == 0).sum()) if "n_picks" in df.columns else 0
    result = {
        "name": name,
        "top_k": top_k,
        "weighting": weighting,
        "regime": regime or "none",
        "cagr": round(float(metrics["cagr"]), 4),
        "sharpe": round(float(metrics["sharpe"]), 3),
        "max_dd": round(float(metrics["max_dd"]), 4),
        "win_rate": round(float(metrics["win_rate"]), 3),
        "ann_vol": round(float(metrics["ann_vol"]), 4),
        "n_months": int(metrics["n_months"]),
        "cash_months": cash_m,
        "mean_m": round(float(metrics["mean_m"]), 5),
        "std_m": round(float(metrics["std_m"]), 5),
        "elapsed_s": round(elapsed, 2),
    }
    RESULTS.append(result)
    gc = "✓" if result["cagr"] >= 0.50 else "✗"
    gs = "✓" if result["sharpe"] >= 2.0 else "✗"
    print(f"  {name:55s} CAGR={result['cagr']:.1%}{gc} Sharpe={result['sharpe']:.2f}{gs} "
          f"MaxDD={result['max_dd']:.1%} Cash={cash_m}m {elapsed:.1f}s")


print("=" * 90)
print("EXPERIMENT 001: BASELINE LADDER")
print("Period: 2003-09-30 to 2021-12-31 (OOS — 2022-2023 reserved for lockbox)")
print("=" * 90)

# ----- RUNG 1: Pure momentum -----
print("\n--- RUNG 1: Pure 12-1 Momentum ---")
for k in [3, 5, 10, 20, 30]:
    run(f"mom_12_1 K={k:2d} EW", momentum_12_1, k)

# ----- RUNG 2: Momentum + low-vol -----
print("\n--- RUNG 2: Momentum + Low-Vol Filter ---")
for k in [5, 10, 20]:
    run(f"mom_lowvol K={k:2d} EW", momentum_lowvol, k)
for k in [5, 10, 20]:
    run(f"mom_lowvol K={k:2d} inv_vol", momentum_lowvol, k, weighting="inv_vol")

# ----- RUNG 3: Quality screen -----
print("\n--- RUNG 3: Quality + Low-Vol ---")
for k in [5, 10, 20]:
    run(f"mom_qual K={k:2d} EW", momentum_quality_lowvol, k)
for k in [5, 10, 20]:
    run(f"mom_qual K={k:2d} inv_vol", momentum_quality_lowvol, k, weighting="inv_vol")

# ----- RUNG 4: Regime gate -----
print("\n--- RUNG 4: With Regime Gate ---")
for k in [5, 10, 20]:
    run(f"mom_qual K={k:2d} + regime_200ma", momentum_quality_lowvol, k, regime="200ma")
for k in [5, 10, 20]:
    run(f"smooth K={k:2d} + regime_200ma", smooth_compounder, k, regime="200ma")
for k in [10, 20]:
    run(f"smooth K={k:2d} inv_vol + regime", smooth_compounder, k, "inv_vol", "conservative")

# ----- RUNG 5: Cross-sectional composite -----
print("\n--- RUNG 5: Composite v1 ---")
for k in [5, 10, 20]:
    run(f"composite_v1 K={k:2d} EW", composite_v1, k)
for k in [5, 10, 20]:
    run(f"composite_v1 K={k:2d} inv_vol", composite_v1, k, weighting="inv_vol")
for k in [10, 20]:
    run(f"composite_v1 K={k:2d} + regime", composite_v1, k, regime="200ma")

# ----- ML-based scores -----
print("\n--- ML Predictions (v3 GBM) ---")
for k in [3, 5, 10]:
    run(f"ml_v3 K={k:2d} EW", ml_score_v3, k)
for k in [3, 5, 10]:
    run(f"ml_v3 K={k:2d} + regime_200ma", ml_score_v3, k, regime="200ma")
for k in [5, 10]:
    run(f"ml_v3 K={k:2d} inv_vol + regime", ml_score_v3, k, "inv_vol", "200ma")

# ML + filters
print("\n--- ML + Quality/Vol Filters ---")
for k in [3, 5, 10]:
    run(f"ml_sharpe K={k:2d} EW", ml_plus_sharpe, k)
for k in [3, 5, 10]:
    run(f"ml_lowvol K={k:2d} EW", ml_plus_lowvol, k)
for k in [5, 10, 20]:
    run(f"ml_smooth K={k:2d} EW", ml_plus_smooth, k)
for k in [5, 10]:
    run(f"ml_smooth K={k:2d} inv_vol + regime", ml_plus_smooth, k, "inv_vol", "200ma")

# ----- Summary -----
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
pd.set_option("display.max_colwidth", 60)
print(df_res[["name", "cagr", "sharpe", "max_dd", "win_rate", "cash_months", "ann_vol",
              "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_001_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved to {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} -> CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"Total hypotheses this run: {TOTAL_HYP}")
