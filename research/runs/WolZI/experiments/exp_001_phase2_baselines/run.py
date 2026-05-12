"""Phase 2 baseline ladder — Section 7 of CLAUDE.md.

Rungs (all walk-forward on research window 2003-09 → 2023-12):
  1. Equal-weight top-N by 12-1 momentum (Jegadeesh-Titman)
  2. +low-volatility filter (drop top-30% vol names before ranking)
  3. +quality screen (keep above-median trend_health_5y)
  4. +regime gate (SPY 200-day MA + momentum)
  5. +inverse-vol weighting within selected names
  6. Using v3 GBM score (pred col in panel) as ranker — best prior result

Also sweeps K ∈ {3, 5, 10, 20, 30} for each rung to show sensitivity.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
from quant_research.backtest.engine import (
    BacktestConfig, run_backtest, summarize, load_panel, load_monthly_returns, load_spy_monthly
)
from quant_research.backtest.metrics import (
    annualized_sharpe, annualized_cagr, max_drawdown,
    deflated_sharpe_ratio, block_bootstrap_sharpe
)

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data...")
panel = load_panel()
mr = load_monthly_returns()
spy = load_spy_monthly()
print(f"Panel: {panel.shape}, MR: {mr.shape}, SPY: {spy.shape}")

# -----------------------------------------------------------------------
# Define the ladder configs
# -----------------------------------------------------------------------
K_SWEEP = [3, 5, 10, 20, 30]
CONFIGS = []

for K in K_SWEEP:
    # Rung 1: Pure 12-1 momentum, EW, no filters, no regime gate
    CONFIGS.append(BacktestConfig(
        name=f"R1_mom12_1_K{K}",
        K=K, weighting="ew", score_col="mom_12_1",
        use_regime=False, vol_filter=False, quality_filter=False,
    ))

    # Rung 2: + low-vol filter
    CONFIGS.append(BacktestConfig(
        name=f"R2_mom_lowvol_K{K}",
        K=K, weighting="ew", score_col="mom_12_1",
        use_regime=False, vol_filter=True, quality_filter=False,
    ))

    # Rung 3: + quality screen (trend health)
    CONFIGS.append(BacktestConfig(
        name=f"R3_mom_qual_K{K}",
        K=K, weighting="ew", score_col="mom_12_1",
        use_regime=False, vol_filter=True, quality_filter=True,
    ))

    # Rung 4: + regime gate
    CONFIGS.append(BacktestConfig(
        name=f"R4_mom_qual_regime_K{K}",
        K=K, weighting="ew", score_col="mom_12_1",
        use_regime=True, vol_filter=True, quality_filter=True,
    ))

    # Rung 5: + inv-vol weighting
    CONFIGS.append(BacktestConfig(
        name=f"R5_mom_qual_regime_ivw_K{K}",
        K=K, weighting="invvol", score_col="mom_12_1",
        use_regime=True, vol_filter=True, quality_filter=True,
    ))

    # Rung 6: v3 GBM ranker (pred col = blend of pred_1m + pred_3m + pred_6m)
    CONFIGS.append(BacktestConfig(
        name=f"R6_gbm_pred_K{K}",
        K=K, weighting="ew", score_col="pred",
        use_regime=True, vol_filter=False, quality_filter=False,
    ))

    # Rung 7: v3 GBM ranker + inv-vol weights
    CONFIGS.append(BacktestConfig(
        name=f"R7_gbm_pred_ivw_K{K}",
        K=K, weighting="invvol", score_col="pred",
        use_regime=True, vol_filter=False, quality_filter=False,
    ))

    # Rung 8: v3 GBM ranker + lowvol filter + inv-vol weights
    CONFIGS.append(BacktestConfig(
        name=f"R8_gbm_lowvol_ivw_K{K}",
        K=K, weighting="invvol", score_col="pred",
        use_regime=True, vol_filter=True, quality_filter=False,
    ))

# -----------------------------------------------------------------------
# Run all configs
# -----------------------------------------------------------------------
results = []
for cfg in CONFIGS:
    res = run_backtest(cfg, panel=panel, mr=mr, spy=spy)
    print(summarize(res))
    row = {k: v for k, v in res.items() if k != "monthly_returns"}
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(OUT / "results.csv", index=False)
print(f"\nSaved {len(df)} rows to {OUT / 'results.csv'}")

# -----------------------------------------------------------------------
# Summary table: best result per rung-type across K
# -----------------------------------------------------------------------
print("\n=== Best per rung (by CAGR) ===")
for rung in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
    sub = df[df["name"].str.startswith(rung)].sort_values("cagr", ascending=False)
    if len(sub):
        best = sub.iloc[0]
        print(f"  {best['name']:40s} CAGR={best['cagr']:.1%}  Sharpe={best['sharpe']:.2f}  MaxDD={best['max_dd']:.1%}")

print("\n=== K sensitivity for best rung (GBM pred, R6) ===")
r6 = df[df["name"].str.startswith("R6")].sort_values("K" if "K" in df.columns else "name")
print(r6[["name", "cagr", "sharpe", "max_dd", "cash_months"]].to_string())

# -----------------------------------------------------------------------
# Deep dive on best overall config
# -----------------------------------------------------------------------
best_row = df.sort_values("cagr", ascending=False).iloc[0]
print(f"\n=== Best overall: {best_row['name']} ===")
best_cfg = next(c for c in CONFIGS if c.name == best_row["name"])
best_res = run_backtest(best_cfg, panel=panel, mr=mr, spy=spy)
mr_best = best_res["monthly_returns"]

# Block bootstrap
bb = block_bootstrap_sharpe(mr_best, block_len=6, n_iter=1000)
print(f"Block bootstrap Sharpe p5/p50/p95: {bb['p5']:.2f}/{bb['p50']:.2f}/{bb['p95']:.2f}")

# DSR (using n_trials = len(CONFIGS) as first-run count)
n_trials = len(CONFIGS)
dsr = deflated_sharpe_ratio(
    sharpe_obs=best_res["sharpe"],
    n_obs=best_res["n_months"],
    n_trials=n_trials,
    skew=float(mr_best.skew()),
    kurt=float(mr_best.kurtosis() + 3),
)
print(f"DSR (n_trials={n_trials}): {dsr:.4f}")
print(f"Full result: CAGR={best_res['cagr']:.1%}  Sharpe={best_res['sharpe']:.2f}  MaxDD={best_res['max_dd']:.1%}")

# Save best monthly returns
mr_best.to_csv(OUT / "best_monthly_returns.csv")

summary = {
    "best_config": best_row["name"],
    "cagr": float(best_row["cagr"]),
    "sharpe": float(best_row["sharpe"]),
    "max_dd": float(best_row["max_dd"]),
    "n_months": int(best_row["n_months"]),
    "cash_months": int(best_row["cash_months"]),
    "bb_p5": bb["p5"],
    "bb_p50": bb["p50"],
    "bb_p95": bb["p95"],
    "dsr": dsr,
    "n_trials": n_trials,
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone. Results saved to {OUT}")
