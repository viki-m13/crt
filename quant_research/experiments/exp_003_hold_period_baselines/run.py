"""Phase 2 rung 5: Cross-sectional OLS and hold-period GBM baselines.

Tests the YLOka-compatible architecture (hold_months=1..12) with GBM signals.
Reproduces the YLOka v3 result (~40% CAGR) and shows the hold-period sensitivity.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
from quant_research.backtest.hold_engine import simulate, load_data
from quant_research.backtest.metrics import block_bootstrap_sharpe, deflated_sharpe_ratio

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data (SPY features may take ~30s)...")
panel, mr, mr_idx, spy_feat = load_data()
print(f"Ready: panel={panel.shape}, mr={mr.shape}")

# Helper
def run(name, score_col, K=3, hold_months=6, weighting="ew", use_crash=True):
    r = simulate(
        score_col=score_col, K=K, hold_months=hold_months,
        weighting=weighting, use_crash_gate=use_crash,
        panel=panel, mr=mr, mr_idx=mr_idx, spy_feat=spy_feat,
    )
    print(f"  {name:50s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  "
          f"MaxDD={r['max_dd']:6.1%}  Cash={r['cash_months']}m")
    return r

results = []

print("\n=== Reproducing YLOka v3 baseline (GBM pred, K=3, hold=6) ===")
r_v3 = run("v3_pred_K3_h6", "pred", K=3, hold_months=6)
results.append({"name": "v3_pred_K3_h6", **{k: v for k, v in r_v3.items() if k not in ("monthly_rets","detail")}})

print("\n=== Hold period sweep (GBM pred, K=3) ===")
for h in [1, 2, 3, 4, 6, 9, 12]:
    r = run(f"pred_K3_h{h}", "pred", K=3, hold_months=h)
    results.append({"name": f"pred_K3_h{h}", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== K sweep (GBM pred, hold=6) ===")
for K in [1, 2, 3, 5, 7, 10, 15, 20]:
    r = run(f"pred_K{K}_h6", "pred", K=K, hold_months=6)
    results.append({"name": f"pred_K{K}_h6", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== Score variants (K=3, hold=6) ===")
for score in ["pred_1m", "pred_3m", "pred_6m", "pred", "score", "pred_12m"]:
    if score in panel.columns:
        r = run(f"{score}_K3_h6", score, K=3, hold_months=6)
        results.append({"name": f"{score}_K3_h6", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== Composite GBM score (K=3, hold=6) ===")
# Build composite from GBM heads only
gbm_feats = ["pred_1m", "pred_3m", "pred_6m", "pred_12m"]
gbm_feats = [f for f in gbm_feats if f in panel.columns]
for f in gbm_feats:
    panel[f"rk_{f}"] = panel.groupby("asof")[f].rank(pct=True)
rk_cols = [f"rk_{f}" for f in gbm_feats]
panel["gbm_composite"] = panel[rk_cols].mean(axis=1)

r = run("gbm_composite_K3_h6", "gbm_composite", K=3, hold_months=6)
results.append({"name": "gbm_composite_K3_h6", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

r = run("gbm_composite_K5_h6", "gbm_composite", K=5, hold_months=6)
results.append({"name": "gbm_composite_K5_h6", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

# With inv-vol weighting
r = run("gbm_composite_K3_h6_ivw", "gbm_composite", K=3, hold_months=6, weighting="invvol")
results.append({"name": "gbm_composite_K3_h6_ivw", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== No crash gate (to isolate regime effect) ===")
r = run("pred_K3_h6_nogate", "pred", K=3, hold_months=6, use_crash=False)
results.append({"name": "pred_K3_h6_nogate", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

# Save results
df = pd.DataFrame(results)
df.to_csv(OUT / "results.csv", index=False)

print("\n=== Summary by CAGR ===")
df_s = df.sort_values("cagr", ascending=False)
print(df_s[["name","cagr","sharpe","max_dd","cash_months","n_months"]].to_string())

# Deep dive on best
best = df_s.iloc[0]
print(f"\n=== Deep dive on best: {best['name']} ===")
best_score = best["name"].split("_")[0]
best_K = int([x for x in best["name"].split("_") if x.startswith("K")][0][1:])
best_h = int([x for x in best["name"].split("_") if x.startswith("h")][0][1:])

best_r = simulate(
    score_col=best_score if best_score in panel.columns else "gbm_composite",
    K=best_K, hold_months=best_h,
    panel=panel, mr=mr, mr_idx=mr_idx, spy_feat=spy_feat,
)
mr_best = best_r["monthly_rets"]

# Sub-period analysis
n = len(mr_best)
chunk = n // 3
for i in range(3):
    ch = mr_best.iloc[i*chunk:(i+1)*chunk]
    sh = ch.mean()/ch.std()*np.sqrt(12) if ch.std() > 0 else 0
    cagr_ch = (1+ch).prod()**(12/len(ch))-1
    print(f"  Period {i+1}: CAGR={cagr_ch:.1%}  Sharpe={sh:.2f}")

# Block bootstrap
bb = block_bootstrap_sharpe(mr_best, block_len=6, n_iter=500)
print(f"Block bootstrap Sharpe: p5={bb['p5']:.2f}  p50={bb['p50']:.2f}  p95={bb['p95']:.2f}")

# DSR
n_trials_total = 88 + 40 + 40 + len(results)  # YLOka + exp001 + exp002 + exp003
dsr = deflated_sharpe_ratio(
    sharpe_obs=best_r["sharpe"], n_obs=best_r["n_months"],
    n_trials=n_trials_total,
    skew=float(mr_best.skew()), kurt=float(mr_best.kurtosis()+3),
)
print(f"DSR (n_trials={n_trials_total}): {dsr:.4f}")

summary = {
    "best_config": best["name"],
    "cagr": float(best["cagr"]),
    "sharpe": float(best["sharpe"]),
    "max_dd": float(best["max_dd"]),
    "bb_p5": bb["p5"], "bb_p50": bb["p50"], "bb_p95": bb["p95"],
    "dsr": dsr,
    "n_trials_total": n_trials_total,
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone. {len(results)} configs. Results in {OUT}")
