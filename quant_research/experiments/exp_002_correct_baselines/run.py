"""Phase 2 baseline ladder — REVISED with correct feature signals.

IC analysis reveals:
  - d_sma50, rsi_14, crt_3m, breakout_strength_60 have IR > 0.45 vs next-month return
  - pred/score/pred_Xm have NEGATIVE IC (they predict longer horizons, not 1m)

Rungs:
  1. top-K by d_sma50 (best single IR), EW, no regime gate
  2. top-K by rsi_14
  3. top-K by crt_3m
  4. top-K by composite rank (equal-weight rank of top 5 features)
  5. composite + regime gate
  6. composite + regime gate + inv-vol weighting
  7. composite + regime gate + low-vol filter
  8. composite + SPY-above-50SMA regime (tighter)
  9. cross-sectional rank-blended OLS (computed expanding window)
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[3]))
from quant_research.backtest.engine import (
    load_panel, load_monthly_returns, load_spy_monthly,
    compute_costs, equal_weight, inv_vol_weight, next_month_end,
    RESEARCH_END, LOCKBOX_START, COST_BPS_FLOOR, EXCLUDE
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
spy_feat = load_spy_monthly()

# Filter research window
panel = panel[panel["asof"] <= RESEARCH_END].copy()
print(f"Research panel: {panel.shape}, dates: {panel['asof'].min().date()} → {panel['asof'].max().date()}")

# Top 5 predictive features (by IR from IC analysis)
FEATURES = ["d_sma50", "rsi_14", "crt_3m", "breakout_strength_60", "mom_3"]

# Add composite cross-sectional rank score
def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    """At each asof, compute cross-sectional percentile rank of each feature, then average."""
    df = df.copy()
    for f in FEATURES:
        if f in df.columns:
            df[f"rank_{f}"] = df.groupby("asof")[f].rank(pct=True)
    rank_cols = [f"rank_{f}" for f in FEATURES if f"rank_{f}" in df.columns]
    df["composite"] = df[rank_cols].mean(axis=1)
    return df

panel = add_composite(panel)
print("Composite score added.")

# -------------------------------------------------------------------
# Regime gate functions
# -------------------------------------------------------------------
def regime_tight(spy_row: dict) -> str:
    r1m = spy_row.get("ret_1m", 0.0)
    r6m = spy_row.get("ret_6m", 0.0)
    d200 = spy_row.get("d_sma200", 0.0)
    d50 = spy_row.get("d_sma50", 0.0)
    if pd.isna(r1m):
        return "equity"
    if r1m <= -0.08 or (r6m <= -0.10 and d200 < 0):
        return "cash"
    if d200 < -0.05:
        return "cash"
    return "equity"

def regime_strict(spy_row: dict) -> str:
    """Stricter: also gate on SPY below its 50 SMA."""
    r1m = spy_row.get("ret_1m", 0.0)
    r6m = spy_row.get("ret_6m", 0.0)
    d200 = spy_row.get("d_sma200", 0.0)
    d50 = spy_row.get("d_sma50", 0.0)
    if pd.isna(r1m):
        return "equity"
    if r1m <= -0.05 or (r6m <= -0.05 and d200 < 0):
        return "cash"
    if d50 < -0.03:
        return "cash"
    return "equity"

# -------------------------------------------------------------------
# Core backtest runner
# -------------------------------------------------------------------
def run(name: str, score_col: str, K: int, weighting: str = "ew",
        regime_fn=None, vol_filter: bool = False,
        quality_filter: bool = False, cost_bps: float = COST_BPS_FLOOR,
        cash_yield_apr: float = 0.04) -> dict:
    asof_dates = sorted(panel["asof"].unique())
    equity = 1.0
    monthly_rets, dates_out = [], []
    prev_weights: dict = {}
    cash_months = equity_months = 0

    for asof in asof_dates:
        snap = panel[panel["asof"] == asof].copy()
        snap = snap.dropna(subset=[score_col])
        next_date = next_month_end(asof)

        # Regime
        spy_row = spy_feat.loc[asof].to_dict() if asof in spy_feat.index else {}
        regime = "equity"
        if regime_fn is not None:
            regime = regime_fn(spy_row)

        if regime == "cash":
            ret = cash_yield_apr / 12.0
            cost = compute_costs(prev_weights, {}, cost_bps)
            equity *= (1 + ret - cost)
            monthly_rets.append(ret - cost)
            dates_out.append(next_date)
            prev_weights = {}
            cash_months += 1
            continue

        # Filters
        if vol_filter and "vol_12m" in snap.columns:
            threshold = snap["vol_12m"].quantile(0.70)
            snap = snap[snap["vol_12m"] <= threshold]
        if quality_filter and "trend_health_5y" in snap.columns:
            threshold = snap["trend_health_5y"].median()
            snap = snap[snap["trend_health_5y"] >= threshold]

        top_k = snap.sort_values(score_col, ascending=False).head(K)["ticker"].tolist()
        if not top_k:
            monthly_rets.append(cash_yield_apr / 12.0)
            dates_out.append(next_date)
            continue

        if weighting == "invvol" and "vol_12m" in snap.columns:
            vol_map = snap.set_index("ticker")["vol_12m"].to_dict()
            new_weights = inv_vol_weight(top_k, vol_map)
        else:
            new_weights = equal_weight(top_k)

        cost = compute_costs(prev_weights, new_weights, cost_bps)

        if next_date in mr.index:
            port_ret = sum(
                w * mr.loc[next_date, t]
                for t, w in new_weights.items()
                if t in mr.columns and not np.isnan(mr.loc[next_date, t])
            )
        else:
            port_ret = 0.0

        net_ret = port_ret - cost
        equity *= (1 + net_ret)
        monthly_rets.append(net_ret)
        dates_out.append(next_date)
        prev_weights = new_weights
        equity_months += 1

    rets = pd.Series(monthly_rets, index=dates_out)
    n_months = len(rets)
    n_years = n_months / 12.0
    cagr = equity ** (1 / n_years) - 1 if n_years > 0 else np.nan
    std_m = rets.std()
    sharpe = rets.mean() / std_m * np.sqrt(12) if std_m > 0 else np.nan
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "name": name, "K": K, "score": score_col, "weighting": weighting,
        "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
        "n_months": n_months, "cash_months": cash_months,
        "monthly_returns": rets,
    }

# -------------------------------------------------------------------
# Experiment grid
# -------------------------------------------------------------------
K_SWEEP = [3, 5, 10, 20, 30]
all_results = []

print("\nRunning baselines...")
for K in K_SWEEP:
    for score, label in [
        ("d_sma50",            "R1_dsma50"),
        ("rsi_14",             "R2_rsi14"),
        ("crt_3m",             "R3_crt3m"),
        ("composite",          "R4_composite"),
    ]:
        r = run(f"{label}_K{K}", score, K, regime_fn=None)
        print(f"  {r['name']:40s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_dd']:6.1%}")
        all_results.append({k: v for k, v in r.items() if k != "monthly_returns"})

    # With tight regime gate
    r = run(f"R5_composite_regime_K{K}", "composite", K, regime_fn=regime_tight)
    print(f"  {r['name']:40s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_dd']:6.1%}")
    all_results.append({k: v for k, v in r.items() if k != "monthly_returns"})

    # With strict regime gate
    r = run(f"R6_composite_strict_regime_K{K}", "composite", K, regime_fn=regime_strict)
    print(f"  {r['name']:40s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_dd']:6.1%}")
    all_results.append({k: v for k, v in r.items() if k != "monthly_returns"})

    # Composite + regime + inv-vol
    r = run(f"R7_composite_regime_ivw_K{K}", "composite", K, weighting="invvol", regime_fn=regime_tight)
    print(f"  {r['name']:40s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_dd']:6.1%}")
    all_results.append({k: v for k, v in r.items() if k != "monthly_returns"})

    # Composite + regime + low-vol filter
    r = run(f"R8_composite_regime_lowvol_K{K}", "composite", K, regime_fn=regime_tight, vol_filter=True)
    print(f"  {r['name']:40s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_dd']:6.1%}")
    all_results.append({k: v for k, v in r.items() if k != "monthly_returns"})

df = pd.DataFrame(all_results)
df.to_csv(OUT / "results.csv", index=False)

# -------------------------------------------------------------------
# Best result analysis
# -------------------------------------------------------------------
print("\n=== Best per rung (by Sharpe) ===")
for rung in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
    sub = df[df["name"].str.startswith(rung)].sort_values("sharpe", ascending=False)
    if len(sub):
        b = sub.iloc[0]
        print(f"  {b['name']:42s} CAGR={b['cagr']:.1%}  Sharpe={b['sharpe']:.2f}  MaxDD={b['max_dd']:.1%}  Cash={b['cash_months']}m")

# Re-run best config for detailed stats
best_row = df.sort_values("cagr", ascending=False).iloc[0]
print(f"\n=== Best by CAGR: {best_row['name']} ===")

# Run with returns captured
best_name = best_row["name"]
if "R5" in best_name or "R7" in best_name or "R8" in best_name:
    rfn = regime_tight
elif "R6" in best_name:
    rfn = regime_strict
else:
    rfn = None

K_best = int(best_row["K"])
score_best = str(best_row["score"])
weighting_best = str(best_row["weighting"])

best_res = run(
    best_name + "_full",
    score_best, K_best, weighting=weighting_best,
    regime_fn=rfn,
    vol_filter=("lowvol" in best_name)
)
mr_best = best_res["monthly_returns"]
bb = block_bootstrap_sharpe(mr_best, block_len=6, n_iter=1000)
dsr = deflated_sharpe_ratio(
    sharpe_obs=best_res["sharpe"], n_obs=best_res["n_months"],
    n_trials=len(all_results),
    skew=float(mr_best.skew()), kurt=float(mr_best.kurtosis() + 3),
)
print(f"CAGR={best_res['cagr']:.1%}  Sharpe={best_res['sharpe']:.2f}  MaxDD={best_res['max_dd']:.1%}")
print(f"Block bootstrap Sharpe p5/p50/p95: {bb['p5']:.2f}/{bb['p50']:.2f}/{bb['p95']:.2f}")
print(f"DSR (n_trials={len(all_results)}): {dsr:.4f}")

# Also best by Sharpe
best_sharpe_row = df.sort_values("sharpe", ascending=False).iloc[0]
print(f"\n=== Best by Sharpe: {best_sharpe_row['name']} ===")
print(f"CAGR={best_sharpe_row['cagr']:.1%}  Sharpe={best_sharpe_row['sharpe']:.2f}  MaxDD={best_sharpe_row['max_dd']:.1%}")

mr_best.to_csv(OUT / "best_monthly_returns.csv")
summary = {
    "best_by_cagr": best_row["name"],
    "best_by_sharpe": best_sharpe_row["name"],
    "cagr": float(best_row["cagr"]),
    "sharpe": float(best_row["sharpe"]),
    "max_dd": float(best_row["max_dd"]),
    "bb_p5": bb["p5"], "bb_p50": bb["p50"], "bb_p95": bb["p95"],
    "dsr": dsr, "n_trials_total": len(all_results) + 40,  # +40 from exp_001
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll done. {len(df)} configs. Results in {OUT}")
