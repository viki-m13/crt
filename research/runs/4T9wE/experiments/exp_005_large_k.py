"""
Experiment 005: Very Large K + Regime for Sharpe 2.0

Analysis from exp_004: IC ≈ 0 (signals don't predict returns strongly).
Performance comes from: universe selection (semi-survivorship) + regime timing.

Key insight: For Sharpe 2.0 with current mean returns (~4%/mo), need monthly
vol ≤ 5.94%. Monthly vol ≈ stock_vol / sqrt(K × correlation_adj).

With current K=40 → vol=9.12%: need K≈100 to get vol≈5.76%.

Hypothesis: lgbm K=80-120 inv_vol + regime → CAGR≥50% AND Sharpe≥2.0

Also test: regime-only (EW all stocks with regime gate) as the fundamental baseline.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from backtest.engine import (
    run_backtest, make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import composite_v1, smooth_compounder, set_date_context

print("=" * 90)
print("EXPERIMENT 005: LARGE K FOR SHARPE 2.0")
print("=" * 90)

# Load LGBM models
print("\nLoading LGBM models...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()
monthly_px = get_monthly_prices()

wf = WalkForwardLGBM(train_months=48, embargo_months=3, min_train_months=24)
wf.prepare_data(prices, dates_all)
score_fn_cache = {d: fn for d in dates_all
                  if (fn := wf.get_score_fn(d, dates_all)) is not None}
print(f"LGBM ready for {len(score_fn_cache)} dates")

# Score functions
def lgbm_fn(f):
    from features.signals import _CURRENT_DATE as D
    return score_fn_cache[D](f) if D in score_fn_cache else composite_v1(f)

def lgbm_smooth_fn(f):
    from features.signals import _CURRENT_DATE as D
    lg = score_fn_cache[D](f) if D in score_fn_cache else composite_v1(f)
    sm = smooth_compounder(f)
    def z(s): mu, si = s.mean(), s.std(); return (s-mu)/si if si>1e-10 else pd.Series(0.,index=s.index)
    comb = z(lg)*1.5 + z(sm)*1.0
    mom12 = f.get("mom_12_1", pd.Series(1.,index=f.index)) if "mom_12_1" in f.columns else pd.Series(1.,index=f.index)
    comb[mom12<=0] = np.nan
    return comb

# ---------------------------------------------------------------------------
# Regime-only baseline: EW top-Q% by quality (not ML)
# ---------------------------------------------------------------------------
def regime_only_baseline(feat_df: pd.DataFrame) -> pd.Series:
    """Simple quality-ranked composite — no ML."""
    return composite_v1(feat_df)


def run(name, score_fn, top_k, weighting="ew", regime=None,
        start="2007-01-31", end="2021-12-31"):
    t0 = time.time()
    regime_fn_obj = make_regime_fn(regime) if regime else None
    df, m = run_backtest(score_fn=score_fn, start=start, end=end,
                         top_k=top_k, weighting=weighting,
                         cost_bps=5.0, regime_fn=regime_fn_obj)
    elapsed = time.time() - t0
    if not m:
        print(f"  {name}: NO RESULTS"); return None
    cash_m = int((df["n_picks"]==0).sum()) if "n_picks" in df.columns else 0
    res = {
        "name": name, "top_k": top_k, "weighting": weighting,
        "cagr": round(float(m["cagr"]),4), "sharpe": round(float(m["sharpe"]),3),
        "max_dd": round(float(m["max_dd"]),4), "win_rate": round(float(m["win_rate"]),3),
        "ann_vol": round(float(m["ann_vol"]),4),
        "n_months": int(m["n_months"]), "cash_months": cash_m,
        "mean_m": round(float(m["mean_m"]),5), "std_m": round(float(m["std_m"]),5),
    }
    gc = "✓" if res["cagr"]>=0.50 else "✗"
    gs = "✓" if res["sharpe"]>=2.0 else "✗"
    print(f"  {name:65s} CAGR={res['cagr']:.1%}{gc} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Cash={cash_m}m Vol={res['ann_vol']:.1%} {elapsed:.0f}s")
    return res

RESULTS = []

print("\n--- Regime-Only Baseline (composite_v1, no ML) ---")
for k in [20, 50, 100, 150, 200]:
    r = run(f"composite_v1 K={k:3d} + regime_200ma", composite_v1, k, "inv_vol", "200ma")
    if r: RESULTS.append(r)

print("\n--- LGBM, Very Large K + Regime ---")
for k in [50, 60, 70, 80, 100, 120, 150]:
    r = run(f"lgbm K={k:3d} inv_vol + regime_200ma", lgbm_fn, k, "inv_vol", "200ma")
    if r: RESULTS.append(r)

print("\n--- LGBM_Smooth, Very Large K + Regime ---")
for k in [40, 50, 60, 70, 80, 100]:
    r = run(f"lgbm_smooth K={k:3d} inv_vol + regime_200ma", lgbm_smooth_fn, k, "inv_vol", "200ma")
    if r: RESULTS.append(r)

# Try: subset selection within large K using composite_v1 as pre-filter
print("\n--- Hybrid: Pre-filter top-2K by composite, then LGBM for final K ---")
def hybrid_2x(k_inner: int):
    def score_fn(feat_df: pd.DataFrame) -> pd.Series:
        from features.signals import _CURRENT_DATE as D
        # Pre-filter: top-2K by composite_v1
        comp = composite_v1(feat_df)
        top_2k = comp.sort_values(ascending=False).head(k_inner * 2).index
        feat_sub = feat_df.loc[feat_df.index.isin(top_2k)]
        if feat_sub.empty:
            return comp
        # Then LGBM on the subset
        if D in score_fn_cache:
            lgbm_scores = score_fn_cache[D](feat_sub)
        else:
            lgbm_scores = composite_v1(feat_sub)
        # Fill non-selected with NaN
        result = pd.Series(np.nan, index=feat_df.index)
        result.loc[lgbm_scores.index] = lgbm_scores.values
        return result
    return score_fn

for k in [20, 30, 40, 50]:
    r = run(f"hybrid_2x K={k:2d} inv_vol + regime", hybrid_2x(k), k, "inv_vol", "200ma")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "win_rate", "cash_months",
              "ann_vol", "top_k"]].to_string(index=False))

out = Path(__file__).parent / "exp_005_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")

passed_both = df_res[(df_res["cagr"]>=0.50) & (df_res["sharpe"]>=2.0)]
if len(passed_both) > 0:
    print(f"\n{'='*40}")
    print(f"*** {len(passed_both)} CONFIGS PASS BOTH GATES ***")
    print(passed_both[["name","cagr","sharpe","max_dd","ann_vol"]].to_string(index=False))
else:
    best_sharpe = df_res.iloc[0]["sharpe"]
    best_cagr_row = df_res[df_res["cagr"]>=0.50].iloc[0] if len(df_res[df_res["cagr"]>=0.50]) > 0 else None
    print(f"\nBest Sharpe achieved: {best_sharpe:.2f} (target: 2.0)")
    if best_cagr_row is not None:
        print(f"Best config with CAGR≥50%: {best_cagr_row['name']} → Sharpe={best_cagr_row['sharpe']:.2f}")
