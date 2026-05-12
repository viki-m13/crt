"""
Experiment 002: Walk-forward LightGBM Ranker
Tests LightGBM ranker trained on cross-sectional features to predict forward returns.
Walk-forward: 36-month rolling training window, 3-month embargo, tested 2007-2021.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from backtest.engine import (
    run_backtest, make_regime_fn, get_feat_dates, get_prices, load_features, get_monthly_prices
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import composite_v1, smooth_compounder, set_date_context

print("=" * 90)
print("EXPERIMENT 002: WALK-FORWARD LGBM RANKER")
print("=" * 90)

# Prepare data
print("\nPreparing data...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()

wf_lgbm = WalkForwardLGBM(
    train_months=48,
    embargo_months=3,
    min_train_months=24,
)
wf_lgbm.prepare_data(prices, dates_all)
print(f"Data prepared. Total dates: {len(dates_all)}")

# Cache all score functions up front (saves time during backtest)
print("Pre-computing score functions...")
score_fn_cache = {}
for d in dates_all:
    fn = wf_lgbm.get_score_fn(d, dates_all)
    if fn is not None:
        score_fn_cache[d] = fn

first_avail = min(score_fn_cache.keys()) if score_fn_cache else None
print(f"Score functions available from: {first_avail}")
print(f"Total prediction dates: {len(score_fn_cache)}")


def lgbm_score_fn(feat_df: pd.DataFrame) -> pd.Series:
    """Dispatch to cached LightGBM model for current date."""
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    if date is None or date not in score_fn_cache:
        return composite_v1(feat_df)
    return score_fn_cache[date](feat_df)


def lgbm_plus_quality(feat_df: pd.DataFrame) -> pd.Series:
    """LightGBM rank + Sharpe12m quality filter (must have positive Sharpe)."""
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    score = score_fn_cache.get(date, composite_v1)(feat_df) if date else composite_v1(feat_df)
    sharpe12 = feat_df["sharpe_12m"] if "sharpe_12m" in feat_df.columns else pd.Series(1.0, index=feat_df.index)
    score[sharpe12 <= 0] = np.nan
    return score


def lgbm_smooth(feat_df: pd.DataFrame) -> pd.Series:
    """LightGBM rank combined with smooth_compounder features."""
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    if date is None or date not in score_fn_cache:
        return smooth_compounder(feat_df)
    lgbm = score_fn_cache[date](feat_df)
    smooth = smooth_compounder(feat_df)
    # Normalize and combine
    def znorm(s):
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 1e-10 else pd.Series(0.0, index=s.index)
    combined = znorm(lgbm) * 1.5 + znorm(smooth) * 1.0
    mom12 = feat_df["mom_12_1"] if "mom_12_1" in feat_df.columns else pd.Series(1.0, index=feat_df.index)
    combined[mom12 <= 0] = np.nan
    return combined


RESULTS = []
TOTAL_HYP = 0
OOS_START = "2007-01-31"  # At least 4y training before first prediction
OOS_END = "2021-12-31"


def run(name, score_fn, top_k, weighting="ew", regime=None, n_hyp=1):
    global TOTAL_HYP
    TOTAL_HYP += n_hyp
    t0 = time.time()
    regime_fn = make_regime_fn(regime) if regime else None
    df, metrics = run_backtest(
        score_fn=score_fn,
        start=OOS_START, end=OOS_END,
        top_k=top_k, weighting=weighting,
        cost_bps=5.0, regime_fn=regime_fn,
    )
    elapsed = time.time() - t0
    if not metrics:
        print(f"  {name}: NO RESULTS")
        return

    cash_m = int((df["n_picks"] == 0).sum()) if "n_picks" in df.columns else 0
    result = {
        "name": name, "top_k": top_k, "weighting": weighting,
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
    }
    RESULTS.append(result)
    gc = "✓" if result["cagr"] >= 0.50 else "✗"
    gs = "✓" if result["sharpe"] >= 2.0 else "✗"
    print(f"  {name:60s} CAGR={result['cagr']:.1%}{gc} Sharpe={result['sharpe']:.2f}{gs} "
          f"MaxDD={result['max_dd']:.1%} Cash={cash_m}m {elapsed:.1f}s")


print(f"\n--- LightGBM Ranker vs Baseline ---")
print("Baseline (composite_v1 K=20 EW, no regime) — reproduced for comparison:")
run("composite_v1 K=20 EW [baseline]", composite_v1, 20, n_hyp=0)

print("\nLightGBM Ranker configurations:")
for k in [5, 10, 20]:
    run(f"lgbm K={k:2d} EW", lgbm_score_fn, k)
for k in [5, 10, 20]:
    run(f"lgbm K={k:2d} + regime_200ma", lgbm_score_fn, k, regime="200ma")
for k in [10, 20]:
    run(f"lgbm K={k:2d} inv_vol + regime", lgbm_score_fn, k, "inv_vol", "200ma")

print("\nLGBM + Quality filter:")
for k in [5, 10, 20]:
    run(f"lgbm_qual K={k:2d} EW", lgbm_plus_quality, k)
for k in [5, 10]:
    run(f"lgbm_qual K={k:2d} + regime", lgbm_plus_quality, k, regime="200ma")

print("\nLGBM + Smooth compounder ensemble:")
for k in [5, 10, 20]:
    run(f"lgbm_smooth K={k:2d} EW", lgbm_smooth, k)
for k in [5, 10, 20]:
    run(f"lgbm_smooth K={k:2d} + regime", lgbm_smooth, k, regime="200ma")
for k in [10, 20]:
    run(f"lgbm_smooth K={k:2d} inv_vol + regime", lgbm_smooth, k, "inv_vol", "200ma")

# Summary
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "win_rate", "cash_months",
              "ann_vol", "top_k"]].head(15).to_string(index=False))

out = Path(__file__).parent / "exp_002_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(f"\nBest: {df_res.iloc[0]['name']} -> CAGR={df_res.iloc[0]['cagr']:.1%} Sharpe={df_res.iloc[0]['sharpe']:.2f}")
print(f"Total hypotheses: {TOTAL_HYP}")
