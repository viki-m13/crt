"""
Experiment 017: LGBM Training Variants

MOTIVATION:
Signal blending ceiling is at Sharpe~1.841. The LGBM itself might be better
calibrated with different training targets:

A) Multi-period target: train on 3m forward return rank instead of 1m.
   Hypothesis: 3m signal is more stable → better OOS IC → higher Sharpe.
   Implementation: use the 3m forward return from the precomputed panel.

B) Ensemble diversity: average scores from K different LGBM random seeds.
   Hypothesis: variance reduction improves ranking quality → higher effective IC.
   Implementation: train N=5 LGBM models with different random_state, average.

C) Different LGBM hyperparameters: deeper trees (num_leaves=127), more estimators.
   Hypothesis: more complex model captures nonlinear interactions.
   Risk: overfitting.

D) Longer training window: 60m or 72m instead of 48m.
   Hypothesis: more stable model → less overfitting.

Prior best: LGBM×0.63+sh12×0.18+sh5y×0.09+vol_asym_60×0.10, K=30 → Sharpe=1.841
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import lightgbm as lgb

from backtest.engine import (
    make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import set_date_context

print("=" * 90)
print("EXPERIMENT 017: LGBM TRAINING VARIANTS")
print("=" * 90)

print("\nLoading data...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()
monthly_px = get_monthly_prices()

OOS_START = "2007-01-31"
OOS_END = "2021-12-31"
COST = 5.0 / 10_000.0

# Best blend from exp_014
BLEND = {"lgbm": 0.63, "sharpe_12m": 0.18, "sharpe_5y": 0.09, "vol_asym_60": 0.10}


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


# ---------------------------------------------------------------------------
# Build model caches for different configurations
# ---------------------------------------------------------------------------
print("\nBuilding LGBM cache (standard 48m, 1m target)...")
wf_std = WalkForwardLGBM(train_months=48, embargo_months=3, min_train_months=24)
wf_std.prepare_data(prices, dates_all)
lgbm_cache_std = {d: fn for d in dates_all
                  if (fn := wf_std.get_score_fn(d, dates_all)) is not None}
print(f"  Standard LGBM: {len(lgbm_cache_std)} dates")

print("\nBuilding LGBM cache (60m training window)...")
wf_60m = WalkForwardLGBM(train_months=60, embargo_months=3, min_train_months=36)
wf_60m.prepare_data(prices, dates_all)
lgbm_cache_60m = {d: fn for d in dates_all
                  if (fn := wf_60m.get_score_fn(d, dates_all)) is not None}
print(f"  60m LGBM: {len(lgbm_cache_60m)} dates")

print("\nBuilding LGBM cache (72m training window)...")
wf_72m = WalkForwardLGBM(train_months=72, embargo_months=3, min_train_months=48)
wf_72m.prepare_data(prices, dates_all)
lgbm_cache_72m = {d: fn for d in dates_all
                  if (fn := wf_72m.get_score_fn(d, dates_all)) is not None}
print(f"  72m LGBM: {len(lgbm_cache_72m)} dates")


def make_ensemble_fn(caches, blend_weights=BLEND):
    """Score function that averages across multiple LGBM caches."""
    def score_fn(feat_df, date):
        scores_list = []
        for cache in caches:
            fn = cache.get(date)
            if fn is None:
                continue
            s = fn(feat_df).dropna()
            s = s[~s.index.isin(EXCLUDE)]
            if len(s) > 10:
                scores_list.append(znorm(s))
        if not scores_list:
            return pd.Series(dtype=float)
        # Average the z-normalized LGBM scores
        base = pd.concat(scores_list, axis=1).mean(axis=1).dropna()
        blended = base * blend_weights.get("lgbm", 0.0)
        for feat, wt in blend_weights.items():
            if feat == "lgbm":
                continue
            if feat in feat_df.columns:
                idx = base.index
                s = feat_df.loc[feat_df.index.isin(idx), feat].reindex(idx).fillna(0.0)
                blended = blended + znorm(s) * wt
        return blended.dropna()
    return score_fn


def run(name, score_fn, top_k, target_vol=0.18, regime="200ma_loose",
        start=OOS_START, end=OOS_END):
    t0 = time.time()
    regime_fn = make_regime_fn(regime)
    feat_dates = get_feat_dates()
    dates_range = [d for d in feat_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates_range) < 6:
        return None

    records = []
    for i, date in enumerate(dates_range[:-1]):
        next_date = dates_range[i + 1]
        feats = load_features(date)
        if feats.empty or not regime_fn(date, feats):
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
            continue

        stats = get_spy_stats_at(date)
        spy_vol = stats.get("vol_21d", target_vol) if stats else target_vol
        scale = min(target_vol / spy_vol, 1.0) if target_vol and spy_vol > 1e-6 else 1.0

        set_date_context(date)
        scores = score_fn(feats, date).dropna()
        scores = scores[~scores.index.isin(EXCLUDE)]
        if scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        top = scores.sort_values(ascending=False).head(top_k)
        tickers = top.index.tolist()

        d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
        if d0_idx > 0 and monthly_px.index[d0_idx] > date:
            d0_idx -= 1
        d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
        if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
            d1_idx -= 1
        p0 = monthly_px.iloc[d0_idx]
        p1 = monthly_px.iloc[d1_idx]

        common = [t for t in tickers if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        vols = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols.append(0.20)
        inv_v = 1.0 / np.array(vols)
        weights = inv_v / inv_v.sum()

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        raw_ret = float((weights * rets).sum())
        port_ret = scale * raw_ret - 2 * COST * scale
        records.append({"date": date, "ret_m": port_ret, "n_picks": len(common), "scale": scale})

    if not records:
        return None
    df = pd.DataFrame(records).set_index("date")
    m = compute_metrics(df["ret_m"])
    if not m:
        return None

    elapsed = time.time() - t0
    res = {
        "name": name, "top_k": top_k,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4), "n_months": int(m["n_months"]),
        "cash_months": int((df["n_picks"] == 0).sum()),
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(df["scale"].mean(), 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:78s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference (standard 48m window)
# ---------------------------------------------------------------------------
def make_blend_fn_single(cache, blend_weights=BLEND):
    def score_fn(feat_df, date):
        fn = cache.get(date)
        lgbm = (fn(feat_df) if fn else pd.Series(dtype=float)).dropna()
        lgbm = lgbm[~lgbm.index.isin(EXCLUDE)]
        idx = lgbm.index
        blended = znorm(lgbm) * blend_weights.get("lgbm", 0.0)
        for feat, wt in blend_weights.items():
            if feat == "lgbm": continue
            if feat in feat_df.columns:
                s = feat_df.loc[feat_df.index.isin(idx), feat].reindex(idx).fillna(0.0)
                blended = blended + znorm(s) * wt
        return blended.dropna()
    return score_fn


print("\n--- Reference: 48m standard LGBM ---")
for k in [25, 30, 40]:
    r = run(f"REF 48m LGBM K={k:3d}", make_blend_fn_single(lgbm_cache_std), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# D. Longer training windows
# ---------------------------------------------------------------------------
print("\n--- D. Longer training windows (60m, 72m) ---")
for name, cache in [("60m", lgbm_cache_60m), ("72m", lgbm_cache_72m)]:
    for k in [20, 25, 30, 40]:
        r = run(f"{name}_LGBM K={k:3d}", make_blend_fn_single(cache), k)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# B. Ensemble: average 48m + 60m + 72m (diversity through window length)
# ---------------------------------------------------------------------------
print("\n--- B. Ensemble diversity through window lengths ---")
for k in [20, 25, 30, 40]:
    r = run(f"ensemble_48+60+72 K={k:3d}",
            make_ensemble_fn([lgbm_cache_std, lgbm_cache_60m, lgbm_cache_72m]), k)
    if r: RESULTS.append(r)

for k in [25, 30]:
    r = run(f"ensemble_48+72 K={k:3d}",
            make_ensemble_fn([lgbm_cache_std, lgbm_cache_72m]), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
passing = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]

print(f"\nTop 20 by Sharpe:")
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_017_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      ratio={best['ratio']:.4f} (target 0.5774)")

if len(passing) > 0:
    print(f"\n*** {len(passing)} CONFIGS PASS BOTH GATES ***")
    print(passing[["name", "cagr", "sharpe", "max_dd"]].to_string(index=False))
    import subprocess
    best_pass = passing.iloc[0]
    subprocess.run(["/home/user/crt/quant_research/notify/send_success.sh",
                    best_pass["name"], str(best_pass["cagr"]),
                    str(best_pass["sharpe"]), str(best_pass["max_dd"])],
                   capture_output=True)
else:
    print(f"\nBest Sharpe: {best['sharpe']:.2f} (target 2.0)")
    best_cagr = df_res[df_res["cagr"] >= 0.50]
    if len(best_cagr):
        print(f"Best with CAGR≥50%: {best_cagr.iloc[0]['name']} → Sharpe={best_cagr.iloc[0]['sharpe']:.2f}")

n_configs = len(df_res)
print(f"\nTotal configs: {n_configs}")
