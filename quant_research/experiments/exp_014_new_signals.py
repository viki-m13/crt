"""
Experiment 014: New High-IC Signals in Blend

FULL IC AUDIT FINDINGS (active months, OOS 2007-2021):
  rank_now:              IC=0.223  (SKIP — likely look-ahead; IC drops to 0.015 at 3m lag)
  ret_21d:               IC=0.221  (SKIP — already in LGBM; short-term reversal in most markets)
  breakout_strength_60:  IC=0.159  (USED in exp_011 but at lower weight)
  crt_3m:                IC=0.135  (NEW — higher IC than sharpe_12m=0.081, similar concept)
  prerunner_dist:        IC=0.137  (NEW — unused signal)
  vol_asym_60:           IC=0.105  (NEW — unused signal)
  rs_6m_spy:             IC=0.107  (USED as filter in exp_007, never as blend component)
  cst_score:             IC=0.060  (NEW — unused signal)

HYPOTHESIS: Replacing sharpe_12m (IC=0.081) with crt_3m (IC=0.135) or adding
prerunner_dist/vol_asym_60 should increase portfolio IC and break Sharpe ceiling.

Key caution: High IC of crt_3m, prerunner_dist may partially reflect survivorship bias.
The OOS backtest will give the true test.

Prior best: LGBM×0.70+sh12×0.20+sh5y×0.10, K=30, vt18%, loose → CAGR=63.1%, Sharpe=1.82
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from backtest.engine import (
    make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import set_date_context

print("=" * 90)
print("EXPERIMENT 014: NEW HIGH-IC SIGNALS IN BLEND")
print("=" * 90)
print("\nNew signals: crt_3m (IC=0.135), prerunner_dist (IC=0.137), vol_asym_60 (IC=0.105)")
print("Prior best blend: LGBM×0.70+sh12×0.20+sh5y×0.10 → Sharpe=1.82")

print("\nLoading data and LGBM models...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()
monthly_px = get_monthly_prices()

wf = WalkForwardLGBM(train_months=48, embargo_months=3, min_train_months=24)
wf.prepare_data(prices, dates_all)
lgbm_cache = {d: fn for d in dates_all
              if (fn := wf.get_score_fn(d, dates_all)) is not None}
print(f"LGBM ready for {len(lgbm_cache)} dates")

OOS_START = "2007-01-31"
OOS_END = "2021-12-31"
COST = 5.0 / 10_000.0


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


def make_blend_fn(weights: dict):
    """Build a score function from a weight dict keyed by feature name or 'lgbm'."""
    def score_fn(feat_df: pd.DataFrame, date) -> pd.Series:
        lgbm_fn = lgbm_cache.get(date)
        lgbm = (lgbm_fn(feat_df) if lgbm_fn else pd.Series(dtype=float)).dropna()
        lgbm = lgbm[~lgbm.index.isin(EXCLUDE)]
        idx = lgbm.index

        blended = znorm(lgbm) * weights.get("lgbm", 0.0)
        for feat, wt in weights.items():
            if feat == "lgbm" or wt == 0.0:
                continue
            if feat in feat_df.columns:
                s = feat_df.loc[feat_df.index.isin(idx), feat].reindex(idx).fillna(0.0)
                blended = blended + znorm(s) * wt
        return blended.dropna()
    return score_fn


def run(name, score_fn, top_k, regime="200ma_loose", target_vol=0.18,
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
        if feats.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
            continue
        if not regime_fn(date, feats):
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
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference: current best 3-way blend (exp_009)
# ---------------------------------------------------------------------------
print("\n--- Reference: 3-way blend (LGBM×0.70+sh12×0.20+sh5y×0.10) ---")
ref_blend = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}
for k in [25, 30, 40]:
    r = run(f"REF 3way K={k:3d}", make_blend_fn(ref_blend), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 1. Replace sharpe_12m with crt_3m (higher IC: 0.135 vs 0.081)
# ---------------------------------------------------------------------------
print("\n--- Replace sh12 with crt_3m (IC=0.135 vs 0.081) ---")
for k in [20, 25, 30, 40]:
    blend = {"lgbm": 0.70, "crt_3m": 0.20, "sharpe_5y": 0.10}
    r = run(f"lgbm70+crt3m20+sh5y10 K={k:3d}", make_blend_fn(blend), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Add crt_3m alongside the 3-way blend (renormalized)
# ---------------------------------------------------------------------------
print("\n--- 4-way: LGBM+sh12+sh5y+crt_3m ---")
for crt3_wt in [0.10, 0.15, 0.20]:
    for k in [25, 30, 40]:
        # Renormalize: reduce all other weights proportionally
        base = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}
        total = sum(base.values())
        blend = {k2: v * (1 - crt3_wt) / total for k2, v in base.items()}
        blend["crt_3m"] = crt3_wt
        wts = "+".join(f"{k2[:4]}{v:.0%}" for k2, v in blend.items())
        r = run(f"4way+crt3m{crt3_wt:.0%} K={k:3d}", make_blend_fn(blend), k)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Add prerunner_dist (IC=0.137) to the blend
# ---------------------------------------------------------------------------
print("\n--- 4-way: LGBM+sh12+sh5y+prerunner_dist ---")
for pd_wt in [0.10, 0.15, 0.20]:
    for k in [25, 30, 40]:
        base = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}
        total = sum(base.values())
        blend = {k2: v * (1 - pd_wt) / total for k2, v in base.items()}
        blend["prerunner_dist"] = pd_wt
        r = run(f"4way+prerunner{pd_wt:.0%} K={k:3d}", make_blend_fn(blend), k)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. Add vol_asym_60 (IC=0.105) to the blend
# ---------------------------------------------------------------------------
print("\n--- 4-way: LGBM+sh12+sh5y+vol_asym_60 ---")
for va_wt in [0.10, 0.15]:
    for k in [25, 30, 40]:
        base = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}
        total = sum(base.values())
        blend = {k2: v * (1 - va_wt) / total for k2, v in base.items()}
        blend["vol_asym_60"] = va_wt
        r = run(f"4way+volasym{va_wt:.0%} K={k:3d}", make_blend_fn(blend), k)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Replace sharpe_12m and sh5y with crt_3m and prerunner_dist
# ---------------------------------------------------------------------------
print("\n--- LGBM + crt_3m + prerunner_dist (replace quality signals) ---")
for k in [20, 25, 30, 40]:
    blend = {"lgbm": 0.65, "crt_3m": 0.20, "prerunner_dist": 0.15}
    r = run(f"lgbm65+crt3m20+prerun15 K={k:3d}", make_blend_fn(blend), k)
    if r: RESULTS.append(r)

for k in [25, 30, 40]:
    blend = {"lgbm": 0.60, "crt_3m": 0.20, "prerunner_dist": 0.10, "sharpe_5y": 0.10}
    r = run(f"lgbm60+crt3m20+prerun10+sh5y10 K={k:3d}", make_blend_fn(blend), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 6. 5-way blend: LGBM + sh12 + sh5y + crt_3m + prerunner_dist
# ---------------------------------------------------------------------------
print("\n--- 5-way blend: LGBM+sh12+sh5y+crt3m+prerunner ---")
for k in [25, 30, 40]:
    blend = {"lgbm": 0.60, "sharpe_12m": 0.15, "sharpe_5y": 0.10,
             "crt_3m": 0.10, "prerunner_dist": 0.05}
    r = run(f"5way_v1 K={k:3d}", make_blend_fn(blend), k)
    if r: RESULTS.append(r)

for k in [25, 30, 40]:
    blend = {"lgbm": 0.55, "sharpe_12m": 0.15, "sharpe_5y": 0.10,
             "crt_3m": 0.12, "prerunner_dist": 0.08}
    r = run(f"5way_v2 K={k:3d}", make_blend_fn(blend), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 7. crt_6m + crt_3m blend (both high IC, complementary timeframes)
# ---------------------------------------------------------------------------
print("\n--- Adding both crt_3m and crt_6m ---")
for k in [25, 30, 40]:
    blend = {"lgbm": 0.60, "sharpe_12m": 0.10, "sharpe_5y": 0.10,
             "crt_3m": 0.10, "crt_6m": 0.10}
    r = run(f"lgbm60+sh12_10+sh5y_10+crt3m_10+crt6m_10 K={k:3d}", make_blend_fn(blend), k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 8. Best new blend at different K values (K=15,20 — concentrate in winners)
# ---------------------------------------------------------------------------
print("\n--- Best new blend candidates at small K (concentration test) ---")
# Based on IC, crt_3m should give better stock selection → smaller K might work
for blend_label, blend in [
    ("lgbm70+crt3m20+sh5y10", {"lgbm": 0.70, "crt_3m": 0.20, "sharpe_5y": 0.10}),
    ("lgbm65+crt3m20+prerun15", {"lgbm": 0.65, "crt_3m": 0.20, "prerunner_dist": 0.15}),
]:
    for k in [15, 20, 25]:
        r = run(f"{blend_label} K={k:3d}", make_blend_fn(blend), k)
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

out = Path(__file__).parent / "exp_014_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      ratio={best['ratio']:.4f} (target 0.5774)")

if len(passing) > 0:
    print(f"\n{'='*50}")
    print(f"*** {len(passing)} CONFIGS PASS BOTH GATES (CAGR≥50% AND Sharpe≥2.0) ***")
    print(passing[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio"]].to_string(index=False))
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
print(f"Running total hypotheses: 435 (from exp_001-013) + {n_configs} = {435 + n_configs}")
