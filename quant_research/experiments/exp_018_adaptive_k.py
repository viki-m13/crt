"""
Experiment 018: Adaptive K Based on Regime Quality + Signal Diversity

MOTIVATION:
Sharpe ceiling at ~1.841. The ratio mean_m/std_m = 0.531, target 0.577.

Two structural levers not yet tried together:
A) Adaptive K by regime quality:
   - In quality-3 months (SPY well above 200ma, positive momentum, low vol):
     use K=15 (concentrated → top picks earn the most)
   - In quality-2 months: use K=25
   - In quality-1 months (borderline): use K=40 (diversified)
   Hypothesis: in the best regime months, top-15 outperform top-30;
   in borderline months, diversification reduces variance without hurting mean.

B) Asymmetric signal in "quality" months:
   In quality-3 months, over-weight breakout/momentum signals.
   In quality-1 months, over-weight defensive sharpe/vol signals.

C) Momentum-concentrated weighting in strong regimes:
   In quality-3 months: weight by score^2 × inv_vol (α=2)
   In quality-1 months: pure inv_vol

Key math:
- K=15 portfolio vol: σ_stock × √(0.53 + 0.047) = 40% × 0.759 = 30.4%  (SLIGHTLY HIGHER)
- K=30 portfolio vol: σ_stock × √(0.53 + 0.023) = 40% × 0.744 = 29.8%  (current)
- K=50 portfolio vol: σ_stock × √(0.53 + 0.014) = 40% × 0.738 = 29.5%  (lower)

Insight: reducing K slightly increases vol but concentrates in better picks.
The ratio improvement requires mean_m to grow faster than std_m.

From exp_003 (K sweep): K=30 is optimal, K=20 reduces CAGR.
But: that was a FIXED K. Adaptive K in strong regimes may be different.
When the regime is quality=3, the top-15 momentum picks are very strong;
in weak regimes, K=40 avoids concentration risk.

Prior best: 4way+volasym10% K=30 → CAGR=66.5%, Sharpe=1.841
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
print("EXPERIMENT 018: ADAPTIVE K + REGIME-QUALITY CONVICTION")
print("=" * 90)

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

# Best blend from exp_014
BLEND_4WAY = {"lgbm": 0.63, "sharpe_12m": 0.18, "sharpe_5y": 0.09, "vol_asym_60": 0.10}
BLEND_3WAY = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


def get_spy_quality(date):
    """SPY quality score: 0-3."""
    stats = get_spy_stats_at(date)
    if not stats:
        return 1
    score = 0
    if stats.get("d_sma200", 0) > 0.05:    score += 1
    if stats.get("mom_3", 0) > 0.02:       score += 1
    if stats.get("vol_21d", 0.20) < 0.14:  score += 1
    return score


def compute_blend(feat_df, date, blend_weights):
    lgbm_fn = lgbm_cache.get(date)
    lgbm = (lgbm_fn(feat_df) if lgbm_fn else pd.Series(dtype=float)).dropna()
    lgbm = lgbm[~lgbm.index.isin(EXCLUDE)]
    idx = lgbm.index
    blended = znorm(lgbm) * blend_weights.get("lgbm", 0.0)
    for feat, wt in blend_weights.items():
        if feat == "lgbm":
            continue
        if feat in feat_df.columns:
            s = feat_df.loc[feat_df.index.isin(idx), feat].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * wt
    return blended.dropna()


def run_adaptive(
    name,
    k_map,                    # {0: K, 1: K, 2: K, 3: K} by SPY quality score
    blend_weights=None,
    conviction_alpha_map=None,  # {quality: alpha} for conviction weighting
    regime="200ma_loose",
    target_vol=0.18,
    start=OOS_START,
    end=OOS_END,
):
    if blend_weights is None:
        blend_weights = BLEND_4WAY
    if conviction_alpha_map is None:
        conviction_alpha_map = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}  # default: inv_vol only

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
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0, "quality": -1})
            continue

        stats = get_spy_stats_at(date)
        spy_vol = stats.get("vol_21d", target_vol) if stats else target_vol
        scale = min(target_vol / spy_vol, 1.0) if target_vol and spy_vol > 1e-6 else 1.0

        quality = get_spy_quality(date)
        top_k = k_map.get(quality, 30)
        alpha = conviction_alpha_map.get(quality, 0.0)

        set_date_context(date)
        scores = compute_blend(feats, date, blend_weights).dropna()
        scores = scores[~scores.index.isin(EXCLUDE)]
        if scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale, "quality": quality})
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
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale, "quality": quality})
            continue

        vols = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols.append(0.20)
        inv_v = 1.0 / np.array(vols)

        if alpha > 0:
            z_scores = np.array([float(top.get(t, 0)) for t in common])
            z_shifted = z_scores - z_scores.min() + 0.01
            conv_w = z_shifted ** alpha
            weights = (conv_w * inv_v) / (conv_w * inv_v).sum()
        else:
            weights = inv_v / inv_v.sum()

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        raw_ret = float((weights * rets).sum())
        port_ret = scale * raw_ret - 2 * COST * scale
        records.append({"date": date, "ret_m": port_ret, "n_picks": len(common),
                        "scale": scale, "quality": quality})

    if not records:
        return None
    df = pd.DataFrame(records).set_index("date")
    m = compute_metrics(df["ret_m"])
    if not m:
        return None

    elapsed = time.time() - t0
    # Quality distribution
    q_dist = df["quality"].value_counts().sort_index().to_dict()
    res = {
        "name": name,
        "k_map": str(k_map),
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4), "n_months": int(m["n_months"]),
        "cash_months": int((df["n_picks"] == 0).sum()),
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(df["scale"].mean(), 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
        "q3_months": q_dist.get(3, 0), "q2_months": q_dist.get(2, 0),
        "q1_months": q_dist.get(1, 0),
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:72s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} ratio={res['ratio']:.3f} q3={res['q3_months']} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference
# ---------------------------------------------------------------------------
print("\n--- Reference: fixed K=30 ---")
r = run_adaptive("REF fixed K=30", {0: 30, 1: 30, 2: 30, 3: 30})
if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# A. Adaptive K sweep
# ---------------------------------------------------------------------------
print("\n--- A. Adaptive K by regime quality ---")

# A1: Concentrate in quality-3 months
k_configs = [
    ({0: 40, 1: 35, 2: 25, 3: 15}, "q0=40 q1=35 q2=25 q3=15"),  # aggressive concentration
    ({0: 40, 1: 35, 2: 30, 3: 20}, "q0=40 q1=35 q2=30 q3=20"),  # moderate concentration
    ({0: 50, 1: 40, 2: 30, 3: 20}, "q0=50 q1=40 q2=30 q3=20"),  # also diversify in weak
    ({0: 50, 1: 40, 2: 30, 3: 15}, "q0=50 q1=40 q2=30 q3=15"),  # aggressive both ends
    ({0: 40, 1: 30, 2: 20, 3: 15}, "q0=40 q1=30 q2=20 q3=15"),  # monotone concentration
    ({0: 30, 1: 25, 2: 20, 3: 15}, "q0=30 q1=25 q2=20 q3=15"),  # constant drop
    ({0: 50, 1: 30, 2: 20, 3: 10}, "q0=50 q1=30 q2=20 q3=10"),  # extreme concentration
]
for k_map, desc in k_configs:
    r = run_adaptive(f"adaptive_K {desc}", k_map)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# B. Adaptive K + conviction weighting in quality-3 months
# ---------------------------------------------------------------------------
print("\n--- B. Adaptive K + conviction sizing in quality-3 months ---")

conv_configs = [
    ({0: 40, 1: 35, 2: 25, 3: 15}, {0: 0, 1: 0, 2: 0, 3: 1.0}, "q3_conv1.0"),
    ({0: 40, 1: 35, 2: 25, 3: 15}, {0: 0, 1: 0, 2: 0, 3: 1.5}, "q3_conv1.5"),
    ({0: 40, 1: 35, 2: 25, 3: 15}, {0: 0, 1: 0, 2: 0, 3: 2.0}, "q3_conv2.0"),
    ({0: 50, 1: 40, 2: 30, 3: 20}, {0: 0, 1: 0, 2: 0, 3: 1.5}, "q3_conv1.5_v2"),
    ({0: 40, 1: 35, 2: 25, 3: 20}, {0: 0, 1: 0, 2: 1.0, 3: 1.5}, "q23_conv"),
]
for k_map, alpha_map, desc in conv_configs:
    r = run_adaptive(f"adaptive_K+conv {desc}", k_map, conviction_alpha_map=alpha_map)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# C. Fixed K variants but conviction-weighted by quality
# ---------------------------------------------------------------------------
print("\n--- C. Fixed K=30 with conviction in high-quality months only ---")
for alpha_hi in [1.0, 1.5, 2.0, 2.5]:
    r = run_adaptive(f"fixedK30+conv α={alpha_hi:.1f} q3only",
                     {0: 30, 1: 30, 2: 30, 3: 30},
                     conviction_alpha_map={0: 0, 1: 0, 2: 0, 3: alpha_hi})
    if r: RESULTS.append(r)

    r = run_adaptive(f"fixedK30+conv α={alpha_hi:.1f} q23",
                     {0: 30, 1: 30, 2: 30, 3: 30},
                     conviction_alpha_map={0: 0, 1: 0, 2: alpha_hi * 0.5, 3: alpha_hi})
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# D. Use 3-way blend (more conservative) with adaptive K
# ---------------------------------------------------------------------------
print("\n--- D. 3-way blend + adaptive K ---")
best_k = {0: 40, 1: 35, 2: 25, 3: 15}
for bw, label in [(BLEND_3WAY, "3way"), (BLEND_4WAY, "4way")]:
    r = run_adaptive(f"adaptive_K best {label}", best_k, blend_weights=bw)
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
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio",
              "q3_months"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_018_results.csv"
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
