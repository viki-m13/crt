"""
Experiment 016: Conviction-Based Sizing + Vol-Screen Universe

MOTIVATION:
Current: pick top-K by blend score → weight by inv_vol (ignoring score magnitudes)
Problem: all K stocks get treated equally; marginal K-th pick gets same weight as #1 pick

APPROACH A — Conviction-based sizing:
  Weight within top-K by (blend_z_score^α) × inv_vol
  where α > 0 concentrates more weight in the highest-scoring picks.
  Hypothesis: if top picks have higher realized returns (positive IC), ratio improves.
  Test: α = 0.5, 1.0, 2.0, 3.0

APPROACH B — Vol-screen universe (vol_12m < threshold):
  Filter out high-volatility stocks before ranking.
  Individual stock σ drops → portfolio σ drops → std_m drops → Sharpe up.
  Risk: lower-vol stocks have lower raw returns → CAGR might fall below 50%.
  Test: vol_thresh = 0.60, 0.50, 0.45, 0.40, 0.35

APPROACH C — Regime quality exposure scaling:
  Within active months, scale by SPY quality beyond just 200ma gate.
  SPY quality score: far above 200ma + positive short-term momentum + low vol.
  In low-quality active months: reduce exposure to 60-80%.
  Hypothesis: reduces std_m variance while keeping mean_m mostly intact.

APPROACH D — Combined: conviction sizing + vol-screen + regime quality scale

Math target: ratio needs to move from 0.531 to 0.577 (+8.6%).
Options:
  A) mean_m up 8.6% (needs genuinely better stock selection)
  B) std_m down 8.6% (needs portfolio vol reduction from 8.51% to 7.78%)
  C) Both A and B partially

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
print("EXPERIMENT 016: CONVICTION SIZING + VOL-SCREEN + REGIME QUALITY SCALING")
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

# Best blend from exp_014: LGBM×0.70+sh12×0.20+sh5y×0.10 with vol_asym at 10% (but using 3way for simplicity)
# Actually use vol_asym10% blend for best results
BLEND_4WAY = {"lgbm": 0.63, "sharpe_12m": 0.18, "sharpe_5y": 0.09, "vol_asym_60": 0.10}
BLEND_3WAY = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


def compute_blend(feat_df, date, blend_weights=BLEND_3WAY):
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


def get_spy_quality(date):
    """SPY quality score: 0-3 based on trend strength, short-term momentum, low vol."""
    stats = get_spy_stats_at(date)
    if not stats:
        return 1  # neutral
    score = 0
    if stats.get("d_sma200", 0) > 0.05:   score += 1  # SPY well above 200ma
    if stats.get("mom_3", 0) > 0.02:       score += 1  # SPY positive 3m momentum
    if stats.get("vol_21d", 0.20) < 0.14:  score += 1  # Low vol environment
    return score


def run_experiment(
    name,
    top_k,
    blend_weights=None,
    conviction_alpha=0.0,    # 0=inv_vol only; >0 concentrates by score^alpha×inv_vol
    vol_thresh=None,          # Filter to stocks with vol_12m < vol_thresh
    regime_quality_scale=False,  # Scale by SPY quality score
    quality_scale_map=None,  # {0: 0.5, 1: 0.7, 2: 0.9, 3: 1.0}
    regime="200ma_loose",
    target_vol=0.18,
    start=OOS_START,
    end=OOS_END,
):
    if blend_weights is None:
        blend_weights = BLEND_4WAY
    if quality_scale_map is None:
        quality_scale_map = {0: 0.5, 1: 0.75, 2: 0.90, 3: 1.0}

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
        spy_scale = min(target_vol / spy_vol, 1.0) if target_vol and spy_vol > 1e-6 else 1.0

        # Regime quality scaling
        quality_scale = 1.0
        if regime_quality_scale:
            q = get_spy_quality(date)
            quality_scale = quality_scale_map.get(q, 1.0)

        total_scale = spy_scale * quality_scale

        set_date_context(date)
        scores = compute_blend(feats, date, blend_weights).dropna()
        scores = scores[~scores.index.isin(EXCLUDE)]

        # Vol-screen: filter by individual stock volatility
        if vol_thresh is not None and "vol_12m" in feats.columns:
            vol_data = feats["vol_12m"].dropna()
            low_vol_tickers = vol_data[vol_data < vol_thresh].index
            scores = scores[scores.index.isin(low_vol_tickers)]

        if scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": total_scale})
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
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": total_scale})
            continue

        # Compute weights
        vols = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols.append(0.20)
        inv_v = 1.0 / np.array(vols)

        if conviction_alpha > 0:
            # Conviction-weighted: (score_z^alpha) × inv_vol
            z_scores = np.array([float(top.get(t, 0)) for t in common])
            # Shift to positive (z-scores can be negative)
            z_shifted = z_scores - z_scores.min() + 0.01  # ensure positive
            conv_weights = z_shifted ** conviction_alpha
            weights = (conv_weights * inv_v) / (conv_weights * inv_v).sum()
        else:
            weights = inv_v / inv_v.sum()

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        raw_ret = float((weights * rets).sum())
        port_ret = total_scale * raw_ret - 2 * COST * total_scale
        records.append({"date": date, "ret_m": port_ret, "n_picks": len(common),
                        "scale": total_scale})

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
# 0. Reference
# ---------------------------------------------------------------------------
print("\n--- Reference: 4-way blend, K=30, inv_vol, no conviction, no vol-screen ---")
r = run_experiment("REF 4way K= 30 inv_vol", 30, blend_weights=BLEND_4WAY, conviction_alpha=0.0)
if r: RESULTS.append(r)
r = run_experiment("REF 3way K= 30 inv_vol", 30, blend_weights=BLEND_3WAY, conviction_alpha=0.0)
if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# A. Conviction-based sizing (weight by score^α × inv_vol)
# ---------------------------------------------------------------------------
print("\n--- A. Conviction-based sizing (α sweep) ---")
for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for k in [20, 25, 30, 40]:
        r = run_experiment(f"conviction α={alpha:.1f} K={k:3d} 4way",
                           k, conviction_alpha=alpha)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# B. Vol-screen universe
# ---------------------------------------------------------------------------
print("\n--- B. Vol-screen universe (vol_12m < threshold) ---")
for vt in [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]:
    for k in [20, 25, 30, 40]:
        r = run_experiment(f"vol_screen<{vt:.0%} K={k:3d} 4way",
                           k, vol_thresh=vt)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# C. Regime quality exposure scaling
# ---------------------------------------------------------------------------
print("\n--- C. Regime quality exposure scaling (SPY quality → exposure) ---")
for scale_map in [
    {0: 0.5, 1: 0.75, 2: 0.90, 3: 1.0},
    {0: 0.5, 1: 0.70, 2: 0.85, 3: 1.0},
    {0: 0.0, 1: 0.60, 2: 0.80, 3: 1.0},
]:
    map_str = "/".join([f"{v:.0%}" for v in scale_map.values()])
    for k in [25, 30, 40]:
        r = run_experiment(f"quality_scale {map_str} K={k:3d}",
                           k, regime_quality_scale=True, quality_scale_map=scale_map)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# D. Combined: best A + best B candidates
# ---------------------------------------------------------------------------
print("\n--- D. Conviction α=1.5 + vol_screen ---")
for vt in [0.50, 0.55]:
    for k in [25, 30]:
        r = run_experiment(f"conviction+volscreen<{vt:.0%} K={k:3d}",
                           k, conviction_alpha=1.5, vol_thresh=vt)
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

out = Path(__file__).parent / "exp_016_results.csv"
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
print(f"Running total hypotheses: 488 + exp015_n + {n_configs}")
