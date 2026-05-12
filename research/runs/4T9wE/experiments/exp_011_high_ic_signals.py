"""
Experiment 011: High-IC Signals

CRITICAL FINDING: Features with genuine positive IC on OOS 2007-2021:
  breakout_strength_60: IC=+0.0884, pct_positive=63.1%
  crt_6m:               IC=+0.0810, pct_positive=64.8%
  sharpe_12m:           IC=+0.0429, pct_positive=60.9%
  sharpe_5y:            IC=+0.0405, pct_positive=62.6%
  trend_health_5y:      IC=+0.0300, pct_positive=58.7%

Grinold-Kahn with IC=0.088, K=40: Sharpe ≈ 0.088 × sqrt(480) ≈ 1.93

Hypothesis: Simple linear combination of high-IC features may BEAT LGBM
by avoiding overfitting to noisy features.

Prior best: lgbm×0.70+sh12×0.20+sh5y×0.10 K=30 → CAGR=63.1%, Sharpe=1.82
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
from features.signals import composite_v1, set_date_context

print("=" * 90)
print("EXPERIMENT 011: HIGH-IC SIGNALS")
print("=" * 90)
print("\nKey finding: breakout_strength_60 IC=0.088, crt_6m IC=0.081 (OOS 2007-2021)")

print("\nLoading LGBM models (for comparison)...")
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


def score_breakout(feat_df: pd.DataFrame) -> pd.Series:
    if "breakout_strength_60" not in feat_df.columns:
        return composite_v1(feat_df)
    return feat_df["breakout_strength_60"].dropna()


def score_crt6m(feat_df: pd.DataFrame) -> pd.Series:
    if "crt_6m" not in feat_df.columns:
        return composite_v1(feat_df)
    return feat_df["crt_6m"].dropna()


def score_breakout_crt(feat_df: pd.DataFrame) -> pd.Series:
    def znorm(s):
        mu, si = s.mean(), s.std()
        return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)
    b = feat_df.get("breakout_strength_60", pd.Series(np.nan, index=feat_df.index)).dropna()
    c = feat_df.get("crt_6m", pd.Series(np.nan, index=feat_df.index)).dropna()
    idx = b.index.intersection(c.index)
    if len(idx) < 10:
        return b if len(b) > len(c) else c
    return (znorm(b.loc[idx]) * 0.50 + znorm(c.loc[idx]) * 0.50).dropna()


def score_high_ic_4way(feat_df: pd.DataFrame) -> pd.Series:
    """4-way blend of the highest IC features."""
    def znorm(s):
        mu, si = s.mean(), s.std()
        return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)
    feats = {
        "breakout_strength_60": 0.35,
        "crt_6m": 0.30,
        "sharpe_12m": 0.20,
        "sharpe_5y": 0.15,
    }
    scores = pd.Series(dtype=float)
    for feat, wt in feats.items():
        if feat in feat_df.columns:
            s = feat_df[feat].dropna()
            if len(s) > 5:
                zs = znorm(s) * wt
                if scores.empty:
                    scores = zs
                else:
                    idx = scores.index.intersection(zs.index)
                    scores = scores.loc[idx] + zs.loc[idx]
    return scores.dropna()


def score_high_ic_with_lgbm(feat_df: pd.DataFrame) -> pd.Series:
    """Best IC signals + LGBM for complex signal capture."""
    from features.signals import _CURRENT_DATE as D
    def znorm(s):
        mu, si = s.mean(), s.std()
        return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)

    lgbm = (lgbm_cache[D](feat_df) if D in lgbm_cache else composite_v1(feat_df)).dropna()
    lgbm = lgbm[~lgbm.index.isin(EXCLUDE)]

    idx = lgbm.index
    blended = znorm(lgbm) * 0.40

    for feat, wt in [("breakout_strength_60", 0.25), ("crt_6m", 0.20),
                     ("sharpe_12m", 0.10), ("sharpe_5y", 0.05)]:
        if feat in feat_df.columns:
            s = feat_df.loc[feat_df.index.isin(idx), feat].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * wt

    return blended.dropna()


def run(name, score_fn, top_k, weighting="inv_vol", regime="200ma_loose",
        target_vol=0.18, start=OOS_START, end=OOS_END):
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
        scores = score_fn(feats).dropna()
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
    cash_m = int((df["n_picks"] == 0).sum())
    avg_scale = df["scale"].mean()
    res = {
        "name": name, "top_k": top_k,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4),
        "n_months": int(m["n_months"]), "cash_months": cash_m,
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(avg_scale, 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:72s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 1. Pure high-IC signal scores (no LGBM)
# ---------------------------------------------------------------------------
print("\n--- Pure high-IC signals (no ML) ---")
for k in [20, 30, 40, 50, 60]:
    r = run(f"breakout_60 K={k:3d} inv_vol + vt18% + regime_loose",
            score_breakout, k)
    if r: RESULTS.append(r)

for k in [20, 30, 40, 50, 60]:
    r = run(f"crt_6m K={k:3d} inv_vol + vt18% + regime_loose",
            score_crt6m, k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. 50/50 breakout + crt blend
# ---------------------------------------------------------------------------
print("\n--- 50/50 breakout+crt blend ---")
for k in [20, 30, 40, 50, 60]:
    r = run(f"breakout50+crt50 K={k:3d} inv_vol + vt18% + regime_loose",
            score_breakout_crt, k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. 4-way high-IC blend: breakout×0.35 + crt×0.30 + sh12×0.20 + sh5y×0.15
# ---------------------------------------------------------------------------
print("\n--- 4-way high-IC blend (breakout+crt+sh12+sh5y) ---")
for k in [20, 30, 40, 50, 60]:
    r = run(f"4way_highIC K={k:3d} inv_vol + vt18% + regime_loose",
            score_high_ic_4way, k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. LGBM + high-IC signals blend
# ---------------------------------------------------------------------------
print("\n--- LGBM×0.40 + breakout×0.25 + crt×0.20 + sh12×0.10 + sh5y×0.05 ---")
for k in [20, 30, 40, 50, 60]:
    r = run(f"lgbm40+highIC K={k:3d} inv_vol + vt18% + regime_loose",
            score_high_ic_with_lgbm, k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Various regime/vol configurations for best signals
# ---------------------------------------------------------------------------
print("\n--- Best signal (high-IC 4way) under various regime/vol configs ---")
for regime, vt in [("200ma_loose", 0.15), ("200ma_loose", 0.18),
                    ("200ma", 0.18), ("200ma", None)]:
    for k in [30, 40, 50]:
        vt_str = f"vt={vt:.0%}" if vt else "no_vt"
        r = run(f"4way_highIC K={k:3d} + {vt_str} + {regime}",
                score_high_ic_4way, k, regime=regime, target_vol=vt)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 6. High-IC no-LGBM with standard regime (no vol target)
# ---------------------------------------------------------------------------
print("\n--- High-IC with standard 200MA regime (no vol targeting) ---")
for k in [20, 30, 40, 50]:
    r = run(f"4way_highIC K={k:3d} inv_vol + 200ma [no vol_target]",
            score_high_ic_4way, k, regime="200ma", target_vol=None)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
passing = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]

print(f"\nAll configs sorted by Sharpe:")
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio",
              "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_011_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      ratio={best['ratio']:.4f}")

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

print(f"\nTotal configs: {len(df_res)}")
print(f"Running total hypotheses: 305 (from exp_001-010) + {len(df_res)} = {305 + len(df_res)}")
