"""
Experiment 007: Stock-Level Trend Filter

Key finding from IC analysis:
  Stocks above own 200MA: mean fwd return 2.92%, pct_positive 73.7%
  Stocks below own 200MA: mean fwd return 1.09%, pct_positive 62.0%
  Spread: 1.84%/month

Hypothesis: Pre-filtering to stocks above their own 200-day MA before LGBM ranking
will improve Sharpe by increasing mean return per unit of vol.

Test matrix:
  - d_sma200 threshold: -5%, 0%, +2%, +5%
  - rs_6m_spy > 0 filter (stock outperforming SPY over 6m)
  - Combined filters
  - K ∈ [20, 30, 40, 50, 60]
  - Vol targeting from exp_006 best: target=15-18% + regime_loose

Prior best: lgbm K=50 inv_vol + voltarget(18%) + regime_loose → CAGR=56.9%, Sharpe=1.74
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
print("EXPERIMENT 007: STOCK-LEVEL TREND FILTER")
print("=" * 90)

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

OOS_START = "2007-01-31"
OOS_END = "2021-12-31"
COST = 5.0 / 10_000.0


def run_with_filter(
    name: str,
    top_k: int,
    d200_thresh: float = 0.0,
    rs_thresh: float = None,
    mom12_positive: bool = False,
    sma50_cross: bool = False,
    target_vol: float = None,
    regime_name: str = "200ma_loose",
    start: str = OOS_START,
    end: str = OOS_END,
) -> dict | None:
    """Backtest with stock-level trend filter + optional vol targeting."""
    t0 = time.time()
    regime_fn = make_regime_fn(regime_name)
    feat_dates = get_feat_dates()
    dates = [d for d in feat_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < 6:
        return None

    records = []
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]

        feats = load_features(date)
        if feats.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
            continue
        if not regime_fn(date, feats):
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
            continue

        # Vol scaling
        if target_vol is not None:
            stats = get_spy_stats_at(date)
            spy_vol = stats.get("vol_21d", target_vol) if stats else target_vol
            scale = min(target_vol / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0
        else:
            scale = 1.0

        set_date_context(date)

        # Score and apply stock-level filters
        D = date
        raw_scores = (score_fn_cache[D](feats) if D in score_fn_cache
                      else composite_v1(feats))
        raw_scores = raw_scores.dropna()
        raw_scores = raw_scores[~raw_scores.index.isin(EXCLUDE)]

        # Stock-level d_sma200 filter
        if d200_thresh is not None and "d_sma200" in feats.columns:
            d_sma = feats.loc[feats.index.isin(raw_scores.index), "d_sma200"]
            raw_scores = raw_scores[raw_scores.index.isin(d_sma[d_sma > d200_thresh].index)]

        # Relative strength filter
        if rs_thresh is not None and "rs_6m_spy" in feats.columns:
            rs = feats.loc[feats.index.isin(raw_scores.index), "rs_6m_spy"]
            raw_scores = raw_scores[raw_scores.index.isin(rs[rs > rs_thresh].index)]

        # 12m momentum positive filter
        if mom12_positive and "mom_12_1" in feats.columns:
            mom12 = feats.loc[feats.index.isin(raw_scores.index), "mom_12_1"]
            raw_scores = raw_scores[raw_scores.index.isin(mom12[mom12 > 0].index)]

        # Golden cross filter
        if sma50_cross and "sma50_above_200" in feats.columns:
            gc = feats.loc[feats.index.isin(raw_scores.index), "sma50_above_200"]
            raw_scores = raw_scores[raw_scores.index.isin(gc[gc > 0.5].index)]

        if raw_scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        top = raw_scores.sort_values(ascending=False).head(top_k)
        tickers = top.index.tolist()

        d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
        if d0_idx > 0 and monthly_px.index[d0_idx] > date:
            d0_idx -= 1
        d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
        if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
            d1_idx -= 1
        p0 = monthly_px.iloc[d0_idx]
        p1 = monthly_px.iloc[d1_idx]

        common = [
            t for t in tickers
            if t in monthly_px.columns
            and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
            and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0
        ]
        if not common:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        # inv_vol weights
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
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:72s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Sc={avg_scale:.2f} Cash={cash_m}m {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# Reference from exp_006 best (reproduce for baseline)
# ---------------------------------------------------------------------------
print("\n--- Reference: best from exp_006 ---")
for k in [40, 50]:
    r = run_with_filter(
        f"lgbm K={k:3d} inv_vol + voltarget(18%) + regime_loose [ref]",
        k, d200_thresh=None, target_vol=0.18, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 1. Stock-level d_sma200 filter alone (no vol targeting)
# ---------------------------------------------------------------------------
print("\n--- Stock-level 200MA filter: sweep threshold ---")
for thresh in [-0.05, 0.0, 0.02, 0.05]:
    for k in [30, 40, 50]:
        r = run_with_filter(
            f"lgbm K={k:3d} + d200>{thresh:+.0%} + regime_loose",
            k, d200_thresh=thresh, target_vol=None, regime_name="200ma_loose")
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. d_sma200 + vol targeting (best combo)
# ---------------------------------------------------------------------------
print("\n--- d_sma200 > 0% + vol targeting (15%) ---")
for k in [20, 30, 40, 50, 60]:
    r = run_with_filter(
        f"lgbm K={k:3d} + d200>0% + voltarget(15%) + regime_loose",
        k, d200_thresh=0.0, target_vol=0.15, regime_name="200ma_loose")
    if r: RESULTS.append(r)

print("\n--- d_sma200 > 0% + vol targeting (18%) ---")
for k in [20, 30, 40, 50, 60]:
    r = run_with_filter(
        f"lgbm K={k:3d} + d200>0% + voltarget(18%) + regime_loose",
        k, d200_thresh=0.0, target_vol=0.18, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Golden cross filter (SMA50 > SMA200)
# ---------------------------------------------------------------------------
print("\n--- Golden cross (sma50_above_200) filter ---")
for k in [20, 30, 40, 50]:
    r = run_with_filter(
        f"lgbm K={k:3d} + golden_cross + voltarget(18%) + regime_loose",
        k, d200_thresh=None, sma50_cross=True, target_vol=0.18, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. RS filter: only stocks outperforming SPY over 6m
# ---------------------------------------------------------------------------
print("\n--- RS filter: rs_6m_spy > 0 ---")
for k in [20, 30, 40, 50]:
    r = run_with_filter(
        f"lgbm K={k:3d} + rs6m>0 + voltarget(18%) + regime_loose",
        k, d200_thresh=None, rs_thresh=0.0, target_vol=0.18, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Combined: d_sma200>0 + golden_cross + vol_target
# ---------------------------------------------------------------------------
print("\n--- Combined: d_sma200>0% + golden_cross + voltarget ---")
for vt in [0.15, 0.18]:
    for k in [20, 30, 40, 50]:
        r = run_with_filter(
            f"lgbm K={k:3d} + d200>0+golden + voltarget({vt:.0%}) + regime_loose",
            k, d200_thresh=0.0, sma50_cross=True, target_vol=vt, regime_name="200ma_loose")
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 6. Strict 200MA regime (no vol target, but strict stock filter)
# ---------------------------------------------------------------------------
print("\n--- d_sma200>0% strict regime (200ma) ---")
for k in [20, 30, 40, 50]:
    r = run_with_filter(
        f"lgbm K={k:3d} + d200>0% + regime_200ma (strict)",
        k, d200_thresh=0.0, target_vol=None, regime_name="200ma")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "avg_scale",
              "top_k"]].to_string(index=False))

out = Path(__file__).parent / "exp_007_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")

passed_both = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]
if len(passed_both) > 0:
    print(f"\n{'='*40}")
    print(f"*** {len(passed_both)} CONFIGS PASS BOTH GATES ***")
    print(passed_both[["name", "cagr", "sharpe", "max_dd", "ann_vol"]].to_string(index=False))
else:
    best_sharpe = df_res.iloc[0]["sharpe"]
    best_with_cagr = df_res[df_res["cagr"] >= 0.50]
    print(f"\nBest Sharpe: {best_sharpe:.2f} (target 2.0)")
    if len(best_with_cagr):
        print(f"Best Sharpe with CAGR≥50%: {best_with_cagr.iloc[0]['name']} → {best_with_cagr.iloc[0]['sharpe']:.2f}")

print(f"\nTotal configs: {len(df_res)}")
print(f"Running total hypotheses: 164 (prior) + {len(df_res)} = {164 + len(df_res)}")
