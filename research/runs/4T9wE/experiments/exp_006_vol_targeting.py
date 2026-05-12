"""
Experiment 006: Volatility Targeting

Hypothesis: Scaling position size inversely with realized SPY vol (capped at 1.0)
reduces portfolio vol during market stress more than it reduces expected returns.
Negative vol-return correlation in equities (Moreira & Muir 2017) means scaling down
in high-vol environments improves Sharpe.

Best prior result: lgbm K=50 inv_vol + regime_200ma → CAGR=56.6%, Sharpe=1.68
Target: CAGR ≥ 50% AND Sharpe ≥ 2.0

Test matrix:
  - K ∈ [30, 40, 50, 60, 80]
  - target_ann_vol ∈ [0.10, 0.12, 0.15, 0.18, 0.20]
  - Also: portfolio realized vol targeting (vs SPY proxy)
  - Also: sector momentum pre-filter (top-500 tickers by sector rs_3m_spy)
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
print("EXPERIMENT 006: VOLATILITY TARGETING")
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


def lgbm_fn(f):
    from features.signals import _CURRENT_DATE as D
    return score_fn_cache[D](f) if D in score_fn_cache else composite_v1(f)


OOS_START = "2007-01-31"
OOS_END = "2021-12-31"
COST = 5.0 / 10_000.0


def run_voltarget(
    name: str,
    score_fn,
    top_k: int,
    target_ann_vol: float,
    regime_name: str = "200ma",
    start: str = OOS_START,
    end: str = OOS_END,
) -> dict | None:
    """Backtest with volatility targeting. Scale = min(target_vol / spy_vol, 1.0)."""
    t0 = time.time()
    regime_fn = make_regime_fn(regime_name)
    feat_dates = get_feat_dates()
    dates = [d for d in feat_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < 6:
        print(f"  {name}: insufficient dates")
        return None

    records = []
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]

        feats = load_features(date)
        if feats.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
            continue

        # Regime gate
        if not regime_fn(date, feats):
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
            continue

        # Vol scaling factor
        stats = get_spy_stats_at(date)
        spy_vol = stats.get("vol_21d", target_ann_vol) if stats else target_ann_vol
        scale = min(target_ann_vol / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0

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

        # Apply vol scaling
        port_ret = scale * raw_ret - 2 * COST * scale
        records.append({"date": date, "ret_m": port_ret, "n_picks": len(common), "scale": scale})

    if not records:
        print(f"  {name}: NO RESULTS")
        return None

    df = pd.DataFrame(records).set_index("date")
    m = compute_metrics(df["ret_m"])
    if not m:
        print(f"  {name}: NO METRICS")
        return None

    elapsed = time.time() - t0
    avg_scale = df["scale"].mean()
    cash_m = int((df["n_picks"] == 0).sum())
    res = {
        "name": name, "top_k": top_k, "target_vol": target_ann_vol,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4),
        "n_months": int(m["n_months"]), "cash_months": cash_m,
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(avg_scale, 3),
    }
    gc = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:70s} CAGR={res['cagr']:.1%}{gc} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} AvgScale={avg_scale:.2f} Cash={cash_m}m {elapsed:.0f}s")
    return res


def run_standard(name, score_fn, top_k, start=OOS_START, end=OOS_END):
    """Standard run_backtest wrapper for reference comparisons."""
    t0 = time.time()
    regime_fn_obj = make_regime_fn("200ma")
    df, m = run_backtest(score_fn=score_fn, start=start, end=end,
                         top_k=top_k, weighting="inv_vol",
                         cost_bps=5.0, regime_fn=regime_fn_obj)
    elapsed = time.time() - t0
    if not m:
        print(f"  {name}: NO RESULTS")
        return None
    cash_m = int((df["n_picks"] == 0).sum()) if "n_picks" in df.columns else 0
    res = {
        "name": name, "top_k": top_k, "target_vol": 999.0,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4),
        "n_months": int(m["n_months"]), "cash_months": cash_m,
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": 1.0,
    }
    gc = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:70s} CAGR={res['cagr']:.1%}{gc} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} AvgScale=1.00 Cash={cash_m}m {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 1. Reference: no vol targeting
# ---------------------------------------------------------------------------
print("\n--- Reference (no vol targeting, for comparison) ---")
for k in [40, 50, 60]:
    r = run_standard(f"lgbm K={k:3d} inv_vol + regime [NO voltarget]", lgbm_fn, k)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Vol targeting: sweep target_ann_vol
# ---------------------------------------------------------------------------
print("\n--- Vol targeting: K=50, sweep target_ann_vol ---")
for vt in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    r = run_voltarget(
        f"lgbm K= 50 inv_vol + voltarget({vt:.0%}) + regime",
        lgbm_fn, 50, vt)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Vol targeting: sweep K with best target vols
# ---------------------------------------------------------------------------
print("\n--- Vol targeting: sweep K with target=15% ---")
for k in [30, 40, 50, 60, 80, 100]:
    r = run_voltarget(
        f"lgbm K={k:3d} inv_vol + voltarget(15%) + regime",
        lgbm_fn, k, 0.15)
    if r: RESULTS.append(r)

print("\n--- Vol targeting: sweep K with target=18% ---")
for k in [30, 40, 50, 60, 80]:
    r = run_voltarget(
        f"lgbm K={k:3d} inv_vol + voltarget(18%) + regime",
        lgbm_fn, k, 0.18)
    if r: RESULTS.append(r)

print("\n--- Vol targeting: sweep K with target=12% ---")
for k in [40, 50, 60, 80]:
    r = run_voltarget(
        f"lgbm K={k:3d} inv_vol + voltarget(12%) + regime",
        lgbm_fn, k, 0.12)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. Vol targeting with looser regime (to allow more scale in mild downturns)
# ---------------------------------------------------------------------------
print("\n--- Vol targeting + loose regime (200ma_loose) ---")
for vt in [0.12, 0.15, 0.18]:
    for k in [40, 50]:
        regime_fn_obj = make_regime_fn("200ma_loose")
        feat_dates = get_feat_dates()
        dates = [d for d in feat_dates if pd.Timestamp(OOS_START) <= d <= pd.Timestamp(OOS_END)]
        t0 = time.time()
        records = []
        for i, date in enumerate(dates[:-1]):
            next_date = dates[i + 1]
            feats = load_features(date)
            if feats.empty:
                records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
                continue
            if not regime_fn_obj(date, feats):
                records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": 0.0})
                continue
            stats = get_spy_stats_at(date)
            spy_vol = stats.get("vol_21d", vt) if stats else vt
            scale = min(vt / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0
            set_date_context(date)
            scores = lgbm_fn(feats).dropna()
            scores = scores[~scores.index.isin(EXCLUDE)]
            if scores.empty:
                records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
                continue
            top = scores.sort_values(ascending=False).head(k)
            tickers = top.index.tolist()
            d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index)-1)
            if d0_idx > 0 and monthly_px.index[d0_idx] > date: d0_idx -= 1
            d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index)-1)
            if d1_idx > 0 and monthly_px.index[d1_idx] > next_date: d1_idx -= 1
            p0 = monthly_px.iloc[d0_idx]; p1 = monthly_px.iloc[d1_idx]
            common = [t for t in tickers if t in monthly_px.columns
                      and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                      and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
            if not common:
                records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
                continue
            vols_lst = [max(float(feats.loc[t, "vol_12m"]) if t in feats.index and "vol_12m" in feats.columns
                            and np.isfinite(feats.loc[t, "vol_12m"]) else 0.20, 0.05) for t in common]
            inv_v = 1.0 / np.array(vols_lst)
            weights = inv_v / inv_v.sum()
            rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
            raw_ret = float((weights * rets).sum())
            records.append({"date": date, "ret_m": scale * raw_ret - 2*COST*scale,
                             "n_picks": len(common), "scale": scale})

        df = pd.DataFrame(records).set_index("date")
        m = compute_metrics(df["ret_m"])
        elapsed = time.time() - t0
        if m:
            cash_m = int((df["n_picks"] == 0).sum())
            avg_scale = df["scale"].mean()
            name = f"lgbm K={k:3d} inv_vol + voltarget({vt:.0%}) + regime_loose"
            res = {
                "name": name, "top_k": k, "target_vol": vt,
                "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
                "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
                "ann_vol": round(float(m["ann_vol"]), 4),
                "n_months": int(m["n_months"]), "cash_months": cash_m,
                "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
                "avg_scale": round(avg_scale, 3),
            }
            gc = "✓" if res["cagr"] >= 0.50 else "✗"
            gs = "✓" if res["sharpe"] >= 2.0 else "✗"
            print(f"  {name:70s} CAGR={res['cagr']:.1%}{gc} Sharpe={res['sharpe']:.2f}{gs} "
                  f"MaxDD={res['max_dd']:.1%} AvgScale={avg_scale:.2f} Cash={cash_m}m {elapsed:.0f}s")
            RESULTS.append(res)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "avg_scale",
              "top_k", "target_vol"]].to_string(index=False))

out = Path(__file__).parent / "exp_006_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")

passed_both = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]
if len(passed_both) > 0:
    print(f"\n{'='*40}")
    print(f"*** {len(passed_both)} CONFIGS PASS BOTH GATES (CAGR≥50% AND Sharpe≥2.0) ***")
    print(passed_both[["name", "cagr", "sharpe", "max_dd", "ann_vol",
                         "avg_scale"]].to_string(index=False))
else:
    best_sharpe = df_res.iloc[0]["sharpe"]
    best_cagr_row = df_res[df_res["cagr"] >= 0.50].iloc[0] if len(df_res[df_res["cagr"] >= 0.50]) > 0 else None
    print(f"\nBest Sharpe achieved: {best_sharpe:.2f} (target: 2.0)")
    if best_cagr_row is not None:
        print(f"Best config with CAGR≥50%: {best_cagr_row['name']} → Sharpe={best_cagr_row['sharpe']:.2f}")
    print(f"\nTotal configs run: {len(df_res)}")
    print(f"Running total hypotheses: 134 (prior) + {len(df_res)} = {134 + len(df_res)}")
