"""
Experiment 009: Score Blend Optimization

Best from exp_008: lgbm×0.7+sh12×0.3 K=40 → CAGR=58.3%, Sharpe=1.80
mean_m/std_m ratio = 0.519; need 0.577 for Sharpe=2.0.

Test: fine-tune blend weights and add additional quality features.
Also try: idio_mom_12_1 (idiosyncratic momentum), mom_per_unit_vol_12, trend_health_5y.
Plus: combine best blend with all best filters from exp_006/007/008.
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
from features.signals import composite_v1, set_date_context

print("=" * 90)
print("EXPERIMENT 009: SCORE BLEND OPTIMIZATION")
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

FEAT_BLENDS = [
    # (lgbm_w, sh12_w, sh5y_w, idio_w, mpuv_w)
    (0.70, 0.30, 0.00, 0.00, 0.00),  # best from exp_008
    (0.65, 0.35, 0.00, 0.00, 0.00),
    (0.75, 0.25, 0.00, 0.00, 0.00),
    (0.80, 0.20, 0.00, 0.00, 0.00),
    (0.85, 0.15, 0.00, 0.00, 0.00),
    (0.60, 0.40, 0.00, 0.00, 0.00),
    # Three-way: lgbm + sh12 + sh5y
    (0.60, 0.25, 0.15, 0.00, 0.00),
    (0.65, 0.20, 0.15, 0.00, 0.00),
    (0.70, 0.20, 0.10, 0.00, 0.00),
    # Adding idio_mom_12_1
    (0.60, 0.25, 0.00, 0.15, 0.00),
    (0.65, 0.20, 0.00, 0.15, 0.00),
    (0.70, 0.15, 0.00, 0.15, 0.00),
    # Adding mom_per_unit_vol_12 (Sharpe of momentum)
    (0.65, 0.20, 0.00, 0.00, 0.15),
    (0.70, 0.15, 0.00, 0.00, 0.15),
    # Four-way
    (0.60, 0.20, 0.10, 0.10, 0.00),
]


def run_blend(
    name: str,
    top_k: int,
    weights: tuple,  # (lgbm_w, sh12_w, sh5y_w, idio_w, mpuv_w)
    d200_thresh: float = 0.0,
    rs_thresh: float = 0.0,
    target_vol: float = 0.18,
    regime_name: str = "200ma_loose",
    start: str = OOS_START,
    end: str = OOS_END,
) -> dict | None:
    t0 = time.time()
    regime_fn = make_regime_fn(regime_name)
    feat_dates = get_feat_dates()
    dates = [d for d in feat_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < 6:
        return None

    def znorm(s):
        mu, si = s.mean(), s.std()
        return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)

    lgbm_w, sh12_w, sh5y_w, idio_w, mpuv_w = weights
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

        # Vol targeting
        stats = get_spy_stats_at(date)
        spy_vol = stats.get("vol_21d", target_vol) if stats else target_vol
        scale = min(target_vol / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0

        set_date_context(date)

        # LGBM base score
        D = date
        lgbm_scores = (score_fn_cache[D](feats) if D in score_fn_cache
                       else composite_v1(feats)).dropna()
        lgbm_scores = lgbm_scores[~lgbm_scores.index.isin(EXCLUDE)]
        if lgbm_scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        # Build blended score over common index
        idx = lgbm_scores.index
        blended = znorm(lgbm_scores) * lgbm_w

        if sh12_w > 0 and "sharpe_12m" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "sharpe_12m"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * sh12_w

        if sh5y_w > 0 and "sharpe_5y" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "sharpe_5y"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * sh5y_w

        if idio_w > 0 and "idio_mom_12_1" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "idio_mom_12_1"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * idio_w

        if mpuv_w > 0 and "mom_per_unit_vol_12" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "mom_per_unit_vol_12"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * mpuv_w

        # Filters
        if d200_thresh is not None and "d_sma200" in feats.columns:
            d_sma = feats.loc[feats.index.isin(idx), "d_sma200"].reindex(idx)
            blended = blended[d_sma > d200_thresh]

        if rs_thresh is not None and "rs_6m_spy" in feats.columns:
            rs = feats.loc[feats.index.isin(blended.index), "rs_6m_spy"].reindex(blended.index)
            blended = blended[rs > rs_thresh]

        blended = blended.dropna()
        if blended.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        top = blended.sort_values(ascending=False).head(top_k)
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

        vols = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols.append(0.20)
        inv_v = 1.0 / np.array(vols)
        weights_arr = inv_v / inv_v.sum()

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        raw_ret = float((weights_arr * rets).sum())
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
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4),
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:75s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 1. Blend weight sweep at K=40 (best from exp_008)
# ---------------------------------------------------------------------------
print("\n--- Blend weight sweep: K=40, voltarget=18%, regime_loose, no filters ---")
for w in FEAT_BLENDS:
    lw, s12w, s5w, iw, pw = w
    blend_str = f"L={lw:.2f}"
    if s12w > 0: blend_str += f"+S12={s12w:.2f}"
    if s5w > 0: blend_str += f"+S5y={s5w:.2f}"
    if iw > 0: blend_str += f"+Idio={iw:.2f}"
    if pw > 0: blend_str += f"+MPUV={pw:.2f}"
    r = run_blend(f"K=40 {blend_str}", 40, w,
                  d200_thresh=None, rs_thresh=None)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Best blend + d200>0 filter, sweep K
# ---------------------------------------------------------------------------
print("\n--- Best blend (0.70/0.30) + d200>0 + rs>0, sweep K ---")
best_blend = (0.70, 0.30, 0.00, 0.00, 0.00)
for k in [20, 30, 40, 50, 60]:
    r = run_blend(f"L=0.70+S12=0.30 K={k:3d} + d200>0+rs>0 + vt18% + loose",
                  k, best_blend, d200_thresh=0.0, rs_thresh=0.0)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Sweep K for top-3 blends (no filters)
# ---------------------------------------------------------------------------
print("\n--- Top blends sweep K (no filters) ---")
for w in [(0.70, 0.30, 0.00, 0.00, 0.00),
           (0.65, 0.35, 0.00, 0.00, 0.00),
           (0.70, 0.20, 0.10, 0.00, 0.00)]:
    lw, s12w, s5w, iw, pw = w
    for k in [30, 50, 60]:
        blend_str = f"L={lw:.2f}+S12={s12w:.2f}"
        if s5w > 0: blend_str += f"+S5y={s5w:.2f}"
        r = run_blend(f"K={k:3d} {blend_str} + vt18% + loose",
                      k, w, d200_thresh=None, rs_thresh=None)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. Best blend + different vol targets
# ---------------------------------------------------------------------------
print("\n--- Best blend (0.70/0.30) + vol target sweep at K=40 ---")
best_blend = (0.70, 0.30, 0.00, 0.00, 0.00)
for vt in [0.12, 0.15, 0.18, 0.20]:
    r = run_blend(f"L=0.70+S12=0.30 K= 40 + vt={vt:.0%} + loose",
                  40, best_blend, d200_thresh=None, rs_thresh=None,
                  target_vol=vt)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Best combo: all filters + best blend
# ---------------------------------------------------------------------------
print("\n--- Kitchen sink: best blend + d200 + rs + optimal K ---")
for k in [30, 40, 50, 60]:
    for vt in [0.15, 0.18]:
        r = run_blend(
            f"BEST: L=0.70+S12=0.30 K={k:3d} d200+rs vt={vt:.0%} loose",
            k, (0.70, 0.30, 0.00, 0.00, 0.00),
            d200_thresh=0.0, rs_thresh=0.0, target_vol=vt)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe, CAGR≥50% highlighted)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
passing = df_res[df_res["cagr"] >= 0.50]
print(f"\nConfigs with CAGR≥50% (N={len(passing)}):")
print(passing[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio",
               "top_k"]].head(15).to_string(index=False))
print(f"\nAll configs:")
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio",
              "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_009_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      mean_m={best['mean_m']:.3%} std_m={best['std_m']:.3%} ratio={best['ratio']:.4f}")

passed_both = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]
if len(passed_both) > 0:
    print(f"\n{'='*40}")
    print(f"*** {len(passed_both)} CONFIGS PASS BOTH GATES ***")
    print(passed_both[["name", "cagr", "sharpe", "max_dd", "ann_vol"]].to_string(index=False))
else:
    best_sharpe = df_res.iloc[0]["sharpe"]
    best_with_cagr = df_res[df_res["cagr"] >= 0.50]
    print(f"\nBest Sharpe: {best_sharpe:.2f}")
    if len(best_with_cagr):
        print(f"Best with CAGR≥50%: {best_with_cagr.iloc[0]['name']} → Sharpe={best_with_cagr.iloc[0]['sharpe']:.2f}")
        print(f"  ratio={best_with_cagr.iloc[0]['ratio']:.4f} (need 0.5774)")

print(f"\nTotal configs: {len(df_res)}")
print(f"Running total hypotheses: 264 (prior) + {len(df_res)} = {264 + len(df_res)}")
