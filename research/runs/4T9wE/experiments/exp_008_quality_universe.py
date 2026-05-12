"""
Experiment 008: Quality Universe + Sharpe-Weighted Scoring

Diagnosis: Sharpe ceiling at 1.76. Need mean_m/std_m ratio = 0.577 for Sharpe=2.0.
Currently: ratio ≈ 0.50-0.51.

Key levers:
1. Low-vol universe filter (vol_12m < threshold):
   - Reduces portfolio vol without proportionally reducing returns
   - Low-vol anomaly: low-vol stocks have better risk-adjusted returns
2. Sharpe_12m as primary/blended score:
   - Selects stocks with best risk-adjusted momentum
   - More stable: sharpe_12m already balances return and risk
3. Score × inv_vol weighted sizing:
   - High-conviction picks get proportionally larger position
   - Reduces effective K (concentrates on best picks)

Prior best: lgbm K=50 + rs6m>0 + voltarget(18%) + regime_loose → CAGR=55.0%, Sharpe=1.76
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
print("EXPERIMENT 008: QUALITY UNIVERSE + SHARPE-WEIGHTED SCORING")
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


def run_quality(
    name: str,
    top_k: int,
    vol_universe_cap: float = None,    # vol_12m < cap (e.g. 0.30)
    sharpe12_blend: float = 0.0,       # weight on z(sharpe_12m) in score
    lgbm_blend: float = 1.0,          # weight on z(lgbm) in score
    score_weighted: bool = False,      # use score×inv_vol weighting (vs pure inv_vol)
    d200_thresh: float = None,         # stock d_sma200 filter
    target_vol: float = None,         # vol targeting
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
        if target_vol is not None:
            stats = get_spy_stats_at(date)
            spy_vol = stats.get("vol_21d", target_vol) if stats else target_vol
            scale = min(target_vol / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0
        else:
            scale = 1.0

        set_date_context(date)

        # Universe filter: low-vol stocks only
        if vol_universe_cap is not None and "vol_12m" in feats.columns:
            vol12 = feats["vol_12m"]
            feats_sub = feats[vol12 < vol_universe_cap]
        else:
            feats_sub = feats

        if feats_sub.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        # Stock-level d_sma200 filter
        if d200_thresh is not None and "d_sma200" in feats_sub.columns:
            feats_sub = feats_sub[feats_sub["d_sma200"] > d200_thresh]

        if feats_sub.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        # Build score
        D = date
        if D in score_fn_cache:
            lgbm_scores = score_fn_cache[D](feats_sub).dropna()
        else:
            lgbm_scores = composite_v1(feats_sub).dropna()

        lgbm_scores = lgbm_scores[~lgbm_scores.index.isin(EXCLUDE)]

        if sharpe12_blend > 0.0 and "sharpe_12m" in feats_sub.columns:
            sharpe12 = feats_sub.loc[feats_sub.index.isin(lgbm_scores.index), "sharpe_12m"].dropna()
            common_idx = lgbm_scores.index.intersection(sharpe12.index)
            if len(common_idx) > 0:
                z_lgbm = znorm(lgbm_scores.loc[common_idx])
                z_sh12 = znorm(sharpe12.loc[common_idx])
                final_scores = z_lgbm * lgbm_blend + z_sh12 * sharpe12_blend
            else:
                final_scores = lgbm_scores
        else:
            final_scores = lgbm_scores

        if final_scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        top = final_scores.sort_values(ascending=False).head(top_k)
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

        # Weights
        vols = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols.append(0.20)
        inv_v = 1.0 / np.array(vols)

        if score_weighted:
            sc = np.array([max(float(final_scores.get(t, 0.0)) - float(final_scores.iloc[min(top_k, len(final_scores))-1]), 1e-6)
                           for t in common])
            raw_w = sc * inv_v
        else:
            raw_w = inv_v

        if raw_w.sum() <= 0:
            raw_w = np.ones(len(common))
        weights = raw_w / raw_w.sum()

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
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} Sc={avg_scale:.2f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# Reference: best from prior experiments
# ---------------------------------------------------------------------------
print("\n--- Reference: best from exp_006/007 ---")
for k, vt in [(40, 0.18), (50, 0.18)]:
    r = run_quality(f"lgbm K={k:3d} inv_vol + voltarget({vt:.0%}) + regime_loose [ref]",
                    k, target_vol=vt, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 1. Low-vol universe: filter by vol_12m
# ---------------------------------------------------------------------------
print("\n--- Low-vol universe filter (vol_12m < threshold) ---")
for vol_cap in [0.20, 0.25, 0.30, 0.35]:
    for k in [20, 30, 40, 50]:
        r = run_quality(
            f"lgbm K={k:3d} + vol_univ<{vol_cap:.0%} + voltarget(18%) + regime_loose",
            k, vol_universe_cap=vol_cap, target_vol=0.18, regime_name="200ma_loose")
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Sharpe_12m blend: combine LGBM with Sharpe_12m
# ---------------------------------------------------------------------------
print("\n--- Sharpe_12m blend (LGBM × w + Sharpe12 × (1-w)) ---")
for sh_wt in [0.3, 0.5, 0.7]:
    for k in [30, 40, 50]:
        r = run_quality(
            f"lgbm×{1-sh_wt:.1f}+sh12×{sh_wt:.1f} K={k:3d} + voltarget(18%) + regime_loose",
            k, sharpe12_blend=sh_wt, lgbm_blend=(1 - sh_wt),
            target_vol=0.18, regime_name="200ma_loose")
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Score-weighted sizing (alpha × inv_vol)
# ---------------------------------------------------------------------------
print("\n--- Score-weighted sizing (alpha conviction × inv_vol) ---")
for k in [30, 40, 50, 60]:
    r = run_quality(
        f"lgbm K={k:3d} score×inv_vol + voltarget(18%) + regime_loose",
        k, score_weighted=True, target_vol=0.18, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. Low-vol universe + sharpe blend + score weighting
# ---------------------------------------------------------------------------
print("\n--- Combo: low-vol universe + sharpe_12m blend + score weighting ---")
for vol_cap in [0.25, 0.30]:
    for sh_wt in [0.3, 0.5]:
        for k in [30, 40, 50]:
            r = run_quality(
                f"lgbm+sh12 K={k:3d} + vol<{vol_cap:.0%} + sw + voltarget(18%)",
                k, vol_universe_cap=vol_cap, sharpe12_blend=sh_wt,
                lgbm_blend=(1 - sh_wt), score_weighted=True,
                target_vol=0.18, regime_name="200ma_loose")
            if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Pure sharpe_12m score (no LGBM)
# ---------------------------------------------------------------------------
print("\n--- Pure sharpe_12m score (no LGBM) ---")
for k in [20, 30, 40, 50]:
    r = run_quality(
        f"pure_sharpe12m K={k:3d} + voltarget(18%) + regime_loose",
        k, sharpe12_blend=1.0, lgbm_blend=0.0,
        target_vol=0.18, regime_name="200ma_loose")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 6. Low-vol universe + d_sma200>0 + best blend
# ---------------------------------------------------------------------------
print("\n--- Low-vol universe + d_sma200>0% filter + voltarget ---")
for vol_cap in [0.25, 0.30]:
    for k in [30, 40, 50]:
        r = run_quality(
            f"lgbm K={k:3d} + vol<{vol_cap:.0%} + d200>0% + voltarget(18%)",
            k, vol_universe_cap=vol_cap, d200_thresh=0.0,
            target_vol=0.18, regime_name="200ma_loose")
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 7. Strict regime (200ma) with best quality filter
# ---------------------------------------------------------------------------
print("\n--- Strict regime + low-vol + sharpe blend ---")
for vol_cap, sh_wt, k in [(0.25, 0.5, 40), (0.30, 0.5, 50), (0.25, 0.3, 50)]:
    r = run_quality(
        f"lgbm+sh12×{sh_wt} K={k:3d} + vol<{vol_cap:.0%} + regime_200ma",
        k, vol_universe_cap=vol_cap, sharpe12_blend=sh_wt,
        lgbm_blend=(1 - sh_wt), target_vol=0.18, regime_name="200ma")
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "mean_m", "std_m",
              "top_k"]].to_string(index=False))

out = Path(__file__).parent / "exp_008_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      mean_m={best['mean_m']:.3%} std_m={best['std_m']:.3%} ratio={best['mean_m']/best['std_m']:.3f}")
print(f"      (need ratio≥0.577 for Sharpe=2.0; current best in exp_007=0.508)")

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

print(f"\nTotal configs: {len(df_res)}")
print(f"Running total hypotheses: 208 (prior) + {len(df_res)} = {208 + len(df_res)}")
