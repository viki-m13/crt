"""
Experiment 013: Portfolio-Level Risk Control

MOTIVATION:
  Current best: K=30, LGBM×0.70+sh12×0.20+sh5y×0.10, inv_vol, vt18%, loose
               CAGR=63.1%, Sharpe=1.82, ratio=0.527 (target 0.577)

  SPY-vol targeting (vt18%) scales by SPY vol, but that may diverge from
  portfolio's realized vol. Key insight:
    - Portfolio ann_vol = 29.5% >> SPY target 18%
    - SPY vol ~18% in normal markets → scale=1.0 (no scaling)
    - Portfolio vol > SPY vol in all regimes → direct portfolio vol targeting
      should reduce std_m more than SPY-vol targeting.

  Options tested:
  1. Portfolio realized vol targeting: scale by min(target/port_vol_3m, 1.0)
     Applied AT NEXT MONTH using last 3m realized portfolio vol.
  2. Trailing drawdown protection: if 2m portfolio loss > threshold → reduce scale
  3. Combined: portfolio vol target + drawdown protection
  4. Replace SPY vol with portfolio vol (no SPY reference at all)
  5. Adaptive target: tighter target after high-vol periods

  Math: std_m=8.51% → ann_vol=29.5%. If we can control ann_vol to ≤ 26.9%:
    Sharpe = 4.48% × √12 / 26.9% = 4.48% × 3.464 / 26.9% = 2.00 ✓

  Key distinction: This is NOT look-ahead. At decision time t (month-end),
  we use portfolio returns from t-3 to t-1 (last 3 months), all known.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest.engine import (
    make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import set_date_context

print("=" * 90)
print("EXPERIMENT 013: PORTFOLIO-LEVEL RISK CONTROL")
print("=" * 90)
print("\nMotivation: Portfolio ann_vol=29.5% >> SPY-vol-target of 18%.")
print("Direct portfolio vol control should push ann_vol → 26.9% → Sharpe = 2.0")

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

# Best blend from exp_009
BLEND = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


def score_3way(feat_df, date):
    lgbm_fn = lgbm_cache.get(date)
    lgbm = (lgbm_fn(feat_df) if lgbm_fn else pd.Series(dtype=float)).dropna()
    lgbm = lgbm[~lgbm.index.isin(EXCLUDE)]
    idx = lgbm.index

    blended = znorm(lgbm) * BLEND["lgbm"]
    for feat, wt in [("sharpe_12m", BLEND["sharpe_12m"]), ("sharpe_5y", BLEND["sharpe_5y"])]:
        if feat in feat_df.columns:
            s = feat_df.loc[feat_df.index.isin(idx), feat].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * wt
    return blended.dropna()


def run_with_port_risk(
    name,
    top_k,
    spy_vol_target=None,       # SPY-vol-based scale (None = disabled)
    port_vol_target=None,      # Portfolio-vol-based scale (None = disabled)
    port_vol_lookback=3,       # months of history for portfolio vol estimate
    dd_thresh=None,            # 2m drawdown threshold to reduce scale (e.g. -0.10)
    dd_scale_factor=0.5,       # scale multiplier when drawdown threshold hit
    regime="200ma_loose",
    start=OOS_START,
    end=OOS_END,
):
    t0 = time.time()
    regime_fn = make_regime_fn(regime)
    feat_dates = get_feat_dates()
    dates_range = [d for d in feat_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates_range) < 6:
        return None

    records = []
    portfolio_rets = []  # track realized portfolio returns for vol estimation

    for i, date in enumerate(dates_range[:-1]):
        next_date = dates_range[i + 1]
        feats = load_features(date)

        # Compute base scale from SPY vol (same as exp_006-012)
        spy_scale = 1.0
        if spy_vol_target is not None:
            stats = get_spy_stats_at(date)
            spy_vol = stats.get("vol_21d", spy_vol_target) if stats else spy_vol_target
            spy_scale = min(spy_vol_target / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0

        # Portfolio realized vol scale (using last N months of portfolio returns)
        port_scale = 1.0
        if port_vol_target is not None and len(portfolio_rets) >= port_vol_lookback:
            recent = portfolio_rets[-port_vol_lookback:]
            port_vol_realized = float(np.std(recent) * np.sqrt(12))
            if port_vol_realized > 1e-6:
                port_scale = min(port_vol_target / port_vol_realized, 1.0)

        # Trailing drawdown protection
        dd_scale = 1.0
        if dd_thresh is not None and len(portfolio_rets) >= 2:
            dd_2m = portfolio_rets[-1] + portfolio_rets[-2]
            if dd_2m < dd_thresh:
                dd_scale = dd_scale_factor

        total_scale = spy_scale * port_scale * dd_scale

        if feats.empty or not regime_fn(date, feats):
            portfolio_rets.append(0.0)
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "spy_scale": spy_scale, "port_scale": port_scale,
                            "dd_scale": dd_scale})
            continue

        set_date_context(date)
        scores = score_3way(feats, date)
        if scores.empty:
            portfolio_rets.append(0.0)
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "spy_scale": spy_scale, "port_scale": port_scale,
                            "dd_scale": dd_scale})
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
            portfolio_rets.append(0.0)
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "spy_scale": spy_scale, "port_scale": port_scale,
                            "dd_scale": dd_scale})
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
        port_ret = total_scale * raw_ret - 2 * COST * total_scale
        portfolio_rets.append(port_ret)
        records.append({"date": date, "ret_m": port_ret, "n_picks": len(common),
                        "spy_scale": spy_scale, "port_scale": port_scale,
                        "dd_scale": dd_scale})

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
        "ann_vol": round(float(m["ann_vol"]), 4),
        "n_months": int(m["n_months"]), "cash_months": int((df["n_picks"] == 0).sum()),
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_spy_scale": round(df["spy_scale"].mean(), 3),
        "avg_port_scale": round(df["port_scale"].mean(), 3),
        "avg_dd_scale": round(df["dd_scale"].mean(), 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:75s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} "
          f"ratio={res['ratio']:.3f} ps={res['avg_port_scale']:.2f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference: current best from exp_009 (spy_vol only, no port vol)
# ---------------------------------------------------------------------------
print("\n--- Reference: K=30, 3-way blend, inv_vol, vt18%, regime_loose ---")
r = run_with_port_risk("REF: spy_vt18% only K=30", 30, spy_vol_target=0.18)
if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 1. Portfolio vol targeting only (no SPY vol reference)
# ---------------------------------------------------------------------------
print("\n--- Portfolio realized vol targeting (3m lookback, no SPY vol) ---")
for pvt in [0.15, 0.18, 0.20, 0.22]:
    for k in [25, 30, 40]:
        r = run_with_port_risk(
            f"port_vt={pvt:.0%} K={k:3d} 3m_lookback",
            k, spy_vol_target=None, port_vol_target=pvt, port_vol_lookback=3
        )
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Portfolio vol targeting + SPY vol (dual scaling)
# ---------------------------------------------------------------------------
print("\n--- Dual: SPY vt18% AND portfolio vol targeting ---")
for pvt in [0.20, 0.22, 0.25]:
    for k in [25, 30, 40]:
        r = run_with_port_risk(
            f"spy18%+port_vt={pvt:.0%} K={k:3d}",
            k, spy_vol_target=0.18, port_vol_target=pvt, port_vol_lookback=3
        )
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Trailing drawdown protection only
# ---------------------------------------------------------------------------
print("\n--- Trailing drawdown protection (2m return < threshold → reduce scale) ---")
for dd_thresh, dd_sf in [(-0.05, 0.5), (-0.08, 0.5), (-0.10, 0.5),
                          (-0.05, 0.0), (-0.08, 0.0)]:
    r = run_with_port_risk(
        f"spy18%+dd_thresh={dd_thresh:.0%}_sf={dd_sf:.1f} K=30",
        30, spy_vol_target=0.18, dd_thresh=dd_thresh, dd_scale_factor=dd_sf
    )
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. Combined: SPY vol + portfolio vol + drawdown protection
# ---------------------------------------------------------------------------
print("\n--- Combined: SPY vt18% + portfolio vol + drawdown protection ---")
for pvt in [0.20, 0.22]:
    for dd_thresh, dd_sf in [(-0.08, 0.5), (-0.10, 0.5)]:
        for k in [25, 30, 40]:
            r = run_with_port_risk(
                f"spy18%+pvt{pvt:.0%}+dd{dd_thresh:.0%}_sf{dd_sf:.1f} K={k:3d}",
                k, spy_vol_target=0.18, port_vol_target=pvt, port_vol_lookback=3,
                dd_thresh=dd_thresh, dd_scale_factor=dd_sf
            )
            if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Portfolio vol targeting with different lookbacks
# ---------------------------------------------------------------------------
print("\n--- Portfolio vol target=20%, different lookbacks ---")
for lookback in [2, 3, 6]:
    for k in [25, 30]:
        r = run_with_port_risk(
            f"pvt20%+lb={lookback}m K={k:3d}",
            k, spy_vol_target=0.18, port_vol_target=0.20, port_vol_lookback=lookback
        )
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
              "avg_port_scale", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_013_results.csv"
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
print(f"Running total hypotheses: 390 (from exp_001-012) + {n_configs} = {390 + n_configs}")
