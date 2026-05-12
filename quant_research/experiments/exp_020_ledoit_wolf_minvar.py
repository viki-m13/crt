"""
Experiment 020: Ledoit-Wolf Shrinkage Minimum-Variance Portfolio

MOTIVATION:
dead_ends.md entry #2 (Min-Variance weighting) explicitly states:
"Dead end UNLESS: Robust covariance estimator (Ledoit-Wolf shrinkage, factor model)"

The standard sample covariance used in exp_003 is noisy (high estimation error):
- With K=30 stocks and T=252 daily obs, condition number >> 1
- Small eigenvalues are underestimated → weights blow up toward min-var solution
- Result: extreme weights that hurt OOS performance

Ledoit-Wolf (2004) provides an optimal shrinkage estimator:
  Σ_LW = (1-α) × Σ_sample + α × Σ_shrinkage
  where α is analytically optimal and Σ_shrinkage = μ_I × I (scaled identity)
  or Σ_shrinkage = diag(Σ_sample) (constant-correlation shrinkage)

Sklearn provides: sklearn.covariance.LedoitWolf and OAS (Oracle Approximating Shrinkage)

MINIMUM VARIANCE WITH LW:
  min w'Σ_LW w  subject to: Σw=1, w_i ≥ 0
  → Reduces portfolio vol by exploiting true low-correlation pairs

HYPOTHESIS:
With properly estimated covariance, min-var should:
1. Reduce portfolio vol from 29.5% toward the true minimum (theory ≈ 20-25%)
2. Keep mean_m near 4.5% (selection unchanged, only weights change)
3. Improve Sharpe from 1.841 toward 2.0

RISK:
The fundamental problem (ρ≈0.53 for momentum stocks) means min-var weights
will tilt strongly toward the lowest-vol stocks within the K=30 selection.
These lower-vol stocks may have lower returns, cutting mean_m proportionally.

Prior experiments:
- exp_003: standard min-var → same Sharpe as inv_vol (estimation noise cancels benefit)
- exp_012: ERC → INCREASED vol (from 29.5% to 34.5%)
- ERC failed because it's an optimization over a noisy covariance matrix

LW is the recommended fix for exp_003's failure mode.

Prior best: 4way+volasym10% K=30 → CAGR=66.5%, Sharpe=1.841, ratio=0.5315
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS
from scipy.optimize import minimize

from backtest.engine import (
    make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import set_date_context

print("=" * 90)
print("EXPERIMENT 020: LEDOIT-WOLF SHRINKAGE MIN-VAR PORTFOLIO")
print("=" * 90)
print("\nKey test: does LW-shrinkage covariance give min-var improvement that standard failed?")

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
LOOKBACK_DAYS = 252

BLEND_4WAY = {"lgbm": 0.63, "sharpe_12m": 0.18, "sharpe_5y": 0.09, "vol_asym_60": 0.10}
BLEND_3WAY = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


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


def compute_lw_minvar_weights(tickers, date, prices_df, lookback=LOOKBACK_DAYS,
                               estimator="lw", max_weight=0.15,
                               inv_vol_blend=0.0):
    """
    Compute minimum-variance weights using Ledoit-Wolf covariance estimate.

    Returns: np.array of weights or None on failure.
    """
    date_idx = prices_df.index.searchsorted(date, side='right')
    if date_idx < lookback:
        return None

    px_slice = prices_df.iloc[date_idx - lookback:date_idx][tickers].dropna(axis=1, how='all')
    valid_tickers = px_slice.columns.tolist()
    if len(valid_tickers) < 3:
        return None

    rets = px_slice.pct_change().dropna().values
    if len(rets) < 30:
        return None

    n = len(valid_tickers)

    # Estimate covariance matrix
    if estimator == "lw":
        cov_est = LedoitWolf().fit(rets)
        cov = cov_est.covariance_ * 252  # annualize
    elif estimator == "oas":
        cov_est = OAS().fit(rets)
        cov = cov_est.covariance_ * 252
    else:
        # Sample covariance
        cov = np.cov(rets.T) * 252

    # Minimum variance: min w'Cov w s.t. sum(w)=1, w >= 0, w_i <= max_weight
    def portfolio_var(w):
        return float(w @ cov @ w)

    def portfolio_var_grad(w):
        return 2 * cov @ w

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    result = minimize(
        portfolio_var, w0,
        jac=portfolio_var_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9}
    )

    if not result.success or np.any(np.isnan(result.x)):
        return None

    w_minvar = result.x
    w_minvar = np.maximum(w_minvar, 0)
    w_sum = w_minvar.sum()
    if w_sum < 1e-6:
        return None
    w_minvar /= w_sum

    # Optionally blend with inv_vol
    if inv_vol_blend > 0:
        vols = np.sqrt(np.diag(cov))
        vols = np.maximum(vols, 0.05)
        inv_v = 1.0 / vols
        w_ivol = inv_v / inv_v.sum()
        w_minvar = (1 - inv_vol_blend) * w_minvar + inv_vol_blend * w_ivol
        w_minvar /= w_minvar.sum()

    # Remap to full tickers list
    ticker_to_w = dict(zip(valid_tickers, w_minvar))
    return ticker_to_w, valid_tickers


def run_lw(
    name,
    top_k,
    blend_weights=None,
    estimator="lw",
    max_weight=0.15,
    inv_vol_blend=0.0,
    regime="200ma_loose",
    target_vol=0.18,
    start=OOS_START,
    end=OOS_END,
):
    if blend_weights is None:
        blend_weights = BLEND_4WAY

    t0 = time.time()
    regime_fn = make_regime_fn(regime)
    feat_dates = get_feat_dates()
    dates_range = [d for d in feat_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates_range) < 6:
        return None

    lw_ok = 0
    fallback = 0

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
        scores = compute_blend(feats, date, blend_weights).dropna()
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

        # Compute LW min-var weights
        inv_vol_weights = None
        weights = None

        lw_result = compute_lw_minvar_weights(
            common, date, prices, estimator=estimator,
            max_weight=max_weight, inv_vol_blend=inv_vol_blend
        )

        if lw_result is not None:
            ticker_to_w, valid = lw_result
            weights = np.array([ticker_to_w.get(t, 0.0) for t in common])
            weights = np.maximum(weights, 0)
            ws = weights.sum()
            if ws > 1e-6:
                weights /= ws
                lw_ok += 1
            else:
                weights = None

        if weights is None:
            # Fallback to inv_vol
            vols = []
            for t in common:
                if t in feats.index and "vol_12m" in feats.columns:
                    v = feats.loc[t, "vol_12m"]
                    vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
                else:
                    vols.append(0.20)
            inv_v = 1.0 / np.array(vols)
            weights = inv_v / inv_v.sum()
            fallback += 1

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
        "name": name, "top_k": top_k, "estimator": estimator,
        "max_weight": max_weight, "inv_vol_blend": inv_vol_blend,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4), "n_months": int(m["n_months"]),
        "cash_months": int((df["n_picks"] == 0).sum()),
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(df["scale"].mean(), 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
        "lw_ok_pct": round(lw_ok / max(lw_ok + fallback, 1), 2),
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:72s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} "
          f"lw={res['lw_ok_pct']:.0%} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference (inv_vol baseline)
# ---------------------------------------------------------------------------
print("\n--- Reference: inv_vol (no LW min-var) ---")
r = run_lw("REF inv_vol K=30", 30, max_weight=1.0, inv_vol_blend=1.0)
if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# A. Pure Ledoit-Wolf min-var
# ---------------------------------------------------------------------------
print("\n--- A. Ledoit-Wolf min-var (max_weight=0.10, 0.15, 0.20) ---")
for mw in [0.08, 0.10, 0.12, 0.15, 0.20]:
    for k in [25, 30, 40]:
        r = run_lw(f"LW minvar K={k:3d} maxw={mw:.0%}", k, max_weight=mw)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# B. OAS estimator (Oracle Approximating Shrinkage)
# ---------------------------------------------------------------------------
print("\n--- B. OAS estimator ---")
for mw in [0.10, 0.15]:
    for k in [25, 30]:
        r = run_lw(f"OAS minvar K={k:3d} maxw={mw:.0%}", k, estimator="oas", max_weight=mw)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# C. LW blend with inv_vol (reduces estimation-error risk)
# ---------------------------------------------------------------------------
print("\n--- C. LW min-var blended with inv_vol ---")
for blend in [0.25, 0.50, 0.75]:
    for k in [25, 30]:
        r = run_lw(f"LW+ivol blend={blend:.0%} K={k:3d}", k, inv_vol_blend=blend, max_weight=0.15)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# D. LW min-var with 3-way blend
# ---------------------------------------------------------------------------
print("\n--- D. LW min-var with 3-way blend ---")
for mw in [0.10, 0.15]:
    r = run_lw(f"LW 3way K= 30 maxw={mw:.0%}", 30, blend_weights=BLEND_3WAY, max_weight=mw)
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
              "lw_ok_pct", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_020_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      ratio={best['ratio']:.4f} (target 0.5774)")

if len(passing) > 0:
    print(f"\n{'='*50}")
    print(f"*** {len(passing)} CONFIGS PASS BOTH GATES ***")
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
