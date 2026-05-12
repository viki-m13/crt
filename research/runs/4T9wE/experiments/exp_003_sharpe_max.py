"""
Experiment 003: Maximize Sharpe via portfolio construction and regime tuning.
Goal: push Sharpe from 1.63 (exp_002 best) toward 2.0.

Levers:
  1. Larger K (more diversification → lower vol)
  2. Min-variance portfolio weighting
  3. More aggressive regime gate (vol-triggered cash)
  4. Multi-level regime: cash/defensive/full

Best so far: lgbm_smooth K=20 inv_vol + regime → CAGR=48.6%, Sharpe=1.63, MaxDD=-11.4%
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest.engine import (
    run_backtest, make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, get_spy_stats_at, get_spy_regime_df,
    EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import (
    composite_v1, smooth_compounder, set_date_context
)

# ---------------------------------------------------------------------------
# Prepare LightGBM models (same as exp_002)
# ---------------------------------------------------------------------------
print("=" * 90)
print("EXPERIMENT 003: SHARPE MAXIMIZATION")
print("=" * 90)

print("\nLoading LightGBM models...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()

wf_lgbm = WalkForwardLGBM(train_months=48, embargo_months=3, min_train_months=24)
wf_lgbm.prepare_data(prices, dates_all)

score_fn_cache = {}
for d in dates_all:
    fn = wf_lgbm.get_score_fn(d, dates_all)
    if fn is not None:
        score_fn_cache[d] = fn
print(f"LGBM models ready for {len(score_fn_cache)} dates")

# ---------------------------------------------------------------------------
# Advanced regime functions (price-derived SPY statistics)
# ---------------------------------------------------------------------------
spy_regime = get_spy_regime_df()


def make_vol_regime(vol_threshold: float = 0.20, dsma_threshold: float = -0.05):
    """Go to cash when realized SPY vol exceeds threshold OR SPY below 200MA."""
    def regime_fn(date: pd.Timestamp, feats: pd.DataFrame) -> bool:
        stats = get_spy_stats_at(date)
        if not stats:
            return True
        d200 = stats.get("d_sma200", 0.0)
        vol = stats.get("vol_21d", 0.0)
        return bool(d200 > dsma_threshold and vol < vol_threshold)
    return regime_fn


def make_dual_regime(dsma_thresh=-0.02, mom6_thresh=-0.10):
    """SPY above 200-day MA AND 6m momentum not deeply negative."""
    def regime_fn(date: pd.Timestamp, feats: pd.DataFrame) -> bool:
        stats = get_spy_stats_at(date)
        if not stats:
            return True
        d200 = stats.get("d_sma200", 0.0)
        mom6 = stats.get("mom_6_1", 0.0)
        return bool(d200 > dsma_thresh and mom6 > mom6_thresh)
    return regime_fn


def make_triple_regime(dsma1=-0.02, dsma2=0.0, mom6_thresh=-0.05):
    """Graded regime: True only in genuinely good markets."""
    def regime_fn(date: pd.Timestamp, feats: pd.DataFrame) -> bool:
        stats = get_spy_stats_at(date)
        if not stats:
            return True
        d200 = stats.get("d_sma200", 0.0)
        mom6 = stats.get("mom_6_1", 0.0)
        mom3 = stats.get("mom_3", 0.0)
        vol = stats.get("vol_21d", 0.15)
        # Must be clearly positive market + not too volatile
        return bool(d200 > dsma1 and (mom6 > mom6_thresh or mom3 > 0) and vol < 0.25)
    return regime_fn


# ---------------------------------------------------------------------------
# Min-variance portfolio weighting
# ---------------------------------------------------------------------------
def min_var_weights(
    tickers: list,
    feats: pd.DataFrame,
    hist_prices: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = 252,
    max_weight: float = 0.25,
) -> np.ndarray:
    """Compute minimum variance portfolio weights using historical returns."""
    d_idx = hist_prices.index.searchsorted(date, side="right") - 1
    if d_idx < lookback:
        return np.ones(len(tickers)) / len(tickers)

    # Daily returns over lookback period
    avail = [t for t in tickers if t in hist_prices.columns]
    if len(avail) < 2:
        return np.ones(len(tickers)) / len(tickers)

    px_hist = hist_prices.iloc[max(0, d_idx - lookback):d_idx + 1][avail]
    rets = px_hist.pct_change().dropna()
    if len(rets) < 60:
        return np.ones(len(tickers)) / len(tickers)

    cov = rets.cov().values * 252  # Annualized covariance
    n = len(avail)

    # Solve: min w'Σw s.t. sum(w)=1, 0<=w<=max_weight
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, max_weight)] * n

    try:
        result = minimize(
            lambda w: w @ cov @ w,
            w0, method="SLSQP",
            constraints=constraints, bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-8},
        )
        if result.success:
            w = np.maximum(result.x, 0)
            w = w / w.sum()
            # Map back to original tickers list (some may be missing)
            w_out = np.zeros(len(tickers))
            for i, t in enumerate(avail):
                if t in tickers:
                    w_out[tickers.index(t)] = w[i]
            return w_out / w_out.sum() if w_out.sum() > 0 else np.ones(len(tickers)) / len(tickers)
    except Exception:
        pass
    return np.ones(len(tickers)) / len(tickers)


# ---------------------------------------------------------------------------
# Extended run_backtest with min-var weighting support
# ---------------------------------------------------------------------------
def run_backtest_minvar(
    score_fn,
    start: str,
    end: str,
    top_k: int = 20,
    cost_bps: float = 5.0,
    regime_fn=None,
    lookback_days: int = 252,
    max_weight: float = 0.25,
) -> tuple:
    from backtest.engine import compute_metrics
    monthly_px = get_monthly_prices()
    daily_px = prices  # full daily price data
    feat_dates_all = get_feat_dates()
    dates = [d for d in feat_dates_all if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < 6:
        return pd.DataFrame(), {}

    cost = cost_bps / 10_000.0
    records = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        feats = load_features(date)
        if feats.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0})
            continue

        if regime_fn is not None and not regime_fn(date, feats):
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0})
            continue

        set_date_context(date)
        scores = score_fn(feats).dropna()
        scores = scores[~scores.index.isin(EXCLUDE)]
        if scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0})
            continue

        top = scores.sort_values(ascending=False).head(top_k)
        tickers = top.index.tolist()

        d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
        if d0_idx > 0 and monthly_px.index[d0_idx] > date:
            d0_idx -= 1
        d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
        if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
            d1_idx -= 1
        p0_row = monthly_px.iloc[d0_idx]
        p1_row = monthly_px.iloc[d1_idx]

        common = [
            t for t in tickers
            if t in monthly_px.columns
            and np.isfinite(p0_row.get(t, np.nan)) and p0_row.get(t, 0) >= 1.0
            and np.isfinite(p1_row.get(t, np.nan)) and p1_row.get(t, 0) >= 1.0
        ]
        if not common:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0})
            continue

        # Min-var weights
        weights = min_var_weights(common, feats, daily_px, date, lookback_days, max_weight)
        weights = weights[:len(common)]
        if weights.sum() <= 0:
            weights = np.ones(len(common)) / len(common)
        else:
            weights = weights / weights.sum()

        rets = np.array([(p1_row[t] - p0_row[t]) / p0_row[t] for t in common])
        port_ret = float((weights * rets).sum()) - 2 * cost

        records.append({"date": date, "ret_m": port_ret, "n_picks": len(common)})

    if not records:
        return pd.DataFrame(), {}
    df = pd.DataFrame(records).set_index("date")
    return df, compute_metrics(df["ret_m"])


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------
def lgbm_score_fn(feat_df: pd.DataFrame) -> pd.Series:
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    if date is None or date not in score_fn_cache:
        return composite_v1(feat_df)
    return score_fn_cache[date](feat_df)


def lgbm_smooth_fn(feat_df: pd.DataFrame) -> pd.Series:
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    if date is None or date not in score_fn_cache:
        return smooth_compounder(feat_df)
    lgbm = score_fn_cache[date](feat_df)
    smooth = smooth_compounder(feat_df)
    def znorm(s):
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 1e-10 else pd.Series(0.0, index=s.index)
    combined = znorm(lgbm) * 1.5 + znorm(smooth) * 1.0
    mom12 = feat_df["mom_12_1"] if "mom_12_1" in feat_df.columns else pd.Series(1.0, index=feat_df.index)
    combined[mom12 <= 0] = np.nan
    return combined


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
RESULTS = []
TOTAL_HYP = 0
OOS_START = "2007-01-31"
OOS_END = "2021-12-31"


def run(name, score_fn, top_k, weighting="ew", regime_fn=None, n_hyp=1, use_minvar=False):
    global TOTAL_HYP
    TOTAL_HYP += n_hyp
    t0 = time.time()

    if use_minvar:
        df, metrics = run_backtest_minvar(
            score_fn=score_fn, start=OOS_START, end=OOS_END,
            top_k=top_k, cost_bps=5.0, regime_fn=regime_fn,
        )
    else:
        df, metrics = run_backtest(
            score_fn=score_fn, start=OOS_START, end=OOS_END,
            top_k=top_k, weighting=weighting,
            cost_bps=5.0, regime_fn=regime_fn,
        )
    elapsed = time.time() - t0
    if not metrics:
        print(f"  {name}: NO RESULTS")
        return

    cash_m = int((df["n_picks"] == 0).sum()) if "n_picks" in df.columns else 0
    result = {
        "name": name, "top_k": top_k, "weighting": weighting if not use_minvar else "min_var",
        "cagr": round(float(metrics["cagr"]), 4),
        "sharpe": round(float(metrics["sharpe"]), 3),
        "max_dd": round(float(metrics["max_dd"]), 4),
        "win_rate": round(float(metrics["win_rate"]), 3),
        "ann_vol": round(float(metrics["ann_vol"]), 4),
        "n_months": int(metrics["n_months"]),
        "cash_months": cash_m,
        "mean_m": round(float(metrics["mean_m"]), 5),
        "std_m": round(float(metrics["std_m"]), 5),
    }
    RESULTS.append(result)
    gc = "✓" if result["cagr"] >= 0.50 else "✗"
    gs = "✓" if result["sharpe"] >= 2.0 else "✗"
    print(f"  {name:65s} CAGR={result['cagr']:.1%}{gc} Sharpe={result['sharpe']:.2f}{gs} "
          f"MaxDD={result['max_dd']:.1%} Cash={cash_m}m {elapsed:.0f}s")


print("\n--- Larger K for more diversification ---")
for k in [25, 30, 40, 50]:
    run(f"lgbm_smooth K={k:2d} inv_vol + regime_200ma", lgbm_smooth_fn, k, "inv_vol",
        make_regime_fn("200ma"))

print("\n--- More aggressive regime gates ---")
for vol_thresh in [0.18, 0.20, 0.22, 0.25]:
    for k in [20, 30]:
        run(f"lgbm_smooth K={k:2d} inv_vol + vol_regime({vol_thresh})",
            lgbm_smooth_fn, k, "inv_vol",
            make_vol_regime(vol_thresh))

print("\n--- Dual + triple regime conditions ---")
for k in [15, 20, 30]:
    run(f"lgbm_smooth K={k:2d} inv_vol + dual_regime", lgbm_smooth_fn, k, "inv_vol",
        make_dual_regime(-0.02, -0.10))
for k in [15, 20]:
    run(f"lgbm_smooth K={k:2d} inv_vol + triple_regime", lgbm_smooth_fn, k, "inv_vol",
        make_triple_regime())

print("\n--- Min-variance portfolio weights ---")
for k in [20, 30]:
    run(f"lgbm_smooth K={k:2d} min_var + regime_200ma", lgbm_smooth_fn, k,
        regime_fn=make_regime_fn("200ma"), use_minvar=True)
for k in [20, 30]:
    run(f"lgbm_smooth K={k:2d} min_var + vol_regime", lgbm_smooth_fn, k,
        regime_fn=make_vol_regime(0.20), use_minvar=True)

print("\n--- Pure lgbm with larger K ---")
for k in [20, 30, 40]:
    run(f"lgbm K={k:2d} inv_vol + regime_200ma", lgbm_score_fn, k, "inv_vol",
        make_regime_fn("200ma"))

# Summary
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "win_rate", "cash_months",
              "ann_vol", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_003_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(f"\nBest by Sharpe: {df_res.iloc[0]['name']} -> CAGR={df_res.iloc[0]['cagr']:.1%} Sharpe={df_res.iloc[0]['sharpe']:.2f}")

best_both = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]
if len(best_both) > 0:
    print(f"\n*** PASSED BOTH GATES: {len(best_both)} configs! ***")
    print(best_both[["name", "cagr", "sharpe", "max_dd"]].to_string(index=False))
else:
    print(f"\nNo config passes CAGR≥50% AND Sharpe≥2.0 yet.")

print(f"Total hypotheses: {TOTAL_HYP}")
