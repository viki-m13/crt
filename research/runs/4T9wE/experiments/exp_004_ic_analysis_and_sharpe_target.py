"""
Experiment 004: IC Analysis and Sharpe-Target LightGBM

Goals:
1. Measure actual IC of the LGBM model to understand the Sharpe ceiling
2. Train LightGBM to predict individual-stock Sharpe (risk-adjusted return)
3. Test if a Sharpe-targeting model achieves higher portfolio Sharpe

The Grinold-Kahn fundamental law:
  Portfolio Sharpe ≈ IC × sqrt(breadth)
  For Sharpe=2.0 with K=20: IC = 2.0/sqrt(20×12) = 0.129

If actual IC < 0.10, portfolio Sharpe 2.0 is impossible with K=20.
"""
import sys, time, warnings, pickle
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

from backtest.engine import (
    run_backtest, make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM, FEATURE_COLS
from features.signals import composite_v1, smooth_compounder, set_date_context

print("=" * 90)
print("EXPERIMENT 004: IC ANALYSIS + SHARPE-TARGET LGBM")
print("=" * 90)

# ---------------------------------------------------------------------------
# Load LGBM models (cached)
# ---------------------------------------------------------------------------
print("\nLoading LightGBM models...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()
monthly_px = get_monthly_prices()

wf = WalkForwardLGBM(train_months=48, embargo_months=3, min_train_months=24)
wf.prepare_data(prices, dates_all)

score_fn_cache = {}
for d in dates_all:
    fn = wf.get_score_fn(d, dates_all)
    if fn is not None:
        score_fn_cache[d] = fn
print(f"LGBM models ready for {len(score_fn_cache)} dates")

# ---------------------------------------------------------------------------
# IC Measurement
# ---------------------------------------------------------------------------
print("\n--- IC Analysis ---")

def compute_actual_fwd_ret(date: pd.Timestamp, next_date: pd.Timestamp, tickers: list) -> pd.Series:
    """Compute actual 1-month forward returns for given tickers."""
    d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
    if d0_idx > 0 and monthly_px.index[d0_idx] > date:
        d0_idx -= 1
    d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
    if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
        d1_idx -= 1

    p0 = monthly_px.iloc[d0_idx]
    p1 = monthly_px.iloc[d1_idx]
    valid = [t for t in tickers if t in monthly_px.columns
             and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
             and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
    if not valid:
        return pd.Series(dtype=float)
    return (p1[valid] - p0[valid]) / p0[valid]


test_dates = [d for d in dates_all if pd.Timestamp('2007-01-01') <= d <= pd.Timestamp('2021-12-31')]
ics_lgbm = []
ics_composite = []
ics_smooth = []

print("Computing IC for each prediction date...")
for i, date in enumerate(test_dates[:-1]):
    next_date = test_dates[i + 1]

    if date not in score_fn_cache:
        continue

    feats = load_features(date)
    if feats.empty:
        continue

    # LGBM scores
    set_date_context(date)
    lgbm_scores = score_fn_cache[date](feats)
    comp_scores = composite_v1(feats)
    smooth_scores = smooth_compounder(feats)

    # Actual returns
    tickers = feats.index.tolist()
    actual = compute_actual_fwd_ret(date, next_date, tickers)

    # Compute IC
    common = [t for t in actual.index if t in lgbm_scores.index and t in comp_scores.index
              and not np.isnan(lgbm_scores.get(t, np.nan))
              and not np.isnan(comp_scores.get(t, np.nan))]
    if len(common) < 20:
        continue

    actual_common = actual[common]
    lgbm_common = lgbm_scores[common]
    comp_common = comp_scores[common]

    ic_lgbm = spearmanr(lgbm_common, actual_common)[0]
    ic_comp = spearmanr(comp_common, actual_common)[0]

    if len([t for t in common if t in smooth_scores.index]) >= 20:
        smooth_common = smooth_scores[[t for t in common if t in smooth_scores.index]]
        actual_smooth = actual[[t for t in common if t in smooth_scores.index and not np.isnan(smooth_scores.get(t, np.nan))]]
        if len(actual_smooth) >= 20:
            ic_smooth = spearmanr(smooth_common.loc[actual_smooth.index], actual_smooth)[0]
            ics_smooth.append(ic_smooth)

    ics_lgbm.append(ic_lgbm)
    ics_composite.append(ic_comp)

print(f"\nIC Statistics (2007-2021, n={len(ics_lgbm)} dates):")
print(f"  LGBM:      mean={np.mean(ics_lgbm):.4f}, std={np.std(ics_lgbm):.4f}, "
      f"pct_positive={np.mean(np.array(ics_lgbm)>0):.1%}")
print(f"  Composite: mean={np.mean(ics_composite):.4f}, std={np.std(ics_composite):.4f}, "
      f"pct_positive={np.mean(np.array(ics_composite)>0):.1%}")
if ics_smooth:
    print(f"  Smooth:    mean={np.mean(ics_smooth):.4f}, std={np.std(ics_smooth):.4f}, "
          f"pct_positive={np.mean(np.array(ics_smooth)>0):.1%}")

mean_ic_lgbm = np.mean(ics_lgbm)
print(f"\n=== Sharpe Ceiling Analysis ===")
for k in [5, 10, 15, 20, 30, 40]:
    sharpe_ceiling = mean_ic_lgbm * np.sqrt(k * 12)
    print(f"  K={k:2d}: Theoretical max Sharpe = {sharpe_ceiling:.2f} "
          f"(need IC={2.0/np.sqrt(k*12):.3f} for Sharpe 2.0)")

# ---------------------------------------------------------------------------
# Train Sharpe-target LightGBM
# ---------------------------------------------------------------------------
print("\n--- Training Sharpe-Target LGBM ---")

cache_dir = Path("/home/user/crt/quant_research/data/cache")
with open(list(cache_dir.glob("lgbm_panel_*.pkl"))[0], "rb") as f:
    panel = pickle.load(f)
with open(list(cache_dir.glob("lgbm_fwd_*.pkl"))[0], "rb") as f:
    fwd_returns = pickle.load(f)

# Build individual stock Sharpe targets
# For each (stock, date), compute trailing 12m Sharpe
def compute_rolling_sharpe(ticker: str, date: pd.Timestamp, lookback: int = 252) -> float:
    """Compute trailing 12m daily return Sharpe for a stock."""
    if ticker not in prices.columns:
        return np.nan
    d_idx = prices.index.searchsorted(date, side="right") - 1
    start_idx = max(0, d_idx - lookback)
    px = prices.iloc[start_idx:d_idx + 1][ticker].dropna()
    if len(px) < 60:
        return np.nan
    rets = px.pct_change().dropna()
    return (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan


# For Sharpe target: use next-month FORWARD Sharpe-like metric
# Proxy: forward return / expected vol (where expected vol = trailing vol)
# This better targets stocks with high risk-adjusted returns

def compute_sharpe_target(ticker: str, date: pd.Timestamp, next_date: pd.Timestamp) -> float:
    """forward_return / trailing_vol = risk-adjusted forward return."""
    if ticker not in prices.columns:
        return np.nan
    # Forward 1m return
    d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
    if d0_idx > 0 and monthly_px.index[d0_idx] > date:
        d0_idx -= 1
    d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
    if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
        d1_idx -= 1
    p0 = monthly_px.iloc[d0_idx].get(ticker, np.nan)
    p1 = monthly_px.iloc[d1_idx].get(ticker, np.nan)
    if not (np.isfinite(p0) and p0 > 0 and np.isfinite(p1) and p1 > 0):
        return np.nan
    fwd_ret = (p1 - p0) / p0

    # Trailing vol (from feature if available, else from prices)
    # Use trailing 63-day vol from price data
    d_idx = prices.index.searchsorted(date, side="right") - 1
    px = prices.iloc[max(0, d_idx - 63):d_idx + 1].get(ticker, pd.Series()).dropna()
    if len(px) < 20:
        return np.nan
    trail_vol = px.pct_change().dropna().std() * np.sqrt(252)
    if trail_vol < 0.05:
        trail_vol = 0.05
    return fwd_ret / trail_vol


# Build Sharpe-target panel (sample for speed)
print("Building Sharpe-target dataset...")

sharpe_target_cache = cache_dir / "sharpe_target_panel.pkl"
if sharpe_target_cache.exists():
    with open(sharpe_target_cache, "rb") as f:
        sharpe_panel = pickle.load(f)
    print(f"  Loaded from cache: {sharpe_panel.shape}")
else:
    train_dates = [d for d in dates_all if pd.Timestamp('2003-01-01') <= d <= pd.Timestamp('2018-12-31')]
    sample_dates = train_dates  # Use all

    records = []
    fwd_reset = fwd_returns.reset_index()
    if "ticker" not in fwd_reset.columns:
        fwd_reset = fwd_reset.rename(columns={"index": "ticker"})

    for i, d in enumerate(sample_dates[:-1]):
        next_d = sample_dates[i + 1]
        p_sub = panel[panel["asof"] == d]
        f_sub = fwd_reset[fwd_reset["asof"] == d]

        if p_sub.empty:
            continue

        merged = p_sub.merge(
            f_sub[["ticker", "fwd_ret_1m"]],
            on="ticker", how="inner"
        )

        available_cols = [c for c in FEATURE_COLS if c in merged.columns]
        for _, row in merged.iterrows():
            ticker = row.get("ticker", "")
            fwd_ret = row.get("fwd_ret_1m", np.nan)
            if not np.isfinite(fwd_ret) or abs(fwd_ret) > 3.0:
                continue
            # Get trailing vol from features
            trail_vol = row.get("vol_3m", row.get("vol_12m", 0.20))
            if not np.isfinite(trail_vol) or trail_vol < 0.05:
                trail_vol = 0.20
            sharpe_target = fwd_ret / trail_vol
            records.append({
                "ticker": ticker,
                "asof": d,
                "sharpe_target": sharpe_target,
                **{c: row.get(c, np.nan) for c in available_cols},
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(sample_dates)} dates, {len(records)} records")

    sharpe_panel = pd.DataFrame(records)
    with open(sharpe_target_cache, "wb") as f:
        pickle.dump(sharpe_panel, f)
    print(f"  Built Sharpe-target panel: {sharpe_panel.shape}")


# Train walk-forward Sharpe-target models
print("Training walk-forward Sharpe-target models...")

sharpe_score_cache = {}
train_months_per_model = 48
embargo_months = 3

pred_start = dates_all[train_months_per_model + embargo_months]
pred_dates = [d for d in dates_all
              if d >= pred_start and d <= pd.Timestamp('2021-12-31')]

for pred_date in pred_dates:
    pred_idx = dates_all.index(pred_date)
    embargo_end = pred_idx - embargo_months
    train_start_idx = max(0, embargo_end - train_months_per_model)
    train_date_set = set(dates_all[train_start_idx:embargo_end])

    # Get training data for Sharpe target
    if "asof" in sharpe_panel.columns:
        train_data = sharpe_panel[sharpe_panel["asof"].isin(train_date_set)]
    else:
        continue

    if len(train_data) < 500:
        continue

    available_cols = [c for c in FEATURE_COLS if c in train_data.columns]
    X = train_data[available_cols].fillna(train_data[available_cols].median())
    y = train_data["sharpe_target"]
    valid = y.notna() & (y.abs() < 5.0)

    if valid.sum() < 300:
        continue

    # Cross-sectional rank of sharpe_target within each date
    y_ranked = y.copy()
    for date_val, grp in train_data[valid].groupby("asof"):
        idx = grp.index
        if len(idx) < 5:
            continue
        y_ranked.loc[idx] = y.loc[idx].rank(pct=True)

    lgb_params = {
        "objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
        "n_estimators": 150, "min_child_samples": 30, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X[valid], y_ranked[valid])

    def make_sharpe_score(m, cols):
        def sharpe_score_fn(feat_df: pd.DataFrame) -> pd.Series:
            avail = [c for c in cols if c in feat_df.columns]
            if not avail:
                return pd.Series(dtype=float)
            X_pred = feat_df[avail].fillna(feat_df[avail].median())
            return pd.Series(m.predict(X_pred), index=feat_df.index)
        return sharpe_score_fn

    sharpe_score_cache[pred_date] = make_sharpe_score(model, available_cols)


print(f"Sharpe-target models ready for {len(sharpe_score_cache)} dates")


# ---------------------------------------------------------------------------
# Backtest with Sharpe-target model
# ---------------------------------------------------------------------------
def sharpe_target_score_fn(feat_df: pd.DataFrame) -> pd.Series:
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    if date is None or date not in sharpe_score_cache:
        return composite_v1(feat_df)
    return sharpe_score_cache[date](feat_df)


def sharpe_target_smooth(feat_df: pd.DataFrame) -> pd.Series:
    from features.signals import _CURRENT_DATE
    date = _CURRENT_DATE
    st = sharpe_score_cache.get(date, composite_v1)(feat_df) if date else composite_v1(feat_df)
    smooth = smooth_compounder(feat_df)
    def znorm(s):
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 1e-10 else pd.Series(0.0, index=s.index)
    combined = znorm(st) * 1.5 + znorm(smooth) * 1.0
    mom12 = feat_df.get("mom_12_1", pd.Series(1.0, index=feat_df.index)) if "mom_12_1" in feat_df.columns else pd.Series(1.0, index=feat_df.index)
    combined[mom12 <= 0] = np.nan
    return combined


RESULTS = []
OOS_START = "2007-01-31"
OOS_END = "2021-12-31"


def run(name, score_fn, top_k, weighting="ew", regime=None):
    t0 = time.time()
    regime_fn = make_regime_fn(regime) if regime else None
    df, metrics = run_backtest(
        score_fn=score_fn, start=OOS_START, end=OOS_END,
        top_k=top_k, weighting=weighting, cost_bps=5.0, regime_fn=regime_fn,
    )
    elapsed = time.time() - t0
    if not metrics:
        print(f"  {name}: NO RESULTS"); return
    cash_m = int((df["n_picks"] == 0).sum()) if "n_picks" in df.columns else 0
    result = {
        "name": name, "top_k": top_k, "weighting": weighting,
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
    print(f"  {name:60s} CAGR={result['cagr']:.1%}{gc} Sharpe={result['sharpe']:.2f}{gs} "
          f"MaxDD={result['max_dd']:.1%} Cash={cash_m}m {elapsed:.0f}s")


print("\n--- Sharpe-Target LGBM Results ---")
print("Baseline for comparison:")
def lgbm_smooth_ref(feat_df: pd.DataFrame) -> pd.Series:
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

RESULTS.clear()

run("lgbm_smooth [EXP002_BEST] K=20 inv_vol + regime", lgbm_smooth_ref, 20, "inv_vol", "200ma")

for k in [10, 15, 20, 30]:
    run(f"sharpe_target K={k:2d} inv_vol + regime", sharpe_target_score_fn, k, "inv_vol", "200ma")
for k in [10, 15, 20, 30]:
    run(f"sharpe_smooth K={k:2d} inv_vol + regime", sharpe_target_smooth, k, "inv_vol", "200ma")
for k in [10, 20]:
    run(f"sharpe_target K={k:2d} inv_vol [no regime]", sharpe_target_score_fn, k, "inv_vol")
for k in [10, 20]:
    run(f"sharpe_smooth K={k:2d} inv_vol [no regime]", sharpe_target_smooth, k, "inv_vol")

# IC analysis for Sharpe-target model
print("\n--- IC of Sharpe-Target Model ---")
st_ics = []
for i, date in enumerate(test_dates[:-1]):
    next_date = test_dates[i + 1]
    if date not in sharpe_score_cache:
        continue
    feats = load_features(date)
    if feats.empty:
        continue
    set_date_context(date)
    st_scores = sharpe_score_cache[date](feats)
    tickers = feats.index.tolist()
    actual = compute_actual_fwd_ret(date, next_date, tickers)
    common = [t for t in actual.index if t in st_scores.index
              and not np.isnan(st_scores.get(t, np.nan))]
    if len(common) < 20:
        continue
    ic = spearmanr(st_scores[common], actual[common])[0]
    st_ics.append(ic)

if st_ics:
    print(f"  Sharpe-target IC: mean={np.mean(st_ics):.4f}, "
          f"std={np.std(st_ics):.4f}, pct_positive={np.mean(np.array(st_ics)>0):.1%}")

# Summary
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "win_rate", "cash_months",
              "ann_vol", "top_k"]].to_string(index=False))

out = Path(__file__).parent / "exp_004_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(f"\nBest: {df_res.iloc[0]['name']} -> CAGR={df_res.iloc[0]['cagr']:.1%} Sharpe={df_res.iloc[0]['sharpe']:.2f}")

best_both = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]
if len(best_both) > 0:
    print(f"\n*** PASSED BOTH GATES ***")
    print(best_both[["name", "cagr", "sharpe", "max_dd"]].to_string(index=False))
