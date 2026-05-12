"""
Experiment 015: Correlation-Penalized Greedy Selection

INSIGHT: The Sharpe ceiling at ~1.83 is caused by high cross-correlation between
momentum picks. Analysis of the portfolio structure:

  Observed: std_m=8.51%, ann_vol=29.5% (K=30, inv_vol)
  If avg pairwise ρ=0.50 and individual σ=40%:
    Portfolio σ ≈ √(ρ + (1-ρ)/K) × σ_ind = √(0.517) × 40% = 28.8% ✓

  If we can reduce ρ from 0.50 to 0.30:
    Portfolio σ ≈ √(0.323) × 40% = 22.7% → std_m=6.55%
    Sharpe = 4.48%/6.55% × √12 = 2.37 ✓✓

APPROACH: Greedy sequential selection that maximizes:
    net_score_i = blend_score_i - λ × avg_correlation(i, already_selected)

First, filter to top 100 candidates by blend score. Then greedily pick K stocks
that maximize score while penalizing correlation with already-selected stocks.

This is analogous to a greedy portfolio optimization but computationally efficient.
Correlation estimated from daily returns over the last 252 trading days.
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
print("EXPERIMENT 015: CORRELATION-PENALIZED GREEDY SELECTION")
print("=" * 90)
print("\nHypothesis: Reducing pairwise correlation from 0.50 to 0.30 → Sharpe 2.0+")
print("Method: Greedy selection penalizing avg_correlation with already-selected stocks")

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
LOOKBACK_DAYS = 252  # days of daily returns for correlation estimation

# Best blend from exp_009 (will also try new signals from exp_014)
BLEND_BASE = {"lgbm": 0.70, "sharpe_12m": 0.20, "sharpe_5y": 0.10}


def znorm(s):
    mu, si = s.mean(), s.std()
    return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)


def compute_blend_scores(feat_df, date, blend_weights=BLEND_BASE):
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


def get_corr_matrix(tickers, date, prices_df, lookback=LOOKBACK_DAYS):
    """Compute correlation matrix of daily returns for given tickers up to date."""
    date_idx = prices_df.index.searchsorted(date, side='right')
    if date_idx < lookback:
        return None
    start_idx = max(0, date_idx - lookback)
    px_slice = prices_df.iloc[start_idx:date_idx][tickers].dropna(axis=1, how='all')
    if px_slice.shape[1] < 2:
        return None
    rets = px_slice.pct_change().dropna()
    if len(rets) < 30:
        return None
    corr = rets.corr()
    return corr


def greedy_select(scores, corr_matrix, top_k, lambda_corr):
    """Greedy correlation-penalized selection."""
    available = [t for t in scores.index if t in corr_matrix.columns]
    if len(available) < top_k:
        return scores.index[:top_k].tolist()

    scores_avail = scores.loc[available].sort_values(ascending=False)
    selected = []

    for _ in range(top_k):
        if not selected:
            # First pick: highest score
            best = scores_avail.index[0]
            selected.append(best)
        else:
            best = None
            best_net = -np.inf
            for t in scores_avail.index:
                if t in selected:
                    continue
                # Average correlation with already selected
                corr_vals = [corr_matrix.loc[t, s] for s in selected
                             if s in corr_matrix.columns and t in corr_matrix.index]
                avg_corr = np.nanmean(corr_vals) if corr_vals else 0.0
                net_score = float(scores_avail[t]) - lambda_corr * avg_corr
                if net_score > best_net:
                    best_net = net_score
                    best = t
            if best is not None:
                selected.append(best)

    return selected


def run_diversified(
    name,
    top_k,
    lambda_corr,          # correlation penalty coefficient
    candidate_k=100,      # pre-filter to top N by score before greedy selection
    blend_weights=None,
    regime="200ma_loose",
    target_vol=0.18,
    start=OOS_START,
    end=OOS_END,
):
    if blend_weights is None:
        blend_weights = BLEND_BASE
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
        scores = compute_blend_scores(feats, date, blend_weights).dropna()
        scores = scores[~scores.index.isin(EXCLUDE)]
        if scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        # Pre-filter to top candidate_k
        candidates = scores.sort_values(ascending=False).head(candidate_k)
        candidate_tickers = [t for t in candidates.index if t in prices.columns]

        if lambda_corr > 0 and len(candidate_tickers) >= 2:
            corr_mat = get_corr_matrix(candidate_tickers, date, prices)
            if corr_mat is not None:
                tickers = greedy_select(candidates.loc[candidate_tickers], corr_mat, top_k, lambda_corr)
            else:
                tickers = candidates.index[:top_k].tolist()
        else:
            tickers = candidates.index[:top_k].tolist()

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
    res = {
        "name": name, "top_k": top_k, "lambda_corr": lambda_corr,
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
    print(f"  {name:72s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference (no correlation penalty = standard approach)
# ---------------------------------------------------------------------------
print("\n--- Reference: no corr penalty (λ=0), K=30 ---")
r = run_diversified("REF λ=0.0 K= 30 3way_blend inv_vol", 30, lambda_corr=0.0)
if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 1. Correlation penalty sweep
# ---------------------------------------------------------------------------
print("\n--- Correlation penalty sweep (λ=0.1 to 1.0) ---")
for lam in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    for k in [25, 30, 40]:
        r = run_diversified(f"corr_div λ={lam:.1f} K={k:3d} 3way", k, lambda_corr=lam)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Different candidate set sizes
# ---------------------------------------------------------------------------
print("\n--- Different candidate pool sizes before greedy selection ---")
for cand_k in [50, 75, 150]:
    for lam in [0.3, 0.5]:
        r = run_diversified(
            f"corr_div λ={lam:.1f} K= 30 cand={cand_k}",
            30, lambda_corr=lam, candidate_k=cand_k
        )
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Larger K with correlation penalty (more room to diversify)
# ---------------------------------------------------------------------------
print("\n--- Larger K with correlation penalty ---")
for lam in [0.3, 0.5]:
    for k in [50, 75, 100]:
        r = run_diversified(f"corr_div λ={lam:.1f} K={k:3d} large", k, lambda_corr=lam,
                            candidate_k=200)
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
              "lambda_corr", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_015_results.csv"
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
print(f"Running total hypotheses: 435 + exp014_n + {n_configs}")
