"""
Experiment 019: Hierarchical Risk Parity (HRP) Weighting

MOTIVATION:
ERC weighting (exp_012) increased portfolio vol from 29.5% to 34.5% — WORSE.
Root cause: ERC optimization is unstable with high cross-correlations;
forces over-weighting low-vol stocks in ways that increase portfolio vol.

HRP is fundamentally different from ERC:
- No matrix inversion or optimization
- Hierarchical clustering of assets by correlation
- Recursive bisection to allocate weight across the cluster tree
- Known to be more robust than MVO and ERC out-of-sample (Lopez de Prado 2016)

HOW HRP WORKS:
1. Compute correlation matrix of the K picks (252d rolling daily returns)
2. Convert to distance matrix: d_ij = √(½(1-ρ_ij))
3. Hierarchical clustering (single linkage = most connected = lower vol clusters)
4. Quasi-diagonalize: sort stocks by cluster
5. Recursive bisection: split portfolio into sub-portfolios at each cluster node,
   allocate by inverse-variance across branches
6. Final weight: HRP weight × inv_vol (size by vol within HRP structure)

KEY DIFFERENCE FROM ERC:
- ERC equalizes marginal risk contribution (requires solving for weights that
  satisfy w_i × (Σw)_i = const for all i)
- HRP allocates by cluster structure; if K=30 stocks are all in one cluster,
  it degenerates to inv_var weighting (similar to inv_vol but by variance not vol)

HRP vs inv_vol:
- inv_vol: w_i ∝ 1/σ_i (ignores cross-correlations entirely)
- HRP: weights cluster sub-portfolios by their cluster variance (accounts for
  cross-correlation within clusters)
- When ρ is high and uniform, HRP ≈ inv_var ≈ inv_vol
- When ρ is heterogeneous (some clusters more correlated than others),
  HRP diversifies across clusters → lower actual portfolio vol

HYPOTHESIS: Our K=30 momentum picks may have heterogeneous within-cluster
correlation. Tech momentum stocks highly correlated among themselves but less
with energy/healthcare momentum stocks. HRP would underweight the tech cluster
and overweight the less-correlated clusters → lower portfolio vol.

Prior best: 4way+volasym10% K=30 → CAGR=66.5%, Sharpe=1.841, ratio=0.5315
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from backtest.engine import (
    make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM
from features.signals import set_date_context

print("=" * 90)
print("EXPERIMENT 019: HIERARCHICAL RISK PARITY (HRP) WEIGHTING")
print("=" * 90)

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


def get_corr_and_cov(tickers, date, prices_df, lookback=LOOKBACK_DAYS):
    """Return (corr_matrix, cov_matrix, ticker_list) using last `lookback` daily returns."""
    date_idx = prices_df.index.searchsorted(date, side='right')
    if date_idx < lookback:
        return None, None, None
    px_slice = prices_df.iloc[date_idx - lookback:date_idx][tickers].dropna(axis=1, how='all')
    valid_tickers = px_slice.columns.tolist()
    if len(valid_tickers) < 3:
        return None, None, None
    rets = px_slice.pct_change().dropna()
    if len(rets) < 30:
        return None, None, None
    cov = rets.cov() * 252
    corr = rets.corr()
    return corr, cov, valid_tickers


def hrp_weights(cov_df, corr_df):
    """
    Hierarchical Risk Parity weights.
    Returns: pd.Series of weights indexed by ticker.
    """
    tickers = cov_df.index.tolist()
    n = len(tickers)
    if n == 1:
        return pd.Series(1.0, index=tickers)

    # Distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
    corr_arr = corr_df.values
    dist = np.sqrt(0.5 * (1.0 - np.clip(corr_arr, -1, 1)))
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering (single linkage)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method='single')

    # Get sorted order from dendrogram
    def get_quasi_diag(link, n):
        """Get assets sorted by hierarchical clustering."""
        root, node_list = to_tree(link, rd=True)
        stack = [root]
        ordered = []
        while stack:
            node = stack.pop()
            if node.is_leaf():
                ordered.append(node.id)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return ordered

    sorted_idx = get_quasi_diag(link, n)
    sorted_tickers = [tickers[i] for i in sorted_idx]

    # Recursive bisection
    var_arr = np.diag(cov_df.values)  # annualized variances

    def recursive_bisect(items):
        if len(items) == 1:
            return {items[0]: 1.0}
        # Split into two halves
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Cluster variance for each half (using diagonal cov = ignoring cross-corr within half)
        def cluster_var(group):
            sub_cov = cov_df.loc[group, group].values
            inv_diag = 1.0 / np.diag(sub_cov)
            w = inv_diag / inv_diag.sum()
            return float(w @ sub_cov @ w)

        var_left = cluster_var(left)
        var_right = cluster_var(right)

        # Allocate proportional to inverse cluster variance
        alpha = 1.0 - var_left / (var_left + var_right)  # weight to left

        w_left = recursive_bisect(left)
        w_right = recursive_bisect(right)

        combined = {}
        for t, w in w_left.items():
            combined[t] = alpha * w
        for t, w in w_right.items():
            combined[t] = (1.0 - alpha) * w
        return combined

    w_dict = recursive_bisect(sorted_tickers)
    return pd.Series(w_dict).reindex(tickers).fillna(0.0)


def run_hrp(
    name,
    top_k,
    blend_weights=None,
    hrp_scale=1.0,     # blend: hrp_scale * HRP + (1-hrp_scale) * inv_vol
    inv_vol_floor=0.0,  # minimum inv_vol weight before HRP adjustment
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

    hrp_used = 0
    fallback_used = 0

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

        # Compute inv_vol weights (baseline)
        vols = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols.append(0.20)
        inv_v = np.array([1.0 / v for v in vols])
        inv_vol_w = inv_v / inv_v.sum()

        # Attempt HRP
        weights = inv_vol_w  # default fallback
        if hrp_scale > 0:
            corr_df, cov_df, valid = get_corr_and_cov(common, date, prices)
            if cov_df is not None and len(valid) >= 3:
                # Use only tickers with valid price data
                hrp_w = hrp_weights(cov_df.loc[valid, valid], corr_df.loc[valid, valid])
                hrp_w = hrp_w.reindex(common).fillna(0.0)
                hrp_sum = hrp_w.sum()
                if hrp_sum > 1e-6:
                    hrp_w = hrp_w / hrp_sum
                    # Blend HRP with inv_vol
                    hrp_arr = hrp_w.values
                    weights = hrp_scale * hrp_arr + (1.0 - hrp_scale) * inv_vol_w
                    weights = weights / weights.sum()
                    hrp_used += 1
                else:
                    fallback_used += 1
            else:
                fallback_used += 1

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
        "name": name, "top_k": top_k, "hrp_scale": hrp_scale,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4), "n_months": int(m["n_months"]),
        "cash_months": int((df["n_picks"] == 0).sum()),
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(df["scale"].mean(), 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
        "hrp_used_pct": round(hrp_used / max(hrp_used + fallback_used, 1), 2),
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:72s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} "
          f"hrp={res['hrp_used_pct']:.0%} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 0. Reference
# ---------------------------------------------------------------------------
print("\n--- Reference: inv_vol only (hrp_scale=0) ---")
r = run_hrp("REF inv_vol K=30", 30, hrp_scale=0.0)
if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# A. Pure HRP weighting
# ---------------------------------------------------------------------------
print("\n--- A. Pure HRP (hrp_scale=1.0) ---")
for k in [20, 25, 30, 40, 50]:
    r = run_hrp(f"HRP pure K={k:3d}", k, hrp_scale=1.0)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# B. HRP blend with inv_vol (reduce sensitivity to estimation error)
# ---------------------------------------------------------------------------
print("\n--- B. HRP blend with inv_vol ---")
for scale in [0.25, 0.50, 0.75]:
    for k in [25, 30, 40]:
        r = run_hrp(f"HRP blend={scale:.0%} K={k:3d}", k, hrp_scale=scale)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# C. Pure HRP with 3-way blend
# ---------------------------------------------------------------------------
print("\n--- C. HRP with 3-way blend ---")
for k in [25, 30]:
    r = run_hrp(f"HRP pure K={k:3d} 3way", k, hrp_scale=1.0, blend_weights=BLEND_3WAY)
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
              "hrp_used_pct", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_019_results.csv"
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
