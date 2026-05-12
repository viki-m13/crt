"""
Experiment 012: Equal Risk Contribution (ERC) + Double-Inv-Vol Weighting

Math: With K=30, 3-way blend:
  mean_m=4.48%, std_m=8.51%, ann_vol=29.9% → Sharpe=1.82

To reach Sharpe=2.0 with same mean_m: need std_m ≤ 7.78% → ann_vol ≤ 27.0%
Required vol reduction: (29.9 - 27.0) / 29.9 = 9.7%

Hypothesis: ERC (equal risk contribution) weighting uses pairwise correlations
to more precisely balance risk. For K=30 stocks with avg correlation ~0.5,
ERC vs inv_vol can reduce portfolio vol by 5-12%.

Also test: 1/vol² weighting (more aggressive tilt to low-vol stocks).

Prior best: K=30 lgbm×0.70+sh12×0.20+sh5y×0.10 → CAGR=63.1%, Sharpe=1.82
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
from features.signals import composite_v1, set_date_context

print("=" * 90)
print("EXPERIMENT 012: ERC + DOUBLE-INV-VOL WEIGHTING")
print("=" * 90)

print("\nLoading LGBM models...")
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

DAILY_PX = prices  # full daily price data


def erc_weights(tickers: list, date: pd.Timestamp, lookback_days: int = 252) -> np.ndarray:
    """Equal risk contribution weights using trailing covariance."""
    n = len(tickers)
    if n <= 1:
        return np.ones(n) / n

    # Get daily returns
    d_idx = DAILY_PX.index.searchsorted(date, side="right") - 1
    if d_idx < lookback_days:
        return np.ones(n) / n

    avail = [t for t in tickers if t in DAILY_PX.columns]
    if len(avail) < 2:
        return np.ones(n) / n

    px_hist = DAILY_PX.iloc[max(0, d_idx - lookback_days):d_idx + 1][avail]
    rets = px_hist.pct_change().dropna()
    if len(rets) < 60:
        return np.ones(n) / n

    cov = rets.cov().values * 252  # annualized covariance

    def erc_objective(w):
        w = np.maximum(w, 1e-8)
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-10:
            return 0.0
        mc = cov @ w / sigma_p
        rc = w * mc
        target = sigma_p / n  # equal contribution
        return float(np.sum((rc - target) ** 2))

    w0 = np.ones(len(avail)) / len(avail)
    bounds = [(0.001, 0.20)] * len(avail)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    try:
        result = minimize(erc_objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 200, "ftol": 1e-9})
        if result.success:
            w = np.maximum(result.x, 0)
            w_out = np.zeros(n)
            for i, t in enumerate(avail):
                if t in tickers:
                    w_out[tickers.index(t)] = w[i]
            return w_out / w_out.sum() if w_out.sum() > 0 else np.ones(n) / n
    except Exception:
        pass
    return np.ones(n) / n


def run_custom_weighting(
    name: str,
    top_k: int,
    weighting: str = "inv_vol",    # "inv_vol", "inv_vol2", "erc", "max_cap"
    max_weight_cap: float = None,  # e.g., 0.10 for 10% max
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

        stats = get_spy_stats_at(date)
        spy_vol = stats.get("vol_21d", target_vol) if stats else target_vol
        scale = min(target_vol / spy_vol, 1.0) if target_vol and spy_vol > 1e-6 else 1.0

        set_date_context(date)
        D = date
        lgbm_sc = (lgbm_cache[D](feats) if D in lgbm_cache else composite_v1(feats)).dropna()
        lgbm_sc = lgbm_sc[~lgbm_sc.index.isin(EXCLUDE)]

        # 3-way blend: lgbm×0.70 + sh12×0.20 + sh5y×0.10
        idx = lgbm_sc.index
        blended = znorm(lgbm_sc) * 0.70
        if "sharpe_12m" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "sharpe_12m"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * 0.20
        if "sharpe_5y" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "sharpe_5y"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * 0.10

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

        common = [t for t in tickers if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        # Portfolio weights
        vols_raw = []
        for t in common:
            if t in feats.index and "vol_12m" in feats.columns:
                v = feats.loc[t, "vol_12m"]
                vols_raw.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
            else:
                vols_raw.append(0.20)
        vols_arr = np.array(vols_raw)

        if weighting == "inv_vol":
            raw_w = 1.0 / vols_arr
        elif weighting == "inv_vol2":
            raw_w = 1.0 / (vols_arr ** 2)
        elif weighting == "erc":
            raw_w = erc_weights(common, date)
            if raw_w.sum() <= 0:
                raw_w = 1.0 / vols_arr
        elif weighting == "inv_vol_capped":
            raw_w = 1.0 / vols_arr
        else:
            raw_w = np.ones(len(common))

        # Cap max weight
        if max_weight_cap is not None and weighting != "erc":
            raw_w = raw_w / raw_w.sum()  # normalize first
            # Iterative capping
            for _ in range(20):
                mask = raw_w > max_weight_cap
                if not mask.any():
                    break
                excess = (raw_w[mask] - max_weight_cap).sum()
                raw_w[mask] = max_weight_cap
                free = ~mask
                if free.sum() > 0:
                    raw_w[free] += excess / free.sum()

        weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(len(common)) / len(common)

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
        "name": name, "top_k": top_k, "weighting": weighting,
        "cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 3),
        "max_dd": round(float(m["max_dd"]), 4), "win_rate": round(float(m["win_rate"]), 3),
        "ann_vol": round(float(m["ann_vol"]), 4),
        "n_months": int(m["n_months"]), "cash_months": cash_m,
        "mean_m": round(float(m["mean_m"]), 5), "std_m": round(float(m["std_m"]), 5),
        "avg_scale": round(avg_scale, 3),
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:75s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# Reference: best inv_vol from exp_009
# ---------------------------------------------------------------------------
print("\n--- Reference (inv_vol, best blend) ---")
for k in [25, 30, 35, 40]:
    r = run_custom_weighting(
        f"inv_vol K={k:3d} 3way_blend + vt18% + loose [REF]",
        k, "inv_vol", target_vol=0.18)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 1. Double inv_vol (1/σ²)
# ---------------------------------------------------------------------------
print("\n--- Double inv_vol (1/σ²) ---")
for k in [25, 30, 35, 40, 50]:
    r = run_custom_weighting(
        f"inv_vol² K={k:3d} 3way_blend + vt18% + loose",
        k, "inv_vol2", target_vol=0.18)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. ERC weighting
# ---------------------------------------------------------------------------
print("\n--- ERC (equal risk contribution) weighting ---")
for k in [20, 25, 30, 40]:
    r = run_custom_weighting(
        f"ERC K={k:3d} 3way_blend + vt18% + loose",
        k, "erc", target_vol=0.18)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Inv_vol with max-weight cap
# ---------------------------------------------------------------------------
print("\n--- Inv_vol with max-weight cap ---")
for cap in [0.05, 0.07, 0.10]:
    for k in [25, 30, 40]:
        r = run_custom_weighting(
            f"inv_vol cap={cap:.0%} K={k:3d} 3way_blend + vt18% + loose",
            k, "inv_vol_capped", max_weight_cap=cap, target_vol=0.18)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. ERC with max-weight cap
# ---------------------------------------------------------------------------
print("\n--- ERC with max-weight cap ---")
for cap in [0.07, 0.10]:
    for k in [25, 30]:
        r = run_custom_weighting(
            f"ERC cap={cap:.0%} K={k:3d} 3way_blend + vt18% + loose",
            k, "erc", max_weight_cap=cap, target_vol=0.18)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Best weighting + different vol targets
# ---------------------------------------------------------------------------
print("\n--- Best weighting (ERC + inv_vol²) with vol target sweep ---")
for w_type in ["erc", "inv_vol2"]:
    for vt in [0.15, 0.18]:
        r = run_custom_weighting(
            f"{w_type} K= 30 3way_blend + vt={vt:.0%} + loose",
            30, w_type, target_vol=vt)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
passing = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]

print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio",
              "weighting", "top_k"]].head(20).to_string(index=False))

out = Path(__file__).parent / "exp_012_results.csv"
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
    best_cagr = df_res[df_res["cagr"] >= 0.50]
    print(f"\nBest with CAGR≥50%: {best_cagr.iloc[0]['name']} → Sharpe={best_cagr.iloc[0]['sharpe']:.2f}" if len(best_cagr) else "\nNo config passes CAGR≥50%")

print(f"\nTotal: {len(df_res)} configs, hypotheses: 346+{len(df_res)}={346+len(df_res)}")
