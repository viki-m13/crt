"""
Experiment 010: LGBM with Sharpe-Adjusted Training Target

Current bottleneck: mean_m/std_m ratio = 0.527, need 0.577.
The LGBM is trained to predict rank of raw 1m return.

New hypothesis: if we train LGBM to predict rank of risk-adjusted return
(fwd_ret_1m / vol_12m), the model will select stocks that provide better
risk-adjusted momentum. This should improve portfolio Sharpe directly.

Implementation: subclass WalkForwardLGBM, override _fit_model to use
risk-adjusted target: rank(fwd_ret_1m / vol_12m) within each date.

Prior best: lgbm×0.70+sh12×0.20+sh5y×0.10 K=30 → CAGR=63.1%, Sharpe=1.82
"""
import sys, time, warnings, hashlib, pickle
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest.engine import (
    make_regime_fn, get_feat_dates, get_prices, get_monthly_prices,
    load_features, compute_metrics, get_spy_stats_at, EXCLUDE
)
from models.lgbm_ranker import WalkForwardLGBM, FEATURE_COLS, CACHE_DIR
from features.signals import composite_v1, set_date_context

print("=" * 90)
print("EXPERIMENT 010: LGBM WITH SHARPE-ADJUSTED TRAINING TARGET")
print("=" * 90)


class WalkForwardLGBMSharpeTarget(WalkForwardLGBM):
    """LGBM trained to predict rank(fwd_ret / vol) instead of rank(fwd_ret)."""

    def _get_cache_key(self, params_str: str) -> str:
        return hashlib.md5(f"sharpe_{params_str}".encode()).hexdigest()[:12]

    def _fit_model(self, train_dates: list) -> tuple:
        if self._panel is None or self._fwd_returns is None:
            return None, []

        train_date_set = set(train_dates)
        X_full = self._panel[self._panel["asof"].isin(train_date_set)].copy()
        if X_full.empty:
            return None, []

        fwd_train = self._fwd_returns[self._fwd_returns["asof"].isin(train_date_set)].copy()
        if fwd_train.empty:
            return None, []

        fwd_train_reset = fwd_train.reset_index()
        if "ticker" not in fwd_train_reset.columns:
            fwd_train_reset = fwd_train_reset.rename(columns={"index": "ticker"})
        fwd_train_reset = fwd_train_reset[["ticker", "asof", "fwd_ret_1m"]]
        fwd_train_reset["ticker"] = fwd_train_reset["ticker"].astype(str)

        merged = X_full.merge(fwd_train_reset, on=["ticker", "asof"], how="inner")

        available_cols = [c for c in FEATURE_COLS if c in merged.columns]
        if not available_cols:
            return None, []

        X_data = merged[available_cols].copy()
        y = merged["fwd_ret_1m"]
        valid = y.notna() & (y.abs() < 3.0)
        if valid.sum() < 200:
            return None, []

        X_train = X_data[valid].fillna(X_data[valid].median())
        y_train = y[valid]
        merged_valid = merged[valid].copy()

        # Risk-adjusted target: rank(fwd_ret / vol_12m) within each date
        y_ranked = y_train.copy()
        for date_val, grp in merged_valid.groupby("asof"):
            idx = grp.index
            if len(idx) < 5:
                continue
            raw_ret = y_train.loc[idx]
            if "vol_12m" in merged_valid.columns:
                vol = merged_valid.loc[idx, "vol_12m"].clip(lower=0.05)
                sharpe_adj = raw_ret / vol
            else:
                sharpe_adj = raw_ret
            y_ranked.loc[idx] = sharpe_adj.rank(pct=True)

        model = lgb.LGBMRegressor(**self.lgb_params)
        try:
            model.fit(X_train, y_ranked)
        except Exception as e:
            print(f"    Fit error: {e}")
            return None, []

        return model, available_cols


print("\nLoading data and training Sharpe-target LGBM models...")
dates = get_feat_dates()
dates_all = [d for d in dates if d >= pd.Timestamp('2003-01-01')]
prices = get_prices()
monthly_px = get_monthly_prices()

wf_std = WalkForwardLGBM(train_months=48, embargo_months=3, min_train_months=24)
wf_std.prepare_data(prices, dates_all)
std_cache = {d: fn for d in dates_all
             if (fn := wf_std.get_score_fn(d, dates_all)) is not None}
print(f"Standard LGBM ready for {len(std_cache)} dates")

wf_sh = WalkForwardLGBMSharpeTarget(train_months=48, embargo_months=3, min_train_months=24)
wf_sh.prepare_data(prices, dates_all)  # reuses panel/fwd from cache
sh_cache = {d: fn for d in dates_all
            if (fn := wf_sh.get_score_fn(d, dates_all)) is not None}
print(f"Sharpe-target LGBM ready for {len(sh_cache)} dates")

OOS_START = "2007-01-31"
OOS_END = "2021-12-31"
COST = 5.0 / 10_000.0


def run_backtest_blend(
    name: str,
    top_k: int,
    use_sharpe_lgbm: bool = False,
    sharpe12_wt: float = 0.20,
    sharpe5y_wt: float = 0.10,
    lgbm_wt: float = 0.70,
    d200_thresh: float = None,
    rs_thresh: float = None,
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

    score_cache = sh_cache if use_sharpe_lgbm else std_cache

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
        scale = min(target_vol / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0

        set_date_context(date)
        D = date
        raw_scores = (score_cache[D](feats) if D in score_cache
                      else composite_v1(feats)).dropna()
        raw_scores = raw_scores[~raw_scores.index.isin(EXCLUDE)]
        if raw_scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        idx = raw_scores.index
        blended = znorm(raw_scores) * lgbm_wt

        if sharpe12_wt > 0 and "sharpe_12m" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "sharpe_12m"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * sharpe12_wt

        if sharpe5y_wt > 0 and "sharpe_5y" in feats.columns:
            s = feats.loc[feats.index.isin(idx), "sharpe_5y"].reindex(idx).fillna(0.0)
            blended = blended + znorm(s) * sharpe5y_wt

        if d200_thresh is not None and "d_sma200" in feats.columns:
            d_sma = feats.loc[feats.index.isin(blended.index), "d_sma200"].reindex(blended.index)
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
        "ratio": round(float(m["mean_m"]) / float(m["std_m"]), 4) if float(m["std_m"]) > 0 else 0,
    }
    gc_c = "✓" if res["cagr"] >= 0.50 else "✗"
    gs = "✓" if res["sharpe"] >= 2.0 else "✗"
    print(f"  {name:75s} CAGR={res['cagr']:.1%}{gc_c} Sharpe={res['sharpe']:.2f}{gs} "
          f"MaxDD={res['max_dd']:.1%} Vol={res['ann_vol']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
    return res


RESULTS = []

# ---------------------------------------------------------------------------
# 1. Direct comparison: standard LGBM vs Sharpe-target LGBM (no blending)
# ---------------------------------------------------------------------------
print("\n--- Direct comparison: standard vs Sharpe-target LGBM ---")
for use_sh, label in [(False, "STANDARD"), (True, "SHARPE_TARGET")]:
    for k in [30, 40, 50]:
        r = run_backtest_blend(
            f"{label} lgbm K={k:3d} inv_vol + vt18% + loose",
            k, use_sharpe_lgbm=use_sh,
            sharpe12_wt=0.0, sharpe5y_wt=0.0, lgbm_wt=1.0)
        if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 2. Sharpe-target LGBM with best blend from exp_009
# ---------------------------------------------------------------------------
print("\n--- Sharpe-target LGBM + best blend (0.70/0.20/0.10) ---")
for k in [20, 30, 40, 50, 60]:
    r = run_backtest_blend(
        f"SH_LGBM×0.70+sh12×0.20+sh5y×0.10 K={k:3d} + vt18% + loose",
        k, use_sharpe_lgbm=True,
        sharpe12_wt=0.20, sharpe5y_wt=0.10, lgbm_wt=0.70)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 3. Sharpe-target + 2-way blend sweep at K=40
# ---------------------------------------------------------------------------
print("\n--- Sharpe-target LGBM + sh12 blend sweep at K=40 ---")
for lgbm_wt, sh12_wt in [(1.0, 0.0), (0.85, 0.15), (0.70, 0.30), (0.60, 0.40), (0.50, 0.50)]:
    r = run_backtest_blend(
        f"SH_LGBM×{lgbm_wt:.2f}+sh12×{sh12_wt:.2f} K= 40 + vt18% + loose",
        40, use_sharpe_lgbm=True,
        sharpe12_wt=sh12_wt, sharpe5y_wt=0.0, lgbm_wt=lgbm_wt)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 4. Best Sharpe-target combo + filters
# ---------------------------------------------------------------------------
print("\n--- Sharpe-target best combo + d200 + rs filters ---")
for k in [30, 40, 50]:
    r = run_backtest_blend(
        f"SH_LGBM blend K={k:3d} + d200+rs + vt18% + loose",
        k, use_sharpe_lgbm=True,
        sharpe12_wt=0.20, sharpe5y_wt=0.10, lgbm_wt=0.70,
        d200_thresh=0.0, rs_thresh=0.0)
    if r: RESULTS.append(r)

# ---------------------------------------------------------------------------
# 5. Ensemble: Sharpe-target + Standard LGBM blend
# ---------------------------------------------------------------------------
print("\n--- Ensemble: standard LGBM + Sharpe-target LGBM ---")
for sh_weight in [0.3, 0.5, 0.7]:
    std_w = 1.0 - sh_weight

    feat_dates = get_feat_dates()
    dates = [d for d in feat_dates if pd.Timestamp(OOS_START) <= d <= pd.Timestamp(OOS_END)]
    regime_fn = make_regime_fn("200ma_loose")
    t0 = time.time()
    records = []

    def znorm(s):
        mu, si = s.mean(), s.std()
        return (s - mu) / si if si > 1e-10 else pd.Series(0.0, index=s.index)

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
        spy_vol = stats.get("vol_21d", 0.18) if stats else 0.18
        scale = min(0.18 / spy_vol, 1.0) if spy_vol > 1e-6 else 1.0

        set_date_context(date)
        D = date
        std_sc = std_cache.get(D, lambda f: composite_v1(f))(feats).dropna()
        sh_sc = sh_cache.get(D, lambda f: composite_v1(f))(feats).dropna()
        std_sc = std_sc[~std_sc.index.isin(EXCLUDE)]
        sh_sc = sh_sc[~sh_sc.index.isin(EXCLUDE)]

        common_idx = std_sc.index.intersection(sh_sc.index)
        if len(common_idx) < 10:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0, "scale": scale})
            continue

        blended = znorm(std_sc.loc[common_idx]) * std_w + znorm(sh_sc.loc[common_idx]) * sh_weight

        # Add sharpe12+sh5y blend
        if "sharpe_12m" in feats.columns:
            s = feats.loc[feats.index.isin(common_idx), "sharpe_12m"].reindex(common_idx).fillna(0.0)
            blended = blended + znorm(s) * 0.20
        if "sharpe_5y" in feats.columns:
            s = feats.loc[feats.index.isin(common_idx), "sharpe_5y"].reindex(common_idx).fillna(0.0)
            blended = blended + znorm(s) * 0.10

        top = blended.sort_values(ascending=False).head(40)
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

        vols = [max(float(feats.loc[t, "vol_12m"]) if t in feats.index and "vol_12m" in feats.columns
                    and np.isfinite(feats.loc[t, "vol_12m"]) else 0.20, 0.05) for t in common]
        inv_v = 1.0 / np.array(vols)
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
        name = f"ENSEMBLE std×{std_w:.1f}+sh_lgbm×{sh_weight:.1f}+sh12+sh5y K=40 vt18%"
        res = {
            "name": name, "top_k": 40,
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
              f"MaxDD={res['max_dd']:.1%} ratio={res['ratio']:.3f} {elapsed:.0f}s")
        RESULTS.append(res)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (sorted by Sharpe)")
print("=" * 90)
df_res = pd.DataFrame(RESULTS).sort_values("sharpe", ascending=False)
print(df_res[["name", "cagr", "sharpe", "max_dd", "ann_vol", "ratio",
              "top_k"]].to_string(index=False))

out = Path(__file__).parent / "exp_010_results.csv"
df_res.to_csv(out, index=False)
print(f"\nSaved: {out}")

best = df_res.iloc[0]
print(f"\nBest: {best['name']} → CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.2f}")
print(f"      ratio={best['ratio']:.4f} (target 0.5774)")

passed_both = df_res[(df_res["cagr"] >= 0.50) & (df_res["sharpe"] >= 2.0)]
if len(passed_both) > 0:
    print(f"\n{'='*40}")
    print(f"*** {len(passed_both)} CONFIGS PASS BOTH GATES ***")
    print(passed_both[["name", "cagr", "sharpe", "max_dd", "ann_vol"]].to_string(index=False))
    # Trigger success notify
    import subprocess
    best_pass = passed_both.iloc[0]
    subprocess.run([
        "/home/user/crt/quant_research/notify/send_success.sh",
        best_pass["name"],
        str(best_pass["cagr"]),
        str(best_pass["sharpe"]),
        str(best_pass["max_dd"])
    ])
else:
    best_sharpe = df_res.iloc[0]["sharpe"]
    best_with_cagr = df_res[df_res["cagr"] >= 0.50]
    print(f"\nBest Sharpe: {best_sharpe:.2f}")
    if len(best_with_cagr):
        r0 = best_with_cagr.iloc[0]
        print(f"Best with CAGR≥50%: {r0['name']} → Sharpe={r0['sharpe']:.2f}, ratio={r0['ratio']:.4f}")

print(f"\nTotal configs: {len(df_res)}")
print(f"Running total hypotheses: 305 (prior) + {len(df_res)} = {305 + len(df_res)}")
