"""
Diagnostic analyses for understanding PIT SP500 alpha ceiling.

Produces:
- summary_oracle.json:     perfect-foresight upper bound (best 30 stocks each month)
- summary_baselines.json:  naive ladder on PIT (random / mom / sharpe_12m / quality / mom_x_lowvol)
- summary_ksweep.json:     K=3,5,10,20,30,50,100 with the 4T9wE blend
- summary_picks_diag.json: 4T9wE PIT picks diagnostics (top-1 contribution, concentration)
- summary_decomposition.json: month-by-month attribution

Reads the PIT SP500 augmented panel from PR #177's branch (read-only).
Same regime + vol target + cost stack as the 4T9wE config so numbers are
comparable to the validation report.
"""
from __future__ import annotations
import io, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Shared config (mirrors research/validation/sp500_pit/run_pit_sp500_validation.py)
# ---------------------------------------------------------------------------
TOP_K_DEFAULT = 30
LGBM_W, SH12_W, SH5Y_W = 0.70, 0.20, 0.10
WEIGHT_CAP = 0.05
TARGET_VOL = 0.18
REGIME_THRESHOLD = -0.05
COST_BPS = 5.0
TRAIN_MONTHS = 48
EMBARGO_MONTHS = 3
MIN_TRAIN_MONTHS = 24
OOS_START = pd.Timestamp("2007-01-31")
OOS_END = pd.Timestamp("2024-04-30")  # full PIT window (before lockbox)

LGB_PARAMS = dict(
    objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=200,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_jobs=-1,
)
FEATURE_COLS = [
    "mom_12_1","mom_6_1","mom_3","mom_3y","mom_5y","mom_2y",
    "vol_1y","vol_12m","vol_3m","vol_6m",
    "sharpe_1y","sharpe_12m","sharpe_5y",
    "trend_health_5y","trend_r2_12m","trend_slope_252",
    "frac_above_50dma_1y","mom_consistency_12m",
    "d_sma200","d_sma50","sma50_above_200",
    "rsi_14","rsi_zone_score",
    "dd_from_52wh","below_52wh","near_52wh_60d",
    "recovery_rate","beta_2y","max_dd_5y",
    "rs_3m_spy","rs_6m_spy","rs_12m_spy","mom_accel","accel",
    "idio_mom_12_1","mom_per_unit_vol_12",
    "fip_score","breakout_strength_60",
    "tight_consolidation_60","vol_asym_60","vol_asym_126",
    "crt_6m","crt_3m","rbi_60","rbi_120","prerunner_dist",
    "range_pos_1y","bb_width_pct","multibagger_ratio_24m","quality_score_5y",
    "dist_from_low_1y","drawdown_age_days","excess_5y_logret","acceleration_2y",
]

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE
DATA_BRANCH = "origin/claude/search-sp500-dataset-KmGNG"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _blob(branch, path):
    sha = subprocess.check_output(["git","rev-parse",f"{branch}:{path}"]).decode().strip()
    return subprocess.check_output(["git","cat-file","-p",sha])


def load_panel():
    p = pd.read_parquet(io.BytesIO(_blob(DATA_BRANCH,
        "experiments/monthly_dca/cache/v2/sp500_pit/augmented/sp500_pit_panel.parquet")))
    p["asof"] = pd.to_datetime(p["asof"])
    return p


def load_daily():
    d = pd.read_parquet(io.BytesIO(_blob(DATA_BRANCH,
        "experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet")))
    d.index = pd.to_datetime(d.index)
    return d


def load_membership():
    m = pd.read_parquet(io.BytesIO(_blob("origin/main",
        "experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet")))
    m["asof"] = pd.to_datetime(m["asof"])
    return m


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def znorm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sigma = s.mean(), s.std()
    if not np.isfinite(sigma) or sigma < 1e-10:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def iterative_weight_cap(w, cap):
    w = w / w.sum()
    for _ in range(50):
        m = w > cap
        if not m.any(): break
        excess = (w[m] - cap).sum()
        w[m] = cap
        free = ~m
        if not free.any(): break
        w[free] += excess / free.sum()
        w[free] = np.maximum(w[free], 0)
    return w / w.sum()


def metrics(rets):
    rets = rets.dropna()
    if len(rets) < 6: return {}
    cagr = (1+rets).prod()**(12/len(rets)) - 1
    std = rets.std()
    sharpe = (rets.mean()/std)*np.sqrt(12) if std > 0 else 0.0
    cum = (1+rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return dict(
        cagr=float(cagr), ann_vol=float(std*np.sqrt(12)),
        sharpe=float(sharpe), max_dd=float(dd.min()),
        win_rate=float((rets>0).mean()),
        n_months=int(len(rets)),
        mean_m=float(rets.mean()), std_m=float(std),
        ratio=float(rets.mean()/std) if std>0 else 0.0,
    )


def spy_regime(daily):
    spy = daily["SPY"].dropna()
    sma200 = spy.rolling(200, min_periods=100).mean()
    d_sma200 = (spy - sma200) / sma200
    vol_21 = spy.pct_change().rolling(21).std() * np.sqrt(252)
    return pd.DataFrame({"d_sma200": d_sma200, "vol_21d": vol_21}).resample("ME").last()


# ---------------------------------------------------------------------------
# Backtest core (shared)
# ---------------------------------------------------------------------------
def run_with_score_fn(panel, daily, membership, oos_start, oos_end, score_fn,
                      top_k=TOP_K_DEFAULT, use_regime=True, use_vol_target=True,
                      weighting="invvol_cap5", label="?"):
    """
    score_fn: (snap_df) -> Series indexed by ticker
    weighting: 'invvol_cap5', 'invvol', 'ew'
    """
    monthly_px = daily.resample("ME").last().ffill(limit=5)
    sreg = spy_regime(daily)
    mem_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}
    all_dates = sorted(panel["asof"].unique())
    rebalance = [d for d in all_dates if oos_start <= d <= oos_end]

    rows = []
    for i, date in enumerate(rebalance):
        idx_all = all_dates.index(date)
        if idx_all == len(all_dates) - 1: break

        snap = panel[panel["asof"] == date].copy()
        snap = snap[snap["ticker"].apply(lambda t: (date,t) in mem_set)]
        if snap.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_mem"))
            continue

        # Regime gate
        if use_regime:
            reg = sreg.reindex([date]).iloc[0]
            d_sma200 = reg["d_sma200"]
            spy_v = reg["vol_21d"]
            if not (np.isfinite(d_sma200) and d_sma200 > REGIME_THRESHOLD):
                rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="regime"))
                continue
        else:
            reg = sreg.reindex([date]).iloc[0]
            spy_v = reg["vol_21d"]

        sc = score_fn(snap)
        if sc is None or sc.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_score"))
            continue
        sc = sc.dropna()
        if sc.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_score"))
            continue

        top = sc.sort_values(ascending=False).head(top_k)
        tickers = top.index.tolist()

        next_date = all_dates[idx_all+1]
        d0 = monthly_px.index[monthly_px.index.searchsorted(date, side="right") - 1]
        d1 = monthly_px.index[monthly_px.index.searchsorted(next_date, side="right") - 1]
        p0, p1 = monthly_px.loc[d0], monthly_px.loc[d1]
        common = [t for t in tickers if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t,0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t,0) >= 1.0]
        if not common:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_common"))
            continue

        # Weights
        vol_src = "vol_12m" if "vol_12m" in snap.columns else "vol_1y"
        vmap = dict(zip(snap["ticker"].values, snap[vol_src].values))
        vols = np.array([max(float(vmap.get(t,0.2)),0.05)
                         if np.isfinite(vmap.get(t,np.nan)) else 0.2
                         for t in common])
        if weighting == "invvol_cap5":
            w = iterative_weight_cap(1.0/vols, WEIGHT_CAP)
        elif weighting == "invvol":
            inv = 1.0/vols
            w = inv/inv.sum()
        else:  # ew
            w = np.ones(len(common))/len(common)

        rets = np.array([(p1[t]-p0[t])/p0[t] for t in common])
        raw_port = float((w*rets).sum())

        scale = 1.0
        if use_vol_target and np.isfinite(spy_v) and spy_v > 1e-6:
            scale = min(TARGET_VOL/spy_v, 1.0)
        cost = COST_BPS/10_000.0
        port = scale*raw_port - 2*cost*scale

        rows.append(dict(date=date, ret_m=port, n_picks=len(common),
                         scale=scale, picks=",".join(common),
                         pick_ret_min=float(rets.min()), pick_ret_max=float(rets.max()),
                         pick_ret_mean=float(rets.mean()),
                         top_w_pick=common[int(np.argmax(w))], top_w=float(w.max()),
                         reason="ok"))

    df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return df, metrics(df["ret_m"]) if not df.empty else {}


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------
def score_random(seed=42):
    rng = np.random.default_rng(seed)
    def fn(snap):
        return pd.Series(rng.uniform(0,1,len(snap)),
                         index=snap["ticker"].values)
    return fn

def score_col(col, sign=1):
    def fn(snap):
        if col not in snap.columns:
            return pd.Series(dtype=float)
        return pd.Series(sign*snap[col].values, index=snap["ticker"].values).dropna()
    return fn

def score_mom_x_lowvol():
    """12-1 momentum restricted to bottom-2/3 vol_12m."""
    def fn(snap):
        m = pd.Series(snap["mom_12_1"].values, index=snap["ticker"].values)
        v = pd.Series(snap["vol_12m"].values, index=snap["ticker"].values)
        cut = v.quantile(2/3)
        m[v > cut] = np.nan
        return m.dropna()
    return fn

def score_quality_x_mom():
    """rank-combine: sharpe_5y + trend_health_5y - vol_1y + mom_12_1."""
    def fn(snap):
        s = (znorm(pd.Series(snap["sharpe_5y"].values, index=snap["ticker"].values))
             + znorm(pd.Series(snap["trend_health_5y"].values, index=snap["ticker"].values))
             - znorm(pd.Series(snap["vol_1y"].values, index=snap["ticker"].values))
             + znorm(pd.Series(snap["mom_12_1"].values, index=snap["ticker"].values)))
        return s.dropna()
    return fn

def score_oracle():
    """Score = fwd_1m_ret (perfect foresight). Upper bound only."""
    def fn(snap):
        if "fwd_1m_ret" not in snap.columns: return pd.Series(dtype=float)
        return pd.Series(snap["fwd_1m_ret"].values, index=snap["ticker"].values).dropna()
    return fn


# ---------------------------------------------------------------------------
# Walk-forward LGBM (for 4T9wE blend)
# ---------------------------------------------------------------------------
def fit_lgbm(panel, train_dates, feat_cols):
    sub = panel[panel["asof"].isin(train_dates)].copy().dropna(subset=["fwd_1m_ret"])
    if sub.empty: return None
    X = sub[feat_cols].copy().fillna(sub[feat_cols].median(numeric_only=True))
    y = sub["fwd_1m_ret"].values
    m = lgb.LGBMRegressor(**LGB_PARAMS); m.fit(X,y)
    return m

def build_lgbm_cache(panel, oos_start, oos_end):
    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    all_dates = sorted(panel["asof"].unique())
    fwd_dates = sorted(panel[panel["fwd_1m_ret"].notna()]["asof"].unique())
    cache = {}
    for i, date in enumerate(all_dates):
        if date < oos_start: continue
        if date > oos_end: break
        cutoff_idx = i - EMBARGO_MONTHS
        if cutoff_idx < 0: continue
        train_end = all_dates[cutoff_idx]
        pool = [d for d in fwd_dates if d <= train_end and d < date]
        if len(pool) < MIN_TRAIN_MONTHS: continue
        td = pool[-TRAIN_MONTHS:]
        key = (td[0], td[-1])
        if key in cache:
            cache[date] = cache[key]
            continue
        m = fit_lgbm(panel, td, feat_cols)
        cache[key] = m
        cache[date] = m
    return cache, feat_cols

def score_4t9we_blend(panel, oos_start, oos_end):
    cache, feat_cols = build_lgbm_cache(panel, oos_start, oos_end)
    def fn(snap):
        d = snap["asof"].iloc[0] if len(snap) else None
        m = cache.get(d)
        if m is None: return pd.Series(dtype=float)
        X = snap[feat_cols].copy().fillna(snap[feat_cols].median(numeric_only=True))
        pred = m.predict(X)
        lz = znorm(pd.Series(pred, index=snap["ticker"].values))
        s12 = znorm(pd.Series(snap.get("sharpe_12m", np.zeros(len(snap))),
                              index=snap["ticker"].values))
        s5  = znorm(pd.Series(snap.get("sharpe_5y", np.zeros(len(snap))),
                              index=snap["ticker"].values))
        return (LGBM_W*lz + SH12_W*s12 + SH5Y_W*s5).dropna()
    return fn


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Loading PIT SP500 augmented panel + prices + membership ...")
    panel = load_panel(); daily = load_daily(); mem = load_membership()
    print(f"  panel {panel.shape}  prices {daily.shape}  members {mem.shape}")

    OUT_DIR.mkdir(exist_ok=True)
    summary = {}

    # ---- 1. Oracle K=30 (perfect foresight) ----
    print("\n[1/4] Oracle K=30 (perfect foresight ceiling)")
    for use_reg, use_vt, lab in [(False, False, "no_reg_no_vt"),
                                  (True,  False, "reg_only"),
                                  (True,  True,  "reg_vt18")]:
        df, m = run_with_score_fn(panel, daily, mem, OOS_START, OOS_END,
                                  score_oracle(), top_k=30,
                                  use_regime=use_reg, use_vol_target=use_vt,
                                  weighting="invvol_cap5", label=f"oracle_{lab}")
        if m:
            print(f"  oracle_{lab:<18}  CAGR={m['cagr']:>7.2%}  Sharpe={m['sharpe']:>5.2f}  "
                  f"MaxDD={m['max_dd']:>6.1%}  AnnVol={m['ann_vol']:>5.1%}  N={m['n_months']}")
            summary[f"oracle_{lab}"] = m
    json.dump(summary, open(OUT_DIR/"summary_oracle.json","w"), indent=2, default=str)

    # ---- 2. Baseline ladder ----
    print("\n[2/4] Baseline ladder (K=30, regime+vt18 stack)")
    baselines = [
        ("random_seed42",       score_random(42)),
        ("mom_12_1",            score_col("mom_12_1")),
        ("mom_6_1",             score_col("mom_6_1")),
        ("sharpe_12m",          score_col("sharpe_12m")),
        ("sharpe_5y",           score_col("sharpe_5y")),
        ("trend_health_5y",     score_col("trend_health_5y")),
        ("quality_score_5y",    score_col("quality_score_5y")),
        ("low_vol",             score_col("vol_1y", sign=-1)),
        ("mom_x_lowvol",        score_mom_x_lowvol()),
        ("quality_x_mom",       score_quality_x_mom()),
    ]
    baseline_summary = {}
    for name, sfn in baselines:
        df, m = run_with_score_fn(panel, daily, mem, OOS_START, OOS_END,
                                  sfn, top_k=30, weighting="invvol_cap5",
                                  label=name)
        if m:
            print(f"  {name:<22}  CAGR={m['cagr']:>7.2%}  Sharpe={m['sharpe']:>5.2f}  "
                  f"MaxDD={m['max_dd']:>6.1%}  AnnVol={m['ann_vol']:>5.1%}  "
                  f"ratio={m['ratio']:.3f}")
            baseline_summary[name] = m
    json.dump(baseline_summary, open(OUT_DIR/"summary_baselines.json","w"),
              indent=2, default=str)

    # ---- 3. K sweep with 4T9wE blend ----
    print("\n[3/4] K-sweep with 4T9wE blend (this is the slow one — ~5 min)")
    blend_score = score_4t9we_blend(panel, OOS_START, OOS_END)
    ksweep = {}
    for k in [3, 5, 10, 20, 30, 50, 100]:
        df, m = run_with_score_fn(panel, daily, mem, OOS_START, OOS_END,
                                  blend_score, top_k=k,
                                  weighting="invvol_cap5", label=f"k={k}")
        if m:
            print(f"  K={k:<3}  CAGR={m['cagr']:>7.2%}  Sharpe={m['sharpe']:>5.2f}  "
                  f"MaxDD={m['max_dd']:>6.1%}  AnnVol={m['ann_vol']:>5.1%}  "
                  f"ratio={m['ratio']:.3f}")
            ksweep[f"k_{k}"] = m
    json.dump(ksweep, open(OUT_DIR/"summary_ksweep.json","w"), indent=2, default=str)

    # ---- 4. Pick concentration / per-month picks diagnostics ----
    print("\n[4/4] 4T9wE blend K=30 picks diagnostics")
    df, m = run_with_score_fn(panel, daily, mem, OOS_START, OOS_END,
                              blend_score, top_k=30, weighting="invvol_cap5",
                              label="diag")
    picks_diag = {
        "headline": m,
        "n_cash_months": int((df["ret_m"] == 0.0).sum()) if "ret_m" in df.columns else None,
        "n_total_months": int(len(df)),
        "pick_ret_distribution": (
            {"mean_pick_mean": float(df["pick_ret_mean"].dropna().mean()),
             "mean_pick_min":  float(df["pick_ret_min"].dropna().mean()),
             "mean_pick_max":  float(df["pick_ret_max"].dropna().mean()),
             "median_pick_mean": float(df["pick_ret_mean"].dropna().median())}
            if "pick_ret_mean" in df.columns else {}
        ),
        "scale_distribution": (
            {"mean": float(df["scale"].dropna().mean()),
             "median": float(df["scale"].dropna().median()),
             "min": float(df["scale"].dropna().min()),
             "max": float(df["scale"].dropna().max())}
            if "scale" in df.columns else {}
        ),
    }
    json.dump(picks_diag, open(OUT_DIR/"summary_picks_diag.json","w"),
              indent=2, default=str)
    df.to_csv(OUT_DIR/"backtest_diag.csv")

    print("\nDone. Output files:")
    for f in sorted(OUT_DIR.glob("summary_*.json")):
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
