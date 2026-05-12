"""
PIT SP500 validation of 4T9wE's winning config (exp_012).

Spec replayed verbatim from research/runs/4T9wE/STATE.md:
- K=30, monthly rebalance, hold 1m
- Score = z(LGBM) * 0.70 + z(sharpe_12m) * 0.20 + z(sharpe_5y) * 0.10
- LGBM: WalkForwardLGBM(train=48m, embargo=3m, min_train=24m), 200 trees / 31 leaves
- Weighting: inv-vol on vol_12m (or vol_1y if vol_12m absent), capped at 5% per name
- Vol-target: scale = min(0.18 / spy_vol_21d, 1.0); applied to net portfolio return
- Regime gate: 200ma_loose -> invest iff d_sma200(SPY) > -0.05
- Cost: 5 bps round-trip * 2 = 10 bps per rebalance

Data sources:
- Augmented PIT SP500 panel (from PR #177): per-(asof, ticker) features + fwd returns
- Augmented PIT SP500 daily prices: monthly resample for portfolio returns + SPY for regime
- sp500_membership_monthly.parquet: PIT membership filter at each rebalance

OOS window: 2007-01-31 -> 2024-04-30 (last asof in panel that has fwd_1m_ret).
Lockbox 2024-05 onward is NOT touched.

No retraining on OOS, no parameter tuning, no cherry-picking.
"""
from __future__ import annotations
import io, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Config — replayed verbatim from 4T9wE
# ---------------------------------------------------------------------------
TOP_K = 30
LGBM_W, SH12_W, SH5Y_W = 0.70, 0.20, 0.10
WEIGHT_CAP = 0.05            # 5% per name
TARGET_VOL = 0.18            # 18% ann
REGIME_THRESHOLD = -0.05     # 200ma_loose
COST_BPS = 5.0
TRAIN_MONTHS = 48
EMBARGO_MONTHS = 3
MIN_TRAIN_MONTHS = 24

OOS_START = pd.Timestamp("2007-01-31")
# OOS ends at last asof where fwd_1m_ret is valid AND before lockbox (2024-01).
# 4T9wE quotes OOS through 2021-12-31; we use the same window to make the
# comparison apples-to-apples. We separately report an "extended" window
# through 2024-04 to show 2022-2024 performance.
OOS_END_PRIMARY = pd.Timestamp("2021-12-31")
OOS_END_EXTENDED = pd.Timestamp("2024-04-30")

LGB_PARAMS = dict(
    objective="regression",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1,
    n_jobs=-1,
)

# Feature columns — restricted to the 4T9wE FEATURE_COLS set, intersected with
# columns actually present in the augmented panel. Any missing cols are dropped
# (LGBM handles a different-but-non-empty feature set; we just need enough
# signal to mirror 4T9wE's training data shape as closely as possible).
FEATURE_COLS = [
    "mom_12_1", "mom_6_1", "mom_3", "mom_3y", "mom_5y", "mom_2y",
    "vol_1y", "vol_12m", "vol_3m", "vol_6m",
    "sharpe_1y", "sharpe_12m", "sharpe_5y",
    "trend_health_5y", "trend_r2_12m", "trend_slope_252",
    "frac_above_50dma_1y", "mom_consistency_12m",
    "d_sma200", "d_sma50", "sma50_above_200",
    "rsi_14", "rsi_zone_score",
    "dd_from_52wh", "below_52wh", "near_52wh_60d",
    "recovery_rate",
    "beta_2y", "max_dd_5y",
    "rs_3m_spy", "rs_6m_spy", "rs_12m_spy",
    "mom_accel", "accel",
    "idio_mom_12_1", "mom_per_unit_vol_12",
    "fip_score", "breakout_strength_60",
    "tight_consolidation_60", "vol_asym_60", "vol_asym_126",
    "crt_6m", "crt_3m", "rbi_60", "rbi_120",
    "prerunner_dist",
    "range_pos_1y", "bb_width_pct",
    "multibagger_ratio_24m", "quality_score_5y",
    "dist_from_low_1y", "drawdown_age_days",
    "excess_5y_logret", "acceleration_2y",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE
DATA_BRANCH = "origin/claude/search-sp500-dataset-KmGNG"
DATA_PATHS = {
    "panel":  "experiments/monthly_dca/cache/v2/sp500_pit/augmented/sp500_pit_panel.parquet",
    "prices": "experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet",
}
MAIN_BRANCH = "origin/main"
MAIN_PATHS = {
    "membership": "experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet",
}

def _blob(branch: str, path: str) -> bytes:
    sha = subprocess.check_output(["git", "rev-parse", f"{branch}:{path}"]).decode().strip()
    return subprocess.check_output(["git", "cat-file", "-p", sha])


def load_panel() -> pd.DataFrame:
    raw = _blob(DATA_BRANCH, DATA_PATHS["panel"])
    df = pd.read_parquet(io.BytesIO(raw))
    df["asof"] = pd.to_datetime(df["asof"])
    return df


def load_daily_prices() -> pd.DataFrame:
    raw = _blob(DATA_BRANCH, DATA_PATHS["prices"])
    df = pd.read_parquet(io.BytesIO(raw))
    df.index = pd.to_datetime(df.index)
    return df


def load_membership() -> pd.DataFrame:
    raw = _blob(MAIN_BRANCH, MAIN_PATHS["membership"])
    df = pd.read_parquet(io.BytesIO(raw))
    df["asof"] = pd.to_datetime(df["asof"])
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def znorm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sigma = s.mean(), s.std()
    if not np.isfinite(sigma) or sigma < 1e-10:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def iterative_weight_cap(w: np.ndarray, cap: float) -> np.ndarray:
    w = w / w.sum()
    for _ in range(50):
        mask = w > cap
        if not mask.any():
            break
        excess = (w[mask] - cap).sum()
        w[mask] = cap
        free = ~mask
        if free.sum() == 0:
            break
        w[free] += excess / free.sum()
        w[free] = np.maximum(w[free], 0)
    return w / w.sum()


def metrics(rets: pd.Series) -> dict:
    rets = rets.dropna()
    if len(rets) < 6:
        return {}
    cagr = (1 + rets).prod() ** (12 / len(rets)) - 1
    std = rets.std()
    sharpe = (rets.mean() / std) * np.sqrt(12) if std > 0 else 0.0
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return dict(
        cagr=float(cagr),
        ann_vol=float(std * np.sqrt(12)),
        sharpe=float(sharpe),
        max_dd=float(dd.min()),
        win_rate=float((rets > 0).mean()),
        n_months=int(len(rets)),
        mean_m=float(rets.mean()),
        std_m=float(std),
    )


# ---------------------------------------------------------------------------
# SPY regime + vol target from daily prices
# ---------------------------------------------------------------------------
def spy_regime_table(daily: pd.DataFrame) -> pd.DataFrame:
    if "SPY" not in daily.columns:
        raise ValueError("SPY missing from daily prices")
    spy = daily["SPY"].dropna()
    sma200 = spy.rolling(200, min_periods=100).mean()
    d_sma200 = (spy - sma200) / sma200
    vol_21 = spy.pct_change().rolling(21).std() * np.sqrt(252)
    out = pd.DataFrame({"d_sma200": d_sma200, "vol_21d": vol_21})
    return out.resample("ME").last()


# ---------------------------------------------------------------------------
# Walk-forward LGBM
# ---------------------------------------------------------------------------
def fit_lgbm(panel: pd.DataFrame, train_dates: list, feat_cols: list) -> tuple:
    sub = panel[panel["asof"].isin(train_dates)].copy()
    sub = sub.dropna(subset=["fwd_1m_ret"])
    if sub.empty:
        return None, []
    X = sub[feat_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = sub["fwd_1m_ret"].values
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model, feat_cols


def lgbm_score_for_date(model, feat_cols: list, snap: pd.DataFrame) -> pd.Series:
    if model is None or snap.empty:
        return pd.Series(dtype=float)
    X = snap[feat_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    pred = model.predict(X)
    return pd.Series(pred, index=snap["ticker"].values)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run_backtest(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    membership: pd.DataFrame,
    oos_end: pd.Timestamp,
    label: str = "primary",
) -> tuple[pd.DataFrame, dict]:
    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    print(f"  [{label}] features used: {len(feat_cols)}/{len(FEATURE_COLS)}")

    monthly_px = daily.resample("ME").last().ffill(limit=5)
    spy_reg = spy_regime_table(daily)

    all_dates = sorted(panel["asof"].unique())
    fwd_valid_dates = sorted(panel[panel["fwd_1m_ret"].notna()]["asof"].unique())

    membership_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}

    rebalance_dates = [d for d in all_dates
                       if pd.Timestamp("2003-01-31") <= d <= oos_end]

    records = []
    model_cache = {}

    for i, date in enumerate(rebalance_dates):
        idx = all_dates.index(date)
        if idx == len(all_dates) - 1:
            break

        # Walk-forward window: train on dates with valid fwd_1m_ret up to date - embargo
        cutoff_idx = idx - EMBARGO_MONTHS
        if cutoff_idx < 0:
            continue
        train_end_date = all_dates[cutoff_idx]
        train_dates_pool = [d for d in fwd_valid_dates if d <= train_end_date]
        if len(train_dates_pool) < MIN_TRAIN_MONTHS:
            # Not enough history -> skip (still record as cash month)
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="insufficient_train"))
            continue
        train_dates = train_dates_pool[-TRAIN_MONTHS:]
        # Only outside OOS window (no leakage)
        train_dates = [d for d in train_dates if d < date]

        # Skip months before OOS start in scoring (we still need to train them
        # in-sample for the WF model; rebalance only on OOS_START+)
        if date < OOS_START:
            continue

        key = (train_dates[0], train_dates[-1])
        if key in model_cache:
            model = model_cache[key]
        else:
            model, _ = fit_lgbm(panel, train_dates, feat_cols)
            model_cache[key] = model

        snap = panel[panel["asof"] == date].copy()
        if snap.empty:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="no_snapshot"))
            continue

        # PIT membership filter
        snap["in_sp500"] = snap["ticker"].apply(lambda t: (date, t) in membership_set)
        snap = snap[snap["in_sp500"]]
        if snap.empty:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="empty_after_membership"))
            continue

        # Regime gate
        reg = spy_reg.reindex([date]).iloc[0]
        d_sma200 = reg["d_sma200"]
        spy_vol = reg["vol_21d"]
        gate_pass = bool(np.isfinite(d_sma200) and d_sma200 > REGIME_THRESHOLD)
        if not gate_pass:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="regime"))
            continue

        # Score: z(LGBM) * 0.70 + z(sharpe_12m) * 0.20 + z(sharpe_5y) * 0.10
        lgbm_raw = lgbm_score_for_date(model, feat_cols, snap)
        lgbm_z = znorm(lgbm_raw)
        # sharpe_12m / sharpe_5y from the panel (already cross-sectional features)
        sh12 = pd.Series(snap["sharpe_12m"].values, index=snap["ticker"].values) \
            if "sharpe_12m" in snap.columns else pd.Series(0.0, index=snap["ticker"].values)
        sh5y = pd.Series(snap["sharpe_5y"].values, index=snap["ticker"].values) \
            if "sharpe_5y" in snap.columns else pd.Series(0.0, index=snap["ticker"].values)
        score = LGBM_W * lgbm_z + SH12_W * znorm(sh12) + SH5Y_W * znorm(sh5y)
        score = score.dropna()
        if score.empty:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=True, reason="empty_score"))
            continue

        top = score.sort_values(ascending=False).head(TOP_K)
        tickers = top.index.tolist()

        # Returns from monthly_px (point-in-time aware)
        next_date = all_dates[idx + 1]
        # Use the augmented monthly resample
        if date not in monthly_px.index or next_date not in monthly_px.index:
            # fall back to nearest <= date
            d0 = monthly_px.index[monthly_px.index.searchsorted(date, side="right") - 1]
            d1 = monthly_px.index[monthly_px.index.searchsorted(next_date, side="right") - 1]
        else:
            d0, d1 = date, next_date

        p0 = monthly_px.loc[d0]
        p1 = monthly_px.loc[d1]

        common = [t for t in tickers
                  if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=True, reason="no_common"))
            continue

        # Weighting: inv-vol on vol_12m (fall back to vol_1y), capped at 5%
        vol_src = "vol_12m" if "vol_12m" in snap.columns else "vol_1y"
        vols_map = dict(zip(snap["ticker"].values, snap[vol_src].values))
        vols = np.array([max(float(vols_map.get(t, 0.20)), 0.05)
                         if np.isfinite(vols_map.get(t, np.nan)) else 0.20
                         for t in common])
        raw_w = 1.0 / vols
        w = iterative_weight_cap(raw_w, WEIGHT_CAP)

        # Portfolio return
        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        raw_port = float((w * rets).sum())

        # Vol target
        scale = min(TARGET_VOL / spy_vol, 1.0) if np.isfinite(spy_vol) and spy_vol > 1e-6 else 1.0
        cost = COST_BPS / 10_000.0
        port = scale * raw_port - 2 * cost * scale

        records.append(dict(date=date, ret_m=port, n_picks=len(common), scale=scale,
                            gate_pass=True, reason="ok",
                            picks=",".join(common)))

    df = pd.DataFrame(records).set_index("date")
    rets = df.loc[(df.index >= OOS_START) & (df.index <= oos_end), "ret_m"]
    m = metrics(rets)
    return df, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading PIT SP500 augmented panel ...")
    t0 = time.time()
    panel = load_panel()
    print(f"  panel: {panel.shape}, {panel['asof'].nunique()} months, "
          f"{panel['ticker'].nunique()} tickers ({time.time()-t0:.1f}s)")

    print("Loading augmented daily prices ...")
    t0 = time.time()
    daily = load_daily_prices()
    print(f"  daily: {daily.shape} ({time.time()-t0:.1f}s)")

    print("Loading PIT SP500 membership ...")
    membership = load_membership()
    print(f"  membership: {membership.shape}, "
          f"{membership['asof'].nunique()} months, {membership['ticker'].nunique()} tickers")

    OUT_DIR.mkdir(exist_ok=True)

    results = {}
    for label, oos_end in [("primary_2007_2021", OOS_END_PRIMARY),
                           ("extended_2007_2024", OOS_END_EXTENDED)]:
        print(f"\n=== Run {label} (OOS_END={oos_end.date()}) ===")
        t0 = time.time()
        df, m = run_backtest(panel, daily, membership, oos_end, label=label)
        print(f"  walltime: {time.time()-t0:.1f}s")
        if m:
            print(f"  CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.3f}  "
                  f"MaxDD={m['max_dd']:.1%}  AnnVol={m['ann_vol']:.1%}  "
                  f"N={m['n_months']}")
        # Persist
        df.to_csv(OUT_DIR / f"backtest_{label}.csv")
        with open(OUT_DIR / f"summary_{label}.json", "w") as fh:
            json.dump({"label": label, "oos_end": str(oos_end.date()), **m,
                       "config": {
                           "top_k": TOP_K, "lgbm_w": LGBM_W, "sh12_w": SH12_W,
                           "sh5y_w": SH5Y_W, "weight_cap": WEIGHT_CAP,
                           "target_vol": TARGET_VOL, "regime_thresh": REGIME_THRESHOLD,
                           "train_months": TRAIN_MONTHS, "embargo_months": EMBARGO_MONTHS,
                           "min_train_months": MIN_TRAIN_MONTHS, "cost_bps": COST_BPS,
                       }}, fh, indent=2, default=str)
        results[label] = m

    print("\n=== Final summary vs 4T9wE claim ===")
    print(f"{'window':<22}  {'CAGR':>8}  {'Sharpe':>8}  {'MaxDD':>8}  {'AnnVol':>8}  {'N':>4}")
    for label, m in results.items():
        if m:
            print(f"{label:<22}  {m['cagr']:>7.2%}  {m['sharpe']:>8.3f}  "
                  f"{m['max_dd']:>7.1%}  {m['ann_vol']:>7.1%}  {m['n_months']:>4}")
    print(f"{'4T9wE claim (synth)':<22}  {0.651:>7.2%}  {1.834:>8.3f}  "
          f"{-0.136:>7.1%}  {0.301:>7.1%}  {180:>4}")
    print("\nGates: CAGR>=50% (4T9wE PASSED), Sharpe>=2.0 (4T9wE FAILED at 1.834)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
