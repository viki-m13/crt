"""
PIT NDX validation of 4T9wE's winning config (exp_012).

Same strategy spec as PIT SP500 validation (research/validation/sp500_pit/).
Same walk-forward protocol. Same K=30, blend, weighting, vol-target, regime.

Data sources (all on main):
- experiments/monthly_dca/v5/qqq_pit/ndx_pit_membership_monthly_full.parquet
  -> PIT NDX membership (137 months: 2015-01-31 -> 2026-05-31, 207 historical tickers)
- experiments/monthly_dca/v5/qqq_pit/ndx_monthly_prices.parquet
  -> NDX monthly close prices (377 months: 1995-01-31 -> 2026-05-31, 173 tickers)
- experiments/monthly_dca/cache/features/*.parquet
  -> 79-feature monthly snapshots (1997-01-31 -> 2026-05-07, ~1022 tickers/month)
- experiments/monthly_dca/cache/prices_extended.parquet
  -> daily SPY + universe prices for regime + vol target

Limitations (honest disclosure):
- NDX PIT membership only goes back to 2015-01-31, so OOS window is ~2019-2024.
- 34 of 207 historical NDX tickers are missing from ndx_monthly_prices.
- The features snapshot universe omits some non-SP500 NDX names (ADRs, late
  IPOs, spinoffs). Coverage is ~70% of NDX historical members. Those names
  are simply unavailable for selection -- they are NOT silently kept.
"""
from __future__ import annotations
import io, json, subprocess, sys, time, glob
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Config — IDENTICAL to PIT SP500 harness (no per-universe tuning)
# ---------------------------------------------------------------------------
TOP_K = 30
LGBM_W, SH12_W, SH5Y_W = 0.70, 0.20, 0.10
WEIGHT_CAP = 0.05
TARGET_VOL = 0.18
REGIME_THRESHOLD = -0.05
COST_BPS = 5.0
TRAIN_MONTHS = 48
EMBARGO_MONTHS = 3
MIN_TRAIN_MONTHS = 24

# OOS window: NDX membership starts 2015-01-31 -> first feasible OOS
# is 2015 + 48m train + 3m embargo + 24m min => ~2019-04. We start OOS
# at 2019-04 to leave a clean training history.
OOS_START = pd.Timestamp("2019-04-30")
OOS_END_PRIMARY = pd.Timestamp("2024-04-30")     # apples-to-apples cutoff
OOS_END_EXTENDED = pd.Timestamp("2025-12-31")    # extended (still before "today")

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

# Feature columns — intersected with whatever the on-main features snapshots have
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
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE
REPO = Path("/home/user/crt")
NDX_DIR = REPO / "experiments/monthly_dca/v5/qqq_pit"
FEAT_DIR = REPO / "experiments/monthly_dca/cache/features"
DAILY_PRICES_PATH = REPO / "experiments/monthly_dca/cache/prices_extended.parquet"


# ---------------------------------------------------------------------------
# Helpers (same shape as SP500 harness)
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
        cagr=float(cagr), ann_vol=float(std * np.sqrt(12)),
        sharpe=float(sharpe), max_dd=float(dd.min()),
        win_rate=float((rets > 0).mean()),
        n_months=int(len(rets)),
        mean_m=float(rets.mean()), std_m=float(std),
    )


def spy_regime_table(daily: pd.DataFrame) -> pd.DataFrame:
    spy = daily["SPY"].dropna()
    sma200 = spy.rolling(200, min_periods=100).mean()
    d_sma200 = (spy - sma200) / sma200
    vol_21 = spy.pct_change().rolling(21).std() * np.sqrt(252)
    return pd.DataFrame({"d_sma200": d_sma200, "vol_21d": vol_21}).resample("ME").last()


# ---------------------------------------------------------------------------
# Build a NDX feature panel from cached features + NDX prices for fwd returns
# ---------------------------------------------------------------------------
def build_ndx_panel(start: pd.Timestamp = pd.Timestamp("2010-01-31"),
                    end: pd.Timestamp = pd.Timestamp("2026-02-28")) -> pd.DataFrame:
    """
    Returns: long DataFrame with columns ['asof', 'ticker', *FEATURE_COLS, 'fwd_1m_ret']
    restricted to NDX-historical tickers (no membership filter applied here --
    that's applied at rebalance time).
    """
    ndx_mem = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_universe = set(ndx_mem["ticker"].unique())

    ndx_monthly_px = pd.read_parquet(NDX_DIR / "ndx_monthly_prices.parquet")
    ndx_monthly_ret = ndx_monthly_px.pct_change()

    feat_files = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
    frames = []
    for f in feat_files:
        date = pd.Timestamp(Path(f).stem)
        if date < start or date > end:
            continue
        df = pd.read_parquet(f)
        # FEAT_DIR snapshots are indexed by ticker
        df = df[df.index.isin(ndx_universe)].copy()
        if df.empty:
            continue
        df["asof"] = date
        df.index.name = "ticker"
        df = df.reset_index()
        frames.append(df)
    if not frames:
        raise RuntimeError("No NDX features found")
    panel = pd.concat(frames, ignore_index=True)

    # Forward 1m return from NDX monthly prices
    # Map asof -> month-end in ndx_monthly_px
    next_month = ndx_monthly_px.shift(-1)
    fwd = (next_month - ndx_monthly_px) / ndx_monthly_px

    def get_fwd(row):
        d, t = row["asof"], row["ticker"]
        if d not in fwd.index or t not in fwd.columns:
            return np.nan
        v = fwd.at[d, t]
        return float(v) if np.isfinite(v) else np.nan

    panel["fwd_1m_ret"] = panel.apply(get_fwd, axis=1)
    return panel


# ---------------------------------------------------------------------------
# Walk-forward LGBM
# ---------------------------------------------------------------------------
def fit_lgbm(panel: pd.DataFrame, train_dates: list, feat_cols: list):
    sub = panel[panel["asof"].isin(train_dates)].copy()
    sub = sub.dropna(subset=["fwd_1m_ret"])
    if sub.empty:
        return None
    X = sub[feat_cols].copy().fillna(sub[feat_cols].median(numeric_only=True))
    y = sub["fwd_1m_ret"].values
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model


def lgbm_score(model, feat_cols: list, snap: pd.DataFrame) -> pd.Series:
    if model is None or snap.empty:
        return pd.Series(dtype=float)
    X = snap[feat_cols].copy().fillna(snap[feat_cols].median(numeric_only=True))
    pred = model.predict(X)
    return pd.Series(pred, index=snap["ticker"].values)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_backtest(panel, ndx_monthly_px, membership, daily, oos_end, label):
    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    print(f"  [{label}] features used: {len(feat_cols)}/{len(FEATURE_COLS)}")

    spy_reg = spy_regime_table(daily)
    membership_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}

    all_dates = sorted(panel["asof"].unique())
    fwd_valid_dates = sorted(panel[panel["fwd_1m_ret"].notna()]["asof"].unique())

    records = []
    model_cache = {}

    for i, date in enumerate(all_dates):
        if date > oos_end:
            break
        if i == len(all_dates) - 1:
            break

        # Training window with embargo
        idx = i
        cutoff_idx = idx - EMBARGO_MONTHS
        if cutoff_idx < 0:
            continue
        train_end = all_dates[cutoff_idx]
        pool = [d for d in fwd_valid_dates if d <= train_end and d < date]
        if len(pool) < MIN_TRAIN_MONTHS:
            if date >= OOS_START:
                records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                    gate_pass=False, reason="insufficient_train"))
            continue
        train_dates = pool[-TRAIN_MONTHS:]

        if date < OOS_START:
            continue

        key = (train_dates[0], train_dates[-1])
        if key not in model_cache:
            model_cache[key] = fit_lgbm(panel, train_dates, feat_cols)
        model = model_cache[key]

        snap = panel[panel["asof"] == date].copy()
        if snap.empty:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="no_snapshot"))
            continue

        # PIT NDX membership filter
        snap = snap[snap["ticker"].apply(lambda t: (date, t) in membership_set)]
        if snap.empty:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="empty_after_membership"))
            continue

        # Regime gate (SPY 200ma_loose)
        reg = spy_reg.reindex([date]).iloc[0]
        d_sma200 = reg["d_sma200"]
        spy_vol = reg["vol_21d"]
        if not (np.isfinite(d_sma200) and d_sma200 > REGIME_THRESHOLD):
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=False, reason="regime"))
            continue

        # Score = z(LGBM)*0.70 + z(sharpe_12m)*0.20 + z(sharpe_5y)*0.10
        lgbm_z = znorm(lgbm_score(model, feat_cols, snap))
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

        # Returns from NDX monthly prices
        if date not in ndx_monthly_px.index:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=True, reason="no_px_date"))
            continue
        next_idx = ndx_monthly_px.index.searchsorted(date) + 1
        if next_idx >= len(ndx_monthly_px.index):
            break
        next_date = ndx_monthly_px.index[next_idx]
        p0 = ndx_monthly_px.loc[date]
        p1 = ndx_monthly_px.loc[next_date]
        common = [t for t in tickers
                  if t in ndx_monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            records.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0,
                                gate_pass=True, reason="no_common"))
            continue

        vol_src = "vol_12m" if "vol_12m" in snap.columns else "vol_1y"
        vols_map = dict(zip(snap["ticker"].values, snap[vol_src].values))
        vols = np.array([max(float(vols_map.get(t, 0.20)), 0.05)
                         if np.isfinite(vols_map.get(t, np.nan)) else 0.20
                         for t in common])
        raw_w = 1.0 / vols
        w = iterative_weight_cap(raw_w, WEIGHT_CAP)

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        raw_port = float((w * rets).sum())
        scale = min(TARGET_VOL / spy_vol, 1.0) if np.isfinite(spy_vol) and spy_vol > 1e-6 else 1.0
        cost = COST_BPS / 10_000.0
        port = scale * raw_port - 2 * cost * scale

        records.append(dict(date=date, ret_m=port, n_picks=len(common),
                            scale=scale, gate_pass=True, reason="ok",
                            picks=",".join(common)))

    df = pd.DataFrame(records).set_index("date")
    rets = df.loc[(df.index >= OOS_START) & (df.index <= oos_end), "ret_m"]
    m = metrics(rets)
    return df, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building NDX feature panel ...")
    t0 = time.time()
    panel = build_ndx_panel()
    print(f"  panel: {panel.shape}, {panel['asof'].nunique()} months, "
          f"{panel['ticker'].nunique()} tickers, "
          f"fwd_1m valid: {panel['fwd_1m_ret'].notna().sum()} ({time.time()-t0:.1f}s)")

    print("Loading NDX monthly prices ...")
    ndx_monthly_px = pd.read_parquet(NDX_DIR / "ndx_monthly_prices.parquet")
    ndx_monthly_px.index = pd.to_datetime(ndx_monthly_px.index)
    print(f"  ndx monthly px: {ndx_monthly_px.shape}")

    print("Loading PIT NDX membership ...")
    membership = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    membership["asof"] = pd.to_datetime(membership["asof"])
    print(f"  membership: {membership.shape}")

    print("Loading daily prices for SPY regime + vol target ...")
    daily = pd.read_parquet(DAILY_PRICES_PATH)
    daily.index = pd.to_datetime(daily.index)
    print(f"  daily: {daily.shape}")

    OUT_DIR.mkdir(exist_ok=True)
    results = {}
    for label, oos_end in [("primary_2019_2024", OOS_END_PRIMARY),
                           ("extended_2019_2025", OOS_END_EXTENDED)]:
        print(f"\n=== Run {label} (OOS={OOS_START.date()} -> {oos_end.date()}) ===")
        t0 = time.time()
        df, m = run_backtest(panel, ndx_monthly_px, membership, daily, oos_end, label)
        print(f"  walltime: {time.time()-t0:.1f}s")
        if m:
            print(f"  CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.3f}  "
                  f"MaxDD={m['max_dd']:.1%}  AnnVol={m['ann_vol']:.1%}  "
                  f"N={m['n_months']}")
        df.to_csv(OUT_DIR / f"backtest_{label}.csv")
        with open(OUT_DIR / f"summary_{label}.json", "w") as fh:
            json.dump({"label": label, "oos_start": str(OOS_START.date()),
                       "oos_end": str(oos_end.date()), **m,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
