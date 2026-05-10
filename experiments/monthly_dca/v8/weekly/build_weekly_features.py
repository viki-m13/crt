"""Build a weekly feature panel for walk-forward GBM on PIT S&P 500.

Weekly asofs are Friday closes. Features are PIT — every value at asof T
uses only data with index <= T. Cross-sectional ranks are produced by the
fit step (`fit_weekly_gbm.py`), not here.

Output (single parquet, long format):
  experiments/monthly_dca/v8/weekly/cache/features_weekly.parquet
  columns: asof, ticker, <feature columns...>, fwd_4w_ret
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE_IN = ROOT / "experiments" / "monthly_dca" / "cache"
OUT_DIR = Path(__file__).resolve().parent / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rolling_max_drawdown(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling max drawdown over the past `window` trading days (negative or zero)."""
    cummax = close.rolling(window, min_periods=window // 2).max()
    return (close / cummax - 1.0)


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_dn = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _trend_health_5y(close: pd.DataFrame) -> pd.DataFrame:
    sma200 = close.rolling(200, min_periods=120).mean()
    above = (close > sma200).astype(float)
    # 5y = ~1260 trading days
    return above.rolling(1260, min_periods=600).mean()


def main():
    print("[load] prices_extended")
    px = pd.read_parquet(CACHE_IN / "prices_extended.parquet")
    print(f"  shape={px.shape}, range={px.index.min().date()} -> {px.index.max().date()}")

    # Drop tickers with too little history (< 252 days)
    valid_count = px.notna().sum(axis=0)
    keep = valid_count[valid_count >= 252].index
    px = px[keep]
    print(f"  kept {len(keep)} tickers with >=252 days")

    print("[compute] features (vectorised across tickers)")
    t0 = time.time()
    log_px = np.log(px.where(px > 0))
    ret_1d = log_px.diff()

    # Multi-horizon returns
    ret_5d = log_px - log_px.shift(5)
    ret_10d = log_px - log_px.shift(10)
    ret_21d = log_px - log_px.shift(21)
    ret_63d = log_px - log_px.shift(63)
    ret_126d = log_px - log_px.shift(126)
    ret_252d = log_px - log_px.shift(252)

    # Momentum (skip-1m)
    mom_12_1 = ret_252d - ret_21d
    mom_6_1 = ret_126d - ret_21d
    mom_3_1 = ret_63d - ret_21d

    # Volatility
    vol_21d = ret_1d.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_63d = ret_1d.rolling(63, min_periods=30).std() * np.sqrt(252)
    vol_252d = ret_1d.rolling(252, min_periods=126).std() * np.sqrt(252)

    # Sharpe-momentum
    mom_per_vol_12 = mom_12_1 / vol_252d.replace(0, np.nan)

    # SMA distances
    sma50 = px.rolling(50, min_periods=25).mean()
    sma200 = px.rolling(200, min_periods=120).mean()
    d_sma50 = (px / sma50) - 1.0
    d_sma200 = (px / sma200) - 1.0

    # RSI
    rsi_14 = _rsi(px, 14)

    # 1Y pullback (price / 252d max - 1, negative or zero)
    high_252 = px.rolling(252, min_periods=126).max()
    pullback_1y = (px / high_252) - 1.0

    # 1Y rolling drawdown
    dd_252 = _rolling_max_drawdown(px, 252)

    # Trend health: % days above 200dma in last 5y
    trend_health_5y = _trend_health_5y(px)

    # Acceleration: 21d ret minus 63d ret (positive = accelerating)
    accel = ret_21d - (ret_63d / 3.0)

    # Vol contraction: vol_21 / vol_63 (low = contracting)
    vol_contr = vol_21d / vol_63d.replace(0, np.nan)

    # Above-200dma streak (proxy via rolling sum of (px > 200dma))
    above_200 = (px > sma200).astype(float)
    streak_above_200_60 = above_200.rolling(60, min_periods=30).sum()

    print(f"  features computed in {time.time()-t0:.1f}s")

    # Friday weekly resample
    print("[resample] weekly Friday closes")
    weekly_idx = pd.date_range(px.index.min(), px.index.max(), freq="W-FRI")
    # Asof index: most recent trading day at or before each Friday
    asof_idx = []
    px_idx = px.index
    for w in weekly_idx:
        pos = px_idx.searchsorted(w, side="right") - 1
        if pos >= 0:
            asof_idx.append(px_idx[pos])
    asof_idx = pd.DatetimeIndex(sorted(set(asof_idx)))

    feat_dict = {
        "log_px": log_px,
        "ret_5d": ret_5d, "ret_10d": ret_10d, "ret_21d": ret_21d,
        "ret_63d": ret_63d, "ret_126d": ret_126d, "ret_252d": ret_252d,
        "mom_12_1": mom_12_1, "mom_6_1": mom_6_1, "mom_3_1": mom_3_1,
        "vol_21d": vol_21d, "vol_63d": vol_63d, "vol_252d": vol_252d,
        "mom_per_vol_12": mom_per_vol_12,
        "d_sma50": d_sma50, "d_sma200": d_sma200,
        "rsi_14": rsi_14,
        "pullback_1y": pullback_1y,
        "dd_252d": dd_252,
        "trend_health_5y": trend_health_5y,
        "accel": accel, "vol_contraction": vol_contr,
        "streak_above_200_60": streak_above_200_60,
    }

    print("[build] long-format weekly panel")
    t0 = time.time()
    feat_long_list = []
    for fname, fdf in feat_dict.items():
        sub = fdf.loc[asof_idx]
        sub = sub.stack(future_stack=True).rename(fname)
        feat_long_list.append(sub)
    feat_long = pd.concat(feat_long_list, axis=1).reset_index()
    feat_long.columns = ["asof", "ticker"] + list(feat_dict.keys())
    print(f"  long panel rows={len(feat_long)} in {time.time()-t0:.1f}s")

    # Forward 4-week return target (close-to-close 20 trading days)
    print("[target] fwd_4w_ret (20 trading days)")
    t0 = time.time()
    fwd_4w = (px.shift(-20) / px) - 1.0
    fwd_4w_long = (
        fwd_4w.loc[asof_idx]
        .stack(future_stack=True)
        .rename("fwd_4w_ret")
        .reset_index()
    )
    fwd_4w_long.columns = ["asof", "ticker", "fwd_4w_ret"]
    feat_long = feat_long.merge(fwd_4w_long, on=["asof", "ticker"], how="left")
    print(f"  done in {time.time()-t0:.1f}s")

    # Drop rows with missing core features (e.g. tickers without 252d history at asof)
    core = ["mom_12_1", "vol_252d", "d_sma200"]
    pre_n = len(feat_long)
    feat_long = feat_long.dropna(subset=core)
    print(f"  dropped {pre_n - len(feat_long)} rows missing core features")

    out_path = OUT_DIR / "features_weekly.parquet"
    feat_long.to_parquet(out_path, index=False)
    print(f"[save] {out_path}  ({len(feat_long)} rows, {feat_long['asof'].nunique()} weeks, {feat_long['ticker'].nunique()} tickers)")


if __name__ == "__main__":
    main()
