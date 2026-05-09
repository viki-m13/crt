"""Compute additional alpha features and append to per-month feature parquets.

These are designed from cross-sectional IC analysis (see REPORT.md §11). The
features below have shown high cross-sectional signal-to-noise on the 1997-2024
universe and form the foundation of the new `strategies_alpha.py` library.

We append:
  vol_3m, vol_6m              — recent volatility
  vol_contraction             — vol_3m / vol_1y (institutional accumulation proxy)
  vol_expansion_24m           — current vol / 24m median (for blow-off/freefall detect)
  rs_3m_spy, rs_6m_spy, rs_12m_spy — relative strength vs SPY
  rs_3m_zscore                — cross-sectional z of rs_3m_spy
  mom_accel                   — mom_3m - mom_12m (recent reversal)
  mom_consistency_12m         — fraction of months in last 12 with positive return
  dist_from_recent_low_1y     — % above 1y low (rebound strength)
  near_52wh_60d               — was within 5% of 52wh sometime in last 60 trading days
  bb_width_pct                — Bollinger Band width % (compression)
  bb_width_contraction        — current BB width / median 1y BB width
  drawdown_age_days           — days since last 252-day high
  price_x_vol                 — price * vol (proxy for rough $-traded)
  log_price                   — log of price (size proxy stand-in)
  excess_5y_logret            — 5y log-return - SPY 5y log-return
  multibagger_ratio_24m       — ratio of best month to second-worst month (more robust tail)
  trend_slope_252             — annualized slope of log-price 12m
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import (
    CACHE,
    FEATURES_DIR,
    load_panel,
)
from experiments.monthly_dca.backtester import month_end_dates


SENTINEL_COL = "trend_slope_252"  # presence => already computed


def compute_alpha_features(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    sub = panel.loc[panel.index <= asof]
    last_row = sub.iloc[-1]
    alive_mask = last_row.notna()
    counts = sub.notna().sum()
    eligible = alive_mask & (counts >= 252)
    cols = sub.columns[eligible]
    sub = sub[cols].astype("float64")
    out = pd.DataFrame(index=cols)

    px = sub.iloc[-1]

    # Daily returns matrix
    rets = sub.pct_change()

    # Recent and longer-window vol (annualized)
    if len(sub) >= 63:
        out["vol_3m"] = rets.iloc[-63:].std() * math.sqrt(252)
    if len(sub) >= 126:
        out["vol_6m"] = rets.iloc[-126:].std() * math.sqrt(252)
    if "vol_3m" in out.columns and len(sub) >= 252:
        vol_1y = rets.iloc[-252:].std() * math.sqrt(252)
        out["vol_contraction"] = (out["vol_3m"] / vol_1y).replace([np.inf, -np.inf], np.nan)

    # Vol expansion vs 24m median vol (current 1m vs prior history)
    if len(sub) >= 504:
        # Compute monthly vol over last 24 months
        monthly_vols = []
        for i in range(24):
            start = -((i + 1) * 21)
            end = -(i * 21) if i > 0 else None
            if -start > len(sub):
                continue
            window = rets.iloc[start:end]
            if len(window) < 15:
                continue
            monthly_vols.append(window.std() * math.sqrt(252))
        if len(monthly_vols) >= 6:
            mat = pd.concat(monthly_vols, axis=1)
            med = mat.median(axis=1)
            current = monthly_vols[0]
            out["vol_expansion_24m"] = (current / med).replace([np.inf, -np.inf], np.nan)

    # Relative strength vs SPY
    if "SPY" in sub.columns:
        spy = sub["SPY"]
        spy_now = float(spy.iloc[-1])
        if len(sub) >= 63:
            spy_3m = float(spy.iloc[-63])
            stock_3m = sub.iloc[-63]
            out["rs_3m_spy"] = (px / stock_3m - 1.0) - (spy_now / spy_3m - 1.0)
        if len(sub) >= 126:
            spy_6m = float(spy.iloc[-126])
            stock_6m = sub.iloc[-126]
            out["rs_6m_spy"] = (px / stock_6m - 1.0) - (spy_now / spy_6m - 1.0)
        if len(sub) >= 252:
            spy_12m = float(spy.iloc[-252])
            stock_12m = sub.iloc[-252]
            out["rs_12m_spy"] = (px / stock_12m - 1.0) - (spy_now / spy_12m - 1.0)
        if len(sub) >= 1260 and "SPY" in sub.columns:
            try:
                logp = np.log(sub.iloc[-1] / sub.iloc[-1260])
                logspy = np.log(spy_now / float(spy.iloc[-1260]))
                out["excess_5y_logret"] = logp - logspy
            except Exception:
                pass

    # Cross-sectional z of rs_3m_spy will be computed per-asof at scoring time
    # but we precompute a simple z here for convenience
    if "rs_3m_spy" in out.columns:
        rs3 = out["rs_3m_spy"]
        med = rs3.median(); sd = rs3.std()
        if sd and np.isfinite(sd) and sd > 0:
            out["rs_3m_zscore"] = (rs3 - med) / sd

    # Momentum acceleration
    if len(sub) >= 252:
        mom_3m = px / sub.iloc[-63] - 1.0
        mom_12m = px / sub.iloc[-252] - 1.0
        out["mom_accel"] = mom_3m - mom_12m

    # Momentum consistency: fraction of last 12 months with positive return
    if len(sub) >= 252:
        monthly = sub.iloc[-252:].resample("ME").last()
        mret = monthly.pct_change()
        out["mom_consistency_12m"] = (mret > 0).sum() / mret.count().replace(0, np.nan)

    # Distance from 1y low (rebound strength)
    if len(sub) >= 252:
        low_1y = sub.iloc[-252:].min()
        out["dist_from_low_1y"] = (px / low_1y - 1.0)

    # Near 52wh in last 60 days?
    if len(sub) >= 252:
        last60 = sub.iloc[-60:]
        max60 = last60.max()
        high252 = sub.iloc[-252:].max()
        out["near_52wh_60d"] = (max60 >= 0.95 * high252).astype(float)

    # Bollinger band width % over 252d
    if len(sub) >= 252:
        last20 = sub.iloc[-20:]
        m20 = last20.mean()
        sd20 = last20.std()
        out["bb_width_pct"] = (4 * sd20) / m20.replace(0, np.nan)
        # 1y median BB width
        bb_widths = []
        for i in range(0, 252, 5):
            window = sub.iloc[-(252 - i + 20):-(252 - i)] if i < 252 else sub.iloc[-20:]
            if len(window) < 20:
                continue
            wmean = window.mean()
            wsd = window.std()
            bb_widths.append((4 * wsd / wmean.replace(0, np.nan)))
        if bb_widths:
            mat = pd.concat(bb_widths, axis=1)
            med = mat.median(axis=1)
            out["bb_width_contraction"] = (out["bb_width_pct"] / med).replace([np.inf, -np.inf], np.nan)

    # Drawdown age: days since last 252-day high
    if len(sub) >= 252:
        last_252 = sub.iloc[-252:]
        # For each ticker, find argmax then days_since
        argmax_idx = last_252.values.argmax(axis=0)
        days_since = (len(last_252) - 1 - argmax_idx).astype(float)
        # Map to series
        out["drawdown_age_days"] = pd.Series(days_since, index=last_252.columns)

    # log price (very rough size proxy in lieu of market cap)
    out["log_price"] = np.log(px.replace(0, np.nan))

    # Multibagger ratio: best 24m month / abs(median worst-half month) — robust tail
    if len(sub) >= 504:
        monthly = sub.iloc[-504:].resample("ME").last().pct_change().dropna()
        if len(monthly) >= 6:
            bm = monthly.max()
            # average of worst 6 monthly returns
            sorted_neg = monthly.apply(lambda s: s.nsmallest(6).abs().mean(), axis=0)
            out["multibagger_ratio_24m"] = (bm / sorted_neg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Trend slope (12m) as annualized log-return slope per day
    if len(sub) >= 252:
        last = sub.iloc[-252:]
        x = np.arange(252)
        x_mean = x.mean()
        x_var = float(((x - x_mean) ** 2).sum())
        slopes = {}
        for c in last.columns:
            y = np.log(last[c].to_numpy())
            if not np.all(np.isfinite(y)):
                slopes[c] = np.nan
                continue
            y_mean = y.mean()
            slope = float(((x - x_mean) * (y - y_mean)).sum() / x_var)
            slopes[c] = slope * 252.0  # annualized
        out["trend_slope_252"] = pd.Series(slopes)

    return out


def main(start: str = "1997-01-01", end: str = "2099-01-01") -> None:
    panel = load_panel()
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    print(f"Adding alpha features to {len(months)} months")
    for i, asof in enumerate(months):
        feat_path = FEATURES_DIR / f"{asof.date()}.parquet"
        if not feat_path.exists():
            continue
        existing = pd.read_parquet(feat_path)
        if SENTINEL_COL in existing.columns:
            continue
        try:
            extras = compute_alpha_features(panel, asof)
        except Exception as e:
            print(f"  skip {asof.date()}: {e}")
            continue
        # Drop overlap columns from extras to avoid duplication on existing
        for col in list(extras.columns):
            if col in existing.columns:
                extras = extras.drop(columns=[col])
        merged = existing.join(extras, how="left")
        merged.to_parquet(feat_path)
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  [{i+1}/{len(months)}] {asof.date()}: {merged.shape}")


if __name__ == "__main__":
    main()
