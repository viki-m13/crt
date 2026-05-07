"""Compute additional 'explosive winner' features and append to cached panels."""
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
    load_features,
    load_panel,
)
from experiments.monthly_dca.backtester import month_end_dates


# ---------------------------------------------------------------------------
def compute_extras(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    sub = panel.loc[panel.index <= asof]
    last_row = sub.iloc[-1]
    alive_mask = last_row.notna()
    counts = sub.notna().sum()
    eligible = alive_mask & (counts >= 504)
    cols = sub.columns[eligible]
    sub = sub[cols].astype("float64")

    px = sub.iloc[-1]
    out = pd.DataFrame(index=cols)

    # Long-horizon momentum
    if len(sub) >= 756:
        out["mom_3y"] = sub.iloc[-1] / sub.iloc[-756] - 1.0
    if len(sub) >= 1260:
        out["mom_5y"] = sub.iloc[-1] / sub.iloc[-1260] - 1.0
    if len(sub) >= 504:
        out["mom_2y"] = sub.iloc[-1] / sub.iloc[-504] - 1.0

    # Distance from N-week high
    if len(sub) >= 756:
        h_3y = sub.iloc[-756:].max()
        out["pullback_3y"] = (px / h_3y - 1.0)
    if len(sub) >= 1260:
        h_5y = sub.iloc[-1260:].max()
        out["pullback_5y"] = (px / h_5y - 1.0)

    # Breakout proximity: % distance below 52-week high * (-1) so positive = below high
    high_252 = sub.iloc[-252:].max()
    out["below_52wh"] = high_252 / px - 1.0   # >=0; smaller = closer to/above high
    # Above 252d high?
    out["new_52wh"] = (px >= high_252 * 0.999).astype(float)

    # Volatility-adjusted momentum (Sharpe-like over 12m)
    if len(sub) >= 252:
        rets = sub.iloc[-252:].pct_change()
        m = rets.mean() * 252
        v = rets.std() * math.sqrt(252)
        out["sharpe_12m"] = m / v.replace(0, np.nan)
        out["mean_ret_12m"] = m
        out["vol_12m"] = v

    # Beta to SPY (24m)
    if "SPY" in sub.columns and len(sub) >= 504:
        spy_rets = sub["SPY"].iloc[-504:].pct_change().dropna()
        rets_2y = sub.iloc[-504:].pct_change()
        cov = rets_2y.apply(lambda x: x.cov(spy_rets))
        var = spy_rets.var()
        out["beta_2y"] = cov / var

    # Tail-asymmetry: ratio of best-month to worst-month over last 24m
    if len(sub) >= 504:
        monthly = sub.iloc[-504:].resample("ME").last().pct_change()
        out["best_month_24m"] = monthly.max()
        out["worst_month_24m"] = monthly.min()
        out["tail_ratio_24m"] = monthly.max() / monthly.min().abs().replace(0, np.nan)

    # Smoothness of trend: R^2 of log price vs time over 12m
    if len(sub) >= 252:
        last = sub.iloc[-252:]
        x = np.arange(252)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        out_r2 = {}
        for c in last.columns:
            y = np.log(last[c].to_numpy())
            if not np.all(np.isfinite(y)):
                out_r2[c] = np.nan
                continue
            y_mean = y.mean()
            ss_tot = float(((y - y_mean) ** 2).sum())
            slope = float(((x - x_mean) * (y - y_mean)).sum() / x_var)
            intercept = y_mean - slope * x_mean
            y_hat = slope * x + intercept
            ss_res = float(((y - y_hat) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            out_r2[c] = r2
        out["trend_r2_12m"] = pd.Series(out_r2)

    # Drawdown duration: avg days to recover from prior 20% drawdowns
    # (cheap proxy: longest streak below 200dma over last 5y)
    sma200 = sub.rolling(200, min_periods=200).mean()
    below = (sub < sma200).astype(int)
    last_5y = below.iloc[-1260:] if len(below) >= 1260 else below
    # Count max consecutive below
    def _max_streak(s: pd.Series) -> float:
        m = 0
        cur = 0
        for v in s:
            if v == 1:
                cur += 1
                if cur > m:
                    m = cur
            else:
                cur = 0
        return m
    out["max_below_200_streak"] = last_5y.apply(_max_streak)

    return out


def main(start: str = "2017-12-31", end: str = "2025-12-31") -> None:
    panel = load_panel()
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    print(f"Adding extra features to {len(months)} months")
    for i, asof in enumerate(months):
        feat_path = FEATURES_DIR / f"{asof.date()}.parquet"
        if not feat_path.exists():
            continue
        existing = pd.read_parquet(feat_path)
        # Skip if extras already merged
        if "mom_3y" in existing.columns and "trend_r2_12m" in existing.columns:
            continue
        extras = compute_extras(panel, asof)
        merged = existing.join(extras, how="left")
        merged.to_parquet(feat_path)
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  [{i+1}/{len(months)}] {asof.date()}: {merged.shape}")


if __name__ == "__main__":
    main()
