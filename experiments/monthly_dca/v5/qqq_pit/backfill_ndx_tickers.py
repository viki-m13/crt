"""Backfill missing PIT-NDX tickers via yfinance into the broader panel.

Builds:
  experiments/monthly_dca/v5/qqq_pit/ndx_monthly_prices.parquet
  experiments/monthly_dca/v5/qqq_pit/ndx_monthly_returns.parquet

Approach:
  - Load the PIT NDX membership (~207 unique tickers).
  - For each ticker not already in our broader panel, fetch its full
    daily price history from yfinance (including delisted / merged
    names — yfinance returns historical data for many of them).
  - Resample to month-end last close and compute monthly returns.
  - Merge with the existing broader 1833 panel (for tickers we already
    have) to produce a clean NDX-restricted monthly panel.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
QQQ_DIR = ROOT / "experiments" / "monthly_dca" / "v5" / "qqq_pit"
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"


def backfill_yfinance(missing: list[str]) -> dict[str, pd.Series]:
    """Fetch daily Close from yfinance for each ticker. Returns dict
    ticker -> Series of daily closes (tz-naive index)."""
    import yfinance as yf
    print(f"Fetching {len(missing)} tickers via yfinance.download(threads=True)...")
    out: dict[str, pd.Series] = {}
    df = yf.download(missing, start="2014-01-01", end="2026-06-01",
                     auto_adjust=True, progress=False, threads=True)
    if df is None or df.empty:
        print("  → empty response")
        return out
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(0)
        if "Close" in levels:
            close = df["Close"]
        elif "Adj Close" in levels:
            close = df["Adj Close"]
        else:
            print(f"  → unexpected columns: {set(levels)}")
            return out
    else:
        close = df[["Close"]].rename(columns={"Close": missing[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    for tk in missing:
        if tk in close.columns:
            s = close[tk].dropna()
            if len(s) >= 60:
                out[tk] = s
    print(f"  → got {len(out)} of {len(missing)} tickers with usable history")
    return out


def main():
    QQQ_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Backfill missing PIT-NDX tickers")
    print("=" * 60)

    mem = pd.read_parquet(QQQ_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_tickers = sorted(mem["ticker"].unique().tolist())
    print(f"NDX universe ({len(ndx_tickers)} unique tickers across 2015-2026)")

    mr = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    mp = pd.read_parquet(CACHE / "v2" / "monthly_prices_clean.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    if not isinstance(mp.index, pd.DatetimeIndex):
        mp.index = pd.to_datetime(mp.index)
    panel_cols = set(mr.columns.tolist())

    have = [t for t in ndx_tickers if t in panel_cols]
    missing = [t for t in ndx_tickers if t not in panel_cols]
    print(f"In existing panel: {len(have)}; missing: {len(missing)}")

    fetched = backfill_yfinance(missing) if missing else {}

    # Build daily Close DF for fetched
    if fetched:
        fc = pd.DataFrame(fetched)
        # Resample to month-end last close
        fc_m = fc.resample("ME").last()
        # Compute monthly returns
        fc_ret = fc_m.pct_change(fill_method=None)
        print(f"Backfilled monthly panel: {fc_m.shape}, returns: {fc_ret.shape}")
    else:
        fc_m = pd.DataFrame()
        fc_ret = pd.DataFrame()

    # Merge: take existing columns for `have`, append new columns for backfilled
    have_mp = mp[have].copy() if have else pd.DataFrame()
    have_mr = mr[have].copy() if have else pd.DataFrame()

    if not fc_m.empty:
        # Align indices
        all_idx = have_mp.index.union(fc_m.index)
        have_mp = have_mp.reindex(all_idx)
        have_mr = have_mr.reindex(all_idx)
        fc_m = fc_m.reindex(all_idx)
        fc_ret = fc_ret.reindex(all_idx)
        merged_mp = pd.concat([have_mp, fc_m], axis=1)
        merged_mr = pd.concat([have_mr, fc_ret], axis=1)
    else:
        merged_mp = have_mp
        merged_mr = have_mr

    print(f"\nMerged NDX monthly panel: {merged_mp.shape}")
    print(f"NDX tickers in final panel: "
          f"{sum(1 for t in ndx_tickers if t in merged_mp.columns)}/{len(ndx_tickers)}")

    merged_mp.to_parquet(QQQ_DIR / "ndx_monthly_prices.parquet")
    merged_mr.to_parquet(QQQ_DIR / "ndx_monthly_returns.parquet")
    print(f"\nSaved:")
    print(f"  {QQQ_DIR / 'ndx_monthly_prices.parquet'}")
    print(f"  {QQQ_DIR / 'ndx_monthly_returns.parquet'}")

    # Re-check pool size with full data
    final_set = set(merged_mp.columns.tolist())
    pool = mem[mem["ticker"].isin(final_set)].groupby("asof").size()
    print(f"\nPanel-resolvable pool size after backfill: "
          f"median={int(pool.median())}, min={int(pool.min())}, max={int(pool.max())}")


if __name__ == "__main__":
    main()
