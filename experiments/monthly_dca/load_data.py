"""Load all ticker price series from docs/data/tickers/*.json into a panel parquet.

Backfills SPY (and a small set of bench/factor ETFs) via yfinance to cover the
gap between the freshest ticker date (2026-03-20) and SPY's stale snapshot
(2025-09-19). Persists the result so we never have to redo I/O.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TICKER_DIR = ROOT / "docs" / "data" / "tickers"
OUT_DIR = ROOT / "experiments" / "monthly_dca" / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_PARQUET = OUT_DIR / "prices.parquet"
META_PARQUET = OUT_DIR / "meta.parquet"

EXTRA_TICKERS = ["SPY", "QQQ", "IWM", "VTI", "RSP"]  # benchmarks we may need fresh


def _load_one(path: Path) -> pd.Series | None:
    with open(path) as f:
        d = json.load(f)
    s = d.get("series") or {}
    dates = s.get("dates") or []
    prices = s.get("prices") or []
    if not dates or not prices or len(dates) != len(prices):
        return None
    idx = pd.to_datetime(dates)
    out = pd.Series(prices, index=idx, dtype="float64", name=d["ticker"])
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_all_local() -> pd.DataFrame:
    files = sorted(TICKER_DIR.glob("*.json"))
    series_map: dict[str, pd.Series] = {}
    for p in files:
        s = _load_one(p)
        if s is None or len(s) < 252:  # require >= 1y history
            continue
        series_map[s.name] = s
    if not series_map:
        raise RuntimeError("No tickers loaded")
    df = pd.concat(series_map.values(), axis=1).sort_index()
    df.index.name = "date"
    return df


def fetch_yf(tickers: list[str], start: str = "2014-01-01") -> pd.DataFrame:
    import yfinance as yf

    print(f"Fetching {len(tickers)} via yfinance from {start}...")
    df = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # auto_adjust=True puts 'Close' as the adjusted close
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        else:
            close = df.xs("Adj Close", axis=1, level=0)
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close


def merge_yf_into_local(local: pd.DataFrame, yf_df: pd.DataFrame) -> pd.DataFrame:
    """Splice yfinance data on top of local where local is missing/stale.

    Strategy: for each ticker in yf_df, scale yf series so it agrees with the
    local series on the last common date, then fill local where local is NaN.
    Avoids drift from differing dividend-adjustment conventions.
    """
    out = local.copy()
    for col in yf_df.columns:
        yfs = yf_df[col].dropna()
        if yfs.empty:
            continue
        if col not in out.columns:
            out[col] = yfs
            continue
        loc = out[col].dropna()
        common = loc.index.intersection(yfs.index)
        if len(common) == 0:
            continue
        last_common = common.max()
        scale = loc.loc[last_common] / yfs.loc[last_common]
        scaled = yfs * scale
        out[col] = out[col].combine_first(scaled)
    return out.sort_index()


def main(force: bool = False) -> pd.DataFrame:
    if PRICE_PARQUET.exists() and not force:
        print(f"Loading cached panel: {PRICE_PARQUET}")
        return pd.read_parquet(PRICE_PARQUET)

    print("Loading local tickers...")
    local = load_all_local()
    print(f"Local: {local.shape[0]} days x {local.shape[1]} tickers")
    print(f"  date range: {local.index.min().date()} .. {local.index.max().date()}")

    # Refresh stale tickers (SPY etc) so benchmarks reach the same date
    target_end = local.index.max()
    stale = [t for t in EXTRA_TICKERS if t in local.columns and local[t].dropna().index.max() < target_end]
    if stale:
        print(f"Stale benchmarks: {stale} -- backfilling via yfinance")
        yfd = fetch_yf(stale, start="2014-01-01")
        if not yfd.empty:
            local = merge_yf_into_local(local, yfd)

    # Save panel
    local.index = pd.to_datetime(local.index)
    local.to_parquet(PRICE_PARQUET, compression="zstd")
    print(f"Wrote {PRICE_PARQUET} ({local.shape[0]}x{local.shape[1]})")

    # Meta: first/last valid date per ticker (used for PIT eligibility)
    meta = pd.DataFrame(
        {
            "first_date": local.apply(lambda s: s.first_valid_index()),
            "last_date": local.apply(lambda s: s.last_valid_index()),
            "n_days": local.notna().sum(),
        }
    )
    meta.to_parquet(META_PARQUET)
    print(f"Wrote {META_PARQUET}")
    return local


if __name__ == "__main__":
    main()
