"""Download daily adjusted-close history for the ETF universes via yfinance.

Saves three parquets:
  data/prices_broad.parquet     — un-leveraged ETFs + SPY
  data/prices_levered.parquet   — leveraged/inverse ETFs + SPY
  data/prices_combined.parquet  — union + SPY

Each panel: index=trading day (UTC-naive), columns=tickers, values=adj close.

Idempotent: if a parquet exists and the latest bar is within `freshness_days`
of today, the panel is reused. Set FORCE=1 to refetch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
from universe import BROAD_ETFS, LEVERAGED_ETFS, ALWAYS_INCLUDE, combined_universe  # noqa: E402


START = "1995-01-01"


def _fetch_yf(tickers: list[str], start: str = START) -> pd.DataFrame:
    import yfinance as yf
    print(f"  yfinance: {len(tickers)} tickers from {start}", flush=True)
    df = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        else:
            close = df.xs("Adj Close", axis=1, level=0)
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    # Keep only columns where we actually got data
    keep = [c for c in close.columns if close[c].notna().sum() > 60]
    close = close[keep]
    return close


def fetch_universe(name: str, tickers: list[str], force: bool = False) -> pd.DataFrame:
    out = DATA / f"prices_{name}.parquet"
    if out.exists() and not force:
        df = pd.read_parquet(out)
        last = df.index.max()
        # If panel is "fresh enough" (within 7 days of today), reuse.
        if (pd.Timestamp.utcnow().tz_localize(None) - last).days <= 7:
            print(f"[{name}] cached {out.name} shape={df.shape} last={last.date()}", flush=True)
            return df
    print(f"[{name}] fetching {len(tickers)} tickers via yfinance...", flush=True)
    # Fetch in chunks to avoid 'Read timed out' on huge requests
    chunks = []
    chunk_size = 50
    uniq = sorted(set(tickers) | set(ALWAYS_INCLUDE))
    for i in range(0, len(uniq), chunk_size):
        batch = uniq[i : i + chunk_size]
        try:
            ch = _fetch_yf(batch, start=START)
            if not ch.empty:
                chunks.append(ch)
        except Exception as e:
            print(f"  chunk {i}: {e}", flush=True)
    if not chunks:
        raise RuntimeError(f"No data for {name}")
    df = pd.concat(chunks, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    # Drop columns with too few obs
    df = df.loc[:, df.notna().sum() >= 60]
    df.to_parquet(out, compression="zstd")
    print(f"[{name}] wrote {out.name} shape={df.shape} range={df.index.min().date()} → {df.index.max().date()}", flush=True)
    return df


def main():
    force = bool(int(os.environ.get("FORCE", "0")))
    print("=== fetch_prices ===")
    fetch_universe("broad", BROAD_ETFS, force=force)
    fetch_universe("levered", LEVERAGED_ETFS, force=force)
    fetch_universe("combined", combined_universe(), force=force)
    print("done.")


if __name__ == "__main__":
    main()
