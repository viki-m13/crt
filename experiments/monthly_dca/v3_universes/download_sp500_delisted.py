"""
Download price history for S&P 500 historical members that are NOT in our existing panel.

These are mostly delisted / acquired / renamed tickers. Yahoo Finance keeps
data for many of them because they were major US listings.

Saves:
  - data/sp500_delisted_prices.parquet — daily closes for each delisted ticker
  - data/sp500_delisted_download_log.csv — fetch attempt log (success / failure / n_obs)

Run:
    python3 -m experiments.monthly_dca.v3_universes.download_sp500_delisted
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
DATA = ROOT / "experiments" / "monthly_dca" / "v3_universes" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def main():
    panel = pd.read_parquet(CACHE / "prices_extended.parquet")
    panel_tickers = set(panel.columns)

    mem = pd.read_parquet(DATA / "sp500_pit_membership.parquet")
    sp500_ever = set(mem["ticker"].unique())
    missing = sorted(sp500_ever - panel_tickers)
    print(f"Need to download {len(missing)} tickers")

    log_rows = []
    series_map: dict[str, pd.Series] = {}

    for i, tkr in enumerate(missing):
        if i % 25 == 0:
            print(f"  [{i}/{len(missing)}] {tkr}...")
        try:
            t = yf.Ticker(tkr)
            # auto_adjust=True for split/dividend adjustment
            h = t.history(period="max", interval="1d", auto_adjust=True, actions=False)
            if h.empty or "Close" not in h.columns:
                log_rows.append({"ticker": tkr, "status": "empty", "n_obs": 0,
                                 "first_date": None, "last_date": None})
                continue
            s = h["Close"].dropna()
            if len(s) < 30:
                log_rows.append({"ticker": tkr, "status": "too_short", "n_obs": len(s),
                                 "first_date": str(s.index.min().date()),
                                 "last_date": str(s.index.max().date())})
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
            s.name = tkr
            series_map[tkr] = s
            log_rows.append({"ticker": tkr, "status": "ok", "n_obs": len(s),
                             "first_date": str(s.index.min().date()),
                             "last_date": str(s.index.max().date())})
        except Exception as e:
            log_rows.append({"ticker": tkr, "status": "error", "n_obs": 0,
                             "first_date": None, "last_date": None, "err": str(e)[:100]})
        if (i + 1) % 50 == 0:
            time.sleep(1.0)  # gentle on Yahoo

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(DATA / "sp500_delisted_download_log.csv", index=False)
    print(f"\nLog saved. Status counts:")
    print(log_df["status"].value_counts())

    if series_map:
        # Concat into a single panel
        delisted_panel = pd.concat(series_map, axis=1).sort_index()
        delisted_panel.to_parquet(DATA / "sp500_delisted_prices.parquet")
        print(f"\nSaved {delisted_panel.shape[1]} ticker series to sp500_delisted_prices.parquet")
        print(f"  Date range: {delisted_panel.index.min()} - {delisted_panel.index.max()}")
        print(f"  Total non-NaN cells: {delisted_panel.notna().sum().sum():,}")
    else:
        print("No data downloaded.")


if __name__ == "__main__":
    main()
