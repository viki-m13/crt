"""
Build the international universe panel:
  - intl_prices.parquet (524 individual international stocks across 10 countries)
  - SPY appended for the regime gate

Saves: cache/v3_universes/intl/prices.parquet

Note: this universe has SURVIVORSHIP BIAS (constituents are current major-index
names). For a more honest test we'd need historical international index
membership which is not freely available. Documented in REPORT.

Run: python3 -m experiments.monthly_dca.v3_universes.build_intl_panel
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
DATA = ROOT / "experiments" / "monthly_dca" / "v3_universes" / "data"
OUT = CACHE / "v3_universes" / "intl"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading international panel...")
    intl = pd.read_parquet(DATA / "intl_prices.parquet")
    print(f"  Intl: {intl.shape}")

    print("Loading US panel for SPY...")
    us = pd.read_parquet(CACHE / "prices_extended.parquet")
    spy = us["SPY"].rename("SPY")

    # Combine
    combined = intl.copy()
    combined["SPY"] = spy.reindex(combined.index)
    # Forward-fill SPY where intl has trading days but US doesn't
    combined["SPY"] = combined["SPY"].ffill()

    combined.to_parquet(OUT / "prices.parquet")
    print(f"Combined: {combined.shape}")
    print(f"Date range: {combined.index.min()} - {combined.index.max()}")

    # Coverage stats
    coverage = {
        "n_tickers": int(combined.shape[1]),
        "n_intl_tickers": int(intl.shape[1]),
        "date_range": f"{combined.index.min().date()} - {combined.index.max().date()}",
        "n_days": int(combined.shape[0]),
    }
    # Per-country counts
    tlist = pd.read_csv(DATA / "intl_tickers_list.csv")
    in_panel = tlist[tlist["ticker"].isin(intl.columns)]
    coverage["per_country"] = in_panel.groupby("country").size().to_dict()
    with open(OUT / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)
    print("\n=== Coverage ===")
    for k, v in coverage.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
