"""
Combine the existing US-equities panel + the delisted-S&P-500 backfill
into a single panel covering all S&P 500 historical members where data exists.

Saves:
  - cache/v3_universes/sp500_pit/prices.parquet — daily price panel
  - cache/v3_universes/sp500_pit/coverage.json — coverage stats

Run:
    python3 -m experiments.monthly_dca.v3_universes.build_sp500_panel
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
DATA = ROOT / "experiments" / "monthly_dca" / "v3_universes" / "data"
OUT = CACHE / "v3_universes" / "sp500_pit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading existing US panel...")
    panel_us = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"  US panel: {panel_us.shape}")

    print("Loading delisted S&P 500 backfill...")
    delisted = pd.read_parquet(DATA / "sp500_delisted_prices.parquet")
    print(f"  Delisted backfill: {delisted.shape}")

    print("Loading S&P 500 PIT membership...")
    mem = pd.read_parquet(DATA / "sp500_pit_membership.parquet")
    sp500_ever = sorted(mem["ticker"].unique())
    print(f"  S&P 500 ever-members (1996-2026): {len(sp500_ever)}")

    # Combine: take SP500-ever tickers from US panel where available, else delisted backfill
    cols_us = [t for t in sp500_ever if t in panel_us.columns]
    cols_dl = [t for t in sp500_ever if t in delisted.columns and t not in cols_us]
    print(f"  From US panel: {len(cols_us)}")
    print(f"  From delisted backfill: {len(cols_dl)}")

    parts = [panel_us[cols_us]]
    if cols_dl:
        parts.append(delisted[cols_dl])
    # Always include SPY for regime gate (it's in panel_us)
    if "SPY" not in cols_us and "SPY" in panel_us.columns:
        parts.append(panel_us[["SPY"]])

    combined = pd.concat(parts, axis=1).sort_index()
    # Deduplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated()]
    print(f"  Combined panel: {combined.shape}")
    print(f"  Date range: {combined.index.min()} - {combined.index.max()}")

    combined.to_parquet(OUT / "prices.parquet")

    # Symlink the membership file for run_universe convenience
    membership_link = OUT / "membership.parquet"
    if not membership_link.exists():
        # Just copy
        mem.to_parquet(membership_link)

    coverage = {
        "sp500_ever_members": len(sp500_ever),
        "in_us_panel": len(cols_us),
        "in_delisted_backfill": len(cols_dl),
        "total_in_combined_panel": combined.shape[1],
        "missing_completely": len(sp500_ever) - len(cols_us) - len(cols_dl),
        "coverage_pct": round(100 * (len(cols_us) + len(cols_dl)) / len(sp500_ever), 2),
        "date_range": f"{combined.index.min().date()} - {combined.index.max().date()}",
    }
    with open(OUT / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"\n=== Coverage ===")
    for k, v in coverage.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
