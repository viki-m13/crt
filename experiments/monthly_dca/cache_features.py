"""Precompute & cache per-month-end features for all eligible tickers.

Iterating on strategies becomes ~free because we just rescore cached feature frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.backtester import (
    compute_features,
    load_panel,
    month_end_dates,
)

OUT = Path(__file__).resolve().parent / "cache" / "features"


def main(start: str = "2002-01-01", end: str = "2099-01-01") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    print(f"Computing features for {len(months)} month-ends")
    for i, asof in enumerate(months):
        out = OUT / f"{asof.date()}.parquet"
        if out.exists():
            continue
        try:
            pack = compute_features(panel, asof)
        except Exception as e:
            print(f"  skip {asof.date()}: {e}")
            continue
        df = pack.df()
        df.index.name = "ticker"
        df.to_parquet(out)
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  [{i+1}/{len(months)}] {asof.date()} ({len(df)} tickers)")
    print(f"Cached features in {OUT}")


if __name__ == "__main__":
    main()
