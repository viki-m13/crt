"""Build alternative forward targets at weekly cadence: 1w, 2w, 4w, 8w
forward returns for each (asof, ticker). Used by alternate GBM fits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PRICES_PATH = ROOT / "experiments" / "monthly_dca" / "cache" / "prices_extended.parquet"
WEEKLY_CACHE = Path(__file__).resolve().parent / "cache"


def main():
    feat = pd.read_parquet(WEEKLY_CACHE / "features_weekly.parquet",
                            columns=["asof", "ticker"])
    weekly_asofs = pd.DatetimeIndex(sorted(feat["asof"].unique()))
    px = pd.read_parquet(PRICES_PATH).reindex(weekly_asofs).ffill(limit=2)

    targets = pd.DataFrame(index=feat.set_index(["asof", "ticker"]).index)
    for h in (1, 2, 4, 8):
        fwd = (px.shift(-h) / px) - 1.0
        long = fwd.stack(future_stack=True).rename(f"fwd_{h}w_ret").to_frame()
        targets = targets.join(long, how="left")
    targets = targets.reset_index()
    out = WEEKLY_CACHE / "targets_weekly.parquet"
    targets.to_parquet(out, index=False)
    print(f"saved {len(targets)} rows -> {out}")


if __name__ == "__main__":
    main()
