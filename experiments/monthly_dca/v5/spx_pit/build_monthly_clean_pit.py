"""Phase 2: build augmented monthly_prices / monthly_returns from the
augmented daily panel built in Phase 1.

Mirrors experiments/monthly_dca/v2/build_dataset.py:_detect_bad_months and
build_clean_monthly, but reads the augmented daily panel instead of the
original prices_extended.parquet. Writes outputs to a parallel directory
so the original v5 pipeline stays intact for side-by-side comparison.

Inputs:
  experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet

Outputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/monthly_prices_clean.parquet
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/monthly_returns_clean.parquet
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/bad_data_tickers.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
OUT = PIT / "augmented"


def _detect_bad_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Replica of v2/build_dataset.py._detect_bad_months.

    Detects 80%+ drop followed by 200%+ surge within 3 months (ticker reuse
    signature), or 200%+ surge followed by 80%+ drop. Around each detected
    cell, masks ±3 months.
    """
    mret = monthly.pct_change()

    extreme_drops = mret < -0.80
    mret_max_3m_ahead = mret.rolling(3).max().shift(-3)
    sig1 = extreme_drops & (mret_max_3m_ahead > 2.0)

    extreme_surge = mret > 2.0
    mret_min_3m_ahead = mret.rolling(3).min().shift(-3)
    sig2 = extreme_surge & (mret_min_3m_ahead < -0.8)

    reuse = sig1 | sig2
    counts = reuse.sum().sort_values(ascending=False)

    mask_bad = pd.DataFrame(False, index=monthly.index, columns=monthly.columns)
    for tk in counts[counts > 0].index:
        sig = reuse[tk]
        bad_dates = sig[sig].index
        for d in bad_dates:
            i = monthly.index.get_loc(d)
            lo = max(0, i - 3)
            hi = min(len(monthly.index) - 1, i + 3)
            for j in range(lo, hi + 1):
                mask_bad.iat[j, monthly.columns.get_loc(tk)] = True
    return mask_bad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 64)
    print("Phase 2: rebuild monthly clean panels (augmented PIT universe)")
    print("=" * 64)

    panel = pd.read_parquet(PIT / "prices_extended_pit.parquet")
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)
    print(f"[1] daily panel: {panel.shape} "
          f"({panel.index.min().date()}..{panel.index.max().date()})")

    monthly = panel.resample("ME").last()
    print(f"[2] monthly resampled: {monthly.shape}")

    mask_bad = _detect_bad_months(monthly)
    n_bad_cells = int(mask_bad.values.sum())
    n_bad_tickers = int(mask_bad.any(axis=0).sum())
    print(f"[3] bad-data cells masked: {n_bad_cells} "
          f"across {n_bad_tickers} tickers")

    monthly_clean = monthly.where(~mask_bad)
    mret_clean = monthly_clean.pct_change().clip(lower=-1.0, upper=2.0)

    bad_list = sorted(
        mask_bad.any(axis=0)[mask_bad.any(axis=0)].index.tolist()
    )
    with open(OUT / "bad_data_tickers.json", "w") as f:
        json.dump(bad_list, f, indent=2)

    monthly_clean.to_parquet(OUT / "monthly_prices_clean.parquet")
    mret_clean.to_parquet(OUT / "monthly_returns_clean.parquet")

    print(f"[4] wrote:")
    print(f"      {OUT / 'monthly_prices_clean.parquet'}")
    print(f"      {OUT / 'monthly_returns_clean.parquet'}")
    print(f"      {OUT / 'bad_data_tickers.json'}")

    # Compare to original
    orig_mp = pd.read_parquet(CACHE / "v2" / "monthly_prices_clean.parquet")
    orig_cols = set(orig_mp.columns)
    new_cols = set(monthly_clean.columns)
    added = new_cols - orig_cols
    dropped = orig_cols - new_cols
    print(f"\n[5] vs original monthly_prices_clean:")
    print(f"      original cols: {len(orig_cols)}")
    print(f"      new cols:      {len(new_cols)}  (+{len(added)} added, "
          f"-{len(dropped)} removed)")
    print(f"      added sample:  {sorted(added)[:20]}")
    if dropped:
        print(f"      dropped:       {sorted(dropped)[:20]}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
