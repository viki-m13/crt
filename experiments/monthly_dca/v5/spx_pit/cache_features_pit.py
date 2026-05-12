"""Phase 3: recompute per-month features for the augmented PIT universe.

Same compute_features() function the original v2 pipeline uses (from
experiments/monthly_dca/backtester.py), but the panel passed in is the
augmented daily panel from Phase 1. Writes to a parallel features
directory so the original feature cache is untouched.

Inputs:
  experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet
  experiments/monthly_dca/cache/features/                        (used to discover month-ends)

Outputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/features/<asof>.parquet

Compute: ~5-12 min for 353 month-ends on the 1994-ticker augmented panel.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG = PIT / "augmented"
ORIG_FEATURES = CACHE / "features"
OUT_FEATURES = AUG / "features"

sys.path.insert(0, str(ROOT))
from experiments.monthly_dca.backtester import compute_features  # noqa: E402


def main():
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 64)
    print("Phase 3: cache features for augmented PIT universe")
    print("=" * 64)

    panel = pd.read_parquet(PIT / "prices_extended_pit.parquet")
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)
    print(f"[1] augmented daily panel: {panel.shape} "
          f"({panel.index.min().date()}..{panel.index.max().date()})")

    # Use the SAME month-end dates as the original cache so downstream code
    # finds matching files. (Don't expand the date set.)
    months = sorted(pd.Timestamp(p.stem) for p in ORIG_FEATURES.glob("*.parquet"))
    print(f"[2] {len(months)} target month-ends from {months[0].date()} "
          f"to {months[-1].date()}")

    done = skipped = errored = 0
    err_list: list[tuple[str, str]] = []
    for i, asof in enumerate(months):
        out_path = OUT_FEATURES / f"{asof.date()}.parquet"
        if out_path.exists():
            skipped += 1
            continue
        try:
            pack = compute_features(panel, asof)
        except Exception as e:
            err_list.append((str(asof.date()), str(e)[:80]))
            errored += 1
            continue
        df = pack.df()
        df.index.name = "ticker"
        df.to_parquet(out_path)
        done += 1
        if (i + 1) % 24 == 0 or i == len(months) - 1:
            print(f"    [{i+1}/{len(months)}] {asof.date()}  "
                  f"({len(df)} tickers, elapsed={time.time()-t0:.1f}s)")

    print(f"\n[3] Done in {time.time()-t0:.1f}s")
    print(f"      wrote: {done}, skipped: {skipped}, errored: {errored}")
    if err_list:
        print(f"      first 5 errors: {err_list[:5]}")

    # Compare ticker counts vs original
    sample = pd.Timestamp("2010-12-31")
    if (OUT_FEATURES / f"{sample.date()}.parquet").exists():
        new_df = pd.read_parquet(OUT_FEATURES / f"{sample.date()}.parquet")
        orig_df = pd.read_parquet(ORIG_FEATURES / f"{sample.date()}.parquet")
        print(f"\n[4] sample asof {sample.date()}:")
        print(f"      original features: {orig_df.shape}")
        print(f"      augmented features: {new_df.shape}")
        new_t = set(new_df.index)
        orig_t = set(orig_df.index)
        print(f"      added tickers (sample): {sorted(new_t - orig_t)[:15]}")


if __name__ == "__main__":
    main()
