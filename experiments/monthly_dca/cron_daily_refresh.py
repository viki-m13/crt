"""Cron-friendly daily refresh for the monthly-DCA strategy webapp.

Cheap to run (~30-60s):
  1. Rebuild prices.parquet from docs/data/tickers/*.json (force).
  2. Compute base + extra features for any month-end not yet cached
     (and refresh the most recent two months, which may have grown).
  3. Rebuild experiments/docs/monthly-dca/data.json.

Does NOT redo the full sweep / walk-forward — those are slow and
historical (immutable). The webapp page reads the static aggregate
CSVs already committed; only `live_picks` and the latest `as_of`
need to refresh daily.

Designed to be idempotent and fail-soft: if the panel cannot be
rebuilt, it leaves the existing data in place.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.backtester import month_end_dates
from experiments.monthly_dca.cache_features import main as run_features
from experiments.monthly_dca.extra_features import main as run_extras
from experiments.monthly_dca.load_data import main as run_load
from experiments.monthly_dca.fast_score import load_features_long


def refresh_recent_months(panel: pd.DataFrame, lookback_months: int = 3) -> None:
    """Re-compute features for the lookback_months most recent month-ends.

    The latest month-end keeps growing as new daily data lands inside the
    current month, so we re-compute it (and the previous two) every run.
    Older months are immutable price-history -> no need to recompute.
    """
    me = month_end_dates(panel.index)
    target = me[-lookback_months:]
    if len(target) == 0:
        return
    start = target[0].strftime("%Y-%m-%d")
    end = target[-1].strftime("%Y-%m-%d")

    # Force-rebuild the recent months by deleting their parquets first
    feat_dir = Path(__file__).resolve().parent / "cache" / "features"
    for d in target:
        p = feat_dir / f"{d.date()}.parquet"
        if p.exists():
            p.unlink()

    print(f"Refreshing features for {len(target)} month-ends: {start} → {end}")
    run_features(start=start, end=end)
    run_extras(start=start, end=end)


def main() -> int:
    try:
        # 1. Rebuild prices panel from latest tickers
        print("=== Step 1: Rebuilding prices panel ===")
        panel = run_load(force=True)
        print(f"  panel shape={panel.shape}  date range={panel.index.min().date()} → {panel.index.max().date()}")

        # 2. Compute features for any new month-ends, refresh recent ones
        print("\n=== Step 2: Refreshing features for recent month-ends ===")
        # Cover everything since 2017 — incremental (skips already-cached unless they're in the recent window)
        run_features(start="2017-01-01", end="2099-01-01")
        run_extras(start="2017-01-01", end="2099-01-01")
        # Always re-run the most recent 3 month-ends (they may have grown intra-month)
        refresh_recent_months(panel, lookback_months=3)

        # 3. Rebuild webapp data.json
        print("\n=== Step 3: Rebuilding webapp data.json ===")
        load_features_long.cache_clear()  # reset lru cache
        from experiments.monthly_dca.build_webapp_json import main as build_json
        build_json()

        print("\nDaily refresh complete.")
        return 0
    except Exception as e:
        print(f"ERROR during daily refresh: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
