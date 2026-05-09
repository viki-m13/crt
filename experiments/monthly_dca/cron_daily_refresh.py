"""Cron-friendly daily refresh for the v3 PIT-S&P-500 strategy webapp.

Cheap to run (~30-60s):
  1. Rebuild prices.parquet from docs/data/tickers/*.json (force).
  2. Compute base + extra features for any month-end not yet cached
     (and refresh the most recent two months, which may have grown).
  3. Refresh PIT S&P 500 membership panel (extends through latest live month).
  4. Rebuild experiments/docs/monthly-dca/data.json with the v3 strategy.

Does NOT redo the full strategy sweep / walk-forward — those are slow,
historical (immutable), and live as static CSVs in
experiments/monthly_dca/cache/v2/sp500_pit/v3_*.csv. They were generated
once by the v3 sweep and validation pipeline and only need re-running
when the strategy logic or universe materially changes (re-run via
sp500_pit_strategy_sweep.py + sp500_pit_v3_validate.py).

The ML model (ml_preds_v2.parquet for WF backtest, ml_preds_live.parquet
for current-month picks) needs to be retrained annually — not in the
daily cron — by re-running experiments/monthly_dca/v2/ml_strategy.py.

Designed to be idempotent and fail-soft: if the panel cannot be
rebuilt, it leaves the existing data in place.
"""
from __future__ import annotations

import subprocess
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
    """Re-compute features for the lookback_months most recent month-ends."""
    me = month_end_dates(panel.index)
    target = me[-lookback_months:]
    if len(target) == 0:
        return
    start = target[0].strftime("%Y-%m-%d")
    end = target[-1].strftime("%Y-%m-%d")

    feat_dir = Path(__file__).resolve().parent / "cache" / "features"
    for d in target:
        p = feat_dir / f"{d.date()}.parquet"
        if p.exists():
            p.unlink()

    print(f"Refreshing features for {len(target)} month-ends: {start} → {end}")
    run_features(start=start, end=end)
    run_extras(start=start, end=end)


def refresh_pit_membership() -> None:
    """Re-roll the PIT S&P 500 membership panel forward through the latest
    live-pred month.  Idempotent — the membership history is immutable;
    this only extends to the current month-end if it isn't already there."""
    print("Refreshing PIT S&P 500 membership panel...")
    script = Path(__file__).resolve().parent / "v2" / "build_sp500_pit_membership.py"
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> int:
    try:
        # 1. Rebuild prices panel from latest tickers
        print("=== Step 1: Rebuilding prices panel ===")
        panel = run_load(force=True)
        print(f"  panel shape={panel.shape}  date range={panel.index.min().date()} → {panel.index.max().date()}")

        # 2. Compute features for any new month-ends, refresh recent ones
        print("\n=== Step 2: Refreshing features for recent month-ends ===")
        run_features(start="2017-01-01", end="2099-01-01")
        run_extras(start="2017-01-01", end="2099-01-01")
        refresh_recent_months(panel, lookback_months=3)

        # 3. Refresh PIT S&P 500 membership panel.  The S&P 500 changes
        # post-2019 are sourced from a small CSV that the user updates
        # manually; the daily cron rolls forward to the latest live-pred
        # month using the rolled-forward last-known-set.
        print("\n=== Step 3: Refreshing PIT S&P 500 membership ===")
        refresh_pit_membership()

        # 4. Rebuild webapp data.json with the v3 strategy.
        # Static walk-forward / bias / sub-period / sensitivity / generalise
        # CSVs live under cache/v2/sp500_pit/v3_*.csv and are ingested by
        # the builder. They only re-generate when v3 strategy logic changes.
        print("\n=== Step 4: Rebuilding webapp data.json (v3 PIT-S&P-500) ===")
        load_features_long.cache_clear()
        from experiments.monthly_dca.v2.build_webapp_v3_pit import main as build_v3
        build_v3()

        print("\nDaily refresh complete.")
        return 0
    except Exception as e:
        print(f"ERROR during daily refresh: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
