"""Phase 4: re-train the v2 walk-forward GBM on the augmented PIT cross-section.

Just imports fit_walkforward() from the original ml_strategy.py and calls
it with the augmented panel. The training procedure (HistGradientBoosting
per-horizon, retrained every January, 7-month embargo) is unchanged —
only the input data is richer.

Inputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/panel_cross_section_v3.parquet

Outputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/ml_preds.parquet

Run time: ~1 hour (CPU; same as the original v2 training).
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

sys.path.insert(0, str(ROOT))
from experiments.monthly_dca.v2.ml_strategy import fit_walkforward  # noqa: E402


def main():
    t0 = time.time()
    print("=" * 64)
    print("Phase 4: walk-forward GBM training (augmented PIT cross-section)")
    print("=" * 64)

    panel_path = AUG / "panel_cross_section_v3.parquet"
    print(f"[1] loading {panel_path} ...")
    big = pd.read_parquet(panel_path)
    print(f"    cross-section shape: {big.shape}, "
          f"tickers: {big.reset_index()['ticker'].nunique()}, "
          f"asofs: {big.reset_index()['asof'].nunique()}")

    print(f"[2] fitting walk-forward (HistGBM, 7-month embargo, retrain every Jan) ...")
    # fit_walkforward defaults train_end to 2025-12-31; that hard cap
    # would freeze live predictions. Extend it to the panel's latest
    # asof so the current/live months are always scored (predictions
    # need no realized target — only training rows do). This is what
    # advances the deployed `as_of` to the current month.
    panel_last = pd.to_datetime(
        big.reset_index()["asof"]).max().normalize()
    print(f"    extending train_end -> {panel_last.date()} "
          f"(was the 2025-12-31 default cap)")
    preds = fit_walkforward(big, target_horizons=(1, 3, 6),
                            train_end=panel_last)
    print(f"    predictions: {len(preds)}")

    out_path = AUG / "ml_preds.parquet"
    preds.to_parquet(out_path)
    print(f"[3] saved -> {out_path}")
    print(f"Done in {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
