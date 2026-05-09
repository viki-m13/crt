"""
Fast universe runner: reuses the existing v2 feature cache instead of re-computing.

For US universes that are subsets of the 1,833-ticker v2 panel, this is much
faster (skips the 30-60 min feature compute step). The trade-off: we use the
features that were computed on the full 1,833-ticker panel, NOT recomputed on
the universe-restricted panel. This means:
  - Cross-sectional ranks were computed across the FULL panel originally
    (we re-rank in the model fit step using only universe members, so
    that's fine for the model).
  - Per-ticker time-series features (momentum, vol, recovery_rate) are
    identical regardless of universe — same prices, same compute.
  - The cross-sectional features that compare to SPY (rs_*_spy) use SPY
    which is unchanged.

So reusing the cache is *equivalent* for these features.

Usage:
    python3 -m experiments.monthly_dca.v3_universes.run_universe_fast \\
        --name sp500_pit \\
        --membership experiments/monthly_dca/v3_universes/data/sp500_pit_membership.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.monthly_dca.v2.ml_strategy import (
    build_strategy_outputs, simulate_strategy, cagr,
)
from experiments.monthly_dca.v3_universes.run_universe import (
    fit_walkforward, simulate_universe,
)

UV_CACHE_BASE = ROOT / "experiments" / "monthly_dca" / "cache" / "v3_universes"
V2_CACHE = ROOT / "experiments" / "monthly_dca" / "cache" / "v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--membership", default=None,
                    help="Optional PIT membership parquet")
    ap.add_argument("--K-normal", type=int, default=15)
    ap.add_argument("--K-recovery", type=int, default=7)
    ap.add_argument("--K-bull", type=int, default=7)
    ap.add_argument("--year-min", type=int, default=2003)
    ap.add_argument("--year-max", type=int, default=2024)
    ap.add_argument("--regime-mode", default="tight")
    ap.add_argument("--min-train-rows", type=int, default=10000)
    args = ap.parse_args()

    out_dir = UV_CACHE_BASE / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Fast universe run: {args.name} ===")

    # Load v2 cross-section + cleaned monthly returns
    big = pd.read_parquet(V2_CACHE / "panel_cross_section_v3.parquet")
    monthly_returns = pd.read_parquet(V2_CACHE / "monthly_returns_clean.parquet")
    print(f"  v2 cross-section: {big.shape}")

    membership = None
    if args.membership:
        membership = pd.read_parquet(args.membership)
        print(f"  Membership: {len(membership)} (date,ticker) rows")

    # Apply membership filter
    if membership is not None:
        membership["date"] = pd.to_datetime(membership["date"])
        # For each asof, keep only members
        big_flat = big.reset_index()
        big_flat["asof"] = pd.to_datetime(big_flat["asof"])
        # Build membership lookup (date -> set of tickers)
        mem_by_date = {}
        for d, gd in membership.groupby("date"):
            mem_by_date[pd.Timestamp(d)] = set(gd["ticker"].unique())
        mem_dates = sorted(mem_by_date.keys())

        def is_member(row):
            d = row["asof"]
            t = row["ticker"]
            if t == "SPY":  # always keep SPY for regime gate
                return True
            pos = np.searchsorted(mem_dates, d, side="right") - 1
            if pos < 0:
                return False
            return t in mem_by_date[mem_dates[pos]]

        # Vectorized filter: build a mask
        keep_mask = []
        # group by asof for efficiency
        for d, gd in big_flat.groupby("asof"):
            d = pd.Timestamp(d)
            pos = np.searchsorted(mem_dates, d, side="right") - 1
            if pos < 0:
                members = set()
            else:
                members = mem_by_date[mem_dates[pos]] | {"SPY"}
            mask_g = gd["ticker"].isin(members)
            keep_mask.append(pd.Series(mask_g.values, index=gd.index))
        keep_mask = pd.concat(keep_mask).sort_index()
        big_flat = big_flat[keep_mask.values]
        big = big_flat.set_index(["asof", "ticker"])
        print(f"  Filtered cross-section: {big.shape}")

    # Save filtered cross-section
    big.to_parquet(out_dir / "panel_cross_section_v3.parquet")

    print("[1/2] Walk-forward fit + simulate...")
    preds = fit_walkforward(
        big, train_start=f"{args.year_min}-01-01",
        train_end=f"{args.year_max}-12-31",
        min_train_rows=args.min_train_rows,
    )
    if preds.empty:
        print("ERROR: empty predictions")
        return
    preds.to_parquet(out_dir / "ml_preds.parquet")
    print(f"  Predictions: {len(preds)}")

    print("[2/2] Simulating equity curve...")
    eq, summary = simulate_universe(
        preds, big, monthly_returns,
        K_normal=args.K_normal, K_recovery=args.K_recovery, K_bull=args.K_bull,
        regime_mode=args.regime_mode,
        year_min=args.year_min, year_max=args.year_max,
    )
    if eq.empty:
        print("ERROR: empty equity curve")
        return
    eq.to_csv(out_dir / "equity_curve.csv", index=False)
    yr_df = pd.DataFrame(list(summary["year_by_year"].items()), columns=["year", "ret_pct"])
    yr_df.to_csv(out_dir / "year_by_year.csv", index=False)
    summary["universe"] = args.name
    summary["K_normal"] = args.K_normal
    summary["K_recovery"] = args.K_recovery
    summary["K_bull"] = args.K_bull
    summary["regime_mode"] = args.regime_mode
    summary["fast_path"] = True
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ({args.name}) ===")
    print(f"  CAGR: {summary['CAGR_pct']}%")
    print(f"  Sharpe: {summary['Sharpe']}")
    print(f"  MaxDD: {summary['MaxDD_pct']}%")
    print(f"  Final equity: {summary['Final_equity']}")
    print(f"  Win rate (months): {summary['win_rate_months_pct']}%")
    print(f"  Positive years: {summary['n_positive_years']}/{summary['n_total_years']}")
    print(f"  Worst year: {summary['worst_year_pct']}%, Best: {summary['best_year_pct']}%")


if __name__ == "__main__":
    main()
