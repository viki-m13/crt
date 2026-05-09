"""
Run walk-forward on the already-built SP500 cross-section.

Reads the pre-built panel_cross_section_v3.parquet (from build_sp500_extended_panel)
and runs walk-forward + simulate.

Run: python3 -m experiments.monthly_dca.v3_universes.run_sp500_only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.monthly_dca.v3_universes.run_universe import (
    fit_walkforward, simulate_universe,
)

OUT = ROOT / "experiments" / "monthly_dca" / "cache" / "v3_universes" / "sp500_pit"


def main():
    print("=== Run SP500 PIT walk-forward ===")
    big = pd.read_parquet(OUT / "panel_cross_section_v3.parquet")
    monthly_returns = pd.read_parquet(OUT / "monthly_returns_clean.parquet")
    print(f"  Cross-section: {big.shape}")

    print("[1/2] Walk-forward fit + simulate...")
    preds = fit_walkforward(big, train_start="2003-01-01", train_end="2024-12-31",
                              min_train_rows=10000, min_train_per_target=2500)
    if preds.empty:
        print("ERROR: empty preds")
        return
    preds.to_parquet(OUT / "ml_preds.parquet")
    print(f"  Predictions: {len(preds)}")

    print("[2/2] Simulating equity curve...")
    eq, summary = simulate_universe(
        preds, big, monthly_returns,
        K_normal=15, K_recovery=7, K_bull=7,
        regime_mode="tight", year_min=2003, year_max=2024,
    )
    if eq.empty:
        print("ERROR: empty equity")
        return
    eq.to_csv(OUT / "equity_curve.csv", index=False)
    yr_df = pd.DataFrame(list(summary["year_by_year"].items()), columns=["year", "ret_pct"])
    yr_df.to_csv(OUT / "year_by_year.csv", index=False)
    summary["universe"] = "sp500_pit"
    summary["K_normal"] = 15
    summary["K_recovery"] = 7
    summary["K_bull"] = 7
    summary["regime_mode"] = "tight"
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (sp500_pit) ===")
    for k in ("CAGR_pct", "Sharpe", "MaxDD_pct", "Final_equity",
              "win_rate_months_pct", "n_positive_years", "n_total_years",
              "worst_year_pct", "best_year_pct"):
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()
