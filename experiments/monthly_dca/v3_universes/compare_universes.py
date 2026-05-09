"""
Compare strategy performance across universes.

Reads summary.json from each universe's cache directory and produces a
comparison table + CSV.

Run: python3 -m experiments.monthly_dca.v3_universes.compare_universes
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache" / "v3_universes"
OUT = ROOT / "experiments" / "monthly_dca" / "v3_universes"


# v2 baseline (full 1,833-ticker US universe, from existing cache)
BASELINE_SUMMARY = {
    "universe": "v2_baseline_full_us",
    "n_months": 256,
    "CAGR_pct": 80.79,
    "Sharpe": 1.469,
    "MaxDD_pct": -45.02,
    "Final_equity": 306452.81,
    "win_rate_months_pct": 67.6,
    "n_positive_years": 20,
    "n_total_years": 22,
    "worst_year_pct": -31.7,
    "best_year_pct": 874.1,
    "K_normal": 15,
    "K_recovery": 7,
    "K_bull": 7,
    "regime_mode": "tight",
}


def main():
    rows = [BASELINE_SUMMARY]
    for sub in sorted(CACHE.iterdir()):
        if not sub.is_dir():
            continue
        sj = sub / "summary.json"
        if sj.exists():
            with open(sj) as f:
                d = json.load(f)
            rows.append(d)

    df = pd.DataFrame(rows)
    cols = ["universe", "n_months", "CAGR_pct", "Sharpe", "MaxDD_pct",
            "win_rate_months_pct", "n_positive_years", "n_total_years",
            "worst_year_pct", "best_year_pct", "Final_equity"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("CAGR_pct", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "comparison.csv", index=False)
    print("=== Universe comparison (sorted by CAGR) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
