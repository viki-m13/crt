"""Pick the most robust strategy (smallest worst-year SPY underperformance)
on the extended 2002-2024 history.

For each (strategy, top_k, exit_rule):
  - Compute year-by-year CAGR vs SPY DCA
  - Score by: edge_vs_spy_dca - 2 * (worst_year_underperformance vs SPY)
    This penalizes massive bad years.
  - Filter: must beat SPY DCA in >=70% of years
  - Filter: minimum 80 picks (across the window)

Output to cache/recommended_strategy.json so the webapp can use it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"


def main() -> None:
    sweep_path = CACHE / "sweep_extended.csv"
    yedges_path = CACHE / "year_edges_extended.csv"
    if not sweep_path.exists() or not yedges_path.exists():
        print("Run run_extended.py first")
        return

    sweep = pd.read_csv(sweep_path)
    yedges = pd.read_csv(yedges_path)

    # Per-strategy yearly stats
    agg = yedges.groupby(["strategy", "top_k", "exit"]).agg(
        n_years=("year", "nunique"),
        min_year_edge=("edge", "min"),
        median_year_edge=("edge", "median"),
        n_bad_years=("edge", lambda s: (s < -0.10).sum()),
        n_terrible_years=("edge", lambda s: (s < -0.20).sum()),
        n_great_years=("edge", lambda s: (s > 0.30).sum()),
        years_beating_spy=("edge", lambda s: (s > 0).sum()),
        min_year_cagr=("cagr_dca_picks", "min"),
        worst_year=("year", lambda s: int(yedges.loc[s.index, "year"][yedges.loc[s.index, "edge"].idxmin()])
                    if not s.empty else 0),
    ).reset_index()

    agg["pct_years_beating_spy"] = agg["years_beating_spy"] / agg["n_years"]

    # Join with full sweep
    full = agg.merge(
        sweep[["strategy", "top_k", "exit", "n_picks", "win_rate",
               "win_rate_bias_corr", "cagr_dca_portfolio", "cagr_spy_dca",
               "edge_vs_spy_dca"]],
        on=["strategy", "top_k", "exit"],
        how="inner",
    )
    full = full[full["n_picks"] >= 80].copy()
    full = full[full["pct_years_beating_spy"] >= 0.70].copy()
    full = full[full["n_terrible_years"] == 0].copy()
    full["robust_score"] = full["edge_vs_spy_dca"] + 2.0 * full["min_year_edge"]
    full = full.sort_values("robust_score", ascending=False)
    full.to_csv(CACHE / "robust_ranking.csv", index=False)

    cols = ["strategy", "top_k", "exit", "n_picks", "n_years",
            "pct_years_beating_spy", "min_year_edge", "n_bad_years",
            "n_terrible_years", "cagr_dca_portfolio", "cagr_spy_dca",
            "edge_vs_spy_dca", "robust_score"]
    print("=== ROBUST STRATEGIES (>=70% years beat SPY, no year worse than -20% vs SPY) ===")
    if full.empty:
        print("  none — relaxing constraints")
        # Fallback: relax constraints
        full = agg.merge(sweep[["strategy", "top_k", "exit", "n_picks", "win_rate",
                                "cagr_dca_portfolio", "cagr_spy_dca",
                                "edge_vs_spy_dca"]],
                          on=["strategy", "top_k", "exit"], how="inner")
        full = full[full["n_picks"] >= 80].copy()
        full["robust_score"] = full["edge_vs_spy_dca"] + 2.0 * full["min_year_edge"]
        full = full.sort_values("robust_score", ascending=False)
        full.to_csv(CACHE / "robust_ranking.csv", index=False)
        print(full.head(15)[cols + (["win_rate_bias_corr"] if "win_rate_bias_corr" in full.columns else [])].to_string(index=False))
    else:
        print(full.head(15)[cols].to_string(index=False))

    # Pick the top one as recommended
    if not full.empty:
        best = full.iloc[0]
        rec = {
            "strategy": best["strategy"],
            "top_k": int(best["top_k"]),
            "exit": best["exit"],
            "n_picks": int(best["n_picks"]),
            "cagr_dca_portfolio": float(best["cagr_dca_portfolio"]),
            "cagr_spy_dca": float(best["cagr_spy_dca"]),
            "edge_vs_spy_dca": float(best["edge_vs_spy_dca"]),
            "min_year_edge": float(best["min_year_edge"]),
            "n_bad_years": int(best["n_bad_years"]),
            "n_terrible_years": int(best["n_terrible_years"]),
            "pct_years_beating_spy": float(best["pct_years_beating_spy"]),
            "robust_score": float(best["robust_score"]),
        }
        with open(CACHE / "recommended_strategy.json", "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nRECOMMENDED: {rec}")


if __name__ == "__main__":
    main()
