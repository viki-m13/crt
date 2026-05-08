"""Run the full sweep + walk-forward on the extended 2002-2026 panel.

Goal: find the strategy (among our 16) at top_k ∈ {1, 5, 10} and exit ∈
{hold_forever, fixed_3y, fixed_5y} that has:
  - High DCA-portfolio CAGR
  - Few "bad years" (entry-year cohorts where edge vs SPY DCA is < -10%)
  - Robust across walk-forward splits

Output:
  - cache/sweep_extended.csv  (all combos)
  - cache/year_edges_extended.csv  (per-strategy / per-year edge vs SPY)
  - cache/wf_extended_aggregate.csv  (walk-forward over the longer window)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED,
    evaluate_strategy,
    load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.deepdive import merge_fwd, picks_for, per_year_breakdown
from experiments.monthly_dca.strategies_fast import all_strategies


CACHE = Path(__file__).resolve().parent / "cache"


def per_year_edges(name: str, fn, top_k: int, panel: pd.DataFrame,
                   start: str, end: str, eval_at: pd.Timestamp,
                   ret_col: str = "ret__hold_forever") -> pd.DataFrame:
    picks = picks_for(fn, top_k=top_k, start=start, end=end)
    if picks.empty:
        return pd.DataFrame()
    merged = merge_fwd(picks)
    yb = per_year_breakdown(merged, ret_col, panel, eval_at)
    yb["strategy"] = name
    yb["top_k"] = top_k
    yb["exit"] = ret_col.replace("ret__", "")
    return yb


def main() -> None:
    panel = load_panel()
    print(f"Panel: {panel.shape}  date range {panel.index.min().date()} → {panel.index.max().date()}")
    eval_at = panel.index.max()

    # Window: start when we have 2y of feature lookback (need >= 504 days history)
    # Panel starts 2000-01, so feature month-ends start ~2002-01.
    # End: 2024-12 to leave 1+ year of forward data.
    START = "2002-01-31"
    END = "2024-12-31"

    print(f"\n=== Sweep over extended window: {START} → {END} ===")
    summaries = []
    year_rows = []
    for top_k in (1, 5, 10):
        for strat in all_strategies(top_k=top_k):
            try:
                er = evaluate_strategy(strat.score_fn, top_k=top_k, name=strat.name,
                                       start=START, end=END, panel=panel,
                                       delist_iters=50)
            except Exception as e:
                print(f"  ERROR {strat.name} k={top_k}: {e}")
                continue
            if er.summary.empty:
                continue
            summ = er.summary.copy()
            summ["top_k"] = top_k
            summaries.append(summ)
            # Per-year for hold_forever and fixed_3y
            for ret_col in ("ret__hold_forever", "ret__fixed_3y"):
                yb = per_year_edges(strat.name, strat.score_fn, top_k, panel,
                                    START, END, eval_at, ret_col)
                if not yb.empty:
                    year_rows.append(yb)
        print(f"  done top_k={top_k}")

    big = pd.concat(summaries, ignore_index=True)
    big.to_csv(CACHE / "sweep_extended.csv", index=False)
    print(f"\nWrote {CACHE/'sweep_extended.csv'}: {big.shape}")

    yedges = pd.concat(year_rows, ignore_index=True) if year_rows else pd.DataFrame()
    if not yedges.empty:
        yedges.to_csv(CACHE / "year_edges_extended.csv", index=False)
        print(f"Wrote {CACHE/'year_edges_extended.csv'}: {yedges.shape}")

    cols = ["strategy", "top_k", "exit", "n_picks", "win_rate", "win_rate_bias_corr",
            "beat_spy_rate", "median_ret", "cagr_dca_portfolio", "cagr_spy_dca",
            "edge_vs_spy_dca"]

    # Filter to meaningful results: at least 50 picks (otherwise it's a degenerate exit rule)
    big = big[big["n_picks"] >= 50]

    print("\n=== TOP 30 BY DCA-PORTFOLIO CAGR (n_picks >= 50) ===")
    print(big.sort_values("cagr_dca_portfolio", ascending=False).head(30)[cols].to_string(index=False))

    # For each (strategy, k, exit) compute year-edge stats: min, mean, fraction of years > 0
    if not yedges.empty:
        agg = yedges.groupby(["strategy", "top_k", "exit"]).agg(
            n_years=("year", "nunique"),
            min_year_edge=("edge", "min"),
            mean_year_edge=("edge", "mean"),
            n_bad_years=("edge", lambda s: (s < -0.10).sum()),
            n_great_years=("edge", lambda s: (s > 0.30).sum()),
            min_year_cagr=("cagr_dca_picks", "min"),
            min_year_win=("win_rate", "min"),
            mean_year_win=("win_rate", "mean"),
        ).reset_index()
        agg.to_csv(CACHE / "year_stability_extended.csv", index=False)
        # Join with sweep_extended for combined view
        joined = agg.merge(
            big[["strategy", "top_k", "exit", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca", "win_rate", "n_picks"]],
            on=["strategy", "top_k", "exit"],
            how="inner",
        )
        joined.to_csv(CACHE / "stability_full_extended.csv", index=False)

        print("\n=== STRATEGIES WITH HIGH CAGR + FEW BAD YEARS ===")
        # Filter: edge_vs_spy_dca > 0.05, n_bad_years <= 2
        filt = joined[(joined["edge_vs_spy_dca"] > 0.05) & (joined["n_bad_years"] <= 2)]
        filt = filt.sort_values("cagr_dca_portfolio", ascending=False)
        print(filt.head(15).to_string(index=False))

        print("\n=== STRATEGIES WITH BEST WORST-YEAR EDGE ===")
        filt2 = joined[joined["edge_vs_spy_dca"] > 0.0]
        filt2 = filt2.sort_values("min_year_edge", ascending=False)
        print(filt2.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
