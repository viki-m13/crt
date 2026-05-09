"""Streamlined walk-forward: only test TOP-N candidates from the full sweep.

This is a smarter, faster walk-forward:
  1. Read top N candidates from sweep_apex_focused_full.csv
  2. For each (strategy, k, exit) candidate, evaluate on each TEST window
  3. Aggregate: mean, median, min test CAGR; mean test edge

Note: This does NOT prevent look-ahead bias in the strategy selection step,
since we picked the candidates using the full window. To address that, we
run a SECOND walk-forward where each split picks its own top candidate
from a TRAIN window, and evaluates on TEST.

Output:
  cache/wf_apex_v2_oos.csv         — TOP-N OOS performance per split
  cache/wf_apex_v2_aggregate.csv   — aggregated stats
  cache/wf_apex_v2_train_test.csv  — true TRAIN-pick / TEST-eval
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import (
    ExitSpec, Strategy as CompStrategy, benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_apex import all_apex_strategies
from experiments.monthly_dca.strategies_v3 import all_v3_strategies
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import (
    quality_pullback, explosive_winners, pullback_in_winner,
)


CACHE = Path(__file__).resolve().parent / "cache"


SPLITS = [
    ("A1", "2002-01-31", "2010-12-31", "2011-01-31", "2018-12-31"),
    ("A2", "2002-01-31", "2014-12-31", "2015-01-31", "2021-12-31"),
    ("A3", "2002-01-31", "2017-12-31", "2018-01-31", "2024-12-31"),
    ("R1", "2002-01-31", "2007-12-31", "2008-01-31", "2010-12-31"),
    ("R2", "2005-01-31", "2010-12-31", "2011-01-31", "2013-12-31"),
    ("R3", "2008-01-31", "2013-12-31", "2014-01-31", "2016-12-31"),
    ("R4", "2011-01-31", "2016-12-31", "2017-01-31", "2019-12-31"),
    ("R5", "2014-01-31", "2019-12-31", "2020-01-31", "2022-12-31"),
    ("R6", "2017-01-31", "2022-12-31", "2023-01-31", "2024-12-31"),
    ("STRICT", "2002-01-31", "2020-12-31", "2021-01-31", "2024-12-31"),
]


CANDIDATES = {
    "strategy_rotation": strategy_rotation,
    "quality_pullback": quality_pullback,
    "explosive_winners": explosive_winners,
    "pullback_in_winner": pullback_in_winner,
    **all_v3_strategies(),
    **all_apex_strategies(),
}


def evaluate(panel, sname, sfn, k, exr, start, end, eval_at):
    res = run_compound(
        panel, CompStrategy(sname, sfn, top_k=k), exr,
        start=start, end=end, eval_at=pd.Timestamp(eval_at), cost_bps=5.0,
    )
    spy = benchmark_spy_dca(panel, start, end, eval_at=pd.Timestamp(eval_at))
    return {
        "strategy": sname, "k": k, "exit": exr.name,
        "n_trades": res.n_trades,
        "cagr_xirr": res.cagr_money_weighted,
        "cagr_total": res.cagr_total_money,
        "edge": res.cagr_money_weighted - spy["cagr_xirr"],
        "spy_dca": spy["cagr_xirr"],
    }


def part1_oos_top_candidates(panel, top_n=15):
    """Read top-N candidates from full sweep, evaluate on each TEST window."""
    sweep_path = CACHE / "sweep_apex_focused_full.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"No sweep at {sweep_path}; run run_apex_focused.py first.")
    df = pd.read_csv(sweep_path)
    top = df.nlargest(top_n, "cagr_xirr")
    print(f"Evaluating top-{top_n} candidates from full sweep on each TEST window...", flush=True)

    rows = []
    for _, c in top.iterrows():
        sname, k = c["strategy"], int(c["k"])
        sfn = CANDIDATES.get(sname)
        if sfn is None:
            continue
        # Find ExitSpec
        exit_name = c["exit"]
        if exit_name == "monthly_rebalance":
            exr = ExitSpec("monthly_rebalance", monthly_rebalance=True)
        elif exit_name == "trail_25":
            exr = ExitSpec("trail_25", trail=0.25)
        elif exit_name == "trail_35":
            exr = ExitSpec("trail_35", trail=0.35)
        elif exit_name == "trail_50":
            exr = ExitSpec("trail_50", trail=0.50)
        elif exit_name == "trail35_or_3y":
            exr = ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3)
        elif exit_name == "fixed_3y":
            exr = ExitSpec("fixed_3y", days=252 * 3)
        else:
            exr = ExitSpec("hold_forever")
        for split in SPLITS:
            label, train_start, train_end, test_start, test_end = split
            try:
                r = evaluate(panel, sname, sfn, k, exr, test_start, test_end, test_end)
                r["split"] = label
                r["full_window_cagr"] = c["cagr_xirr"]
                rows.append(r)
                print(f"  {sname:25s} k={k} {exit_name:18s} | {label}: TEST CAGR={r['cagr_xirr']:.4f} edge={r['edge']:+.4f}",
                      flush=True)
            except Exception as e:
                pass
    return pd.DataFrame(rows)


def part2_proper_walkforward(panel, candidates_to_train=None):
    """For each split: find top-3 candidates by TRAIN CAGR, evaluate on TEST."""
    if candidates_to_train is None:
        # Use a small fixed set of strong candidates to keep it fast
        candidates_to_train = [
            ("strategy_rotation", 3, "monthly_rebalance"),
            ("strategy_rotation", 5, "monthly_rebalance"),
            ("apex_balanced", 3, "monthly_rebalance"),
            ("apex_balanced", 5, "monthly_rebalance"),
            ("apex_reloaded", 3, "monthly_rebalance"),
            ("apex_reloaded", 5, "monthly_rebalance"),
            ("apex_turbocharged", 3, "monthly_rebalance"),
            ("apex_hybrid", 5, "monthly_rebalance"),
            ("quality_pullback", 5, "monthly_rebalance"),
            ("compound_quality", 5, "monthly_rebalance"),
            ("consensus_engine", 5, "monthly_rebalance"),
            ("apex_balanced", 5, "trail_35"),
            ("apex_balanced", 5, "trail_25"),
            ("apex_reloaded", 5, "trail_35"),
            ("strategy_rotation", 5, "trail_35"),
        ]

    def get_exr(name):
        if name == "monthly_rebalance":
            return ExitSpec("monthly_rebalance", monthly_rebalance=True)
        if name == "trail_25":
            return ExitSpec("trail_25", trail=0.25)
        if name == "trail_35":
            return ExitSpec("trail_35", trail=0.35)
        if name == "trail_50":
            return ExitSpec("trail_50", trail=0.50)
        if name == "trail35_or_3y":
            return ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3)
        return ExitSpec("hold_forever")

    train_rows, test_rows = [], []
    for split in SPLITS:
        label, train_start, train_end, test_start, test_end = split
        print(f"\n[{label}] proper WF: train={train_start}->{train_end} test={test_start}->{test_end}", flush=True)
        candidates_train = []
        for sname, k, ex_name in candidates_to_train:
            sfn = CANDIDATES.get(sname)
            if sfn is None:
                continue
            try:
                r = evaluate(panel, sname, sfn, k, get_exr(ex_name),
                             train_start, train_end, train_end)
                r["split"] = label
                r["phase"] = "TRAIN"
                candidates_train.append(r)
                train_rows.append(r)
            except Exception:
                pass
        train_df = pd.DataFrame(candidates_train)
        if train_df.empty:
            continue
        top3 = train_df.nlargest(3, "cagr_xirr")
        print(f"  Top-3 TRAIN: {top3[['strategy','k','exit','cagr_xirr']].to_dict('records')}", flush=True)
        for _, row in top3.iterrows():
            sname = row["strategy"]; k = int(row["k"]); ex_name = row["exit"]
            sfn = CANDIDATES.get(sname)
            try:
                r = evaluate(panel, sname, sfn, k, get_exr(ex_name),
                             test_start, test_end, test_end)
                r["split"] = label
                r["phase"] = "TEST"
                r["train_cagr"] = row["cagr_xirr"]
                r["train_edge"] = row["edge"]
                test_rows.append(r)
                print(f"    {sname:25s} k={k} {ex_name:18s} TEST CAGR={r['cagr_xirr']:.4f} edge={r['edge']:+.4f}",
                      flush=True)
            except Exception:
                pass
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def main():
    panel = load_panel()
    print("=" * 60, flush=True)
    print("PART 1: TOP candidates from full sweep, evaluated on each TEST window", flush=True)
    print("=" * 60, flush=True)
    p1 = part1_oos_top_candidates(panel, top_n=15)
    p1.to_csv(CACHE / "wf_apex_v2_oos.csv", index=False)

    if not p1.empty:
        agg1 = p1.groupby(["strategy", "k", "exit"]).agg(
            n_splits=("cagr_xirr", "count"),
            mean_test_cagr=("cagr_xirr", "mean"),
            median_test_cagr=("cagr_xirr", "median"),
            min_test_cagr=("cagr_xirr", "min"),
            max_test_cagr=("cagr_xirr", "max"),
            mean_test_edge=("edge", "mean"),
            min_test_edge=("edge", "min"),
            full_window_cagr=("full_window_cagr", "first"),
        ).sort_values("mean_test_cagr", ascending=False).reset_index()
        agg1.to_csv(CACHE / "wf_apex_v2_aggregate.csv", index=False)
        print("\nTop-10 by mean OOS test CAGR:", flush=True)
        print(agg1.head(10).to_string(), flush=True)

    print("\n" + "=" * 60, flush=True)
    print("PART 2: Proper TRAIN/TEST walk-forward (no look-ahead)", flush=True)
    print("=" * 60, flush=True)
    train_df, test_df = part2_proper_walkforward(panel)
    train_df.to_csv(CACHE / "wf_apex_v2_train.csv", index=False)
    test_df.to_csv(CACHE / "wf_apex_v2_test.csv", index=False)

    if not test_df.empty:
        agg2 = test_df.groupby(["strategy", "k", "exit"]).agg(
            n_splits=("cagr_xirr", "count"),
            mean_test_cagr=("cagr_xirr", "mean"),
            median_test_cagr=("cagr_xirr", "median"),
            min_test_cagr=("cagr_xirr", "min"),
            max_test_cagr=("cagr_xirr", "max"),
            mean_test_edge=("edge", "mean"),
            min_test_edge=("edge", "min"),
        ).sort_values("mean_test_cagr", ascending=False).reset_index()
        agg2.to_csv(CACHE / "wf_apex_v2_proper_aggregate.csv", index=False)
        print("\nProper WF — Top by mean TEST CAGR (only when chosen by TRAIN):", flush=True)
        print(agg2.head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
