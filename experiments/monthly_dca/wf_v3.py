"""Walk-forward validation for V3 strategies on the COMPOUNDING engine.

10 disjoint TRAIN/TEST splits, picks the best (strategy, k, exit) on each
TRAIN window, evaluates on TEST. Reports aggregate stats.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import (
    BENCH_EXCLUDED, ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_v3 import all_v3_strategies
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import (
    quality_pullback, explosive_winners, pullback_in_winner,
    blended_pullback_momentum,
)


CACHE = Path(__file__).resolve().parent / "cache"


# Walk-forward splits (TRAIN_START, TRAIN_END, TEST_START, TEST_END, eval_at)
SPLITS = [
    ("A1", "2002-01-31", "2010-12-31", "2011-01-31", "2018-12-31", "2018-12-31"),
    ("A2", "2002-01-31", "2014-12-31", "2015-01-31", "2021-12-31", "2021-12-31"),
    ("A3", "2002-01-31", "2017-12-31", "2018-01-31", "2024-12-31", "2024-12-31"),
    ("R1", "2002-01-31", "2007-12-31", "2008-01-31", "2010-12-31", "2010-12-31"),
    ("R2", "2005-01-31", "2010-12-31", "2011-01-31", "2013-12-31", "2013-12-31"),
    ("R3", "2008-01-31", "2013-12-31", "2014-01-31", "2016-12-31", "2016-12-31"),
    ("R4", "2011-01-31", "2016-12-31", "2017-01-31", "2019-12-31", "2019-12-31"),
    ("R5", "2014-01-31", "2019-12-31", "2020-01-31", "2022-12-31", "2022-12-31"),
    ("R6", "2017-01-31", "2022-12-31", "2023-01-31", "2024-12-31", "2024-12-31"),
    ("STRICT", "2002-01-31", "2020-12-31", "2021-01-31", "2024-12-31", "2024-12-31"),
]


CANDIDATES = {
    **{
        "strategy_rotation": strategy_rotation,
        "quality_pullback": quality_pullback,
        "explosive_winners": explosive_winners,
        "pullback_in_winner": pullback_in_winner,
        "blended_pullback_momentum": blended_pullback_momentum,
    },
    **all_v3_strategies(),
}

EXITS = [
    ExitSpec("hold_forever"),
    ExitSpec("trail_25", trail=0.25),
    ExitSpec("trail_35", trail=0.35),
    ExitSpec("trail_50", trail=0.50),
    ExitSpec("monthly_rebalance", monthly_rebalance=True),
    ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3),
]
KS = [1, 2, 3, 5]


def evaluate_combo(panel, strat_name, sfn, k, exit_spec, start, end, eval_at) -> dict:
    cstrat = CompStrategy(strat_name, sfn, top_k=k)
    res = run_compound(panel, cstrat, exit_spec, start=start, end=end,
                       eval_at=pd.Timestamp(eval_at), cost_bps=5.0)
    spy = benchmark_spy_dca(panel, start, end, eval_at=pd.Timestamp(eval_at))
    return {
        "strategy": strat_name, "k": k, "exit": exit_spec.name,
        "n_trades": res.n_trades, "cagr_xirr": res.cagr_money_weighted,
        "cagr_total": res.cagr_total_money,
        "edge": res.cagr_money_weighted - spy["cagr_xirr"],
        "spy_dca": spy["cagr_xirr"],
        "final_eq": res.final_equity, "deposited": res.total_deposited,
    }


def run_one_split(panel, split):
    label, train_start, train_end, test_start, test_end, eval_at = split
    print(f"\n[{label}] TRAIN={train_start}->{train_end}  TEST={test_start}->{test_end}")
    train_rows = []
    for sname, sfn in CANDIDATES.items():
        for k in KS:
            for exr in EXITS:
                if sname == "perfect_storm" and k == 5:
                    continue
                try:
                    train_rows.append({
                        **evaluate_combo(panel, sname, sfn, k, exr,
                                         train_start, train_end, train_end),
                        "phase": "TRAIN",
                    })
                except Exception as e:
                    pass
    train_df = pd.DataFrame(train_rows)
    if train_df.empty:
        return [], []
    # Pick top-10 train combos by cagr_xirr
    train_top = train_df.nlargest(10, "cagr_xirr")
    test_rows = []
    for _, row in train_top.iterrows():
        sname, k, exit_name = row["strategy"], int(row["k"]), row["exit"]
        sfn = CANDIDATES[sname]
        # Find matching ExitSpec
        exr = next((e for e in EXITS if e.name == exit_name), None)
        if exr is None:
            continue
        try:
            test_rows.append({
                **evaluate_combo(panel, sname, sfn, k, exr,
                                 test_start, test_end, eval_at),
                "phase": "TEST",
                "split": label,
                "train_cagr": row["cagr_xirr"],
                "train_edge": row["edge"],
            })
        except Exception:
            pass
    return train_rows, test_rows


def main():
    panel = load_panel()
    all_train, all_test = [], []
    for split in SPLITS:
        train, test = run_one_split(panel, split)
        for t in train:
            t["split"] = split[0]
        all_train.extend(train)
        all_test.extend(test)
        # Show top 5 for this split
        if test:
            t_df = pd.DataFrame(test)
            print(t_df.head(5).to_string())

    train_df = pd.DataFrame(all_train)
    test_df = pd.DataFrame(all_test)
    train_df.to_csv(CACHE / "wf_v3_train.csv", index=False)
    test_df.to_csv(CACHE / "wf_v3_test.csv", index=False)

    # Aggregate: for each (strategy, k, exit) combo, count splits where it was
    # in train top-10 and average TEST cagr.
    if not test_df.empty:
        agg = test_df.groupby(["strategy", "k", "exit"]).agg(
            n_splits=("cagr_xirr", "count"),
            mean_test_cagr=("cagr_xirr", "mean"),
            median_test_cagr=("cagr_xirr", "median"),
            min_test_cagr=("cagr_xirr", "min"),
            max_test_cagr=("cagr_xirr", "max"),
            mean_test_edge=("edge", "mean"),
            min_test_edge=("edge", "min"),
        ).sort_values("mean_test_cagr", ascending=False).reset_index()
        agg.to_csv(CACHE / "wf_v3_aggregate.csv", index=False)
        print("\nTop-20 by mean TEST CAGR:")
        print(agg.head(20).to_string())


if __name__ == "__main__":
    main()
