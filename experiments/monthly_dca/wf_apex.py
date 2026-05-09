"""Walk-forward validation for APEX strategies on the COMPOUNDING engine.

10 disjoint TRAIN/TEST splits. For each split:
  1. Evaluate top candidates on TRAIN window
  2. Pick top-10 candidates by TRAIN CAGR
  3. Evaluate those candidates on TEST window
  4. Aggregate stats

Saves: cache/wf_apex_train.csv, cache/wf_apex_test.csv, cache/wf_apex_aggregate.csv
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

EXITS = [
    ExitSpec("hold_forever"),
    ExitSpec("trail_25", trail=0.25),
    ExitSpec("trail_35", trail=0.35),
    ExitSpec("monthly_rebalance", monthly_rebalance=True),
    ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3),
]
KS = [3, 5]


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


def run_split(panel, split):
    label, train_start, train_end, test_start, test_end = split
    print(f"\n[{label}] TRAIN={train_start}..{train_end} TEST={test_start}..{test_end}", flush=True)
    train_rows = []
    for sname, sfn in CANDIDATES.items():
        for k in KS:
            for exr in EXITS:
                if "perfect_storm" in sname and k == 5:
                    continue
                try:
                    r = evaluate(panel, sname, sfn, k, exr,
                                 train_start, train_end, train_end)
                    r["split"] = label
                    r["phase"] = "TRAIN"
                    train_rows.append(r)
                except Exception as e:
                    pass
    train_df = pd.DataFrame(train_rows)
    if train_df.empty:
        return [], []
    # Top-10 by train CAGR
    top10 = train_df.nlargest(10, "cagr_xirr")
    print(f"  Top-3 TRAIN: {top10.head(3)[['strategy','k','exit','cagr_xirr']].to_dict('records')}", flush=True)
    test_rows = []
    for _, row in top10.iterrows():
        sname, k, exit_name = row["strategy"], int(row["k"]), row["exit"]
        sfn = CANDIDATES[sname]
        exr = next((e for e in EXITS if e.name == exit_name), None)
        if exr is None:
            continue
        try:
            r = evaluate(panel, sname, sfn, k, exr,
                         test_start, test_end, test_end)
            r["split"] = label
            r["phase"] = "TEST"
            r["train_cagr"] = row["cagr_xirr"]
            r["train_edge"] = row["edge"]
            test_rows.append(r)
        except Exception:
            pass
    if test_rows:
        df = pd.DataFrame(test_rows)
        print(f"  TOP TEST: {df.nlargest(3, 'cagr_xirr')[['strategy','k','exit','cagr_xirr','edge']].to_dict('records')}", flush=True)
    return train_rows, test_rows


def main():
    panel = load_panel()
    all_train, all_test = [], []
    for split in SPLITS:
        train, test = run_split(panel, split)
        all_train.extend(train)
        all_test.extend(test)

    train_df = pd.DataFrame(all_train)
    test_df = pd.DataFrame(all_test)
    train_df.to_csv(CACHE / "wf_apex_train.csv", index=False)
    test_df.to_csv(CACHE / "wf_apex_test.csv", index=False)

    # For each (strategy, k, exit), aggregate across all splits where it
    # appeared in train top-10
    if not test_df.empty:
        agg = test_df.groupby(["strategy", "k", "exit"]).agg(
            n_splits=("cagr_xirr", "count"),
            mean_test_cagr=("cagr_xirr", "mean"),
            median_test_cagr=("cagr_xirr", "median"),
            min_test_cagr=("cagr_xirr", "min"),
            max_test_cagr=("cagr_xirr", "max"),
            mean_test_edge=("edge", "mean"),
            min_test_edge=("edge", "min"),
            mean_train_cagr=("train_cagr", "mean"),
        ).sort_values("mean_test_cagr", ascending=False).reset_index()
        agg.to_csv(CACHE / "wf_apex_aggregate.csv", index=False)
        print("\n=== TOP-20 by mean TEST CAGR ===")
        print(agg.head(20).to_string(), flush=True)


if __name__ == "__main__":
    main()
