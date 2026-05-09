"""Walk-forward validate the winner: strategy_rotation k=5 monthly_rebalance.

Tests across 10 distinct TRAIN/TEST splits.

Approach:
  1. For each split, evaluate top candidates on TRAIN and TEST windows.
  2. Top candidates: strategy_rotation, quality_pullback, explosive_winners,
     consensus_engine, rotation_apex (top from sweep).
  3. Report mean OOS test CAGR + edge for the winner.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import benchmark_spy_dca
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.fast_monthly_rebalance import run_monthly_rebalance
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import quality_pullback, explosive_winners, pullback_in_winner
from experiments.monthly_dca.strategies_v3 import consensus_engine
from experiments.monthly_dca.strategies_rotation_plus import rotation_apex


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
    "consensus_engine": consensus_engine,
    "rotation_apex": rotation_apex,
}
KS = [3, 5, 7]


def evaluate(panel, sname, sfn, k, start, end, eval_at):
    res = run_monthly_rebalance(panel, sfn, top_k=k,
                                 start=start, end=end,
                                 eval_at=pd.Timestamp(eval_at), cost_bps=5.0)
    spy = benchmark_spy_dca(panel, start, end, eval_at=pd.Timestamp(eval_at))
    return {
        "strategy": sname, "k": k,
        "n_trades": res["n_trades"],
        "cagr_xirr": res["cagr_xirr"],
        "cagr_total": res["cagr_total"],
        "edge": res["cagr_xirr"] - spy["cagr_xirr"],
        "spy_dca": spy["cagr_xirr"],
        "final_equity": res["final_equity"],
        "deposited": res["deposited"],
    }


def main():
    panel = load_panel()
    train_rows, test_rows = [], []

    for label, ts, te, vs, ve in SPLITS:
        print(f"\n[{label}] TRAIN={ts}..{te} TEST={vs}..{ve}", flush=True)
        for sname, sfn in CANDIDATES.items():
            for k in KS:
                try:
                    rt = evaluate(panel, sname, sfn, k, ts, te, te)
                    rt["split"] = label; rt["phase"] = "TRAIN"
                    train_rows.append(rt)
                except Exception as e:
                    pass
        td = pd.DataFrame([r for r in train_rows if r.get("split") == label])
        if td.empty:
            continue
        top3 = td.nlargest(3, "cagr_xirr")
        print(f"  Top-3 TRAIN: {top3[['strategy','k','cagr_xirr']].to_dict('records')}", flush=True)
        for _, row in top3.iterrows():
            sname, k = row["strategy"], int(row["k"])
            try:
                rv = evaluate(panel, sname, CANDIDATES[sname], k, vs, ve, ve)
                rv["split"] = label; rv["phase"] = "TEST"
                rv["train_cagr"] = row["cagr_xirr"]
                rv["train_edge"] = row["edge"]
                test_rows.append(rv)
            except Exception as e:
                pass
        if test_rows:
            t_df = pd.DataFrame([r for r in test_rows if r["split"] == label])
            print(f"    TOP TEST: {t_df.nlargest(3, 'cagr_xirr')[['strategy','k','cagr_xirr','edge']].to_dict('records')}", flush=True)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    train_df.to_csv(CACHE / "wf_winner_train.csv", index=False)
    test_df.to_csv(CACHE / "wf_winner_test.csv", index=False)

    if not test_df.empty:
        agg = test_df.groupby(["strategy", "k"]).agg(
            n_splits=("cagr_xirr", "count"),
            mean_test_cagr=("cagr_xirr", "mean"),
            median_test_cagr=("cagr_xirr", "median"),
            min_test_cagr=("cagr_xirr", "min"),
            max_test_cagr=("cagr_xirr", "max"),
            mean_edge=("edge", "mean"),
            min_edge=("edge", "min"),
        ).sort_values("mean_test_cagr", ascending=False).reset_index()
        agg.to_csv(CACHE / "wf_winner_aggregate.csv", index=False)
        print("\n=== Walk-forward AGGREGATE ===", flush=True)
        print(agg.to_string(), flush=True)

    # Also: full-window stats per (strategy, k)
    if not train_df.empty:
        per_combo = train_df.groupby(["strategy", "k"]).size().rename("n").reset_index()
        all_train_top10 = train_df.groupby(["strategy", "k"]).apply(
            lambda d: d.nlargest(0, "cagr_xirr")  # placeholder
        )


if __name__ == "__main__":
    main()
