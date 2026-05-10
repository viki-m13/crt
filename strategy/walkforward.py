"""Walk-forward validation harness with embargo.

The existing experiments/monthly_dca/wf_*.py scripts split TRAIN/TEST
without embargo.  We add a 6-month embargo and use a purged splitter:
TRAIN ends at t1, embargo from t1 to t1+6mo, TEST starts at t1+6mo.

Each split runs the strategy on the TEST window using compound_engine
and reports CAGR XIRR + edge vs SPY DCA.  Aggregate stats reported.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import (
    BENCH_EXCLUDED, ExitSpec, REINVEST_RULES, Strategy, run_compound,
    benchmark_spy_dca,
)
from experiments.monthly_dca.fast_engine import load_panel


# 10-split walk-forward windows used by the existing harness, with
# a 6-month embargo added between TRAIN-end and TEST-start.
DEFAULT_SPLITS_EMBARGO = [
    # name, train_start, train_end, test_start, test_end
    ("A1", "2002-01-31", "2010-06-30", "2011-01-31", "2018-12-31"),
    ("A2", "2002-01-31", "2014-06-30", "2015-01-31", "2021-12-31"),
    ("A3", "2002-01-31", "2017-06-30", "2018-01-31", "2024-12-31"),
    ("R1", "2002-01-31", "2007-06-30", "2008-01-31", "2010-12-31"),
    ("R2", "2005-01-31", "2010-06-30", "2011-01-31", "2013-12-31"),
    ("R3", "2008-01-31", "2013-06-30", "2014-01-31", "2016-12-31"),
    ("R4", "2011-01-31", "2016-06-30", "2017-01-31", "2019-12-31"),
    ("R5", "2014-01-31", "2019-06-30", "2020-01-31", "2022-12-31"),
    ("R6", "2017-01-31", "2022-06-30", "2023-01-31", "2024-12-31"),
    ("STRICT", "2002-01-31", "2020-06-30", "2021-01-31", "2024-12-31"),
]


def walk_forward(strategy_factory: Callable[[], Strategy],
                  splits=DEFAULT_SPLITS_EMBARGO,
                  monthly_deposit: float = 1.0,
                  cost_bps: float = 5.0) -> pd.DataFrame:
    panel = load_panel()
    rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)
    rows = []
    for name, train_s, train_e, test_s, test_e in splits:
        strat = strategy_factory()
        try:
            res = run_compound(panel, strat, rule,
                                start=test_s, end=test_e,
                                monthly_deposit=monthly_deposit,
                                cost_bps=cost_bps)
            spy = benchmark_spy_dca(panel, test_s, test_e,
                                      monthly_deposit=monthly_deposit)
            rows.append({
                "split": name,
                "test_start": test_s, "test_end": test_e,
                "cagr_strat": res.cagr_money_weighted,
                "cagr_spy": spy["cagr_xirr"],
                "edge": res.cagr_money_weighted - spy["cagr_xirr"],
                "n_months": res.n_months,
                "n_trades": res.n_trades,
                "final_eq": res.final_equity,
                "deposited": res.total_deposited,
            })
        except Exception as e:
            rows.append({"split": name, "error": str(e)})
    return pd.DataFrame(rows)


def aggregate(wf: pd.DataFrame) -> dict:
    f = wf.dropna(subset=["cagr_strat"]) if "cagr_strat" in wf.columns else wf
    if f.empty:
        return {}
    return {
        "n_splits": len(f),
        "mean_cagr": float(f["cagr_strat"].mean()),
        "median_cagr": float(f["cagr_strat"].median()),
        "min_cagr": float(f["cagr_strat"].min()),
        "max_cagr": float(f["cagr_strat"].max()),
        "mean_edge": float(f["edge"].mean()),
        "n_positive": int((f["cagr_strat"] > 0).sum()),
        "n_beat_spy": int((f["edge"] > 0).sum()),
    }


if __name__ == "__main__":
    from strategy.selection import make_strategy, make_no_gate_strategy, make_crt_only_strategy

    print("=" * 70)
    print("WALK-FORWARD: prerunner_v1 (full strategy)")
    print("=" * 70)
    wf = walk_forward(lambda: make_strategy(top_k=5))
    print(wf.to_string(index=False))
    print()
    print("Aggregate:")
    for k, v in aggregate(wf).items():
        print(f"  {k}: {v}")
    wf.to_csv("backtests/wf_prerunner_v1.csv", index=False)

    print()
    print("=" * 70)
    print("WALK-FORWARD: crt_only (diagnostic)")
    print("=" * 70)
    wf2 = walk_forward(lambda: make_crt_only_strategy(top_k=5))
    print(wf2.to_string(index=False))
    print()
    for k, v in aggregate(wf2).items():
        print(f"  {k}: {v}")
    wf2.to_csv("backtests/wf_crt_only.csv", index=False)

    print()
    print("=" * 70)
    print("WALK-FORWARD: no_gate (diagnostic)")
    print("=" * 70)
    wf3 = walk_forward(lambda: make_no_gate_strategy(top_k=5))
    print(wf3.to_string(index=False))
    print()
    for k, v in aggregate(wf3).items():
        print(f"  {k}: {v}")
    wf3.to_csv("backtests/wf_no_gate.csv", index=False)
