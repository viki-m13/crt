"""Quick feasibility test: run prerunner_v1 + diagnostics on the FULL window
2002-2024 with monthly_rebalance compounding, vs the existing baseline.

Output:  CAGR XIRR, edge vs SPY DCA, n_trades.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from experiments.monthly_dca.compound_engine import (
    ExitSpec, run_compound, benchmark_spy_dca,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.compound_engine import Strategy as CompoundStrategy

from strategy.selection import (
    make_strategy, make_no_gate_strategy, make_crt_only_strategy,
)


def run_one(panel, strat, start: str, end: str, cost_bps: float = 5.0) -> dict:
    rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)
    res = run_compound(panel, strat, rule, start=start, end=end, cost_bps=cost_bps)
    spy = benchmark_spy_dca(panel, start, end)
    return {
        "strategy": strat.name,
        "cagr_xirr": res.cagr_money_weighted,
        "cagr_total": res.cagr_total_money,
        "cagr_spy": spy["cagr_xirr"],
        "edge": res.cagr_money_weighted - spy["cagr_xirr"],
        "n_trades": res.n_trades,
        "n_months": res.n_months,
        "deposited": res.total_deposited,
        "final_eq": res.final_equity,
    }


def main():
    panel = load_panel()
    rows = []

    print("=" * 70)
    print("Full-window feasibility (2002-01 → 2024-12)")
    print("=" * 70)

    # 1. Existing baseline (recompute as a sanity check)
    baseline = CompoundStrategy(
        name="baseline_strategy_rotation",
        score_fn=strategy_rotation,
        top_k=5,
        description="Existing winner",
    )
    rows.append(run_one(panel, baseline, "2002-01-31", "2024-12-31"))
    print(f"  baseline_strategy_rotation: CAGR={rows[-1]['cagr_xirr']:.2%}, "
           f"edge={rows[-1]['edge']:.2%}, trades={rows[-1]['n_trades']}")

    # 2. Pre-runner v1 — full strategy
    rows.append(run_one(panel, make_strategy(top_k=5), "2002-01-31", "2024-12-31"))
    print(f"  prerunner_v1: CAGR={rows[-1]['cagr_xirr']:.2%}, "
           f"edge={rows[-1]['edge']:.2%}, trades={rows[-1]['n_trades']}")

    # 3. CRT only (diagnostic)
    rows.append(run_one(panel, make_crt_only_strategy(top_k=5),
                         "2002-01-31", "2024-12-31"))
    print(f"  crt_only: CAGR={rows[-1]['cagr_xirr']:.2%}, "
           f"edge={rows[-1]['edge']:.2%}, trades={rows[-1]['n_trades']}")

    # 4. No-gate composite (diagnostic)
    rows.append(run_one(panel, make_no_gate_strategy(top_k=5),
                         "2002-01-31", "2024-12-31"))
    print(f"  no_gate: CAGR={rows[-1]['cagr_xirr']:.2%}, "
           f"edge={rows[-1]['edge']:.2%}, trades={rows[-1]['n_trades']}")

    # 5. Pre-runner v1, k=3 and k=7 sensitivity
    for k in (3, 7, 10):
        rows.append(run_one(panel, make_strategy(top_k=k),
                             "2002-01-31", "2024-12-31"))
        print(f"  prerunner_v1 k={k}: CAGR={rows[-1]['cagr_xirr']:.2%}, "
               f"edge={rows[-1]['edge']:.2%}, trades={rows[-1]['n_trades']}")

    df = pd.DataFrame(rows)
    df.to_csv("backtests/feasibility_full_window.csv", index=False)
    print()
    print(df[["strategy", "cagr_xirr", "edge", "n_trades", "final_eq"]]
            .to_string(index=False))


if __name__ == "__main__":
    main()
