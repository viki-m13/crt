"""Quick smoke test of the compound engine."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.compound_engine import (
    ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import quality_pullback


def main():
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")
    start, end = "2002-01-31", "2024-12-31"

    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)
    print(f"SPY DCA  CAGR XIRR={spy['cagr_xirr']:.4f}  total={spy['cagr_total']:.4f}")
    print(f"SPY DCA  final={spy['final']:.2f}  deposited={spy['deposited']:.2f}")
    print()

    cases = [
        ("strategy_rotation k=5 hold_forever", strategy_rotation, 5, ExitSpec("hold_forever")),
        ("strategy_rotation k=5 trail_25",     strategy_rotation, 5, ExitSpec("trail_25", trail=0.25)),
        ("strategy_rotation k=5 trail_35",     strategy_rotation, 5, ExitSpec("trail_35", trail=0.35)),
        ("strategy_rotation k=5 monthly_rebalance", strategy_rotation, 5, ExitSpec("monthly_rebalance", monthly_rebalance=True)),
        ("strategy_rotation k=3 trail_35",     strategy_rotation, 3, ExitSpec("trail_35", trail=0.35)),
        ("strategy_rotation k=1 trail_35",     strategy_rotation, 1, ExitSpec("trail_35", trail=0.35)),
        ("quality_pullback k=5 monthly_rebalance", quality_pullback, 5, ExitSpec("monthly_rebalance", monthly_rebalance=True)),
        ("quality_pullback k=5 trail_35",      quality_pullback, 5, ExitSpec("trail_35", trail=0.35)),
        ("quality_pullback k=3 trail_35",      quality_pullback, 3, ExitSpec("trail_35", trail=0.35)),
    ]

    for label, fn, k, exr in cases:
        t0 = time.time()
        strat = CompStrategy(label, fn, top_k=k)
        res = run_compound(panel, strat, exr, start=start, end=end,
                           eval_at=eval_at, cost_bps=5.0)
        dt = time.time() - t0
        print(f"{label}")
        print(f"  CAGR XIRR={res.cagr_money_weighted:.4f}  total={res.cagr_total_money:.4f}  "
              f"final={res.final_equity:.1f}  trades={res.n_trades}  edge_vs_spy="
              f"{res.cagr_money_weighted - spy['cagr_xirr']:+.4f}  ({dt:.1f}s)")
        print()


if __name__ == "__main__":
    main()
