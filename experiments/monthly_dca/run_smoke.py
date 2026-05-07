"""Smoke test: run one strategy over a short window to validate engine."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.backtester import (
    BacktestConfig,
    DEFAULT_EXITS,
    load_panel,
    run_strategy,
    summarize_result,
)
from experiments.monthly_dca.strategies import all_strategies


def main() -> None:
    panel = load_panel()
    print("Panel:", panel.shape, panel.index.min(), "->", panel.index.max())
    eval_at = panel.index.max()
    cfg = BacktestConfig(
        start="2018-01-01",
        end="2019-12-31",
        eval_at=eval_at,
    )
    strats = all_strategies(top_k=5)
    rows = []
    for strat in strats[:3]:
        res = run_strategy(panel, strat, cfg)
        if res.fwd_returns.empty:
            print(f"  no picks for {strat.name}")
            continue
        summ = summarize_result(res, eval_at=eval_at, n_iter=100)
        rows.append(summ)
        print(f"\n=== {strat.name} ===")
        print(summ.to_string())

    if rows:
        big = pd.concat(rows, ignore_index=True)
        out = Path(__file__).parent / "cache" / "smoke_summary.csv"
        big.to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
