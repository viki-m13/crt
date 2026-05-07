"""Run all strategies and rank them. Save full summary to CSV."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.fast_engine import (
    BacktestConfig,
    backtest,
    load_panel,
)
from experiments.monthly_dca.strategies_fast import all_strategies


def main() -> None:
    panel = load_panel()
    print("Panel:", panel.shape, panel.index.min(), "->", panel.index.max())
    eval_at = panel.index.max()

    cfg = BacktestConfig(
        start="2017-12-31",
        end="2024-12-31",
        eval_at=eval_at,
        delist_iters=200,
        delist_prob_annual=0.04,
    )

    summaries = []
    for strat in all_strategies(top_k=5):
        print(f"\n=== {strat.name} ===")
        out = backtest(panel, strat, cfg)
        summ = out["summary"]
        if summ.empty:
            print("  no picks")
            continue
        summaries.append(summ)
        # Print top exits by edge_vs_spy_dca
        cols = ["exit", "n_picks", "win_rate", "win_rate_bias_corr", "beat_spy_rate",
                "median_ret", "mean_ret", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca"]
        print(summ[cols].sort_values("edge_vs_spy_dca", ascending=False).to_string(index=False))

    if summaries:
        big = pd.concat(summaries, ignore_index=True)
        out = Path(__file__).parent / "cache" / "summary_all.csv"
        big.to_csv(out, index=False)
        print(f"\nWrote {out}")
        print("\n=== TOP 15 BY DCA-PORTFOLIO CAGR ===")
        cols = ["strategy", "exit", "n_picks", "win_rate", "win_rate_bias_corr",
                "beat_spy_rate", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca"]
        print(big.sort_values("cagr_dca_portfolio", ascending=False).head(15)[cols].to_string(index=False))
        print("\n=== TOP 15 BY EDGE VS SPY DCA ===")
        print(big.sort_values("edge_vs_spy_dca", ascending=False).head(15)[cols].to_string(index=False))
        print("\n=== TOP 15 BY WIN RATE (BIAS-CORRECTED) ===")
        print(big.sort_values("win_rate_bias_corr", ascending=False).head(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
