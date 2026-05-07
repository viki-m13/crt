"""Step 1: Print oracle ceilings.
Step 2: Sweep all strategies x top_k x rules.
Step 3: Print top-N tables for CAGR / win-rate / edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.fast_score import (
    evaluate_strategy,
    load_panel,
    oracle_bound,
)
from experiments.monthly_dca.strategies_fast import all_strategies


def main() -> None:
    panel = load_panel()
    print("Panel:", panel.shape, "eval_at:", panel.index.max())

    # ---- ORACLE ----
    print("\n=== ORACLE (theoretical ceiling, perfect foresight) ===")
    rules_to_check = ["hold_forever", "fixed_3y", "fixed_5y", "fixed_1y", "tp200"]
    rows = []
    for k in (1, 2, 3, 5, 10):
        for r in rules_to_check:
            res = oracle_bound(top_k=k, rule_name=r)
            rows.append(res)
            print(f"  top_k={k:2d}  rule={r:14s}  n={res['n_picks']:5d}  "
                  f"median={res['median_pick_ret']:+.3f}  win={res['win_rate']:.3f}  "
                  f"CAGR={res['cagr_dca']:.3f}  SPY={res['cagr_spy_dca']:.3f}")
    pd.DataFrame(rows).to_csv("experiments/monthly_dca/cache/oracle.csv", index=False)

    # ---- STRATEGY SWEEP ----
    print("\n=== STRATEGY SWEEP (15 strategies x 5 top_k x 13 rules) ===")
    summaries: list[pd.DataFrame] = []
    for top_k in (1, 3, 5, 10, 20):
        for strat in all_strategies(top_k=top_k):
            er = evaluate_strategy(strat.score_fn, top_k=top_k, name=strat.name,
                                   start="2017-12-31", end="2024-12-31",
                                   panel=panel, delist_iters=100)
            if er.summary.empty:
                continue
            summ = er.summary.copy()
            summ["top_k"] = top_k
            summaries.append(summ)
        print(f"  done top_k={top_k}")
    big = pd.concat(summaries, ignore_index=True)
    big.to_csv("experiments/monthly_dca/cache/sweep_v1.csv", index=False)

    cols = ["strategy", "top_k", "exit", "n_picks", "win_rate", "win_rate_bias_corr",
            "beat_spy_rate", "median_ret", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca"]

    print("\n=== TOP 25 BY DCA-PORTFOLIO CAGR ===")
    print(big.sort_values("cagr_dca_portfolio", ascending=False).head(25)[cols].to_string(index=False))

    print("\n=== TOP 25 BY EDGE VS SPY DCA ===")
    print(big.sort_values("edge_vs_spy_dca", ascending=False).head(25)[cols].to_string(index=False))

    print("\n=== TOP 25 BY WIN RATE (BIAS-CORRECTED), with positive edge ===")
    pos_edge = big[big["edge_vs_spy_dca"] > 0]
    print(pos_edge.sort_values("win_rate_bias_corr", ascending=False).head(25)[cols].to_string(index=False))

    print("\n=== HIGHEST CAGR among bias-corrected win_rate >= 60% ===")
    hi_win = big[big["win_rate_bias_corr"] >= 0.60]
    print(hi_win.sort_values("cagr_dca_portfolio", ascending=False).head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
