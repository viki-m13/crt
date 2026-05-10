"""Robustness sweep: vary top_n, top_k, cost_bps for v3_topn_comp.

Checks that the headline result isn't a knife-edge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from experiments.monthly_dca.compound_engine import ExitSpec, run_compound, benchmark_spy_dca, Strategy
from experiments.monthly_dca.fast_engine import load_panel
from strategy.selection_v3 import v3_topn_composite

panel = load_panel()
rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)

rows = []

print("=" * 60)
print("Robustness: top_n, top_k, cost_bps sweep on v3_topn_comp")
print("=" * 60)

for top_n in [8, 10, 12, 15, 20, 25]:
    for top_k in [3, 5, 7]:
        for cost_bps in [5.0]:
            def make_score(N=top_n):
                return lambda df: v3_topn_composite(df, top_n=N)
            strat = Strategy(name=f"v3_n{top_n}_k{top_k}",
                              score_fn=make_score(top_n), top_k=top_k)
            try:
                res = run_compound(panel, strat, rule, start="2002-01-31",
                                     end="2024-12-31", cost_bps=cost_bps)
                spy = benchmark_spy_dca(panel, "2002-01-31", "2024-12-31")
                edge = res.cagr_money_weighted - spy["cagr_xirr"]
                rows.append({"top_n": top_n, "top_k": top_k, "cost_bps": cost_bps,
                             "cagr": res.cagr_money_weighted, "edge": edge,
                             "trades": res.n_trades, "final": res.final_equity})
                print(f"  n={top_n}, k={top_k}, cost={cost_bps}bp: "
                       f"CAGR={res.cagr_money_weighted:.2%}, edge={edge:.2%}")
            except Exception as e:
                print(f"  n={top_n}, k={top_k}: ERR {e}")

print()
print("=" * 60)
print("Cost sensitivity at n=10, k=5")
print("=" * 60)
for cost_bps in [2.5, 5.0, 10.0, 25.0, 50.0]:
    strat = Strategy(name=f"v3_n10_k5_c{cost_bps}",
                      score_fn=lambda df: v3_topn_composite(df, top_n=10),
                      top_k=5)
    res = run_compound(panel, strat, rule, start="2002-01-31",
                        end="2024-12-31", cost_bps=cost_bps)
    spy = benchmark_spy_dca(panel, "2002-01-31", "2024-12-31")
    edge = res.cagr_money_weighted - spy["cagr_xirr"]
    rows.append({"top_n": 10, "top_k": 5, "cost_bps": cost_bps,
                 "cagr": res.cagr_money_weighted, "edge": edge,
                 "trades": res.n_trades, "final": res.final_equity})
    print(f"  cost={cost_bps}bp/side: CAGR={res.cagr_money_weighted:.2%}, "
           f"edge={edge:.2%}")

pd.DataFrame(rows).to_csv("backtests/robustness.csv", index=False)
print()
print("Saved to backtests/robustness.csv")
