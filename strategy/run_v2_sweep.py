"""V2 quick sweep across variants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.monthly_dca.compound_engine import ExitSpec, run_compound, benchmark_spy_dca
from experiments.monthly_dca.fast_engine import load_panel
from strategy.selection_v2 import make_v2_strategy
import pandas as pd

panel = load_panel()
rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)

rows = []
for variant in ["blended", "pure", "no_gate"]:
    for k in [3, 5, 7]:
        strat = make_v2_strategy(top_k=k, variant=variant)
        res = run_compound(panel, strat, rule, start="2002-01-31", end="2024-12-31", cost_bps=5.0)
        spy = benchmark_spy_dca(panel, "2002-01-31", "2024-12-31")
        edge = res.cagr_money_weighted - spy["cagr_xirr"]
        rows.append({"variant": variant, "k": k, "cagr": res.cagr_money_weighted,
                     "edge": edge, "trades": res.n_trades, "final": res.final_equity})
        print(f"  {variant} k={k}: CAGR={res.cagr_money_weighted:.2%}, edge={edge:.2%}, trades={res.n_trades}, final=${res.final_equity:.0f}")

df = pd.DataFrame(rows)
df.to_csv("backtests/v2_sweep.csv", index=False)
