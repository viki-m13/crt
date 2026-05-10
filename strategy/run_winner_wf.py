"""Walk-forward + holdout validation for the FHtzX winner: v3_topn_comp_10."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from experiments.monthly_dca.compound_engine import (
    ExitSpec, Strategy, run_compound, benchmark_spy_dca,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from strategy.selection_v3 import make_v3
from strategy.holdout import time_holdout_run, universe_holdout_run


# Walk-forward splits with 6-month embargo (TEST start 6mo after TRAIN end).
SPLITS_EMBARGO = [
    ("A1", "2011-01-31", "2018-12-31"),
    ("A2", "2015-01-31", "2021-12-31"),
    ("A3", "2018-01-31", "2024-12-31"),
    ("R1", "2008-01-31", "2010-12-31"),
    ("R2", "2011-01-31", "2013-12-31"),
    ("R3", "2014-01-31", "2016-12-31"),
    ("R4", "2017-01-31", "2019-12-31"),
    ("R5", "2020-01-31", "2022-12-31"),
    ("R6", "2023-01-31", "2024-12-31"),
    ("STRICT", "2021-01-31", "2024-12-31"),
]


def wf_one(strategy_factory, name: str, panel) -> pd.DataFrame:
    rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)
    rows = []
    for split_name, test_s, test_e in SPLITS_EMBARGO:
        strat = strategy_factory()
        try:
            res = run_compound(panel, strat, rule, start=test_s, end=test_e, cost_bps=5.0)
            spy = benchmark_spy_dca(panel, test_s, test_e)
            rows.append({
                "strategy": name,
                "split": split_name,
                "test_start": test_s, "test_end": test_e,
                "cagr": res.cagr_money_weighted,
                "spy_cagr": spy["cagr_xirr"],
                "edge": res.cagr_money_weighted - spy["cagr_xirr"],
                "n_trades": res.n_trades,
                "final_eq": res.final_equity,
                "deposited": res.total_deposited,
            })
        except Exception as e:
            rows.append({"strategy": name, "split": split_name, "error": str(e)})
    return pd.DataFrame(rows)


def main():
    panel = load_panel()

    # 1. WF on v3_topn_comp_10 (winner)
    print("=" * 72)
    print("WALK-FORWARD: v3_topn_comp_10 (FHtzX winner)")
    print("=" * 72)
    wf_winner = wf_one(lambda: make_v3("topn_comp_10", top_k=5),
                        "v3_topn_comp_10", panel)
    print(wf_winner.to_string(index=False))

    # 2. WF on baseline strategy_rotation
    print()
    print("=" * 72)
    print("WALK-FORWARD: baseline strategy_rotation k=5")
    print("=" * 72)
    baseline_factory = lambda: Strategy(
        name="baseline_strategy_rotation", score_fn=strategy_rotation, top_k=5,
        description="Existing winner",
    )
    wf_base = wf_one(baseline_factory, "baseline_strategy_rotation", panel)
    print(wf_base.to_string(index=False))

    # Also run a few diagnostic variants
    print()
    print("=" * 72)
    print("WALK-FORWARD: v3_topn_comp_15 (sensitivity)")
    print("=" * 72)
    wf_15 = wf_one(lambda: make_v3("topn_comp_15", top_k=5),
                    "v3_topn_comp_15", panel)
    print(wf_15.to_string(index=False))

    # Aggregate
    print()
    all_wf = pd.concat([wf_winner, wf_base, wf_15])
    all_wf.to_csv("backtests/wf_full.csv", index=False)

    print("AGGREGATE TEST METRICS:")
    for name in ["v3_topn_comp_10", "baseline_strategy_rotation", "v3_topn_comp_15"]:
        sub = all_wf[(all_wf.strategy == name) & all_wf.cagr.notna()]
        if sub.empty: continue
        print(f"  {name}:")
        print(f"    n={len(sub)}, mean_cagr={sub.cagr.mean():.2%}, "
               f"median={sub.cagr.median():.2%}, "
               f"min={sub.cagr.min():.2%}, max={sub.cagr.max():.2%}, "
               f"mean_edge={sub.edge.mean():.2%}, "
               f"n_pos={(sub.cagr>0).sum()}/{len(sub)}")

    # 3. Frozen holdouts on the winner only
    print()
    print("=" * 72)
    print("FROZEN HOLDOUTS — v3_topn_comp_10 (run once, never tuned on)")
    print("=" * 72)

    time_h = time_holdout_run(lambda: make_v3("topn_comp_10", top_k=5),
                               start="2024-07-31", end="2026-04-30")
    print(f"  TIME HOLDOUT 2024-07 → 2026-04: CAGR={time_h['cagr_strat']:.2%}, "
           f"SPY={time_h['cagr_spy']:.2%}, edge={time_h['edge']:.2%}, "
           f"trades={time_h['n_trades']}")

    univ_h = universe_holdout_run(lambda: make_v3("topn_comp_10", top_k=5),
                                    start="2002-01-31", end="2024-06-30")
    print(f"  UNIVERSE HOLDOUT (30% bucketed tickers): "
           f"CAGR={univ_h['cagr_strat']:.2%}, SPY={univ_h['cagr_spy']:.2%}, "
           f"edge={univ_h['edge']:.2%}, n_trades={univ_h['n_trades']}, "
           f"n_tickers={univ_h['n_tickers_holdout']}")

    # Save
    pd.DataFrame([time_h, univ_h]).to_csv("backtests/holdouts.csv", index=False)


if __name__ == "__main__":
    main()
