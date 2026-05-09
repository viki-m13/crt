"""Comprehensive sweep of V3 strategies on the COMPOUNDING engine.

Tests every (strategy × top_k × exit_rule) combination on the full panel,
plus a recent window. Saves results to cache/sweep_v3_compound.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import (
    BENCH_EXCLUDED, REINVEST_RULES, ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_v3 import all_v3_strategies, dyn_conc_score, dyn_conc_k
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import (
    quality_pullback, explosive_winners, pullback_in_winner,
    blended_pullback_momentum,
)


CACHE = Path(__file__).resolve().parent / "cache"


def sweep(start: str, end: str, eval_at: pd.Timestamp, label: str) -> pd.DataFrame:
    panel = load_panel()
    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)
    print(f"[{label}] SPY DCA CAGR XIRR={spy['cagr_xirr']:.4f}  total={spy['cagr_total']:.4f}")

    rows = []
    # Baselines
    baselines = {
        "strategy_rotation": strategy_rotation,
        "quality_pullback": quality_pullback,
        "explosive_winners": explosive_winners,
        "pullback_in_winner": pullback_in_winner,
        "blended_pullback_momentum": blended_pullback_momentum,
    }
    v3 = all_v3_strategies()
    all_strats = {**baselines, **v3}

    # We will test each strategy with k in {1, 2, 3, 5} and a curated exit set
    exits_to_test = [
        ExitSpec("hold_forever"),
        ExitSpec("trail_25", trail=0.25),
        ExitSpec("trail_35", trail=0.35),
        ExitSpec("trail_50", trail=0.50),
        ExitSpec("monthly_rebalance", monthly_rebalance=True),
        ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3),
    ]

    for sname, sfn in all_strats.items():
        for k in [1, 2, 3, 5]:
            for exr in exits_to_test:
                if sname == "perfect_storm" and k == 5:
                    continue  # rare to have 5 perfect storms
                strat = CompStrategy(sname, sfn, top_k=k)
                try:
                    t0 = time.time()
                    res = run_compound(
                        panel, strat, exr,
                        start=start, end=end, eval_at=eval_at,
                        cost_bps=5.0,
                    )
                    dt = time.time() - t0
                    rows.append({
                        "strategy": sname,
                        "top_k": k,
                        "exit": exr.name,
                        "n_months": res.n_months,
                        "n_trades": res.n_trades,
                        "deposited": res.total_deposited,
                        "final_equity": res.final_equity,
                        "cagr_xirr": res.cagr_money_weighted,
                        "cagr_total": res.cagr_total_money,
                        "cagr_spy_dca": spy["cagr_xirr"],
                        "edge_vs_spy": res.cagr_money_weighted - spy["cagr_xirr"],
                        "window": label,
                        "elapsed_sec": dt,
                    })
                except Exception as e:
                    print(f"FAIL {sname} k={k} {exr.name}: {e}")
                    continue
        # Print progress
        df_so_far = pd.DataFrame(rows)
        if not df_so_far.empty:
            top = df_so_far[df_so_far.strategy == sname].nlargest(3, "cagr_xirr")
            print(f"  {sname}: best CAGR XIRR = {top['cagr_xirr'].max():.4f} ", flush=True)
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("V3 COMPOUND SWEEP")
    print("=" * 60)

    # Recent window — most relevant to current investing
    eval_at = pd.Timestamp("2026-05-07")
    df_recent = sweep("2018-01-31", "2024-12-31", eval_at, "recent_2018_2024")
    df_recent.to_csv(CACHE / "sweep_v3_compound_recent.csv", index=False)
    print(f"\nWrote sweep_v3_compound_recent.csv ({len(df_recent)} rows)")
    print(df_recent.nlargest(20, "cagr_xirr").to_string())

    # Full window
    df_full = sweep("2002-01-31", "2024-12-31", eval_at, "full_2002_2024")
    df_full.to_csv(CACHE / "sweep_v3_compound_full.csv", index=False)
    print(f"\nWrote sweep_v3_compound_full.csv ({len(df_full)} rows)")
    print(df_full.nlargest(20, "cagr_xirr").to_string())


if __name__ == "__main__":
    main()
