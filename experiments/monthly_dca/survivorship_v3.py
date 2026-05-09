"""Survivorship overlay for the V3 (compounding) winning strategy.

Runs the winning strategy with synthetic delisting injection at α ∈ {0%, 4%,
8%, 12%, 16%, 20%}/yr, with 50 Monte-Carlo iterations each. Reports median +
p10/p90 CAGR.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import (
    ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_apex import (
    apex_reloaded, apex_turbocharged, apex_balanced, apex_hybrid,
)
from experiments.monthly_dca.strategies_ensemble import strategy_rotation


CACHE = Path(__file__).resolve().parent / "cache"


def run_overlay(panel, sname, sfn, k, exr, start, end, eval_at,
                alphas=(0.0, 0.04, 0.08, 0.12, 0.16, 0.20),
                iters=30):
    spy = benchmark_spy_dca(panel, start, end, eval_at=pd.Timestamp(eval_at))
    rows = []
    for a in alphas:
        cagrs, edges, finals, win_rates = [], [], [], []
        for seed in range(iters):
            r = run_compound(panel, CompStrategy(sname, sfn, top_k=k), exr,
                             start=start, end=end, eval_at=pd.Timestamp(eval_at),
                             delist_alpha=a, delist_seed=seed, cost_bps=5.0)
            cagrs.append(r.cagr_money_weighted)
            edges.append(r.cagr_money_weighted - spy["cagr_xirr"])
            finals.append(r.final_equity)
            n_wins = 0
            n_total = 0
            if not r.trades.empty:
                rets = r.trades["ret"].dropna()
                n_wins = (rets > 0).sum()
                n_total = len(rets)
            win_rates.append(n_wins / n_total if n_total else float("nan"))
        rows.append({
            "alpha": a,
            "cagr_p10": float(np.percentile(cagrs, 10)),
            "cagr_median": float(np.median(cagrs)),
            "cagr_mean": float(np.mean(cagrs)),
            "cagr_p90": float(np.percentile(cagrs, 90)),
            "edge_median": float(np.median(edges)),
            "final_median": float(np.median(finals)),
            "win_rate_median": float(np.nanmedian(win_rates)),
        })
    return pd.DataFrame(rows)


def main(strategy_name="apex_balanced", k=5, exit_name="monthly_rebalance"):
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")
    start, end = "2002-01-31", "2024-12-31"

    name_to_fn = {
        "apex_balanced": apex_balanced,
        "apex_reloaded": apex_reloaded,
        "apex_turbocharged": apex_turbocharged,
        "apex_hybrid": apex_hybrid,
        "strategy_rotation": strategy_rotation,
    }
    sfn = name_to_fn[strategy_name]
    if exit_name == "monthly_rebalance":
        exr = ExitSpec("monthly_rebalance", monthly_rebalance=True)
    elif exit_name == "trail_35":
        exr = ExitSpec("trail_35", trail=0.35)
    elif exit_name == "trail_25":
        exr = ExitSpec("trail_25", trail=0.25)
    else:
        exr = ExitSpec("hold_forever")

    print(f"Running survivorship overlay: {strategy_name} k={k} {exit_name} on {start}..{end}", flush=True)
    df = run_overlay(panel, strategy_name, sfn, k, exr, start, end, eval_at, iters=30)
    out = CACHE / f"survivorship_v3_{strategy_name}_k{k}_{exit_name}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "apex_balanced"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    e = sys.argv[3] if len(sys.argv) > 3 else "monthly_rebalance"
    main(s, k, e)
