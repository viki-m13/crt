"""Survivorship overlay for the winner: strategy_rotation k=5 monthly_rebalance.

Run synthetic delisting injection at α ∈ {0%, 4%, 8%, 12%, 16%, 20%}/yr
with multiple seeds, report median + p10/p90.

Each iteration: with probability p_del per month per pick, replace the entry
with a forced -100% return. The engine subtracts that from cash immediately.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import benchmark_spy_dca
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.fast_monthly_rebalance import run_monthly_rebalance
from experiments.monthly_dca.strategies_ensemble import strategy_rotation


CACHE = Path(__file__).resolve().parent / "cache"


def main():
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")
    start, end = "2002-01-31", "2024-12-31"

    spy = benchmark_spy_dca(panel, start, end, eval_at=eval_at)
    print(f"SPY DCA: cagr_xirr={spy['cagr_xirr']:.4f}", flush=True)

    rows = []
    for alpha in [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20]:
        cagrs, finals = [], []
        for seed in range(30):
            res = run_monthly_rebalance(
                panel, strategy_rotation, top_k=5,
                start=start, end=end, eval_at=eval_at,
                delist_alpha=alpha, delist_seed=seed, cost_bps=5.0,
            )
            cagrs.append(res["cagr_xirr"])
            finals.append(res["final_equity"])
        rows.append({
            "alpha": alpha,
            "cagr_p10": float(np.percentile(cagrs, 10)),
            "cagr_p25": float(np.percentile(cagrs, 25)),
            "cagr_median": float(np.median(cagrs)),
            "cagr_mean": float(np.mean(cagrs)),
            "cagr_p75": float(np.percentile(cagrs, 75)),
            "cagr_p90": float(np.percentile(cagrs, 90)),
            "edge_median": float(np.median(cagrs) - spy["cagr_xirr"]),
            "final_median": float(np.median(finals)),
        })
        print(f"  alpha={alpha:.2f}: median CAGR={np.median(cagrs):.4f}  "
              f"p10={np.percentile(cagrs, 10):.4f}  p90={np.percentile(cagrs, 90):.4f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "winner_bias_sensitivity_v3.csv", index=False)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
