"""Frozen holdouts for generalization testing.

Two holdouts, applied independently:
  1. TIME HOLDOUT: 2024-07 -> latest. Strategy must perform on the most
     recent 18+ months never seen during candidate selection.
  2. UNIVERSE HOLDOUT: 30% of tickers (alphabetical-hash split) reserved.
     Strategy is tested on this ticker subset only.

These run AFTER the walk-forward gauntlet, never before.  The frozen
holdout is allowed exactly one run.
"""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import ExitSpec, run_compound, benchmark_spy_dca
from experiments.monthly_dca.fast_engine import load_panel


def universe_holdout_mask(tickers, hash_mod: int = 10, holdout_buckets: tuple = (7, 8, 9)):
    """Deterministic split of tickers into 'in-universe' and 'holdout'."""
    out = {}
    for t in tickers:
        bucket = int(hashlib.sha256(t.encode()).hexdigest(), 16) % hash_mod
        out[t] = bucket in holdout_buckets
    return out


def time_holdout_run(strategy_factory, start: str = "2024-07-31",
                      end: str = "2026-04-30", cost_bps: float = 5.0) -> dict:
    panel = load_panel()
    strat = strategy_factory()
    rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)
    res = run_compound(panel, strat, rule, start=start, end=end, cost_bps=cost_bps)
    spy = benchmark_spy_dca(panel, start, end)
    return {
        "kind": "time_holdout",
        "start": start, "end": end,
        "cagr_strat": res.cagr_money_weighted,
        "cagr_spy": spy["cagr_xirr"],
        "edge": res.cagr_money_weighted - spy["cagr_xirr"],
        "n_months": res.n_months,
        "n_trades": res.n_trades,
        "final_equity": res.final_equity,
        "deposited": res.total_deposited,
    }


def universe_holdout_run(strategy_factory, start: str = "2002-01-31",
                          end: str = "2024-06-30", cost_bps: float = 5.0,
                          hash_mod: int = 10, holdout_buckets=(7, 8, 9)) -> dict:
    """Run the strategy on the held-out ticker subset only.

    Score function is the same; we restrict the panel to the holdout
    tickers and the SPY benchmark is kept.
    """
    panel = load_panel()
    holdout = universe_holdout_mask(panel.columns, hash_mod, holdout_buckets)
    keep = [t for t, isout in holdout.items() if isout or t == "SPY"]
    sub_panel = panel[keep]
    strat = strategy_factory()
    rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)
    res = run_compound(sub_panel, strat, rule, start=start, end=end, cost_bps=cost_bps)
    spy = benchmark_spy_dca(sub_panel, start, end)
    return {
        "kind": "universe_holdout",
        "start": start, "end": end,
        "cagr_strat": res.cagr_money_weighted,
        "cagr_spy": spy["cagr_xirr"],
        "edge": res.cagr_money_weighted - spy["cagr_xirr"],
        "n_months": res.n_months,
        "n_trades": res.n_trades,
        "n_tickers_holdout": int(sum(holdout.values())),
    }
