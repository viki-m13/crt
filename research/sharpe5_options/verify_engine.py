#!/usr/bin/env python3
"""Final-verification driver: run a signal-driven sleeve in the event-loop
engine with full M/W/F marks. This produces the ONLY Sharpe numbers we report.

A sleeve spec:
  structure builder(day, chain, quotes, spots, feats_row) -> list[Leg] | None
  entry_filter(feats up to day) -> bool / weights per symbol
Overlapping cycles: a new cohort may open at each obs date; capital is split
across cohorts; each cohort held to expiry (settled by engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as E


def run_sleeve(dates, spot_panel, open_positions_fn, capital=1_000_000.0,
               stock_hedge=False, verbose=False):
    """open_positions_fn(day, chain, quotes, spots, vol, state) -> list[E.Leg]
    Called each obs date; returns new legs to open (engine executes at worst
    side). Position sizing/cohort management is the fn's responsibility via
    state (it can see its own open cohorts in state)."""
    bt = E.Backtester(dates, capital=capital, stock_hedge=stock_hedge)

    def strat(day, chain, quotes, spots, vol, state, eng):
        return open_positions_fn(day, chain, quotes, spots, vol, state)

    eq, fills, state = bt.run(strat, spot_panel, verbose=verbose)
    return eq, fills


def report(eq: pd.Series, label: str, n_trials: int = 1) -> dict:
    m = E.weekly_sharpe(eq)
    lo, hi = E.block_bootstrap_ci(eq)
    w = eq.resample("W-FRI").last().dropna().pct_change().dropna()
    from scipy.stats import skew as _sk, kurtosis as _ku
    dsr = E.deflated_sharpe(m["sharpe"], len(w), n_trials,
                            float(_sk(w)), float(_ku(w, fisher=False)))
    out = {"label": label, **{k: round(v, 4) for k, v in m.items() if isinstance(v, float)},
           "ci95": (round(lo, 2), round(hi, 2)), "dsr": round(dsr, 4),
           "n_weeks": m.get("n_weeks")}
    print(out)
    # per-year Sharpe (excess) for regime dependence
    by_year = {}
    for y, wy in w.groupby(w.index.year):
        if len(wy) >= 20 and wy.std() > 0:
            ex = wy - E.RF_BY_YEAR.get(y, 0.03) / 52.0
            by_year[y] = round(float(ex.mean() / wy.std() * np.sqrt(52)), 2)
    print("  yearly Sharpe:", by_year)
    out["yearly"] = by_year
    return out
