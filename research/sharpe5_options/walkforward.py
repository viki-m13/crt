#!/usr/bin/env python3
"""Walk-forward harness: parameters chosen only on data strictly before each
test year, then applied out-of-sample. Reports per-fold and pooled OOS Sharpe.

This is the anti-overfit backbone for any sleeve whose gate has a tunable
threshold. A sleeve that only works with hindsight-chosen parameters shows up
here as pooled OOS Sharpe far below its in-sample value.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import engine as E
import portfolio as P

HERE = os.path.dirname(os.path.abspath(__file__))


def walk_forward(build_series, grid: list, years=(2021, 2022, 2023, 2024, 2025, 2026),
                 min_train_obs=60, label="wf"):
    """build_series(param) -> pd.Series indexed by date (entry-bucketed returns).

    For each test year: pick the param maximizing screening Sharpe on all data
    strictly before Jan 1 of that year, then record that param's returns
    DURING the test year. Pooled OOS = concatenation of all test-year slices.
    """
    series = {p: build_series(p) for p in grid}
    picks, oos_parts = {}, []
    for y in years:
        cut = pd.Timestamp(f"{y}-01-01")
        best, best_sh = None, -np.inf
        for p, s in series.items():
            tr = s[s.index < cut]
            if len(tr) < min_train_obs:
                continue
            sh = P.screen_sharpe(tr).get("sharpe_scr", np.nan)
            if np.isfinite(sh) and sh > best_sh:
                best, best_sh = p, sh
        if best is None:
            continue
        te = series[best]
        te = te[(te.index >= cut) & (te.index < pd.Timestamp(f"{y+1}-01-01"))]
        if len(te) < 5:
            continue
        picks[y] = (best, round(best_sh, 2), round(
            P.screen_sharpe(te).get("sharpe_scr", np.nan), 2), len(te))
        oos_parts.append(te)
    if not oos_parts:
        print(f"{label}: no folds")
        return None
    pooled = pd.concat(oos_parts).sort_index()
    print(f"\n=== walk-forward: {label} ===")
    for y, (p, tr_sh, te_sh, n) in picks.items():
        print(f"  {y}: param={p} train_sh={tr_sh:+.2f} -> TEST_sh={te_sh:+.2f} (n={n})")
    pool = P.screen_sharpe(pooled)
    print(f"  POOLED OOS: sharpe={pool.get('sharpe_scr'):+.2f} "
          f"hit={pool.get('hit', float('nan')):.2f} n={pool.get('n_weeks')}")
    return pooled
