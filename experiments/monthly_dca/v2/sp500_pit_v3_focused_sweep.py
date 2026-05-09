"""Focused sweep on the v3 winner config space.

The base sweep (1755 variants) found K=3 EW tight h=6 is the composite
winner.  Sensitivity confirmed the result is robust.  This focused sweep
tests:
  - New scorers: ml_h1, ml_h3, ml_h6, ml_3plus6, ml_filter_winsor, ml_q,
    ml_filter_softmax
  - K = 2, 3
  - hold = 6, 9
  - EW only
  - tight gate only

Goal: see if any new scorer variant beats ml_filter K=3 EW tight h=6.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v2"))

from sp500_pit_extended_sweep import (  # noqa: E402
    build_panel_with_score, simulate_variant, evaluate, load_spy_features,
    Variant,
)


def main():
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()

    full_dates = pd.DatetimeIndex(sorted(pd.read_parquet(PIT / "sp500_pit_panel.parquet")["asof"].unique()))
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })

    scorers = ["ml_filter", "ml_h1", "ml_h3", "ml_h6", "ml_3plus6",
               "ml_filter_winsor", "ml_q", "ml_filter_softmax"]
    panel_cache = {}
    rows = []
    t0 = time.time()
    for scorer in scorers:
        if scorer not in panel_cache:
            panel_cache[scorer] = build_panel_with_score(scorer)
        for k in [2, 3]:
            for h in [6, 9]:
                for w in ["ew", "conv"]:
                    v = Variant(name=f"{scorer}|K={k}|{w}|tight|h={h}",
                                scorer=scorer, k_normal=k, k_recovery=k, k_bull=k,
                                weighting=w, regime_gate="tight",
                                hold_months=h, cap_per_pick=1.0)
                    eq = simulate_variant(panel_cache[scorer], monthly_returns,
                                          spy_features, v)
                    m = evaluate(eq, spy_aligned, v.name)
                    m["scorer"] = scorer
                    m["k"] = k
                    m["hold"] = h
                    m["weighting"] = w
                    rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(PIT / "v3_focused_sweep.csv", index=False)
    print(f"\nDone in {time.time()-t0:.0f}s. {len(df)} variants.")
    cols = ["name", "cagr_full", "edge_full_pp", "wf_mean_cagr", "wf_min_cagr",
            "wf_max_cagr", "wf_mean_edge_pp", "wf_n_beats", "max_dd", "sharpe"]
    print("\n=== Sorted by wf_mean_cagr ===")
    print(df.sort_values("wf_mean_cagr", ascending=False)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
