"""Parameter sensitivity of the v3 winner — make sure performance is not
on a knife-edge.

Around the winner (K=3 EW tight h=6), perturb:
  - K = 2, 3, 4, 5
  - hold = 3, 6, 9, 12
  - cost_bps = 5, 10, 15, 20, 30
  - regime gate = tight, strict, ddgate

For each perturbation, simulate and report WF mean CAGR + edge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    panel = build_panel_with_score("ml_3plus6")

    full_dates = pd.DatetimeIndex(sorted(panel["asof"].unique()))
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })

    # Baseline: K=3 EW tight h=6
    base = Variant(name="base", scorer="ml_3plus6",
                   k_normal=3, k_recovery=3, k_bull=3,
                   weighting="ew", regime_gate="tight",
                   hold_months=6, cap_per_pick=1.0)

    rows = []

    # K perturbation (hold=6 EW tight)
    for k in [1, 2, 3, 4, 5, 7]:
        v = Variant(name=f"K={k}", scorer="ml_3plus6",
                    k_normal=k, k_recovery=max(2, k-1), k_bull=max(2, k-1),
                    weighting="ew", regime_gate="tight",
                    hold_months=6, cap_per_pick=1.0)
        eq = simulate_variant(panel, monthly_returns, spy_features, v)
        m = evaluate(eq, spy_aligned, v.name)
        m["param"] = "K"
        m["value"] = k
        rows.append(m)

    # Hold perturbation (K=3 EW tight)
    for h in [1, 2, 3, 4, 6, 9, 12]:
        v = Variant(name=f"H={h}", scorer="ml_3plus6",
                    k_normal=3, k_recovery=3, k_bull=3,
                    weighting="ew", regime_gate="tight",
                    hold_months=h, cap_per_pick=1.0)
        eq = simulate_variant(panel, monthly_returns, spy_features, v)
        m = evaluate(eq, spy_aligned, v.name)
        m["param"] = "hold"
        m["value"] = h
        rows.append(m)

    # Cost perturbation
    for cost in [0, 5, 10, 15, 20, 30, 50]:
        v = Variant(name=f"cost={cost}bp", scorer="ml_3plus6",
                    k_normal=3, k_recovery=3, k_bull=3,
                    weighting="ew", regime_gate="tight",
                    hold_months=6, cap_per_pick=1.0)
        eq = simulate_variant(panel, monthly_returns, spy_features, v, cost_bps=cost)
        m = evaluate(eq, spy_aligned, v.name)
        m["param"] = "cost_bps"
        m["value"] = cost
        rows.append(m)

    # Gate perturbation
    for g in ["tight", "strict", "ddgate"]:
        v = Variant(name=f"gate={g}", scorer="ml_3plus6",
                    k_normal=3, k_recovery=3, k_bull=3,
                    weighting="ew", regime_gate=g,
                    hold_months=6, cap_per_pick=1.0)
        eq = simulate_variant(panel, monthly_returns, spy_features, v)
        m = evaluate(eq, spy_aligned, v.name)
        m["param"] = "gate"
        m["value"] = g
        rows.append(m)

    # Weighting perturbation
    for w in ["ew", "conv", "invvol", "softmax"]:
        v = Variant(name=f"w={w}", scorer="ml_3plus6",
                    k_normal=3, k_recovery=3, k_bull=3,
                    weighting=w, regime_gate="tight",
                    hold_months=6, cap_per_pick=1.0)
        eq = simulate_variant(panel, monthly_returns, spy_features, v)
        m = evaluate(eq, spy_aligned, v.name)
        m["param"] = "weighting"
        m["value"] = w
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(PIT / "v3_winner_sensitivity.csv", index=False)
    cols = ["param", "value", "name", "cagr_full", "edge_full_pp",
            "wf_mean_cagr", "wf_min_cagr", "wf_max_cagr",
            "wf_mean_edge_pp", "wf_n_beats", "max_dd", "sharpe"]
    print("\n=== Parameter sensitivity (K=3 EW tight h=6 baseline) ===")
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
