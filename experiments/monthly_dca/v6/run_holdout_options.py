"""2024-2026 holdout test for v3, A, B (and a recovery probe for 2025 dip).

The walk-forward splits include data up to 2024; the 2025-01 → 2026-04 window
is essentially fresh OOS for all candidates. Both v3 and the v8 2qHxY agent's
k=1 winner reported -32% on 2025; let's see what v6 invvol and v8 kb=2 do.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_engine import (
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2024-05-01")
HOLDOUT_END = pd.Timestamp("2026-12-31")


def evaluate_window(eq: pd.DataFrame, mr: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    eq2 = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy().reset_index(drop=True)
    if len(eq2) == 0:
        return {"cagr_full": float("nan"), "spy_cagr_full": float("nan"),
                "edge_full_pp": float("nan"), "sharpe": float("nan"),
                "max_dd": float("nan")}
    spy_aln = build_spy_aligned(eq2, mr)
    return evaluate(eq2, spy_aln, "holdout")


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    cfgs = {
        "v3": V6Config(name="v3", scorer="ml_3plus6", universe="sp500_pit",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                       weighting="ew", hold_months=6, cost_bps=10.0),
        "A":  V6Config(name="A", scorer="ml_3plus6", universe="sp500_pit",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                       weighting="invvol", hold_months=6, cost_bps=10.0,
                       cash_yield_yr=0.03),
        "B":  V6Config(name="B", scorer="ml_3plus6", universe="sp500_pit",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=2,
                       weighting="invvol", hold_months=6, cost_bps=10.0,
                       cash_yield_yr=0.03),
    }

    rows = []
    for label, cfg in cfgs.items():
        eq = simulate(cfg, panel, monthly_returns, spy_feats)
        eq.to_csv(OUT / f"{label}_full_equity.csv", index=False)
        # Full-window metrics
        full = evaluate(eq, build_spy_aligned(eq, monthly_returns), label)
        # Holdout (2024-05 → 2026)
        hh = evaluate_window(eq, monthly_returns, HOLDOUT_START, HOLDOUT_END)
        # 2025 calendar-year
        cy25 = evaluate_window(eq, monthly_returns, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
        rows.append({
            "strategy": label,
            "full_cagr": full["cagr_full"],
            "full_sharpe": full["sharpe"],
            "full_mdd": full["max_dd"],
            "ho_cagr": hh["cagr_full"], "ho_spy": hh["spy_cagr_full"],
            "ho_edge_pp": hh["edge_full_pp"], "ho_sharpe": hh["sharpe"], "ho_mdd": hh["max_dd"],
            "cy25_cagr": cy25["cagr_full"], "cy25_spy": cy25["spy_cagr_full"],
            "cy25_edge_pp": cy25["edge_full_pp"],
        })
        print(f"\n[{label}] full CAGR={full['cagr_full']*100:.2f}% Sh={full['sharpe']:.3f}")
        print(f"        24-05→26 holdout CAGR={hh['cagr_full']*100:.2f}% vs SPY {hh['spy_cagr_full']*100:.2f}% edge {hh['edge_full_pp']:.1f}pp MDD={hh['max_dd']*100:.2f}%")
        print(f"        CY2025 CAGR={cy25['cagr_full']*100:.2f}% vs SPY {cy25['spy_cagr_full']*100:.2f}% edge {cy25['edge_full_pp']:.1f}pp")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "holdout_options_summary.csv", index=False)
    print("\n=== Holdout summary (2024-05 → present) ===")
    print(df[["strategy", "ho_cagr", "ho_spy", "ho_edge_pp", "ho_mdd"]].round(4).to_string(index=False))
    print("\n=== 2025 calendar year ===")
    print(df[["strategy", "cy25_cagr", "cy25_spy", "cy25_edge_pp"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
