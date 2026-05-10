"""Per-split walk-forward decomposition for v3, A, B.

Verifies the 10/10 beats-SPY claim for option B and looks for whether
the lift survives in each split (not just on average).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib_engine import (
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


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

    from lib_engine import WF_SPLITS, cagr_monthly, sharpe_monthly, maxdd_monthly

    rows = []
    for label, cfg in cfgs.items():
        eq = simulate(cfg, panel, monthly_returns, spy_feats)
        spy_aln = build_spy_aligned(eq, monthly_returns)
        for split, lo, hi in WF_SPLITS:
            lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
            e = eq[(eq["date"] >= lo_t) & (eq["date"] <= hi_t)]
            if len(e) == 0: continue
            r = e["ret_m"].astype(float)
            s_e = spy_aln[(spy_aln["date"] >= lo_t) & (spy_aln["date"] <= hi_t)]
            sr = s_e["spy_ret_m"].astype(float)
            rows.append({
                "strategy": label, "split": split,
                "cagr": cagr_monthly(r), "sharpe": sharpe_monthly(r),
                "max_dd": maxdd_monthly(r),
                "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100,
                "spy_cagr": cagr_monthly(sr),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_split_options.csv", index=False)

    print("\n=== Per-split CAGR by strategy ===")
    piv = df.pivot_table(index="split", columns="strategy", values="cagr").round(4)
    print(piv.to_string())
    print("\n=== Per-split edge vs SPY by strategy ===")
    piv = df.pivot_table(index="split", columns="strategy", values="edge_pp").round(2)
    print(piv.to_string())
    print("\n=== Per-split Sharpe by strategy ===")
    piv = df.pivot_table(index="split", columns="strategy", values="sharpe").round(3)
    print(piv.to_string())
    print("\n=== Per-split MaxDD by strategy ===")
    piv = df.pivot_table(index="split", columns="strategy", values="max_dd").round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
