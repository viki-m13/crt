"""Validate Option A: v6 invvol weighting + 3% cash yield."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def run(cfg, panel, monthly_returns, spy_feats):
    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    return evaluate(eq, spy_aln, cfg.name), eq


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    cfgs = [
        V6Config(name="v3", scorer="ml_3plus6", universe="sp500_pit",
                 regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="ew", hold_months=6, cost_bps=10.0),
        V6Config(name="A_v6_invvol_only", scorer="ml_3plus6", universe="sp500_pit",
                 regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="invvol", hold_months=6, cost_bps=10.0),
        V6Config(name="A_v6_invvol_cy3", scorer="ml_3plus6", universe="sp500_pit",
                 regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="invvol", hold_months=6, cost_bps=10.0,
                 cash_yield_yr=0.03),
        V6Config(name="A_v6_ew_cy3", scorer="ml_3plus6", universe="sp500_pit",
                 regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="ew", hold_months=6, cost_bps=10.0,
                 cash_yield_yr=0.03),
    ]

    rows = []
    for cfg in cfgs:
        t0 = time.time()
        m, eq = run(cfg, panel, monthly_returns, spy_feats)
        eq.to_csv(OUT / f"{cfg.name}_eq.csv", index=False)
        rows.append(m)
        print(f"[{cfg.name}] {time.time()-t0:.1f}s  "
              f"CAGR={m['cagr_full']*100:.2f}% Sh={m['sharpe']:.3f} "
              f"MDD={m['max_dd']*100:.2f}% WFmean={m['wf_mean_cagr']*100:.2f}% "
              f"WFmin={m['wf_min_cagr']*100:.2f}% pos={m['wf_n_pos']}/{m['wf_n_splits']} "
              f"beatSPY={m['wf_n_beats_spy']}/{m['wf_n_splits']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "optionA_summary.csv", index=False)
    print("\n=== Summary ===")
    print(df[["name", "cagr_full", "sharpe", "max_dd", "wf_mean_cagr",
              "wf_min_cagr", "wf_n_beats_spy", "wf_mean_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
