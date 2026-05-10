"""Test Options A/B (and v3) on multiple universes — independent verification."""
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


def random_subset(panel: pd.DataFrame, k: int = 500, seed: int = 1) -> pd.DataFrame:
    tickers = sorted(panel["ticker"].unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(tickers, size=min(k, len(tickers)), replace=False))
    return panel[panel["ticker"].isin(pick)].reset_index(drop=True)


def main():
    print("[load monthly_returns/spy]")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    panels = {
        "sp500_pit": load_score_panel("ml_3plus6", "sp500_pit"),
        "broader_1811": load_score_panel("ml_3plus6", "broader"),
        "non_sp500": load_score_panel("ml_3plus6", "non_sp500"),
    }
    for seed in [1, 2, 3, 4, 5]:
        panels[f"rand500_s{seed}"] = random_subset(panels["broader_1811"], k=500, seed=seed)

    cfgs = {
        "v3":  V6Config(name="v3",  scorer="ml_3plus6", regime_gate="tight",
                        k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                        hold_months=6, cost_bps=10.0),
        "A":   V6Config(name="A",   scorer="ml_3plus6", regime_gate="tight",
                        k_normal=3, k_recovery=3, k_bull=3, weighting="invvol",
                        hold_months=6, cost_bps=10.0, cash_yield_yr=0.03),
        "B":   V6Config(name="B",   scorer="ml_3plus6", regime_gate="tight",
                        k_normal=3, k_recovery=3, k_bull=2, weighting="invvol",
                        hold_months=6, cost_bps=10.0, cash_yield_yr=0.03),
    }

    rows = []
    for u, panel in panels.items():
        for label, cfg0 in cfgs.items():
            cfg = V6Config(**{**cfg0.__dict__, "universe": u, "name": f"{label}__{u}"})
            try:
                t0 = time.time()
                eq = simulate(cfg, panel, monthly_returns, spy_feats)
                spy_aln = build_spy_aligned(eq, monthly_returns)
                m = evaluate(eq, spy_aln, cfg.name)
                m["universe"] = u
                m["strategy"] = label
                rows.append(m)
                print(f"[{u:15s}|{label}] {time.time()-t0:.1f}s "
                      f"CAGR={m['cagr_full']*100:6.2f}% Sh={m['sharpe']:.3f} "
                      f"MDD={m['max_dd']*100:6.2f}% WFmean={m['wf_mean_cagr']*100:6.2f}% "
                      f"WFmin={m['wf_min_cagr']*100:6.2f}% beats={m['wf_n_beats_spy']}/{m['wf_n_splits']}")
            except Exception as e:
                print(f"[{u}|{label}] ERROR: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "generalize_options_summary.csv", index=False)

    print("\n=== Sharpe by universe ===")
    print(df.pivot_table(index="universe", columns="strategy", values="sharpe").round(3).to_string())
    print("\n=== WF mean CAGR by universe ===")
    print(df.pivot_table(index="universe", columns="strategy", values="wf_mean_cagr").round(3).to_string())
    print("\n=== Full CAGR by universe ===")
    print(df.pivot_table(index="universe", columns="strategy", values="cagr_full").round(3).to_string())
    print("\n=== MaxDD by universe ===")
    print(df.pivot_table(index="universe", columns="strategy", values="max_dd").round(3).to_string())
    print("\n=== beats SPY by universe ===")
    print(df.pivot_table(index="universe", columns="strategy", values="wf_n_beats_spy").to_string())


if __name__ == "__main__":
    main()
