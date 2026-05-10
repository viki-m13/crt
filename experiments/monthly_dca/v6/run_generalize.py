"""Generalization test — apply the v6 winner to non-S&P 500 universes.

Universes tested:
  - sp500_pit       (the home universe) — sanity check
  - broader_1811    (full ml_preds_v2 panel — non-PIT)
  - non_sp500       (only tickers NEVER in S&P 500 at the asof in question)
  - random_500_x5   (5 different seeded random subsets of 500 tickers from the broader)

For each, we report CAGR / Sharpe / MaxDD / WF mean.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib_engine import (
    EXCLUDE_TICKERS, V2, V6Config, build_spy_aligned, evaluate,
    load_score_panel, load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def random_subset(panel: pd.DataFrame, k: int = 500, seed: int = 1) -> pd.DataFrame:
    """Random subset of `k` tickers from the broader panel, fixed for all asofs."""
    tickers = sorted(panel["ticker"].unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(tickers, size=min(k, len(tickers)), replace=False))
    return panel[panel["ticker"].isin(pick)].reset_index(drop=True)


def run_panel(panel: pd.DataFrame, mr: pd.DataFrame, spy: pd.DataFrame, cfg: V6Config) -> dict:
    eq = simulate(cfg, panel, mr, spy)
    spy_aln = build_spy_aligned(eq, mr)
    return evaluate(eq, spy_aln, cfg.name), eq


def main():
    print("[load]")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    # Define the "winner" config from sweep3 — placeholder for now
    # We'll override with actual best after sweep3 finishes.
    winner_cfg_kwargs = dict(
        scorer="ml_3plus6", regime_gate="tight",
        k_normal=3, k_recovery=3, k_bull=3,
        weighting="invvol", hold_months=6, cost_bps=10.0,
        cash_yield_yr=0.03, monthly_exposure=False,
        spy_dd_scale=0.0, spy_dd_floor=0.5,
        pullback_filter=0.0, vol_penalty=0.0,
        crash_persist=1, ts_reset_on_reentry=True,
    )

    rows = []
    eqs = {}

    panels = {
        "sp500_pit": load_score_panel("ml_3plus6", "sp500_pit", attach_pullback=True),
        "broader_1811": load_score_panel("ml_3plus6", "broader", attach_pullback=True),
        "non_sp500": load_score_panel("ml_3plus6", "non_sp500", attach_pullback=True),
    }
    for seed in [1, 2, 3, 4, 5]:
        panels[f"random_500_seed{seed}"] = random_subset(panels["broader_1811"], k=500, seed=seed)

    for univ, panel in panels.items():
        print(f"\n[{univ}] panel rows={len(panel)} asofs={panel['asof'].nunique()} tickers={panel['ticker'].nunique()}")
        for variant in ("v3_baseline", "v6_winner"):
            kw = dict(winner_cfg_kwargs)
            if variant == "v3_baseline":
                kw.update(weighting="ew", cash_yield_yr=0.0)
            cfg = V6Config(name=f"{univ}|{variant}", universe=univ, **kw)
            try:
                m, _ = run_panel(panel, monthly_returns, spy_feats, cfg)
                m["universe"] = univ
                m["variant"] = variant
                rows.append(m)
                print(f"  {variant:13s}: cagr={m['cagr_full']:.4f} sh={m['sharpe']:.4f} mdd={m['max_dd']:.4f} wf={m['wf_mean_cagr']:.4f} wmin={m['wf_min_cagr']:.4f} npos={m['wf_n_pos']} beats={m['wf_n_beats_spy']}")
            except Exception as e:
                print(f"  {variant} FAILED: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v6_generalize_results.csv", index=False)
    print(f"\nSaved -> {OUT}/v6_generalize_results.csv")


if __name__ == "__main__":
    main()
