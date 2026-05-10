"""Sweep over pillar parameters to find the actually-best multi-pillar config.

Tests:
  - drop_failure_pct in {0.0, 0.10, 0.20, 0.30}
  - trend_gate on/off, with PERMISSIVE setting (mom_12_1_min = -0.50)
  - failure as filter vs failure as score-penalty
  - composite weights for archetype and novel
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))

from lib_engine import (  # noqa: E402
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)
from experiments.multi_pillar_43Agh.strategy import selection, trend_regime  # noqa: E402

OUT = ROOT / "experiments" / "multi_pillar_43Agh" / "backtests"
OUT.mkdir(parents=True, exist_ok=True)


PERMISSIVE_GATE = {
    "mom_12_1_min": -0.50,
    "mom_3_min": -0.20,
    "d_sma200_min": -0.20,
    "dd_from_52wh_min": -0.65,
    "frac_above_50dma_1y_min": 0.10,
}


def run_cfg(panel: pd.DataFrame, monthly_returns, spy_feats, name: str,
            weighting="invvol", cash_yield_yr=0.03, k=3, hold_months=6):
    cfg = V6Config(name=name, scorer="ml_3plus6", universe="sp500_pit",
                   regime_gate="tight", k_normal=k, k_recovery=k, k_bull=k,
                   weighting=weighting, hold_months=hold_months, cost_bps=10.0,
                   cash_yield_yr=cash_yield_yr)
    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    return evaluate(eq, spy_aln, name)


def fmt(m: dict) -> str:
    return (f"CAGR={m['cagr_full']*100:6.2f}%  S={m['sharpe']:5.3f}  "
            f"DD={m['max_dd']*100:6.2f}%  WFmCAGR={m['wf_mean_cagr']*100:5.1f}%  "
            f"WFmS={m['wf_mean_sharpe']:.2f}  beats={m['wf_n_beats_spy']}/10")


def main():
    print("[load] base data ...")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    rows = []

    # Sweep 1: failure-filter only at varying drop %
    for drop in [0.0, 0.10, 0.20, 0.30, 0.40]:
        t0 = time.time()
        panel = selection.build_composite_panel(
            drop_failure_pct=drop, apply_trend_gate=False,
            w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.0)
        m = run_cfg(panel, monthly_returns, spy_feats, f"fail_{int(drop*100)}pct")
        rows.append({"sweep": "fail_pct", "drop": drop, **m,
                     "elapsed_s": time.time() - t0})
        print(f"fail_{int(drop*100):02d}%: {fmt(m)}")

    # Sweep 2: failure as score-penalty (no filter), varying weight
    for wf in [0.0, 0.20, 0.40, 0.60]:
        t0 = time.time()
        panel = selection.build_composite_panel(
            drop_failure_pct=0.0, apply_trend_gate=False,
            w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=wf)
        m = run_cfg(panel, monthly_returns, spy_feats, f"penalty_w{int(wf*100)}")
        rows.append({"sweep": "fail_penalty", "w_failure": wf, **m,
                     "elapsed_s": time.time() - t0})
        print(f"pen_w{int(wf*100):02d}: {fmt(m)}")

    # Sweep 3: permissive trend gate alone
    saved_default = trend_regime.TREND_GATE_DEFAULT.copy()
    trend_regime.TREND_GATE_DEFAULT.update(PERMISSIVE_GATE)
    t0 = time.time()
    panel = selection.build_composite_panel(
        drop_failure_pct=0.0, apply_trend_gate=True,
        w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.0)
    m = run_cfg(panel, monthly_returns, spy_feats, "trend_permissive")
    rows.append({"sweep": "trend_permissive", "drop": 0, **m,
                 "elapsed_s": time.time() - t0})
    print(f"trend_perm: {fmt(m)}")
    trend_regime.TREND_GATE_DEFAULT.update(saved_default)

    # Sweep 4: best-from-above combinations
    for drop, wf, wa, wn, wc in [
        (0.10, 0.0, 0.0, 0.0, 0.0),
        (0.10, 0.20, 0.0, 0.0, 0.0),
        (0.10, 0.40, 0.20, 0.0, 0.0),
        (0.20, 0.0, 0.20, 0.0, 0.20),
        (0.10, 0.0, 0.10, 0.10, 0.0),
        (0.20, 0.40, 0.10, 0.0, 0.0),
    ]:
        t0 = time.time()
        panel = selection.build_composite_panel(
            drop_failure_pct=drop, apply_trend_gate=False,
            w_ml=1.0, w_archetype=wa, w_novel=wn, w_classic=wc, w_failure=wf)
        name = f"combo_drop{int(drop*100)}_wf{int(wf*10)}_wa{int(wa*10)}_wn{int(wn*10)}_wc{int(wc*10)}"
        m = run_cfg(panel, monthly_returns, spy_feats, name)
        rows.append({"sweep": "combo", "drop": drop, "w_failure": wf,
                     "w_arch": wa, "w_novel": wn, "w_classic": wc,
                     **m, "elapsed_s": time.time() - t0})
        print(f"{name}: {fmt(m)}")

    # Sweep 5: K variations with best filter
    for k in [3, 4, 5]:
        for drop in [0.10, 0.20]:
            t0 = time.time()
            panel = selection.build_composite_panel(
                drop_failure_pct=drop, apply_trend_gate=False,
                w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.20)
            m = run_cfg(panel, monthly_returns, spy_feats, f"k{k}_drop{int(drop*100)}", k=k)
            rows.append({"sweep": "k_drop", "k": k, "drop": drop, **m,
                         "elapsed_s": time.time() - t0})
            print(f"k{k}_drop{int(drop*100)}: {fmt(m)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep_results.csv", index=False)
    print(f"\nSweep complete; saved -> {OUT}/sweep_results.csv")
    # Sort by Sharpe-CAGR composite
    df["composite"] = df["sharpe"] * 0.5 + df["cagr_full"]
    print("\ntop-10 by composite (Sharpe*0.5 + CAGR):")
    print(df.nlargest(10, "composite")[["sweep", "cagr_full", "sharpe", "max_dd",
                                          "wf_mean_cagr", "wf_n_beats_spy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
