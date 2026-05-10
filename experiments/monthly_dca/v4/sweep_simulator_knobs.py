"""Sweep v4 simulator knobs holding ML scorer constant (ml_3plus6).

Tests:
  - Stop-loss thresholds: -0.20, -0.30, -0.40, disabled
  - Take-profit thresholds: +0.50, +0.75, +1.00, disabled
  - Stop-to-cash vs redistribute
  - Score thresholds (top-K avg score): 0.50, 0.55, 0.60, none
  - Regime gates: tight, breadth_tight, multi
  - K combos: (3,3,3), (2,2,2), (3,2,4), (5,3,3)

Reports best variants by WF mean OOS CAGR.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import pandas as pd

from v4_engine import (
    V4Variant, simulate_v4, evaluate_v4, load_spy_features, build_panel_with_score,
    build_spy_aligned, PIT, V2,
)


def main():
    print("=== Loading inputs ===")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    panel = build_panel_with_score("ml_3plus6")
    spy_aligned = build_spy_aligned(PIT / "sp500_pit_panel.parquet", monthly_returns)

    # Sweep grid
    stop_losses = [-1.0, -0.40, -0.30, -0.25, -0.20]
    take_profits = [1e9, 1.50, 1.00, 0.75]
    stop_modes = [False, True]   # False=redistribute, True=stop_to_cash
    score_thresholds = [-1e9, 0.50, 0.55, 0.60]
    regime_gates = ["tight", "breadth_tight", "multi"]
    k_combos = [(3, 3, 3), (2, 2, 2), (3, 2, 4), (4, 3, 4)]
    holds = [6]
    weightings = ["ew"]

    rows = []
    t0 = time.time()
    n = 0
    total = (len(stop_losses) * len(take_profits) * len(stop_modes) *
             len(score_thresholds) * len(regime_gates) * len(k_combos) *
             len(holds) * len(weightings))
    print(f"=== Sweep: {total} variants ===")

    for sl in stop_losses:
        for tp in take_profits:
            for sm in stop_modes:
                for st in score_thresholds:
                    for g in regime_gates:
                        for kn, kr, kb in k_combos:
                            for h in holds:
                                for w in weightings:
                                    name = f"sl{sl}|tp{tp}|sm{int(sm)}|st{st}|{g}|k{kn}{kr}{kb}|h{h}|{w}"
                                    v = V4Variant(
                                        name=name, scorer="ml_3plus6",
                                        k_normal=kn, k_recovery=kr, k_bull=kb,
                                        weighting=w, regime_gate=g,
                                        hold_months=h, cap_per_pick=1.0,
                                        score_threshold=st,
                                        stop_loss=sl, take_profit=tp,
                                        stop_to_cash=sm,
                                    )
                                    eq = simulate_v4(panel, monthly_returns, spy, v)
                                    res = evaluate_v4(eq, spy_aligned, name)
                                    res.update({
                                        "sl": sl, "tp": tp, "sm": int(sm), "st": st,
                                        "gate": g, "kn": kn, "kr": kr, "kb": kb, "h": h, "w": w,
                                    })
                                    rows.append(res)
                                    n += 1
                                    if n % 100 == 0:
                                        elapsed = time.time() - t0
                                        eta = elapsed / n * (total - n)
                                        print(f"  {n}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    df = pd.DataFrame(rows).sort_values("wf_mean_cagr", ascending=False)
    out = PIT / "v4_simulator_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n=== Top 25 by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_n_pos", "wf_n_beats", "wf_min_cagr", "max_dd", "n_cash"]
    print(df[cols].head(25).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
