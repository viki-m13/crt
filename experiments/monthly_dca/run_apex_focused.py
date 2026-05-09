"""Focused APEX sweep: test smaller candidate set, full window first.

Designed for fast iteration. Outputs progress every combo.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import (
    ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_apex import all_apex_strategies
from experiments.monthly_dca.strategies_v3 import all_v3_strategies
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import quality_pullback


CACHE = Path(__file__).resolve().parent / "cache"


# Focused candidate set
CANDIDATES = {
    "strategy_rotation": strategy_rotation,
    "quality_pullback": quality_pullback,
    **all_v3_strategies(),
    **all_apex_strategies(),
}

EXITS = [
    ExitSpec("hold_forever"),
    ExitSpec("trail_25", trail=0.25),
    ExitSpec("trail_35", trail=0.35),
    ExitSpec("monthly_rebalance", monthly_rebalance=True),
    ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3),
]
KS = [3, 5]


def sweep_window(panel, start: str, end: str, eval_at: pd.Timestamp, label: str) -> pd.DataFrame:
    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)
    print(f"\n[{label}] SPY DCA CAGR XIRR={spy['cagr_xirr']:.4f}", flush=True)

    rows = []
    total = len(CANDIDATES) * len(KS) * len(EXITS)
    done = 0
    for sname, sfn in CANDIDATES.items():
        for k in KS:
            for exr in EXITS:
                if "perfect_storm" in sname and k == 5:
                    continue
                done += 1
                t0 = time.time()
                try:
                    res = run_compound(
                        panel, CompStrategy(sname, sfn, top_k=k), exr,
                        start=start, end=end, eval_at=eval_at, cost_bps=5.0,
                    )
                    dt = time.time() - t0
                    edge = res.cagr_money_weighted - spy["cagr_xirr"]
                    rows.append({
                        "strategy": sname, "k": k, "exit": exr.name,
                        "n_trades": res.n_trades,
                        "deposited": res.total_deposited,
                        "final_equity": res.final_equity,
                        "cagr_xirr": res.cagr_money_weighted,
                        "cagr_total": res.cagr_total_money,
                        "cagr_spy": spy["cagr_xirr"],
                        "edge": edge, "elapsed_sec": dt, "window": label,
                    })
                    print(f"  [{done}/{total}] {sname:25s} k={k} {exr.name:18s}: "
                          f"CAGR={res.cagr_money_weighted:.4f} edge={edge:+.4f} ({dt:.1f}s)",
                          flush=True)
                except Exception as e:
                    print(f"  FAIL {sname} k={k} {exr.name}: {e}", flush=True)
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")

    df_full = sweep_window(panel, "2002-01-31", "2024-12-31", eval_at, "full_2002_2024")
    df_full.to_csv(CACHE / "sweep_apex_focused_full.csv", index=False)
    print(f"\nWrote sweep_apex_focused_full.csv ({len(df_full)} rows)")
    print("\n=== TOP-30 (full 2002-2024) ===")
    print(df_full.nlargest(30, "cagr_xirr")[["strategy", "k", "exit", "cagr_xirr", "edge", "n_trades"]].to_string())


if __name__ == "__main__":
    main()
