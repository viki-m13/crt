"""Targeted APEX sweep: focus on most promising combos.

Based on early results: monthly_rebalance is the best exit by far.
Test ALL strategies (incl apex_v2) with monthly_rebalance + a couple alternatives.
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
from experiments.monthly_dca.strategies_apex_v2 import all_apex_v2_strategies
from experiments.monthly_dca.strategies_v3 import all_v3_strategies
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import quality_pullback


CACHE = Path(__file__).resolve().parent / "cache"


CANDIDATES = {
    "strategy_rotation": strategy_rotation,
    "quality_pullback": quality_pullback,
    **all_v3_strategies(),
    **all_apex_strategies(),
    **all_apex_v2_strategies(),
}

# Target only the most promising exits
EXITS = [
    ExitSpec("monthly_rebalance", monthly_rebalance=True),
    ExitSpec("trail_35", trail=0.35),
    ExitSpec("hold_forever"),
]
KS = [3, 5]


def sweep(panel, start, end, eval_at, label):
    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)
    print(f"\n[{label}] SPY DCA={spy['cagr_xirr']:.4f}", flush=True)
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
                        "n_trades": res.n_trades, "deposited": res.total_deposited,
                        "final_equity": res.final_equity,
                        "cagr_xirr": res.cagr_money_weighted,
                        "cagr_total": res.cagr_total_money,
                        "cagr_spy": spy["cagr_xirr"],
                        "edge": edge, "elapsed_sec": dt, "window": label,
                    })
                    print(f"  [{done}/{total}] {sname:25s} k={k} {exr.name:18s}: "
                          f"CAGR={res.cagr_money_weighted:.4f} edge={edge:+.4f}", flush=True)
                except Exception as e:
                    print(f"  FAIL {sname} k={k} {exr.name}: {e}", flush=True)
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")
    df_full = sweep(panel, "2002-01-31", "2024-12-31", eval_at, "full_2002_2024")
    df_full.to_csv(CACHE / "sweep_apex_targeted_full.csv", index=False)
    df_modern = sweep(panel, "2010-01-31", "2024-12-31", eval_at, "modern_2010_2024")
    df_modern.to_csv(CACHE / "sweep_apex_targeted_modern.csv", index=False)
    df_recent = sweep(panel, "2018-01-31", "2024-12-31", eval_at, "recent_2018_2024")
    df_recent.to_csv(CACHE / "sweep_apex_targeted_recent.csv", index=False)

    print("\n=== TOP-30 by FULL window CAGR ===", flush=True)
    print(df_full.nlargest(30, "cagr_xirr").to_string(), flush=True)
    print("\n=== TOP-30 by MODERN window CAGR ===", flush=True)
    print(df_modern.nlargest(30, "cagr_xirr").to_string(), flush=True)


if __name__ == "__main__":
    main()
