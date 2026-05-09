"""APEX sweep: full strategy × k × exit grid on compounding engine.

Tests V3 strategies AND APEX strategies on multiple windows, plus
survivorship overlay.
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
    BENCH_EXCLUDED, ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel
from experiments.monthly_dca.strategies_apex import all_apex_strategies
from experiments.monthly_dca.strategies_v3 import all_v3_strategies
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_fast import (
    quality_pullback, explosive_winners, pullback_in_winner,
)


CACHE = Path(__file__).resolve().parent / "cache"


CANDIDATES = {
    # Baselines for comparison
    "strategy_rotation": strategy_rotation,
    "quality_pullback": quality_pullback,
    "explosive_winners": explosive_winners,
    "pullback_in_winner": pullback_in_winner,
    # V3 strategies
    **all_v3_strategies(),
    # APEX strategies
    **all_apex_strategies(),
}

EXITS_TO_TEST = [
    ExitSpec("hold_forever"),
    ExitSpec("trail_25", trail=0.25),
    ExitSpec("trail_35", trail=0.35),
    ExitSpec("trail_50", trail=0.50),
    ExitSpec("monthly_rebalance", monthly_rebalance=True),
    ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3),
    ExitSpec("fixed_3y", days=252 * 3),
]
KS = [1, 2, 3, 5]


def sweep_window(panel, start: str, end: str, eval_at: pd.Timestamp, label: str) -> pd.DataFrame:
    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)
    print(f"\n[{label}] SPY DCA CAGR XIRR={spy['cagr_xirr']:.4f}")

    rows = []
    total = len(CANDIDATES) * len(KS) * len(EXITS_TO_TEST)
    done = 0
    for sname, sfn in CANDIDATES.items():
        for k in KS:
            for exr in EXITS_TO_TEST:
                if "perfect_storm" in sname and k == 5:
                    continue
                done += 1
                try:
                    t0 = time.time()
                    res = run_compound(
                        panel, CompStrategy(sname, sfn, top_k=k), exr,
                        start=start, end=end, eval_at=eval_at, cost_bps=5.0,
                    )
                    dt = time.time() - t0
                    rows.append({
                        "strategy": sname, "k": k, "exit": exr.name,
                        "n_trades": res.n_trades,
                        "deposited": res.total_deposited,
                        "final_equity": res.final_equity,
                        "cagr_xirr": res.cagr_money_weighted,
                        "cagr_total": res.cagr_total_money,
                        "cagr_spy": spy["cagr_xirr"],
                        "edge": res.cagr_money_weighted - spy["cagr_xirr"],
                        "elapsed_sec": dt, "window": label,
                    })
                except Exception as e:
                    print(f"FAIL {sname} k={k} {exr.name}: {e}")
        if done % 20 == 0 or sname in ("apex_balanced", "apex_turbocharged"):
            df_now = pd.DataFrame(rows)
            if not df_now.empty:
                top = df_now.nlargest(5, "cagr_xirr")
                print(f"  Progress {done}/{total}, top so far:")
                for _, r in top.iterrows():
                    print(f"    {r.strategy:25s} k={r.k} {r.exit:18s} -> CAGR {r.cagr_xirr:.4f} edge={r.edge:+.4f}")
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")

    # Run on three windows
    df_recent = sweep_window(panel, "2018-01-31", "2024-12-31", eval_at, "recent_2018_2024")
    df_recent.to_csv(CACHE / "sweep_apex_recent.csv", index=False)
    print(f"\nWrote sweep_apex_recent.csv ({len(df_recent)} rows)")

    df_full = sweep_window(panel, "2002-01-31", "2024-12-31", eval_at, "full_2002_2024")
    df_full.to_csv(CACHE / "sweep_apex_full.csv", index=False)
    print(f"\nWrote sweep_apex_full.csv ({len(df_full)} rows)")

    df_modern = sweep_window(panel, "2010-01-31", "2024-12-31", eval_at, "modern_2010_2024")
    df_modern.to_csv(CACHE / "sweep_apex_modern.csv", index=False)
    print(f"\nWrote sweep_apex_modern.csv ({len(df_modern)} rows)")

    # Combine and report top
    combined = pd.concat([df_full, df_modern, df_recent])
    combined.to_csv(CACHE / "sweep_apex_combined.csv", index=False)

    print("\n=== TOP-30 (full window 2002-2024, by CAGR XIRR) ===")
    print(df_full.nlargest(30, "cagr_xirr").to_string())
    print("\n=== TOP-30 (recent 2018-2024) ===")
    print(df_recent.nlargest(30, "cagr_xirr").to_string())


if __name__ == "__main__":
    main()
