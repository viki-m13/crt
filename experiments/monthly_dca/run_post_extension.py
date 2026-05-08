"""End-to-end pipeline run after forward_returns.parquet is finalised:
  1. run_extended (sweep 16 strategies × 5 K × 13 exits on 2002-2024)
  2. pick_robust (find strategy with best worst-year edge vs SPY)
  3. survivorship (re-run with extended panel)
  4. walk_forward_v2 (more splits with 23 years of data)
  5. save_winning_picks (with newly-chosen recommended strategy)
  6. build_webapp_json

This is a single entry point so we don't have to babysit each step.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = Path(__file__).resolve().parent / "cache"


def run(cmd: list[str], description: str) -> None:
    print(f"\n{'='*70}\n=== {description} ===\n{'='*70}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, check=False)
    dt = time.time() - t0
    print(f"\n=== {description}: exit={r.returncode}  elapsed={dt:.1f}s ===")
    if r.returncode != 0:
        print(f"  WARNING: non-zero exit code")


def main() -> None:
    if not (CACHE / "fwd_returns.parquet").exists():
        print("forward returns not yet finalised; run forward_returns.py first")
        return

    py = sys.executable

    # Run sweep on extended history
    run([py, "experiments/monthly_dca/run_extended.py"],
        "Step 1: Strategy sweep on extended 2002-2024 history")

    # Pick robust strategy
    run([py, "experiments/monthly_dca/pick_robust.py"],
        "Step 2: Find robust strategy (best worst-year edge)")

    # Read recommendation
    rec_path = CACHE / "recommended_strategy.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        print(f"\nRECOMMENDED: {rec.get('strategy')}::{rec.get('top_k')}::{rec.get('exit')}")
        print(f"  CAGR={rec.get('cagr_dca_portfolio'):.3f}  edge={rec.get('edge_vs_spy_dca'):.3f}  "
              f"min_year={rec.get('min_year_edge'):.3f}")

    # Survivorship analysis on extended data
    run([py, "experiments/monthly_dca/survivorship.py"],
        "Step 3: Survivorship analysis (re-run on extended panel)")

    # Walk-forward over extended period
    run([py, "experiments/monthly_dca/walk_forward_v2.py"],
        "Step 4: Walk-forward across 8 splits on extended history")

    # Save winning picks
    run([py, "experiments/monthly_dca/save_winning_picks.py"],
        "Step 5: Save full pick logs")

    # Build webapp JSON
    run([py, "experiments/monthly_dca/build_webapp_json.py"],
        "Step 6: Build webapp data.json")

    print("\n" + "=" * 70)
    print("All steps complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
