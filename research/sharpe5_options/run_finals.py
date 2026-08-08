#!/usr/bin/env python3
"""Final gauntlet driver: run every surviving sleeve in the event-loop engine
on DEV, build the frozen-weight ensemble, then touch the holdout ONCE.

Usage:
  python3 run_finals.py dev      # dev-only sleeve runs + ensemble weights
  python3 run_finals.py full     # same sleeves over the full period (holdout)
"""
from __future__ import annotations

import os
import sys

import pandas as pd

import engine as E
import ensemble as X
import strategies as S

HERE = os.path.dirname(os.path.abspath(__file__))
SLEEVES = ["A_ps", "A_ps_2_7", "A_ps_6_12", "A_ps_mom", "A_ps_spydia",
           "A_ss", "B_str25", "B_str25_hi", "C_short", "C_short_ivr",
           "C_short_wings"]


def main(mode="dev"):
    dev_only = mode == "dev"
    n_trials = 80  # running trial count from RESEARCH_LOG
    for name in SLEEVES:
        try:
            S.run_sleeve(name, dev_only=dev_only, n_trials=n_trials)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    X.combine(label=f"ensemble_{mode}", n_trials=n_trials)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
