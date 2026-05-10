"""Build v5 webapp data.json for the PIT-S&P-500 strategy with Chronos filter.

Strategy: ml_3plus6 (v3 baseline) + Chronos-bolt-tiny p70 confidence filter.
At each rebalance:
  1. Compute v3 ml_3plus6 score for each PIT S&P 500 member.
  2. Compute Chronos-bolt-tiny p70 (3m forecast) for each member from
     252-day daily price history.
  3. Cross-sectionally rank Chronos predictions within S&P 500.
  4. Filter to stocks with Chronos rank >= 0.4 (top 60%).
  5. Pick top-3 by ml_3plus6 score from filtered pool.
  6. Equal-weight, hold 6 months.
  7. Tight regime gate (cash on SPY 21d ≤ -8% etc).

Backtest:
  - Full window 2003-2025: 44.81% CAGR, +32.87pp edge over SPY
  - Walk-forward (10 splits): 45.86% mean test CAGR, 17.01% min, 10/10 beat SPY
  - vs deployed v3: +5.04pp full / +3.06pp WF mean / +2.52pp WF min / 10/10 vs 9/10
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.monthly_dca.v2.ml_strategy import EXCLUDE  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
WEBAPP_OUT = ROOT / "experiments" / "docs" / "monthly-dca"
WEBAPP_OUT.mkdir(parents=True, exist_ok=True)

STRATEGY_SPEC = {
    "scorer": "ml_3plus6",
    "scorer_description": "Mean of multi-horizon GBM 3m and 6m forward-rank predictions",
    "K_normal": 3,
    "K_recovery": 3,
    "K_bull": 3,
    "weighting": "equal-weight",
    "regime_gate": "tight",
    "regime_gate_rule": (
        "crash if SPY 21d <= -8% OR (SPY 6m <= -5% AND SPY 21d <= -3%); "
        "recovery if SPY below 200dma streak >= 40d AND SPY just back above 200dma AND SPY 21d > 0; "
        "bull if SPY 12m >= 10% AND above 200dma; else normal."
    ),
    "hold_months": 6,
    "cost_bps": 10,
    "universe": "PIT S&P 500 members at each rebalance month-end",
    "rebalance_rule": "Hold each basket for 6 months. Reform basket on month T if (T - last_rebalance) >= 6m or regime transitions to/from cash.",
    "chronos_filter": {
        "model": "amazon/chronos-bolt-tiny (HuggingFace)",
        "model_size": "9M params",
        "input": "trailing 252-day daily prices",
        "horizon": "64 trading days (~3 months)",
        "metric": "70th percentile of probabilistic forecast distribution",
        "filter_quantile": 0.4,
        "rule": (
            "At each rebalance, before picking top-K, restrict candidate pool to "
            "stocks where the Chronos p70 3m-forecast cross-sectional rank is >= "
            "0.4 (top 60% of S&P 500 members).  This eliminates the bottom 40% "
            "by Chronos confidence, then v3 picks top 3 by ml_3plus6 score."
        ),
    },
}

WINNER_NAME = "v5_pit_sp500_ml_3plus6_chronos_p70_filter_k3_h6"
