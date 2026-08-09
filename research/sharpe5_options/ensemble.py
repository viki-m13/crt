#!/usr/bin/env python3
"""Combine sleeve equity curves into one portfolio, honestly.

- Each sleeve runs on its own capital in the event-loop engine (equity curve
  saved under results/eq_<name>.parquet).
- Weights: inverse of DEV-period weekly vol (no return-based optimization,
  no holdout contact). Weights are then FROZEN and applied to the whole
  period; holdout numbers reported separately.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import engine as E
import verify_engine as V

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DEV_END = pd.Timestamp("2024-12-31")


def save_eq(eq: pd.Series, name: str):
    os.makedirs(RES, exist_ok=True)
    eq.rename("equity").to_frame().to_parquet(os.path.join(RES, f"eq_{name}.parquet"))


def load_all() -> dict[str, pd.Series]:
    out = {}
    for p in glob.glob(os.path.join(RES, "eq_*.parquet")):
        name = os.path.basename(p)[3:-8]
        out[name] = pd.read_parquet(p)["equity"]
    return out


def combine(names: list[str] | None = None, label: str = "ensemble",
            n_trials: int = 1):
    eqs = load_all()
    if names:
        eqs = {k: v for k, v in eqs.items() if k in names}
    # Drop sleeves that went bust. A sizing rule that divides capital by a
    # narrow defined-risk width can lever a position far enough to take equity
    # through zero; such a curve has no meaningful return series and must not
    # be blended into a portfolio, however small its inverse-vol weight.
    broke = [k for k, v in eqs.items() if (v <= 0).any()]
    for k in broke:
        print(f"EXCLUDED (equity went non-positive): {k}")
        eqs.pop(k)
    rets = {}
    for k, eq in eqs.items():
        w = eq.resample("W-FRI").last().dropna()
        rets[k] = w.pct_change().dropna()
    R = pd.DataFrame(rets).dropna()
    dev = R[R.index <= DEV_END]
    vols = dev.std()
    wts = (1.0 / vols) / (1.0 / vols).sum()
    print("weights:", wts.round(3).to_dict())
    port = (R * wts).sum(axis=1)
    eq_port = (1 + port).cumprod() * 1e6
    eq_port.index.name = None
    print("== DEV ==")
    V.report(eq_port[eq_port.index <= DEV_END], f"{label}_dev", n_trials)
    hold = eq_port[eq_port.index > DEV_END]
    if len(hold) > 25:
        print("== HOLDOUT (2025+) ==")
        V.report(hold, f"{label}_holdout", n_trials)
    print("== FULL ==")
    out = V.report(eq_port, f"{label}_full", n_trials)
    print("sleeve corr (weekly, full):")
    print(R.corr().round(2).to_string())
    return eq_port, wts, out
