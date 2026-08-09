#!/usr/bin/env python3
"""Run the illusion audit on the best sleeve: how much fake Sharpe does each
common backtesting shortcut buy, measured on identical underlying trades?
"""
from __future__ import annotations

import os

import pandas as pd

import illusion_audit as IA

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    eq = pd.read_parquet(os.path.join(HERE, "results", "eq_A_ps.parquet"))["equity"]
    st = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    tr = st[(st.structure == "credit_putspread") & (st.act_symbol == "SPY")].copy()
    tr["date"] = pd.to_datetime(tr.date)
    print(f"trades: {len(tr)}  dates: {tr.date.nunique()}  "
          f"{tr.date.min().date()} -> {tr.date.max().date()}")
    out = IA.audit(eq, tr, "SPY put credit spread (best sleeve)")
    out.to_csv(os.path.join(HERE, "results_illusion_audit.csv"), index=False)


if __name__ == "__main__":
    main()
