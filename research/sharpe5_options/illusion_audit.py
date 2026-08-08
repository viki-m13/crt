#!/usr/bin/env python3
"""Illusion audit: how much fake Sharpe does each common shortcut buy?

Takes one real sleeve and scores it under progressively dishonest conventions.
This is the control experiment for the whole project: it shows what a Sharpe-5
options backtest actually looks like from the inside, and which specific
accounting choice manufactured it.

Conventions audited (each vs the honest baseline):
  H0 honest      : worst-side fills, marked equity, margin capital, cash drag
  I1 mid fills   : enter/exit at mid instead of bid/ask
  I2 no marks    : entry-bucketed P&L (no intermediate marks) = screening layer
  I3 per-trade   : Sharpe of the trade-return distribution x sqrt(trades/yr)
  I4 premium base: returns divided by premium collected, not margin
  I5 active-only : drop flat weeks (Sharpe "when in a position")
  I6 no-crash    : drop the worst 1% of weeks ("outlier removal")
  I7 all combined
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))


def sharpe_of(r: pd.Series, ppy: float = 52.0) -> float:
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(ppy))


def audit(eq: pd.Series, trades: pd.DataFrame, label: str,
          eq_mid: pd.Series | None = None) -> pd.DataFrame:
    """eq: honest equity curve. trades: df with pnl, margin, credit, date."""
    rows = []
    w = eq.resample("W-FRI").last().dropna()
    r = w.pct_change().dropna()
    rf = pd.Series([E.rf_at(t) for t in r.index], index=r.index) / 52.0
    rows.append(("H0 honest (worst-side, marked, margin, cash drag)",
                 sharpe_of(r - rf)))

    if eq_mid is not None:
        wm = eq_mid.resample("W-FRI").last().dropna()
        rm = wm.pct_change().dropna()
        rfm = pd.Series([E.rf_at(t) for t in rm.index], index=rm.index) / 52.0
        rows.append(("I1 mid fills instead of bid/ask", sharpe_of(rm - rfm)))

    t = trades.copy()
    t["date"] = pd.to_datetime(t.date)
    t["ret_margin"] = t.pnl / t.margin

    # I2: entry-bucketed, no intermediate marks (the screening layer)
    s2 = t.groupby("date").ret_margin.mean()
    w2 = s2.resample("W-FRI").mean().dropna()
    rows.append(("I2 no intermediate marks (entry-bucketed)",
                 sharpe_of(w2, ppy=12.0)))

    # I3: per-trade Sharpe annualized by trade count
    tpy = len(t) / max((t.date.max() - t.date.min()).days / 365.25, 1e-9)
    rows.append((f"I3 per-trade Sharpe x sqrt({tpy:.0f} trades/yr)",
                 sharpe_of(t.ret_margin, ppy=tpy)))

    # I4: returns on premium collected rather than capital at risk
    if "credit" in t.columns:
        prem = t.credit.abs().clip(lower=1.0)
        s4 = (t.pnl / prem).groupby(t.date).mean()
        w4 = s4.resample("W-FRI").mean().dropna()
        rows.append(("I4 return on premium, not margin", sharpe_of(w4, ppy=12.0)))

    # I5: drop flat weeks (Sharpe "while active")
    active = r[r.abs() > 1e-9]
    rows.append((f"I5 active weeks only (n={len(active)}/{len(r)})",
                 sharpe_of(active - rf.reindex(active.index))))

    # I6: winsorize away the worst 1% of weeks
    cut = r.quantile(0.01)
    r6 = r[r > cut]
    rows.append((f"I6 drop worst 1% of weeks (n={len(r)-len(r6)} removed)",
                 sharpe_of(r6 - rf.reindex(r6.index))))

    # I7: stacked — mid fills + no marks + premium base + drop worst 1%
    base = t.copy()
    if "pnl_mid" in base.columns:
        base["r7"] = base.pnl_mid / base.credit.abs().clip(lower=1.0)
    else:
        base["r7"] = base.pnl / base.credit.abs().clip(lower=1.0)
    s7 = base.groupby("date").r7.mean()
    w7 = s7.resample("W-FRI").mean().dropna()
    w7 = w7[w7 > w7.quantile(0.01)]
    rows.append(("I7 stacked (mid + no marks + premium base + trim)",
                 sharpe_of(w7, ppy=12.0)))

    df = pd.DataFrame(rows, columns=["convention", "sharpe"])
    df.insert(0, "sleeve", label)
    print(f"\n=== ILLUSION AUDIT: {label} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    return df
