#!/usr/bin/env python3
"""Screening-layer portfolio math over the precomputed structures table.

IMPORTANT (honesty): returns here are entry→expiry P&L bucketed at ENTRY date
and divided by committed margin. Intermediate marks are absent, so vol is
understated for overlapping positions. Use ONLY for ranking/screening.
Final numbers must come from engine.Backtester with M/W/F marks.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_structures() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    df["ret"] = df.pnl / df.margin
    df["date"] = pd.to_datetime(df.date)
    return df


def sleeve_series(df: pd.DataFrame, mask, weight_col=None, dev_end="2024-12-31") -> pd.Series:
    """Equal-weight (or weighted) mean return per entry date for rows in mask."""
    g = df[mask]
    if weight_col is None:
        s = g.groupby("date").ret.mean()
    else:
        s = g.groupby("date").apply(
            lambda x: np.average(x.ret, weights=x[weight_col]), include_groups=False)
    return s.sort_index()


def screen_sharpe(s: pd.Series) -> dict:
    """Annualized Sharpe of entry-bucketed structure returns.

    Each entry's capital is committed ~1 month; entries overlap 3x/week. A
    portfolio that splits capital across all concurrent cycles earns the MEAN
    monthly-cycle return per cycle. Approximate annualization: aggregate by
    ISO week (mean), then x sqrt(12) on monthly non-overlap basis — i.e.
    treat each week's cohort as 1/4.33 of a month's capital.
    """
    if len(s) < 30:
        return {"sharpe_scr": np.nan, "n": len(s)}
    w = s.resample("W-FRI").mean().dropna()
    mu, sd = w.mean(), w.std()
    if sd == 0 or not np.isfinite(sd):
        return {"sharpe_scr": np.nan, "n": len(w)}
    # weekly cohorts of monthly trades: successive cohorts overlap; the
    # independent unit is ~1 month → sqrt(12). This is approximate; the
    # event-loop engine gives the real number.
    sh = mu / sd * math.sqrt(12)
    return {"sharpe_scr": float(sh), "mean_m": float(mu), "sd_m": float(sd),
            "n_weeks": len(w), "hit": float((w > 0).mean())}


def dev_test_split(s: pd.Series, dev_end="2024-12-31"):
    return s[s.index <= dev_end], s[s.index > dev_end]
