#!/usr/bin/env python3
"""How often is a stock simply higher N days later? The number 90% must beat.

Before building anything, this establishes the yardstick. A method that calls
"higher in 30 days" correctly 90% of the time is extraordinary if the base
rate is 55% and worthless if the base rate is 89%. Almost every published
"90% win rate" is the second thing.

Three questions:

  1. The unconditional base rate by horizon, on a universe that includes the
     4,725 tickers that stopped trading.
  2. How much of any high number is just the horizon. P(up) rises with N for
     the trivial reason that drift accumulates and the choice of N is ours.
  3. What the cheapest conditioning buys — low volatility, an uptrend, a
     market above its own 200-day average. If simple filters already reach
     85%, a model has to beat that, not the unconditional rate.

DELISTING IS REPORTED BOTH WAYS. When a stock stops trading inside the
horizon its outcome is unknown: acquisitions and bankruptcies both just end.
Dropping those observations is the optimistic choice and is what almost
everyone does silently. Scoring them as "not higher" is the pessimistic one.
Both are printed, and the gap between them is the size of the problem.

    python research/uptrend/baserates.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load_panel  # noqa: E402

HORIZONS = [21, 42, 63, 126, 252, 504]
SAMPLE_EVERY = 5           # rows sampled, to keep the sweep quick


def summarise(px: pd.DataFrame):
    arr = px.to_numpy(dtype=np.float32)
    n, k = arr.shape
    ok = np.isfinite(arr)
    # last traded bar per ticker: where a name's data ends
    last = np.where(ok.any(0), n - 1 - ok[::-1].argmax(0), -1)
    rows = np.arange(n)[:, None]

    print("=" * 78)
    print("1. UNCONDITIONAL: IS THE STOCK HIGHER N TRADING DAYS LATER?")
    print("=" * 78)
    print(f"  {'horizon':<10}{'observations':>14}{'strict':>10}"
          f"{'delisting=loss':>16}{'gap':>8}")
    base = {}
    for H in HORIZONS:
        fut = np.full_like(arr, np.nan)
        fut[:n - H] = arr[H:]
        live = ok & (rows + H <= last[None, :])       # survived the window
        gone = ok & (rows + H > last[None, :]) & (last[None, :] >= 0)
        live[::SAMPLE_EVERY] = live[::SAMPLE_EVERY]   # (sampling below)
        sel = np.zeros_like(live)
        sel[::SAMPLE_EVERY] = True
        L, G = live & sel, gone & sel
        up = (fut > arr) & L
        strict = up.sum() / max(L.sum(), 1)
        withdead = up.sum() / max(L.sum() + G.sum(), 1)
        base[H] = strict
        print(f"  {H:<10}{L.sum():>14,}{100*strict:>9.1f}%"
              f"{100*withdead:>15.1f}%{100*(strict-withdead):>7.1f}")
    print("\n  'strict' drops the observations where the stock stopped")
    print("  trading mid-horizon. That is the number everyone quotes, and it")
    print("  is the optimistic one — the gap column is what it costs.")
    return arr, ok, last, rows, base


def conditioned(arr, ok, last, rows, base):
    n, k = arr.shape
    logp = np.log(np.where(arr > 0, arr, np.nan))
    ret1 = np.diff(logp, axis=0, prepend=np.nan)

    # --- features, all strictly backward-looking -------------------------
    def roll_mean(a, w):
        c = pd.DataFrame(a).rolling(w, min_periods=w).mean().to_numpy()
        return c

    def roll_std(a, w):
        return pd.DataFrame(a).rolling(w, min_periods=w).std().to_numpy()

    sma200 = roll_mean(arr, 200)
    vol60 = roll_std(ret1, 60) * np.sqrt(252)
    above = arr > sma200
    sma_rising = np.full_like(arr, np.nan, dtype=np.float32)
    sma_rising[21:] = sma200[21:] - sma200[:-21]
    rising = sma_rising > 0

    # market regime: the equal-weight universe above its own 200-day average
    mkt = np.nanmean(arr / np.where(sma200 > 0, sma200, np.nan), axis=1)
    mkt_up = (mkt > 1.0)[:, None]

    print()
    print("=" * 78)
    print("2. WHAT THE CHEAPEST CONDITIONING BUYS")
    print("=" * 78)
    tests = [
        ("no filter", np.ones_like(ok)),
        ("above rising 200d SMA", above & rising),
        ("...and vol < 30%", above & rising & (vol60 < 0.30)),
        ("...and market above its SMA", above & rising & (vol60 < 0.30) & mkt_up),
        ("...and vol < 20%", above & rising & (vol60 < 0.20) & mkt_up),
    ]
    for H in (21, 63, 252, 504):
        fut = np.full_like(arr, np.nan)
        fut[:n - H] = arr[H:]
        live = ok & (rows + H <= last[None, :])
        sel = np.zeros_like(live)
        sel[::SAMPLE_EVERY] = True
        print(f"\n  horizon {H} days   (base rate {100*base[H]:.1f}%)")
        print(f"  {'filter':<32}{'n':>12}{'P(higher)':>12}{'lift':>8}")
        for name, f in tests:
            m = live & sel & np.nan_to_num(f, nan=False).astype(bool)
            if m.sum() < 2000:
                print(f"  {name:<32}{m.sum():>12,}   too few")
                continue
            p = ((fut > arr) & m).sum() / m.sum()
            print(f"  {name:<32}{m.sum():>12,}{100*p:>11.1f}%"
                  f"{100*(p-base[H]):>+7.1f}")
    return above, rising, vol60, mkt_up


def main():
    px, _ = load_panel()
    print(f"panel {px.shape[0]:,} bars x {px.shape[1]:,} tickers  "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")
    arr, ok, last, rows, base = summarise(px)
    conditioned(arr, ok, last, rows, base)

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    print(f"  A stock is higher 30 trading days later {100*base[21]:.0f}% of")
    print(f"  the time and higher a year later {100*base[252]:.0f}% of the")
    print("  time, before any skill at all. Any '90% accurate' claim has to")
    print("  be measured against the row of this table with the same horizon")
    print("  and the same universe, or it is measuring the calendar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
