#!/usr/bin/env python3
"""Can this panel detect an effect that is known to exist?

A negative result is only worth believing if the machinery could have found a
positive one. Before concluding that analog matching predicts nothing, this
runs two textbook cross-sectional effects through the same price panel:

  12-1 MOMENTUM — buy what went up over the last year skipping the last
      month. Robust, published since Jegadeesh & Titman (1993), and expected
      to show a small positive information coefficient.
  1-MONTH REVERSAL — expected to be weak-to-absent in a large-cap universe
      after the 1990s, so it doubles as a check that the harness is not
      simply printing positive numbers.

If momentum shows up and analog matching does not, the difference is in the
signal, not in the test.

    python research/analog/sanity_known_effects.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL = os.path.join(ROOT, "experiments", "monthly_dca", "cache",
                     "prices_extended.parquet")


def ic(signal: pd.DataFrame, fwd: pd.DataFrame, skip: int) -> pd.Series:
    """Rank correlation of the signal with next month's return, per month."""
    out = []
    for d in signal.index[skip:-1]:
        a, b = signal.loc[d], fwd.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() > 50:
            out.append(a[ok].rank().corr(b[ok].rank()))
    return pd.Series(out).dropna()


def report(name: str, s: pd.Series):
    t = s.mean() / (s.std(ddof=1) / len(s) ** 0.5)
    print(f"  {name:<22}{s.mean():>+9.4f}   t={t:>+6.2f}   {len(s)} months")
    return t


def main():
    px = pd.read_parquet(PANEL)
    m = px.resample("ME").last()
    fwd = m.shift(-1) / m - 1
    print(f"panel {px.shape[0]:,} bars x {px.shape[1]:,} tickers  "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")

    print("=" * 78)
    print("KNOWN EFFECTS, SAME PANEL, SAME HARNESS")
    print("=" * 78)
    print(f"  {'effect':<22}{'IC':>9}{'':>12}")
    tm = report("12-1 momentum", ic(m.shift(1) / m.shift(12) - 1, fwd, 12))
    report("1-month reversal", ic(m.shift(1) / m.shift(2) - 1, fwd, 3))

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    if tm > 2:
        print("  Momentum is detectable here at t>2, as it should be. The")
        print("  harness can find a real cross-sectional signal.")
        print("  Analog matching's IC on the same panel is -0.0071, t=-0.43")
        print("  (research/analog/validate.py). That difference is the signal,")
        print("  not the test.")
    else:
        print("  Momentum did NOT show up. Treat every other result from this")
        print("  panel as unproven until that is explained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
