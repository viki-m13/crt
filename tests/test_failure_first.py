#!/usr/bin/env python3
"""Leakage and construction checks for the failure-channel gate.

The whole premise is that each channel is computable on the day, from that
day's past. If any channel peeks, the conjunction inherits it and the result
is meaningless — and a peeking channel makes the answer BETTER, so it must be
checked mechanically rather than read.

    python tests/test_failure_first.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "failure_first"))

import channels as C  # noqa: E402

FAIL: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok    " if cond else "  FAIL  ") + label + " " + str(detail))
    if not cond:
        FAIL.append(label)


def panel(n=1600, k=60, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        {f"S{j}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.017, n)))
         for j in range(k)}, index=idx)


def main():
    print("=" * 78)
    print("FAILURE-CHANNEL GATE")
    print("=" * 78)
    px = panel()
    vol = pd.DataFrame(
        np.random.default_rng(1).lognormal(12, 1.2, px.shape),
        index=px.index, columns=px.columns)

    print("\n--- 1. every channel is built and bounded ---")
    S = C.build(px, vol)
    check(set(S) == set(C.CHANNELS), "all seven channels present",
          f"{len(S)}")
    for c in C.CHANNELS:
        v = S[c][np.isfinite(S[c])]
        check(v.size > 0 and v.min() >= -1e-6 and v.max() <= 1 + 1e-6,
              f"{c} is a risk score in [0,1]",
              f"[{v.min():.3f}, {v.max():.3f}]")

    print("\n--- 2. NO LOOK-AHEAD (the check that matters) ---")
    cut = 1000
    tampered = px.copy()
    tampered.iloc[cut:] *= 5.0
    S2 = C.build(tampered, vol)
    worst, where = 0.0, ""
    for c in C.CHANNELS:
        a, b = S[c][:cut], S2[c][:cut]
        m = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
        if d > worst:
            worst, where = d, c
    check(worst < 1e-6,
          "quintupling every bar after 1000 changes no score before it",
          f"worst drift {worst:.2e} ({where})")

    # the market channel is the one with a time-series threshold, so it gets
    # its own check: an expanding percentile must never see its own future
    tam2 = px.copy()
    tam2.iloc[cut:] *= 0.2
    S3 = C.build(tam2, vol)
    a, b = S["market"][:cut, 0], S3["market"][:cut, 0]
    m = np.isfinite(a) & np.isfinite(b)
    check(np.max(np.abs(a[m] - b[m])) < 1e-6,
          "the market channel's expanding percentile is past-only")

    print("\n--- 3. the gate is a conjunction ---")
    g = C.gate(S, 0.5)
    manual = np.ones(g.shape, dtype=bool)
    for c in C.CHANNELS:
        manual &= np.isfinite(S[c]) & (S[c] <= 0.5)
    check(np.array_equal(g, manual), "gate is exactly AND over all channels")
    check(C.gate(S, 1.0).sum() >= C.gate(S, 0.5).sum() >= C.gate(S, 0.1).sum(),
          "tightening the gate never adds signals")
    check(C.gate(S, 0.0).sum() <= C.gate(S, 0.5).sum(),
          "a zero gate is the tightest possible")

    print("\n--- 4. a NaN channel blocks the gate, it does not pass it ---")
    S4 = {c: v.copy() for c, v in S.items()}
    S4["tail"][:] = np.nan
    check(C.gate(S4, 1.0).sum() == 0,
          "an all-NaN channel means no signal, never a free pass")

    print("\n--- 5. worst_channel is the conjunction as a score ---")
    w = C.worst_channel(S)
    for q in (0.3, 0.5, 0.7):
        a = (np.isfinite(w) & (w <= q))
        b = C.gate(S, q)
        check(np.array_equal(a, b),
              f"thresholding worst_channel at {q} == gate at {q}")

    print("\n--- 6. ranks are within-date, not across the whole panel ---")
    r = C._rank(np.array([[1.0, 2, 3, 4], [40, 30, 20, 10]] * 15))
    check(np.isnan(r).all(), "fewer than 20 names in a row gives no rank")
    big = np.tile(np.arange(30.0), (3, 1))
    rb = C._rank(big)
    check(abs(rb[0, 0]) < 1e-6 and abs(rb[0, -1] - 1.0) < 1e-6,
          "within a row the min ranks 0 and the max ranks 1")
    shifted = big.copy()
    shifted[1] += 1000.0
    rs = C._rank(shifted)
    check(np.allclose(rs[0], rs[1]),
          "adding 1000 to another DATE does not change this date's ranks")

    print("\n" + "=" * 78)
    print(f"{N[0] - len(FAIL)}/{N[0]} checks passed")
    if FAIL:
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
