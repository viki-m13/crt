#!/usr/bin/env python3
"""Validation of the analog matcher (api/_analog.py).

Every number this tool ever reports depends on the matcher being unable to
see the future. These checks exist because each of the failures below would
produce a *better looking* result, which is exactly the kind of bug that
survives casual review:

  - a match whose forward window overlaps the period being forecast
  - five matches that are really one window counted five times
  - matching on price level so only similarly-priced names ever come back
  - an agreement score computed off duplicated analogs

    python tests/test_analog.py
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import _analog as A  # noqa: E402

FAIL: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    if cond:
        print(f"  ok    {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label} {detail}")


def series(n, seed=0, drift=0.0004, vol=0.012):
    r = random.Random(seed)
    px, out = 100.0, []
    for _ in range(n):
        px *= math.exp(r.gauss(drift, vol))
        out.append(px)
    return out


def dates(n, start=0):
    """Sequential ISO dates, one per bar. Good enough to order by."""
    out = []
    y, m, d = 2000, 1, 1
    for i in range(start + n):
        d += 1
        if d > 28:
            d, m = 1, m + 1
        if m > 12:
            m, y = 1, y + 1
        out.append(f"{y:04d}-{m:02d}-{d:02d}")
    return out[start:]


def main():
    print("=" * 78)
    print("ANALOG MATCHER")
    print("=" * 78)

    L, H = 40, 20
    px = series(1200, seed=7)
    dt = dates(1200)
    lib = {"TEST": (dt, px)}

    print("\n--- 1. it finds something and reports it honestly ---")
    fc = A.find_analogs(px, lib, symbol="TEST", lookback=L, horizon=H, top_k=5)
    check(fc.ok, "returns analogs on a normal series",
          f"{len(fc.analogs)} found from {fc.searched:,} windows")
    check(len(fc.query_path) == L + 1, "query path is lookback+1 bars",
          f"{len(fc.query_path)}")
    check(abs(fc.query_path[0] - 100.0) < 1e-6, "query path is rebased to 100")
    check(all(len(a.forward) == H + 1 for a in fc.analogs),
          "every forward path is horizon+1 bars")
    check(all(abs(a.forward[0] - 100.0) < 1e-6 for a in fc.analogs),
          "forward paths start at 100 so they are comparable")

    print("\n--- 2. NO LOOK-AHEAD (the check that matters) ---")
    # as_of is bar 600; nothing at or past bar 600 may inform the answer
    as_of = 600
    fc2 = A.find_analogs(px, lib, symbol="TEST", as_of_index=as_of,
                         lookback=L, horizon=H, top_k=5)
    worst = max((dt.index(a.end) + H for a in fc2.analogs), default=-1)
    check(worst < as_of,
          "no analog's forward window reaches as_of",
          f"latest bar used = {worst}, as_of = {as_of}")
    check(all(dt.index(a.end) < as_of for a in fc2.analogs),
          "no analog even ENDS at or after as_of")

    # the decisive one: change the future and the answer must not move
    px_alt = list(px)
    for i in range(as_of + 1, len(px_alt)):
        px_alt[i] = px_alt[i] * 3.0
    fc3 = A.find_analogs(px_alt, {"TEST": (dt, px_alt)}, symbol="TEST",
                         as_of_index=as_of, lookback=L, horizon=H, top_k=5)
    same = [(a.end, a.distance) for a in fc2.analogs] == \
           [(a.end, a.distance) for a in fc3.analogs]
    check(same, "tripling every future bar changes nothing about the answer")

    print("\n--- 3. matches are distinct events, not one window five times ---")
    ends = [dt.index(a.end) for a in fc.analogs]
    gaps = [abs(x - y) for i, x in enumerate(ends) for y in ends[i + 1:]]
    check(all(g >= A.MIN_GAP for g in gaps) or len(ends) < 2,
          "every pair of matches is at least MIN_GAP apart",
          f"closest pair {min(gaps) if gaps else '-'}")

    # and without dedupe they WOULD collide, so the guard is doing work
    raw = A.find_analogs(px, lib, symbol="TEST", lookback=L, horizon=H,
                         top_k=5, min_gap=1)
    raw_ends = sorted(dt.index(a.end) for a in raw.analogs)
    raw_gaps = [b - a for a, b in zip(raw_ends, raw_ends[1:])]
    check(any(g < A.MIN_GAP for g in raw_gaps) if raw_gaps else True,
          "with min_gap=1 the matches DO cluster, so dedupe is load-bearing",
          f"gaps {raw_gaps}")

    print("\n--- 4. it matches shape, not price level ---")
    # identical shape, 50x the price, different symbol
    scaled = [p * 50.0 for p in px[:400]]
    lib2 = {"TEST": (dt, px), "RICH": (dates(400), scaled)}
    fc4 = A.find_analogs(px[:200], lib2, symbol="TEST", lookback=L,
                         horizon=H, top_k=3)
    check(any(a.symbol == "RICH" for a in fc4.analogs),
          "a 50x-priced series with the same shape is reachable")

    # a vol-scaled copy must match essentially perfectly
    calm = [100.0]
    for r in A._log_returns(px[:L + 1]):
        calm.append(calm[-1] * math.exp(r * 0.25))
    d = A._distance(A._shape(A._log_returns(px[:L + 1])),
                    A._shape(A._log_returns(calm)))
    check(d < 1e-6, "quarter-volatility copy is a near-exact shape match",
          f"distance {d:.2e}")

    print("\n--- 5. the distribution is a distribution ---")
    check(len(fc.median_path) == len(fc.low_path) == len(fc.high_path),
          "median/low/high paths are the same length")
    check(all(lo <= m <= hi for lo, m, hi in
              zip(fc.low_path, fc.median_path, fc.high_path)),
          "median sits inside the low/high envelope at every step")
    check(0.0 <= fc.agreement <= 1.0, "agreement is a share", f"{fc.agreement}")
    ups = sum(1 for a in fc.analogs if a.forward_return > 0)
    expect = max(ups, len(fc.analogs) - ups) / len(fc.analogs)
    check(abs(fc.agreement - expect) < 1e-9,
          "agreement counts the majority direction, not the up direction")

    print("\n--- 6. it refuses rather than guesses ---")
    short = A.find_analogs(px[:10], lib, symbol="TEST", lookback=L, horizon=H)
    check(not short.ok and "needs" in short.reason,
          "too little history is a refusal with a reason", short.reason)
    flat = [100.0] * 500
    ff = A.find_analogs(flat, {"F": (dates(500), flat)}, symbol="F",
                        lookback=L, horizon=H)
    check(not ff.ok, "a flat line has no shape and returns nothing",
          ff.reason)
    check(not A.find_analogs(px, {}, symbol="TEST", lookback=L,
                             horizon=H).ok,
          "an empty library is an empty answer, not a crash")

    print("\n--- 7. the shape transform ---")
    check(A._shape([0.0] * 10) is None, "zero-variance window has no shape")
    check(A._shape([0.01, float('nan'), 0.02]) is None,
          "a gap in the window disqualifies it rather than being filled")
    check(A._shape([0.01] * 20) is None,
          "a CONSTANT-return window is also shapeless — a smooth exponential "
          "has no pattern to match")
    s = A._shape([0.01, -0.02, 0.03, -0.01, 0.02])
    check(s is not None and len(s) == 5 and s[0] != 0,
          "a window with real variation produces a path")
    check(abs(A._corr([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9,
          "correlation of a scaled copy is 1")
    check(abs(A._corr([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9,
          "and of a reversed copy is -1")

    print("\n--- 8. serialisation ---")
    d = A.to_dict(fc)
    check(d["ok"] and d["symbol"] == "TEST", "round-trips the basics")
    check(len(d["analogs"]) == len(fc.analogs), "keeps every analog")
    import json
    check(len(json.dumps(d)) > 100, "is JSON-serialisable")

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
