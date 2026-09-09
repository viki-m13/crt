#!/usr/bin/env python3
"""The fast search in validate.py must agree with api/_analog.py.

validate.py rewrites the matcher in numpy because the pure-Python version
cannot sweep 400 tickers of history per test point. That optimisation is only
legitimate if it returns the SAME analogs — otherwise the accuracy figures
describe code that no user ever runs, which is a subtler version of the
look-ahead problem.

This picks real tickers and real dates out of the panel and requires both
implementations to select the same windows.

    python research/analog/check_equivalence.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, HERE)

import _analog as A            # noqa: E402
from validate import build_library, pick, query_shape, PANEL  # noqa: E402

LOOKBACK, HORIZON, TOP_K = A.LOOKBACK, A.HORIZON, A.TOP_K
FAIL, N = [], [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok    " if cond else "  FAIL  ") + label + " " + detail)
    if not cond:
        FAIL.append(label)


def main():
    px = pd.read_parquet(PANEL)
    cov = px.notna().sum().sort_values(ascending=False)
    # a handful of tickers with full history; stride 1 so both see the same
    # candidate set (validate.py strides for speed, which is a sampling
    # choice, not an algorithmic one — here we remove it to compare like
    # for like)
    cols = sorted(cov.index[:6])
    px = px[cols].dropna()
    print(f"panel {px.shape[0]} bars x {px.shape[1]} tickers "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")

    shapes, end_ix, col_ix, fwd = build_library(px, LOOKBACK, stride=1)
    logp = np.log(px.to_numpy(dtype=np.float64))
    dates = [str(d.date()) for d in px.index]
    library = {c: (dates, px[c].tolist()) for c in px.columns}

    print("=" * 78)
    print("FAST NUMPY SEARCH vs REFERENCE api/_analog.py")
    print("=" * 78)
    rng = np.random.default_rng(3)
    tested = 0
    for t in rng.choice(range(LOOKBACK + HORIZON + 400, len(px) - HORIZON),
                        size=6, replace=False):
        t = int(t)
        for c in (0, 3):
            sym = px.columns[c]
            q = query_shape(logp[:, c], t, LOOKBACK)
            if q is None:
                continue
            mask = (end_ix + HORIZON) < t
            sel = pick(shapes, mask, q, TOP_K, end_ix, col_ix)
            fast = sorted((px.columns[col_ix[i]], dates[end_ix[i]])
                          for i in sel)

            fc = A.find_analogs(px[sym].tolist(), library, as_of_index=t,
                                lookback=LOOKBACK, horizon=HORIZON,
                                top_k=TOP_K, symbol=sym, dates=dates)
            ref = sorted((a.symbol, a.end) for a in fc.analogs)
            check(fast == ref, f"{sym} @ {dates[t]}",
                  "same analogs" if fast == ref
                  else f"\n      fast={fast}\n      ref ={ref}")
            tested += 1

    print(f"\n{N[0] - len(FAIL)}/{N[0]} comparisons matched "
          f"({tested} ticker-dates)")
    if FAIL:
        return 1
    print("The validated code and the shipped code are the same algorithm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
