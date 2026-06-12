"""Build the causal market-breadth series for the CreditFloor universe.

For every trading date: the fraction of universe tickers (with at least
200 sessions of history at that date) whose close is at or above their
own 200-day SMA. This is a fully causal, panel-derived market-regime
indicator — no external index data needed, no look-ahead (the value for
date t uses only closes <= t).

Output: results/breadth.json  {"dates": [...], "pct_above_sma200": [...],
"n_tickers": [...]}.

Honors CS_DATA_DIR (full-history cache) and CS_LIMIT.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import list_tickers, load_series, _rolling_mean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results",
                   os.environ.get("CS_BREADTH_OUT", "breadth.json"))


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    above: dict[np.datetime64, int] = {}
    total: dict[np.datetime64, int] = {}
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None:
            continue
        sma = _rolling_mean(ts.close, 200)
        ok = np.isfinite(sma)
        up = ok & (ts.close >= sma)
        for d in ts.dates[ok]:
            total[d] = total.get(d, 0) + 1
        for d in ts.dates[up]:
            above[d] = above.get(d, 0) + 1
        if i % 200 == 0:
            print(f"  {i}/{len(tickers)} elapsed={time.time()-t0:.0f}s", flush=True)
    dates = sorted(total.keys())
    blob = {
        "dates": [str(d) for d in dates],
        "pct_above_sma200": [above.get(d, 0) / total[d] for d in dates],
        "n_tickers": [total[d] for d in dates],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(blob, fh)
    print(f"Wrote {OUT}: {len(dates)} dates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
