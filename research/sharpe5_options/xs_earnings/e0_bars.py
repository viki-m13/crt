#!/usr/bin/env python3
"""Fetch adjusted daily closes for the option-chain symbols that bonds/ lacks,
so the cross-section is not artificially thinned by a data gap.
Written to extra_bars/<SYM>.csv with the same (ts, close) shape as
bonds/data/intraday_daily so the loader is unchanged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "extra_bars")
SYMS = ["AMD", "BA", "C", "COP", "F", "GM", "IBM", "MMM", "MU", "NKE",
        "PYPL", "SBUX", "T", "WFC"]


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in SYMS:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{s}"
               f"?period1=1420070400&period2=1790000000&interval=1d&events=div%2Csplit")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            j = json.loads(r.read().decode())
        res = j["chart"]["result"][0]
        ts = pd.to_datetime(res["timestamp"], unit="s", utc=True)
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        d = pd.DataFrame({"ts": ts, "close": adj}).dropna()
        d["ts"] = d.ts.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        d = d.drop_duplicates("ts").sort_values("ts")
        d.to_csv(os.path.join(OUT, f"{s}.csv"), index=False)
        print(f"{s}: {len(d)} bars {d.ts.min().date()}..{d.ts.max().date()}")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
