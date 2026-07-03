"""Average-daily-dollar-volume map for the universe (liquidity gate).

Underlying dollar volume is the coarse liquidity filter: a name that
trades tens of millions of dollars a day has a real, continuous options
market; a thin name does not, no matter how many strikes are listed.
This complements the per-option gates in reality.py (open interest,
bid/ask spread) — the underlying floor removes structurally illiquid
names, the option gates verify the specific contract will fill.

Uses the Yahoo v8 chart endpoint directly (robust; the yfinance
download path 401s in some environments). Writes results/adv.json:

    {"as_of": "...", "adv_usd": {ticker: float_dollars_per_day, ...}}

Run weekly (liquidity is sticky):
    python3 fetch_adv.py
    CS_LIMIT=25 python3 fetch_adv.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import list_tickers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "adv.json")
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36"}
LOOKBACK = 90


def _adv_one(ticker: str) -> float | None:
    import requests
    for host in ("query1", "query2"):
        try:
            u = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
                 f"{ticker}?range=6mo&interval=1d")
            r = requests.get(u, headers=HDR, timeout=20)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            close = np.array([x for x in q["close"] if x is not None], float)
            vol = np.array([x for x in q["volume"] if x is not None], float)
            n = min(len(close), len(vol))
            if n < 20:
                continue
            close, vol = close[-n:], vol[-n:]
            return float((close[-LOOKBACK:] * vol[-LOOKBACK:]).mean())
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    existing: dict = {}
    if os.path.exists(OUT):
        try:
            existing = json.load(open(OUT)).get("adv_usd", {})
        except Exception:  # noqa: BLE001
            existing = {}
    out = dict(existing)
    t0 = time.time()
    fetched = 0
    for i, t in enumerate(tickers, 1):
        adv = _adv_one(t)
        if adv is not None:
            out[t] = round(adv, 1)
            fetched += 1
        if i % 50 == 0:
            with open(OUT, "w") as fh:
                json.dump({"as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "adv_usd": out}, fh)
            print(f"  {i}/{len(tickers)} fetched={fetched} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        time.sleep(0.15)
    with open(OUT, "w") as fh:
        json.dump({"as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "adv_usd": out}, fh)
    n = sum(1 for v in out.values() if v and v >= 50e6)
    print(f"Done: {len(out)} tickers, {n} with ADV>=$50M -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
