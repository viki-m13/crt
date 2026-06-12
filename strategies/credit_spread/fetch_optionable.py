"""Determine which universe tickers actually have listed options.

A credit spread can only be sold on an underlying with a listed options
chain. The price panel alone can't tell us that — e.g. SEB (Seaboard,
~$5,800/share) has NO listed options at all, yet the legacy engine
happily published "signals" for it (43% of one experimental config's
inventory). Conservative-execution claims are meaningless for such
names, so the engine fails closed: a ticker is publishable only if this
scan found a non-empty options chain for it.

Output: results/optionable.json
    {"as_of": "...", "optionable": {ticker: true/false, ...}}

Run weekly (or before any scan); listings are sticky.
    python3 fetch_optionable.py
    CS_LIMIT=25 python3 fetch_optionable.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import yfinance as yf

from common import list_tickers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "optionable.json")


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    # Start from the existing file so reruns only fill gaps.
    existing: dict[str, bool] = {}
    if os.path.exists(OUT):
        try:
            with open(OUT) as fh:
                existing = json.load(fh).get("optionable", {})
        except Exception:  # noqa: BLE001
            existing = {}
    out = dict(existing)
    t0 = time.time()
    fetched = 0
    for i, t in enumerate(tickers, 1):
        if t in out:
            continue
        ok = False
        for attempt in range(2):
            try:
                chains = yf.Ticker(t).options
                ok = bool(chains)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1 + attempt)
        out[t] = ok
        fetched += 1
        if fetched % 50 == 0:
            print(f"  {i}/{len(tickers)} fetched={fetched} "
                  f"optionable={sum(out.values())} elapsed={time.time()-t0:.0f}s",
                  flush=True)
            # checkpoint
            with open(OUT, "w") as fh:
                json.dump({"as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "optionable": out}, fh, indent=0)
    with open(OUT, "w") as fh:
        json.dump({"as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "optionable": out}, fh, indent=0)
    n_opt = sum(1 for v in out.values() if v)
    print(f"Done: {len(out)} tickers checked, {n_opt} optionable -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
