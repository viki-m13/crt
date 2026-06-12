"""Fetch full (period=max) adjusted-close history for the CreditFloor
universe into a standalone cache directory, on a single consistent
adjustment basis per ticker.

The main site's docs/data/tickers/*.json series start in 2015 and are
append-maintained; deep replay validation (2008 GFC, 2011, 2015-16,
2018 Q4, 2020 COVID) needs decades of history. This writes
``cache_full/{TICKER}.json`` with the same ``{"series": {"dates":
[...], "prices": [...]}}`` shape that ``common.load_series`` consumes,
selectable via the ``CS_DATA_DIR`` environment variable.

    python3 fetch_full_history.py            # fill gaps only
    CS_REFRESH=1 python3 fetch_full_history.py   # refetch everything (~3 min)
    CS_LIMIT=25 python3 fetch_full_history.py

CS_REFRESH=1 is what the daily cron uses: refetching the whole series
every run guarantees a single consistent adjustment basis (no
split/dividend seams, ever) at the cost of ~3 minutes of yfinance
traffic.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
import yfinance as yf

from common import list_tickers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "cache_full")
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK = 40
RETRIES = 3


def fetch_chunk(tickers: list[str]) -> pd.DataFrame | None:
    last = None
    for attempt in range(RETRIES):
        try:
            df = yf.download(tickers=tickers, period="max", interval="1d",
                             auto_adjust=True, progress=False, threads=True,
                             group_by="column")
            if df is None or df.empty:
                last = RuntimeError("empty")
                continue
            close = df["Close"] if len(tickers) > 1 else df[["Close"]].rename(
                columns={"Close": tickers[0]})
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            return close
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    print(f"  chunk failed: {last}", file=sys.stderr)
    return None


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    refresh = os.environ.get("CS_REFRESH") == "1"
    t0 = time.time()
    done = failed = skipped = 0
    for s in range(0, len(tickers), CHUNK):
        chunk = [t for t in tickers[s:s + CHUNK]
                 if refresh or not os.path.exists(os.path.join(OUT_DIR, f"{t}.json"))]
        skipped += len(tickers[s:s + CHUNK]) - len(chunk)
        if not chunk:
            continue
        close = fetch_chunk(chunk)
        if close is None:
            failed += len(chunk)
            continue
        for t in chunk:
            if t not in close.columns:
                failed += 1
                continue
            col = close[t].dropna()
            col = col[col > 0]
            if len(col) < 300:
                failed += 1
                continue
            blob = {"ticker": t, "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in col.index],
                "prices": [float(v) for v in col.values],
            }}
            tmp = os.path.join(OUT_DIR, f"{t}.json.tmp")
            with open(tmp, "w") as fh:
                json.dump(blob, fh)
            os.replace(tmp, os.path.join(OUT_DIR, f"{t}.json"))
            done += 1
        print(f"  {s + CHUNK}/{len(tickers)} done={done} failed={failed} "
              f"skipped={skipped} elapsed={time.time() - t0:.0f}s", flush=True)
        time.sleep(0.4)
    print(f"Done: {done} written, {failed} failed, {skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
