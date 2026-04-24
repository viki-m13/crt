"""v2 OHLCV backfill — batched yfinance fetch to a dedicated cache.

Writes per-ticker JSON files to `strategies/touch_predict/data/ohlcv/`
with { dates, open, high, low, close, volume } — everything we need
for volume-climax and intraday-range signals. Does NOT touch the
main `docs/data/tickers/*.json` files.

Run:
    python strategies/touch_predict/v2_backfill.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "data", "ohlcv")
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, _HERE)
from v2_liquid_universe import LIQUID_TICKERS

START = "2015-01-02"
CHUNK = 20
RETRIES = 3


def _latest_trading_day() -> str:
    import pandas_market_calendars as mcal
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    nyse = mcal.get_calendar("NYSE")
    sessions = nyse.valid_days(
        start_date=(today - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )
    return (sessions[-1] if len(sessions) else today).strftime("%Y-%m-%d")


def _fetch_ohlcv(tickers: list[str], start: str, end: str):
    last_err = None
    for attempt in range(RETRIES):
        try:
            df = yf.download(
                tickers=tickers, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=True,
                group_by="column",
            )
            if df is None or df.empty:
                last_err = RuntimeError("empty dataframe")
                continue
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 ** attempt)
    print(f"  chunk failed after {RETRIES}: {last_err}", file=sys.stderr)
    return None


def _write(ticker: str, frame: pd.DataFrame) -> None:
    frame = frame.dropna()
    if frame.empty:
        return
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    blob = {
        "dates":  [d.strftime("%Y-%m-%d") for d in frame.index],
        "open":   frame["Open"].astype(float).tolist(),
        "high":   frame["High"].astype(float).tolist(),
        "low":    frame["Low"].astype(float).tolist(),
        "close":  frame["Close"].astype(float).tolist(),
        "volume": frame["Volume"].astype(float).tolist(),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(blob, fh)
    os.replace(tmp, path)


def main() -> int:
    target_end = _latest_trading_day()
    fetch_end = (np.datetime64(target_end, "D") + 1).astype(str)
    print(f"Fetching OHLCV {START} → {target_end} for {len(LIQUID_TICKERS)} liquid tickers")

    t0 = time.time()
    ok = fail = 0
    for chunk_start in range(0, len(LIQUID_TICKERS), CHUNK):
        chunk = LIQUID_TICKERS[chunk_start : chunk_start + CHUNK]
        df = _fetch_ohlcv(chunk, START, fetch_end)
        if df is None:
            fail += len(chunk)
            continue
        for t in chunk:
            try:
                if len(chunk) == 1:
                    frame = df[["Open", "High", "Low", "Close", "Volume"]]
                else:
                    frame = pd.DataFrame({
                        "Open":   df["Open"][t],
                        "High":   df["High"][t],
                        "Low":    df["Low"][t],
                        "Close":  df["Close"][t],
                        "Volume": df["Volume"][t],
                    })
                _write(t, frame)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {t}: {exc}", file=sys.stderr)
                fail += 1
        print(f"  chunk {chunk_start // CHUNK + 1}: ok={ok} fail={fail} elapsed={time.time()-t0:.1f}s")

    print(f"\nDone. ok={ok} fail={fail} total_elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
