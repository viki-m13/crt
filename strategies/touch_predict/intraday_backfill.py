"""Intraday OHLCV downloader (2-min) for the QSIT strategy.

Downloads 60 days of 2-min bars for the target universe + a
correlated basket. Stores per-ticker JSON in
strategies/touch_predict/data/intraday/.

yfinance limit: 60 days of 2-min data. We pull it all in one call.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "data", "intraday")
os.makedirs(CACHE_DIR, exist_ok=True)


# Target universe — user-mentioned + cross-asset basket for the
# coordination signal.
UNIVERSE = [
    # Core targets
    "QCOM", "INTC", "SLV", "GLD",
    # Tech / semis (correlated to QCOM, INTC)
    "AVGO", "AMD", "NVDA", "XLK", "SMH",
    # Precious metals (correlated to SLV, GLD)
    "GDX", "PLTR",   # PLTR for high-vol single name
    # Broad market
    "SPY", "QQQ", "IWM", "TLT", "VIX",
    # Energy (different correlation cluster)
    "XLE", "USO",
]


def _fetch_one(ticker: str, period: str = "60d") -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            d = yf.download(
                ticker, period=period, interval="2m",
                progress=False, auto_adjust=True,
            )
            if d is None or d.empty:
                return None
            # Flatten multi-index columns
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            return d
        except Exception as exc:  # noqa: BLE001
            print(f"  {ticker}: attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def _write(ticker: str, frame: pd.DataFrame) -> None:
    frame = frame.dropna()
    if frame.empty:
        return
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    blob = {
        "ticker": ticker,
        "datetimes": [d.strftime("%Y-%m-%dT%H:%M:%S%z") for d in frame.index],
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
    print(f"Downloading 60d × 2-min for {len(UNIVERSE)} tickers…")
    t0 = time.time()
    ok = fail = 0
    for tk in UNIVERSE:
        d = _fetch_one(tk)
        if d is None or len(d) < 100:
            print(f"  {tk}: failed or sparse")
            fail += 1
            continue
        _write(tk, d)
        print(f"  {tk}: {len(d)} bars  range {d.index[0]} → {d.index[-1]}")
        ok += 1
    print(f"\nDone. ok={ok} fail={fail} elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
