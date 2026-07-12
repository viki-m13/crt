"""Refresh the SPY daily panel from the Yahoo v8 chart endpoint (robust;
the yfinance download path 401s in some environments). Self-contained —
writes strategies/spx_predict/data/SPY.json so the signal never depends
on the credit_spread universe refresh.

    python3 fetch_spy.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "SPY.json")
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_spy() -> dict | None:
    import requests
    for host in ("query1", "query2"):
        try:
            u = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
                 f"SPY?range=max&interval=1d&events=div%2Csplit")
            r = requests.get(u, headers=HDR, timeout=30)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            # adjclose carries split+dividend adjustment (total-return basis,
            # matching the existing cache_full/SPY.json panel).
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            closes = adj if adj else q["close"]
            dates, prices = [], []
            for t, c in zip(ts, closes):
                if c is None:
                    continue
                dates.append(time.strftime("%Y-%m-%d", time.gmtime(t)))
                prices.append(float(c))
            if len(dates) < 1000:
                continue
            return {"ticker": "SPY", "series": {"dates": dates, "prices": prices}}
        except Exception as e:  # noqa: BLE001
            print(f"  {host} failed: {e}", file=sys.stderr)
            continue
    return None


def fetch_spy_yf() -> dict | None:
    """Fallback: yfinance handles Yahoo's cookie/crumb auth (works on CI
    runners where the raw chart endpoint 401s)."""
    try:
        import yfinance as yf
        h = yf.Ticker("SPY").history(period="max", auto_adjust=True)
        if h is None or len(h) < 1000:
            return None
        dates = [d.strftime("%Y-%m-%d") for d in h.index]
        prices = [float(x) for x in h["Close"].tolist()]
        return {"ticker": "SPY", "series": {"dates": dates, "prices": prices}}
    except Exception as e:  # noqa: BLE001
        print(f"  yfinance fallback failed: {e}", file=sys.stderr)
        return None


def main() -> int:
    blob = fetch_spy() or fetch_spy_yf()
    if blob is None:
        print("FAILED to fetch SPY (chart endpoint and yfinance fallback)",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(blob, fh)
    d = blob["series"]["dates"]
    print(f"SPY refreshed: {d[0]}..{d[-1]}  n={len(d)}  last={blob['series']['prices'][-1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
