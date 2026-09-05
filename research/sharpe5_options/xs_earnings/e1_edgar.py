#!/usr/bin/env python3
"""Build a survivorship-free earnings calendar from SEC EDGAR.

Earnings releases are 8-K filings carrying Item 2.02 ("Results of Operations and
Financial Condition"). The submissions API exposes `items` per filing, so no
document parsing is needed.

Convention for the market-moving session, made explicit:
  acceptanceDateTime is in ET (the API returns e.g. "2023-11-02T16:30:21.000Z"
  which is actually ET wall-clock despite the Z suffix -- verified against known
  cases in e1b_verify.py).
    accepted >= 16:00 ET on day D   -> event session = next trading day after D
    accepted <  09:30 ET on day D   -> event session = D
    09:30 <= accepted < 16:00 on D  -> event session = D  (intraday release)

Writes cache/earnings_8k.parquet
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "earnings_8k.parquet")
UA = "Viktor Mashalov independent quant research viktormashalov@gmail.com"

# the 61-63 symbols present in the option-chain cache
UNIVERSE = [
    "AAPL", "ABBV", "ADBE", "AMD", "AMGN", "AMZN", "AVGO", "BA", "BAC", "BMY",
    "C", "CAT", "CMCSA", "COP", "COST", "CRM", "CSCO", "CVX", "DIS", "F",
    "GE", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM",
    "KO", "LLY", "LOW", "MCD", "META", "MMM", "MRK", "MS", "MSFT", "MU",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PYPL", "QCOM", "SBUX",
    "T", "TSLA", "TXN", "UNH", "UPS", "VZ", "WFC", "WMT", "XOM",
]

_last = [0.0]


def get(url: str, tries: int = 4):
    for i in range(tries):
        # hard rate limit: <= 8 req/s (SEC allows 10)
        gap = time.time() - _last[0]
        if gap < 0.125:
            time.sleep(0.125 - gap)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode())
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print(f"  FAIL {url}: {e}", flush=True)
                return None
            time.sleep(1.5 * (i + 1))
    return None


def main():
    tk = json.load(open(os.path.join(HERE, "company_tickers.json")))
    cikmap = {}
    for v in tk.values():
        cikmap.setdefault(v["ticker"], int(v["cik_str"]))
    missing = [s for s in UNIVERSE if s not in cikmap]
    print(f"universe {len(UNIVERSE)}  cik-resolved {len(UNIVERSE)-len(missing)}  missing {missing}")

    rows = []
    for i, sym in enumerate(UNIVERSE):
        cik = cikmap.get(sym)
        if cik is None:
            continue
        c10 = f"CIK{cik:010d}"
        j = get(f"https://data.sec.gov/submissions/{c10}.json")
        if j is None:
            continue
        frames = []
        rec = j.get("filings", {}).get("recent", {})
        if rec:
            frames.append(pd.DataFrame(rec))
        for extra in j.get("filings", {}).get("files", []):
            je = get(f"https://data.sec.gov/submissions/{extra['name']}")
            if je:
                frames.append(pd.DataFrame(je))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        keep = [c for c in ("form", "items", "filingDate", "reportDate",
                            "acceptanceDateTime", "accessionNumber") if c in df.columns]
        df = df[keep]
        df = df[df["form"] == "8-K"]
        if "items" not in df.columns:
            continue
        df = df[df["items"].fillna("").str.contains("2.02")]
        df["symbol"] = sym
        df["cik"] = cik
        rows.append(df)
        print(f"[{i+1}/{len(UNIVERSE)}] {sym} cik={cik} 8-K/2.02 filings={len(df)}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out["filingDate"] = pd.to_datetime(out["filingDate"])
    out["acceptance"] = pd.to_datetime(out["acceptanceDateTime"].str.replace("Z", "", regex=False),
                                       errors="coerce")
    out = out.sort_values(["symbol", "filingDate"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out)} filings, {out.symbol.nunique()} symbols, "
          f"{out.filingDate.min().date()} .. {out.filingDate.max().date()}")


if __name__ == "__main__":
    main()
