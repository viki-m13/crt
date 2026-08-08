#!/usr/bin/env python3
"""Resumable fetcher for real EOD option chains from DoltHub post-no-preference/options.

Per observation date: fetch option_chain rows for the UNIVERSE (batched IN-lists,
paginated along the PK), plus volatility_history rows. Writes one parquet per
date into cache/chains/ and cache/vol/. A manifest records empty/done dates so
reruns skip finished work. Safe to kill and restart.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
TIER = os.environ.get("FETCH_TIER", "1")
SUF = "" if TIER == "1" else "2"
CHAINS_DIR = os.path.join(CACHE, "chains" + SUF)
VOL_DIR = os.path.join(CACHE, "vol" + SUF)
MANIFEST = os.path.join(CACHE, f"manifest{SUF}.json")

UNIVERSE2 = [
    "ADI", "AMAT", "ANET", "AXP", "BKNG", "BLK", "CB", "CI", "CL", "CMG",
    "COF", "DE", "DHR", "DUK", "EBAY", "ETN", "FDX", "GD", "GILD", "GOOG",
    "HCA", "ISRG", "KLAC", "LIN", "LMT", "LRCX", "MA", "MDLZ", "MDT", "MET",
    "MRVL", "NEE", "NOC", "NOW", "ON", "ORLY", "PANW", "PGR", "PLTR", "PM",
    "PNC", "REGN", "ROP", "RTX", "SCHW", "SLB", "SO", "SPG", "SYK", "TGT",
    "TJX", "TMO", "TMUS", "UNP", "USB", "V", "VLO", "VRTX", "WDC", "WMB",
    "ZTS", "ABNB", "UBER", "DASH", "COIN", "ROKU", "SMCI",
]

UNIVERSE = [
    # ETFs present in the DB
    "SPY", "DIA", "XLE", "XLF",
    # mega/large caps (S&P + Dow components, liquid options)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    "XOM", "CVX", "COP",
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "DIS", "NKE",
    "JNJ", "PFE", "MRK", "UNH", "LLY", "ABBV", "BMY", "AMGN",
    "CAT", "BA", "GE", "HON", "UPS", "MMM",
    "T", "VZ", "CMCSA",
    "CSCO", "ORCL", "CRM", "ADBE", "INTC", "AMD", "QCOM", "NFLX", "MU", "TXN",
    "IBM", "PYPL", "SBUX", "LOW", "GM", "F",
]
if TIER == "2":
    UNIVERSE = UNIVERSE2

BATCH = 8
PAGE = 1000
START = dt.date(2019, 2, 1)
END = dt.date(2026, 8, 7)

COLS = "`date`,act_symbol,expiration,strike,call_put,bid,ask,vol,delta,gamma,theta,vega"


def q(sql: str, tries: int = 5, timeout: int = 150):
    url = BASE + "?" + urllib.parse.urlencode({"q": sql})
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            if d.get("query_execution_status") in ("Success", "RowLimit"):
                return d["rows"]
            last = d.get("query_execution_message", "")
            # deadline exceeded → retry (server-side variance)
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"query failed after {tries}: {last} :: {sql[:120]}")


def fetch_date(day: str) -> tuple[str, str]:
    """Fetch one observation date. Returns (day, status)."""
    # cheap probe: AAPL is always covered when a scrape ran
    probe_sym = "AAPL" if TIER == "1" else "GILD"
    probe = q(f"SELECT COUNT(*) n FROM option_chain WHERE `date`='{day}' AND act_symbol='{probe_sym}'")
    if int(probe[0]["n"]) == 0:
        return day, "empty"

    frames = []
    for i in range(0, len(UNIVERSE), BATCH):
        batch = UNIVERSE[i : i + BATCH]
        inlist = ",".join(f"'{s}'" for s in batch)
        offset = 0
        while True:
            rows = q(
                f"SELECT {COLS} FROM option_chain WHERE `date`='{day}' AND act_symbol IN ({inlist}) "
                f"ORDER BY act_symbol, expiration, strike, call_put LIMIT {PAGE} OFFSET {offset}"
            )
            if rows:
                frames.append(pd.DataFrame(rows))
            if len(rows) < PAGE:
                break
            offset += PAGE
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        for c in ("bid", "ask", "vol", "delta", "gamma", "theta", "vega", "strike"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_parquet(os.path.join(CHAINS_DIR, f"{day}.parquet"), index=False)

    vrows = q(
        f"SELECT * FROM volatility_history WHERE `date`='{day}' AND act_symbol IN "
        f"({','.join(chr(39)+s+chr(39) for s in UNIVERSE)})"
    )
    if vrows:
        pd.DataFrame(vrows).to_parquet(os.path.join(VOL_DIR, f"{day}.parquet"), index=False)
    return day, f"ok:{len(df)}"


def main():
    os.makedirs(CHAINS_DIR, exist_ok=True)
    os.makedirs(VOL_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))

    days = []
    d = START
    while d <= END:
        # scrapes land Mon-Sat (Sat carries Friday close in 2019)
        if d.weekday() != 6:
            days.append(d.isoformat())
        d += dt.timedelta(days=1)
    todo = [x for x in days if x not in manifest]
    print(f"{len(todo)} dates to fetch ({len(days)} candidates, {len(manifest)} done)", flush=True)

    done_ct = 0
    lock_write = time.time()
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_date, day): day for day in todo}
        for fut in cf.as_completed(futs):
            day = futs[fut]
            try:
                _, status = fut.result()
                manifest[day] = status
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {day}: {e}", flush=True)
                manifest[day] = "fail"
            done_ct += 1
            if time.time() - lock_write > 15:
                json.dump(manifest, open(MANIFEST, "w"))
                lock_write = time.time()
            if done_ct % 50 == 0:
                nok = sum(1 for v in manifest.values() if v.startswith("ok"))
                print(f"progress {done_ct}/{len(todo)} (ok dates total: {nok})", flush=True)
    json.dump(manifest, open(MANIFEST, "w"))
    nok = sum(1 for v in manifest.values() if v.startswith("ok"))
    nfail = sum(1 for v in manifest.values() if v == "fail")
    print(f"DONE. ok={nok} empty={sum(1 for v in manifest.values() if v=='empty')} fail={nfail}", flush=True)


if __name__ == "__main__":
    main()
