#!/usr/bin/env python3
"""Fetch 10-Q / 10-K filing dates per symbol, used to disambiguate which
Item-2.02 8-K is the quarterly EARNINGS release.

Some issuers file several Item 2.02 8-Ks per quarter (TSLA files quarterly
production/delivery numbers under 2.02; ABBV files segment updates). The
earnings release is the one accompanied by the periodic report: every issuer
files its 10-Q/10-K within a few days of the earnings press release.
"""
from __future__ import annotations

import os

import pandas as pd

import e1_edgar as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "periodic.parquet")


def main():
    ek = pd.read_parquet(os.path.join(HERE, "earnings_8k.parquet"))
    ciks = ek[["symbol", "cik"]].drop_duplicates().values.tolist()
    rows = []
    for i, (sym, cik) in enumerate(ciks):
        j = E.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        if j is None:
            continue
        frames = [pd.DataFrame(j["filings"]["recent"])]
        for x in j["filings"].get("files", []):
            je = E.get("https://data.sec.gov/submissions/" + x["name"])
            if je:
                frames.append(pd.DataFrame(je))
        df = pd.concat(frames, ignore_index=True)
        df = df[df["form"].isin(["10-Q", "10-K"])][["form", "filingDate", "reportDate"]]
        df["symbol"] = sym
        rows.append(df)
        print(f"[{i+1}/{len(ciks)}] {sym} periodic={len(df)}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out["filingDate"] = pd.to_datetime(out["filingDate"])
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows")


if __name__ == "__main__":
    main()
