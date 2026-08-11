#!/usr/bin/env python3
"""0DTE chain snapshot collector.

No free historical intraday 0DTE data exists (verified 2026-08-11: the only
free EOD archives stop at 2013, before daily expirations began in late 2022).
So this project records its own: CBOE delayed quotes (free, ~15-min delay,
real bid/ask with sizes) snapshotted several times per trading day, filtered
to expirations within 1 calendar day and strikes within +/-2.5% of spot.

Each run writes research/zerodte/snaps/<YYYY-MM-DD>/<HHMM>Z_<SYM>.csv.gz
(~10-50 KB each). The pre-registered strategies in PREREGISTRATION.md are
scored against these snapshots by evaluate.py — entry fills at the recorded
worst side, settlement at the official index close. The quote delay is part
of the record (snap_utc is the collection time, not the quote time); a live
implementation would see quotes ~15 minutes fresher.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import re
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = os.path.join(HERE, "snaps")
SYMS = {"_SPX": "SPX", "SPY": "SPY"}
BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"
OCC = re.compile(r"^(?P<root>[A-Z]+)(?P<d>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")


def fetch(sym_api: str) -> dict:
    req = urllib.request.Request(BASE.format(sym_api),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def snapshot(sym_api: str, sym: str, now_utc: dt.datetime) -> pd.DataFrame | None:
    d = fetch(sym_api)["data"]
    spot = float(d["current_price"])
    rows = []
    for o in d["options"]:
        m = OCC.match(o["option"])
        if not m:
            continue
        exp = dt.date(2000 + int(m["d"][:2]), int(m["d"][2:4]), int(m["d"][4:6]))
        cal_dte = (exp - now_utc.date()).days
        if not 0 <= cal_dte <= 1:
            continue
        k = int(m["k"]) / 1000.0
        if abs(k / spot - 1) > 0.025:
            continue
        rows.append(dict(
            root=m["root"], expiration=exp.isoformat(), call_put=m["cp"],
            strike=k, bid=o.get("bid"), ask=o.get("ask"),
            bid_size=o.get("bid_size"), ask_size=o.get("ask_size"),
            iv=o.get("iv"), delta=o.get("delta"), gamma=o.get("gamma"),
            volume=o.get("volume"), open_interest=o.get("open_interest"),
            last=o.get("last_trade_price")))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["spot"] = spot
    df["snap_utc"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["symbol"] = sym
    return df


def main():
    now = dt.datetime.now(dt.timezone.utc)
    if now.weekday() >= 5:
        print("weekend, skip")
        return
    day_dir = os.path.join(SNAPS, now.strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)
    for sym_api, sym in SYMS.items():
        try:
            df = snapshot(sym_api, sym, now)
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: FAIL {e}")
            continue
        if df is None or df.empty:
            print(f"{sym}: no rows in window")
            continue
        path = os.path.join(day_dir, f"{now.strftime('%H%M')}Z_{sym}.csv.gz")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        with gzip.open(path, "wt") as f:
            f.write(buf.getvalue())
        print(f"{sym}: {len(df)} rows, spot={df.spot.iloc[0]:.2f} -> {path}")


if __name__ == "__main__":
    main()
