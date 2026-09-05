#!/usr/bin/env python3
"""Score the pre-registered 0DTE strategies against collected snapshots.

Implements PREREGISTRATION.md exactly — S1 (morning richness condor) and
S2 (last-hour momentum ride) on SPX, worst-side fills from the recorded
snapshots, cash settlement at the official SPX close (^GSPC). Refuses to
draw conclusions before the pre-stated 60-valid-day minimum; until then it
prints the running tally only.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = os.path.join(HERE, "snaps")
HDR = {"User-Agent": "Mozilla/5.0"}
MIN_DAYS = 60


def spx_closes() -> pd.Series:
    for host in ("query1", "query2"):
        try:
            u = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
                 f"%5EGSPC?range=1y&interval=1d")
            req = urllib.request.Request(u, headers=HDR)
            with urllib.request.urlopen(req, timeout=30) as r:
                res = json.loads(r.read().decode())["chart"]["result"][0]
            import time as _t
            dates = [_t.strftime("%Y-%m-%d", _t.gmtime(t)) for t in res["timestamp"]]
            closes = res["indicators"]["quote"][0]["close"]
            s = pd.Series(closes, index=dates, dtype=float).dropna()
            if len(s):
                return s
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no ^GSPC closes")


def load_snap(day: str, prefix: str) -> pd.DataFrame | None:
    """First SPX snapshot at/after HHMM `prefix` on `day`."""
    files = sorted(glob.glob(os.path.join(SNAPS, day, "*Z_SPX.csv.gz")))
    for p in files:
        hhmm = os.path.basename(p)[:4]
        if hhmm >= prefix:
            with gzip.open(p, "rt") as f:
                df = pd.read_csv(f)
            return df[df.expiration == day]      # 0DTE legs only
    return None


def pick(df, side, target_k):
    g = df[df.call_put == side].copy()
    if g.empty:
        return None
    return g.loc[(g.strike - target_k).abs().idxmin()]


def s1_condor(day: str, close: float) -> dict | None:
    df = load_snap(day, "1330")
    if df is None or df.empty:
        return None
    spot = float(df.spot.iloc[0])
    legs = {}
    for side, sgn in (("C", 1), ("P", -1)):
        g = df[(df.call_put == side) & df.delta.notna() & (df.bid > 0)]
        if g.empty:
            return None
        sh = g.loc[(g.delta.abs() - 0.15).abs().idxmin()]
        lg = pick(df[df.ask > 0], side, float(sh.strike) + sgn * 0.003 * spot)
        if lg is None:
            return None
        legs[side] = (sh, lg)
    (cs, cl), (ps, pl) = legs["C"], legs["P"]
    if not (float(pl.strike) < float(ps.strike) < float(cs.strike) < float(cl.strike)):
        return None
    credit = float(cs.bid) - float(cl.ask) + float(ps.bid) - float(pl.ask)
    if credit <= 0:
        return None
    width = max(float(cl.strike) - float(cs.strike),
                float(ps.strike) - float(pl.strike))
    loss_c = max(close - float(cs.strike), 0) - max(close - float(cl.strike), 0)
    loss_p = max(float(ps.strike) - close, 0) - max(float(pl.strike) - close, 0)
    pnl = credit - loss_c - loss_p
    return dict(day=day, strat="S1", pnl_frac=pnl / (width - credit),
                credit=credit, width=width)


def s2_momentum(day: str, close: float) -> dict | None:
    early, late = load_snap(day, "1330"), load_snap(day, "1900")
    if early is None or late is None or early.empty or late.empty:
        return None
    s0, s1 = float(early.spot.iloc[0]), float(late.spot.iloc[0])
    r = s1 / s0 - 1
    if abs(r) < 0.003:
        return dict(day=day, strat="S2", pnl_frac=np.nan, skipped="no signal")
    side = "C" if r > 0 else "P"
    sgn = 1 if r > 0 else -1
    atm = pick(late[late.ask > 0], side, s1)
    far = pick(late[late.bid > 0], side, s1 + sgn * 0.003 * s1)
    if atm is None or far is None or float(atm.strike) == float(far.strike):
        return None
    debit = float(atm.ask) - float(far.bid)
    gap = abs(float(far.strike) - float(atm.strike))
    if debit <= 0 or debit >= gap:
        return None
    if side == "C":
        pay = (max(close - float(atm.strike), 0) - max(close - float(far.strike), 0))
    else:
        pay = (max(float(atm.strike) - close, 0) - max(float(far.strike) - close, 0))
    return dict(day=day, strat="S2", pnl_frac=(pay - debit) / debit)


def main():
    days = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SNAPS, "*")))
    if not days:
        print("no snapshots yet — run collect.py / wait for the cron")
        return
    closes = spx_closes()
    rows = []
    for day in days:
        if day not in closes.index:
            continue
        c = float(closes.loc[day])
        for fn in (s1_condor, s2_momentum):
            r = fn(day, c)
            if r:
                rows.append(r)
    if not rows:
        print("no scorable days yet")
        return
    res = pd.DataFrame(rows)
    for strat, g in res.groupby("strat"):
        v = g.pnl_frac.dropna()
        print(f"\n{strat}: {len(v)} valid days (min {MIN_DAYS} before any conclusion)")
        if not len(v):
            continue
        print(f"  mean P&L/maxloss={v.mean():+.4f}  win={(v>0).mean():.1%}  "
              f"worst={v.min():+.3f}")
        if len(v) >= MIN_DAYS:
            boots = [np.random.default_rng(i).choice(v, len(v)).mean()
                     for i in range(2000)]
            lo, hi = np.percentile(boots, [5, 95])
            verdict = "PASS" if lo > 0 else "REJECT (pre-stated kill criterion)"
            print(f"  bootstrap 90% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
        else:
            print(f"  {MIN_DAYS - len(v)} more valid days required — no conclusion yet")


if __name__ == "__main__":
    main()
