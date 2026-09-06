#!/usr/bin/env python3
"""Daily muni dislocation scan -> docs/data/muni_signals.json

    python scripts/muni_scan.py --tape-dir <dir> [--asof YYYY-MM-DD]

Writes the current signals with the bid limit for each. An empty list is a
valid result and is written as such: this strategy is dormant unless someone
is force-selling, and a scanner that always finds something is broken.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "api"))

import _munisignal as S  # noqa: E402

OUT = os.path.join(ROOT, "docs", "data", "muni_signals.json")


def load_tape(tape_dir: str, limit: int | None = None):
    bonds = {}
    files = sorted(glob.glob(os.path.join(tape_dir, "*.csv.gz")))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["ts", "price", "par", "side"])
            d["date"] = pd.to_datetime(d.ts, errors="coerce").dt.strftime("%Y-%m-%d")
            d = d.dropna(subset=["date", "price", "par"])
            if d.empty:
                continue
            d["pw"] = d.price * d.par
            g = d.groupby(["date", "side"], as_index=False).agg(
                pw=("pw", "sum"), par=("par", "sum"))
            g["px"] = g.pw / g.par.replace(0, float("nan"))
            piv = g.pivot(index="date", columns="side", values="px")
            hist = [S.DayPrints(date=str(i), buy=r.get("S"), sell=r.get("P"),
                                dealer=r.get("D")) for i, r in piv.iterrows()]
            bonds[os.path.basename(f)[:12]] = ("", hist)
        except Exception:
            continue
    return bonds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape-dir", required=True)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    bonds = load_tape(a.tape_dir, a.limit)
    if not bonds:
        print("no tape data"); return 1
    latest = max(d.date for _, h in bonds.values() for d in h)
    asof = a.asof or latest
    hits = S.scan(bonds, asof)
    tradeable = [h for h in hits
                 if h.limit_price is not None and h.buy_price <= h.limit_price]

    payload = {
        "as_of": asof,
        "tape_latest": latest,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "bonds_scanned": len(bonds),
        "dislocated": len(hits),
        "actionable": len(tradeable),
        "rules": {"discount_pts": S.DISCOUNT_PTS, "limit_cap": S.LIMIT_CAP,
                  "min_active_days": S.MIN_ACTIVE_DAYS},
        "signals": [{
            "id": h.security_id, "description": h.description,
            "buy_price": round(h.buy_price, 3),
            "trailing_median": round(h.trailing_median, 3),
            "discount_pts": round(h.discount_pts, 2),
            "limit_price": round(h.limit_price, 3) if h.limit_price else None,
            "actionable": bool(h.limit_price is not None
                               and h.buy_price <= h.limit_price),
            "active_days": h.active_days,
            "notes": h.notes,
        } for h in hits],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"{asof}: {len(bonds)} bonds, {len(hits)} dislocated, "
          f"{len(tradeable)} inside limit -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
