#!/usr/bin/env python3
"""Which muni product has enough inventory to sell: the list, or the price check?

Two candidate products, and only one question decides between them — how often
does each one have something to say?

  LIST     scan the tape nightly, publish bonds trading >= 3 points under their
           own 60-day trend. Sells only if the list is non-empty most days AND
           the names are ones an advisor could actually buy.

  CHECK    advisor pastes a CUSIP and the price a dealer quoted; we answer
           within limit / too rich, and by how much. Sells only if a
           meaningful share of real quotes come back "too rich" — a tool that
           says "fine" every time is a tool nobody renews.

Both are measured on the same MSRB tape and the same locked rules in
api/_munisignal.py. Nothing here is a backtest of returns; it is a count of
how often the product fires, which is the thing I do not yet know.

    python research/muni_edge/product_shape.py [--days 500]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "api"))

import _munisignal as S  # noqa: E402

TAPE = "/home/user/viki-m13/bonds/munis/data/trades"
RETAIL_MAX_PAR = 100_000        # the lot size an advisor actually buys


def load_bond(path: str) -> tuple[str, list[S.DayPrints], pd.DataFrame] | None:
    """Collapse one bond's tape into daily prints, keeping the raw rows too."""
    try:
        d = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if not {"ts", "price", "side"} <= set(d.columns):
        return None
    d["date"] = pd.to_datetime(d.ts, errors="coerce").dt.date
    d["price"] = pd.to_numeric(d.price, errors="coerce")
    d["par"] = pd.to_numeric(d.get("par"), errors="coerce")
    d = d.dropna(subset=["date", "price"])
    d = d[(d.price > 20) & (d.price < 200)]        # data-error guard
    if d.empty:
        return None

    days = []
    for date, g in d.groupby("date"):
        buys = g[g.side == "S"].price
        sells = g[g.side == "P"].price
        deal = g[g.side == "D"].price
        days.append(S.DayPrints(
            date=str(date),
            buy=float(buys.median()) if len(buys) else None,
            sell=float(sells.median()) if len(sells) else None,
            dealer=float(deal.median()) if len(deal) else None))
    days.sort(key=lambda x: x.date)
    return os.path.basename(path).split(".")[0], days, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500,
                    help="how many recent trading days to measure")
    ap.add_argument("--limit-bonds", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(TAPE, "*.csv.gz")))
    if a.limit_bonds:
        paths = paths[:a.limit_bonds]
    print(f"loading {len(paths):,} bonds")

    bonds: dict[str, list[S.DayPrints]] = {}
    raw: dict[str, pd.DataFrame] = {}
    for i, p in enumerate(paths, 1):
        r = load_bond(p)
        if r:
            bonds[r[0]] = r[1]
            raw[r[0]] = r[2]
        if i % 800 == 0:
            print(f"  …{i:,}/{len(paths):,}")
    print(f"  usable: {len(bonds):,}")

    # ---- the calendar we measure over -------------------------------------
    all_days = sorted({d.date for h in bonds.values() for d in h})
    cal = all_days[-a.days:]
    print(f"  measuring {cal[0]} -> {cal[-1]} ({len(cal)} days)\n")

    # ============ PRODUCT 1: the nightly list ==============================
    print("=" * 78)
    print("PRODUCT 1 — THE NIGHTLY LIST OF DISLOCATED BONDS")
    print("=" * 78)

    per_day = collections.Counter()
    per_day_actionable = collections.Counter()
    disc = []
    fired_bonds = set()
    for day in cal:
        for sid, hist in bonds.items():
            # cheap pre-filter: did this bond print a customer buy that day?
            if not any(h.date == day and h.buy is not None for h in hist):
                continue
            sig = S.evaluate(sid, hist, day)
            if not sig.dislocated:
                continue
            per_day[day] += 1
            fired_bonds.add(sid)
            disc.append(sig.discount_pts)
            if (sig.limit_price is not None and sig.buy_price is not None
                    and sig.buy_price <= sig.limit_price):
                per_day_actionable[day] += 1

    counts = [per_day.get(d, 0) for d in cal]
    act = [per_day_actionable.get(d, 0) for d in cal]
    empty = sum(1 for c in counts if c == 0)
    empty_act = sum(1 for c in act if c == 0)
    print(f"  dislocations found:      {sum(counts):,} across {len(cal)} days")
    print(f"  distinct bonds involved: {len(fired_bonds):,} of {len(bonds):,}")
    print(f"  per day  median {sorted(counts)[len(counts)//2]}   "
          f"mean {sum(counts)/len(counts):.1f}   max {max(counts)}")
    print(f"  days with an EMPTY list: {empty}/{len(cal)} ({100*empty/len(cal):.0f}%)")
    print(f"  ...and where the print was also inside the limit (buyable):")
    print(f"     per day  median {sorted(act)[len(act)//2]}   "
          f"mean {sum(act)/len(act):.1f}")
    print(f"     days with nothing buyable: {empty_act}/{len(cal)} "
          f"({100*empty_act/len(cal):.0f}%)")
    if disc:
        disc.sort()
        print(f"  discount when it fires: median {disc[len(disc)//2]:.2f} pts, "
              f"90th pct {disc[int(.9*len(disc))]:.2f} pts")

    # ============ PRODUCT 2: the price check ===============================
    print()
    print("=" * 78)
    print("PRODUCT 2 — THE PRE-TRADE PRICE CHECK")
    print("=" * 78)
    print("  Every retail-size customer BUY on the tape is re-run as if the")
    print("  advisor had asked us first. How often would we have said stop?\n")

    verdicts = collections.Counter()
    over_pts: list[float] = []
    saved: list[float] = []
    for sid, hist in bonds.items():
        r = raw[sid]
        r = r[(r.side == "S") & (r.par <= RETAIL_MAX_PAR)]
        if r.empty:
            continue
        by_day = {d.date: d for d in hist}
        for row in r.itertuples():
            day = str(row.date)
            if day not in by_day or day < cal[0] or day > cal[-1]:
                continue
            sig = S.evaluate(sid, hist, day)
            v = S.price_verdict(float(row.price), sig)
            verdicts[v["verdict"]] += 1
            if v["verdict"] == "too rich":
                over_pts.append(v["over_limit_pts"])
                saved.append(v["over_limit_pts"] / 100.0 * float(row.par))

    tot = sum(verdicts.values())
    print(f"  quotes assessed: {tot:,}")
    for k, n in verdicts.most_common():
        print(f"    {k:<14} {n:>9,}  {100*n/max(tot,1):>5.1f}%")
    if over_pts:
        over_pts.sort()
        saved.sort()
        print(f"\n  when we say 'too rich':")
        print(f"    median overpayment  {over_pts[len(over_pts)//2]:.2f} pts")
        print(f"    median $ on the lot ${saved[len(saved)//2]:,.0f}")
        print(f"    90th pct $          ${saved[int(.9*len(saved))]:,.0f}")
        print(f"    total across all    ${sum(saved):,.0f}")

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    hit = 100 * verdicts.get("too rich", 0) / max(tot, 1)
    print(f"  The list is empty on {100*empty/len(cal):.0f}% of days.")
    print(f"  The price check has an opinion on {100*(tot-verdicts.get('unknown',0))/max(tot,1):.0f}% "
          f"of real quotes and objects to {hit:.0f}% of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
