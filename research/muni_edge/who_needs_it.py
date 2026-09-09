#!/usr/bin/env python3
"""Which buyer actually needs the price check?

MSRB research says odd-lot muni trading is no longer mostly retail — SMA
managers now account for something like half of it, and fee-based accounts
execute materially better than non-fee-based ones. If that is true here, the
execution penalty should FALL as lot size rises, and the question of who to
sell to answers itself: the tool is worth most to whoever still pays the
widest markup.

This slices the same symmetry-controlled measurement by trade size. It uses
the cache built by price_check_control.py.

    python research/muni_edge/who_needs_it.py
"""
from __future__ import annotations

import os
import statistics as st
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "api"))

import _munisignal as S  # noqa: E402

CACHE = os.path.join(HERE, ".tape_cache.pkl.gz")
BUCKETS = [(0, 25_000), (25_000, 100_000), (100_000, 500_000),
           (500_000, 1_000_000), (1_000_000, float("inf"))]
DAYS = 250


def label(lo, hi):
    h = "+" if hi == float("inf") else f"-${hi/1000:,.0f}k"
    return f"${lo/1000:,.0f}k{h}"


def main():
    if not os.path.exists(CACHE):
        print("run price_check_control.py first to build the cache")
        return 1
    t = pd.read_pickle(CACHE)
    print(f"{len(t):,} prints, {t.sid.nunique():,} bonds")

    g = t.groupby(["sid", "date", "side"]).price.median().unstack("side")
    for c in ("S", "P", "D"):
        if c not in g.columns:
            g[c] = float("nan")
    g = g.reset_index()
    g["mid"] = [
        S.DayPrints(date=r.date, buy=None if pd.isna(r.S) else float(r.S),
                    sell=None if pd.isna(r.P) else float(r.P),
                    dealer=None if pd.isna(r.D) else float(r.D)).mid
        for r in g.itertuples()]
    g = g.dropna(subset=["mid"]).sort_values(["sid", "date"])
    g["prior_mid"] = g.groupby("sid")["mid"].shift(1)
    g = g.dropna(subset=["prior_mid"])

    days = sorted(g.date.unique())[-DAYS:]
    g = g[g.date.isin(days)]
    ref = {(r.sid, r.date): r.prior_mid for r in g.itertuples()}
    print(f"window {days[0]} -> {days[-1]}\n")

    r = t[t.date.isin(days)].dropna(subset=["par"]).copy()
    r["ref"] = [ref.get((s, d)) for s, d in zip(r.sid, r.date)]
    r = r.dropna(subset=["ref"])
    r["gap"] = r.price - r.ref                       # buys: + is overpaying
    r.loc[r.side == "P", "gap"] = r.ref - r.price    # sells: + is underselling

    print("=" * 78)
    print("EXECUTION PENALTY VS THE PRIOR MARK, BY LOT SIZE")
    print("=" * 78)
    print(f"  {'lot size':<16}{'trades':>10}{'median':>10}{'mean':>9}"
          f"{'>0.25 cap':>11}{'bonds':>8}{'t':>8}")
    for side, name in (("S", "CUSTOMER BUYS"), ("P", "CUSTOMER SELLS")):
        print(f"\n  {name}")
        for lo, hi in BUCKETS:
            b = r[(r.side == side) & (r.par >= lo) & (r.par < hi)]
            if len(b) < 200:
                print(f"  {label(lo,hi):<16}{len(b):>10,}   too few")
                continue
            per = b.groupby("sid").gap.mean()
            tstat = (per.mean() / (st.stdev(per) / len(per) ** 0.5)
                     if len(per) > 30 else float("nan"))
            print(f"  {label(lo,hi):<16}{len(b):>10,}{b.gap.median():>+10.3f}"
                  f"{b.gap.mean():>+9.3f}"
                  f"{100*(b.gap>S.LIMIT_CAP).mean():>10.1f}%"
                  f"{len(per):>8,}{tstat:>+8.1f}")

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    small = r[(r.side == "S") & (r.par < 100_000)].gap.median()
    big = r[(r.side == "S") & (r.par >= 1_000_000)].gap.median()
    print(f"  An odd-lot buyer (under $100k) pays {small:+.3f} pts over the "
          f"prior mark;")
    print(f"  a $1m+ block buyer pays {big:+.3f}.")
    if small > big:
        print(f"  The penalty is {small - big:+.3f} pts larger at the small end —")
        print("  the tool is worth most to whoever trades in odd lots, and")
        print("  worth nearly nothing to a block buyer who already gets the mid.")
    else:
        print("  The penalty does NOT shrink with size — the story that big")
        print("  buyers execute better is not visible in this tape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
