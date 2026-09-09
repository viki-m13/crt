#!/usr/bin/env python3
"""If we tell an advisor a quote is too rich, can they do anything about it?

This is the question that decides whether the price check is a product or a
statistic. We measured that 58% of retail-size muni buys print above the
prior mark by more than the cap. That gap is only worth money if the advisor
has an alternative — walk away, and buy the same bond cheaper soon after.
Munis are idiosyncratic: if the bond never prints again, "too rich" is
information the advisor cannot use, and the honest saving is zero.

So for every flagged buy, look forward and ask what actually happened in
that bond:
  - did ANY customer buy print again within the window?
  - was one of them cheaper than the flagged price?
  - was it cheaper by more than the excess we flagged?

Reported against the full flagged denominator, not just the bonds that
happened to trade again — conditioning on a later print is the selection
bias that would make this look far better than it is.

    python research/muni_edge/can_they_act.py [--window 10]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "api"))

import _munisignal as S  # noqa: E402

CACHE = os.path.join(HERE, ".tape_cache.pkl.gz")
RETAIL_MAX_PAR = 100_000
DAYS = 250


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10,
                    help="trading days the advisor is willing to wait")
    a = ap.parse_args()

    if not os.path.exists(CACHE):
        print("run price_check_control.py first to build the cache")
        return 1
    t = pd.read_pickle(CACHE)

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
    day_ix = {d: i for i, d in enumerate(days)}
    ref = {(r.sid, r.date): r.prior_mid
           for r in g[g.date.isin(days)].itertuples()}

    # every retail-size customer buy in the window, with our verdict
    r = t[(t.side == "S") & (t.par <= RETAIL_MAX_PAR)
          & t.date.isin(days)].copy()
    r["ref"] = [ref.get((s, d)) for s, d in zip(r.sid, r.date)]
    r = r.dropna(subset=["ref"])
    r["excess"] = r.price - r.ref - S.LIMIT_CAP
    flagged = r[r.excess > 0].copy()
    print(f"flagged buys: {len(flagged):,} of {len(r):,} "
          f"({100*len(flagged)/len(r):.1f}%)")
    print(f"window: the next {a.window} trading days\n")

    # forward customer buys per bond, indexed by day position
    buys = r[["sid", "date", "price"]].copy()
    buys["i"] = buys.date.map(day_ix)
    by_bond: dict[str, list[tuple[int, float]]] = {}
    for row in buys.sort_values("i").itertuples():
        by_bond.setdefault(row.sid, []).append((row.i, row.price))

    n = len(flagged)
    printed_again = cheaper = beat_excess = 0
    realised = []
    # The honest version. Taking the MINIMUM of the next ten days is
    # look-ahead — nobody knows in advance which day is cheapest — so the
    # headline uses the NEXT print, which is what an advisor who declines the
    # quote and waits actually gets. The best-of-window figure is kept
    # alongside it as the unattainable ceiling.
    nx_cheaper = nx_beat = 0
    nx_realised = []
    for row in flagged.itertuples():
        i0 = day_ix[row.date]
        seq = by_bond.get(row.sid, ())
        later = [(i, p) for i, p in seq if i0 < i <= i0 + a.window]
        if not later:
            continue
        printed_again += 1

        nxt = min(i for i, _ in later)
        nxt_px = min(p for i, p in later if i == nxt)   # worst-case same-day
        if nxt_px < row.price:
            nx_cheaper += 1
            gain = row.price - nxt_px
            nx_realised.append(gain)
            if gain > row.excess:
                nx_beat += 1

        best = min(p for _, p in later)
        if best < row.price:
            cheaper += 1
            gain = row.price - best
            realised.append(gain)
            if gain > row.excess:
                beat_excess += 1

    def pct(x):
        return f"{100*x/max(n,1):>5.1f}%"

    print("=" * 78)
    print("WHAT AN ADVISOR WHO WALKED AWAY WOULD HAVE FOUND")
    print("=" * 78)
    print(f"  {'flagged buys':<44}{n:>10,}")
    print(f"  {'the bond printed a customer buy again':<44}"
          f"{printed_again:>10,}  {pct(printed_again)}")
    print("\n  ATTAINABLE — take the NEXT print, no foresight required")
    print(f"  {'  it was cheaper':<44}{nx_cheaper:>10,}  {pct(nx_cheaper)}")
    print(f"  {'  cheaper by MORE than we flagged':<44}{nx_beat:>10,}  "
          f"{pct(nx_beat)}")
    if nx_realised:
        s = pd.Series(nx_realised)
        print(f"  {'  median improvement':<44}{s.median():>10.3f} pts"
              f"  = ${s.median()*250:,.0f} on $25k")

    print("\n  CEILING — best of the window, needs foresight, not attainable")
    print(f"  {'  one of them was cheaper':<44}{cheaper:>10,}  {pct(cheaper)}")
    print(f"  {'  cheaper by MORE than we flagged':<44}"
          f"{beat_excess:>10,}  {pct(beat_excess)}")
    if realised:
        s = pd.Series(realised)
        print(f"  {'  median improvement':<44}{s.median():>10.3f} pts")

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    hit = 100 * nx_beat / max(n, 1)
    print(f"  Walking away recovers more than the flagged excess on "
          f"{hit:.0f}% of")
    print(f"  flagged trades. On the other {100-hit:.0f}% the advisor either "
          f"never sees the")
    print(f"  bond again ({100*(n-printed_again)/max(n,1):.0f}%) or it does "
          f"not come back cheaper enough.")
    print("  Note this ignores the cost of not owning the bond meanwhile, and")
    print("  assumes the later print was available to THIS buyer, which the")
    print("  tape cannot confirm. Both push the true figure lower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
