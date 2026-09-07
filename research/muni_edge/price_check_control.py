#!/usr/bin/env python3
"""Is the pre-trade price check measuring a dealer markup, or just drift?

product_shape.py found the check would tell an advisor "too rich" on 58% of
real retail-size buys, median 1.13 points over. That number is worthless on
its own, because the limit is set off the PRIOR day's mid — so in any rising
market a fair buy sits above it mechanically, and the check would look
valuable while measuring nothing but the trend.

The control is symmetry. If the gap is a dealer spread, customers are
penalised on BOTH sides: buys print above the prior mid AND sells print below
it, by similar amounts. If it is drift, buys print above and sells print
above too, and the two numbers do not mirror.

Also reported: the raw drift of the mid over the same window, which is the
size of the effect the symmetry test has to overcome.

    python research/muni_edge/price_check_control.py [--days 250]
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "api"))

import _munisignal as S  # noqa: E402

TAPE = "/home/user/viki-m13/bonds/munis/data/trades"
CACHE = os.path.join(HERE, ".tape_cache.pkl.gz")
RETAIL_MAX_PAR = 100_000


def build_cache() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(TAPE, "*.csv.gz")))
    print(f"building cache from {len(paths):,} bonds (one time)")
    frames = []
    for i, p in enumerate(paths, 1):
        try:
            d = pd.read_csv(p, usecols=lambda c: c in ("ts", "price", "par", "side"))
        except Exception:  # noqa: BLE001
            continue
        if not {"ts", "price", "side"} <= set(d.columns):
            continue
        d["date"] = pd.to_datetime(d.ts, errors="coerce").dt.date
        d["price"] = pd.to_numeric(d.price, errors="coerce")
        d["par"] = pd.to_numeric(d.get("par"), errors="coerce")
        d = d.dropna(subset=["date", "price"])
        d = d[(d.price > 20) & (d.price < 200)]
        if d.empty:
            continue
        d["sid"] = os.path.basename(p).split(".")[0]
        frames.append(d[["sid", "date", "price", "par", "side"]])
        if i % 800 == 0:
            print(f"  …{i:,}/{len(paths):,}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out.date.astype(str)
    out.to_pickle(CACHE)
    return out


def load() -> pd.DataFrame:
    if os.path.exists(CACHE):
        print("using cached tape")
        return pd.read_pickle(CACHE)
    return build_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    a = ap.parse_args()

    t = load()
    print(f"  {len(t):,} prints, {t.sid.nunique():,} bonds")

    # ---- daily mid per bond, exactly as the scanner builds it -------------
    g = t.groupby(["sid", "date", "side"]).price.median().unstack("side")
    for col in ("S", "P", "D"):
        if col not in g.columns:
            g[col] = float("nan")
    g = g.reset_index()
    mid = []
    for r in g.itertuples():
        mid.append(S.DayPrints(date=r.date, buy=None if pd.isna(r.S) else float(r.S),
                               sell=None if pd.isna(r.P) else float(r.P),
                               dealer=None if pd.isna(r.D) else float(r.D)).mid)
    g["mid"] = mid
    g = g.dropna(subset=["mid"]).sort_values(["sid", "date"])

    # the PRIOR mid is the mark the limit is set off — shift(1) within bond
    g["prior_mid"] = g.groupby("sid")["mid"].shift(1)
    g = g.dropna(subset=["prior_mid"])

    days = sorted(g.date.unique())[-a.days:]
    g = g[g.date.isin(days)]
    print(f"  window {days[0]} -> {days[-1]} ({len(days)} days)\n")

    ref = {(r.sid, r.date): r.prior_mid for r in g.itertuples()}

    # ---- the drift the control has to beat -------------------------------
    drift = (g["mid"] - g["prior_mid"]).dropna()
    print("=" * 78)
    print("HOW MUCH DID PRICES DRIFT IN THIS WINDOW?")
    print("=" * 78)
    print(f"  median day-over-day change in mid: {drift.median():+.4f} pts")
    print(f"  mean:                              {drift.mean():+.4f} pts")
    print("  (a check set at prior_mid + 0.25 fires on drift alone if this "
          "is large)\n")

    # ---- the symmetry test ------------------------------------------------
    r = t[t.par <= RETAIL_MAX_PAR].copy()
    r = r[r.date.isin(days)]
    r["ref"] = [ref.get((s, d)) for s, d in zip(r.sid, r.date)]
    r = r.dropna(subset=["ref"])

    buys = r[r.side == "S"]
    sells = r[r.side == "P"]
    b_gap = (buys.price - buys.ref)      # + means customer paid ABOVE the mark
    s_gap = (sells.ref - sells.price)    # + means customer sold BELOW the mark

    print("=" * 78)
    print("SYMMETRY: ARE CUSTOMERS PENALISED ON BOTH SIDES?")
    print("=" * 78)
    print(f"  {'':22} {'buys':>14} {'sells':>14}")
    print(f"  {'trades':22} {len(buys):>14,} {len(sells):>14,}")
    print(f"  {'median penalty (pts)':22} {b_gap.median():>+14.3f} "
          f"{s_gap.median():>+14.3f}")
    print(f"  {'mean penalty (pts)':22} {b_gap.mean():>+14.3f} "
          f"{s_gap.mean():>+14.3f}")
    print(f"  {'% worse than mark':22} {100*(b_gap>0).mean():>13.1f}% "
          f"{100*(s_gap>0).mean():>13.1f}%")
    print(f"  {'% past the 0.25 cap':22} "
          f"{100*(b_gap>S.LIMIT_CAP).mean():>13.1f}% "
          f"{100*(s_gap>S.LIMIT_CAP).mean():>13.1f}%")

    sym = min(b_gap.median(), s_gap.median()) / max(
        abs(b_gap.median()), abs(s_gap.median()), 1e-9)
    print()
    if b_gap.median() > 0 and s_gap.median() > 0:
        print(f"  BOTH SIDES ARE PENALISED (symmetry ratio {sym:.2f}).")
        print("  A markup, not a trend: drift can only push one side.")
    else:
        print("  ONE-SIDED. This is drift, not a spread — the check is not")
        print("  measuring what it claims to.")

    # ---- clustered t-stat, one observation per bond -----------------------
    print()
    print("=" * 78)
    print("CLUSTERED BY BOND (a bond that trades 500 times is one vote)")
    print("=" * 78)
    for label, gap, key in (("buys", b_gap, buys), ("sells", s_gap, sells)):
        per = pd.Series(gap.values, index=key.sid.values).groupby(level=0).mean()
        n = len(per)
        if n < 30:
            print(f"  {label}: too few bonds")
            continue
        se = st.stdev(per) / (n ** 0.5)
        print(f"  {label:6} {n:,} bonds   mean {per.mean():+.3f} pts   "
              f"t = {per.mean()/se:+.1f}")

    # ---- what the advisor keeps ------------------------------------------
    over = b_gap[b_gap > S.LIMIT_CAP] - S.LIMIT_CAP
    par = buys.par[b_gap > S.LIMIT_CAP].fillna(RETAIL_MAX_PAR)
    dollars = (over / 100.0 * par).dropna()
    print()
    print("=" * 78)
    print("WHAT A 'TOO RICH' VERDICT IS WORTH ON ONE TRADE")
    print("=" * 78)
    print(f"  median ${dollars.median():,.0f}   "
          f"mean ${dollars.mean():,.0f}   "
          f"90th pct ${dollars.quantile(.9):,.0f}")
    print(f"  median lot size ${par.median():,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
