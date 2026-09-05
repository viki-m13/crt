#!/usr/bin/env python3
"""Does retail systematically overpay for municipal bonds? Measure it.

This is the business-validation test, not a trading study. The proposed
product tells a wealth manager what their clients' muni fills actually cost
versus the tape. That claim is only worth selling if the penalty is real,
large, and measurable from data we already hold — so measure it before
building anything.

Method, chosen to be conservative and hard to argue with:

  * Same bond, same day, same side. Comparing a retail buy to an
    institutional buy of THE SAME security on THE SAME day removes bond
    quality, coupon, maturity, credit and the day's rate move. What is left
    is size.
  * Customer buys only (EMMA side 'S' = dealer sold to a customer). Inter-
    dealer prints are not available to a customer and are excluded.
  * Par-weighted prices within each size bucket, so one tiny print cannot
    swing a day.
  * Retail is <= $100k par, institutional >= $1m par. The gap between the
    buckets is deliberate: it avoids arguing about the boundary.
  * Yield is reported alongside price. Price alone could mislead across
    bonds; a retail buyer receiving a LOWER yield for the same bond on the
    same day is unambiguously worse off.

Also measured: the same-day round trip (what a customer buys at versus what
another customer simultaneously sells at), which is the dealer spread a
client pays instantly and never sees on a statement.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

MUNIS = "/home/user/viki-m13/bonds/munis"
RETAIL_MAX = 100_000
INSTIT_MIN = 1_000_000


def pw(g: pd.DataFrame, col: str = "price") -> float:
    """Par-weighted average."""
    w = g["par"].to_numpy(dtype=float)
    v = g[col].to_numpy(dtype=float)
    m = np.isfinite(w) & np.isfinite(v) & (w > 0)
    if not m.any():
        return np.nan
    return float((v[m] * w[m]).sum() / w[m].sum())


def analyse_bond(path: str):
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    if not len(d) or "side" not in d.columns:
        return None
    d["date"] = pd.to_datetime(d.ts, errors="coerce").dt.date
    d = d.dropna(subset=["date", "price", "par"])
    d = d[(d.price > 20) & (d.price < 250) & (d.par > 0)]
    if not len(d):
        return None

    buys = d[d.side == "S"]
    sells = d[d.side == "P"]
    rows = []

    # ---- size penalty: retail vs institutional, same bond, same day, buys
    for day, g in buys.groupby("date"):
        r = g[g.par <= RETAIL_MAX]
        i = g[g.par >= INSTIT_MIN]
        if len(r) and len(i):
            pr, pi = pw(r), pw(i)
            yr = pw(r, "ytw") if "ytw" in g else np.nan
            yi = pw(i, "ytw") if "ytw" in g else np.nan
            if np.isfinite(pr) and np.isfinite(pi):
                rows.append({"kind": "size", "date": day,
                             "retail_px": pr, "instit_px": pi,
                             "px_penalty": pr - pi,
                             "retail_ytw": yr, "instit_ytw": yi,
                             "ytw_penalty": (yi - yr) if np.isfinite(yr) and np.isfinite(yi) else np.nan,
                             "retail_par": float(r.par.sum())})

    # ---- same-day round trip: customer buys at S, another sells at P
    rt = []
    for day, g in d.groupby("date"):
        b = g[(g.side == "S") & (g.par <= RETAIL_MAX)]
        s = g[(g.side == "P") & (g.par <= RETAIL_MAX)]
        if len(b) and len(s):
            pb, ps = pw(b), pw(s)
            if np.isfinite(pb) and np.isfinite(ps):
                rt.append({"date": day, "buy_px": pb, "sell_px": ps,
                           "spread": pb - ps})
    return rows, rt


def main():
    files = sorted(glob.glob(os.path.join(MUNIS, "data", "trades", "*.csv.gz")))
    if not files:
        print("no trade files found"); return 1
    print(f"scanning {len(files)} bonds from the EMMA tape…\n")

    size_rows, rt_rows, n_ok = [], [], 0
    for i, f in enumerate(files):
        out = analyse_bond(f)
        if not out:
            continue
        rows, rt = out
        if rows or rt:
            n_ok += 1
        size_rows.extend(rows)
        rt_rows.extend(rt)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} bonds…")

    S = pd.DataFrame(size_rows)
    R = pd.DataFrame(rt_rows)
    print(f"\nbonds contributing data: {n_ok:,}")
    print("=" * 74)
    print("1. THE SIZE PENALTY — same bond, same day, both buying")
    print("=" * 74)
    if not len(S):
        print("  no overlapping retail/institutional days found")
    else:
        p = S.px_penalty
        print(f"  comparable bond-days: {len(S):,}")
        print(f"  retail paid MORE on:  {100*(p>0).mean():.1f}% of them")
        print(f"  mean penalty:         {p.mean():+.3f} points of par "
              f"(${p.mean()*10:,.0f} per $100k)")
        print(f"  median penalty:       {p.median():+.3f} points "
              f"(${p.median()*10:,.0f} per $100k)")
        for q in (0.25, 0.5, 0.75, 0.90, 0.99):
            print(f"    {int(q*100):>2}th pct:          {p.quantile(q):+.3f} points "
                  f"(${p.quantile(q)*10:,.0f} per $100k)")
        y = S.ytw_penalty.dropna()
        if len(y):
            print(f"\n  yield given up by retail: {y.mean()*100:+.1f} bp mean, "
                  f"{y.median()*100:+.1f} bp median  (n={len(y):,})")
            print(f"  retail got a WORSE yield on {100*(y>0).mean():.1f}% of bond-days")
        # a t-stat on bond-day observations, clustered crudely by bond count
        t = p.mean() / (p.std() / np.sqrt(len(p)))
        print(f"\n  t-stat on the mean penalty: {t:+.1f}")

    print("\n" + "=" * 74)
    print("2. THE ROUND TRIP — what a retail customer pays to get in and out")
    print("=" * 74)
    if not len(R):
        print("  no same-day two-sided retail days found")
    else:
        s = R.spread
        print(f"  bond-days with retail on both sides: {len(R):,}")
        print(f"  mean spread:   {s.mean():+.3f} points  "
              f"(${s.mean()*10:,.0f} per $100k, instantly)")
        print(f"  median spread: {s.median():+.3f} points  "
              f"(${s.median()*10:,.0f} per $100k)")
        for q in (0.75, 0.90):
            print(f"    {int(q*100)}th pct:    {s.quantile(q):+.3f} points "
                  f"(${s.quantile(q)*10:,.0f} per $100k)")

    print("\n" + "=" * 74)
    print("3. WHAT THIS MEANS FOR A $2M MUNI LADDER")
    print("=" * 74)
    if len(S):
        med = S.px_penalty.median()
        print(f"  A $2,000,000 muni allocation bought in retail-size lots, at the")
        print(f"  MEDIAN observed size penalty, costs {med:.3f} points more than")
        print(f"  the same bonds bought institutionally the same day:")
        print(f"      ${med/100*2_000_000:,.0f} of client money, once.")
        if len(R):
            rtm = R.spread.median()
            print(f"  Selling it back at the median observed retail spread costs a")
            print(f"  further ${rtm/100*2_000_000:,.0f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
