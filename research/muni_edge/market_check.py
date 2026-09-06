#!/usr/bin/env python3
"""Two market questions, answered from the MSRB tape rather than from opinion.

Q1. IS DIRECT RETAIL MUNI BUYING DECLINING? The SMA growth story implies
    advisors are handing muni buying to institutional managers. If so, the
    retail-size share of trading should be falling year over year. If it is
    flat, the addressable market is not disappearing and the product thesis
    survives.

Q2. DO INSTITUTIONS ALREADY CAPTURE THE DISLOCATIONS? If institutional-size
    buyers are already the ones buying bonds that print far below trend,
    then the signal is not news to them and an SMA manager will not pay for
    it. If retail is doing the buying while institutions sit out, the edge
    is unharvested — which is the only condition under which selling it to
    institutions makes sense.

Both are measured on the same trades: retail <= $100k par, institutional
>= $1m, with the gap between them left out of the comparison.
"""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

TAPE = "/home/user/viki-m13/bonds/munis/data/trades"
RETAIL_MAX = 100_000
INSTIT_MIN = 1_000_000
DISLOCATION_PTS = 3.0


def main():
    files = sorted(glob.glob(os.path.join(TAPE, "*.csv.gz")))
    print(f"scanning {len(files):,} bonds\n")

    yr_cnt = defaultdict(lambda: defaultdict(int))     # year -> bucket -> trades
    yr_par = defaultdict(lambda: defaultdict(float))
    disloc_buy = {"retail": 0, "instit": 0}            # who buys dislocations
    normal_buy = {"retail": 0, "instit": 0}
    disloc_par = {"retail": 0.0, "instit": 0.0}

    for n, f in enumerate(files, 1):
        try:
            d = pd.read_csv(f, usecols=["ts", "price", "par", "side"])
        except Exception:
            continue
        d["date"] = pd.to_datetime(d.ts, errors="coerce")
        d = d.dropna(subset=["date", "price", "par"])
        d = d[(d.price > 20) & (d.price < 250) & (d.par > 0)]
        if d.empty:
            continue
        d["year"] = d.date.dt.year
        d["day"] = d.date.dt.strftime("%Y-%m-%d")

        # --- Q1: retail vs institutional share of trading, by year
        for yr, g in d.groupby("year"):
            r = g[g.par <= RETAIL_MAX]
            i = g[g.par >= INSTIT_MIN]
            yr_cnt[yr]["retail"] += len(r)
            yr_cnt[yr]["instit"] += len(i)
            yr_par[yr]["retail"] += float(r.par.sum())
            yr_par[yr]["instit"] += float(i.par.sum())

        # --- Q2: who buys when a bond prints far below its own trend
        day = d.groupby("day", as_index=False).agg(
            px=("price", "median"))
        day = day.sort_values("day")
        day["trend"] = day.px.rolling(30, min_periods=5).median().shift(1)
        trend = dict(zip(day.day, day.trend))
        buys = d[d.side == "S"]
        for row in buys.itertuples(index=False):
            t = trend.get(row.day)
            if t is None or not np.isfinite(t):
                continue
            bucket = ("retail" if row.par <= RETAIL_MAX
                      else "instit" if row.par >= INSTIT_MIN else None)
            if bucket is None:
                continue
            if (t - row.price) >= DISLOCATION_PTS:
                disloc_buy[bucket] += 1
                disloc_par[bucket] += float(row.par)
            else:
                normal_buy[bucket] += 1
        if n % 800 == 0:
            print(f"  …{n:,}/{len(files):,}", flush=True)

    print("\n" + "=" * 78)
    print("Q1. IS RETAIL-SIZE MUNI TRADING DECLINING?")
    print("=" * 78)
    print(f"  {'year':<6} {'retail trades':>14} {'share of trades':>16} "
          f"{'retail par $m':>14} {'share of par':>13}")
    years = sorted(y for y in yr_cnt if 2013 <= y <= 2026)
    shares = []
    for y in years:
        rc, ic = yr_cnt[y]["retail"], yr_cnt[y]["instit"]
        rp, ip = yr_par[y]["retail"], yr_par[y]["instit"]
        if rc + ic < 500:
            continue
        s_cnt = rc / (rc + ic)
        s_par = rp / (rp + ip) if (rp + ip) else float("nan")
        shares.append((y, s_cnt))
        print(f"  {y:<6} {rc:>14,} {s_cnt:>15.1%} {rp/1e6:>14,.0f} {s_par:>12.1%}")
    if len(shares) >= 6:
        ys = np.array([s[0] for s in shares], float)
        vs = np.array([s[1] for s in shares], float)
        slope = np.polyfit(ys, vs, 1)[0]
        first3 = vs[:3].mean()
        last3 = vs[-3:].mean()
        print(f"\n  trend in retail share of trades: {slope*100:+.2f} pts/year")
        print(f"  first 3 years {first3:.1%} -> last 3 years {last3:.1%} "
              f"({(last3-first3)*100:+.1f} pts)")
        if abs(last3 - first3) < 0.05:
            print("  -> STABLE. Direct retail-size muni trading is not "
                  "disappearing.")
        elif last3 < first3:
            print("  -> DECLINING. The addressable segment is shrinking, as "
                  "the SMA story implies.")
        else:
            print("  -> GROWING.")

    print("\n" + "=" * 78)
    print("Q2. WHO BUYS THE DISLOCATIONS?")
    print("=" * 78)
    dr, di = disloc_buy["retail"], disloc_buy["instit"]
    nr, ni = normal_buy["retail"], normal_buy["instit"]
    if dr + di:
        print(f"  buys >= {DISLOCATION_PTS:.0f} pts below trend:  "
              f"retail {dr:>7,} ({dr/(dr+di):.1%})   "
              f"institutional {di:>7,} ({di/(dr+di):.1%})")
    if nr + ni:
        print(f"  all other buys:                 "
              f"retail {nr:>7,} ({nr/(nr+ni):.1%})   "
              f"institutional {ni:>7,} ({ni/(nr+ni):.1%})")
    if (dr + di) and (nr + ni):
        d_share = di / (dr + di)
        n_share = ni / (nr + ni)
        print(f"\n  institutional share of dislocation buys: {d_share:.1%}")
        print(f"  institutional share of ordinary buys:    {n_share:.1%}")
        if d_share > n_share + 0.05:
            print("  -> Institutions LEAN IN to dislocations. The edge is")
            print("     already being harvested; an SMA manager does not need")
            print("     to be told about it.")
        elif d_share < n_share - 0.05:
            print("  -> Institutions STEP BACK from dislocations while retail")
            print("     buys them. The edge is unharvested by the people best")
            print("     placed to take it — which is the condition under which")
            print("     selling them the signal could make sense.")
        else:
            print("  -> No meaningful difference: institutions neither chase")
            print("     nor avoid dislocations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
