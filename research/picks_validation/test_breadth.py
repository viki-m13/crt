#!/usr/bin/env python3
"""Was the picks idea failing on the signal, or on the portfolio?

The signal survey found quality_x_cheap has rank-IC +0.047 at twelve months
with t=+3.00 on independent periods, stable dev-to-holdout (+0.0476 vs
+0.0467) and still positive in the 2020s. But the earlier portfolio test of
the SAME idea came in at t=+1.91.

Those are not contradictory. IC measures the whole cross-section; that test
bought only ~14 names a month. A 14-name portfolio is mostly idiosyncratic
noise, so a real but modest signal can be genuine in the cross-section and
still invisible in a concentrated list — which is a portfolio-construction
problem, not a signal problem.

So: hold the signal fixed and vary only the number of names held. If breadth
is the binding constraint, excess return should be similar across widths
while its t-statistic climbs as idiosyncratic noise averages out. If the
signal is not real, nothing improves and the whole idea is finished.

Also tests pct_above_sma, the strongest single signal in the survey
(IC +0.0747, t=+2.76), which no previous test tried on its own.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "experiments", "monthly_dca", "cache")
PIT = os.path.join(CACHE, "v2", "sp500_pit")
HOLD = 252


def load():
    p = os.path.join(PIT, "prices_extended_pit.parquet")
    if not os.path.exists(p):
        p = os.path.join(CACHE, "prices_extended.parquet")
    px = pd.read_parquet(p).sort_index()
    px.index = pd.to_datetime(px.index)
    mem = pd.read_parquet(os.path.join(PIT, "sp500_membership_monthly.parquet"))
    mem["asof"] = pd.to_datetime(mem["asof"])
    return px, mem


def make_signals(hist: pd.DataFrame) -> pd.DataFrame:
    last = hist.iloc[-1]
    r = hist.pct_change()
    sma200 = hist.rolling(200).mean()
    pct_above = (hist.iloc[-756:] > sma200.iloc[-756:]).mean()
    downside = r.iloc[-252:].clip(upper=0).std() * np.sqrt(252)
    long_run = last / hist.iloc[-1260] - 1.0
    hi = hist.iloc[-252:].max()
    dd = last / hi - 1.0

    df = pd.DataFrame({"pct_above_sma": pct_above, "downside": downside,
                       "long_run": long_run, "dd": dd}).dropna()
    if df.empty:
        return df
    q = (df.pct_above_sma.rank(pct=True) * 0.45
         + (1 - df.downside.rank(pct=True)) * 0.30
         + df.long_run.rank(pct=True) * 0.25)
    df["quality"] = q
    df["quality_x_cheap"] = q * (1 - df.dd.rank(pct=True))
    return df


def main():
    px, mem = load()
    months = sorted(mem["asof"].unique())
    recs = []
    for asof in months:
        asof = pd.Timestamp(asof)
        if asof < px.index[0] or asof > px.index[-1] - pd.Timedelta(days=400):
            continue
        uni = [t for t in mem.loc[mem["asof"] == asof, "ticker"].unique()
               if t in px.columns]
        if len(uni) < 100:
            continue
        hist = px[uni].loc[:asof]
        if len(hist) < 1300:
            continue
        sig = make_signals(hist)
        if len(sig) < 100:
            continue
        fut = px[uni].loc[px.index > asof].iloc[:HOLD]
        if len(fut) < HOLD * 0.8:
            continue
        fwd = (fut.ffill().iloc[-1] / hist.iloc[-1] - 1.0)
        fwd = fwd.replace([np.inf, -np.inf], np.nan)
        fwd = fwd[(fwd > -0.995) & (fwd < 9.0)]
        sig = sig.join(fwd.rename("fwd"), how="inner").dropna(subset=["fwd"])
        if len(sig) < 100:
            continue

        base = sig.fwd.median()
        row = {"month": asof, "base": base, "n_uni": len(sig)}
        for name in ("quality_x_cheap", "pct_above_sma", "quality"):
            ranked = sig.sort_values(name, ascending=False)
            for k in (10, 25, 50, 100, 150):
                if len(ranked) >= k:
                    row[f"{name}_{k}"] = ranked.head(k).fwd.median() - base
        recs.append(row)

    R = pd.DataFrame(recs)
    if R.empty:
        print("no data"); return 1
    R["year"] = R.month.dt.year

    print("=" * 94)
    print("EXCESS RETURN vs the equal-weight universe, BY NUMBER OF NAMES HELD")
    print("  12-month holds. t is on non-overlapping annual cohorts.")
    print("=" * 94)
    print(f"  {'signal':<18} {'names':>6} {'excess':>9} {'t(yr)':>7} "
          f"{'dev':>8} {'hold':>8} {'2020s':>8} {'yrs+':>6}")
    best = []
    for name in ("quality_x_cheap", "pct_above_sma", "quality"):
        for k in (10, 25, 50, 100, 150):
            col = f"{name}_{k}"
            if col not in R:
                continue
            s = R[col].dropna()
            if len(s) < 40:
                continue
            ann = R.dropna(subset=[col]).groupby("year")[col].mean()
            t = ann.mean() / (ann.std() / np.sqrt(len(ann))) if len(ann) > 3 else np.nan
            dev = R[R.month < "2017-01-01"][col].mean()
            hold = R[R.month >= "2017-01-01"][col].mean()
            d20 = R[R.month >= "2020-01-01"][col].mean()
            print(f"  {name:<18} {k:>6} {s.mean():>+9.2%} {t:>+7.2f} "
                  f"{dev:>+8.2%} {hold:>+8.2%} {d20:>+8.2%} "
                  f"{100*(ann>0).mean():>5.0f}%")
            best.append((name, k, s.mean(), t, dev, hold, d20))
        print()

    print("=" * 94)
    print("VERDICT (criteria fixed before the run)")
    print("=" * 94)
    ok = [b for b in best if b[3] > 2.0 and b[2] > 0.01 and b[5] > 0 and b[6] > 0]
    if not ok:
        print("  NOTHING clears t>2 with >1% excess, positive holdout AND")
        print("  positive 2020s. The idea does not survive.")
    else:
        print("  Configurations clearing t>2, >1% excess, positive holdout and")
        print("  a positive 2020s:\n")
        for name, k, ex, t, dev, hold, d20 in sorted(ok, key=lambda b: -b[3]):
            print(f"    {name} holding {k} names: {ex:+.2%}/yr excess, t={t:+.2f}, "
                  f"holdout {hold:+.2%}, 2020s {d20:+.2%}")
        print("\n  Note what this does NOT include: trading costs, taxes, or the")
        print("  tracking error of holding a portfolio that differs this much")
        print("  from the index. It is a signal result, not a product claim.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breadth.csv")
    R.to_csv(out, index=False)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
