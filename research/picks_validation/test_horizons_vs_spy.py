#!/usr/bin/env python3
"""Does the screen beat SPY, and does a longer horizon help?

Reframed from the earlier tests, which asked the wrong two questions. They
compared against the EQUAL-WEIGHT universe, but a buyer of this product would
otherwise own SPY, so SPY is the benchmark that matters. And they stopped at
twelve months, when the claim being made is about quality compounding — which
is a multi-year proposition.

So this measures, from point-in-time S&P membership:

  1. Forward returns at 1, 2, 3 and 5 years against BOTH benchmarks — the
     equal-weight universe (does the screen beat the average stock?) and SPY
     (does it beat what the customer already owns?). These differ a lot: SPY
     is cap-weighted, and cap-weight beat equal-weight heavily in the 2020s,
     so an equal-weight comparison flatters any screen in that period.

  2. An actual compounded portfolio — buy the screen, hold, rebalance
     annually, track the equity curve against SPY. A per-window average
     return is not what an investor gets; the compounded path is.

  3. Face validity — the names the screen actually produces. A screen sold as
     "high quality companies that keep going up and are currently cheap" has
     to visibly name that kind of company. If the list is unrecognisable
     junk, no statistic rescues it.

HONESTY NOTE ON SIGNIFICANCE: with membership from 2003 and prices to 2026, a
five-year horizon leaves roughly four NON-OVERLAPPING periods. That is far too
few for a meaningful t-statistic, and one is not quoted for the long horizons.
Consistency across formation months and across regimes is the evidence
available at those horizons; it is weaker evidence, and is labelled as such.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "experiments", "monthly_dca", "cache")
PIT = os.path.join(CACHE, "v2", "sp500_pit")
HORIZONS = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}
N_HOLD = 25


def load():
    px = pd.read_parquet(os.path.join(PIT, "prices_extended_pit.parquet")).sort_index()
    px.index = pd.to_datetime(px.index)
    base = pd.read_parquet(os.path.join(CACHE, "prices_extended.parquet"))
    base.index = pd.to_datetime(base.index)
    spy = base["SPY"].dropna() if "SPY" in base.columns else px["SPY"].dropna()
    mem = pd.read_parquet(os.path.join(PIT, "sp500_membership_monthly.parquet"))
    mem["asof"] = pd.to_datetime(mem["asof"])
    return px, spy, mem


def screen(hist: pd.DataFrame) -> pd.DataFrame:
    """High quality, still trending, currently below its high."""
    last = hist.iloc[-1]
    r = hist.pct_change()
    sma200 = hist.rolling(200).mean()
    pct_above = (hist.iloc[-756:] > sma200.iloc[-756:]).mean()
    downside = r.iloc[-252:].clip(upper=0).std() * np.sqrt(252)
    long_run = last / hist.iloc[-1260] - 1.0
    dd = last / hist.iloc[-252:].max() - 1.0
    df = pd.DataFrame({"pct_above": pct_above, "downside": downside,
                       "long_run": long_run, "dd": dd}).dropna()
    if df.empty:
        return df
    df["quality"] = (df.pct_above.rank(pct=True) * 0.45
                     + (1 - df.downside.rank(pct=True)) * 0.30
                     + df.long_run.rank(pct=True) * 0.25)
    df["score"] = df.quality * (1 - df.dd.rank(pct=True))
    return df


def fwd(px: pd.DataFrame, asof, tickers, days):
    fut = px.loc[px.index > asof]
    if len(fut) < days * 0.9:
        return None
    w = fut.iloc[:days].ffill().iloc[-1]
    start = px.loc[:asof].iloc[-1]
    out = (w[tickers] / start[tickers] - 1.0).replace([np.inf, -np.inf], np.nan)
    return out[(out > -0.995) & (out < 30.0)].dropna()


def spy_fwd(spy: pd.Series, asof, days):
    fut = spy.loc[spy.index > asof]
    if len(fut) < days * 0.9:
        return None
    return float(fut.iloc[:days].iloc[-1] / spy.loc[:asof].iloc[-1] - 1.0)


def main():
    px, spy, mem = load()
    months = sorted(mem["asof"].unique())
    recs, examples = [], []

    for asof in months:
        asof = pd.Timestamp(asof)
        if asof < px.index[0] or asof > px.index[-1]:
            continue
        uni = [t for t in mem.loc[mem["asof"] == asof, "ticker"].unique()
               if t in px.columns]
        if len(uni) < 100:
            continue
        hist = px[uni].loc[:asof]
        if len(hist) < 1300:
            continue
        s = screen(hist)
        if len(s) < 100:
            continue
        picks = s.sort_values("score", ascending=False).head(N_HOLD).index.tolist()
        if asof.month == 6 and asof.year in (2008, 2015, 2021):
            examples.append((asof, picks[:12]))

        row = {"month": asof}
        for hl, hd in HORIZONS.items():
            f = fwd(px[uni], asof, s.index, hd)
            sp = spy_fwd(spy, asof, hd)
            if f is None or sp is None:
                continue
            pk = f.reindex(picks).dropna()
            if len(pk) < N_HOLD * 0.6:
                continue
            row[f"pick_{hl}"] = pk.median()
            row[f"univ_{hl}"] = f.median()
            row[f"spy_{hl}"] = sp
            row[f"vs_univ_{hl}"] = pk.median() - f.median()
            row[f"vs_spy_{hl}"] = pk.median() - sp
            row[f"beat_spy_{hl}"] = 1.0 if pk.median() > sp else 0.0
        recs.append(row)

    R = pd.DataFrame(recs)
    print("=" * 96)
    print(f"FORWARD RETURNS BY HORIZON — screen of {N_HOLD} names, "
          f"{len(R)} formation months")
    print("=" * 96)
    print(f"  {'horizon':<9} {'n':>5} {'picks':>9} {'universe':>10} {'SPY':>9} "
          f"{'vs univ':>9} {'vs SPY':>9} {'beat SPY':>9}")
    for hl in HORIZONS:
        c = f"pick_{hl}"
        if c not in R:
            continue
        g = R.dropna(subset=[c])
        if len(g) < 12:
            continue
        print(f"  {hl:<9} {len(g):>5} {g[c].median():>+9.1%} "
              f"{g[f'univ_{hl}'].median():>+10.1%} {g[f'spy_{hl}'].median():>+9.1%} "
              f"{g[f'vs_univ_{hl}'].median():>+9.1%} "
              f"{g[f'vs_spy_{hl}'].median():>+9.1%} "
              f"{g[f'beat_spy_{hl}'].mean():>8.0%}")

    print("\n" + "=" * 96)
    print("DOES A LONGER HORIZON HELP? (share of formation months beating SPY)")
    print("=" * 96)
    for hl in HORIZONS:
        c = f"beat_spy_{hl}"
        if c in R and R[c].notna().sum() > 12:
            g = R.dropna(subset=[c])
            dev = g[g.month < "2017-01-01"][c].mean()
            hold = g[g.month >= "2017-01-01"][c].mean()
            print(f"  {hl:<4} beat SPY on {g[c].mean():>5.0%} of months  "
                  f"(pre-2017 {dev:.0%}, 2017+ "
                  f"{'n/a' if np.isnan(hold) else f'{hold:.0%}'})  n={len(g)}")

    # ---------------------------------------------------------------- equity
    print("\n" + "=" * 96)
    print("COMPOUNDED: buy the screen each June, hold 12 months, repeat")
    print("  This is the path an investor actually experiences.")
    print("=" * 96)
    eq, eq_spy, yrs = 1.0, 1.0, []
    for asof in [pd.Timestamp(m) for m in months]:
        if asof.month != 6:
            continue
        uni = [t for t in mem.loc[mem["asof"] == asof, "ticker"].unique()
               if t in px.columns]
        if len(uni) < 100:
            continue
        hist = px[uni].loc[:asof]
        if len(hist) < 1300:
            continue
        s = screen(hist)
        if len(s) < 100:
            continue
        picks = s.sort_values("score", ascending=False).head(N_HOLD).index.tolist()
        f = fwd(px[uni], asof, picks, 252)
        sp = spy_fwd(spy, asof, 252)
        if f is None or sp is None or len(f) < N_HOLD * 0.6:
            continue
        r = float(f.mean())            # equal-weighted across held names
        eq *= (1 + r)
        eq_spy *= (1 + sp)
        yrs.append((asof.year, r, sp))
    if yrs:
        print(f"  {'year':<6} {'screen':>9} {'SPY':>9} {'diff':>9}")
        for y, r, sp in yrs:
            print(f"  {y:<6} {r:>+9.1%} {sp:>+9.1%} {r-sp:>+9.1%}")
        n = len(yrs)
        print(f"\n  compounded over {n} years: screen {eq:.2f}x  vs  SPY {eq_spy:.2f}x")
        print(f"  CAGR: screen {eq**(1/n)-1:+.2%}  SPY {eq_spy**(1/n)-1:+.2%}")
        wins = sum(1 for _, r, sp in yrs if r > sp)
        print(f"  beat SPY in {wins}/{n} years ({wins/n:.0%})")

    print("\n" + "=" * 96)
    print("FACE VALIDITY — what the screen actually named")
    print("=" * 96)
    for asof, names in examples:
        print(f"  {asof.date()}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
