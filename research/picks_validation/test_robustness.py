#!/usr/bin/env python3
"""Is the 14x-vs-6.8x result real, or an artifact of two choices?

The horizon test produced a contradiction that has to be resolved before
anyone quotes it. The MEDIAN pick loses to SPY at every horizon and loses by
more as the horizon lengthens. Yet an equal-weighted portfolio of the same
picks compounds to 14.0x against SPY's 6.8x over twenty years.

Both are arithmetically correct. An equal-weighted portfolio earns the MEAN
of its holdings, and equity returns are right-skewed, so mean and median can
point in opposite directions. But that implies the result rests on a handful
of large winners rather than on the typical name — which is a completely
different product, with completely different risk, from "these names
reliably beat the market".

Three things could make the compounded number an artifact, and each is
tested here:

  1. FORMATION MONTH. The compounding test formed portfolios every June.
     June was an arbitrary choice; if the result only exists in June, it is
     luck. All twelve formation months are run.

  2. SKEW DEPENDENCE. If dropping the single best holding each year destroys
     the edge, the strategy is a lottery-ticket book, not a quality screen.
     Reported with the best name removed, and with the top two removed.

  3. CONCENTRATION. 25 names is a high-tracking-error bet. Whether the edge
     survives at wider breadth says whether it is a signal or a few picks.

Every run is against SPY, the benchmark a customer would otherwise hold.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "experiments", "monthly_dca", "cache")
PIT = os.path.join(CACHE, "v2", "sp500_pit")


def load():
    px = pd.read_parquet(os.path.join(PIT, "prices_extended_pit.parquet")).sort_index()
    px.index = pd.to_datetime(px.index)
    base = pd.read_parquet(os.path.join(CACHE, "prices_extended.parquet"))
    base.index = pd.to_datetime(base.index)
    spy = base["SPY"].dropna()
    mem = pd.read_parquet(os.path.join(PIT, "sp500_membership_monthly.parquet"))
    mem["asof"] = pd.to_datetime(mem["asof"])
    return px, spy, mem


def screen(hist: pd.DataFrame) -> pd.DataFrame:
    last = hist.iloc[-1]
    r = hist.pct_change()
    sma200 = hist.rolling(200).mean()
    df = pd.DataFrame({
        "pct_above": (hist.iloc[-756:] > sma200.iloc[-756:]).mean(),
        "downside": r.iloc[-252:].clip(upper=0).std() * np.sqrt(252),
        "long_run": last / hist.iloc[-1260] - 1.0,
        "dd": last / hist.iloc[-252:].max() - 1.0,
    }).dropna()
    if df.empty:
        return df
    df["quality"] = (df.pct_above.rank(pct=True) * 0.45
                     + (1 - df.downside.rank(pct=True)) * 0.30
                     + df.long_run.rank(pct=True) * 0.25)
    df["score"] = df.quality * (1 - df.dd.rank(pct=True))
    return df


def run(px, spy, mem, form_month: int, n_hold: int, drop_best: int = 0):
    """Annual buy-and-hold from `form_month`, equal weighted."""
    eq = eq_spy = 1.0
    rows = []
    for m in sorted(mem["asof"].unique()):
        asof = pd.Timestamp(m)
        if asof.month != form_month:
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
        picks = s.sort_values("score", ascending=False).head(n_hold).index.tolist()
        fut = px[uni].loc[px.index > asof]
        if len(fut) < 230:
            continue
        f = (fut.iloc[:252].ffill().iloc[-1][picks] / hist.iloc[-1][picks] - 1.0)
        f = f.replace([np.inf, -np.inf], np.nan).dropna()
        f = f[(f > -0.995) & (f < 30.0)]
        if len(f) < n_hold * 0.6:
            continue
        sfut = spy.loc[spy.index > asof]
        if len(sfut) < 230:
            continue
        sp = float(sfut.iloc[:252].iloc[-1] / spy.loc[:asof].iloc[-1] - 1.0)
        if drop_best:
            f = f.sort_values().iloc[:-drop_best] if len(f) > drop_best else f
        r = float(f.mean())
        eq *= (1 + r)
        eq_spy *= (1 + sp)
        rows.append((asof.year, r, sp))
    if len(rows) < 8:
        return None
    n = len(rows)
    wins = sum(1 for _, r, sp in rows if r > sp)
    return {"n": n, "eq": eq, "eq_spy": eq_spy,
            "cagr": eq ** (1 / n) - 1, "cagr_spy": eq_spy ** (1 / n) - 1,
            "wins": wins, "win_rate": wins / n,
            "rows": rows}


def main():
    px, spy, mem = load()

    print("=" * 92)
    print("1. FORMATION MONTH — was June lucky?")
    print("   25 names, annual hold, equal weighted, vs SPY")
    print("=" * 92)
    print(f"  {'month':<7} {'yrs':>4} {'screen':>9} {'SPY':>9} {'diff':>8} "
          f"{'screen x':>9} {'SPY x':>8} {'beat':>6}")
    months, diffs = [], []
    for mo in range(1, 13):
        res = run(px, spy, mem, mo, 25)
        if not res:
            continue
        d = res["cagr"] - res["cagr_spy"]
        diffs.append(d)
        months.append((mo, res))
        print(f"  {pd.Timestamp(2020, mo, 1).strftime('%b'):<7} {res['n']:>4} "
              f"{res['cagr']:>+9.2%} {res['cagr_spy']:>+9.2%} {d:>+8.2%} "
              f"{res['eq']:>8.2f}x {res['eq_spy']:>7.2f}x "
              f"{res['win_rate']:>5.0%}")
    if diffs:
        arr = np.array(diffs)
        print(f"\n  across all 12 formation months: mean edge {arr.mean():+.2%}/yr, "
              f"worst {arr.min():+.2%}, best {arr.max():+.2%}")
        print(f"  months with a positive edge: {(arr > 0).sum()}/12")
        if (arr > 0).sum() >= 10:
            print("  -> not a June artifact; the edge is present whenever you form.")
        else:
            print("  -> DEPENDS ON WHEN YOU FORM. That is a red flag, not a strategy.")

    print("\n" + "=" * 92)
    print("2. SKEW DEPENDENCE — does it survive without its best holding?")
    print("   If removing one name per year kills it, it is a lottery book.")
    print("=" * 92)
    print(f"  {'variant':<26} {'screen CAGR':>12} {'SPY':>9} {'edge':>9} {'beat':>7}")
    for drop, lab in ((0, "as-is"), (1, "best name removed"),
                      (2, "best two removed")):
        got = []
        for mo in range(1, 13):
            r = run(px, spy, mem, mo, 25, drop_best=drop)
            if r:
                got.append(r)
        if not got:
            continue
        c = np.mean([g["cagr"] for g in got])
        cs = np.mean([g["cagr_spy"] for g in got])
        w = np.mean([g["win_rate"] for g in got])
        print(f"  {lab:<26} {c:>+12.2%} {cs:>+9.2%} {c-cs:>+9.2%} {w:>6.0%}")

    print("\n" + "=" * 92)
    print("3. BREADTH — signal or a few picks?")
    print("=" * 92)
    print(f"  {'names':<8} {'screen CAGR':>12} {'SPY':>9} {'edge':>9} {'beat':>7}")
    for k in (10, 25, 50, 100):
        got = [run(px, spy, mem, mo, k) for mo in range(1, 13)]
        got = [g for g in got if g]
        if not got:
            continue
        c = np.mean([g["cagr"] for g in got])
        cs = np.mean([g["cagr_spy"] for g in got])
        w = np.mean([g["win_rate"] for g in got])
        print(f"  {k:<8} {c:>+12.2%} {cs:>+9.2%} {c-cs:>+9.2%} {w:>6.0%}")

    print("\n" + "=" * 92)
    print("READ THIS BEFORE QUOTING ANY OF THE ABOVE")
    print("=" * 92)
    print("  * Equal weighted and rebalanced annually, with no trading costs,")
    print("    no taxes and no slippage. A 25-name book turned over yearly in a")
    print("    taxable account loses a meaningful part of any edge shown here.")
    print("  * ~20 annual observations per formation month. Overlapping across")
    print("    months, so the twelve columns are NOT twelve independent tests.")
    print("  * The MEDIAN pick still loses to SPY. Any edge here comes from the")
    print("    mean beating the median — a few large winners — which is a very")
    print("    different risk profile from reliable outperformance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
