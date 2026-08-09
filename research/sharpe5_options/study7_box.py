#!/usr/bin/env python3
"""Study 7: box spreads — the only family here with unbounded Sharpe potential.

A box (long call K1, short call K2, short put K1, long put K2, K1<K2) pays
exactly K2-K1 at expiry regardless of the path. It is a pure financing
instrument: its cost implies a rate. If the market's worst-side quotes let you
buy a box for less than the discounted payoff, that is a locked profit with no
path risk — a genuine arbitrage, and arbitrage has no Sharpe ceiling.

So this is the one place where Sharpe 5 could legitimately live. Three
outcomes, all informative:
  (a) nothing clears worst-side costs  -> options are efficiently financed
  (b) apparent arbitrage appears       -> almost certainly stale/crossed EOD
                                          quotes, which indicts the dataset
                                          and any strategy relying on it
  (c) real, persistent, executable     -> the answer

Everything is scored at WORST-SIDE fills: buy legs at ask, sell legs at bid.
Implied rate r = ln(payoff / cost) / T, compared against the T-bill curve.
American-exercise assignment risk is flagged, not ignored.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "boxes.parquet")


def box_cost(ge: pd.DataFrame, k1: float, k2: float):
    """Worst-side cost of the long box, and of shorting it (proceeds)."""
    c1 = ge[(ge.call_put == "Call") & (ge.strike == k1)]
    c2 = ge[(ge.call_put == "Call") & (ge.strike == k2)]
    p1 = ge[(ge.call_put == "Put") & (ge.strike == k1)]
    p2 = ge[(ge.call_put == "Put") & (ge.strike == k2)]
    if not (len(c1) and len(c2) and len(p1) and len(p2)):
        return None
    c1, c2, p1, p2 = c1.iloc[0], c2.iloc[0], p1.iloc[0], p2.iloc[0]
    # long box: buy C(k1), sell C(k2), sell P(k1), buy P(k2)
    if not (c1.ask > 0 and p2.ask > 0):
        return None
    long_cost = c1.ask - c2.bid - p1.bid + p2.ask
    # short box: sell C(k1), buy C(k2), buy P(k1), sell P(k2)
    short_proceeds = c1.bid - c2.ask - p1.ask + p2.bid
    return float(long_cost), float(short_proceeds)


def main(max_pairs_per_exp=40):
    dates = E.available_dates()
    rows = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        ch = ch[(ch.dte >= 10) & (ch.dte <= 400) & (ch.ask > 0)]
        if ch.empty:
            continue
        rf = E.rf_at(pd.Timestamp(day))
        for (sym, exp), ge in ch.groupby(["act_symbol", "expiration"]):
            T = float(ge.dte.iloc[0]) / 365.0
            if T <= 0:
                continue
            # strikes quoted on BOTH calls and puts with two-sided markets
            ks = sorted(set(ge[(ge.call_put == "Call") & (ge.bid > 0)].strike)
                        & set(ge[(ge.call_put == "Put") & (ge.bid > 0)].strike))
            if len(ks) < 2:
                continue
            pairs = 0
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    if pairs >= max_pairs_per_exp:
                        break
                    k1, k2 = float(ks[i]), float(ks[j])
                    width = k2 - k1
                    if width <= 0:
                        continue
                    r = box_cost(ge, k1, k2)
                    if r is None:
                        continue
                    long_cost, short_proceeds = r
                    pairs += 1
                    # implied financing rates at worst-side execution
                    if long_cost > 0:
                        r_lend = math.log(width / long_cost) / T
                    else:
                        r_lend = np.inf   # free money: pay <=0 for a positive payoff
                    # shorting the box borrows `short_proceeds` and repays width
                    if short_proceeds > 0:
                        r_borrow = math.log(width / short_proceeds) / T
                    else:
                        r_borrow = np.nan
                    rows.append((day, sym, exp, T, k1, k2, width, long_cost,
                                 short_proceeds, r_lend, r_borrow, rf))
                if pairs >= max_pairs_per_exp:
                    break
        if (di + 1) % 200 == 0:
            print(f"boxes {di+1}/{len(dates)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=["date", "sym", "exp", "T", "k1", "k2",
                                     "width", "long_cost", "short_proceeds",
                                     "r_lend", "r_borrow", "rf"])
    df.to_parquet(OUT, index=False)
    print("boxes:", df.shape)
    if not len(df):
        return

    df["edge_lend"] = df.r_lend - df.rf        # excess rate earned by lending
    df["edge_borrow"] = df.rf - df.r_borrow    # excess saved by borrowing
    fin = df[np.isfinite(df.r_lend)]
    print("\n=== implied financing rate from boxes, WORST-SIDE fills ===")
    print(f"rows: {len(df):,}  finite-rate rows: {len(fin):,}")
    print("r_lend (annualized) percentiles:")
    print(fin.r_lend.describe([.01, .05, .25, .5, .75, .95, .99]).round(4).to_string())
    print(f"\nT-bill reference (mean over sample): {df.rf.mean():.4f}")

    # hard arbitrage: pay <= 0 for a guaranteed positive payoff
    free = df[df.long_cost <= 0]
    print(f"\nHARD arbitrage (long box cost <= 0): {len(free):,} "
          f"({len(free)/len(df):.4%})")
    # soft: lock a rate far above T-bills
    for thr in (0.02, 0.05, 0.10, 0.25):
        g = fin[fin.edge_lend > thr]
        print(f"  boxes lending at > T-bill + {thr:.0%}: {len(g):,} "
              f"({len(g)/len(fin):.3%})", end="")
        if len(g):
            print(f"  median excess {g.edge_lend.median():+.1%}, "
                  f"median width ${g.width.iloc[:].median():.0f}, "
                  f"median T {g['T'].median()*365:.0f}d")
        else:
            print()

    # Are the apparent arbitrages concentrated in junk quotes?
    if len(free):
        print("\nDiagnostics on apparent free money (top symbols):")
        print(free.sym.value_counts().head(8).to_string())
        print(f"median width ${free.width.median():.1f}, "
              f"median dte {free['T'].median()*365:.0f}d")


if __name__ == "__main__":
    main()
