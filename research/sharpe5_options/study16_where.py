#!/usr/bin/env python3
"""Study 16: where is the edge largest RELATIVE TO THE TOLL?

Study 15 established the governing quantity. For a vertical held to expiry:

    gross premium  = (mid credit - realized loss) / width      [the prize]
    half-spread    = (mid credit - worst-side credit) / width  [the toll]
    f*             = breakeven fraction of the half-spread you must capture

f* < 0   profitable even when crossing the spread
f* < 0.5 easier than the average cell measured in study 15 (f* = 0.504)
f* > 1   unprofitable even at a perfect mid fill

Rather than hunting for a better breach forecast — which study 15 showed is
not available, since the market's own quoted odds beat any model here — this
searches the DESIGN SPACE for corners where the toll is structurally small:

  TENOR   the bid-ask is roughly fixed per leg in dollar terms, while premium
          grows with sqrt(T). Cost/premium should fall like 1/sqrt(T), so
          longer-dated spreads may clear their toll where 30-day ones cannot.

  WIDTH   the toll is two legs regardless of width, but the risk (and margin)
          scales with width. Wider spreads dilute a fixed cost over more
          risk capital.

  MONEYNESS / LIQUIDITY  spread-to-premium varies by an order of magnitude
          across strikes and names.

Every cell reports its own f*, so the output is a map of where this trade is
executable rather than a single verdict.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E
from structures import pick, settle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "where.parquet")

TIER1 = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX "
            "AVGO ORCL CRM ADBE TXN INTC CSCO".split())
TIER2 = set("JPM BAC C WFC MS GS XOM CVX COP WMT HD COST PG KO PEP MCD DIS "
            "NKE JNJ PFE MRK UNH LLY ABBV BMY AMGN CAT BA GE HON UPS MMM "
            "T VZ CMCSA IBM PYPL SBUX LOW GM F".split())

TENORS = [(15, 50, "30d"), (50, 100, "75d"), (100, 200, "150d"), (200, 400, "300d")]
WIDTHS = [(0.03, "3%"), (0.05, "5%"), (0.10, "10%"), (0.20, "20%")]
OTMS = [(0.02, "2%"), (0.04, "4%"), (0.07, "7%"), (0.12, "12%")]


def main(step=3):
    dates = E.available_dates()[::step]
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    hist = {s: panel[s].dropna() for s in panel.columns}

    def se_of(sym, exp):
        c = hist.get(sym)
        if c is None:
            return None
        b = c[c.index <= exp]
        a = c[c.index > exp]
        if len(b) and (pd.Timestamp(exp) - pd.Timestamp(b.index[-1])).days <= 4:
            return float(b.iloc[-1])
        if len(a) and (pd.Timestamp(a.index[0]) - pd.Timestamp(exp)).days <= 4:
            return float(a.iloc[0])
        return None

    rows = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        if ch.empty or day not in panel.index:
            continue
        for sym, g in ch.groupby("act_symbol"):
            tier = 1 if sym in TIER1 else (2 if sym in TIER2 else 3)
            if tier == 3 or sym not in panel.columns:
                continue
            s = panel.at[day, sym]
            if not np.isfinite(s):
                continue
            for lo, hi, tlab in TENORS:
                gt = g[(g.dte >= lo) & (g.dte <= hi)]
                if gt.empty:
                    continue
                exp = gt.expiration.min()
                ge = gt[gt.expiration == exp]
                se = se_of(sym, exp)
                if se is None or abs(math.log(se / s)) > 0.7:
                    continue
                for otm, olab in OTMS:
                    for wpct, wlab in WIDTHS:
                        if wpct >= otm + 0.25:
                            continue
                        sl = pick(ge, "Put", s * (1 - otm))
                        wl = pick(ge, "Put", s * (1 - otm - wpct))
                        if sl is None or wl is None:
                            continue
                        ks, kl = float(sl.strike), float(wl.strike)
                        if kl >= ks:
                            continue
                        width = (ks - kl) * 100.0
                        cw = (sl.bid - wl.ask) * 100.0 / width          # worst-side
                        cm = ((sl.bid + sl.ask) / 2
                              - (wl.bid + wl.ask) / 2) * 100.0 / width  # mid
                        if not (np.isfinite(cw) and np.isfinite(cm)) or cm <= 0:
                            continue
                        loss = (settle(se, ks, "Put") - settle(se, kl, "Put")) * 100.0 / width
                        rows.append((day, sym, tier, tlab, olab, wlab,
                                     cw, cm, loss, int(ge.dte.iloc[0])))
        if (di + 1) % 100 == 0:
            print(f"where {di+1}/{len(dates)} rows={len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=["date", "sym", "tier", "tenor", "otm",
                                    "width", "cw", "cm", "loss", "dte"])
    d.to_parquet(OUT, index=False)
    print(f"\nrows: {len(d):,}")
    report(d)


def fstar(g):
    """Breakeven fraction of the half-spread that must be captured."""
    gross = g.cm.mean() - g.loss.mean()      # premium at a mid fill
    net = g.cw.mean() - g.loss.mean()        # premium crossing the spread
    if not np.isfinite(gross) or gross <= net:
        return np.nan, gross, net
    return -net / (gross - net), gross, net


def show(d, by, label, min_n=400):
    print(f"\n--- by {label} ---")
    print(f"  {'cell':>14} {'f*':>8} {'gross':>9} {'net(worst)':>11} "
          f"{'toll':>8} {'n':>8}")
    res = []
    for key, g in d.groupby(by):
        if len(g) < min_n:
            continue
        f, gross, net = fstar(g)
        toll = g.cm.mean() - g.cw.mean()
        res.append((key, f, gross, net, toll, len(g)))
    for key, f, gross, net, toll, n in sorted(res, key=lambda r: (np.inf if not np.isfinite(r[1]) else r[1])):
        flag = ""
        if np.isfinite(f):
            if f < 0:
                flag = "  <== PROFITABLE CROSSING THE SPREAD"
            elif f < 0.5:
                flag = "  <- easier than average"
        k = "/".join(map(str, key)) if isinstance(key, tuple) else str(key)
        print(f"  {k:>14} {f:>8.3f} {gross:>+9.4f} {net:>+11.4f} "
              f"{toll:>8.4f} {n:>8,}{flag}")


def report(d):
    print("=" * 82)
    print("STUDY 16 — breakeven fill quality f* across the design space")
    print("  f* < 0    profitable even crossing the spread")
    print("  f* = 0.50 the study-15 baseline (30d, 5% wide, 4% OTM)")
    print("  f* > 1    unprofitable even at a mid fill")
    print("=" * 82)
    show(d, "tenor", "TENOR")
    show(d, "width", "WIDTH")
    show(d, "otm", "MONEYNESS (short strike OTM by)")
    show(d, "tier", "LIQUIDITY TIER")
    show(d, ["tenor", "width"], "TENOR x WIDTH", min_n=300)
    best = d[(d.tenor.isin(["150d", "300d"])) & (d.width == "20%")]
    if len(best) > 300:
        show(best, "otm", "MONEYNESS within long-dated + widest", min_n=200)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(pd.read_parquet(OUT))
    else:
        main()
