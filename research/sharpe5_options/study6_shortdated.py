#!/usr/bin/env python3
"""Study 6: short-dated (8-14 DTE) premium selling, held to expiry.

Theta density is highest in the last two weeks, and this is the one regime
where the premium collected might outrun the half-spread paid to enter. The
database carries no sub-week expirations, so 8-14 DTE is the shortest tenor
available (present on ~94% of observation dates).

Same honesty rules as everything else: sell at bid, buy at ask, settle at
intrinsic against the corrected parity spot, returns on committed margin.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right

import numpy as np
import pandas as pd

import engine as E
import portfolio as P
from structures import pick, leg_entry, settle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "shortdated.parquet")
LIQUID = {"SPY", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
          "AMD", "MU", "QCOM", "NFLX", "BA", "XOM", "JPM", "BAC", "C", "GM", "F",
          "INTC", "CSCO", "PYPL", "AVGO", "ORCL", "CRM", "ADBE", "TXN", "COST",
          "WMT", "DIS", "CAT", "GE", "PFE", "CVX", "WFC", "MS", "GS"}


def main(lo=8, hi=14):
    dates = E.available_dates()
    spots = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = spots.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    hist = {s: panel[s].dropna() for s in panel.columns}

    def settle_spot(sym, exp):
        if sym not in hist:
            return None
        col = hist[sym]
        before = col[col.index <= exp]
        after = col[col.index > exp]
        if len(before) and (pd.Timestamp(exp) - pd.Timestamp(before.index[-1])).days <= 4:
            return float(before.iloc[-1])
        if len(after) and (pd.Timestamp(after.index[0]) - pd.Timestamp(exp)).days <= 4:
            return float(after.iloc[0])
        return None

    rows = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        ch = ch[(ch.dte >= lo) & (ch.dte <= hi)]
        if ch.empty or day not in panel.index:
            continue
        for sym, g in ch.groupby("act_symbol"):
            if sym not in LIQUID or sym not in panel.columns:
                continue
            s = panel.at[day, sym]
            if not np.isfinite(s):
                continue
            exp = g.expiration.min()
            ge = g[g.expiration == exp]
            dte = int(ge.dte.iloc[0])
            se = settle_spot(sym, exp)
            if se is None or abs(math.log(se / s)) > 0.5:
                continue

            # ATM straddle
            cA, pA = pick(ge, "Call", s), pick(ge, "Put", s)
            if cA is not None and pA is not None:
                e1, e2 = leg_entry(cA, -1), leg_entry(pA, -1)
                if e1 and e2:
                    credit = (e1 + e2) * 100.0
                    loss = (settle(se, float(cA.strike), "Call")
                            + settle(se, float(pA.strike), "Put")) * 100.0
                    rows.append((day, sym, "sd_straddle", exp, dte,
                                 credit - loss, 0.30 * s * 100.0, credit, s, se))

            # 25-delta strangle
            gp = ge[(ge.call_put == "Put") & (ge.delta.abs().between(0.15, 0.35)) & (ge.bid > 0)]
            gc = ge[(ge.call_put == "Call") & (ge.delta.between(0.15, 0.35)) & (ge.bid > 0)]
            if len(gp) and len(gc):
                p25 = gp.loc[(gp.delta.abs() - 0.25).abs().idxmin()]
                c25 = gc.loc[(gc.delta - 0.25).abs().idxmin()]
                e1, e2 = leg_entry(p25, -1), leg_entry(c25, -1)
                if e1 and e2:
                    credit = (e1 + e2) * 100.0
                    loss = (settle(se, float(p25.strike), "Put")
                            + settle(se, float(c25.strike), "Call")) * 100.0
                    rows.append((day, sym, "sd_strangle25", exp, dte,
                                 credit - loss, 0.25 * s * 100.0, credit, s, se))

            # put credit spread 3%/7%
            sl, wl = pick(ge, "Put", s * 0.97), pick(ge, "Put", s * 0.93, )
            if sl is not None and wl is not None and float(wl.strike) < float(sl.strike):
                e1, e2 = leg_entry(sl, -1), leg_entry(wl, +1)
                if e1 and e2 and e1 - e2 > 0:
                    credit = (e1 - e2) * 100.0
                    loss = (settle(se, float(sl.strike), "Put")
                            - settle(se, float(wl.strike), "Put")) * 100.0
                    width = (float(sl.strike) - float(wl.strike)) * 100.0
                    rows.append((day, sym, "sd_putspread", exp, dte,
                                 credit - loss, width, credit, s, se))
        if (di + 1) % 200 == 0:
            print(f"shortdated {di+1}/{len(dates)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=["date", "act_symbol", "structure", "expiration",
                                     "dte", "pnl", "margin", "credit", "spot", "spot_exp"])
    df.to_parquet(OUT, index=False)
    print("shortdated rows:", df.shape)
    if not len(df):
        return
    df["date"] = pd.to_datetime(df.date)
    df["ret"] = df.pnl / df.margin
    print(f"\n=== SHORT-DATED ({lo}-{hi} DTE) held to expiry, worst-side fills ===")
    for stx in df.structure.unique():
        for univ, nm in [(LIQUID, "liquid"), ({"SPY"}, "SPY")]:
            g = df[(df.structure == stx) & df.act_symbol.isin(univ)]
            if len(g) < 60:
                continue
            s = g.groupby("date").ret.mean()
            r = P.screen_sharpe(s)
            print(f"  {stx:>14} [{nm:>6}]: scrSharpe={r.get('sharpe_scr', float('nan')):+.2f} "
                  f"hit={r.get('hit', float('nan')):.2f} meanRet={g.ret.mean():+.4f} n={len(g)}")


if __name__ == "__main__":
    main()
