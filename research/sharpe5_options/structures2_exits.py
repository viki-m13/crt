#!/usr/bin/env python3
"""Second pass: short-hold exit variants of the ATM front straddle.

For each (date, symbol): sell ATM front straddle at bid; buy back BOTH legs at
ask at the +1/+2/+3-th observation date; delta-hedged variant re-hedges with
stock (2bps) at each obs in between at parity spots. Also the LONG variant
(buy at ask, sell back at bid). Output cache/exits.parquet.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right

import numpy as np
import pandas as pd

import engine as E
from structures import bs_delta, pick, leg_entry, STOCK_BPS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PQ = os.path.join(HERE, "cache", "exits.parquet")


def main():
    dates = E.available_dates()
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    spot_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    sym_hist = {sym: spot_panel[sym].dropna() for sym in spot_panel.columns}

    rows = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        front = ch[(ch.dte >= 15) & (ch.dte <= 50)]
        for sym, g in front.groupby("act_symbol"):
            if sym not in spot_panel.columns or day not in spot_panel.index:
                continue
            s_now = spot_panel.at[day, sym]
            if not np.isfinite(s_now):
                continue
            exp1 = g.expiration.min()
            ge = g[g.expiration == exp1]
            dte1 = int(ge.dte.iloc[0])
            cA, pA = pick(ge, "Call", s_now), pick(ge, "Put", s_now)
            if cA is None or pA is None:
                continue
            ivm = float(np.nanmean([cA.vol, pA.vol]))
            hs = sym_hist[sym]
            ds = list(hs.index)
            i0 = bisect_right(ds, day) - 1
            if i0 < 0 or ds[i0] != day:
                continue

            sb = (leg_entry(cA, -1), leg_entry(pA, -1))   # sell at bid
            ba = (leg_entry(cA, +1), leg_entry(pA, +1))   # buy at ask
            if None in sb or None in ba:
                continue
            exp_ts = pd.Timestamp(exp1)

            for k in (1, 2, 3):
                j = i0 + k
                if j >= len(ds) or ds[j] > exp1:
                    break
                dx = ds[j]
                ch2 = E.load_chain(dx)
                q2 = ch2[(ch2.act_symbol == sym) & (ch2.expiration == exp1)]
                qc = q2[(q2.call_put == "Call") & (q2.strike == float(cA.strike))]
                qp = q2[(q2.call_put == "Put") & (q2.strike == float(pA.strike))]
                if not (len(qc) and len(qp)):
                    continue
                qc, qp = qc.iloc[0], qp.iloc[0]
                if not (qc.ask > 0 and qp.ask > 0 and qc.bid > 0 and qp.bid > 0):
                    continue
                # stock-hedge P&L over [i0, j] for the SHORT straddle
                hedge = 0.0
                sh = 0.0
                ok = True
                prev_s = float(hs.iloc[i0])
                for m in range(i0, j + 1):
                    dm, sm = ds[m], float(hs.iloc[m])
                    if m > i0 and abs(math.log(sm / prev_s)) > 0.22:
                        ok = False
                        break
                    prev_s = sm
                    T = max((exp_ts - pd.Timestamp(dm)).days, 0) / 365.0
                    dl = (bs_delta(sm, float(cA.strike), T, max(ivm, .05), "Call")
                          + bs_delta(sm, float(pA.strike), T, max(ivm, .05), "Put"))
                    tgt = dl * 100.0 if m < j else 0.0  # short straddle hedge: +delta*100 shares
                    dchg = tgt - sh
                    hedge -= dchg * sm
                    hedge -= abs(dchg) * sm * STOCK_BPS / 1e4
                    sh = tgt
                    if m < j:
                        hedge_mark_day = dm
                if not ok:
                    continue
                # final stock cash-out already handled by tgt=0 at m==j
                short_pnl = (sum(sb) - (qc.ask + qp.ask)) * 100.0
                long_pnl = ((qc.bid + qp.bid) - sum(ba)) * 100.0
                margin = 0.30 * s_now * 100.0
                rows.append((day, sym, k, dx, exp1, dte1, ivm, s_now,
                             short_pnl, short_pnl + hedge, long_pnl,
                             long_pnl - hedge, margin))
        if (di + 1) % 100 == 0:
            print(f"exits {di+1}/{len(dates)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=["date", "act_symbol", "k", "exit_date",
                                     "expiration", "dte", "iv", "spot",
                                     "short_pnl", "short_dh_pnl", "long_pnl",
                                     "long_dh_pnl", "margin"])
    df.to_parquet(OUT_PQ, index=False)
    print("saved", df.shape)


if __name__ == "__main__":
    main()
