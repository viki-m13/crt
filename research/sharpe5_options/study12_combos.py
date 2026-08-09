#!/usr/bin/env python3
"""Study 12: combined STOCK + OPTION positions.

Not options alone and not stock alone, but the classic joint structures where
you hold the underlying and trade options against it. These change the shape of
the equity return rather than trying to extract premium in isolation, and they
are the one family where the option leg is a minority of the risk, so its
bid-ask matters proportionally less.

  buy_write     long stock + short call (covered call)
  put_write     short put, cash-secured (synthetically equivalent to buy_write)
  collar        long stock + long OTM put + short OTM call (financed hedge)
  prot_put      long stock + long OTM put (pure insurance)

Each is held to expiry and compared against BUY-AND-HOLD THE SAME STOCK over
the identical dates. A structure that merely tracks the stock has added nothing;
the question is whether it improves return per unit of risk.

Stock is entered/exited at the corrected parity spot with 5 bps each way.
Option legs are worst-side as always. Returns are on the full stock notional,
which is the capital genuinely at risk.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right

import numpy as np
import pandas as pd

import engine as E
from structures import pick, leg_entry, settle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "combos.parquet")
STOCK_BPS = 5.0
LIQUID = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA "
             "XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST "
             "WMT DIS CAT GE PFE CVX WFC MS GS".split())


def main():
    dates = E.available_dates()
    spots = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = spots.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    hist = {s: panel[s].dropna() for s in panel.columns}

    def settle_spot(sym, exp):
        col = hist.get(sym)
        if col is None:
            return None
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
        ch = ch[(ch.dte >= 15) & (ch.dte <= 50)]
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
            se = settle_spot(sym, exp)
            if se is None or abs(math.log(se / s)) > 0.5:
                continue
            notional = s * 100.0
            # stock leg: buy at spot + 5bp, sell at settle - 5bp
            stock_pnl = (se * (1 - STOCK_BPS / 1e4) - s * (1 + STOCK_BPS / 1e4)) * 100.0
            rec = dict(date=day, act_symbol=sym, expiration=exp,
                       dte=int(ge.dte.iloc[0]), spot=s, spot_exp=se,
                       notional=notional, stock_pnl=stock_pnl)

            # --- buy-write at several call strikes
            for pct, tag in ((1.00, "atm"), (1.02, "2pct"), (1.05, "5pct")):
                c = pick(ge, "Call", s * pct)
                if c is None:
                    continue
                prem = leg_entry(c, -1)
                if prem is None:
                    continue
                call_pnl = prem * 100.0 - settle(se, float(c.strike), "Call") * 100.0
                rows.append({**rec, "structure": f"buy_write_{tag}",
                             "pnl": stock_pnl + call_pnl, "iv": float(c.vol)})

            # --- collar: long 5% put, short 5% call, plus stock
            cp = pick(ge, "Call", s * 1.05)
            pp = pick(ge, "Put", s * 0.95, need_bid=False)
            if cp is not None and pp is not None:
                ec, ep = leg_entry(cp, -1), leg_entry(pp, +1)
                if ec is not None and ep is not None:
                    opt = (ec - ep) * 100.0 \
                        - settle(se, float(cp.strike), "Call") * 100.0 \
                        + settle(se, float(pp.strike), "Put") * 100.0
                    rows.append({**rec, "structure": "collar_5pct",
                                 "pnl": stock_pnl + opt, "iv": float(cp.vol)})

            # --- protective put 5% OTM
            if pp is not None:
                ep = leg_entry(pp, +1)
                if ep is not None:
                    opt = -ep * 100.0 + settle(se, float(pp.strike), "Put") * 100.0
                    rows.append({**rec, "structure": "prot_put_5pct",
                                 "pnl": stock_pnl + opt, "iv": float(pp.vol)})

            # --- cash-secured put write 5% OTM (no stock leg)
            pw = pick(ge, "Put", s * 0.95)
            if pw is not None:
                e = leg_entry(pw, -1)
                if e is not None:
                    opt = e * 100.0 - settle(se, float(pw.strike), "Put") * 100.0
                    rows.append({**rec, "structure": "put_write_5pct",
                                 "pnl": opt, "iv": float(pw.vol)})

            # --- pure stock benchmark over identical dates
            rows.append({**rec, "structure": "stock_only", "pnl": stock_pnl,
                         "iv": np.nan})
        if (di + 1) % 200 == 0:
            print(f"combos {di+1}/{len(dates)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print("combos:", df.shape)
    report(df)


def report(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    df["ret"] = df.pnl / df.notional
    print("\n" + "=" * 78)
    print("STOCK + OPTION COMBINATIONS — returns on full stock notional")
    print("=" * 78)
    print(f"{'structure':>18} {'meanRet':>9} {'sd':>8} {'Sharpe*':>9} "
          f"{'vs stock':>10} {'hit':>7} {'n':>8}")
    base = None
    order = ["stock_only", "buy_write_atm", "buy_write_2pct", "buy_write_5pct",
             "collar_5pct", "prot_put_5pct", "put_write_5pct"]
    for st in order:
        g = df[df.structure == st]
        if len(g) < 500:
            continue
        s = g.groupby("date").ret.mean()
        # monthly-cycle returns; ~12 independent rounds/yr
        mu, sd = s.mean(), s.std()
        sh = mu / sd * math.sqrt(12) if sd > 0 else np.nan
        if st == "stock_only":
            base = sh
        rel = "" if base is None else f"{sh - base:+.2f}"
        print(f"{st:>18} {mu:+9.4f} {sd:8.4f} {sh:+9.2f} {rel:>10} "
              f"{(g.pnl > 0).mean():7.1%} {len(g):8,}")

    # SPY only
    print("\nSPY only:")
    for st in order:
        g = df[(df.structure == st) & (df.act_symbol == "SPY")]
        if len(g) < 100:
            continue
        s = g.groupby("date").ret.mean()
        mu, sd = s.mean(), s.std()
        sh = mu / sd * math.sqrt(12) if sd > 0 else np.nan
        print(f"{st:>18} {mu:+9.4f} {sd:8.4f} {sh:+9.2f} "
              f"{(g.pnl > 0).mean():7.1%} {len(g):8,}")
    print("\n* Sharpe here is on monthly-cycle returns (sqrt(12)); it is a")
    print("  screening figure, directly comparable across rows, not a portfolio")
    print("  Sharpe with intermediate marks.")


if __name__ == "__main__":
    main()
