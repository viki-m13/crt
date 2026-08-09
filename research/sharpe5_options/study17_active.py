#!/usr/bin/env python3
"""Study 17: ACTIVE management of the long-dated wide put spread.

Everything in this project so far was static hold-to-expiry. That assumption
came from trials 18-71, which tested exits on 30-day ATM straddles held 2-3
days — the worst possible case, where a round trip's spread swamps two days of
theta. It was then carried, unexamined, through ~250 further configurations
and never re-tested on the 75d/20% structure that actually works.

Two reasons active management should matter *here* specifically:

  BREADTH. The binding constraint is ~35 independent rounds over the sample.
  A profit target that closes in ~30 days instead of 75 gives ~2.5x the rounds
  per year, and IR scales with sqrt(rounds) — this attacks the exact term that
  caps every strategy in this study.

  TAIL. The static version loses full width on 24% of trades and draws down
  50%. Sharpe is mean/vol, so a rule that cuts the tail faster than it cuts
  the mean raises Sharpe even while lowering total return.

Mechanics, all at WORST-SIDE prices:
  entry credit  = short.bid - long.ask     (what you actually receive)
  cost to close = short.ask - long.bid     (what you actually pay)
  P&L           = entry credit - cost to close
Positions are marked at every observation date and exited the moment a rule
fires, at that date's quotes. Expiry settlement is intrinsic as always.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import engine as E
from structures import settle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "active.parquet")

TIER = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX AVGO "
           "ORCL CRM ADBE TXN INTC CSCO JPM BAC C WFC GS XOM CVX WMT HD COST "
           "PG KO PEP MCD DIS NKE JNJ PFE MRK UNH LLY ABBV CAT BA GE HON UPS "
           "T VZ IBM SBUX LOW GM F".split())

OTM, WIDTH = 0.04, 0.20
DTE_LO, DTE_HI = 50, 100
ENTRY_EVERY = 3          # start a cohort every 3rd observation date


@dataclass
class Pos:
    sym: str
    exp: str
    ks: float
    kl: float
    width: float
    credit: float
    entry_date: str
    entry_dte: int


def close_cost(chain_sym, exp, ks, kl):
    """Worst-side cost to buy back the spread: buy short leg at ask, sell long
    leg at bid. Returns None if either leg is not tradable."""
    g = chain_sym[(chain_sym.expiration == exp) & (chain_sym.call_put == "Put")]
    s_leg = g[g.strike == ks]
    l_leg = g[g.strike == kl]
    if not (len(s_leg) and len(l_leg)):
        return None
    a = float(s_leg.ask.iloc[0])
    b = float(l_leg.bid.iloc[0])
    if a <= 0:
        return None
    return (a - b) * 100.0


def run(rule_name, profit_take=None, stop_mult=None, dte_exit=None,
        breach_exit=False, verbose=False):
    """One forward pass applying a single exit rule."""
    dates = E.available_dates()
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()

    open_pos: list[Pos] = []
    trades = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        if ch.empty:
            continue
        by_sym = {s: g for s, g in ch.groupby("act_symbol")}
        spots = panel.loc[day].dropna().to_dict() if day in panel.index else {}

        # ---- manage existing positions
        still: list[Pos] = []
        for p in open_pos:
            g = by_sym.get(p.sym)
            expired = p.exp <= day
            if expired:
                se = spots.get(p.sym)
                if se is None:
                    still.append(p)
                    continue
                loss = (settle(se, p.ks, "Put") - settle(se, p.kl, "Put")) * 100.0
                trades.append((p.entry_date, day, p.sym, p.credit,
                               p.credit - loss, p.width, "expiry",
                               (pd.Timestamp(day) - pd.Timestamp(p.entry_date)).days))
                continue
            cost = close_cost(g, p.exp, p.ks, p.kl) if g is not None else None
            if cost is None:
                still.append(p)
                continue
            dte = (pd.Timestamp(p.exp) - pd.Timestamp(day)).days
            spot = spots.get(p.sym)
            reason = None
            if profit_take is not None and cost <= (1 - profit_take) * p.credit:
                reason = "profit"
            elif stop_mult is not None and cost >= stop_mult * p.credit:
                reason = "stop"
            elif dte_exit is not None and dte <= dte_exit:
                reason = "time"
            elif breach_exit and spot is not None and spot <= p.ks:
                reason = "breach"
            if reason:
                trades.append((p.entry_date, day, p.sym, p.credit,
                               p.credit - cost, p.width, reason,
                               (pd.Timestamp(day) - pd.Timestamp(p.entry_date)).days))
            else:
                still.append(p)
        open_pos = still

        # ---- new entries
        if di % ENTRY_EVERY == 0:
            for sym, g in by_sym.items():
                if sym not in TIER:
                    continue
                s = spots.get(sym)
                if s is None:
                    continue
                gt = g[(g.dte >= DTE_LO) & (g.dte <= DTE_HI)]
                if gt.empty:
                    continue
                exp = gt.expiration.min()
                ge = gt[(gt.expiration == exp) & (gt.call_put == "Put")]
                if ge.empty:
                    continue
                short_c = ge[ge.bid > 0]
                if short_c.empty:
                    continue
                sl = short_c.loc[(short_c.strike - s * (1 - OTM)).abs().idxmin()]
                long_c = ge[ge.ask > 0]
                if long_c.empty:
                    continue
                wl = long_c.loc[(long_c.strike - s * (1 - OTM - WIDTH)).abs().idxmin()]
                ks, kl = float(sl.strike), float(wl.strike)
                if kl >= ks:
                    continue
                credit = (float(sl.bid) - float(wl.ask)) * 100.0
                if credit <= 0:
                    continue
                open_pos.append(Pos(sym, exp, ks, kl, (ks - kl) * 100.0,
                                    credit, day, int(ge.dte.iloc[0])))
        if verbose and (di + 1) % 200 == 0:
            print(f"  {rule_name}: {di+1}/{len(dates)} open={len(open_pos)} "
                  f"closed={len(trades)}", flush=True)

    t = pd.DataFrame(trades, columns=["entry", "exit", "sym", "credit", "pnl",
                                      "width", "reason", "held_days"])
    t["rule"] = rule_name
    return t


def summarize(t: pd.DataFrame, label: str):
    if not len(t):
        print(f"  {label:<26} no trades")
        return None
    t = t.copy()
    t["ret"] = t.pnl / t.width
    t["entry"] = pd.to_datetime(t.entry)
    hold = t.held_days.mean()
    rounds = 365.0 / max(hold, 1)
    s = t.groupby("entry").ret.mean()
    sh = s.mean() / s.std() * math.sqrt(rounds) if s.std() > 0 else np.nan
    yrs = (t.entry.max() - t.entry.min()).days / 365.25
    ann = (1 + t.ret.mean()) ** rounds - 1
    print(f"  {label:<26} ret/trade={t.ret.mean():+.4f} hold={hold:5.1f}d "
          f"rounds/yr={rounds:5.1f} Sharpe*={sh:+.2f} ann={ann:+7.1%} "
          f"worst={t.ret.min():+.2f} lossRate={(t.pnl<0).mean():.1%} n={len(t):,}")
    return {"label": label, "ret": t.ret.mean(), "hold": hold, "rounds": rounds,
            "sharpe": sh, "ann": ann, "n": len(t)}


def main():
    print("=" * 100)
    print("STUDY 17 — active management of the 75d/20% put credit spread")
    print("  Sharpe* annualizes on ACTUAL rounds/yr implied by realized holding")
    print("  period, so a faster rule is credited with the breadth it creates.")
    print("=" * 100)
    rules = [
        ("hold to expiry", dict()),
        ("profit 25%", dict(profit_take=0.25)),
        ("profit 50%", dict(profit_take=0.50)),
        ("profit 75%", dict(profit_take=0.75)),
        ("stop 2x credit", dict(stop_mult=2.0)),
        ("stop 3x credit", dict(stop_mult=3.0)),
        ("exit at 21 DTE", dict(dte_exit=21)),
        ("breach exit", dict(breach_exit=True)),
        ("profit50 + stop2x", dict(profit_take=0.50, stop_mult=2.0)),
        ("profit50 + stop3x", dict(profit_take=0.50, stop_mult=3.0)),
        ("profit50 + 21DTE", dict(profit_take=0.50, dte_exit=21)),
        ("profit50+stop2x+21DTE", dict(profit_take=0.50, stop_mult=2.0, dte_exit=21)),
        ("profit25 + stop2x", dict(profit_take=0.25, stop_mult=2.0)),
    ]
    allt, res = [], []
    for name, kw in rules:
        t = run(name, **kw)
        allt.append(t)
        r = summarize(t, name)
        if r:
            res.append(r)
    pd.concat(allt, ignore_index=True).to_parquet(OUT, index=False)

    print("\n" + "=" * 100)
    print("DEV vs HOLDOUT for the top rules by Sharpe*")
    print("=" * 100)
    big = pd.concat(allt, ignore_index=True)
    big["entry"] = pd.to_datetime(big.entry)
    big["ret"] = big.pnl / big.width
    for r in sorted(res, key=lambda x: -(x["sharpe"] if np.isfinite(x["sharpe"]) else -9))[:5]:
        t = big[big.rule == r["label"]]
        for lab, seg in (("dev", t[t.entry <= "2024-12-31"]),
                         ("holdout", t[t.entry > "2024-12-31"])):
            if len(seg) < 50:
                continue
            s = seg.groupby("entry").ret.mean()
            sh = s.mean() / s.std() * math.sqrt(r["rounds"]) if s.std() > 0 else np.nan
            print(f"  {r['label']:<26} {lab:<8} ret={seg.ret.mean():+.4f} "
                  f"Sharpe*={sh:+.2f} n={len(seg):,}")


if __name__ == "__main__":
    main()
