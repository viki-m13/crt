#!/usr/bin/env python3
"""Sleeve B — single-name term-inversion (earnings IV crush) harvesting.

Event-loop implementation for honest verification with M/W/F marks.

At each obs date:
  1. close cohorts whose age >= K obs (buy back at ask)
  2. for each name passing the gate (idio term inversion > THR, liquid tier,
     tradable quotes), sell 1 ATM front straddle at bid
  3. maintain BS delta hedges in stock at parity spots (2 bps)

Sizing: qty per position chosen so margin (30% notional) ~= capital * F_POS.
Capital must cover all concurrent margins; engine tracks equity honestly.
"""
from __future__ import annotations

import math
import os
from collections import deque

import numpy as np
import pandas as pd

import engine as E
from structures import bs_delta

HERE = os.path.dirname(os.path.abspath(__file__))


def make_strategy(feats: pd.DataFrame, thr: float = 0.04, K: int = 2,
                  universe: set | None = None, f_pos: float = 0.04,
                  capital: float = 1_000_000.0, hedge: bool = True,
                  max_new_per_day: int = 10):
    """Returns (strategy_fn, uses_hedge). feats must be point-in-time."""
    f = feats.copy()
    f["date_s"] = f.date.dt.strftime("%Y-%m-%d")
    fmap = {}
    for r in f.itertuples(index=False):
        fmap[(r.date_s, r.act_symbol)] = r
    spy_inv = {r.date_s: r.inv for r in f[f.act_symbol == "SPY"].itertuples(index=False)}

    def strategy(day, chain, quotes, spots, vol, state, eng):
        cohorts: deque = state.setdefault("cohorts", deque())
        orders: list[E.Leg] = []

        # 1) close due cohorts (age in obs counts)
        for c in cohorts:
            c["age"] += 1
        while cohorts and cohorts[0]["age"] >= K:
            c = cohorts.popleft()
            for leg in c["legs"]:
                orders.append(E.Leg(leg.symbol, leg.expiration, leg.strike, leg.cp, -leg.qty))

        # 2) new entries
        n_new = 0
        spyi = spy_inv.get(day, np.nan)
        margin_committed = sum(c["margin"] for c in cohorts)
        for sym in (universe or set()) & set(spots):
            if n_new >= max_new_per_day:
                break
            r = fmap.get((day, sym))
            if r is None or not np.isfinite(r.inv) or not np.isfinite(spyi):
                continue
            if (r.inv - spyi) <= thr:
                continue
            s = spots[sym]
            g = chain[(chain.act_symbol == sym) & (chain.dte >= 15) & (chain.dte <= 50)]
            if g.empty:
                continue
            exp1 = g.expiration.min()
            ge = g[g.expiration == exp1]
            cs = ge[(ge.call_put == "Call") & (ge.bid > 0)]
            ps = ge[(ge.call_put == "Put") & (ge.bid > 0)]
            if cs.empty or ps.empty:
                continue
            cA = cs.loc[(cs.strike - s).abs().idxmin()]
            pA = ps.loc[(ps.strike - s).abs().idxmin()]
            # spread sanity: reject if relative spread too wide to ever profit
            prem = cA.bid + pA.bid
            rt_cost = (cA.spread + pA.spread)
            if prem <= 0 or rt_cost / max(prem, 1e-9) > 0.25:
                continue
            margin1 = 0.30 * s * 100.0
            qty = max(int((capital * f_pos) / margin1), 0)
            if qty == 0 or margin_committed + qty * margin1 > 0.95 * capital:
                continue
            legs = [E.Leg(sym, exp1, float(cA.strike), "Call", -qty),
                    E.Leg(sym, exp1, float(pA.strike), "Put", -qty)]
            orders.extend(legs)
            cohorts.append({"age": 0, "legs": legs, "margin": qty * margin1,
                            "iv": float(np.nanmean([cA.vol, pA.vol])), "exp": exp1})
            margin_committed += qty * margin1
            n_new += 1

        # 3) hedge targets
        if hedge:
            tgt: dict[str, float] = {}
            for c in cohorts:
                for leg in c["legs"]:
                    s = spots.get(leg.symbol)
                    if s is None:
                        continue
                    T = max((pd.Timestamp(leg.expiration) - pd.Timestamp(day)).days, 0) / 365.0
                    d = bs_delta(s, leg.strike, T, max(c["iv"], 0.05), leg.cp)
                    tgt[leg.symbol] = tgt.get(leg.symbol, 0.0) - leg.qty * d * 100.0
            state["hedge_targets"] = tgt
        return orders

    return strategy


def run(thr=0.04, K=2, universe=None, capital=1_000_000.0, hedge=True,
        f_pos=0.04, dates=None, label=None, n_trials=1):
    import verify_engine as V
    feats = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    feats["date"] = pd.to_datetime(feats.date)
    feats["inv"] = feats.iv_front - feats.iv_back
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    spot_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    dates = dates or E.available_dates()
    strat = make_strategy(feats, thr=thr, K=K, universe=universe,
                          f_pos=f_pos, capital=capital, hedge=hedge)
    bt = E.Backtester(dates, capital=capital, stock_hedge=hedge)
    eq, fills, _ = bt.run(strat, spot_panel)
    V.report(eq, label or f"B thr={thr} K={K}", n_trials=n_trials)
    return eq, fills
