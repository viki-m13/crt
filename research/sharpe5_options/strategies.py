#!/usr/bin/env python3
"""Event-loop implementations of the surviving sleeves (hold-to-expiry).

A selector examines (day, chain, spots, feats) and returns a list of
candidate positions: each = dict(symbol, legs=[Leg...], margin, iv, hedged).
The cohort wrapper sizes them (equal margin), enforces a utilization cap,
tracks hedge targets, and lets the engine settle at expiry (no exit orders).
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd

import engine as E
import verify_engine as V
from structures import bs_delta

HERE = os.path.dirname(os.path.abspath(__file__))

LIQUID = {"SPY", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
          "AMD", "MU", "QCOM", "NFLX", "BA", "XOM", "JPM", "BAC", "C", "GM", "F",
          "INTC", "CSCO", "PYPL", "AVGO", "ORCL", "CRM", "ADBE", "TXN", "COST",
          "WMT", "DIS", "CAT", "GE", "PFE", "CVX", "WFC", "MS", "GS"}


def _front(chain, sym, lo=15, hi=50):
    g = chain[(chain.act_symbol == sym) & (chain.dte >= lo) & (chain.dte <= hi)]
    if g.empty:
        return None
    return g[g.expiration == g.expiration.min()]


def _nearest(ge, cp, target, need_bid=True):
    g = ge[(ge.call_put == cp)]
    if need_bid:
        g = g[g.bid > 0]
    if g.empty:
        return None
    return g.loc[(g.strike - target).abs().idxmin()]


# ---------------------------------------------------------------- selectors

def make_sel_putspread(short_pct=0.04, long_pct=0.09, syms=("SPY",),
                       mom_gate=False):
    def sel(day, chain, spots, F):
        out = []
        for sym in syms:
            s = spots.get(sym)
            if s is None:
                continue
            if mom_gate:
                r = F.get((day, sym))
                if r is None or not (r.get("mom1w", 0) > 0):
                    continue
            ge = _front(chain, sym)
            if ge is None:
                continue
            sl = _nearest(ge, "Put", s * (1 - short_pct))
            wl = _nearest(ge, "Put", s * (1 - long_pct), need_bid=False)
            if sl is None or wl is None or float(wl.strike) >= float(sl.strike):
                continue
            if sl.bid - wl.ask <= 0:
                continue
            width = (float(sl.strike) - float(wl.strike)) * 100.0
            legs = [E.Leg(sym, sl.expiration, float(sl.strike), "Put", -1),
                    E.Leg(sym, wl.expiration, float(wl.strike), "Put", +1)]
            out.append(dict(symbol=sym, legs=legs, margin=width,
                            iv=float(sl.vol), hedged=False))
        return out
    return sel


sel_A_putspread = make_sel_putspread()


def sel_A_ss_dh(day, chain, spots, F):
    """SPY ATM short straddle, delta-hedged, only in contango (spy inv<0)."""
    r = F.get((day, "SPY"))
    if r is None or not np.isfinite(r.get("inv", np.nan)) or r["inv"] >= 0:
        return []
    s = spots.get("SPY")
    if s is None:
        return []
    ge = _front(chain, "SPY")
    if ge is None:
        return []
    cA, pA = _nearest(ge, "Call", s), _nearest(ge, "Put", s)
    if cA is None or pA is None:
        return []
    legs = [E.Leg("SPY", cA.expiration, float(cA.strike), "Call", -1),
            E.Leg("SPY", pA.expiration, float(pA.strike), "Put", -1)]
    return [dict(symbol="SPY", legs=legs, margin=0.30 * s * 100.0,
                 iv=float(np.nanmean([cA.vol, pA.vol])), hedged=True)]


def make_sel_B_strangle25(thr=0.06, max_new=6):
    def sel(day, chain, spots, F):
        out = []
        spy = F.get((day, "SPY"))
        if spy is None or not np.isfinite(spy.get("inv", np.nan)):
            return out
        for sym in sorted(LIQUID & set(spots)):
            if len(out) >= max_new:
                break
            r = F.get((day, sym))
            if r is None or not np.isfinite(r.get("inv", np.nan)):
                continue
            if (r["inv"] - spy["inv"]) <= thr:
                continue
            ge = _front(chain, sym)
            if ge is None:
                continue
            s = spots[sym]
            gp = ge[(ge.call_put == "Put") & (ge.delta.abs().between(0.15, 0.35)) & (ge.bid > 0)]
            gc = ge[(ge.call_put == "Call") & (ge.delta.between(0.15, 0.35)) & (ge.bid > 0)]
            if gp.empty or gc.empty:
                continue
            p25 = gp.loc[(gp.delta.abs() - 0.25).abs().idxmin()]
            c25 = gc.loc[(gc.delta - 0.25).abs().idxmin()]
            prem = p25.bid + c25.bid
            if prem <= 0 or (p25.spread + c25.spread) / prem > 0.30:
                continue
            legs = [E.Leg(sym, p25.expiration, float(p25.strike), "Put", -1),
                    E.Leg(sym, c25.expiration, float(c25.strike), "Call", -1)]
            out.append(dict(symbol=sym, legs=legs, margin=0.25 * s * 100.0,
                            iv=float(np.nanmean([p25.vol, c25.vol])), hedged=False))
        return out
    return sel


def make_sel_C_shortonly(q=0.8, max_new=8, wings=None, ivr_min=None):
    def sel(day, chain, spots, F):
        cands = []
        for sym in sorted(LIQUID & set(spots)):
            r = F.get((day, sym))
            if r is None:
                continue
            if ivr_min is not None and not (r.get("iv_rank", np.nan) > ivr_min):
                continue
            z = r.get("z_cy", np.nan) + r.get("z_mom", np.nan)
            if np.isfinite(z):
                cands.append((z, sym))
        if len(cands) < 12:
            return []
        cands.sort(reverse=True)
        cut = np.quantile([z for z, _ in cands], q)
        out = []
        for z, sym in cands:
            if z < cut or len(out) >= max_new:
                break
            ge = _front(chain, sym)
            if ge is None:
                continue
            s = spots[sym]
            cA, pA = _nearest(ge, "Call", s), _nearest(ge, "Put", s)
            if cA is None or pA is None:
                continue
            prem = cA.bid + pA.bid
            if prem <= 0 or (cA.spread + pA.spread) / prem > 0.25:
                continue
            legs = [E.Leg(sym, cA.expiration, float(cA.strike), "Call", -1),
                    E.Leg(sym, pA.expiration, float(pA.strike), "Put", -1)]
            margin = 0.30 * s * 100.0
            if wings:
                pw = _nearest(ge, "Put", s * (1 - wings), need_bid=False)
                cw = _nearest(ge, "Call", s * (1 + wings), need_bid=False)
                if pw is not None and cw is not None and \
                   float(pw.strike) < float(pA.strike) and float(cw.strike) > float(cA.strike):
                    legs += [E.Leg(sym, pw.expiration, float(pw.strike), "Put", +1),
                             E.Leg(sym, cw.expiration, float(cw.strike), "Call", +1)]
                    margin = max(float(pA.strike) - float(pw.strike),
                                 float(cw.strike) - float(cA.strike)) * 100.0
            out.append(dict(symbol=sym, legs=legs, margin=margin,
                            iv=float(np.nanmean([cA.vol, pA.vol])), hedged=True))
        return out
    return sel


# ---------------------------------------------------------------- wrapper

def make_cohort_strategy(selector: Callable, feats: pd.DataFrame,
                         capital: float, f_pos: float = 0.03,
                         util_cap: float = 0.95, profit_take: float | None = None):
    """profit_take: close a cohort once its buy-back cost has fallen to
    (1-profit_take) of the credit received. Study 17 found this is the only
    active rule that survives out of sample — closing winners buys back cheap
    far-OTM options (small toll) and recycles capital into more rounds, whereas
    every loss-cutting rule pays a toll that scales with how badly the trade
    has gone."""
    f = feats.copy()
    f["date_s"] = f.date.dt.strftime("%Y-%m-%d")
    F = {(r["date_s"], r["act_symbol"]): r for r in f.to_dict("records")}

    def strategy(day, chain, quotes, spots, vol, state, eng):
        cohorts = state.setdefault("cohorts", [])
        # drop expired cohorts
        cohorts[:] = [c for c in cohorts if c["exp"] > day]
        orders = []

        # profit-taking: close cohorts whose buy-back cost has fallen enough.
        # Cost is the worst side as always: buy back shorts at ask, sell longs
        # at bid. A leg with no quote cannot be closed, so the cohort is held.
        if profit_take is not None:
            keep = []
            for c in cohorts:
                cost, ok = 0.0, True
                for leg in c["legs"]:
                    q = quotes.get(leg)
                    if q is None:
                        ok = False
                        break
                    px = q[1] if leg.qty < 0 else q[0]   # short->ask, long->bid
                    cost += -leg.qty * px * 100.0
                if ok and c.get("credit", 0) > 0 and \
                        cost <= (1 - profit_take) * c["credit"]:
                    orders.extend([E.Leg(l.symbol, l.expiration, l.strike, l.cp,
                                         -l.qty) for l in c["legs"]])
                else:
                    keep.append(c)
            cohorts[:] = keep

        margin_used = sum(c["margin"] for c in cohorts)
        for cand in selector(day, chain, spots, F):
            m1 = cand["margin"]
            qty = max(int((capital * f_pos) / m1), 0)
            if qty == 0:
                qty = 0 if m1 > capital * f_pos * 3 else 1
            if qty == 0:
                continue
            if margin_used + qty * m1 > util_cap * capital:
                continue
            legs = [E.Leg(l.symbol, l.expiration, l.strike, l.cp, l.qty * qty)
                    for l in cand["legs"]]
            orders.extend(legs)
            credit = 0.0
            for l in legs:
                q = quotes.get(l)
                if q is not None:
                    credit += -l.qty * (q[0] if l.qty < 0 else q[1]) * 100.0
            cohorts.append({"legs": legs, "iv": cand["iv"], "margin": qty * m1,
                            "exp": legs[0].expiration, "hedged": cand["hedged"],
                            "credit": credit})
            margin_used += qty * m1
        # hedge targets
        tgt: dict[str, float] = {}
        for c in cohorts:
            if not c["hedged"]:
                continue
            for leg in c["legs"]:
                sp = spots.get(leg.symbol)
                if sp is None:
                    continue
                T = max((pd.Timestamp(leg.expiration) - pd.Timestamp(day)).days, 0) / 365.0
                d = bs_delta(sp, leg.strike, T, max(c["iv"], 0.05), leg.cp)
                tgt[leg.symbol] = tgt.get(leg.symbol, 0.0) - leg.qty * d * 100.0
        state["hedge_targets"] = tgt
        return orders

    return strategy


def load_feats() -> pd.DataFrame:
    feats = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    feats["date"] = pd.to_datetime(feats.date)
    feats["inv"] = feats.iv_front - feats.iv_back
    feats = feats.sort_values(["act_symbol", "date"])
    feats["mom8"] = feats.groupby("act_symbol").spot.transform(lambda s: s / s.shift(24) - 1)
    feats["mom1w"] = feats.groupby("act_symbol").spot.transform(lambda s: s / s.shift(3) - 1)
    feats["iv_rank"] = feats.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    # credit yield proxy at feature level: iv_front * sqrt(dte) ~ premium/notional;
    # use iv_front directly z-scored + mom8 z-scored per date
    feats["z_cy"] = feats.groupby("date").iv_front.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    feats["z_mom"] = feats.groupby("date").mom8.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return feats


def run_sleeve(name: str, dates=None, capital=1_000_000.0, dev_only=True,
               n_trials=1, f_pos=0.03, profit_take=None):
    import ensemble as X
    feats = load_feats()
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    spot_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    sel, hedge = {
        "A_ps": (make_sel_putspread(), False),
        "A_ps_mom": (make_sel_putspread(mom_gate=True), False),
        "A_ps_2_7": (make_sel_putspread(0.02, 0.07), False),
        "A_ps_6_12": (make_sel_putspread(0.06, 0.12), False),
        "A_ps_spydia": (make_sel_putspread(syms=("SPY", "DIA")), False),
        "A_ss": (sel_A_ss_dh, True),
        "B_str25": (make_sel_B_strangle25(), False),
        "B_str25_hi": (make_sel_B_strangle25(thr=0.09), False),
        "C_short": (make_sel_C_shortonly(), True),
        "C_short_ivr": (make_sel_C_shortonly(ivr_min=0.5), True),
        "C_short_wings": (make_sel_C_shortonly(wings=0.10), True),
        "LD_wide": (make_sel_longdated_wide(), False),
        "LD_wide10": (make_sel_longdated_wide(width=0.10), False),
    }[name]
    dates = dates or E.available_dates()
    if dev_only:
        dates = [d for d in dates if d <= "2024-12-31"]
    strat = make_cohort_strategy(sel, feats, capital, f_pos=f_pos,
                                profit_take=profit_take)
    bt = E.Backtester(dates, capital=capital, stock_hedge=hedge)
    eq, fills, _ = bt.run(strat, spot_panel)
    tag = name + (f"_pt{int(profit_take*100)}" if profit_take else "")
    V.report(eq, f"sleeve_{tag}", n_trials=n_trials)
    X.save_eq(eq, tag)
    return eq


if __name__ == "__main__":
    import sys
    run_sleeve(sys.argv[1] if len(sys.argv) > 1 else "A_ps")


def make_sel_longdated_wide(otm=0.04, width=0.20, dte_lo=50, dte_hi=100,
                            syms=None, max_new=8):
    """Study-16 structure: long-dated, WIDE put credit spread.

    The bid-ask is roughly fixed per leg in dollars while premium scales with
    sqrt(tenor) and risk scales with width, so this corner of the design space
    dilutes a fixed toll over far more premium and far more margin than the
    30-day/5%-wide spreads the rest of this project tested. Requires a positive
    worst-side credit: if the requested strike spacing is unavailable and the
    long leg costs more than the short collects, it is not a credit spread and
    must not be entered.
    """
    universe = syms or LIQUID

    def sel(day, chain, spots, F):
        out = []
        for sym in sorted(universe & set(spots)):
            if len(out) >= max_new:
                break
            s = spots[sym]
            g = chain[(chain.act_symbol == sym) & (chain.dte >= dte_lo)
                      & (chain.dte <= dte_hi)]
            if g.empty:
                continue
            ge = g[g.expiration == g.expiration.min()]
            sl = _nearest(ge, "Put", s * (1 - otm))
            wl = _nearest(ge, "Put", s * (1 - otm - width), need_bid=False)
            if sl is None or wl is None or float(wl.strike) >= float(sl.strike):
                continue
            credit = sl.bid - wl.ask
            if credit <= 0:
                continue
            w = (float(sl.strike) - float(wl.strike)) * 100.0
            legs = [E.Leg(sym, sl.expiration, float(sl.strike), "Put", -1),
                    E.Leg(sym, wl.expiration, float(wl.strike), "Put", +1)]
            out.append(dict(symbol=sym, legs=legs, margin=w,
                            iv=float(sl.vol), hedged=False))
        return out
    return sel
