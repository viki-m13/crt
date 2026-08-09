#!/usr/bin/env python3
"""Study 19: the deployed SPX ladder, priced at real market quotes.

The live site's strategy: SPY/SPX put credit spread, ~83 DTE, short leg ~3% OTM,
long leg ~6% OTM (3% wide), one new rung per week, 3% of equity per rung,
ladder capped at 60% of equity.

Its own live_validation.json reports booked_vs_natural of 1.413 (SPY) and 1.582
(SPX) with model_conservative:false — the backtest books 41-58% more credit than
the market pays, because IV is modelled as 1.15*sqrt(0.3*rv60^2+0.7*rvbar^2)
(0.181) against actual leg IVs of 0.149/0.173.

This runs the identical ladder three ways — model credit, real mid, real
worst-side — so the gap is measured rather than argued. Everything else
(rung cadence, sizing, cap, settlement) is held fixed.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E
from structures import settle

HERE = os.path.dirname(os.path.abspath(__file__))
MARKUP = 1.413            # measured booked_vs_natural for SPY
RUNG_EQUITY = 0.03        # 3% of equity per rung
LADDER_CAP = 0.60         # max 60% of equity at risk
SYM = "SPY"


def build_trades(otm=0.03, width=0.03, dte_lo=60, dte_hi=110, sym=SYM):
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    if sym not in panel.columns:
        return None
    hist = panel[sym].dropna()

    def se_of(exp):
        b = hist[hist.index <= exp]
        a = hist[hist.index > exp]
        if len(b) and (pd.Timestamp(exp) - pd.Timestamp(b.index[-1])).days <= 4:
            return float(b.iloc[-1])
        if len(a) and (pd.Timestamp(a.index[0]) - pd.Timestamp(exp)).days <= 4:
            return float(a.iloc[0])
        return None

    rows = []
    for day in E.available_dates():
        ch = E.load_chain(day)
        g = ch[(ch.act_symbol == sym) & (ch.dte >= dte_lo) & (ch.dte <= dte_hi)
               & (ch.call_put == "Put")]
        if g.empty or day not in panel.index:
            continue
        s = panel.at[day, sym]
        if not np.isfinite(s):
            continue
        exp = g.expiration.min()
        ge = g[g.expiration == exp]
        se = se_of(exp)
        if se is None:
            continue
        sh = ge[ge.bid > 0]
        lg = ge[ge.ask > 0]
        if sh.empty or lg.empty:
            continue
        sl = sh.loc[(sh.strike - s * (1 - otm)).abs().idxmin()]
        wl = lg.loc[(lg.strike - s * (1 - otm - width)).abs().idxmin()]
        ks, kl = float(sl.strike), float(wl.strike)
        if kl >= ks:
            continue
        w = (ks - kl) * 100.0
        nat = (float(sl.bid) - float(wl.ask)) * 100.0
        mid = ((float(sl.bid) + float(sl.ask)) / 2
               - (float(wl.bid) + float(wl.ask)) / 2) * 100.0
        if nat <= 0:
            continue
        loss = (settle(se, ks, "Put") - settle(se, kl, "Put")) * 100.0
        rows.append((day, exp, w, nat, mid, loss))
    d = pd.DataFrame(rows, columns=["date", "exp", "w", "nat", "mid", "loss"])
    d["date"] = pd.to_datetime(d.date)
    return d


def ladder(d: pd.DataFrame, credit_col, label, weekly=True):
    """Simulate the rung ladder: one new rung per week, 3% of equity each,
    ladder capped at 60% of equity. Equity compounds on realized P&L."""
    if d is None or not len(d):
        return None
    d = d.sort_values("date").copy()
    if weekly:                      # one rung per calendar week
        d["wk"] = d.date.dt.to_period("W")
        d = d.groupby("wk", as_index=False).first()
    equity = 1.0
    open_rungs = []               # (exp, risk_capital, credit_frac, loss_frac)
    curve = []
    for r in d.itertuples(index=False):
        # settle matured rungs
        still = []
        for exp, cap, cfrac, lfrac in open_rungs:
            if exp <= r.date.strftime("%Y-%m-%d"):
                equity += cap * (cfrac - lfrac)
            else:
                still.append((exp, cap, cfrac, lfrac))
        open_rungs = still
        at_risk = sum(c for _, c, _, _ in open_rungs)
        if at_risk + RUNG_EQUITY * equity <= LADDER_CAP * equity:
            cap = RUNG_EQUITY * equity
            open_rungs.append((r.exp, cap, getattr(r, credit_col) / r.w,
                               r.loss / r.w))
        curve.append((r.date, equity))
    for exp, cap, cfrac, lfrac in open_rungs:
        equity += cap * (cfrac - lfrac)
    eq = pd.Series(dict(curve)).sort_index()
    if len(eq) < 30:
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (equity / 1.0) ** (1 / max(yrs, 1e-9)) - 1
    rr = eq.pct_change().dropna()
    sh = rr.mean() / rr.std() * math.sqrt(52) if rr.std() > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    print(f"  {label:<40} CAGR={cagr:+7.1%} Sharpe={sh:+.2f} maxDD={dd:+7.1%} "
          f"rungs={len(d)}")
    return {"cagr": cagr, "sharpe": sh, "dd": dd, "eq": eq}


def main():
    print("=" * 84)
    print("STUDY 19 — the deployed SPX ladder at real market quotes")
    print("=" * 84)
    d = build_trades()
    if d is None or not len(d):
        print("no trades built")
        return
    print(f"\nSPY, ~83 DTE, 3% OTM short, 3% wide. entries={len(d):,} "
          f"({d.date.min().date()} -> {d.date.max().date()})")
    print(f"  natural credit/width {(d.nat/d.w).mean():.4f}   "
          f"mid {(d.mid/d.w).mean():.4f}   realized loss {(d.loss/d.w).mean():.4f}\n")
    d["model"] = d.nat * MARKUP
    print("Ladder simulation (weekly rungs, 3% equity each, 60% cap):")
    res = {}
    for col, lab in (("model", f"booked at MODEL credit ({MARKUP:.3f}x natural)"),
                     ("mid", "booked at real MID"),
                     ("nat", "booked at real NATURAL (worst-side)")):
        res[col] = ladder(d, col, lab)
    if res.get("model") and res.get("nat"):
        print(f"\n  overstatement: model CAGR {res['model']['cagr']:.1%} vs "
              f"real worst-side {res['nat']['cagr']:.1%}")
    # per-year on the honest version
    if res.get("nat"):
        eq = res["nat"]["eq"]
        print("\n  honest (worst-side) equity by year:")
        for y, g in eq.groupby(eq.index.year):
            print(f"    {y}: {g.iloc[-1]/g.iloc[0]-1:+7.1%}")


if __name__ == "__main__":
    main()
