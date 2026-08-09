#!/usr/bin/env python3
"""Study 20: optimize the deployed ladder, at honest worst-side fills.

Baseline (deployed): SPY, ~83 DTE, short 3% OTM, 3% wide, weekly rungs, 3% of
equity per rung, 60% cap. At real worst-side quotes: Sharpe 1.07, CAGR 5.3%,
maxDD -18.8%.

Levers, in the order the evidence says they matter:

  WIDTH   the bid-ask toll is fixed per leg while risk scales with width, so
          wider spreads dilute a fixed cost over more margin (study 16). The
          deployed 3% width is the most cost-punished cell measured.
  OTM     further out lowers breach probability but collects less; study 15
          showed the market prices this close to fairly, so the question is
          purely where the toll/premium ratio is best.
  DTE     premium scales ~sqrt(tenor) against a fixed toll.
  CADENCE more rungs = more entry-timing diversification, which is where the
          ladder's Sharpe advantage comes from in the first place.
  CAP     more capital deployed raises return and drawdown together.

Everything is scored at worst-side fills on the same SPY chain data.
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np
import pandas as pd

import engine as E
from structures import settle

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = {}


def load_spy():
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    return panel


def build(otm, width, dte_lo, dte_hi, sym="SPY", panel=None):
    key = (otm, width, dte_lo, dte_hi, sym)
    if key in CACHE:
        return CACHE[key]
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
        sh, lg = ge[ge.bid > 0], ge[ge.ask > 0]
        if sh.empty or lg.empty:
            continue
        sl = sh.loc[(sh.strike - s * (1 - otm)).abs().idxmin()]
        wl = lg.loc[(lg.strike - s * (1 - otm - width)).abs().idxmin()]
        ks, kl = float(sl.strike), float(wl.strike)
        if kl >= ks:
            continue
        w = (ks - kl) * 100.0
        nat = (float(sl.bid) - float(wl.ask)) * 100.0
        if nat <= 0:
            continue
        loss = (settle(se, ks, "Put") - settle(se, kl, "Put")) * 100.0
        rows.append((day, exp, w, nat, loss, s))
    d = pd.DataFrame(rows, columns=["date", "exp", "w", "nat", "loss", "spot"])
    if len(d):
        d["date"] = pd.to_datetime(d.date)
    CACHE[key] = d
    return d


def ladder(d, rung_eq=0.03, cap=0.60, cadence="W", size_fn=None):
    if d is None or len(d) < 40:
        return None
    d = d.sort_values("date").copy()
    d["per"] = d.date.dt.to_period(cadence)
    d = d.groupby("per", as_index=False).first()
    equity, open_r, curve = 1.0, [], []
    for r in d.itertuples(index=False):
        still = []
        for exp, capital, cf, lf in open_r:
            if exp <= r.date.strftime("%Y-%m-%d"):
                equity += capital * (cf - lf)
            else:
                still.append((exp, capital, cf, lf))
        open_r = still
        at_risk = sum(c for _, c, _, _ in open_r)
        mult = 1.0 if size_fn is None else size_fn(r)
        want = rung_eq * equity * mult
        if want > 0 and at_risk + want <= cap * equity:
            open_r.append((r.exp, want, r.nat / r.w, r.loss / r.w))
        curve.append((r.date, equity))
    for exp, capital, cf, lf in open_r:
        equity += capital * (cf - lf)
    eq = pd.Series(dict(curve)).sort_index()
    if len(eq) < 30:
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rr = eq.pct_change().dropna()
    ppy = len(eq) / max(yrs, 1e-9)
    return {"cagr": equity ** (1 / max(yrs, 1e-9)) - 1,
            "sharpe": rr.mean() / rr.std() * math.sqrt(ppy) if rr.std() > 0 else np.nan,
            "dd": float((eq / eq.cummax() - 1).min()), "eq": eq, "n": len(d)}


def main():
    panel = load_spy()
    print("=" * 92)
    print("STUDY 20 — optimizing the deployed ladder at WORST-SIDE fills")
    print("=" * 92)
    base = build(0.03, 0.03, 60, 110, panel=panel)
    b = ladder(base)
    print(f"\nBASELINE (deployed: 3% OTM, 3% wide, 83DTE, weekly, 3%/rung, 60% cap)")
    print(f"  CAGR={b['cagr']:+.1%}  Sharpe={b['sharpe']:+.2f}  maxDD={b['dd']:+.1%}\n")

    print("--- WIDTH sweep (deployed=3%) ---")
    print(f"  {'width':>7} {'CAGR':>8} {'Sharpe':>8} {'maxDD':>8} {'credit/w':>9}")
    for wd in (0.03, 0.05, 0.08, 0.12, 0.20):
        d = build(0.03, wd, 60, 110, panel=panel)
        r = ladder(d)
        if r:
            print(f"  {wd:>7.0%} {r['cagr']:>8.1%} {r['sharpe']:>8.2f} "
                  f"{r['dd']:>8.1%} {(d.nat/d.w).mean():>9.4f}")

    print("\n--- OTM sweep at best width (deployed=3%) ---")
    for otm in (0.02, 0.03, 0.05, 0.07, 0.10):
        d = build(otm, 0.08, 60, 110, panel=panel)
        r = ladder(d)
        if r:
            print(f"  otm={otm:>4.0%} CAGR={r['cagr']:+7.1%} Sharpe={r['sharpe']:+.2f} "
                  f"maxDD={r['dd']:+7.1%} credit/w={(d.nat/d.w).mean():.4f}")

    print("\n--- TENOR sweep (deployed=83d) ---")
    for lo, hi, lab in ((15, 45, "30d"), (45, 75, "60d"), (60, 110, "83d"),
                        (110, 200, "150d"), (200, 400, "300d")):
        d = build(0.03, 0.08, lo, hi, panel=panel)
        r = ladder(d)
        if r:
            print(f"  {lab:>5}: CAGR={r['cagr']:+7.1%} Sharpe={r['sharpe']:+.2f} "
                  f"maxDD={r['dd']:+7.1%} n={r['n']}")

    print("\n--- CADENCE / CAP (deployed = weekly, 60%) ---")
    d = build(0.03, 0.08, 60, 110, panel=panel)
    for cad in ("W", "2W", "M"):
        for cap in (0.4, 0.6, 0.9):
            r = ladder(d, cap=cap, cadence=cad)
            if r:
                print(f"  cadence={cad:>2} cap={cap:>4.0%}: CAGR={r['cagr']:+7.1%} "
                      f"Sharpe={r['sharpe']:+.2f} maxDD={r['dd']:+7.1%}")

    print("\n--- RUNG SIZE (deployed = 3% of equity) ---")
    for re_ in (0.02, 0.03, 0.05, 0.08):
        r = ladder(d, rung_eq=re_, cap=0.9)
        if r:
            print(f"  rung={re_:>4.0%}: CAGR={r['cagr']:+7.1%} "
                  f"Sharpe={r['sharpe']:+.2f} maxDD={r['dd']:+7.1%}")


if __name__ == "__main__":
    main()
