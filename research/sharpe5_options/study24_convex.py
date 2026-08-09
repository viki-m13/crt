#!/usr/bin/env python3
"""Study 24: BUYING convexity — the untested direction.

Everything in this project sold premium or traded stock. Both are
near-Gaussian, so growth is capped at S^2/2 regardless of leverage, which is
why Sharpe 5 and 1000% looked unreachable. That bound does NOT apply to convex
payoffs. For a binary bet at odds b with hit rate p, Kelly log-growth is

    g = p*log(1+f*b) + (1-p)*log(1-f),  maximised at f* = (p*b-(1-p))/b

A 10:1 payoff hit 15% of the time gives f*=6.5% and g=0.018 per bet. At 250
independent bets a year that compounds to ~9000%. Convexity plus frequency
genuinely reaches the target — IF the predictive power exists.

And I have never tested for it. Study 9 measured signals against the LOSS term
of short premium, i.e. it asked "when do spreads get breached?" — which is the
same question as "when do large moves happen?", just used in the selling
direction. Vol-of-vol scored IC -0.033 (t=-5.30): high vol-of-vol predicts
bigger losses for sellers, which means bigger MOVES. I used it to avoid
selling. This buys instead.

Structures tested (all bought at the ASK, held to expiry, settled at intrinsic):
  straddle   ATM call + put            - most expensive, least convex
  strangle5  5% OTM both sides         - cheaper, more convex
  strangle10 10% OTM both sides        - lottery ticket, most convex

For each, the payoff distribution, the Kelly-optimal fraction, and the implied
log-growth are reported both unconditionally and conditioned on the signals.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E
from structures import settle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "convex.parquet")
LIQUID = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA "
             "XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST "
             "WMT DIS CAT GE PFE CVX WFC MS GS".split())


def kelly_growth(returns: np.ndarray, max_f=0.5):
    """Optimal fraction and log-growth per bet for an empirical return dist.

    returns are per-unit-staked P&L (-1 = total loss of the premium).
    """
    r = returns[np.isfinite(returns)]
    if len(r) < 50 or r.min() <= -1.0000001:
        r = np.clip(r, -0.999999, None)
    best_f, best_g = 0.0, 0.0
    for f in np.linspace(0.005, max_f, 200):
        g = np.mean(np.log1p(f * r))
        if np.isfinite(g) and g > best_g:
            best_f, best_g = f, g
    return best_f, best_g


def main(step=2):
    dates = E.available_dates()[::step]
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    hist = {s: panel[s].dropna() for s in panel.columns}

    feats = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    feats["date"] = pd.to_datetime(feats.date)
    feats = feats.sort_values(["act_symbol", "date"])
    feats["vov"] = feats.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(24, min_periods=8).std())
    feats["ivrv"] = feats.iv_front - feats.rv_ewma
    feats["iv_rank"] = feats.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    fmap = {(r.date.strftime("%Y-%m-%d"), r.act_symbol): r
            for r in feats.itertuples(index=False)}

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
        ch = ch[(ch.dte >= 20) & (ch.dte <= 60) & (ch.ask > 0)]
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
            se = se_of(sym, exp)
            if se is None or abs(math.log(se / s)) > 0.8:
                continue
            fr = fmap.get((day, sym))
            move = abs(se / s - 1.0)
            for otm, lab in ((0.0, "straddle"), (0.05, "strangle5"),
                             (0.10, "strangle10")):
                cs = ge[(ge.call_put == "Call") & (ge.ask > 0)]
                ps = ge[(ge.call_put == "Put") & (ge.ask > 0)]
                if cs.empty or ps.empty:
                    continue
                c = cs.loc[(cs.strike - s * (1 + otm)).abs().idxmin()]
                p = ps.loc[(ps.strike - s * (1 - otm)).abs().idxmin()]
                cost = (float(c.ask) + float(p.ask)) * 100.0
                if cost <= 0:
                    continue
                payoff = (settle(se, float(c.strike), "Call")
                          + settle(se, float(p.strike), "Put")) * 100.0
                rows.append((day, sym, lab, cost, payoff, (payoff - cost) / cost,
                             move, int(ge.dte.iloc[0]),
                             getattr(fr, "vov", np.nan) if fr else np.nan,
                             getattr(fr, "ivrv", np.nan) if fr else np.nan,
                             getattr(fr, "iv_rank", np.nan) if fr else np.nan,
                             getattr(fr, "skew25", np.nan) if fr else np.nan))
        if (di + 1) % 100 == 0:
            print(f"convex {di+1}/{len(dates)} rows={len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=["date", "sym", "structure", "cost", "payoff",
                                    "ret", "move", "dte", "vov", "ivrv",
                                    "iv_rank", "skew25"])
    d.to_parquet(OUT, index=False)
    report(d)


def report(d):
    d = d.copy()
    d["date"] = pd.to_datetime(d.date)
    print("\n" + "=" * 90)
    print("STUDY 24 — BUYING convexity. Return is per unit of premium staked.")
    print("=" * 90)
    for st in ("straddle", "strangle5", "strangle10"):
        g = d[d.structure == st]
        if len(g) < 200:
            continue
        f, gr = kelly_growth(g.ret.values)
        rounds = 365 / max(g.dte.mean(), 1)
        print(f"\n--- {st} (n={len(g):,}, mean dte {g.dte.mean():.0f}) ---")
        print(f"  mean ret={g.ret.mean():+.3f}  median={g.ret.median():+.3f}  "
              f"win%={(g.ret>0).mean():.1%}  max={g.ret.max():+.1f}x")
        print(f"  p90={g.ret.quantile(.9):+.2f}  p99={g.ret.quantile(.99):+.2f}")
        print(f"  Kelly f*={f:.3f}  log-growth/bet={gr:+.5f}  "
              f"-> annualized {math.exp(gr*rounds)-1:+.1%}")
        # conditioned on the large-move signals
        for sig, lo_hi in (("vov", "hi"), ("iv_rank", "lo"), ("ivrv", "lo")):
            gg = g.dropna(subset=[sig])
            if len(gg) < 200:
                continue
            gg = gg.copy()
            gg["q"] = gg.groupby("date")[sig].transform(
                lambda s: s.rank(pct=True) if s.notna().sum() >= 8 else np.nan)
            sel = gg[gg.q > 0.75] if lo_hi == "hi" else gg[gg.q < 0.25]
            if len(sel) < 150:
                continue
            f2, g2 = kelly_growth(sel.ret.values)
            print(f"    {sig} {'high' if lo_hi=='hi' else 'low'} quartile: "
                  f"mean={sel.ret.mean():+.3f} win%={(sel.ret>0).mean():.1%} "
                  f"Kelly f*={f2:.3f} g={g2:+.5f} "
                  f"-> ann {math.exp(g2*rounds)-1:+.1%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(pd.read_parquet(OUT))
    else:
        main()
