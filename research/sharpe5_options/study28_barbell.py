#!/usr/bin/env python3
"""Study 28: the self-funding barbell — the last untested structure family.

Three ideas, none tested in studies 1-27:

  A. PUT RATIO BACKSPREAD. Sell one ~5%-OTM put, buy N far-OTM (12-15%) puts
     with the credit. Net cost ~zero: the VRP harvested on the short leg pays
     the tail-insurance bleed. Calm markets: small win. Crash: the long puts
     pay multiples while the short leg's loss is linear. The known weakness is
     the "valley" — a moderate decline that breaches the short strike but not
     far enough for the longs to pay. On SPX the convex skew (b2=12.9) makes
     far puts expensive in IV terms, so the funding is fighting the skew.

  B. CONDITIONAL TAIL BUYING. Study 27 showed unconditional far-OTM puts lose
     even with hindsight exits. Untested: buy them ONLY when the market's own
     stress detector fires (SPY term inversion, iv_front > iv_back), which
     historically precedes the vol spikes that make puts pay.

  C. THE LEVERED COMBINATION. If A or B produces a crash-bounded carry book,
     it can support leverage the naked ladder cannot. Kelly arithmetic decides
     the ceiling honestly.

All entries at worst-side fills (sell bid / buy ask), monthly cohorts, held to
expiry with the pre-stated 10x monetisation trigger on long legs.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E
from structures import settle

HERE = os.path.dirname(os.path.abspath(__file__))


def build_cohorts():
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    spy = panel["SPY"].dropna()
    spy.index = pd.to_datetime(spy.index)
    dates = E.available_dates()

    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    spyf = f[f.act_symbol == "SPY"].set_index("date")
    inv = (spyf.iv_front - spyf.iv_back)          # >0 = stress inversion

    def se_of(exp):
        b = spy[spy.index <= pd.Timestamp(exp)]
        a = spy[spy.index > pd.Timestamp(exp)]
        if len(b) and (pd.Timestamp(exp) - b.index[-1]).days <= 4:
            return float(b.iloc[-1])
        if len(a) and (a.index[0] - pd.Timestamp(exp)).days <= 4:
            return float(a.iloc[0])
        return None

    rows, last_m = [], None
    for day in dates:
        m = day[:7]
        if m == last_m:
            continue
        ch = E.load_chain(day)
        g = ch[(ch.act_symbol == "SPY") & (ch.call_put == "Put")
               & (ch.dte >= 45) & (ch.dte <= 75) & (ch.ask > 0)]
        ts = pd.Timestamp(day)
        if g.empty or ts not in spy.index:
            continue
        s = float(spy.loc[ts])
        exp = g.expiration.min()
        ge = g[g.expiration == exp]
        near = ge[ge.bid > 0]
        if near.empty:
            continue
        sn = near.loc[(near.strike - s * 0.95).abs().idxmin()]     # short leg
        fl = ge.loc[(ge.strike - s * 0.87).abs().idxmin()]         # far long
        Ks, Kf = float(sn.strike), float(fl.strike)
        if Kf >= Ks:
            continue
        se = se_of(exp)
        if se is None:
            continue
        last_m = m
        # 10x monetisation path for the far long (real bids)
        far_exit = None
        for d2 in [x for x in dates if day < x <= exp]:
            ch2 = E.load_chain(d2)
            q = ch2[(ch2.act_symbol == "SPY") & (ch2.expiration == exp)
                    & (ch2.call_put == "Put") & (ch2.strike == Kf)]
            if len(q) and float(q.bid.iloc[0]) >= 10 * float(fl.ask):
                far_exit = float(q.bid.iloc[0])
                break
        far_pay = far_exit if far_exit is not None else settle(se, Kf, "Put")
        stress = float(inv.loc[ts]) if ts in inv.index else np.nan
        rows.append(dict(
            date=ts, exp=exp, spot=s, se=se,
            short_bid=float(sn.bid), short_K=Ks,
            far_ask=float(fl.ask), far_K=Kf, far_pay=far_pay,
            short_loss=settle(se, Ks, "Put"),
            stress=stress))
    return pd.DataFrame(rows)


def perf(r: pd.Series, label: str):
    r = r.dropna()
    if len(r) < 24:
        print(f"  {label:<44} insufficient")
        return
    eq = (1 + r).cumprod()
    if (eq <= 0).any():
        first = eq[eq <= 0].index[0]
        print(f"  {label:<44} RUIN — equity <= 0 at {str(first)[:10]}")
        return None
    yrs = len(r) / 12
    cagr = float(eq.iloc[-1]) ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    sh = r.mean() / r.std() * math.sqrt(12) if r.std() > 0 else np.nan
    wy = (1 + r).groupby(r.index.year).prod() - 1
    print(f"  {label:<44} CAGR={cagr:+7.1%} maxDD={dd:+7.1%} Sharpe={sh:+.2f} "
          f"worstYr={wy.min():+.1%}")
    return cagr, dd, sh


def main():
    d = build_cohorts()
    print(f"cohorts: {len(d)}  {d.date.min().date()} -> {d.date.max().date()}")
    print(f"short 5%-OTM put bid: {(d.short_bid/d.spot).mean():.3%} of spot | "
          f"far 13%-OTM ask: {(d.far_ask/d.spot).mean():.3%}")

    # per-cohort P&L per 1x short notional unit, as fraction of spot
    def pnl(n_far):
        return (d.short_bid - d.short_loss
                + n_far * (d.far_pay - d.far_ask)) / d.spot

    print("\n=== A. put ratio backspreads (monthly, per-cohort P&L % of spot) ===")
    for n in (0, 1, 2, 3, 4):
        p = pnl(n)
        lab = "short put only" if n == 0 else f"1 short : {n} long far"
        cost = (d.short_bid - n * d.far_ask) / d.spot
        print(f"  {lab:<22} net credit={cost.mean():+.3%}  mean={p.mean():+.4%}  "
              f"worst={p.min():+.2%}  hit={100*(p>0).mean():.0f}%")

    # monthly return series on committed capital (reserve = 8% of spot per unit,
    # the max realistic loss of the 5%-OTM short after far-put offset)
    print("\n=== portfolio: monthly cohorts on committed capital (8% spot reserve) ===")
    res = {}
    for n in (0, 2, 3):
        r = pd.Series((pnl(n) / 0.08).values, index=d.date)
        res[n] = r
        perf(r, f"backspread 1:{n}" if n else "short put only")

    print("\n=== B. conditional tail buying (stress = SPY term inversion > 0) ===")
    ds = d.dropna(subset=["stress"])
    far_r = (ds.far_pay - ds.far_ask) / ds.far_ask
    for lab, mask in (("always buy far put", np.ones(len(ds), bool)),
                      ("only when inverted (stress)", (ds.stress > 0).values),
                      ("only when calm", (ds.stress <= 0).values)):
        sel = far_r[mask]
        if len(sel) < 10:
            continue
        print(f"  {lab:<30} n={len(sel):>3}  mean mult={1+sel.mean():.2f}x  "
              f"hit={(sel>0).mean():.1%}")

    print("\n=== C. levered combination with the v3 ladder ===")
    # ladder monthly returns from its saved equity curve
    lad_p = os.path.join(HERE, "results", "eq_LD_wide.parquet")
    if os.path.exists(lad_p):
        lad = pd.read_parquet(lad_p)["equity"]
        lad_m = lad.resample("MS").last().pct_change().dropna()
        best_n = 3
        bs = pd.Series((pnl(best_n) / 0.08).values, index=d.date).resample("MS").sum()
        both = pd.DataFrame({"ladder": lad_m, "bs": bs}).dropna()
        print(f"  overlap n={len(both)}  corr(ladder, backspread)="
              f"{both.corr().iloc[0,1]:+.3f}")
        for L in (1, 2, 3, 5):
            r = L * (0.5 * both.ladder + 0.5 * both.bs) - max(L - 1, 0) * 0.05 / 12
            perf(r, f"50/50 book levered {L}x (5% borrow)")
    print("\nKelly note: max growth = S^2/2 still binds any near-Gaussian book;")
    print("only a crash-POSITIVE book escapes it, and only if its carry is >= 0.")


if __name__ == "__main__":
    main()
