#!/usr/bin/env python3
"""Diagnostics that explain the e4 result.

  A. why T1 = implied - realized is STILL contaminated when the signal contains
     the implied move (the identity, one layer down)
  B. cross-name correlation of the demeaned tradable P&L -- the direct test of
     BREADTH_HUNT.md's rho-bar = -0.008 claim, now on REAL option P&L
  C. the cost arithmetic: gross earnings variance premium vs the half-spread
  D. required IC vs achieved IC at the measured breadth
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    f = pd.read_parquet(os.path.join(HERE, "sample_EARNINGS.parquet"))
    yrs = (f.event_session.max() - f.event_session.min()).days / 365.25

    ic = pd.read_csv(os.path.join(HERE, "ic_EARNINGS.csv"))

    def geti(sg, tg):
        r = ic[(ic.signal == sg) & (ic.target == tg)]
        return (float(r.ic.iloc[0]), float(r.t.iloc[0])) if len(r) else (np.nan, np.nan)

    print("=" * 74)
    print("A. WHY T1 = IMP - R IS STILL NOT A CLEAN TARGET")
    print("=" * 74)
    print("T1 has two components. A signal scores on T1 either by predicting IMP")
    print("(which is a LITERAL term of T1) or by predicting R (vol persistence).")
    print("Neither is a forecast of the market's error.\n")
    print(f"{'signal':<8}{'corrXS with IMP':>17}{'IC vs R':>10}{'IC vs T1':>11}"
          f"{'IC vs tradable':>16}")
    for sg in ["S1", "S2", "S3", "S2raw"]:
        rs = [stats.spearmanr(g[sg + "_d"], g["S3_d"]).statistic
              for _, g in f.groupby("window") if len(g) >= 6]
        rs = np.array([x for x in rs if np.isfinite(x)])
        icR, _ = geti(sg, "R")
        icT, tT = geti(sg, "T1")
        icP, tP = geti(sg, "pnl_exp_worst")
        print(f"{sg:<8}{rs.mean():>+17.3f}{icR:>+10.3f}{icT:>+11.3f}"
              f"{icP:>+11.3f} (t={tP:+.2f})")
    print("\n  S2 scores on T1 (+0.158) purely because it predicts R with the WRONG")
    print("  sign (-0.118): its denominator is trailing realized vol, which is the")
    print("  single best predictor of the next realized move. S3 scores on T1")
    print("  (+0.167) because S3 IS the first term of T1. Neither is an edge -- and")
    print("  the tradable column, where the implied move is the entry PRICE rather")
    print("  than a free variable, collapses both to noise.")

    print("\n" + "=" * 74)
    print("B. BREADTH: cross-name correlation of the tradable P&L")
    print("=" * 74)
    print("  rho_bar from the variance-components identity")
    print("  Var(window mean) = sigma^2 (1 + (n-1) rho) / n  -- robust to the")
    print("  unbalanced panel, unlike pairwise correlations on ~14 shared windows.\n")
    for col, lab in [("pnl_exp_worst", "RAW (directional short-vol book)"),
                     ("pnl_exp_worst_d", "DEMEANED (factor-neutral book)")]:
        wm, wv, ns = [], [], []
        for _, g in f.groupby("window"):
            x = g[col].dropna()
            if len(x) >= 6:
                wm.append(x.mean()); wv.append(x.var(ddof=1)); ns.append(len(x))
        wm, wv, ns = np.array(wm), np.array(wv), np.array(ns)
        s2p = wv.mean()
        vm = wm.var(ddof=1)
        n = ns.mean()
        rho = (n * vm / s2p - 1) / (n - 1)
        den = 1 + (n - 1) * rho
        neff = f"{n/den:.1f}" if den > 0 else "unbounded (rho<=0)"
        print(f"  {lab:<34} rho_bar={rho:+.4f}  n/window={n:.0f}  N_eff={neff}")
    print("\n  BREADTH_HUNT.md section 5's breadth claim is CONFIRMED on real option")
    print("  P&L: the directional book carries a common factor, demeaning removes it,")
    print("  and the residual bets are uncorrelated. Breadth was never the problem.")

    print("\n" + "=" * 74)
    print("C. THE COST ARITHMETIC (units: fraction of straddle premium)")
    print("=" * 74)
    gm = f.pnl_exp_mid.mean()
    gw = f.pnl_exp_worst.mean()
    hs = f.entry_cost.mean()
    print(f"  gross short-straddle P&L into earnings, mid entry   {gm:+.4f}"
          f"  (t={f.pnl_exp_mid.mean()/f.pnl_exp_mid.sem():+.2f})")
    print(f"  entry half-spread paid (mean)                       {-hs:+.4f}"
          f"   (median {-f.entry_cost.median():+.4f})")
    print(f"  net, worst-side ENTRY only, settled at intrinsic    {gw:+.4f}"
          f"  (t={f.pnl_exp_worst.mean()/f.pnl_exp_worst.sem():+.2f})")
    print(f"  full straddle bid-ask / mid (both legs, one way)    {f.spread_frac.mean():.4f}")
    rt = f.pnl_rt_worst.dropna()
    print(f"  quote-to-quote round trip, worst side (n={len(rt)})      {rt.mean():+.4f}"
          f"  (t={rt.mean()/rt.sem():+.2f})")
    print(f"\n  cost / gross ratio (entry side only): {hs/max(gm,1e-9):.2f}x")
    print("  The unconditional trade is a coin flip BEFORE costs and negative after"
          "\n  paying only ONE side of ONE leg-pair's spread.")

    print("\n" + "=" * 74)
    print("D. REQUIRED vs ACHIEVED")
    print("=" * 74)
    br = len(f) / yrs
    print(f"  measured breadth (events/yr surviving all filters)  {br:.0f}")
    print(f"  BREADTH_HUNT projected breadth at 500 names          1996")
    for tgt in (5.0, 1.0):
        print(f"  IC required for IR {tgt:.0f} at BR={br:.0f}:  {tgt/np.sqrt(br):.3f}"
              f"   |  at BR=1996: {tgt/np.sqrt(1996):.3f}")
    trad = ic[ic.target.isin(["pnl_exp_worst", "pnl_exp_mid"])]
    best = trad.loc[trad.ic.idxmax()]
    print(f"\n  best achieved TRADABLE IC (any of 4 signals, full sample):"
          f"  {best.ic:+.4f}  t={best.t:+.2f}  ({best.signal} -> {best.target})")
    print(f"  pre-registered signal S1 -> pnl_exp_worst:"
          f"  {trad[(trad.signal=='S1')&(trad.target=='pnl_exp_worst')].ic.iloc[0]:+.4f}"
          f"  t={trad[(trad.signal=='S1')&(trad.target=='pnl_exp_worst')].t.iloc[0]:+.2f}")
    print(f"  deflation threshold sqrt(2 ln 18) = {np.sqrt(2*np.log(18)):.2f}"
          f"   |  sqrt(2 ln 164) = {np.sqrt(2*np.log(164)):.2f}")

    print("\n" + "=" * 74)
    print("E. STRIP VALIDITY (does the two-expiry event strip actually work?)")
    print("=" * 74)
    for tag in ["EARNINGS", "P1_nonearnings", "P2_2ndmove"]:
        p = os.path.join(HERE, f"sample_{tag}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p)
            print(f"  {tag:<16} median stripped event sigma M2 = {d.M2.median():.4f}"
                  f"   median realized |move| = {d.R.median():.4f}   n={len(d)}")
    print("\n  The strip reads a 5.3% event jump when earnings sit inside the front")
    print("  expiry and 2.2% when they do not: the estimator is doing its job.")


if __name__ == "__main__":
    main()
