#!/usr/bin/env python3
"""Study 9: the ceiling, recomputed on the *predictable* component only.

Study 5 measured IC of credit_yield against realized return and got +0.037
(t=+5.54). That number is contaminated by an accounting identity. For a
premium-selling structure held to expiry:

    ret = credit_yield - loss/margin

and the credit is KNOWN AT ENTRY — it is observed, not forecast. With 76% of
credit spreads expiring worthless, ret literally equals credit_yield most of
the time, so regressing ret on credit_yield partly regresses a variable on
itself. The IC looks strong and carries no tradable information, which is
exactly why every sleeve built on it lost money.

The honest question is: how well can any signal predict the only uncertain
term, the LOSS? This script measures IC against -loss/margin for signals that
are not components of the payoff, and re-derives the fundamental-law ceiling
from that.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
LIQUID = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA "
             "XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST "
             "WMT DIS CAT GE PFE CVX WFC MS GS".split())
# signals that are NOT mechanically part of the payoff
CLEAN = ["mom8", "skew25", "iv_rank", "vov", "idio_inv", "ivrv"]


def main(structure="short_strangle25"):
    st = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    st["date"] = pd.to_datetime(st.date)
    d = st[(st.structure == structure) & st.act_symbol.isin(LIQUID)].copy()
    d["ret"] = d.pnl / d.margin
    d["cy"] = d.credit / d.margin
    d["neg_loss"] = -(d.credit - d.pnl) / d.margin   # -loss/margin

    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    f = f.sort_values(["act_symbol", "date"])
    f["ivrv"] = f.iv_front - f.rv_ewma
    f["inv"] = f.iv_front - f.iv_back
    f["mom8"] = f.groupby("act_symbol").spot.transform(lambda s: s / s.shift(24) - 1)
    f["iv_rank"] = f.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    f["vov"] = f.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(24, min_periods=8).std())
    spy = f[f.act_symbol == "SPY"][["date", "inv"]].rename(columns={"inv": "spy_inv"})
    f = f.merge(spy, on="date", how="left")
    f["idio_inv"] = f.inv - f.spy_inv
    d = d.merge(f.drop(columns=["spot"]), on=["date", "act_symbol"], how="left")

    print("=" * 72)
    print(f"STUDY 9 — true ceiling, {structure}")
    print("=" * 72)
    print(f"\nThe payoff identity: ret = credit_yield - loss/margin")
    print(f"  trades with zero loss: {(d.neg_loss >= -1e-9).mean():.1%}")
    print(f"  IC(credit_yield -> ret)      = "
          f"{spearmanr(d.cy.dropna(), d.ret[d.cy.notna()]).statistic:+.4f}   [CONTAMINATED]")

    print(f"\nIC of clean signals against the only uncertain term (-loss):")
    best_ic = 0.0
    for sig in CLEAN:
        ics = []
        for _, g in d.groupby("date"):
            g = g[[sig, "neg_loss"]].dropna()
            if len(g) >= 10 and g[sig].nunique() > 3:
                ic = spearmanr(g[sig], g.neg_loss).statistic
                if np.isfinite(ic):
                    ics.append(ic)
        if len(ics) < 50:
            continue
        ics = np.array(ics)
        t = ics.mean() / (ics.std() / math.sqrt(len(ics)))
        print(f"  {sig:>10}: IC={ics.mean():+.4f}  t={t:+.2f}  n={len(ics)}")
        best_ic = max(best_ic, abs(ics.mean()))

    # effective breadth (reuse the participation-ratio measure)
    panel = d.pivot_table(index="date", columns="act_symbol", values="ret")
    R = panel.dropna(axis=1, thresh=int(0.4 * len(panel)))
    R = R.loc[:, R.std() > 0]
    C = np.nan_to_num(R.corr().values, nan=0.0)
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    n_eff = float(lam.sum() ** 2 / (lam ** 2).sum())

    rounds = 12.0
    br = n_eff * rounds
    print(f"\nBreadth: {R.shape[1]} names -> {n_eff:.1f} effective; "
          f"{br:.0f} independent bets/yr")
    print(f"\nCEILING on the predictable component:")
    print(f"  best clean |IC| = {best_ic:.4f}")
    print(f"  IR = IC x sqrt(BR) = {best_ic * math.sqrt(br):.2f}")
    print(f"  (Study 5 reported 0.48 using the contaminated IC of 0.037.)")
    need_ic = 5.0 / math.sqrt(br)
    print(f"  For IR=5 at this breadth you need IC={need_ic:.3f} "
          f"= {need_ic/max(best_ic,1e-9):.0f}x the best clean signal.")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "short_strangle25")


