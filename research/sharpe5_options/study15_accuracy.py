#!/usr/bin/env python3
"""Study 15: can breaches be predicted better than the market already prices?

This is the core question stated precisely. For a vertical credit spread held
to expiry:

    return = credit/width  -  loss/width

The credit is KNOWN at entry. All uncertainty lives in the loss. So the entire
predictive problem is forecasting breaches — and there is a hard benchmark for
it, because the market has already published its own forecast:

    credit/width = the RISK-NEUTRAL expected loss ratio

A spread's value under Q is exactly the discounted expected loss, so the credit
you are paid, divided by the width you risk, IS the market's estimate of how
much you will lose. That gives three sharp, falsifiable tests:

  TEST 1  BREAKEVEN. Realized win rate vs the win rate the price requires.
          Winning often is not the objective; winning more often than the
          quoted odds demand is. A 90%-accurate strategy paid 1:9 is a
          coin flip.

  TEST 2  VRP EXISTENCE. Is realized loss below risk-neutral loss at MID
          prices? This separates "there is no premium" from "the premium
          exists but the spread eats it" — a distinction every previous
          study in this project conflated.

  TEST 3  BEATABILITY. Can a model using point-in-time features predict
          breaches better than the market's own implied probability?
          Measured by out-of-sample AUC, walk-forward, against the
          credit/width baseline. If a model cannot beat the quoted odds,
          no structuring or strike selection will rescue it.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIQUID = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA "
             "XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST "
             "WMT DIS CAT GE PFE CVX WFC MS GS".split())
FEATS = ["iv_front", "rv_ewma", "ivrv", "skew25", "iv_rank", "vov", "inv",
         "mom8", "rev3", "cw"]


def auc(y_true, score):
    """Rank-based AUC; no sklearn dependency."""
    y = np.asarray(y_true).astype(float)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    y, s = y[ok], s[ok]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(s).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load():
    st = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    st["date"] = pd.to_datetime(st.date)
    st = st[st.act_symbol.isin(LIQUID)].copy()
    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    f = f.sort_values(["act_symbol", "date"])
    f["ivrv"] = f.iv_front - f.rv_ewma
    f["inv"] = f.iv_front - f.iv_back
    f["iv_rank"] = f.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    f["vov"] = f.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(24, min_periods=8).std())
    f["mom8"] = f.groupby("act_symbol").spot.transform(lambda s: s / s.shift(24) - 1)
    f["rev3"] = f.groupby("act_symbol").spot.transform(lambda s: s / s.shift(3) - 1)
    d = st.merge(f.drop(columns=["spot"]), on=["date", "act_symbol"], how="left")
    return d


def main():
    d = load()
    print("=" * 78)
    print("STUDY 15 — is the market's breach forecast beatable?")
    print("=" * 78)

    for struct in ["credit_putspread", "credit_callspread"]:
        g = d[d.structure == struct].copy()
        if len(g) < 2000:
            continue
        g["cw"] = g.credit / g.margin            # risk-neutral expected loss ratio
        g["lossw"] = (g.credit - g.pnl) / g.margin   # realized loss ratio
        g["breach"] = (g.lossw > 1e-9).astype(int)
        g["win"] = 1 - g.breach
        g = g[(g.cw > 0.01) & (g.cw < 0.95)]

        print(f"\n{'='*78}\n{struct}  (n={len(g):,})")

        # ---- TEST 1: breakeven vs realized
        print("\nTEST 1 — accuracy vs the accuracy the price demands")
        print(f"  {'credit/width bucket':>22} {'breakeven win%':>15} "
              f"{'realized win%':>14} {'edge':>8} {'meanRet':>9} {'n':>7}")
        g["cwb"] = pd.qcut(g.cw.rank(method="first"), 5, labels=False)
        for b in range(5):
            x = g[g.cwb == b]
            be = 1 - x.cw.mean()          # required win rate
            rw = x.win.mean()             # achieved win rate
            mr = (x.pnl / x.margin).mean()
            print(f"  {'Q'+str(b)+' cw~'+format(x.cw.mean(),'.2f'):>22} "
                  f"{be:>15.1%} {rw:>14.1%} {rw-be:>+8.1%} {mr:>+9.4f} {len(x):>7,}")

        # ---- TEST 2: does the premium exist before costs?
        # mid-price credit = credit + half the spread we gave up on each leg.
        # We do not have per-leg spreads stored here, so bound it: worst-side
        # entry costs at most (ask-bid) per leg; use the observed relationship
        # between realized and risk-neutral loss instead.
        print("\nTEST 2 — variance risk premium: is realized loss below Q-implied?")
        rn, real = g.cw.mean(), g.lossw.mean()
        print(f"  risk-neutral expected loss ratio (credit/width) : {rn:.4f}")
        print(f"  realized      loss ratio                        : {real:.4f}")
        print(f"  premium (rn - realized)                         : {rn-real:+.4f}"
              f"   {'PREMIUM EXISTS' if rn > real else 'NO PREMIUM'}")
        print(f"  mean net return on margin (worst-side)          : "
              f"{(g.pnl/g.margin).mean():+.4f}")

        # ---- TEST 3: can a model beat the market's own probability?
        print("\nTEST 3 — out-of-sample AUC for predicting a breach")
        base = auc(g.breach, g.cw)      # market's own forecast
        print(f"  market baseline (credit/width)          AUC = {base:.4f}")
        for feat in FEATS:
            if feat == "cw" or feat not in g.columns:
                continue
            a = auc(g.breach, g[feat])
            a = max(a, 1 - a) if np.isfinite(a) else a   # allow either direction
            print(f"  single feature {feat:<24} AUC = {a:.4f}"
                  + ("   <-- beats market" if np.isfinite(a) and a > base else ""))

        # walk-forward logistic model on all features
        print("\n  walk-forward logistic model (fit on prior years only):")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("    sklearn unavailable — skipping model")
            continue
        aucs = []
        for y in (2021, 2022, 2023, 2024, 2025, 2026):
            tr = g[g.date < pd.Timestamp(f"{y}-01-01")].dropna(subset=FEATS)
            te = g[(g.date >= pd.Timestamp(f"{y}-01-01"))
                   & (g.date < pd.Timestamp(f"{y+1}-01-01"))].dropna(subset=FEATS)
            if len(tr) < 2000 or len(te) < 300 or te.breach.nunique() < 2:
                continue
            sc = StandardScaler().fit(tr[FEATS])
            m = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(tr[FEATS]),
                                                             tr.breach)
            p = m.predict_proba(sc.transform(te[FEATS]))[:, 1]
            a_model = auc(te.breach, p)
            a_mkt = auc(te.breach, te.cw)
            aucs.append((y, a_model, a_mkt, len(te)))
            print(f"    {y}: model AUC={a_model:.4f}  market AUC={a_mkt:.4f}  "
                  f"delta={a_model-a_mkt:+.4f}  n={len(te):,}")
        if aucs:
            dm = np.mean([a - b for _, a, b, _ in aucs])
            print(f"    MEAN model−market AUC = {dm:+.4f}  "
                  + ("MODEL WINS" if dm > 0.01 else
                     "no meaningful improvement over the quoted odds"))


if __name__ == "__main__":
    main()
