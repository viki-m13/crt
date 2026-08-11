#!/usr/bin/env python3
"""Study 26: the literature's actual recipe — NONLINEAR ML on option returns.

Motivated by Bali, Beckmeyer, Moerke & Weigert, "Option Return Predictability
with Machine Learning and Big Data", Review of Financial Studies 36(9) 2023:
12M+ observations, 1996-2020, and the headline claim that *allowing for
nonlinearities* significantly increases out-of-sample predictive performance
for option returns, with long-short equity-option portfolios profitable after
transaction costs. Option-based characteristics dominate, but stock-based
characteristics add incremental power.

My study 8 fitted a LINEAR cross-sectional model and found nothing tradable.
Linearity is precisely what that paper says is the binding limitation, so this
is the missing test rather than a repetition.

Two disciplines carried over from earlier work here, both of which the naive
version of this test would fail:

  1. NO PAYOFF-COMPONENT FEATURES. Study 15 showed credit_yield "predicts"
     returns with corr +0.53 purely because ret = credit_yield - loss/margin
     and 76% of trades have zero loss. Any feature that is mechanically part
     of the payoff is excluded, so the model has to forecast the uncertain
     term rather than rediscover an identity.
  2. THE MARKET IS THE BENCHMARK. The option's own price already encodes a
     forecast. A model is only interesting if it beats that, so the market's
     implied measure is scored alongside on identical data.

Walk-forward: fit on all years strictly before the test year, predict forward,
never refit on test data. Costs are the same worst-side fills used throughout.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
LIQUID = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA "
             "XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST "
             "WMT DIS CAT GE PFE CVX WFC MS GS".split())

# Characteristics. Deliberately EXCLUDES credit/credit_yield and anything else
# that is a component of the realised payoff (see docstring point 1).
FEATURES = ["iv_front", "rv_ewma", "ivrv", "ivrv_ratio", "skew25", "skew_chg",
            "term", "iv_rank", "vov", "mom8", "mom1m", "rev3", "dist_52w",
            "iv_chg", "rv_chg", "spy_iv", "spy_inv", "idio_inv", "dte"]


def build():
    st = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    st["date"] = pd.to_datetime(st.date)
    st = st[(st.structure == "short_straddle_dh") & st.act_symbol.isin(LIQUID)].copy()
    st["ret"] = st.pnl / st.margin
    # the uncertain term only: loss given the credit is known at entry
    st["neg_loss"] = -(st.credit - st.pnl) / st.margin

    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    f = f.sort_values(["act_symbol", "date"])
    g = f.groupby("act_symbol")
    f["ivrv"] = f.iv_front - f.rv_ewma
    f["ivrv_ratio"] = f.iv_front / f.rv_ewma.clip(lower=0.03)
    f["term"] = f.iv_front - f.iv_back
    f["iv_rank"] = g.iv_front.transform(lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    f["vov"] = g.iv_front.transform(lambda s: s.rolling(24, min_periods=8).std())
    f["mom8"] = g.spot.transform(lambda s: s / s.shift(24) - 1)
    f["mom1m"] = g.spot.transform(lambda s: s / s.shift(9) - 1)
    f["rev3"] = g.spot.transform(lambda s: s / s.shift(3) - 1)
    f["dist_52w"] = g.spot.transform(lambda s: s / s.rolling(250, min_periods=60).max() - 1)
    f["iv_chg"] = g.iv_front.transform(lambda s: s - s.shift(3))
    f["rv_chg"] = g.rv_ewma.transform(lambda s: s - s.shift(3))
    f["skew_chg"] = g.skew25.transform(lambda s: s - s.shift(3))
    spy = f[f.act_symbol == "SPY"][["date", "iv_front", "term"]].rename(
        columns={"iv_front": "spy_iv", "term": "spy_inv"})
    f = f.merge(spy, on="date", how="left")
    f["idio_inv"] = f.term - f.spy_inv

    d = st.merge(f.drop(columns=["spot"]), on=["date", "act_symbol"], how="left")
    # the market's own forecast, for the benchmark comparison
    d["mkt_implied"] = d.credit / d.margin
    # REAL long-side returns, priced at the ask. Using the negated short return
    # as a proxy for the long leg is the error that made an earlier sleeve here
    # look like +2.32 Sharpe before collapsing to -1.64 once priced honestly.
    lg = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    lg["date"] = pd.to_datetime(lg.date)
    lg = lg[(lg.structure == "long_straddle_dh") & lg.act_symbol.isin(LIQUID)].copy()
    lg["ret_long"] = lg.pnl / lg.margin
    d = d.merge(lg[["date", "act_symbol", "ret_long"]], on=["date", "act_symbol"], how="left")
    return d


def xs_rank(df, cols):
    """Cross-sectional rank-normalise per date: scale-free, outlier-robust,
    and what the option-ML literature uses."""
    out = df.copy()
    for c in cols:
        out[c] = out.groupby("date")[c].transform(
            lambda s: (s.rank(pct=True) - 0.5) * 2 if s.notna().sum() >= 8 else np.nan)
    return out


def main():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
    except ImportError:
        print("sklearn required"); return
    d = build()
    d = d.dropna(subset=["ret"])
    feats = [c for c in FEATURES if c in d.columns]
    d = xs_rank(d, feats)
    d = d.dropna(subset=feats + ["ret"])
    print("=" * 84)
    print("STUDY 26 — nonlinear ML on option returns (Bali et al. RFS 2023 recipe)")
    print("=" * 84)
    print(f"rows={len(d):,}  dates={d.date.nunique():,}  names={d.act_symbol.nunique()}")
    print(f"features={len(feats)}  target=delta-hedged short straddle return\n")

    years = sorted(y for y in d.date.dt.year.unique() if y >= 2021)
    preds = []
    for y in years:
        tr = d[d.date < pd.Timestamp(f"{y}-01-01")]
        te = d[(d.date >= pd.Timestamp(f"{y}-01-01")) & (d.date < pd.Timestamp(f"{y+1}-01-01"))]
        if len(tr) < 3000 or len(te) < 200:
            continue
        Xtr, ytr = tr[feats].values, tr.ret.values
        gbm = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=80, l2_regularization=1.0, random_state=7)
        gbm.fit(Xtr, ytr)
        rdg = Ridge(alpha=10.0).fit(Xtr, ytr)
        te = te.copy()
        te["p_gbm"] = gbm.predict(te[feats].values)
        te["p_rdg"] = rdg.predict(te[feats].values)
        preds.append(te)
        # IC per model on this test year
        def ic(col):
            v = []
            for _, gg in te.groupby("date"):
                if len(gg) >= 10:
                    s = spearmanr(gg[col], gg.ret).statistic
                    if np.isfinite(s):
                        v.append(s)
            return np.mean(v) if v else np.nan
        print(f"  {y}: OOS IC  nonlinear={ic('p_gbm'):+.4f}  linear={ic('p_rdg'):+.4f}  "
              f"market={ic('mkt_implied'):+.4f}   n={len(te):,}")
    if not preds:
        print("insufficient data"); return
    P = pd.concat(preds)

    print("\n--- pooled OOS information coefficient ---")
    for col, lab in (("p_gbm", "nonlinear (GBM)"), ("p_rdg", "linear (ridge)"),
                     ("mkt_implied", "market implied")):
        v = []
        for _, gg in P.groupby("date"):
            if len(gg) >= 10:
                s = spearmanr(gg[col], gg.ret).statistic
                if np.isfinite(s):
                    v.append(s)
        v = np.array(v)
        t = v.mean() / (v.std() / math.sqrt(len(v))) if len(v) > 5 else np.nan
        print(f"  {lab:<20} IC={v.mean():+.4f}  t={t:+.2f}  n={len(v)}")

    print("\n--- tradable long-short decile portfolio, worst-side fills ---")
    print("  (short the richest decile, long the cheapest; equal weight per date)")
    for col, lab in (("p_gbm", "nonlinear (GBM)"), ("p_rdg", "linear (ridge)"),
                     ("mkt_implied", "market implied")):
        rows = []
        for dt, gg in P.groupby("date"):
            if len(gg) < 20:
                continue
            hi = gg[col].quantile(0.9)
            lo = gg[col].quantile(0.1)
            short_leg = gg[gg[col] >= hi].ret.mean()          # real short at bid
            cheap = gg[gg[col] <= lo].dropna(subset=["ret_long"])
            if not len(cheap):
                continue
            long_leg = cheap.ret_long.mean()                   # real long at ask
            rows.append((dt, 0.5 * (short_leg + long_leg), short_leg, long_leg))
        if len(rows) < 30:
            continue
        R = pd.DataFrame(rows, columns=["date", "ls", "sl", "ll"]).set_index("date").sort_index()
        f_ = lambda x: (x.mean()/x.std()*math.sqrt(12)) if x.std() > 0 else float('nan')
        print(f"  {lab:<20} LS mean={R.ls.mean():+.4f} Sh={f_(R.ls.resample('W-FRI').mean().dropna()):+.2f} | "
              f"short-only mean={R.sl.mean():+.4f} Sh={f_(R.sl.resample('W-FRI').mean().dropna()):+.2f} | "
              f"long-leg mean={R.ll.mean():+.4f}  n={len(R)}")

    print("\n  Interpretation: the paper's claim is that nonlinearity adds")
    print("  out-of-sample power. If GBM does not beat ridge AND the market's")
    print("  own implied measure here, the result does not transfer to this")
    print("  universe at these costs.")


if __name__ == "__main__":
    main()
