#!/usr/bin/env python3
"""What, if anything, actually predicts cross-sectional stock returns here?

The quality-and-cheap combination came in at +2.85%/yr excess with t=+1.91 on
non-overlapping years and turned negative in the 2020s. Rather than guess at
another combination and re-run the same near-miss, this measures a whole
panel of candidate signals the same way at once, so the CEILING is known
before anything is built on top of it — the approach that settled the options
work, where finding the ceiling first prevented a lot of wasted search.

Scored for every signal:
  * cross-sectional Spearman IC against forward returns at 1, 3, 6 and 12
    months (rank correlation, so outliers and the skew of equity returns
    cannot drive it — the corrupted TIE price that broke the earlier run
    would move a mean but barely moves a rank)
  * t-statistic on NON-OVERLAPPING periods, never on overlapping months
  * dev (pre-2017) versus holdout (2017+)
  * decade-by-decade, because a signal that died in the 2020s is not
    sellable today no matter how good its full-sample average looks

Universe is point-in-time S&P 500 membership, so delisted and acquired names
are present. No signal uses information from after its formation date.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "experiments", "monthly_dca", "cache")
PIT = os.path.join(CACHE, "v2", "sp500_pit")
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def load():
    p = os.path.join(PIT, "prices_extended_pit.parquet")
    if not os.path.exists(p):
        p = os.path.join(CACHE, "prices_extended.parquet")
    px = pd.read_parquet(p).sort_index()
    px.index = pd.to_datetime(px.index)
    mem = pd.read_parquet(os.path.join(PIT, "sp500_membership_monthly.parquet"))
    mem["asof"] = pd.to_datetime(mem["asof"])
    return px, mem


def build_signals(hist: pd.DataFrame) -> pd.DataFrame:
    """Every signal from prices up to and including the last row of `hist`."""
    last = hist.iloc[-1]
    r = hist.pct_change()
    out = {}

    # momentum family
    for lab, back in (("mom_1m", 21), ("mom_3m", 63), ("mom_6m", 126),
                      ("mom_12m", 252)):
        if len(hist) > back:
            out[lab] = last / hist.iloc[-1 - back] - 1.0
    # classic 12-1 momentum: skip the most recent month
    if len(hist) > 252:
        out["mom_12_1"] = hist.iloc[-22] / hist.iloc[-253] - 1.0

    # cheapness / mean reversion
    if len(hist) > 252:
        hi = hist.iloc[-252:].max()
        lo = hist.iloc[-252:].min()
        out["dd_from_high"] = last / hi - 1.0
        out["pos_52w"] = (last - lo) / (hi - lo).replace(0, np.nan)
    if len(hist) > 1260:
        out["long_run_5y"] = last / hist.iloc[-1260] - 1.0

    # risk / stability
    if len(hist) > 252:
        out["vol_1y"] = r.iloc[-252:].std() * np.sqrt(252)
        out["downside_vol"] = r.iloc[-252:].clip(upper=0).std() * np.sqrt(252)
        out["max_dd_1y"] = (hist.iloc[-252:] / hist.iloc[-252:].cummax() - 1).min()
    if len(hist) > 756:
        sma200 = hist.rolling(200).mean()
        out["pct_above_sma"] = (hist.iloc[-756:] > sma200.iloc[-756:]).mean()
        out["vol_of_vol"] = (r.rolling(21).std().iloc[-756:].std()
                             / r.rolling(21).std().iloc[-756:].mean())

    # trend quality: how straight the line up has been
    if len(hist) > 252:
        w = hist.iloc[-252:]
        x = np.arange(len(w))
        lg = np.log(w.replace(0, np.nan))
        slope = lg.apply(lambda c: np.polyfit(x[c.notna()], c.dropna(), 1)[0]
                         if c.notna().sum() > 100 else np.nan)
        resid = lg.apply(lambda c: (np.std(c.dropna() - np.poly1d(
            np.polyfit(x[c.notna()], c.dropna(), 1))(x[c.notna()]))
            if c.notna().sum() > 100 else np.nan))
        out["trend_slope"] = slope
        out["trend_smoothness"] = -(resid)

    df = pd.DataFrame(out)
    # composites, built from ranks so scales cannot dominate
    def rk(c):
        return df[c].rank(pct=True) if c in df else pd.Series(np.nan, index=df.index)

    df["quality"] = (rk("pct_above_sma") * 0.45
                     + (1 - rk("downside_vol")) * 0.30
                     + rk("long_run_5y") * 0.25)
    df["quality_x_cheap"] = df["quality"] * (1 - rk("dd_from_high"))
    df["quality_x_momentum"] = df["quality"] * rk("mom_12_1")
    df["cheap_x_trend"] = (1 - rk("dd_from_high")) * rk("trend_smoothness")
    return df


def main():
    px, mem = load()
    print(f"panel {px.shape[1]:,} tickers | PIT tickers {mem['ticker'].nunique():,}\n")
    months = sorted(mem["asof"].unique())

    records = []
    for asof in months:
        asof = pd.Timestamp(asof)
        if asof < px.index[0] or asof > px.index[-1] - pd.Timedelta(days=30):
            continue
        uni = [t for t in mem.loc[mem["asof"] == asof, "ticker"].unique()
               if t in px.columns]
        if len(uni) < 80:
            continue
        hist = px[uni].loc[:asof]
        if len(hist) < 1300:
            continue
        sig = build_signals(hist)
        if sig.empty:
            continue

        fut = px[uni].loc[px.index > asof]
        if fut.empty:
            continue
        start = hist.iloc[-1]
        for hlab, hdays in HORIZONS.items():
            w = fut.iloc[:hdays]
            if len(w) < hdays * 0.8:
                continue
            fwd = (w.ffill().iloc[-1] / start - 1.0)
            fwd = fwd.replace([np.inf, -np.inf], np.nan)
            fwd = fwd[(fwd > -0.995) & (fwd < 9.0)]      # data-error guard
            if len(fwd) < 60:
                continue
            for col in sig.columns:
                s = sig[col].reindex(fwd.index).dropna()
                if len(s) < 60:
                    continue
                y = fwd.reindex(s.index)
                ic = spearmanr(s, y).statistic
                if np.isfinite(ic):
                    records.append({"month": asof, "h": hlab,
                                    "signal": col, "ic": ic})

    D = pd.DataFrame(records)
    if D.empty:
        print("no results"); return 1

    print("=" * 100)
    print("CROSS-SECTIONAL IC BY SIGNAL AND HORIZON")
    print("  IC = rank correlation with forward return. t is on NON-OVERLAPPING")
    print("  periods (a 12m signal gets one observation per year, not twelve).")
    print("=" * 100)
    print(f"  {'signal':<20} {'horizon':<8} {'IC':>8} {'t(indep)':>9} "
          f"{'dev':>8} {'hold':>8} {'2020s':>8}")
    rows = []
    for (sigl, h), g in D.groupby(["signal", "h"]):
        g = g.sort_values("month")
        per_year = max(1, {"1m": 12, "3m": 4, "6m": 2, "12m": 1}[h])
        blocks = g.set_index("month").ic.groupby(
            lambda d: (d.year, (d.month - 1) // (12 // per_year))).mean()
        t = blocks.mean() / (blocks.std() / np.sqrt(len(blocks))) \
            if len(blocks) > 3 and blocks.std() > 0 else np.nan
        dev = g[g.month < "2017-01-01"].ic.mean()
        hold = g[g.month >= "2017-01-01"].ic.mean()
        d20 = g[g.month >= "2020-01-01"].ic.mean()
        rows.append({"signal": sigl, "h": h, "ic": g.ic.mean(), "t": t,
                     "dev": dev, "hold": hold, "d20": d20, "n": len(blocks)})
    T = pd.DataFrame(rows)
    T["absic"] = T.ic.abs()
    for _, r in T.sort_values("absic", ascending=False).head(28).iterrows():
        print(f"  {r.signal:<20} {r.h:<8} {r.ic:>+8.4f} {r.t:>+9.2f} "
              f"{r.dev:>+8.4f} {r.hold:>+8.4f} {r.d20:>+8.4f}")

    print("\n" + "=" * 100)
    print("SURVIVORS — |t| > 2 on independent periods AND same sign in holdout")
    print("=" * 100)
    surv = T[(T.t.abs() > 2) & (np.sign(T.dev) == np.sign(T.hold))
             & (T.ic.abs() > 0.01)]
    if surv.empty:
        print("  NONE. No signal here clears the bar on independent periods")
        print("  with a consistent sign out of sample.")
    else:
        for _, r in surv.sort_values("absic", ascending=False).iterrows():
            still = "yes" if np.sign(r.d20) == np.sign(r.ic) else "NO — dead in the 2020s"
            print(f"  {r.signal:<20} {r.h:<6} IC={r.ic:+.4f} t={r.t:+.2f} "
                  f"holdout={r.hold:+.4f}  still working: {still}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_survey.csv")
    T.to_csv(out, index=False)
    print(f"\nfull table -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
