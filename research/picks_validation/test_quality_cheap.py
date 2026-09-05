#!/usr/bin/env python3
"""Does "high quality but currently cheap" actually beat the market?

This tests the daily-picks product idea. The existing study in
docs/analysis-quality-weight-backtest.md reports an 81% one-year hit rate
for high-quality pullback buys and adopted a production gate on that basis.
It has no baseline. Over 1993-2026 most stocks rose in most twelve-month
windows, so an 81% hit rate may be entirely the base rate of owning equities
— the same trap that made a credit-spread "edge" turn out to be levered beta
elsewhere in this project.

Everything here is built to make the signal FAIL if it is not real:

  SURVIVORSHIP. The universe is point-in-time S&P 500 membership (985
    tickers, including names later acquired or bankrupted), not today's
    survivors. The prior study used 34 tickers that all still exist.

  BASELINE. Every number is reported against three controls measured on the
    SAME months and the SAME horizon:
      - all PIT members that month (the true base rate)
      - random picks of the same size, repeated, from the same universe
      - the equal-weight universe return (the market itself)
    A hit rate without these is not evidence of anything.

  LOOK-AHEAD. Signals use only prices up to and including the formation
    date; membership is as of the formation date; returns start after it.

  OVERLAP. Monthly formation with twelve-month holds means overlapping
    windows. Significance is computed on non-overlapping annual cohorts, and
    the number of independent periods is printed next to every t-stat.

  REGIME. Results are split by decade and dev/holdout so a single bull run
    cannot carry the verdict.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "experiments", "monthly_dca", "cache")
PIT = os.path.join(CACHE, "v2", "sp500_pit")
HOLD_DAYS = 252
RNG = np.random.default_rng(7)


def load():
    px_path = os.path.join(PIT, "prices_extended_pit.parquet")
    if not os.path.exists(px_path):
        px_path = os.path.join(CACHE, "prices_extended.parquet")
    px = pd.read_parquet(px_path).sort_index()
    px.index = pd.to_datetime(px.index)
    mem = pd.read_parquet(os.path.join(PIT, "sp500_membership_monthly.parquet"))
    mem["asof"] = pd.to_datetime(mem["asof"])
    return px, mem


def signals(px: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Quality and cheapness from prices available AT asof only."""
    hist = px.loc[:asof]
    if len(hist) < 760:
        return pd.DataFrame()
    last = hist.iloc[-1]

    # cheapness: how far below the trailing 1-year high (the "washout")
    hi_1y = hist.iloc[-252:].max()
    drawdown = last / hi_1y - 1.0

    # quality, computed only from past prices:
    #   trend      - above its own 200d average over the past 3 years
    #   recovery   - historically regains prior highs after falling
    #   steadiness - low downside volatility
    sma200 = hist.rolling(200).mean()
    above = (hist.iloc[-756:] > sma200.iloc[-756:]).mean()

    r = hist.pct_change()
    downside = r.iloc[-756:].clip(upper=0).std() * np.sqrt(252)
    long_run = hist.iloc[-1] / hist.iloc[-1260] - 1.0 if len(hist) >= 1260 else np.nan

    q = (above.rank(pct=True) * 0.45
         + (1 - downside.rank(pct=True)) * 0.30
         + pd.Series(long_run, index=hist.columns).rank(pct=True) * 0.25) * 100

    out = pd.DataFrame({"quality": q, "drawdown": drawdown, "px": last})
    return out.dropna(subset=["quality", "drawdown", "px"])


def forward(px: pd.DataFrame, asof: pd.Timestamp, tickers, days=HOLD_DAYS):
    """Return over the next `days` sessions. Delisting mid-window is handled
    by using the last available price — which is how a real holder would
    experience an acquisition or a bankruptcy, rather than dropping the name."""
    fut = px.loc[px.index > asof]
    if fut.empty:
        return pd.Series(dtype=float)
    window = fut.iloc[:days]
    if window.empty:
        return pd.Series(dtype=float)
    start = px.loc[:asof].iloc[-1]
    end = window.ffill().iloc[-1]
    out = (end[tickers] / start[tickers] - 1.0)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def main():
    px, mem = load()
    print(f"price panel: {px.shape[1]:,} tickers, {px.index[0].date()} -> "
          f"{px.index[-1].date()}")
    print(f"PIT membership rows: {len(mem):,}, unique tickers: "
          f"{mem['ticker'].nunique():,}\n")

    months = sorted(mem["asof"].unique())
    rows = []
    for asof in months:
        asof = pd.Timestamp(asof)
        if asof < px.index[0] or asof > px.index[-1] - pd.Timedelta(days=400):
            continue
        universe = [t for t in mem.loc[mem["asof"] == asof, "ticker"].unique()
                    if t in px.columns]
        if len(universe) < 50:
            continue
        sig = signals(px[universe], asof)
        if sig.empty or len(sig) < 50:
            continue
        fwd = forward(px, asof, sig.index)
        sig = sig.join(fwd.rename("fwd"), how="inner").dropna(subset=["fwd"])
        if len(sig) < 50:
            continue

        # THE PICK: high quality AND currently cheap
        hi_q = sig.quality >= sig.quality.quantile(0.70)
        cheap = sig.drawdown <= sig.drawdown.quantile(0.30)
        picks = sig[hi_q & cheap]
        if len(picks) < 3:
            continue

        n = len(picks)
        # matched random control: same number of names, same month, same universe
        rand = [sig.fwd.sample(n, random_state=int(RNG.integers(1e9))).mean()
                for _ in range(30)]

        rows.append({
            "asof": asof, "n_universe": len(sig), "n_picks": n,
            "pick_ret": picks.fwd.mean(),
            "pick_hit": (picks.fwd > 0).mean(),
            "base_ret": sig.fwd.mean(),          # equal-weight universe
            "base_hit": (sig.fwd > 0).mean(),    # the TRUE base rate
            "rand_ret": float(np.mean(rand)),
            "cheap_only": sig[cheap].fwd.mean(),
            "quality_only": sig[hi_q].fwd.mean(),
        })

    R = pd.DataFrame(rows)
    if R.empty:
        print("no evaluable months")
        return 1
    R["excess"] = R.pick_ret - R.base_ret
    R["hit_excess"] = R.pick_hit - R.base_hit

    print("=" * 92)
    print("1. THE HEADLINE NUMBER, AND WHAT IT IS WORTH")
    print("=" * 92)
    print(f"  months evaluated: {len(R):,}  "
          f"({R.asof.min().date()} -> {R.asof.max().date()})")
    print(f"  mean picks per month: {R.n_picks.mean():.0f} "
          f"from a universe of {R.n_universe.mean():.0f}\n")
    print(f"  PICKS   1y hit rate {R.pick_hit.mean():6.1%}   mean return {R.pick_ret.mean():+7.2%}")
    print(f"  BASE    1y hit rate {R.base_hit.mean():6.1%}   mean return {R.base_ret.mean():+7.2%}"
          "   <- every S&P member that month")
    print(f"  RANDOM  (same count)                     mean return {R.rand_ret.mean():+7.2%}")
    print(f"\n  EXCESS over the base rate: {R.hit_excess.mean():+.1%} hit, "
          f"{R.excess.mean():+.2%} return")

    print("\n" + "=" * 92)
    print("2. SIGNIFICANCE, WITH THE OVERLAP CORRECTION")
    print("=" * 92)
    naive_t = R.excess.mean() / (R.excess.std() / np.sqrt(len(R)))
    ann = R.set_index("asof").excess.groupby(lambda d: d.year).mean()
    ann_t = ann.mean() / (ann.std() / np.sqrt(len(ann))) if len(ann) > 2 else np.nan
    print(f"  naive t on {len(R)} overlapping monthly windows: {naive_t:+.2f}  "
          "<- OVERSTATED, do not quote")
    print(f"  t on {len(ann)} non-overlapping annual cohorts:  {ann_t:+.2f}  "
          "<- the honest one")
    print(f"  years with positive excess: {100*(ann>0).mean():.0f}% of {len(ann)}")

    print("\n" + "=" * 92)
    print("3. IS IT QUALITY, CHEAPNESS, OR NEITHER?")
    print("=" * 92)
    print(f"  quality alone (top 30%):  {R.quality_only.mean()-R.base_ret.mean():+.2%} vs base")
    print(f"  cheap alone (worst 30%):  {R.cheap_only.mean()-R.base_ret.mean():+.2%} vs base")
    print(f"  both together:            {R.excess.mean():+.2%} vs base")

    print("\n" + "=" * 92)
    print("4. DOES IT SURVIVE OUT OF SAMPLE AND ACROSS REGIMES?")
    print("=" * 92)
    dev = R[R.asof < "2017-01-01"]
    hold = R[R.asof >= "2017-01-01"]
    for lab, part in (("dev   (pre-2017)", dev), ("holdout (2017+)", hold)):
        if len(part) > 5:
            print(f"  {lab:<18} excess {part.excess.mean():+.2%}  "
                  f"hit excess {part.hit_excess.mean():+.1%}  n={len(part)}")
    print()
    for decade, part in R.groupby(R.asof.dt.year // 10 * 10):
        if len(part) > 5:
            print(f"  {decade}s  excess {part.excess.mean():+.2%}  "
                  f"picks {part.pick_ret.mean():+.2%} vs base {part.base_ret.mean():+.2%}  "
                  f"n={len(part)}")

    print("\n" + "=" * 92)
    print("VERDICT TEST (stated before the numbers were seen)")
    print("=" * 92)
    ok_excess = R.excess.mean() > 0.01
    ok_t = np.isfinite(ann_t) and ann_t > 2.0
    ok_hold = len(hold) > 5 and hold.excess.mean() > 0
    print(f"  beats the base rate by >1%/yr: {'YES' if ok_excess else 'NO'} "
          f"({R.excess.mean():+.2%})")
    print(f"  t > 2 on non-overlapping years: {'YES' if ok_t else 'NO'} ({ann_t:+.2f})")
    print(f"  holdout excess positive:        {'YES' if ok_hold else 'NO'}")
    print(f"\n  => {'SELLABLE as a real edge' if (ok_excess and ok_t and ok_hold) else 'NOT PROVEN — do not sell this as an edge'}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    R.to_csv(out, index=False)
    print(f"\n  per-month results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
