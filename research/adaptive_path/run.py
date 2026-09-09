#!/usr/bin/env python3
"""Run the adaptive-path engine on the point-in-time panel and report honestly.

    python research/adaptive_path/run.py [--horizon 30] [--step 21]

The question is whether the 90% precision gate ever opens, and if it does,
what it delivers on the forecasts it actually emitted. Everything printed is
out of sample by construction: the gate is chosen from matured forecasts only,
and the outcome of a forecast is invisible until its horizon has passed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "research", "uptrend"))

import engine as E              # noqa: E402
from data import load_panel     # noqa: E402


def block_bootstrap(sel: pd.DataFrame, h: int, seed: int = 7, iters: int = 2000):
    """Precision CI resampling whole horizon blocks, not individual rows."""
    if sel.empty:
        return float("nan"), float("nan")
    g = sel.assign(_b=sel["i"] // (2 * h)).groupby("_b")["up"].agg(["sum", "count"])
    if len(g) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    out = np.empty(iters)
    for t in range(iters):
        k = rng.integers(0, len(g), len(g))
        out[t] = s[k].sum() / max(c[k].sum(), 1)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def report(R: pd.DataFrame, h: int, cfg: E.Config):
    R = R.copy()
    R["year"] = pd.to_datetime(R["date"]).dt.year
    print("=" * 78)
    print(f"RESULT — HORIZON {h} SESSIONS")
    print("=" * 78)
    print(f"  forecasts issued        {len(R):,}")
    print(f"  distinct dates          {R['i'].nunique():,}")
    print(f"  base rate P(higher)     {100*R['up'].mean():.2f}%")
    print(f"  delisted mid-horizon    {int(R['delisted'].sum()):,} "
          f"({100*R['delisted'].mean():.2f}%, all scored as NOT higher)")

    print()
    print("  PRECISION BY CALIBRATED PROBABILITY BUCKET")
    print(f"  {'p_cal':<14}{'n':>10}{'precision':>12}{'lift vs base':>14}")
    base = R["up"].mean()
    for lo, hi in ((0.0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        b = R[(R.p_cal >= lo) & (R.p_cal < hi)]
        if len(b) < 50:
            print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{len(b):>10}   too few")
            continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{len(b):>10,}"
              f"{100*b.up.mean():>11.2f}%{100*(b.up.mean()-base):>+13.2f}")

    print()
    print("  IS IT CALIBRATED? (does p_cal mean what it says)")
    q = pd.qcut(R.p_cal, 10, duplicates="drop")
    cal = R.groupby(q, observed=True).agg(pred=("p_cal", "mean"),
                                          actual=("up", "mean"),
                                          n=("up", "size"))
    err = (cal.pred - cal.actual).abs().mean()
    print(f"  mean |predicted - actual| across deciles: {err:.4f}")
    print(f"  worst decile gap: {(cal.pred - cal.actual).abs().max():.4f}")

    print()
    print("  THE GATE")
    em = R[R["emitted"]]
    open_dates = R.loc[np.isfinite(R.gate), "i"].nunique()
    print(f"  dates where a gate opened at all: {open_dates:,} "
          f"of {R['i'].nunique():,}")
    if em.empty:
        print("  forecasts emitted: 0")
        print("\n  The gate never opened. On this universe the engine could")
        print("  not find a threshold whose matured precision lower bound")
        print(f"  cleared {cfg.precision_target:.0%}, so it stayed silent —")
        print("  which is the designed behaviour, not a failure.")
        return
    lo, hi = block_bootstrap(em, h)
    print(f"  forecasts emitted:      {len(em):,} "
          f"({100*len(em)/len(R):.2f}% of all)")
    print(f"  realised precision:     {100*em['up'].mean():.2f}%")
    print(f"  block bootstrap 95% CI: [{100*lo:.2f}%, {100*hi:.2f}%]")
    print(f"  base rate same rows:    {100*base:.2f}%")

    print()
    print("  PRECISION OF EMITTED FORECASTS, BY YEAR")
    print(f"  {'year':<8}{'n':>8}{'precision':>12}{'base rate':>12}")
    for y, g in em.groupby("year"):
        allg = R[R.year == y]
        flag = "  << below target" if g.up.mean() < cfg.precision_target else ""
        print(f"  {y:<8}{len(g):>8,}{100*g.up.mean():>11.1f}%"
              f"{100*allg.up.mean():>11.1f}%{flag}")
    worst = em.groupby("year").up.mean().min()
    print(f"\n  worst year: {100*worst:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--names", type=int, default=300)
    ap.add_argument("--eval-year", type=int, default=2005)
    ap.add_argument("--tickers", type=int, default=4000,
                    help="random ticker subsample. Random is unbiased — it "
                         "keeps dead names in proportion — unlike selecting "
                         "by liquidity or length of history, which would "
                         "quietly rebuild the survivorship problem.")
    a = ap.parse_args()

    px, _ = load_panel()
    if a.tickers and a.tickers < px.shape[1]:
        rng = np.random.default_rng(20260909)
        keep = sorted(rng.choice(px.columns, a.tickers, replace=False))
        px = px[keep]
    print(f"panel {px.shape[0]:,} bars x {px.shape[1]:,} tickers  "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")
    cfg = E.Config(horizons=(a.horizon,), step=a.step,
                   names_per_date=a.names, evaluation_year=a.eval_year)
    R = E.run_prequential(px, a.horizon, cfg)
    if R.empty:
        print("no forecasts produced")
        return 1
    out = os.path.join(HERE, f"forecasts_h{a.horizon}.parquet")
    R.drop(columns=[c for c in R.columns if c in E.FEATURES]).to_parquet(out)
    print()
    report(R, a.horizon, cfg)
    print(f"\n  detail -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
