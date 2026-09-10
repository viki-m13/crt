#!/usr/bin/env python3
"""The short side: is predicting which stocks FALL easier than which rise?

There is a real reason to expect yes. Short-side anomalies are documented as
stronger than long-side ones, and the standard explanation is that shorting is
costly and constrained, so mispricing on that side is not arbitraged away. Our
panel is also unusually suited to it: 4,725 of 10,991 tickers stopped trading,
and predicting that a company fails is a genuinely different problem from
predicting that one prospers.

There is an equally real reason the answer may not matter. The explanation for
why the edge survives — that these names are expensive or impossible to borrow
— predicts that the edge lives exactly where it cannot be captured. So this
runs four tests, and the third is the one that decides it:

  1. SYMMETRY. Same channels, same machinery, aimed down instead of up. Is
     short precision above long precision at matched selectivity?
  2. DRIFT. A short pays the market's upward drift, so precision is not the
     metric — mean return is. A 55% hit rate that loses money is not an edge.
  3. BORROWABILITY. Split the signals by the liquidity and price profile that
     determines whether a name can be borrowed at all. If the edge is
     concentrated in sub-$5 microcaps, it is a measurement, not a strategy.
  4. DELISTING. Can we predict which names stop trading? That is the cleanest
     version of the short thesis and the one this dataset can actually answer.

Delisting is scored two ways throughout, because for a SHORT it is genuinely
ambiguous: a bankruptcy is the best possible outcome and an acquisition is
usually the worst, and a close-only tape cannot tell them apart.

    python research/failure_first/shorts.py [--horizon 63]
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
sys.path.insert(0, os.path.join(ROOT, "research", "adaptive_path"))

import channels as C                       # noqa: E402
from data import load_panel                # noqa: E402
from engine import effective_block_wilson  # noqa: E402
from validate import load_volume           # noqa: E402

COVERAGE = (0.05, 0.02, 0.01, 0.005, 0.002)


def short_gate(scores, q):
    """Every channel maximally DANGEROUS — the mirror of the long gate."""
    m = np.ones(next(iter(scores.values())).shape, dtype=bool)
    for c in C.CHANNELS:
        s = scores[c]
        m &= np.isfinite(s) & (s >= 1.0 - q)
    return m


def solve(scores, valid, target, gate_fn, lo=0.01, hi=1.0, iters=24):
    tot = max(int(valid.sum()), 1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        cov = int((gate_fn(scores, mid) & valid).sum()) / tot
        if cov < target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--tickers", type=int, default=4000)
    ap.add_argument("--borrowable", action="store_true",
                    help="restrict the universe to names a short could "
                         "actually be established in BEFORE ranking. Ranking "
                         "across everything makes the most dangerous names "
                         "microcaps by construction, so the signal is only "
                         "meaningful if the ranking happens inside the "
                         "tradeable set.")
    a = ap.parse_args()
    H = a.horizon

    px, _ = load_panel()
    rng = np.random.default_rng(20260909)
    if a.tickers < px.shape[1]:
        px = px[sorted(rng.choice(px.columns, a.tickers, replace=False))]
    vol = load_volume(px)

    if a.borrowable:
        dvf = (pd.DataFrame(vol.to_numpy(dtype=np.float32), index=px.index,
                            columns=px.columns) * px)
        advf = dvf.rolling(63, min_periods=40).median()
        keep = (px >= 5.0) & (advf >= 1e6)
        before = int(px.notna().sum().sum())
        px = px.where(keep)
        vol = vol.where(keep)
        alive = px.notna().sum() >= 400
        px, vol = px.loc[:, alive], vol.loc[:, alive]
        print(f"  borrowable filter: price >= $5 and median 63d dollar volume "
              f">= $1m")
        print(f"  {int(px.notna().sum().sum()):,} of {before:,} observations "
              f"survive, {px.shape[1]:,} tickers")

    print(f"panel {px.shape[0]:,} x {px.shape[1]:,}  horizon {H}\n")

    RAW = C.build(px, vol)
    S = C.orthogonalise(RAW)          # the form that worked on the long side

    arr = px.to_numpy(dtype=np.float32)
    n = arr.shape[0]
    ok = np.isfinite(arr)
    last = np.where(ok.any(0), n - 1 - ok[::-1].argmax(0), -1)
    fut = np.full_like(arr, np.nan)
    fut[:n - H] = arr[H:]
    rows = np.arange(n)[:, None]
    gone = ok & (rows + H > last[None, :]) & (last[None, :] >= 0)
    survived = ok & (rows + H <= last[None, :])
    valid = survived | gone
    rows_ix = np.arange(n)

    with np.errstate(invalid="ignore", divide="ignore"):
        ret = fut / arr - 1.0
    # A short's outcome when the name disappears is genuinely unknown:
    # bankruptcy is the best case, acquisition usually the worst.
    down_opt = np.where(gone, 1.0, (fut < arr).astype(np.float32))
    down_pess = np.where(gone, 0.0, (fut < arr).astype(np.float32))
    ret_excl = np.where(gone, np.nan, ret)

    base_down = float(down_pess[survived].mean())
    # MEDIAN, not mean. The first version of this printed a base mean return
    # of +6729% because ZXZZT — a NASDAQ order-routing test symbol, not a
    # security — prints $0.0001 to $6,000 on one bar. That is now excluded at
    # the data layer, but a right-skewed cross-section of equity returns still
    # has a mean nobody experiences, and the median is what a typical position
    # actually does.
    base_ret = float(np.nanmedian(ret_excl[survived]))
    print(f"base rate P(lower in {H}): {100*base_down:.2f}%   "
          f"mean return {100*base_ret:+.2f}%   "
          f"({100*gone[valid].mean():.2f}% delisted)\n")

    def score(mask, label):
        m = mask & valid
        ns = int(m.sum())
        if ns < 100:
            return None
        blocks = len(np.unique(rows_ix[np.nonzero(m)[0]] // (2 * H)))
        surv = mask & survived
        return dict(
            label=label, n=ns, blocks=blocks,
            p_opt=float(down_opt[m].mean()),
            p_pess=float(down_pess[m].mean()),
            lower=effective_block_wilson(float(down_pess[m].mean()), blocks),
            mret=float(np.nanmedian(ret_excl[surv])) if surv.sum() else np.nan,
            coverage=ns / max(int(valid.sum()), 1))

    print("=" * 78)
    print("1. SYMMETRY — SHORT SIDE vs LONG SIDE, MATCHED SELECTIVITY")
    print("=" * 78)
    print(f"  {'coverage':>9}{'P(down) opt':>13}{'P(down) pess':>14}"
          f"{'median ret':>12}{'P(up) long':>12}{'blocks':>8}")
    for cov in COVERAGE:
        qs = solve(S, valid, cov, short_gate)
        rs = score(short_gate(S, qs), "short")
        ql = solve(S, valid, cov, C.gate)
        gl = C.gate(S, ql) & valid
        up_l = float((~down_pess.astype(bool) & ~gone)[gl].mean()) if gl.sum() else np.nan
        if not rs:
            continue
        print(f"  {100*cov:>8.2f}%{100*rs['p_opt']:>12.1f}%"
              f"{100*rs['p_pess']:>13.1f}%{100*rs['mret']:>11.2f}%"
              f"{100*up_l:>11.1f}%{rs['blocks']:>8}")
    print("\n  'opt' counts a delisting as a winning short; 'pess' counts it")
    print("  as a loser. The truth is between and this tape cannot say where.")

    print()
    print("=" * 78)
    print("2. DOES IT MAKE MONEY? (a short pays the market's drift)")
    print("=" * 78)
    qs = solve(S, valid, 0.002, short_gate)
    rs = score(short_gate(S, qs), "tightest")
    print(f"  tightest short gate: {rs['n']:,} signals, {rs['blocks']} blocks")
    print(f"  P(lower), pessimistic delisting: {100*rs['p_pess']:.2f}%   "
          f"95% lower bound {100*rs['lower']:.2f}%")
    print(f"  median return of the NAME:  {100*rs['mret']:+.2f}%")
    print(f"  the short earns the negative of that: {-100*rs['mret']:+.2f}%")
    print(f"  same-period market mean:  {100*base_ret:+.2f}%")
    print(f"  edge over shorting the index: "
          f"{100*(base_ret - rs['mret']):+.2f}%")

    print()
    print("=" * 78)
    print("3. CAN THESE NAMES BE BORROWED? (the test that decides it)")
    print("=" * 78)
    dv = (pd.DataFrame(vol.to_numpy(dtype=np.float32), index=px.index,
                       columns=px.columns) * px)
    adv = dv.rolling(63, min_periods=40).median().to_numpy(dtype=np.float32)
    g = short_gate(S, qs) & valid
    print(f"  {'bucket':<26}{'signals':>10}{'share':>9}{'P(down)':>10}"
          f"{'median ret':>12}")
    bands = [(0, 1e5, "under $100k/day"), (1e5, 1e6, "$100k-$1m/day"),
             (1e6, 1e7, "$1m-$10m/day"), (1e7, np.inf, "over $10m/day")]
    for lo, hi, name in bands:
        m = g & (adv >= lo) & (adv < hi)
        if m.sum() < 50:
            print(f"  {name:<26}{int(m.sum()):>10,}   too few")
            continue
        surv = m & survived
        print(f"  {name:<26}{int(m.sum()):>10,}{100*m.sum()/g.sum():>8.1f}%"
              f"{100*down_pess[m].mean():>9.1f}%"
              f"{100*np.nanmedian(ret_excl[surv]):>11.2f}%")
    print()
    for lo, hi, name in ((0, 5, "under $5"), (5, 20, "$5-$20"),
                         (20, np.inf, "over $20")):
        m = g & (arr >= lo) & (arr < hi)
        if m.sum() < 50:
            print(f"  {('price ' + name):<26}{int(m.sum()):>10,}   too few")
            continue
        surv = m & survived
        print(f"  {('price ' + name):<26}{int(m.sum()):>10,}"
              f"{100*m.sum()/g.sum():>8.1f}%{100*down_pess[m].mean():>9.1f}%"
              f"{100*np.nanmedian(ret_excl[surv]):>10.2f}%")

    print()
    print("=" * 78)
    print("4. CAN WE PREDICT WHICH NAMES STOP TRADING?")
    print("=" * 78)
    print("  The cleanest short thesis, and the one this panel can answer.")
    hz = 252
    fut2 = np.full_like(arr, np.nan)
    fut2[:n - hz] = arr[hz:]
    gone2 = ok & (rows + hz > last[None, :]) & (last[None, :] >= 0)
    valid2 = ok & ((rows + hz <= last[None, :]) | gone2)
    b = float(gone2[valid2].mean())
    print(f"  base rate P(delists within {hz} sessions): {100*b:.2f}%")
    print(f"  {'gate coverage':>15}{'signals':>10}{'P(delists)':>13}{'lift':>9}")
    for cov in (0.05, 0.02, 0.01, 0.002):
        q2 = solve(S, valid2, cov, short_gate)
        m = short_gate(S, q2) & valid2
        if m.sum() < 100:
            continue
        p = float(gone2[m].mean())
        print(f"  {100*cov:>14.2f}%{int(m.sum()):>10,}{100*p:>12.2f}%"
              f"{p/max(b,1e-9):>8.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
