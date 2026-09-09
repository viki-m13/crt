#!/usr/bin/env python3
"""Does gating on ALL failure channels beat the best single one?

The idea under test: model every way the call fails, buy only when all are
quiet. It has an obvious appeal and one obvious way to be an illusion —
intersecting seven filters produces very few signals, and ANY selective rule
looks precise when it fires ten times a year in a bull market. So the test is
built around three controls, and the middle one is the one that decides it.

  BASE RATE at the same horizon on the same universe.

  BEST SINGLE CHANNEL AT EQUAL COVERAGE. This is the real test. If taking the
  safest 2% by `trend` alone matches the conjunction's precision at 2%
  coverage, then six of the seven channels are decoration and the idea is
  just "buy strong trends, selectively".

  RANDOM SELECTION AT EQUAL COVERAGE, to price in the fact that a small
  sample drawn from good years looks good.

Plus the two things that killed every previous version of this question:
precision BY CALENDAR YEAR (an average built from 1995 and 2008 is not a
promise to anyone), and a Wilson lower bound counted in non-overlapping
horizon blocks rather than in rows.

    python research/failure_first/validate.py [--horizon 63]
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

import channels as C                     # noqa: E402
from data import load_panel              # noqa: E402
from engine import effective_block_wilson  # noqa: E402

# Fixing the per-channel quantile and reading off coverage was the wrong
# parameterisation: at q=0.50 the conjunction already covers only 0.11% (not
# the 0.8% independence would give), and by q=0.10 it is empty. Target the
# COVERAGE instead and solve for the quantile, so the conjunction and its
# controls are always compared on the same number of signals.
TARGET_COVERAGE = (0.05, 0.02, 0.01, 0.005, 0.002)


def solve_quantile(S, valid, target, channels, lo=0.01, hi=1.0, iters=24):
    """Per-channel quantile whose conjunction covers `target` of observations."""
    import channels as _C
    tot = max(int(valid.sum()), 1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = np.ones(valid.shape, dtype=bool)
        for c in channels:
            m &= np.isfinite(S[c]) & (S[c] <= mid)
        cov = int((m & valid).sum()) / tot
        if cov < target:
            lo = mid
        else:
            hi = mid
    return hi


def load_volume(px: pd.DataFrame) -> pd.DataFrame | None:
    import glob
    frames = []
    for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/"
                              "prices/vol_*.parquet")):
        d = pd.read_parquet(f)
        cols = [c for c in d.columns if c in set(px.columns)]
        if cols:
            frames.append(d[cols])
    if not frames:
        return None
    v = pd.concat(frames, axis=1, sort=False)
    v = v.loc[:, ~v.columns.duplicated()]
    v.index = pd.to_datetime(v.index)
    return v.reindex(index=px.index, columns=px.columns)


def outcomes(px: pd.DataFrame, H: int):
    """up[i,c] = 1 if higher H bars later. A name that stops trading is 0."""
    arr = px.to_numpy(dtype=np.float32)
    n = arr.shape[0]
    ok = np.isfinite(arr)
    last = np.where(ok.any(0), n - 1 - ok[::-1].argmax(0), -1)
    fut = np.full_like(arr, np.nan)
    fut[:n - H] = arr[H:]
    rows = np.arange(n)[:, None]
    gone = ok & (rows + H > last[None, :]) & (last[None, :] >= 0)
    up = np.where(gone, 0.0, (fut > arr).astype(np.float32))
    valid = ok & ((rows + H <= last[None, :]) | gone)
    return up, valid, gone


def precision(mask, up, valid, rows_ix, H, label):
    m = mask & valid
    n = int(m.sum())
    if n < 100:
        return dict(label=label, n=n, precision=np.nan, lower=np.nan,
                    blocks=0, coverage=0.0)
    p = float(up[m].sum() / n)
    blocks = len(np.unique(rows_ix[np.nonzero(m)[0]] // (2 * H)))
    return dict(label=label, n=n, precision=p, blocks=blocks,
                lower=effective_block_wilson(p, blocks, 0.05),
                coverage=n / max(int(valid.sum()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--tickers", type=int, default=4000)
    ap.add_argument("--orthogonal", action="store_true",
                    help="residualise each channel against the others, so "
                         "the conjunction really is seven separate risks")
    a = ap.parse_args()
    H = a.horizon

    px, _ = load_panel()
    rng = np.random.default_rng(20260909)
    if a.tickers < px.shape[1]:
        px = px[sorted(rng.choice(px.columns, a.tickers, replace=False))]
    print(f"panel {px.shape[0]:,} x {px.shape[1]:,}  horizon {H} sessions\n")

    print("loading volume for the liquidity channel…", flush=True)
    vol = load_volume(px)
    print("building failure channels…", flush=True)
    S = C.build(px, vol)
    if a.orthogonal:
        print("orthogonalising the channels against each other…", flush=True)
        S = C.orthogonalise(S)
    up, valid, gone = outcomes(px, H)
    n_rows = px.shape[0]
    rows_ix = np.arange(n_rows)
    years = px.index.year.to_numpy()

    base = float(up[valid].sum() / valid.sum())
    print(f"\nbase rate P(higher in {H}): {100*base:.2f}%   "
          f"({int(valid.sum()):,} observations, "
          f"{100*gone[valid].mean():.2f}% delisted mid-horizon)\n")

    # ---- are the channels actually independent? -------------------------
    print("=" * 78)
    print("ARE THE FAILURE CHANNELS INDEPENDENT? (they need to be)")
    print("=" * 78)
    samp = np.zeros(valid.shape, dtype=bool)
    samp[::7] = True
    sel = samp & valid
    M = pd.DataFrame({c: S[c][sel] for c in C.CHANNELS}).dropna()
    corr = M.corr()
    print(f"  pairwise correlation of risk scores (n={len(M):,})")
    print("  " + "".join(f"{c[:6]:>9}" for c in C.CHANNELS))
    for c in C.CHANNELS:
        print(f"  {c[:9]:<9}" + "".join(f"{corr.loc[c, d]:>9.2f}"
                                        for d in C.CHANNELS))
    off = corr.to_numpy()[~np.eye(len(C.CHANNELS), dtype=bool)]
    print(f"\n  mean off-diagonal {off.mean():+.3f}   "
          f"most negative {off.min():+.3f}   most positive {off.max():+.3f}")
    ind = 1.0
    for _ in C.CHANNELS:
        ind *= 0.5
    g50 = C.gate(S, 0.50) & valid
    print(f"  coverage at q=0.50: {100*g50.sum()/valid.sum():.3f}% "
          f"vs {100*ind:.3f}% if the seven were independent")

    print()
    print("=" * 78)
    print("THE CONJUNCTION vs THE CONTROLS, AT EQUAL COVERAGE")
    print("=" * 78)
    rows = []
    for target in TARGET_COVERAGE:
        q = solve_quantile(S, valid, target, C.CHANNELS)
        g = C.gate(S, q)
        r = precision(g, up, valid, rows_ix, H, f"all 7 <= {q:.3f}")
        if not np.isfinite(r["precision"]):
            continue
        cov = r["coverage"]

        # --- control 1: the best SINGLE channel at the same coverage ------
        best_single, best_name = None, ""
        for c in C.CHANNELS:
            s = S[c]
            thr = np.nanquantile(s[valid], cov)
            m = np.isfinite(s) & (s <= thr)
            rr = precision(m, up, valid, rows_ix, H, c)
            if np.isfinite(rr["precision"]) and (
                    best_single is None or rr["precision"] > best_single["precision"]):
                best_single, best_name = rr, c

        # --- control 2: random selection at the same coverage -------------
        rnd = rng.random(up.shape) < cov
        rc = precision(rnd, up, valid, rows_ix, H, "random")

        rows.append((q, r, best_single, best_name, rc))

    print(f"  {'gate':<10}{'coverage':>10}{'n':>10}{'blocks':>8}"
          f"{'conj':>9}{'best-1':>9}{'name':>12}{'random':>9}{'vs base':>9}")
    for q, r, bs, bn, rc in rows:
        print(f"  <={q:<8.3f}{100*r['coverage']:>9.2f}%{r['n']:>10,}"
              f"{r['blocks']:>8}{100*r['precision']:>8.1f}%"
              f"{100*bs['precision']:>8.1f}%{bn:>12}"
              f"{100*rc['precision']:>8.1f}%{100*(r['precision']-base):>+8.1f}")

    print()
    print("=" * 78)
    print("IS THE CONJUNCTION BEATING ITS BEST SINGLE CHANNEL?")
    print("=" * 78)
    wins = sum(1 for _, r, bs, _, _ in rows if r["precision"] > bs["precision"])
    for q, r, bs, bn, _ in rows:
        d = 100 * (r["precision"] - bs["precision"])
        print(f"  <={q:.3f}   conjunction {100*r['precision']:.1f}%  vs  "
              f"{bn} alone {100*bs['precision']:.1f}%   {d:+.1f} pts")
    print(f"\n  conjunction ahead at {wins} of {len(rows)} coverage levels")

    # ---- the tightest gate, examined properly ---------------------------
    q = solve_quantile(S, valid, TARGET_COVERAGE[-1], C.CHANNELS)
    g = C.gate(S, q) & valid
    r = precision(g, up, valid, rows_ix, H, "tightest")
    print()
    print("=" * 78)
    print(f"THE TIGHTEST GATE (all channels <= {q:.3f}) IN DETAIL")
    print("=" * 78)
    print(f"  signals {r['n']:,}  coverage {100*r['coverage']:.2f}%  "
          f"blocks {r['blocks']}")
    print(f"  precision {100*r['precision']:.2f}%   "
          f"Wilson lower bound {100*r['lower']:.2f}%   "
          f"(90% target {'MET' if r['lower'] > .90 else 'NOT met'})")

    print("\n  precision by calendar year of entry:")
    yr = {}
    for y in np.unique(years):
        sel = g[years == y]
        u = up[years == y]
        if sel.sum() < 30:
            continue
        yr[int(y)] = (float(u[sel].mean()), int(sel.sum()))
    for y, (p, n) in sorted(yr.items()):
        bar = "#" * int(round(p * 40))
        print(f"    {y}  {100*p:5.1f}%  n={n:<6,} {bar}")
    if yr:
        wy = min(yr.items(), key=lambda kv: kv[1][0])
        print(f"\n  years with any signal: {len(yr)} of {len(np.unique(years))}")
        print(f"  worst year {wy[0]}: {100*wy[1][0]:.1f}% on {wy[1][1]:,} signals")

    # ---- ablation: is every channel load-bearing? -----------------------
    print()
    print("=" * 78)
    print("ABLATION — DROP ONE CHANNEL, DOES PRECISION FALL?")
    print("=" * 78)
    full = r["precision"]
    print(f"  {'dropped':<14}{'n':>10}{'coverage':>11}{'precision':>11}{'delta':>9}")
    for c in C.CHANNELS:
        sub = {k: v for k, v in S.items() if k != c}
        m = np.ones(up.shape, dtype=bool)
        for cc in sub:
            m &= np.isfinite(sub[cc]) & (sub[cc] <= q)
        rr = precision(m, up, valid, rows_ix, H, c)
        if not np.isfinite(rr["precision"]):
            print(f"  {c:<14}{rr['n']:>10,}   too few")
            continue
        print(f"  {c:<14}{rr['n']:>10,}{100*rr['coverage']:>10.2f}%"
              f"{100*rr['precision']:>10.1f}%{100*(rr['precision']-full):>+8.1f}")
    print("\n  A channel whose removal does not lower precision is not doing")
    print("  work — it is only shrinking the sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
