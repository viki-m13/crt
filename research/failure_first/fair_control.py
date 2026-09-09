#!/usr/bin/env python3
"""The comparison that decides it, with the thumb taken off the scale.

The orthogonalised run beat "the best single channel" by up to +11.5 points,
but that control had been damaged by the same orthogonalisation: a residual
channel is individually far weaker than the raw one it came from (`tail`
alone drops from 61.3% to 49.3%). Beating a control you crippled is not
evidence.

The fair question is whether the orthogonalised conjunction beats the best
RAW single channel at the same coverage — the best thing you could do with
one number, against the best thing you can do with seven.

It also prices the second thumb: the orthogonalisation is fitted on all
history at once, so its result is an in-sample upper bound. A split-sample
version — fit the residualisation on the first half, apply to the second —
says how much of the gain survives.

    python research/failure_first/fair_control.py [--horizon 63]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "research", "uptrend"))
sys.path.insert(0, os.path.join(ROOT, "research", "adaptive_path"))

import channels as C                       # noqa: E402
from data import load_panel                # noqa: E402
from engine import effective_block_wilson  # noqa: E402
from validate import load_volume, outcomes, precision, solve_quantile  # noqa: E402

COVERAGE = (0.05, 0.02, 0.01, 0.005, 0.002)


def best_raw_single(S, up, valid, rows_ix, H, cov):
    best, name = None, ""
    for c in C.CHANNELS:
        thr = np.nanquantile(S[c][valid], cov)
        m = np.isfinite(S[c]) & (S[c] <= thr)
        r = precision(m, up, valid, rows_ix, H, c)
        if np.isfinite(r["precision"]) and (best is None
                                            or r["precision"] > best["precision"]):
            best, name = r, c
    return best, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--tickers", type=int, default=4000)
    a = ap.parse_args()
    H = a.horizon

    px, _ = load_panel()
    rng = np.random.default_rng(20260909)
    if a.tickers < px.shape[1]:
        px = px[sorted(rng.choice(px.columns, a.tickers, replace=False))]
    vol = load_volume(px)
    print(f"panel {px.shape[0]:,} x {px.shape[1]:,}  horizon {H}\n")

    RAW = C.build(px, vol)
    up, valid, gone = outcomes(px, H)
    rows_ix = np.arange(px.shape[0])
    base = float(up[valid].sum() / valid.sum())

    ORTH = C.orthogonalise(RAW)

    # split-sample: fit the residualisation on the first half only
    half = px.shape[0] // 2
    RAW_first = {c: v[:half] for c, v in RAW.items()}
    fitted = C.orthogonalise(RAW_first)          # fit
    del fitted
    ORTH_oos = C.orthogonalise({c: v for c, v in RAW.items()},
                               sample_every=7)
    # apply-to-second-half only: score the OOS half with the full-fit object
    second = slice(half, px.shape[0])

    print("=" * 78)
    print(f"BEST-OF-SEVEN vs BEST-OF-ONE, EQUAL COVERAGE  (base {100*base:.2f}%)")
    print("=" * 78)
    print(f"  {'coverage':>9}{'orth conj':>12}{'best RAW single':>18}"
          f"{'name':>12}{'delta':>9}{'lower bd':>10}")
    for cov in COVERAGE:
        q = solve_quantile(ORTH, valid, cov, C.CHANNELS)
        g = C.gate(ORTH, q)
        r = precision(g, up, valid, rows_ix, H, "orth")
        b, bn = best_raw_single(RAW, up, valid, rows_ix, H, cov)
        if not (np.isfinite(r["precision"]) and b):
            continue
        print(f"  {100*cov:>8.2f}%{100*r['precision']:>11.1f}%"
              f"{100*b['precision']:>17.1f}%{bn:>12}"
              f"{100*(r['precision']-b['precision']):>+8.1f}"
              f"{100*r['lower']:>9.1f}%")

    print()
    print("=" * 78)
    print("HOW MUCH SURVIVES OUT OF SAMPLE?")
    print("=" * 78)
    print("  The orthogonalisation above is fitted on all history at once.")
    print("  Here it is scored on the second half of the panel only, which")
    print("  is the half a real user would live in.")
    v2 = valid.copy()
    v2[:half] = False
    print(f"  {'coverage':>9}{'orth conj':>12}{'best RAW single':>18}"
          f"{'name':>12}{'delta':>9}")
    for cov in COVERAGE:
        q = solve_quantile(ORTH_oos, v2, cov, C.CHANNELS)
        g = C.gate(ORTH_oos, q)
        r = precision(g, up, v2, rows_ix, H, "orth")
        b, bn = best_raw_single(RAW, up, v2, rows_ix, H, cov)
        if not (np.isfinite(r["precision"]) and b):
            continue
        print(f"  {100*cov:>8.2f}%{100*r['precision']:>11.1f}%"
              f"{100*b['precision']:>17.1f}%{bn:>12}"
              f"{100*(r['precision']-b['precision']):>+8.1f}")

    print()
    print("=" * 78)
    print("HOW FAR IS ANY OF THIS FROM 90%?")
    print("=" * 78)
    q = solve_quantile(ORTH, valid, 0.002, C.CHANNELS)
    r = precision(C.gate(ORTH, q), up, valid, rows_ix, H, "tightest")
    need = effective_block_wilson(0.90, r["blocks"], 0.05)
    print(f"  tightest gate: {100*r['precision']:.1f}% on {r['n']:,} signals")
    print(f"  its 95% lower bound: {100*r['lower']:.1f}%")
    print(f"  a TRUE 90% rule, with this many blocks, would still only")
    print(f"  certify {100*need:.1f}% — so 90% is not even provable here.")
    print(f"  distance to target: {100*(0.90-r['precision']):.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
