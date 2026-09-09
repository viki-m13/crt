#!/usr/bin/env python3
"""Does analog matching predict anything? Walk-forward, out-of-sample.

The claim this is testing is the one people actually want: paste a ticker,
see the historical patterns that match, and have what follows them tell you
what happens next. The chart is persuasive. This measures whether it is true.

How the test is built to not flatter itself:

  NO LOOK-AHEAD. At each test date t, a candidate analog window ending at s
  is admissible only if s + H < t — its own outcome had to be knowable at t.
  Matching windows whose futures overlap the forecast period is the standard
  way this technique fakes skill.

  NON-OVERLAPPING OUTCOMES. Test dates for a ticker are spaced at least H
  apart, so no two observations share a forward window.

  CLUSTERED BY DATE. Every stock moves with the market, so a hundred tickers
  on one date is nowhere near a hundred independent observations. The t-stat
  treats each DATE as one observation. Not doing this is how a 52% hit rate
  becomes "t = 9, highly significant".

  A MATCHED RANDOM CONTROL. The same pipeline, same dates, same horizon, but
  the analogs are drawn at random from the admissible set instead of chosen
  for similarity. If matching adds nothing, the two scores are the same, and
  everything the shape metric does is decoration.

  THE RIGHT BASELINE. Stocks go up more often than down, so a coin that
  always says "up" scores well above 50%. The number that matters is the hit
  rate MINUS the always-up rate on the same observations.

    python research/analog/validate.py [--tickers 400] [--points 3000]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "api"))

import _analog as A  # noqa: E402

PANEL = os.path.join(ROOT, "experiments", "monthly_dca", "cache",
                     "prices_extended.parquet")
LOOKBACK = A.LOOKBACK
HORIZON = A.HORIZON
TOP_K = A.TOP_K
STRIDE = 5            # candidate windows are sampled every STRIDE bars
MIN_GAP = A.MIN_GAP


# ------------------------------------------------------------- library ----

def build_library(px: pd.DataFrame, lookback: int, stride: int):
    """All candidate window shapes, with the bar index each one ends on.

    Returns (shapes, end_ix, col_ix, fwd_ret) where shapes is
    (n_windows, lookback) float32, end_ix is the row index of the window's
    last bar, col_ix which ticker, and fwd_ret the realised return over the
    following HORIZON bars. Everything downstream is a mask over these.
    """
    logp = np.log(px.to_numpy(dtype=np.float64))
    rets = np.diff(logp, axis=0)                     # (T-1, N)
    T1, N = rets.shape
    shapes, end_ix, col_ix, fwd = [], [], [], []
    for c in range(N):
        r = rets[:, c]
        p = logp[:, c]
        ok = np.isfinite(r)
        for j in range(lookback, T1 - HORIZON, stride):
            w = r[j - lookback:j]
            if not ok[j - lookback:j].all():
                continue
            if not (np.isfinite(p[j]) and np.isfinite(p[j + HORIZON])):
                continue
            sd = w.std(ddof=1)
            if sd < 1e-9:
                continue
            shapes.append(np.cumsum(w) / sd)
            end_ix.append(j)                 # row index in logp of last bar
            col_ix.append(c)
            fwd.append(p[j + HORIZON] - p[j])
    return (np.asarray(shapes, dtype=np.float32),
            np.asarray(end_ix), np.asarray(col_ix),
            np.asarray(fwd, dtype=np.float64))


def query_shape(logp_col: np.ndarray, t: int, lookback: int):
    w = np.diff(logp_col[t - lookback:t + 1])
    if not np.isfinite(w).all():
        return None
    sd = w.std(ddof=1)
    if sd < 1e-9:
        return None
    return (np.cumsum(w) / sd).astype(np.float32)


def pick(shapes, mask, q, top_k, end_ix, col_ix, rng=None):
    """Top-k least-distant admissible windows, deduped, or random if rng.

    `mask` must already encode BOTH admissibility rules — the time gate and,
    when measuring the shipped own-history configuration, the restriction to
    the query's own column. Getting that second one wrong is how the first
    version of this file measured a cross-sectional search over 400 tickers
    and reported it as the accuracy of a tool that searches one.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty(0, dtype=int)
    if rng is not None:
        take = min(top_k, idx.size)
        return rng.choice(idx, size=take, replace=False)
    d = np.sqrt(((shapes[idx] - q) ** 2).mean(axis=1))
    order = idx[np.argsort(d)]
    kept: list[int] = []
    for i in order:
        if all(not (col_ix[i] == col_ix[k] and abs(end_ix[i] - end_ix[k]) < MIN_GAP)
               for k in kept):
            kept.append(i)
            if len(kept) == top_k:
                break
    return np.asarray(kept, dtype=int)


# ------------------------------------------------------------ the test ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=400)
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cross-sectional", action="store_true",
                    help="search ALL tickers' history, not just the query's. "
                         "Off by default because api/analog.py searches the "
                         "query's own history and the measurement has to "
                         "describe the thing that ships.")
    a = ap.parse_args()

    px = pd.read_parquet(PANEL)
    # keep the tickers with the most usable history, deterministically
    cov = px.notna().sum().sort_values(ascending=False)
    cols = sorted(cov.index[:a.tickers])
    px = px[cols]
    print(f"panel {px.shape[0]:,} bars x {px.shape[1]} tickers  "
          f"{px.index[0].date()} -> {px.index[-1].date()}")

    print("building candidate windows…", flush=True)
    shapes, end_ix, col_ix, fwd = build_library(px, LOOKBACK, a.stride)
    print(f"  {len(shapes):,} admissible windows "
          f"({shapes.nbytes/1e6:.0f} MB)")

    logp = np.log(px.to_numpy(dtype=np.float64))
    T, N = logp.shape
    dates = px.index

    # ---- test points: spaced HORIZON apart so outcomes never overlap -----
    rng = np.random.default_rng(a.seed)
    first_t = LOOKBACK + HORIZON + 250          # need some prior history
    grid = list(range(first_t, T - HORIZON, HORIZON))
    pts = []
    for t in grid:
        live = np.flatnonzero(np.isfinite(logp[t]) & np.isfinite(logp[t - LOOKBACK])
                              & np.isfinite(logp[t + HORIZON]))
        if live.size == 0:
            continue
        take = min(live.size, max(1, a.points // max(len(grid), 1)))
        for c in rng.choice(live, size=take, replace=False):
            pts.append((t, int(c)))
    print(f"  {len(pts):,} test points across {len(grid)} non-overlapping dates")
    print(f"  search scope: "
          + ("ALL tickers (cross-sectional)" if a.cross_sectional
             else "the query ticker's own history — as shipped") + "\n")

    rows = []
    ctrl_rng = np.random.default_rng(a.seed + 1)
    for n, (t, c) in enumerate(pts, 1):
        q = query_shape(logp[:, c], t, LOOKBACK)
        if q is None:
            continue
        # THE ADMISSIBILITY GATE, identical to api/_analog.py: the match and
        # its sequel must both finish before the query window opens.
        mask = (end_ix + HORIZON) < (t - LOOKBACK)
        if not a.cross_sectional:
            mask = mask & (col_ix == c)      # the shipped configuration
        real = logp[t + HORIZON, c] - logp[t, c]

        sel = pick(shapes, mask, q, TOP_K, end_ix, col_ix)
        if sel.size == 0:
            continue
        f_med = float(np.median(fwd[sel]))
        ups = int((fwd[sel] > 0).sum())
        agree = max(ups, sel.size - ups) / sel.size

        csel = pick(shapes, mask, q, TOP_K, end_ix, col_ix, rng=ctrl_rng)
        c_med = float(np.median(fwd[csel])) if csel.size else 0.0

        rows.append(dict(date=dates[t], ticker=px.columns[c],
                         pred=f_med, ctrl=c_med, real=real, agree=agree,
                         dist=float(np.sqrt(((shapes[sel] - q) ** 2)
                                            .mean(axis=1)).mean())))
        if n % 500 == 0:
            print(f"  …{n:,}/{len(pts):,}", flush=True)

    R = pd.DataFrame(rows)
    print(f"\nscored {len(R):,} forecasts\n")
    report(R)
    tag = "cross_sectional" if a.cross_sectional else "own_history"
    R.to_csv(os.path.join(HERE, f"validation_points_{tag}.csv"), index=False)
    print(f"\n  per-forecast detail -> research/analog/"
          f"validation_points_{tag}.csv")
    return 0


def clustered_t(R: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean and t-stat treating each DATE as one observation."""
    per = R.groupby("date")[col].mean()
    n = len(per)
    if n < 3:
        return float(per.mean()), float("nan"), n
    se = per.std(ddof=1) / np.sqrt(n)
    return float(per.mean()), float(per.mean() / se) if se > 0 else float("nan"), n


def report(R: pd.DataFrame):
    R = R.copy()
    R["hit"] = (np.sign(R.pred) == np.sign(R.real)).astype(float)
    R["ctrl_hit"] = (np.sign(R.ctrl) == np.sign(R.real)).astype(float)
    R["up_hit"] = (R.real > 0).astype(float)          # the always-up baseline
    R["edge"] = R.hit - R.up_hit
    R["ctrl_edge"] = R.ctrl_hit - R.up_hit

    print("=" * 78)
    print("DIRECTIONAL ACCURACY OVER THE NEXT", HORIZON, "TRADING DAYS")
    print("=" * 78)
    for lbl, col in (("analog matcher", "hit"),
                     ("random windows (control)", "ctrl_hit"),
                     ("always say up (baseline)", "up_hit")):
        m, t, n = clustered_t(R, col)
        print(f"  {lbl:<28} {100*m:>6.2f}%   t={t:>+6.2f}  ({n} dates)")

    print()
    m, t, n = clustered_t(R, "edge")
    cm, ct, _ = clustered_t(R, "ctrl_edge")
    print(f"  matcher minus always-up      {100*m:>+6.2f} pts  t={t:>+6.2f}")
    print(f"  control minus always-up      {100*cm:>+6.2f} pts  t={ct:>+6.2f}")

    print()
    print("=" * 78)
    print("DOES THE PREDICTED SIZE TRACK THE REALISED SIZE?")
    print("=" * 78)
    print(f"  correlation(pred, real)      {R.pred.corr(R.real):>+7.4f}")
    print(f"  correlation(ctrl, real)      {R.ctrl.corr(R.real):>+7.4f}")
    per = R.groupby("date").apply(
        lambda g: g.pred.corr(g.real) if len(g) > 5 else np.nan,
        include_groups=False).dropna()
    if len(per) > 3:
        t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
        print(f"  cross-sectional IC           {per.mean():>+7.4f}  "
              f"t={t:+.2f} over {len(per)} dates")

    print()
    print("=" * 78)
    print("IS IT BETTER WHEN THE ANALOGS AGREE? (the only honest 'confidence')")
    print("=" * 78)
    print(f"  {'agreement':<14}{'n':>8}{'hit rate':>11}{'always-up':>11}{'edge':>9}")
    for lo, hi, lbl in ((0.0, 0.61, "3 of 5"), (0.61, 0.81, "4 of 5"),
                        (0.81, 1.01, "5 of 5")):
        b = R[(R.agree >= lo) & (R.agree < hi)]
        if len(b) < 30:
            print(f"  {lbl:<14}{len(b):>8}   too few")
            continue
        print(f"  {lbl:<14}{len(b):>8}{100*b.hit.mean():>10.2f}%"
              f"{100*b.up_hit.mean():>10.2f}%{100*(b.hit-b.up_hit).mean():>+8.2f}")

    print()
    print("=" * 78)
    print("IS IT BETTER WHEN THE MATCH IS CLOSER?")
    print("=" * 78)
    q = R.dist.quantile([0.25, 0.5, 0.75]).tolist()
    print(f"  {'match quality':<14}{'n':>8}{'hit rate':>11}{'always-up':>11}{'edge':>9}")
    bands = [(-np.inf, q[0], "closest 25%"), (q[0], q[1], "2nd quartile"),
             (q[1], q[2], "3rd quartile"), (q[2], np.inf, "loosest 25%")]
    for lo, hi, lbl in bands:
        b = R[(R.dist > lo) & (R.dist <= hi)]
        if len(b) < 30:
            continue
        print(f"  {lbl:<14}{len(b):>8}{100*b.hit.mean():>10.2f}%"
              f"{100*b.up_hit.mean():>10.2f}%{100*(b.hit-b.up_hit).mean():>+8.2f}")

    print()
    print("=" * 78)
    print("HOW OFTEN IS THE REALISED PATH ANYWHERE NEAR THE FORECAST?")
    print("=" * 78)
    err = (R.pred - R.real).abs()
    naive = R.real.abs()
    print(f"  median |forecast - realised|  {err.median():.4f} log-return")
    print(f"  median |realised| (predicting zero) {naive.median():.4f}")
    print(f"  the forecast beats predicting zero on "
          f"{100*(err < naive).mean():.1f}% of forecasts")

    m, t, _ = clustered_t(R, "edge")
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    hit = 100 * R.hit.mean()
    print(f"  Directional accuracy is {hit:.1f}%, against an always-up "
          f"baseline of {100*R.up_hit.mean():.1f}%.")
    print(f"  The edge over that baseline is {100*m:+.2f} points, t={t:+.2f} "
          f"clustered by date.")
    if abs(t) < 2:
        print("  That is not distinguishable from zero. The tool shows real")
        print("  history honestly; it does not forecast.")


if __name__ == "__main__":
    sys.exit(main())
