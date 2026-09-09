#!/usr/bin/env python3
"""Where does 90% actually live? The frontier of horizon x diversification.

The adaptive engine's 90% gate never opened at 30, 126 or 252 sessions: no
threshold's conservative precision bound cleared the target, so it stayed
silent. That answers "can a model call a single stock higher with 90%
confidence" — no — without answering the more useful question, which is what
DOES reach 90% and what it costs.

Two dials move the probability that something is higher later, and neither is
a forecast:

  HORIZON. Drift accumulates; a stock is higher 30 sessions later 51.4% of
  the time and 504 sessions later 59.7% of the time.
  DIVERSIFICATION. A basket's idiosyncratic risk falls with sqrt(N) while its
  drift does not, so P(basket higher) rises fast with N even though the
  per-name probability is unchanged. This is the honest route to a high hit
  rate and it is not stock picking.

Every cell is measured on the point-in-time panel including the 4,725 tickers
that stopped trading, and reported twice: once scoring a delisted name at its
last traded price, and once as a total loss. The truth is between them, and
the gap tells you how much of any number is survivorship.

    python research/uptrend/frontier.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load_panel  # noqa: E402

HORIZONS = [30, 63, 126, 252, 504, 756]
SIZES = [1, 5, 10, 20, 50]
DRAWS = 300               # random baskets per (date, size)
STEP = 21                 # dates sampled
SEED = 20260909


def eligible(arr, ok, i, sma200, vol60, rising, mkt_up):
    """The cheap quality filter that scored best in baserates.py."""
    f = ok[i] & (arr[i] > sma200[i]) & rising[i] & (vol60[i] < 0.20)
    return f & bool(mkt_up[i])


def main():
    px, _ = load_panel()
    arr = px.to_numpy(dtype=np.float32)
    n, k = arr.shape
    ok = np.isfinite(arr)
    last = np.where(ok.any(0), n - 1 - ok[::-1].argmax(0), -1)
    print(f"panel {n:,} bars x {k:,} tickers  "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")

    F = pd.DataFrame(arr)
    sma200 = F.rolling(200, min_periods=150).mean().to_numpy(dtype=np.float32)
    lr = np.diff(np.log(np.where(arr > 0, arr, np.nan)), axis=0, prepend=np.nan)
    vol60 = (pd.DataFrame(lr).rolling(60, min_periods=40).std().to_numpy()
             * np.sqrt(252)).astype(np.float32)
    rising = np.zeros_like(sma200, dtype=bool)
    rising[21:] = sma200[21:] > sma200[:-21]
    mkt = np.nanmean(np.where(sma200 > 0, arr / sma200, np.nan), axis=1)
    mkt_up = mkt > 1.0

    rng = np.random.default_rng(SEED)
    years = px.index.year.to_numpy()
    rows = []
    for H in HORIZONS:
        for i in range(260, n - H, STEP):
            elig = np.flatnonzero(eligible(arr, ok, i, sma200, vol60, rising, mkt_up))
            if elig.size < max(SIZES):
                continue
            # forward total return per name; a name that stops trading is
            # held to its last observed price (optimistic) or written to zero
            # (pessimistic). Both are carried through.
            fwd_i = np.minimum(i + H, last[elig])
            r_last = arr[fwd_i, elig] / arr[i, elig] - 1.0
            gone = (i + H) > last[elig]
            r_zero = np.where(gone, -1.0, r_last)
            good = np.isfinite(r_last)
            if good.sum() < max(SIZES):
                continue
            r_last, r_zero = r_last[good], r_zero[good]
            for N in SIZES:
                pick = rng.integers(0, len(r_last), (DRAWS, N))
                rows.append(dict(
                    H=H, N=N, year=years[i], i=i,
                    p_last=float((r_last[pick].mean(1) > 0).mean()),
                    p_zero=float((r_zero[pick].mean(1) > 0).mean())))
        print(f"  horizon {H}: {sum(1 for r in rows if r['H']==H)//len(SIZES)} dates",
              flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "frontier.csv"), index=False)

    print()
    print("=" * 78)
    print("P(EQUAL-WEIGHT BASKET IS HIGHER), delisted held at last price")
    print("=" * 78)
    print(f"  {'horizon':<10}" + "".join(f"{('N=' + str(N)):>10}" for N in SIZES))
    for H in HORIZONS:
        g = R[R.H == H]
        cells = "".join(f"{100*g[g.N==N].p_last.mean():>9.1f}%" for N in SIZES)
        print(f"  {H:<10}{cells}")

    print()
    print("=" * 78)
    print("SAME, delisted written to ZERO (the pessimistic bound)")
    print("=" * 78)
    print(f"  {'horizon':<10}" + "".join(f"{('N=' + str(N)):>10}" for N in SIZES))
    for H in HORIZONS:
        g = R[R.H == H]
        cells = "".join(f"{100*g[g.N==N].p_zero.mean():>9.1f}%" for N in SIZES)
        print(f"  {H:<10}{cells}")

    print()
    print("=" * 78)
    print("THE NUMBER THAT DECIDES IT: WORST CALENDAR YEAR")
    print("=" * 78)
    print("  An average of 90% built from great years and terrible ones is")
    print("  not a 90% promise to whoever shows up in a bad one.")
    print(f"  {'horizon':<10}" + "".join(f"{('N=' + str(N)):>10}" for N in SIZES))
    for H in HORIZONS:
        g = R[R.H == H]
        cells = ""
        for N in SIZES:
            by = g[g.N == N].groupby("year").p_last.mean()
            cells += f"{100*by.min():>9.1f}%" if len(by) else "        -"
        print(f"  {H:<10}{cells}")

    print()
    print("=" * 78)
    print("WHAT CLEARS 90% ON EVERY MEASURE")
    print("=" * 78)
    hit = []
    for H in HORIZONS:
        for N in SIZES:
            g = R[(R.H == H) & (R.N == N)]
            if g.empty:
                continue
            by = g.groupby("year").p_last.mean()
            if g.p_last.mean() > .90 and g.p_zero.mean() > .90 and by.min() > .90:
                hit.append((H, N, g.p_last.mean(), by.min()))
    if hit:
        for H, N, p, w in hit:
            print(f"  horizon {H:>4} sessions, basket of {N:>3}: "
                  f"{100*p:.1f}% overall, worst year {100*w:.1f}%")
    else:
        print("  NOTHING. No combination of horizon and basket size in this")
        print("  grid clears 90% on the average, the pessimistic delisting")
        print("  bound, AND the worst calendar year simultaneously.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
