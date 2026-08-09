#!/usr/bin/env python3
"""Acid test: does the recalibrated surface book the credit the market pays?

calibrate.py fits the surface constants to real chains. This checks what those
constants do to the number that actually matters — the credit the backtest
books versus the credit a natural (worst-side) fill would receive — over every
observation date with a real SPY chain, not the single live snapshot
live_validation.py uses.

Ratio > 1 means the backtest is optimistic (booking money the market will not
pay). The deployed constants give ~1.41 on SPY. Target is <= 1.0.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CHAINS = os.path.join(REPO, "research", "sharpe5_options", "cache", "chains")
sys.path.insert(0, HERE)

import signal as S  # noqa: E402


def model_credit(F, S0, K1, K2, T, r, atm, beta, slip):
    def iv(K):
        return max(atm * (1.0 + beta * math.log(F / K)), 0.03)
    v = S.bs_put_F(F, K1, T, iv(K1), r) - S.bs_put_F(F, K2, T, iv(K2), r)
    return v * (1 - slip)


def main(otm=0.03, width=0.03):
    mkt = S.Market(S.SPY_PATH)
    idx = {str(d): i for i, d in enumerate(mkt.dates)}
    days = sorted(os.path.basename(p)[:-8]
                  for p in glob.glob(os.path.join(CHAINS, "*.parquet")))
    variants = {
        "deployed (IVM 1.15, BETA 1.0)": (1.15, 1.0),
        "IVM 1.05, BETA 1.0": (1.05, 1.0),
        "IVM 1.00, BETA 3.9": (1.00, 3.9),
        "IVM 0.98, BETA 3.9 (calibrated)": (0.98, 3.9),
        "IVM 0.95, BETA 3.9": (0.95, 3.9),
    }
    acc = {k: [] for k in variants}
    n = 0
    for day in days:
        i = idx.get(day)
        if i is None or not np.isfinite(mkt.rv[i]) or not np.isfinite(mkt.rvbar[i]):
            continue
        df = pd.read_parquet(os.path.join(CHAINS, f"{day}.parquet"))
        df = df[(df.act_symbol == "SPY") & (df.call_put == "Put") & (df.ask > 0)]
        if df.empty:
            continue
        df = df.assign(dte=(pd.to_datetime(df.expiration)
                            - pd.to_datetime(df.date)).dt.days)
        g = df[(df.dte >= 60) & (df.dte <= 110)]
        if g.empty:
            continue
        exp = g.expiration.min()
        ge = g[g.expiration == exp]
        S0 = float(mkt.px[i])
        T = float(ge.dte.iloc[0]) / 365.0
        r = float(mkt.rate[i])
        F = S0 * math.exp(r * T)
        K1t, K2t = S0 * (1 - otm), S0 * (1 - otm - width)
        sh = ge[ge.bid > 0]
        if sh.empty:
            continue
        sl = sh.loc[(sh.strike - K1t).abs().idxmin()]
        wl = ge.loc[(ge.strike - K2t).abs().idxmin()]
        K1, K2 = float(sl.strike), float(wl.strike)
        if K2 >= K1:
            continue
        natural = float(sl.bid) - float(wl.ask)      # worst-side credit
        if natural <= 0:
            continue
        base = math.sqrt(S.W_BLEND * mkt.rv[i] ** 2
                         + (1 - S.W_BLEND) * mkt.rvbar[i] ** 2)
        n += 1
        for lab, (ivm, beta) in variants.items():
            mc = model_credit(F, S0, K1, K2, T, r, ivm * base, beta, S.SLIP)
            acc[lab].append(mc / natural)

    print("=" * 78)
    print(f"BOOKED vs NATURAL CREDIT — {n} real SPY chains, {otm:.0%} OTM / {width:.0%} wide")
    print("=" * 78)
    print(f"  {'surface':<36}{'mean':>8}{'median':>9}{'p90':>8}  verdict")
    for lab in variants:
        a = np.array(acc[lab])
        if not len(a):
            continue
        v = "OPTIMISTIC" if np.median(a) > 1.02 else (
            "conservative" if np.median(a) < 0.98 else "MATCHED")
        print(f"  {lab:<36}{a.mean():>8.3f}{np.median(a):>9.3f}"
              f"{np.percentile(a,90):>8.3f}  {v}")
    print("\n  Ratio > 1 = the backtest books credit the market will not pay.")
    print("  live_validation.py reports 1.413 for the deployed surface on one")
    print("  snapshot; this is the same measurement across every chain.")


if __name__ == "__main__":
    import sys as _s
    o = float(_s.argv[1]) if len(_s.argv) > 1 else 0.03
    w = float(_s.argv[2]) if len(_s.argv) > 2 else 0.03
    main(o, w)
