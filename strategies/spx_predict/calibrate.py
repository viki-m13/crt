#!/usr/bin/env python3
"""Calibrate the v2 pricing surface against REAL option chains.

signal.py prices spreads from a modelled surface:

    ATM IV  = IVM * sqrt(W*rv60^2 + (1-W)*rvbar^2)
    IV(K)   = ATM * (1 + BETA*ln(F/K))

Those constants were chosen a priori. live_validation.py compares them to one
live snapshot and reports the model booking 1.413x the natural credit on SPY
and 1.582x on SPX, with `model_conservative: false`. A single snapshot cannot
tell you which constant is wrong, so this calibrates against 1254 observation
dates of real SPY chains (research/sharpe5_options/cache) and fits IVM and BETA
to the market rather than to intuition.

Method, per observation date with a 60-110 DTE SPY expiry:
  - market ATM IV: vega-weighted mean IV of the 4 quotes nearest the forward
  - market skew slope: regress IV on ln(F/K) across quotes within +/-12%
  - model inputs rv60 and rvbar recomputed from the same SPY panel signal.py
    uses, so the comparison isolates the surface constants
Then report the IVM and BETA that minimise squared error, plus what those
imply for booked-vs-natural credit on the deployed structure.

Writes strategies/spx_predict/data/calibration.json.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CHAINS = os.path.join(REPO, "research", "sharpe5_options", "cache", "chains")
OUT = os.path.join(HERE, "data", "calibration.json")

sys.path.insert(0, HERE)


def load_panel():
    from signal import Market, SPY_PATH  # noqa
    return Market(SPY_PATH)


def market_surface(day, spot_hint=None):
    """(atm_iv, skew_beta, dte, forward) from a real chain, or None."""
    try:
        import pandas as pd
    except ImportError:
        return None
    p = os.path.join(CHAINS, f"{day}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df = df[(df.act_symbol == "SPY") & (df.bid > 0) & (df.ask > 0)
            & (df.vol > 0.02) & (df.vol < 3.0)]
    if df.empty:
        return None
    df = df.assign(dte=(pd.to_datetime(df.expiration) - pd.to_datetime(df.date)).dt.days)
    g = df[(df.dte >= 60) & (df.dte <= 110)]
    if g.empty:
        return None
    exp = g.expiration.min()
    ge = g[g.expiration == exp].copy()
    if len(ge) < 6:
        return None
    # forward from put-call parity on the nearest-the-money pair
    c = ge[ge.call_put == "Call"].set_index("strike")
    p_ = ge[ge.call_put == "Put"].set_index("strike")
    ks = c.index.intersection(p_.index)
    if len(ks) < 2:
        return None
    cm = (c.loc[ks, "bid"] + c.loc[ks, "ask"]) / 2
    pm = (p_.loc[ks, "bid"] + p_.loc[ks, "ask"]) / 2
    diff = (cm - pm).abs().sort_values()
    near = diff.index[:5]
    F = float(np.median([cm[k] - pm[k] + float(k) for k in near]))
    if not np.isfinite(F) or F <= 0:
        return None
    ge["x"] = np.log(F / ge.strike.astype(float))
    ge["vega_w"] = np.maximum(ge.vega.astype(float), 1e-4)
    band = ge[ge.x.abs() <= 0.12]
    if len(band) < 6:
        return None
    # ATM IV: vega-weighted mean of the 4 quotes nearest x=0
    near_atm = band.reindex(band.x.abs().sort_values().index).head(4)
    atm = float(np.average(near_atm.vol.astype(float),
                           weights=near_atm.vega_w))
    # skew: IV = atm*(1 + beta*x)  ->  (IV/atm - 1) = beta*x
    y = (band.vol.astype(float) / atm - 1.0).values
    x = band.x.values
    w = band.vega_w.values
    if np.sum(x * x * w) <= 0:
        return None
    beta = float(np.sum(w * x * y) / np.sum(w * x * x))
    return atm, beta, int(ge.dte.iloc[0]), F


def main():
    mkt = load_panel()
    idx = {str(d): i for i, d in enumerate(mkt.dates)}
    days = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(CHAINS, "*.parquet")))
    rows = []
    for day in days:
        i = idx.get(day)
        if i is None:
            continue
        if not (np.isfinite(mkt.rv[i]) and np.isfinite(mkt.rvbar[i])):
            continue
        ms = market_surface(day)
        if ms is None:
            continue
        atm_mkt, beta_mkt, dte, F = ms
        base = math.sqrt(0.30 * mkt.rv[i] ** 2 + 0.70 * mkt.rvbar[i] ** 2)
        if not np.isfinite(base) or base <= 0:
            continue
        rows.append({"date": day, "atm_mkt": atm_mkt, "beta_mkt": beta_mkt,
                     "base": base, "rv60": float(mkt.rv[i]),
                     "rvbar": float(mkt.rvbar[i]), "dte": dte})
    if not rows:
        print("no calibration rows")
        return

    atm = np.array([r["atm_mkt"] for r in rows])
    base = np.array([r["base"] for r in rows])
    beta = np.array([r["beta_mkt"] for r in rows])
    # least-squares multiplier: minimise ||atm - k*base||
    ivm_fit = float(np.sum(atm * base) / np.sum(base * base))
    ivm_median = float(np.median(atm / base))
    beta_fit = float(np.median(beta))

    print("=" * 76)
    print("PRICING-SURFACE CALIBRATION vs REAL SPY CHAINS")
    print("=" * 76)
    print(f"observations: {len(rows)}  ({rows[0]['date']} -> {rows[-1]['date']})")
    print(f"\n  market ATM IV : mean {atm.mean():.4f}  median {np.median(atm):.4f}")
    print(f"  model base    : mean {base.mean():.4f}  (sqrt of blended variance)")
    print(f"\n  IVM currently 1.15  ->  model ATM mean {1.15*base.mean():.4f}")
    print(f"  IVM least-squares fit : {ivm_fit:.4f}")
    print(f"  IVM median ratio      : {ivm_median:.4f}")
    print(f"\n  BETA currently 1.00")
    print(f"  BETA market median    : {beta_fit:.4f}"
          f"   (p25 {np.percentile(beta,25):.3f}, p75 {np.percentile(beta,75):.3f})")
    err_now = float(np.mean((1.15 * base - atm) ** 2) ** 0.5)
    err_fit = float(np.mean((ivm_fit * base - atm) ** 2) ** 0.5)
    print(f"\n  RMSE  IVM=1.15 : {err_now:.4f}")
    print(f"  RMSE  IVM={ivm_fit:.3f} : {err_fit:.4f}   "
          f"({(1-err_fit/err_now)*100:.0f}% better)")
    bias = float(np.mean(1.15 * base / atm - 1))
    print(f"\n  current model overstates ATM IV by {bias*100:+.1f}% on average")

    out = {"n_obs": len(rows), "first": rows[0]["date"], "last": rows[-1]["date"],
           "ivm_current": 1.15, "ivm_fit": round(ivm_fit, 4),
           "ivm_median_ratio": round(ivm_median, 4),
           "beta_current": 1.0, "beta_fit": round(beta_fit, 4),
           "beta_p25": round(float(np.percentile(beta, 25)), 4),
           "beta_p75": round(float(np.percentile(beta, 75)), 4),
           "atm_mkt_mean": round(float(atm.mean()), 4),
           "model_atm_bias_pct": round(bias * 100, 2),
           "rmse_current": round(err_now, 4), "rmse_fit": round(err_fit, 4)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
