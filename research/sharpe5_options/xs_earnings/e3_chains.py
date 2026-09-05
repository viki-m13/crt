#!/usr/bin/env python3
"""Consolidate the EOD chain cache and derive per-(date,symbol,expiry) ATM rows.

Outputs
  panel.parquet  — every chain row, one file (date, symbol, expiration, strike,
                   cbid, cask, pbid, pask, civ, piv) pivoted call/put side by side
  atm.parquet    — per (date, symbol, expiry): put-call-parity spot, ATM strike,
                   straddle bid/ask/mid, ATM IV, DTE

Spot comes from put-call parity so it is internally consistent with the raw
(unadjusted) strikes; adjusted daily bars are never mixed into option maths.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = "/home/user/crt/research/sharpe5_options/cache/chains"
FRED = "/home/user/bonds/data/fred/DGS1MO.csv"


def rf_curve() -> pd.Series:
    d = pd.read_csv(FRED)
    c = [x for x in d.columns if x.upper() != "DATE"][0]
    s = pd.Series(pd.to_numeric(d[c], errors="coerce").values / 100.0,
                  index=pd.to_datetime(d[d.columns[0]]))
    return s.ffill().dropna()


def main():
    files = sorted(glob.glob(os.path.join(CHAINS, "*.parquet")))
    print(f"{len(files)} chain dates")
    rf = rf_curve()

    frames = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[(d.ask > 0) & (d.ask >= d.bid) & (d.bid >= 0)]
        if not len(d):
            continue
        c = d[d.call_put == "Call"][["date", "act_symbol", "expiration", "strike",
                                     "bid", "ask", "vol", "vega"]]
        p = d[d.call_put == "Put"][["date", "act_symbol", "expiration", "strike",
                                    "bid", "ask", "vol"]]
        m = c.merge(p, on=["date", "act_symbol", "expiration", "strike"],
                    suffixes=("_c", "_p"))
        frames.append(m)
    pan = pd.concat(frames, ignore_index=True)
    pan.columns = ["date", "symbol", "expiration", "strike", "cbid", "cask", "civ",
                   "cvega", "pbid", "pask", "piv"]
    pan["date"] = pd.to_datetime(pan["date"])
    pan["expiration"] = pd.to_datetime(pan["expiration"])
    pan["cmid"] = (pan.cbid + pan.cask) / 2
    pan["pmid"] = (pan.pbid + pan.pask) / 2
    pan.to_parquet(os.path.join(HERE, "panel.parquet"), index=False)
    print(f"panel: {len(pan):,} call/put pairs, {pan.symbol.nunique()} symbols, "
          f"{pan.date.min().date()}..{pan.date.max().date()}")

    # ---- put-call parity spot, per (date, symbol) ------------------------
    pan["dte"] = (pan.expiration - pan.date).dt.days
    pan = pan[pan.dte > 0]
    pan["r"] = rf.reindex(pan.date.values, method="ffill").values
    pan["tau"] = pan.dte / 365.0
    pan["synth"] = pan.cmid - pan.pmid + pan.strike * np.exp(-pan.r * pan["tau"])
    pan["parity_gap"] = (pan.cmid - pan.pmid).abs()

    # nearest expiry only, strike with the smallest |C-P| = closest to ATM
    near = pan.sort_values(["date", "symbol", "dte"])
    near = near[near.dte == near.groupby(["date", "symbol"]).dte.transform("min")]
    idx = near.groupby(["date", "symbol"]).parity_gap.idxmin()
    spot = near.loc[idx, ["date", "symbol", "synth"]].rename(columns={"synth": "spot"})
    pan = pan.merge(spot, on=["date", "symbol"], how="inner")

    # ---- ATM row per (date, symbol, expiry) ------------------------------
    pan["mny"] = (pan.strike / pan.spot - 1).abs()
    a = pan.loc[pan.groupby(["date", "symbol", "expiration"]).mny.idxmin()].copy()
    a["str_bid"] = a.cbid + a.pbid          # sell the straddle -> receive
    a["str_ask"] = a.cask + a.pask          # buy the straddle  -> pay
    a["str_mid"] = a.cmid + a.pmid
    a["atm_iv"] = (a.civ + a.piv) / 2
    a["spread_frac"] = (a.str_ask - a.str_bid) / a.str_mid
    a = a[["date", "symbol", "expiration", "dte", "tau", "r", "spot", "strike", "mny",
           "cbid", "cask", "pbid", "pask", "str_bid", "str_ask", "str_mid",
           "civ", "piv", "atm_iv", "spread_frac", "cvega"]]
    a.to_parquet(os.path.join(HERE, "atm.parquet"), index=False)

    print(f"atm: {len(a):,} rows")
    print(f"  DTE of nearest expiry: median {a.groupby(['date','symbol']).dte.min().median():.0f}")
    print(f"  |K/S-1| at ATM: median {a.mny.median():.4f}  p90 {a.mny.quantile(.9):.4f}")
    print(f"  straddle spread as frac of mid: median {a.spread_frac.median():.3f}  "
          f"p25 {a.spread_frac.quantile(.25):.3f}  p75 {a.spread_frac.quantile(.75):.3f}")


if __name__ == "__main__":
    main()
