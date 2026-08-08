#!/usr/bin/env python3
"""Feature panel builder: parity spots, realized vol, ATM IV, IV-RV spreads.

All features at date t use ONLY data from dates <= t (no lookahead).
Outputs cache/spots.parquet and cache/features.parquet.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
SPOTS_PQ = os.path.join(HERE, "cache", "spots.parquet")
FEAT_PQ = os.path.join(HERE, "cache", "features.parquet")


def atm_iv_by_exp(chain: pd.DataFrame, spots: dict[str, float]) -> pd.DataFrame:
    """ATM IV per (symbol, expiration): vega-weighted IV of the 4 quotes
    nearest ATM (|delta|~0.5 side-averaged). Uses the DB's per-quote IV."""
    rows = []
    for (sym, exp), g in chain.groupby(["act_symbol", "expiration"]):
        s = spots.get(sym)
        if s is None or len(g) < 4:
            continue
        g = g[(g.vol > 0.01) & (g.vol < 5.0) & (g.bid > 0)]
        if g.empty:
            continue
        g = g.assign(dist=(g.strike - s).abs()).sort_values("dist")
        top = g.head(4)
        iv = float(np.average(top.vol, weights=np.maximum(top.vega, 1e-4)))
        rows.append((sym, exp, iv, int(top.dte.iloc[0])))
    return pd.DataFrame(rows, columns=["act_symbol", "expiration", "atm_iv", "dte"])


def build(dates: list[str]) -> None:
    spot_rows, feat_rows = [], []
    hist: dict[str, list[tuple[pd.Timestamp, float]]] = {}

    for i, d in enumerate(dates):
        ch = E.load_chain(d)
        sp = E.parity_spots(ch)
        ts = pd.Timestamp(d)
        for sym, s in sp.items():
            spot_rows.append((d, sym, s))
            hist.setdefault(sym, []).append((ts, s))

        ivt = atm_iv_by_exp(ch, sp)
        vol = E.load_vol(d)
        volmap = {}
        if vol is not None:
            volmap = vol.set_index("act_symbol")[["hv_current", "iv_current"]].to_dict("index")

        # realized vol from parity-spot history (EWMA, calendar-day scaled)
        for sym, s in sp.items():
            h = hist[sym]
            if len(h) < 8:
                continue
            arr = h[-40:]
            rets, dts = [], []
            for (t0, s0), (t1, s1) in zip(arr[:-1], arr[1:]):
                ddays = max((t1 - t0).days, 1)
                if s0 > 0 and s1 > 0:
                    rets.append(math.log(s1 / s0) / math.sqrt(ddays))
                    dts.append(ddays)
            if len(rets) < 6:
                continue
            r = np.array(rets)
            lam = 0.94
            w = lam ** np.arange(len(r) - 1, -1, -1)
            ewvar = float(np.sum(w * r * r) / np.sum(w))
            rv = math.sqrt(ewvar * 365.0)
            # front-month ATM IV (nearest dte in [15,50])
            gi = ivt[(ivt.act_symbol == sym) & (ivt.dte >= 15) & (ivt.dte <= 50)]
            iv_front = float(gi.sort_values("dte").atm_iv.iloc[0]) if len(gi) else np.nan
            gi2 = ivt[(ivt.act_symbol == sym) & (ivt.dte > 50)]
            iv_back = float(gi2.sort_values("dte").atm_iv.iloc[0]) if len(gi2) else np.nan
            vm = volmap.get(sym, {})
            # 25-delta skew on the front expiry (put IV minus call IV)
            skew = np.nan
            gfe = ch[(ch.act_symbol == sym) & (ch.dte >= 15) & (ch.dte <= 50)
                     & (ch.bid > 0) & (ch.vol > 0.01)]
            if len(gfe):
                exp1 = gfe.expiration.min()
                gg = gfe[gfe.expiration == exp1]
                p25 = gg[(gg.call_put == "Put") & (gg.delta.abs().between(0.15, 0.35))]
                c25 = gg[(gg.call_put == "Call") & (gg.delta.between(0.15, 0.35))]
                if len(p25) and len(c25):
                    pv = p25.loc[(p25.delta.abs() - 0.25).abs().idxmin()].vol
                    cv = c25.loc[(c25.delta - 0.25).abs().idxmin()].vol
                    skew = float(pv - cv)
            feat_rows.append((d, sym, s, rv, iv_front, iv_back,
                              vm.get("hv_current", np.nan), vm.get("iv_current", np.nan),
                              skew))
        if (i + 1) % 100 == 0:
            print(f"features {i+1}/{len(dates)}", flush=True)

    spots = pd.DataFrame(spot_rows, columns=["date", "act_symbol", "spot"])
    spots.to_parquet(SPOTS_PQ, index=False)
    feats = pd.DataFrame(feat_rows, columns=[
        "date", "act_symbol", "spot", "rv_ewma", "iv_front", "iv_back",
        "hv_db", "iv_db", "skew25"])
    feats.to_parquet(FEAT_PQ, index=False)
    print("saved", spots.shape, feats.shape)


if __name__ == "__main__":
    build(E.available_dates())
