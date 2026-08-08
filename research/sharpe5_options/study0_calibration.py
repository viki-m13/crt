#!/usr/bin/env python3
"""Study 0: calibrate screening Sharpe vs event-loop Sharpe.

Same sleeve, same dates, both layers: SPY short ATM straddle delta-hedged,
one cohort per obs date, held to expiry (screening's implicit assumption).
The ratio engine_sharpe / screen_sharpe quantifies the smoothing bias.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import pandas as pd

import engine as E
import portfolio as P
import verify_engine as V
from structures import bs_delta

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # --- screening layer
    df = P.load_structures()
    m = (df.structure == "short_straddle_dh") & (df.act_symbol == "SPY")
    s = df[m].groupby("date").ret.mean()
    print("screen:", P.screen_sharpe(s))

    # --- event loop: sell 1 ATM straddle every obs, hold to expiry, dh
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    spot_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    capital = 400_000.0

    def strat(day, chain, quotes, spots, vol, state, eng):
        cohorts = state.setdefault("cohorts", [])
        orders = []
        s = spots.get("SPY")
        if s is None:
            return orders
        g = chain[(chain.act_symbol == "SPY") & (chain.dte >= 15) & (chain.dte <= 50)]
        if not g.empty:
            exp1 = g.expiration.min()
            ge = g[g.expiration == exp1]
            cs = ge[(ge.call_put == "Call") & (ge.bid > 0)]
            ps = ge[(ge.call_put == "Put") & (ge.bid > 0)]
            if len(cs) and len(ps):
                cA = cs.loc[(cs.strike - s).abs().idxmin()]
                pA = ps.loc[(ps.strike - s).abs().idxmin()]
                legs = [E.Leg("SPY", exp1, float(cA.strike), "Call", -1),
                        E.Leg("SPY", exp1, float(pA.strike), "Put", -1)]
                orders.extend(legs)
                cohorts.append({"legs": legs, "iv": float(np.nanmean([cA.vol, pA.vol]))})
        # prune expired cohorts; hedge targets
        cohorts[:] = [c for c in cohorts if c["legs"][0].expiration >= day]
        tgt = {}
        for c in cohorts:
            for leg in c["legs"]:
                sp = spots.get(leg.symbol)
                if sp is None:
                    continue
                T = max((pd.Timestamp(leg.expiration) - pd.Timestamp(day)).days, 0) / 365.0
                dlt = bs_delta(sp, leg.strike, T, max(c["iv"], 0.05), leg.cp)
                tgt[leg.symbol] = tgt.get(leg.symbol, 0.0) - leg.qty * dlt * 100.0
        state["hedge_targets"] = tgt
        return orders

    dates = E.available_dates()
    bt = E.Backtester(dates, capital=capital, stock_hedge=True)
    eq, fills, _ = bt.run(strat, spot_panel)
    V.report(eq, "engine_SPY_ss_dh")


if __name__ == "__main__":
    main()
