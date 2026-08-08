#!/usr/bin/env python3
"""Study 1 (dev set only): what predicts cross-sectional option returns?

For each obs date, Spearman IC between point-in-time signals and realized
structure returns (entry→expiry, honest worst-side fills). t-stats over dates.
DEV SET ONLY (<= 2024-12-31). Results appended to RESEARCH_LOG.md by hand.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import portfolio as P

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_END = "2024-12-31"


def main():
    df = P.load_structures()
    feats = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    feats["date"] = pd.to_datetime(feats.date)

    d = df[df.structure == "short_straddle_dh"].merge(
        feats, on=["date", "act_symbol"], how="left", suffixes=("", "_f"))
    d = d[d.date <= DEV_END].copy()
    d["ivrv"] = d.iv - d.rv_ewma
    d["ivrv_ratio"] = d.iv / d.rv_ewma.clip(lower=0.03)
    d["ivrv_db"] = d.iv_db - d.hv_db
    d["term"] = d.iv_back - d.iv_front
    d["credit_yield"] = d.credit / d.margin
    # 8-week spot momentum (uses only past spots)
    d = d.sort_values(["act_symbol", "date"])
    d["mom8"] = d.groupby("act_symbol").spot.transform(lambda s: s / s.shift(24) - 1)
    # realized vol premium outcome: iv at entry minus realized move measure
    d["absret_exp"] = (d.spot_exp / d.spot - 1).abs()

    signals = ["ivrv", "ivrv_ratio", "ivrv_db", "term", "credit_yield", "mom8"]
    print(f"rows={len(d)} dates={d.date.nunique()} syms={d.act_symbol.nunique()}")
    print("\nIC of signal -> short_straddle_dh return (dev set):")
    for sig in signals:
        ics = []
        for _, g in d.groupby("date"):
            g = g[[sig, "ret"]].dropna()
            if len(g) >= 15:
                ic = spearmanr(g[sig], g.ret).statistic
                if np.isfinite(ic):
                    ics.append(ic)
        ics = np.array(ics)
        if len(ics) > 30:
            t = ics.mean() / (ics.std() / np.sqrt(len(ics)))
            print(f"  {sig:>14}: IC={ics.mean():+.4f}  t={t:+.2f}  n={len(ics)}")

    print("\nBaseline sleeve screens (dev):")
    for st in ["short_straddle", "short_straddle_dh", "long_straddle_dh",
               "iron_condor", "short_strangle25", "calendar_sf_lb"]:
        for sym_set, name in [(None, "all"), ({"SPY"}, "SPY"),
                              ({"SPY", "AAPL", "NVDA", "MSFT", "AMZN", "MU", "QCOM", "AMD"}, "liquid8")]:
            m = (df.structure == st) & (df.date <= DEV_END)
            if sym_set:
                m &= df.act_symbol.isin(sym_set)
            s = P.sleeve_series(df, m)
            r = P.screen_sharpe(s)
            if r.get("n_weeks", 0) or not np.isnan(r.get("sharpe_scr", np.nan)):
                print(f"  {st:>18} [{name:>7}]: scrSharpe={r['sharpe_scr']:+.2f} "
                      f"hit={r.get('hit', float('nan')):.2f} nW={r.get('n_weeks')}")


if __name__ == "__main__":
    main()
