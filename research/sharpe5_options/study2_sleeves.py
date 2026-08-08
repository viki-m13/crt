#!/usr/bin/env python3
"""Study 2 (dev only): hypothesis-driven sleeve screens on the full panel.

Each sleeve = structure subset + point-in-time entry gate. Output: screening
Sharpe (approximate), hit rate, breadth. Every config counts as a trial.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import portfolio as P

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_END = pd.Timestamp("2024-12-31")
LIQUID = {"SPY", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
          "AMD", "MU", "QCOM", "NFLX", "BA", "XOM", "JPM", "BAC", "C", "GM", "F",
          "INTC", "CSCO", "PYPL", "AVGO", "ORCL", "CRM", "ADBE", "TXN", "COST",
          "WMT", "DIS", "CAT", "GE", "PFE", "CVX", "WFC", "MS", "GS"}


def build_panel():
    df = P.load_structures()
    feats = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    feats["date"] = pd.to_datetime(feats.date)
    feats = feats.sort_values(["act_symbol", "date"])
    # past-only 1y IV rank per name
    feats["iv_rank"] = feats.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    feats["skew_rank"] = feats.groupby("act_symbol").skew25.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    feats["mom8"] = feats.groupby("act_symbol").spot.transform(lambda s: s / s.shift(24) - 1)
    feats["inv"] = feats.iv_front - feats.iv_back  # >0 = inverted term
    spyf = feats[feats.act_symbol == "SPY"][
        ["date", "iv_front", "inv", "iv_rank"]].rename(
        columns={"iv_front": "spy_iv", "inv": "spy_inv", "iv_rank": "spy_ivrank"})
    feats = feats.merge(spyf, on="date", how="left")
    feats["idio_inv"] = feats.inv - feats.spy_inv

    df = df.merge(feats.drop(columns=["spot"]), on=["date", "act_symbol"],
                  how="left", suffixes=("", "_f"))
    df["credit_yield"] = df.credit / df.margin

    ex = pd.read_parquet(os.path.join(HERE, "cache", "exits.parquet"))
    ex["date"] = pd.to_datetime(ex.date)
    ex = ex.merge(feats.drop(columns=["spot"]), on=["date", "act_symbol"], how="left")
    return df, ex


def scr(name, s, results):
    r = P.screen_sharpe(s)
    if r.get("n_weeks"):
        results.append({"sleeve": name, **{k: round(v, 3) if isinstance(v, float) else v
                                           for k, v in r.items()}})


def main():
    df, ex = build_panel()
    dev = df[df.date <= DEV_END]
    exd = ex[ex.date <= DEV_END]
    res: list[dict] = []

    # ---- A: index VRP gated
    spy = dev[dev.act_symbol == "SPY"]
    for st in ["short_straddle_dh", "iron_condor", "short_strangle25", "credit_putspread"]:
        base = spy[spy.structure == st]
        scr(f"A_{st}_uncond", base.groupby("date").ret.mean(), res)
        g = base[base.inv < 0]
        scr(f"A_{st}_contango", g.groupby("date").ret.mean(), res)
        g = base[(base.inv < 0) & (base.iv_rank < 0.8)]
        scr(f"A_{st}_cont+ivr<.8", g.groupby("date").ret.mean(), res)

    # ---- B: single-name term-inversion (earnings crush)
    for thr in (0.02, 0.04, 0.06, 0.09, 0.13):
        for k in (2, 3):
            g = exd[(exd.k == k) & (exd.idio_inv > thr)]
            scr(f"B_exit{k}_idioinv>{thr}", g.groupby("date").apply(
                lambda x: (x.short_dh_pnl / x.margin).mean(), include_groups=False), res)
            gl = g[g.act_symbol.isin(LIQUID)]
            scr(f"B_exit{k}_idioinv>{thr}_liq", gl.groupby("date").apply(
                lambda x: (x.short_dh_pnl / x.margin).mean(), include_groups=False), res)
        g2 = dev[(dev.structure == "short_straddle_dh") & (dev.idio_inv > thr)]
        scr(f"B_hold_idioinv>{thr}", g2.groupby("date").ret.mean(), res)
        g3 = dev[(dev.structure == "calendar_sf_lb") & (dev.idio_inv > thr)]
        scr(f"B_cal_idioinv>{thr}", g3.groupby("date").ret.mean(), res)
        g3l = g3[g3.act_symbol.isin(LIQUID)]
        scr(f"B_cal_idioinv>{thr}_liq", g3l.groupby("date").ret.mean(), res)
        g4 = dev[(dev.structure == "short_strangle25") & (dev.idio_inv > thr)
                 & dev.act_symbol.isin(LIQUID)]
        scr(f"B_str25_idioinv>{thr}_liq", g4.groupby("date").ret.mean(), res)

    # ---- E: weekend theta (Friday entries, k=1 exits)
    fri = exd[(exd.k == 1) & (exd.date.dt.weekday == 4)]
    scr("E_wkndtheta_all", fri.groupby("date").apply(
        lambda x: (x.short_dh_pnl / x.margin).mean(), include_groups=False), res)
    scr("E_wkndtheta_SPY", fri[fri.act_symbol == "SPY"].groupby("date").apply(
        lambda x: (x.short_dh_pnl / x.margin).mean(), include_groups=False), res)

    # ---- C: cross-sectional rank long-short (market-neutral-ish)
    # short leg uses real short_straddle_dh rows (entered at bid); long leg
    # uses real long_straddle_dh rows (entered at ask) — no spread shortcut.
    sh_ = dev[dev.structure == "short_straddle_dh"].dropna(subset=["credit_yield", "mom8"]).copy()
    lg_ = dev[dev.structure == "long_straddle_dh"][
        ["date", "act_symbol", "ret"]].rename(columns={"ret": "ret_long"})
    d = sh_.merge(lg_, on=["date", "act_symbol"], how="inner")
    d["z_cy"] = d.groupby("date").credit_yield.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    d["z_mom"] = d.groupby("date").mom8.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    for sig, nm in [("z_cy", "cy"), ("z_mom", "mom"), (None, "cy+mom")]:
        dd = d.copy()
        dd["sig"] = dd.z_cy + dd.z_mom if sig is None else dd[sig]
        def ls(x):
            if len(x) < 20:
                return np.nan
            q1 = x.sig.quantile(0.8)
            q0 = x.sig.quantile(0.2)
            short_leg = x[x.sig >= q1].ret.mean()          # real bid-side short
            long_leg = x[x.sig <= q0].ret_long.mean()      # real ask-side long
            return 0.5 * (short_leg + long_leg)
        s = dd.groupby("date").apply(ls, include_groups=False).dropna()
        scr(f"C_ls_{nm}", s, res)
        s2 = dd[dd.act_symbol.isin(LIQUID)].groupby("date").apply(ls, include_groups=False).dropna()
        scr(f"C_ls_{nm}_liq", s2, res)
        # short-only variant (top quintile short, no long leg)
        s3 = dd[dd.act_symbol.isin(LIQUID)].groupby("date").apply(
            lambda x: x[x.sig >= x.sig.quantile(0.8)].ret.mean() if len(x) >= 20 else np.nan,
            include_groups=False).dropna()
        scr(f"C_shortonly_{nm}_liq", s3, res)

    # ---- D: skew-conditioned credit spreads
    for st, cond, nm in [("credit_putspread", "hi", "put_skewrich"),
                         ("credit_callspread", "lo", "call_skewpoor")]:
        g = dev[dev.structure == st].dropna(subset=["skew_rank"])
        gg = g[g.skew_rank > 0.7] if cond == "hi" else g[g.skew_rank < 0.3]
        scr(f"D_{nm}", gg.groupby("date").ret.mean(), res)
        gg2 = gg[gg.act_symbol.isin(LIQUID)]
        scr(f"D_{nm}_liq", gg2.groupby("date").ret.mean(), res)

    out = pd.DataFrame(res).sort_values("sharpe_scr", ascending=False)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "results_study2.csv"), index=False)
    print(f"\ntrials this study: {len(res)}")


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    main()
