#!/usr/bin/env python3
"""Study 8: attack the two levers the ceiling formula actually exposes.

IR = IC x sqrt(BR). Everything I built used one signal (IC 0.037) and one
holding period (12 rounds/yr). Both are choices, not constraints:

  LEVER 1 (IC):  combine weakly-correlated signals into a composite. Weights
                 are fit by cross-sectional rank regression on TRAINING data
                 only and applied forward, so a composite that only works with
                 hindsight shows up as a dead OOS IC.

  LEVER 2 (BR):  trade the shortest tenor available (8-14 DTE) instead of
                 monthly. Same names, ~26 independent rounds per year instead
                 of ~12 -> sqrt(2) on breadth, for free, provided the premium
                 still clears the spread at that tenor.

  LEVER 3 (tail): the holdout failure was a short-premium book with nothing
                 stopping it in a stress regime. Add a de-risking overlay
                 driven only by past data, and let walk-forward judge it.

Everything is scored on WORST-SIDE fills, defined-risk structures only (no
sleeve may go insolvent), and validated by walk-forward inside DEV. The 2025+
holdout has already been spent once and is not touched here.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import portfolio as P

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_END = pd.Timestamp("2024-12-31")
LIQUID = {"SPY", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
          "AMD", "MU", "QCOM", "NFLX", "BA", "XOM", "JPM", "BAC", "C", "GM", "F",
          "INTC", "CSCO", "PYPL", "AVGO", "ORCL", "CRM", "ADBE", "TXN", "COST",
          "WMT", "DIS", "CAT", "GE", "PFE", "CVX", "WFC", "MS", "GS"}

SIGNALS = ["z_cy", "z_ivrv", "z_inv", "z_mom", "z_skew", "z_ivr", "z_vov"]


def build_panel(structure="credit_putspread", short_dated=False):
    src = "shortdated.parquet" if short_dated else "structures.parquet"
    st = pd.read_parquet(os.path.join(HERE, "cache", src))
    st["date"] = pd.to_datetime(st.date)
    st = st[st.act_symbol.isin(LIQUID)]
    key = {"credit_putspread": "sd_putspread", "short_strangle25": "sd_strangle25",
           "short_straddle_dh": "sd_straddle"}.get(structure, structure) \
        if short_dated else structure
    st = st[st.structure == key].copy()
    st["ret"] = st.pnl / st.margin

    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    f = f.sort_values(["act_symbol", "date"])
    f["inv"] = f.iv_front - f.iv_back
    f["ivrv"] = f.iv_front - f.rv_ewma
    f["mom8"] = f.groupby("act_symbol").spot.transform(lambda s: s / s.shift(24) - 1)
    f["iv_rank"] = f.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(150, min_periods=40).rank(pct=True))
    # vol-of-vol: dispersion of the name's own front IV over the past ~2 months
    f["vov"] = f.groupby("act_symbol").iv_front.transform(
        lambda s: s.rolling(24, min_periods=8).std())
    # market stress state, past-only, from SPY alone
    spy = f[f.act_symbol == "SPY"][["date", "iv_front", "inv", "spot"]].rename(
        columns={"iv_front": "spy_iv", "inv": "spy_inv", "spot": "spy_spot"})
    spy = spy.sort_values("date")
    spy["spy_dd"] = spy.spy_spot / spy.spy_spot.cummax() - 1
    spy["spy_iv_z"] = (spy.spy_iv - spy.spy_iv.rolling(150, min_periods=40).mean()) \
        / spy.spy_iv.rolling(150, min_periods=40).std()
    spy["spy_mom"] = spy.spy_spot / spy.spy_spot.shift(12) - 1
    f = f.merge(spy[["date", "spy_iv", "spy_inv", "spy_dd", "spy_iv_z", "spy_mom"]],
                on="date", how="left")
    f["idio_inv"] = f.inv - f.spy_inv

    d = st.merge(f.drop(columns=["spot"]), on=["date", "act_symbol"], how="left")
    d["credit_yield"] = d.credit / d.margin

    def zs(col):
        return d.groupby("date")[col].transform(
            lambda s: (s.rank(pct=True) - 0.5) * 2 if s.notna().sum() >= 8 else np.nan)

    d["z_cy"] = zs("credit_yield")
    d["z_ivrv"] = zs("ivrv")
    d["z_inv"] = zs("idio_inv")
    d["z_mom"] = zs("mom8")
    d["z_skew"] = zs("skew25")
    d["z_ivr"] = zs("iv_rank")
    d["z_vov"] = zs("vov")
    return d


def signal_correlations(d):
    print("\n--- signal cross-correlation (are they independent bets?) ---")
    C = d[SIGNALS].corr()
    print(C.round(2).to_string())
    off = C.values[np.triu_indices_from(C.values, k=1)]
    print(f"mean |off-diagonal| = {np.nanmean(np.abs(off)):.3f}")
    return C


def walkforward_composite(d, years=(2021, 2022, 2023, 2024), min_train=2000):
    """Fit composite weights on data strictly before each test year."""
    out_parts, ic_rows = [], []
    for y in years:
        cut = pd.Timestamp(f"{y}-01-01")
        tr = d[(d.date < cut)].dropna(subset=SIGNALS + ["ret"])
        te = d[(d.date >= cut) & (d.date < pd.Timestamp(f"{y+1}-01-01"))].dropna(
            subset=SIGNALS + ["ret"])
        if len(tr) < min_train or len(te) < 100:
            continue
        # cross-sectionally demeaned OLS of return on rank-signals
        X = tr[SIGNALS].values
        yv = tr.groupby("date").ret.transform(lambda s: s - s.mean()).values
        w, *_ = np.linalg.lstsq(X, yv, rcond=None)
        te = te.copy()
        te["pred"] = te[SIGNALS].values @ w
        out_parts.append(te)
        # OOS IC of the composite
        ics = []
        for _, g in te.groupby("date"):
            if len(g) >= 10:
                ic = spearmanr(g.pred, g.ret).statistic
                if np.isfinite(ic):
                    ics.append(ic)
        ic_rows.append((y, np.mean(ics) if ics else np.nan, len(ics),
                        dict(zip(SIGNALS, w.round(4)))))
    if not out_parts:
        return None, None
    return pd.concat(out_parts), ic_rows


def sleeve_from_pred(te, q=0.7, overlay=None):
    """Short the names the composite says are richest; equal weight per date."""
    def pick(x):
        if len(x) < 8:
            return np.nan
        cut = x.pred.quantile(q)
        sel = x[x.pred >= cut]
        if not len(sel):
            return np.nan
        r = sel.ret.mean()
        if overlay is not None:
            r *= overlay(x)
        return r
    return te.groupby("date").apply(pick, include_groups=False).dropna()


def main():
    print("=" * 70)
    print("STUDY 8 — composite signal x shorter tenor x tail overlay")
    print("=" * 70)

    for short_dated, tag in [(False, "MONTHLY 15-50 DTE"), (True, "SHORT 8-14 DTE")]:
        for structure in ["credit_putspread", "short_strangle25"]:
            try:
                d = build_panel(structure, short_dated)
            except FileNotFoundError:
                continue
            if len(d) < 3000:
                continue
            print(f"\n{'='*70}\n{tag} — {structure}  (rows={len(d):,}, "
                  f"dates={d.date.nunique():,})")
            dev = d[d.date <= DEV_END]
            if short_dated is False and structure == "credit_putspread":
                signal_correlations(dev)

            te, ics = walkforward_composite(dev)
            if te is None:
                print("  insufficient data for walk-forward")
                continue
            print("  walk-forward OOS IC of composite:")
            for y, ic, n, w in ics:
                top = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:3]
                print(f"    {y}: IC={ic:+.4f} (n={n})  top weights: "
                      + ", ".join(f"{k}={v:+.3f}" for k, v in top))
            allic = np.nanmean([r[1] for r in ics])
            print(f"    POOLED OOS IC = {allic:+.4f}")

            for q in (0.6, 0.7, 0.8):
                s = sleeve_from_pred(te, q=q)
                r = P.screen_sharpe(s)
                print(f"  q={q}: OOS scrSharpe={r.get('sharpe_scr', float('nan')):+.2f} "
                      f"hit={r.get('hit', float('nan')):.2f} n={r.get('n_weeks')}")

            # tail overlay: scale exposure down when SPY is stressed
            def make_overlay(iv_z_max, dd_min):
                def ov(x):
                    z = x.spy_iv_z.iloc[0] if "spy_iv_z" in x else np.nan
                    dd = x.spy_dd.iloc[0] if "spy_dd" in x else np.nan
                    if not np.isfinite(z) or not np.isfinite(dd):
                        return 1.0
                    return 0.0 if (z > iv_z_max or dd < dd_min) else 1.0
                return ov

            for iv_z_max, dd_min in [(1.0, -0.08), (1.5, -0.12), (2.0, -0.20)]:
                s = sleeve_from_pred(te, q=0.7, overlay=make_overlay(iv_z_max, dd_min))
                r = P.screen_sharpe(s)
                print(f"  overlay(ivz<{iv_z_max}, dd>{dd_min:.0%}): "
                      f"OOS scrSharpe={r.get('sharpe_scr', float('nan')):+.2f} "
                      f"hit={r.get('hit', float('nan')):.2f}")


if __name__ == "__main__":
    main()
