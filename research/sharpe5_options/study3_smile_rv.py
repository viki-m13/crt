#!/usr/bin/env python3
"""Study 3: intra-chain smile relative value (dev screening).

Per (date, sym, expiration): fit quadratic IV smile in log-moneyness over OTM
quotes (vega-weighted, mid IVs, sane spreads). Residual r_i = iv_i - fit(x_i).
Candidate vertical: SELL quote with r_short >= RICH, BUY nearby same-type quote
with r_long <= r_short - GAP, strikes within 12% of each other. Entry at worst
side (sell bid / buy ask). Defined risk = width. Hold to expiry, settle at
parity spot. Screens: per-entry-date mean return on margin; residual
persistence audit (is the "mispricing" still there next obs? if yes → real
feature/stale quote, not noise).
"""
from __future__ import annotations

import os
import sys
from bisect import bisect_right

import numpy as np
import pandas as pd

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
RICH = 0.025      # short leg must be >= 2.5 vol pts rich
GAP = 0.02        # short-long residual gap
MAX_SPREAD_REL = 0.35
MIN_PTS = 6
EDGE_MIN = 8.0    # $ per spread net estimated edge after worst-side entry


def fit_smile(g: pd.DataFrame, s: float):
    """g: one (sym, exp) chain slice. Returns df with resid, fair columns."""
    g = g[(g.bid > 0) & (g.ask > 0) & (g.vol > 0.02) & (g.vol < 4.0)].copy()
    g["relspread"] = g.spread / g.mid.clip(lower=0.01)
    g = g[g.relspread <= MAX_SPREAD_REL]
    # OTM only: puts below spot, calls above (plus one ITM step tolerated)
    otm = g[((g.call_put == "Put") & (g.strike <= s * 1.02))
            | ((g.call_put == "Call") & (g.strike >= s * 0.98))]
    if len(otm) < MIN_PTS or otm.strike.nunique() < 5:
        return None
    x = np.log(otm.strike.values.astype(float) / s)
    y = otm.vol.values.astype(float)
    w = np.maximum(otm.vega.values.astype(float), 1e-3)
    try:
        coef = np.polyfit(x, y, 2, w=np.sqrt(w))
    except Exception:
        return None
    fit = np.polyval(coef, x)
    res = otm.copy()
    res["fair_iv"] = fit
    res["resid"] = y - fit
    # robust second pass: drop worst outlier, refit (protects fit from the
    # very quote we want to flag)
    if len(res) >= MIN_PTS + 2:
        keep = res.resid.abs().rank() <= len(res) - 2
        x2, y2, w2 = x[keep.values], y[keep.values], w[keep.values]
        try:
            coef2 = np.polyfit(x2, y2, 2, w=np.sqrt(w2))
            res["fair_iv"] = np.polyval(coef2, np.log(res.strike.values.astype(float) / s))
            res["resid"] = res.vol - res.fair_iv
        except Exception:
            pass
    return res


def main(dev_end="2024-12-31"):
    dates = [d for d in E.available_dates() if d <= dev_end]
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    spot_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    sym_hist = {sym: spot_panel[sym].dropna() for sym in spot_panel.columns}

    trades = []
    resid_track = []   # for persistence audit
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        ch = ch[(ch.dte >= 10) & (ch.dte <= 70)]
        for (sym, exp), g in ch.groupby(["act_symbol", "expiration"]):
            if sym not in spot_panel.columns or day not in spot_panel.index:
                continue
            s = spot_panel.at[day, sym]
            if not np.isfinite(s):
                continue
            res = fit_smile(g, s)
            if res is None:
                continue
            rich = res[res.resid >= RICH]
            if rich.empty:
                continue
            # settlement spot
            hs = sym_hist[sym]
            ds = list(hs.index)
            j = bisect_right(ds, exp)
            se = None
            if j < len(ds) and (pd.Timestamp(ds[j]) - pd.Timestamp(exp)).days <= 4:
                se = float(hs.iloc[j])
            elif j - 1 >= 0 and abs((pd.Timestamp(ds[j - 1]) - pd.Timestamp(exp)).days) <= 1:
                se = float(hs.iloc[j - 1])
            if se is None or abs(np.log(se / s)) > 0.60:
                continue
            for _, r_s in rich.iterrows():
                cand = res[(res.call_put == r_s.call_put)
                           & (res.resid <= r_s.resid - GAP)
                           & (res.strike != r_s.strike)
                           & ((res.strike / r_s.strike - 1).abs() <= 0.12)]
                if cand.empty:
                    continue
                r_l = cand.loc[(cand.strike - r_s.strike).abs().idxmin()]
                # worst-side entry
                credit = (r_s.bid - r_l.ask) * 100.0
                width = abs(float(r_s.strike) - float(r_l.strike)) * 100.0
                if width <= 0:
                    continue
                # estimated $ edge: resid gap x vega (per share -> x100)
                edge = (r_s.resid * max(r_s.vega, 1e-3)
                        - r_l.resid * max(r_l.vega, 1e-3)) * 100.0
                if edge < EDGE_MIN:
                    continue
                cp = r_s.call_put
                intr_s = max((se - r_s.strike) if cp == "Call" else (r_s.strike - se), 0.0)
                intr_l = max((se - r_l.strike) if cp == "Call" else (r_l.strike - se), 0.0)
                pnl = credit - (intr_s - intr_l) * 100.0
                # margin: worst case loss of short vertical = width - credit
                # (if credit>0); if debit spread, margin = debit
                margin = max(width - credit, -credit if credit < 0 else 1.0, 50.0)
                trades.append((day, sym, exp, cp, float(r_s.strike), float(r_l.strike),
                               float(r_s.resid), float(r_l.resid), credit, width,
                               edge, pnl, margin, int(r_s.dte)))
                resid_track.append((day, sym, exp, cp, float(r_s.strike), float(r_s.resid)))
        if (di + 1) % 100 == 0:
            print(f"smile {di+1}/{len(dates)} trades={len(trades)}", flush=True)

    df = pd.DataFrame(trades, columns=["date", "sym", "exp", "cp", "k_short",
                                       "k_long", "res_s", "res_l", "credit",
                                       "width", "edge", "pnl", "margin", "dte"])
    df.to_parquet(os.path.join(HERE, "cache", "smile_trades.parquet"), index=False)
    print("trades:", df.shape)
    if not len(df):
        return
    df["ret"] = df.pnl / df.margin
    df["date"] = pd.to_datetime(df.date)
    import portfolio as P
    s = df.groupby("date").ret.mean()
    print("screen:", P.screen_sharpe(s))
    print("mean ret:", df.ret.mean().round(4), "hit:", (df.pnl > 0).mean().round(3),
          "trades/day:", round(len(df) / df.date.nunique(), 1))
    # persistence audit
    rt = pd.DataFrame(resid_track, columns=["date", "sym", "exp", "cp", "k", "resid"])
    rt["date"] = pd.to_datetime(rt.date)
    rt = rt.sort_values("date")
    nxt = rt.copy()
    merged = rt.merge(rt.assign(date_prev=rt.date), on=["sym", "exp", "cp", "k"], suffixes=("", "_n"))
    merged = merged[(merged.date_n > merged.date)]
    if len(merged):
        first_next = merged.sort_values("date_n").groupby(
            ["sym", "exp", "cp", "k", "date"]).first().reset_index()
        gap_days = (first_next.date_n - first_next.date).dt.days
        ok = gap_days <= 5
        if ok.sum() > 50:
            import scipy.stats as st
            corr = st.pearsonr(first_next[ok].resid, first_next[ok].resid_n)
            print(f"resid persistence (next obs): r={corr.statistic:.3f} "
                  f"(n={int(ok.sum())}) — high r = stale/structural, low r = noise-reversion")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2024-12-31")
