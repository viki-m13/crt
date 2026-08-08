#!/usr/bin/env python3
"""Study 4: correlation premium / dispersion (SPY vol vs component vol).

Classic institutional trade: index implied vol is rich relative to the
vol-weighted basket of its components because index options embed a
correlation premium. Trade: SHORT index straddle, LONG component straddles
(vega-matched). Honest version: index legs sold at bid, component legs bought
at ask, all held to the same expiry, settled at parity spots.

Also tests the REVERSE (long index / short components) and a "short-only when
implied correlation is extreme" gate. Implied correlation:
    rho_imp = (iv_idx^2 - sum w_i^2 iv_i^2) / (sum_{i!=j} w_i w_j iv_i iv_j)
computed point-in-time from the chains themselves.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right

import numpy as np
import pandas as pd

import engine as E
from structures import pick, leg_entry, settle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "dispersion.parquet")

# rough S&P weights proxy: mega-caps get more weight. We use equal weight
# within the traded basket (honest: we are not claiming index replication,
# only a vega-matched dispersion book on N liquid names).
BASKET = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
          "JPM", "V", "XOM", "UNH", "JNJ", "WMT", "MA", "PG", "HD", "COST",
          "MRK", "ABBV", "CVX", "CRM", "AMD", "LLY", "PEP", "KO", "ADBE",
          "TMO", "MCD", "CSCO", "ORCL", "ACN", "LIN", "DHR", "TXN", "QCOM"]


def main(min_names=12, dev_end="2026-12-31"):
    dates = E.available_dates()
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    sp_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    hist = {s: sp_panel[s].dropna() for s in sp_panel.columns}

    def settle_spot(sym, exp):
        if sym not in hist:
            return None
        ds = list(hist[sym].index)
        j = bisect_right(ds, exp)
        if j < len(ds) and (pd.Timestamp(ds[j]) - pd.Timestamp(exp)).days <= 4:
            return float(hist[sym].iloc[j])
        if j - 1 >= 0 and abs((pd.Timestamp(ds[j - 1]) - pd.Timestamp(exp)).days) <= 1:
            return float(hist[sym].iloc[j - 1])
        return None

    rows = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        ch = ch[(ch.dte >= 15) & (ch.dte <= 50)]
        if ch.empty or day not in sp_panel.index:
            continue
        idx = ch[ch.act_symbol == "SPY"]
        if idx.empty:
            continue
        s_idx = sp_panel.at[day, "SPY"]
        if not np.isfinite(s_idx):
            continue
        exp = idx.expiration.min()
        gi = idx[idx.expiration == exp]
        ci, pi = pick(gi, "Call", s_idx), pick(gi, "Put", s_idx)
        if ci is None or pi is None:
            continue
        se_idx = settle_spot("SPY", exp)
        if se_idx is None:
            continue
        iv_idx = float(np.nanmean([ci.vol, pi.vol]))
        vega_idx = float(np.nansum([ci.vega, pi.vega])) * 100.0
        if not np.isfinite(iv_idx) or vega_idx <= 0:
            continue

        # index short straddle at bid
        e_ci, e_pi = leg_entry(ci, -1), leg_entry(pi, -1)
        if e_ci is None or e_pi is None:
            continue
        idx_credit = (e_ci + e_pi) * 100.0
        idx_loss = (settle(se_idx, float(ci.strike), "Call")
                    + settle(se_idx, float(pi.strike), "Put")) * 100.0
        idx_pnl_short = idx_credit - idx_loss

        # components: same expiration, long straddle at ask, vega-matched
        comps = []
        for sym in BASKET:
            if sym not in sp_panel.columns:
                continue
            s = sp_panel.at[day, sym]
            if not np.isfinite(s):
                continue
            g = ch[(ch.act_symbol == sym) & (ch.expiration == exp)]
            if g.empty:
                continue
            c, p = pick(g, "Call", s), pick(g, "Put", s)
            if c is None or p is None:
                continue
            a_c, a_p = leg_entry(c, +1), leg_entry(p, +1)
            b_c, b_p = leg_entry(c, -1), leg_entry(p, -1)
            if None in (a_c, a_p, b_c, b_p):
                continue
            se = settle_spot(sym, exp)
            if se is None or abs(math.log(se / s)) > 0.6:
                continue
            iv = float(np.nanmean([c.vol, p.vol]))
            vega = float(np.nansum([c.vega, p.vega])) * 100.0
            if not np.isfinite(iv) or vega <= 0:
                continue
            payoff = (settle(se, float(c.strike), "Call")
                      + settle(se, float(p.strike), "Put")) * 100.0
            rows_c = dict(sym=sym, iv=iv, vega=vega, spot=s,
                          debit=(a_c + a_p) * 100.0, credit=(b_c + b_p) * 100.0,
                          payoff=payoff, notional=s * 100.0)
            comps.append(rows_c)
        if len(comps) < min_names:
            continue

        n = len(comps)
        w = np.ones(n) / n
        ivs = np.array([c["iv"] for c in comps])
        # implied correlation of the equal-weight basket
        var_basket_indep = float(np.sum((w * ivs) ** 2))
        cross = float((np.sum(w * ivs)) ** 2 - var_basket_indep)
        rho_imp = (iv_idx ** 2 - var_basket_indep) / cross if cross > 1e-12 else np.nan
        basket_iv = float(np.sum(w * ivs))

        # vega-match: scale component book so its vega = index vega
        vega_comp_unit = float(np.sum([w[i] * comps[i]["vega"] for i in range(n)]))
        if vega_comp_unit <= 0:
            continue
        scale = vega_idx / vega_comp_unit

        long_comp_pnl = float(np.sum([
            w[i] * scale * (comps[i]["payoff"] - comps[i]["debit"]) for i in range(n)]))
        short_comp_pnl = float(np.sum([
            w[i] * scale * (comps[i]["credit"] - comps[i]["payoff"]) for i in range(n)]))
        comp_margin = float(np.sum([w[i] * scale * 0.30 * comps[i]["notional"]
                                    for i in range(n)]))
        idx_margin = 0.30 * s_idx * 100.0

        rows.append((day, exp, int(gi.dte.iloc[0]), iv_idx, basket_iv, rho_imp, n,
                     idx_pnl_short, long_comp_pnl, short_comp_pnl,
                     idx_margin, comp_margin, idx_credit))
        if (di + 1) % 200 == 0:
            print(f"dispersion {di+1}/{len(dates)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=["date", "expiration", "dte", "iv_idx",
                                     "basket_iv", "rho_imp", "n_names",
                                     "idx_short_pnl", "comp_long_pnl",
                                     "comp_short_pnl", "idx_margin",
                                     "comp_margin", "idx_credit"])
    df.to_parquet(OUT, index=False)
    print("dispersion rows:", df.shape)
    if not len(df):
        return

    import portfolio as P
    df["date"] = pd.to_datetime(df.date)
    df["margin"] = df.idx_margin + df.comp_margin
    # classic dispersion: short index vol, long component vol
    df["disp_pnl"] = df.idx_short_pnl + df.comp_long_pnl
    df["rev_pnl"] = -df.idx_short_pnl + df.comp_short_pnl
    df["idx_only"] = df.idx_short_pnl
    df["ret_disp"] = df.disp_pnl / df.margin
    df["ret_rev"] = df.rev_pnl / df.margin
    df["ret_idx"] = df.idx_only / df.idx_margin

    print("\nimplied correlation:", df.rho_imp.describe([.1, .5, .9]).round(3).to_dict())
    for nm, col in [("dispersion (short idx / long comps)", "ret_disp"),
                    ("reverse (long idx / short comps)", "ret_rev"),
                    ("index short only", "ret_idx")]:
        s = df.groupby("date")[col].mean()
        print(f"{nm:>40}: {P.screen_sharpe(s)}")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 2.0)]:
        g = df[(df.rho_imp >= lo) & (df.rho_imp < hi)]
        if len(g) < 40:
            continue
        s = g.groupby("date").ret_disp.mean()
        s2 = g.groupby("date").ret_rev.mean()
        print(f"  rho in [{lo},{hi}) n={len(g)}: disp={P.screen_sharpe(s).get('sharpe_scr'):+.2f} "
              f"rev={P.screen_sharpe(s2).get('sharpe_scr'):+.2f}")


if __name__ == "__main__":
    main()
