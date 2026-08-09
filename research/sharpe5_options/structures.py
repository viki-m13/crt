#!/usr/bin/env python3
"""Precompute honest returns of canonical option structures per (date, symbol).

Every structure: entered at WORST side of EOD spread at date t, held to front
expiry, settled at intrinsic vs parity spot at the expiry observation. Also
records the structure's defined-risk margin so returns can be computed on
committed capital. Delta-hedged variants re-hedge with stock at each
observation date (2 bps stock cost) using Black-Scholes delta at the DB IV.

Output: cache/structures.parquet with one row per (date, symbol, structure).
"""
from __future__ import annotations

import glob
import math
import os
from bisect import bisect_right

import numpy as np
import pandas as pd
from scipy.stats import norm

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PQ = os.path.join(HERE, "cache", "structures.parquet")
SHARD_DIR = os.path.join(HERE, "cache", "struct_shards")
STOCK_BPS = 2.0

COLUMNS = ["date", "act_symbol", "structure", "expiration", "dte", "pnl",
           "margin", "credit", "iv", "spot", "spot_exp"]


def _flush(rows, upto):
    """Checkpoint a shard so a killed run resumes instead of losing hours."""
    if not rows:
        return
    os.makedirs(SHARD_DIR, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(
        os.path.join(SHARD_DIR, f"{upto:06d}.parquet"), index=False)


def bs_delta(S, K, T, sigma, cp):
    if T <= 0 or sigma <= 0:
        intr = (S > K) if cp == "Call" else (S < K)
        return (1.0 if intr else 0.0) if cp == "Call" else (-1.0 if intr else 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if cp == "Call" else norm.cdf(d1) - 1.0


def pick(ge: pd.DataFrame, cp: str, target_strike: float):
    g = ge[(ge.call_put == cp) & (ge.bid > 0)]
    if g.empty:
        return None
    i = (g.strike - target_strike).abs().idxmin()
    return g.loc[i]


def leg_entry(row, side: int) -> float | None:
    """side +1 buy at ask, -1 sell at bid."""
    if side > 0:
        return float(row.ask) if row.ask > 0 else None
    return float(row.bid) if row.bid > 0 else None


def settle(spot_exp: float, K: float, cp: str) -> float:
    return max(spot_exp - K, 0.0) if cp == "Call" else max(K - spot_exp, 0.0)


def main():
    dates = E.available_dates()
    date_ts = [pd.Timestamp(d) for d in dates]
    spots_df = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    spot_panel = spots_df.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    # map: symbol -> (sorted obs dates with spot, values)
    sym_hist = {}
    for sym in spot_panel.columns:
        s = spot_panel[sym].dropna()
        sym_hist[sym] = (list(s.index), s.values)

    def spot_at_or_after(sym: str, day: str, grace_days: int = 4):
        ds, vs = sym_hist[sym]
        i = bisect_right(ds, day) - 1  # last obs <= day... we want expiry settle
        # find first obs >= day within grace; else last obs before
        j = bisect_right(ds, day)
        if j < len(ds) and (pd.Timestamp(ds[j]) - pd.Timestamp(day)).days <= grace_days:
            return vs[j], ds[j]
        if i >= 0 and (pd.Timestamp(day) - pd.Timestamp(ds[i])).days <= 1:
            return vs[i], ds[i]
        return None, None

    def hedge_path(sym: str, t0: str, texp: str, K: float, sigma: float, cp: str,
                   sign: int):
        """P&L of delta-hedging ONE short/long option (sign=+1 long option)
        with stock, re-hedged at each obs in (t0, texp]. Returns stock P&L."""
        ds, vs = sym_hist[sym]
        i0 = bisect_right(ds, t0) - 1
        if i0 < 0 or ds[i0] != t0:
            return None
        pnl = 0.0
        sh = 0.0
        texp_ts = pd.Timestamp(texp)
        prev_s = vs[i0]
        for j in range(i0, len(ds)):
            d, s = ds[j], vs[j]
            if d > texp:
                break
            if abs(math.log(s / prev_s)) > 0.22 and j > i0:
                return None  # split/discontinuity guard — skip this structure
            prev_s = s
            T = max((texp_ts - pd.Timestamp(d)).days, 0) / 365.0
            delta = bs_delta(s, K, T, sigma, cp)
            tgt = -sign * delta * 100.0  # hedge one contract
            dlt = tgt - sh
            pnl -= dlt * s          # buy/sell stock at parity spot
            pnl -= abs(dlt) * s * STOCK_BPS / 1e4
            sh = dlt + sh
        # close stock at expiry spot
        se, _ = spot_at_or_after(sym, texp)
        if se is None:
            return None
        pnl += sh * se
        pnl -= abs(sh) * se * STOCK_BPS / 1e4
        return pnl

    # resume: skip whole shards already on disk
    done_upto = 0
    if os.path.isdir(SHARD_DIR):
        have = sorted(int(os.path.basename(p)[:-8])
                      for p in glob.glob(os.path.join(SHARD_DIR, "*.parquet")))
        # only trust a contiguous run of 50-date shards
        for k, v in enumerate(have, start=1):
            if v == k * 50:
                done_upto = v
            else:
                break
    if done_upto:
        print(f"resuming after {done_upto} dates", flush=True)

    rows = []
    for di, day in enumerate(dates):
        if di < done_upto:
            continue
        ch = E.load_chain(day)
        ch = ch[ch.dte >= 15]
        for sym, g in ch.groupby("act_symbol"):
            s_now = spot_panel.at[day, sym] if (day in spot_panel.index and sym in spot_panel.columns) else np.nan
            if not np.isfinite(s_now):
                continue
            front = g[(g.dte >= 15) & (g.dte <= 50)]
            if front.empty:
                continue
            exp1 = front.expiration.min()
            ge = g[g.expiration == exp1]
            dte1 = int(ge.dte.iloc[0])
            se, sed = spot_at_or_after(sym, exp1)
            if se is None:
                continue
            # split guard between entry and settle
            if abs(math.log(se / s_now)) > 0.60:
                continue

            cA = pick(ge, "Call", s_now)
            pA = pick(ge, "Put", s_now)
            if cA is None or pA is None:
                continue
            ivm = float(np.nanmean([cA.vol, pA.vol]))

            # ---- 1) short ATM straddle (+ delta-hedged variant)
            eb_c, eb_p = leg_entry(cA, -1), leg_entry(pA, -1)
            if eb_c and eb_p:
                credit = (eb_c + eb_p) * 100.0
                loss_at_exp = (settle(se, float(cA.strike), "Call")
                               + settle(se, float(pA.strike), "Put")) * 100.0
                margin = 0.30 * s_now * 100.0  # reserve 30% notional
                pnl = credit - loss_at_exp
                rows.append((day, sym, "short_straddle", exp1, dte1, pnl, margin,
                             credit, ivm, s_now, se))
                hp_c = hedge_path(sym, day, exp1, float(cA.strike), max(ivm, 0.05), "Call", -1)
                hp_p = hedge_path(sym, day, exp1, float(pA.strike), max(ivm, 0.05), "Put", -1)
                if hp_c is not None and hp_p is not None:
                    rows.append((day, sym, "short_straddle_dh", exp1, dte1,
                                 pnl + hp_c + hp_p, margin, credit, ivm, s_now, se))

            # ---- 2) long ATM straddle (+ delta-hedged)
            ea_c, ea_p = leg_entry(cA, +1), leg_entry(pA, +1)
            if ea_c and ea_p:
                debit = (ea_c + ea_p) * 100.0
                pay = (settle(se, float(cA.strike), "Call")
                       + settle(se, float(pA.strike), "Put")) * 100.0
                pnl = pay - debit
                rows.append((day, sym, "long_straddle", exp1, dte1, pnl, debit,
                             -debit, ivm, s_now, se))
                hp_c = hedge_path(sym, day, exp1, float(cA.strike), max(ivm, 0.05), "Call", +1)
                hp_p = hedge_path(sym, day, exp1, float(pA.strike), max(ivm, 0.05), "Put", +1)
                if hp_c is not None and hp_p is not None:
                    rows.append((day, sym, "long_straddle_dh", exp1, dte1,
                                 pnl + hp_c + hp_p, debit, -debit, ivm, s_now, se))

            # ---- 3) iron condor: short 3% strangle, long 8% wings
            ps, cs = pick(ge, "Put", s_now * 0.97), pick(ge, "Call", s_now * 1.03)
            pw, cw = pick(ge, "Put", s_now * 0.92), pick(ge, "Call", s_now * 1.08)
            if all(x is not None for x in (ps, cs, pw, cw)) and \
               float(pw.strike) < float(ps.strike) and float(cw.strike) > float(cs.strike):
                e1, e2 = leg_entry(ps, -1), leg_entry(cs, -1)
                e3, e4 = leg_entry(pw, +1), leg_entry(cw, +1)
                if all(v is not None for v in (e1, e2, e3, e4)):
                    credit = (e1 + e2 - e3 - e4) * 100.0
                    if credit > 0:
                        loss = ((settle(se, float(ps.strike), "Put") - settle(se, float(pw.strike), "Put"))
                                + (settle(se, float(cs.strike), "Call") - settle(se, float(cw.strike), "Call"))) * 100.0
                        width = max(float(ps.strike) - float(pw.strike),
                                    float(cw.strike) - float(cs.strike)) * 100.0
                        pnl = credit - loss
                        rows.append((day, sym, "iron_condor", exp1, dte1, pnl,
                                     width, credit, ivm, s_now, se))

            # ---- 4) 25-delta short strangle (defined by 8% wings → iron-ish)
            g25p = ge[(ge.call_put == "Put") & (ge.delta.abs().between(0.18, 0.32)) & (ge.bid > 0)]
            g25c = ge[(ge.call_put == "Call") & (ge.delta.between(0.18, 0.32)) & (ge.bid > 0)]
            if len(g25p) and len(g25c):
                p25 = g25p.loc[(g25p.delta.abs() - 0.25).abs().idxmin()]
                c25 = g25c.loc[(g25c.delta - 0.25).abs().idxmin()]
                e1, e2 = leg_entry(p25, -1), leg_entry(c25, -1)
                if e1 and e2:
                    credit = (e1 + e2) * 100.0
                    loss = (settle(se, float(p25.strike), "Put")
                            + settle(se, float(c25.strike), "Call")) * 100.0
                    margin = 0.25 * s_now * 100.0
                    rows.append((day, sym, "short_strangle25", exp1, dte1,
                                 credit - loss, margin, credit, ivm, s_now, se))

            # ---- 4b) OTM credit spreads: sell 4% OTM, buy 9% OTM (both sides)
            for side, cp, k_s, k_w in (("put", "Put", 0.96, 0.91), ("call", "Call", 1.04, 1.09)):
                sl, wl = pick(ge, cp, s_now * k_s), pick(ge, cp, s_now * k_w)
                if sl is None or wl is None or float(sl.strike) == float(wl.strike):
                    continue
                if cp == "Put" and not float(wl.strike) < float(sl.strike):
                    continue
                if cp == "Call" and not float(wl.strike) > float(sl.strike):
                    continue
                e1, e2 = leg_entry(sl, -1), leg_entry(wl, +1)
                if e1 is None or e2 is None:
                    continue
                credit = (e1 - e2) * 100.0
                if credit <= 0:
                    continue
                loss = (settle(se, float(sl.strike), cp) - settle(se, float(wl.strike), cp)) * 100.0
                width = abs(float(sl.strike) - float(wl.strike)) * 100.0
                rows.append((day, sym, f"credit_{side}spread", exp1, dte1,
                             credit - loss, width, credit, ivm, s_now, se))

            # NOTE: the calendar structure (sell front / buy back straddle) was
            # dropped after Trial-set 18-71 showed it dead (screen -1.9..-0.3:
            # two entry spreads plus an exit spread on the back leg). It also
            # required loading a second observation date's chain inside this
            # loop, which thrashed the chain cache and made the pass
            # ~100x slower. Both reasons are documented in RESEARCH_LOG.md.
        if (di + 1) % 50 == 0:
            _flush(rows, di + 1)
            rows = []
            print(f"structures {di+1}/{len(dates)}", flush=True)

    _flush(rows, len(dates))
    parts = sorted(glob.glob(os.path.join(SHARD_DIR, "*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(OUT_PQ, index=False)
    print("saved", df.shape)


if __name__ == "__main__":
    main()
