#!/usr/bin/env python3
"""Study 30: SELECTIVITY on the high-c/w spread — targeting CAGR>50%, DD<10%.

The user's target is Calmar > 5 sustained. Study 29 established the base
instrument (SPY 5%-wide, 45-75 DTE put credit spreads, short strike ITM) is
levered equity beta with a fair price. This study tests whether TIMING
selectivity + EXIT control can reshape the path enough to approach the target,
with three upgrades over study 29:

  1. TRUE MARK-TO-MARKET: a single pass over every chain date records the
     worst-side liquidation cost (buy short back at ask, sell long at bid) of
     every open cohort on every date. Drawdowns are measured on MTM equity,
     not settlement marks. This is STRICTER than study 29.
  2. PRE-REGISTERED GATES: 12 entry filters computed causally from features
     known at entry. Ranked on dev (<=2024-12-31) ONLY; holdout reported for
     the top 3 by dev Sharpe — the rest's holdout is not consulted.
  3. EXIT RULES on real unwind quotes: loss-cap, spot-trigger, profit-take.

Multiple-testing: every configuration evaluated is counted and logged.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E
import study29_highcw as S29

HERE = os.path.dirname(os.path.abspath(__file__))
MTM_PATH = os.path.join(HERE, "results", "study30_mtm.parquet")
DEV_END = pd.Timestamp("2024-12-31")


# ---------------------------------------------------------------- cohorts+MTM
def cohorts_with_mtm(off=0.05, width=0.05, dte_lo=45, dte_hi=75):
    panel = S29.load_panel()
    d = S29.build(off, width, dte_lo, dte_hi, panel).copy()
    d = d.reset_index(drop=True)
    d["cid"] = d.index
    if os.path.exists(MTM_PATH):
        mtm = pd.read_parquet(MTM_PATH)
        if len(mtm) and mtm.cid.max() == d.cid.max():
            return d, mtm, panel
    # single pass: for each chain date, liquidation cost of every open cohort.
    # DATA CONSTRAINT (verified): the DoltHub scrape stores only ~3 rolling
    # tenor buckets per date, and later scrapes of an expiration carry a
    # COARSE $10-ish strike grid whose offset shifts — exact-strike lookups
    # fail ~94% of the time. Marks therefore use LINEAR INTERPOLATION in
    # strike across the coarse grid. Put price is convex in strike, so the
    # chord OVERSTATES both legs' prices: conservative (an upper bound) for
    # the short-leg buy-back at ask, anti-conservative by only the convexity
    # gap for the long leg sold at bid — a $0.25 penalty is added whenever
    # either leg is interpolated, which more than covers the ~$0.13-0.25
    # curvature gap of a $10 grid near ATM. Marks exist only on dates whose
    # scrape carries the cohort's expiration (~9 per cohort, clustered near
    # the 28d/14d buckets): exit rules act with up to ~3 weeks of latency,
    # and MTM drawdowns remain a LOWER bound on the true path.
    def interp(grid, k, col):
        ex = grid[grid.strike == k]
        if len(ex):
            return float(ex[col].iloc[0]), True
        lo = grid[grid.strike < k]
        hi = grid[grid.strike > k]
        if lo.empty or hi.empty:
            return None, False
        k1, v1 = float(lo.strike.iloc[-1]), float(lo[col].iloc[-1])
        k2, v2 = float(hi.strike.iloc[0]), float(hi[col].iloc[0])
        return v1 + (v2 - v1) * (k - k1) / (k2 - k1), False

    by_exp: dict = {}
    for r in d.itertuples(index=False):
        by_exp.setdefault(r.exp, []).append((r.cid, r.date, r.ks, r.kl))
    rows = []
    for day in E.available_dates():
        ts = pd.Timestamp(day)
        ch = None
        for exp, lst in by_exp.items():
            if not (min(t for _, t, _, _ in lst) < ts <= pd.Timestamp(exp)):
                continue
            if ch is None:
                ch = E.load_chain(day)
                g_all = ch[(ch.act_symbol == "SPY") & (ch.call_put == "Put")]
            ge = g_all[(g_all.expiration == exp) & (g_all.ask > 0)
                       ].sort_values("strike")
            if len(ge) < 3:
                continue
            for cid, t0, ks, kl in lst:
                if not (t0 < ts):
                    continue
                a_s, ex1 = interp(ge, ks, "ask")
                b_l, ex2 = interp(ge, kl, "bid")
                if a_s is None or b_l is None:
                    continue
                pen = 0.0 if (ex1 and ex2) else 0.25
                rows.append((cid, day, a_s - max(b_l, 0.0) + pen))
    mtm = pd.DataFrame(rows, columns=["cid", "date", "liq"])
    mtm["date"] = pd.to_datetime(mtm.date)
    os.makedirs(os.path.dirname(MTM_PATH), exist_ok=True)
    mtm.to_parquet(MTM_PATH)
    return d, mtm, panel


# ---------------------------------------------------------------- gate features
def gate_table(d, panel):
    """All features computed CAUSALLY from data available at entry."""
    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    spy = f[f.act_symbol == "SPY"].set_index("date").sort_index()
    s = panel["SPY"].dropna()
    obs_yr = len(s) / max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    n200 = max(10, int(round(obs_yr * 200 / 252)))
    n50 = max(5, int(round(obs_yr * 50 / 252)))
    n21 = max(3, int(round(obs_yr * 21 / 252)))
    sma200 = s.rolling(n200, min_periods=n200 // 2).mean()
    sma50 = s.rolling(n50, min_periods=n50 // 2).mean()
    mom21 = s / s.shift(n21) - 1
    hi52 = s.rolling(int(obs_yr), min_periods=int(obs_yr) // 2).max()
    ivrv = (spy.iv_front / spy.rv_ewma.clip(lower=0.03)).reindex(s.index)
    term = (spy.iv_front - spy.iv_back).reindex(s.index)
    vov = spy.iv_front.rolling(24, min_periods=8).std().reindex(s.index)
    rv = spy.rv_ewma.reindex(s.index)
    skew = spy.skew25.reindex(s.index)

    g = pd.DataFrame(index=d.index)
    dt = d.date
    cr = d.credit / d.risk
    cr_med = cr.expanding(min_periods=30).median().shift(1)
    g["G1_trend200"] = (dt.map(s) >= dt.map(sma200)).values
    g["G2_trend50"] = (dt.map(s) >= dt.map(sma50)).values
    g["G3_mom21_pos"] = (dt.map(mom21) > 0).values
    g["G4_near_high"] = (dt.map(s) / dt.map(hi52) - 1 > -0.03).values
    g["G5_ivrv_rich"] = (dt.map(ivrv) > 1.0).values
    g["G6_not_panic"] = (dt.map(ivrv) < 1.3).values
    g["G7_term_calm"] = (dt.map(term) <= 0).values
    g["G8_vov_low"] = (dt.map(vov) <= vov.expanding(30).median().reindex(s.index)
                       .shift(1).reindex(dt).values).fillna(False).values
    g["G9_rv_low"] = (dt.map(rv) < 0.15).values
    g["G10_cr_rich"] = (cr >= cr_med).fillna(False).values
    g["G11_trend_and_calm"] = g.G1_trend200 & g.G7_term_calm
    g["G12_skew_low"] = (dt.map(skew) <= skew.expanding(30).median()
                         .reindex(s.index).shift(1).reindex(dt).values
                         ).fillna(False).values
    return g.fillna(False)


# ---------------------------------------------------------------- MTM ladder
def ladder_mtm(d, mtm, gate=None, rung=0.05, cap=0.60, exit_rule=None,
               cadence="W"):
    """Weekly entries; MTM equity marked on every chain date; exits at real
    worst-side unwind quotes. exit_rule(entry, liq, spot_now) -> bool.
    Rung P&L per unit risk-capital: (credit - cost)/risk, cost = liq at exit
    or settlement loss at expiry."""
    d = d.sort_values("date").copy()
    d["gate"] = True if gate is None else np.asarray(gate, dtype=bool)
    d["per"] = d.date.dt.to_period(cadence)
    entries = d.groupby("per", as_index=False).first()
    ent_by_date = {r.date: r for r in entries.itertuples(index=False)}
    liq = {(int(r.cid), r.date): r.liq for r in mtm.itertuples(index=False)}
    all_dates = sorted(set(mtm.date) | set(d.date))
    fin = d.set_index("cid")

    realized, open_r, curve = 1.0, [], []
    last_liq: dict = {}
    for ts in all_dates:
        still = []
        for o in open_r:                       # settle expiries first
            row = fin.loc[o["cid"]]
            if pd.Timestamp(row.exp) <= ts:
                realized += o["cap"] * float(row.ret_risk)
            else:
                still.append(o)
        open_r = still
        if exit_rule is not None:              # exits at real unwind quotes
            still = []
            for o in open_r:
                q = liq.get((o["cid"], ts))
                if q is not None:
                    last_liq[o["cid"]] = q
                q = last_liq.get(o["cid"])
                if q is not None and exit_rule(o, q):
                    realized += o["cap"] * (o["credit"] - q) / o["risk"]
                else:
                    still.append(o)
            open_r = still
        # MTM equity
        eq = realized
        for o in open_r:
            q = liq.get((o["cid"], ts), last_liq.get(o["cid"], o["credit"]))
            if (o["cid"], ts) in liq:
                last_liq[o["cid"]] = liq[(o["cid"], ts)]
            eq += o["cap"] * (o["credit"] - q) / o["risk"]
        # entries
        r = ent_by_date.get(ts)
        if r is not None and r.gate and eq > 0:
            at_risk = sum(o["cap"] for o in open_r)
            want = rung * eq
            if at_risk + want <= cap * eq:
                open_r.append(dict(cid=int(r.cid), cap=want,
                                   credit=float(r.credit), risk=float(r.risk),
                                   spot0=float(r.spot)))
        curve.append((ts, eq))
    for o in open_r:
        realized += o["cap"] * float(fin.loc[o["cid"]].ret_risk)
    eq = pd.Series(dict(curve)).sort_index()
    if len(eq) < 30 or (eq <= 0).any():
        return {"ruin": True, "eq": eq}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    w = eq.resample("W-FRI").last().dropna()
    rr = w.pct_change().dropna()
    return {"cagr": float(eq.iloc[-1]) ** (1 / max(yrs, 1e-9)) - 1,
            "sharpe": rr.mean() / rr.std() * math.sqrt(52) if rr.std() > 0 else np.nan,
            "dd": float((eq / eq.cummax() - 1).min()),
            "eq": eq, "ruin": False}


def show(r, label):
    if r is None:
        print(f"  {label:<40} insufficient")
        return
    if r.get("ruin"):
        print(f"  {label:<40} RUIN")
        return
    calmar = r["cagr"] / abs(r["dd"]) if r["dd"] < 0 else np.nan
    print(f"  {label:<40} CAGR={r['cagr']:+7.1%} Sharpe={r['sharpe']:+5.2f} "
          f"maxDD={r['dd']:+7.1%} Calmar={calmar:4.1f}")


# ---------------------------------------------------------------- exit rules
def loss_cap(frac):
    def rule(o, q):
        return (o["credit"] - q) / o["risk"] <= -frac
    return rule


def profit_take(frac):
    def rule(o, q):
        return q <= (1 - frac) * o["credit"]
    return rule


def combo(*rules):
    def rule(o, q):
        return any(f(o, q) for f in rules)
    return rule


# ---------------------------------------------------------------- main
def main():
    n_cfg = 0
    d, mtm, panel = cohorts_with_mtm()
    print("=" * 96)
    print("STUDY 30 — selectivity + MTM exits on ITM-5% spreads (target: CAGR>50%, DD<10%)")
    print("=" * 96)
    print(f"cohorts={len(d)}  mtm marks={len(mtm):,}")

    dev_m = (d.date <= DEV_END).values
    g = gate_table(d, panel)

    print("\n--- 1. MTM baseline (no gate, no exit) vs study 29's settlement marks ---")
    for rung in (0.02, 0.05):
        r = ladder_mtm(d, mtm, rung=rung)
        n_cfg += 1
        show(r, f"MTM baseline rung={rung:.0%}")

    print("\n--- 2. gates ranked on DEV ONLY (rung=5%; holdout consulted for top 3 only) ---")
    dev_res = {}
    dd_dev = d[dev_m].copy()
    mtm_dev = mtm[mtm.date <= DEV_END]
    for name in g.columns:
        r = ladder_mtm(dd_dev, mtm_dev, gate=g[name].values[dev_m], rung=0.05)
        n_cfg += 1
        dev_res[name] = r
        cov = g[name].values[dev_m].mean()
        if r and not r.get("ruin"):
            show(r, f"{name} (on {cov:.0%} of weeks)")
        else:
            print(f"  {name:<40} RUIN/insufficient")
    ranked = sorted([k for k, v in dev_res.items() if v and not v.get("ruin")],
                    key=lambda k: -(dev_res[k]["sharpe"] or -9))
    top3 = ranked[:3]
    print(f"\n  top-3 by dev Sharpe: {top3}")
    print("  --- holdout (2025+) for top 3 ---")
    dd_h = d[~dev_m].copy()
    mtm_h = mtm[mtm.date > DEV_END]
    for name in top3:
        r = ladder_mtm(dd_h, mtm_h, gate=g[name].values[~dev_m], rung=0.05)
        n_cfg += 1
        show(r, f"{name} holdout")

    print("\n--- 3. exit rules, no gate, rung=5% (dev / holdout) ---")
    exits = {"hold": None,
             "cut@-50% of risk": loss_cap(0.50),
             "cut@-25% of risk": loss_cap(0.25),
             "take@80% credit": profit_take(0.80),
             "cut50 + take80": combo(loss_cap(0.50), profit_take(0.80))}
    for lab, ex in exits.items():
        rd = ladder_mtm(dd_dev, mtm_dev, rung=0.05, exit_rule=ex)
        rh = ladder_mtm(dd_h, mtm_h, rung=0.05, exit_rule=ex)
        n_cfg += 2
        show(rd, f"{lab} dev")
        show(rh, f"{lab} holdout")

    print("\n--- 4. best gate x best exit x rung frontier (full sample, honest MTM) ---")
    best_gate = top3[0]
    for ex_lab, ex in (("hold", None), ("cut50", loss_cap(0.50)),
                       ("cut50+take80", combo(loss_cap(0.50), profit_take(0.80)))):
        for rung in (0.02, 0.05, 0.10):
            r = ladder_mtm(d, mtm, gate=g[best_gate].values, rung=rung,
                           exit_rule=ex)
            n_cfg += 1
            show(r, f"{best_gate} {ex_lab} rung={rung:.0%}")

    print(f"\nconfigs evaluated this study: {n_cfg} (add to project trial count)")
    print("Pre-stated bar: CAGR>50% AND maxDD<10% (Calmar>5) on MTM equity,")
    print("dev AND holdout. Anything less is reported as what it is.")


if __name__ == "__main__":
    main()
