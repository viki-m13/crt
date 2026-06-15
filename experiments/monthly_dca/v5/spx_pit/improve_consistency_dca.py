"""Improve dual-benchmark DCA consistency: beat BOTH SPY-DCA and QQQ-DCA
in as many rolling windows as possible, honestly.

Objective (user, 2026-06-15): keep CAGR high, maximize the fraction of
rolling monthly-DCA windows (1/3/5/10y, money-weighted) where the strategy
beats BOTH SPY-DCA and QQQ-DCA, and add an OFF-SWITCH that prevents the
deep drawdowns. PIT S&P 500 only.

Anti-overfit discipline (addresses the validation critique):
  * DESIGN period   = 2003-01 .. 2015-12  (levers tuned here)
  * FROZEN HOLDOUT  = 2016-01 .. 2026-12  (evaluated ONCE, never tuned)
  * pre-registered, small lever menu; report the holdout verbatim incl. losses.

Baseline = deployed E2 = 0.5*WIN1 + 0.5*(RC-D + adaptive-breadth), built
from the canonical reproducible sim (improve_sim_v2 / improve_pick_v3),
which reproduces the production stream bit-exactly.

The off-switches are CAUSAL exposure overlays e_t in [0,1] applied to the
strategy's monthly return; exposure at month t uses only SPY features /
realised returns known at month-end t (the t->t+1 return is scaled by e_t).
Parked capital (1-e_t) earns cash (0%) -- the honest "prevent drawdown"
interpretation, and respects the S&P-500-only / no-ETF constraint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from improve_main_strategy import load_inputs  # noqa
from improve_sim_v2 import run_sim_v2  # noqa
from improve_pick_v3 import run_sim_v3  # noqa
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa

DESIGN_END = pd.Timestamp("2015-12-31")
HOLDOUT_START = pd.Timestamp("2016-01-01")


# --------------------------------------------------------------------------- #
#  DCA money-weighted accumulation + dual-benchmark rolling win-rate           #
# --------------------------------------------------------------------------- #
def dca_t(r):
    """Terminal value of contributing $1 at the start of each of len(r)
    months, each contribution then compounding by that month's return."""
    v = 0.0
    for x in r:
        v = (v + 1.0) * (1.0 + x)
    return v


def bench_aligned(dates, mr, col):
    out = []
    for d in dates:
        p = mr.index.searchsorted(d)
        out.append(float(mr[col].iloc[min(p + 1, len(mr) - 1)]))
    return np.array(out)


def dual_dca(r, spv, qqv, dates, lo=None, hi=None):
    """Rolling-window money-weighted win-rate vs SPY-DCA and QQQ-DCA.
    A window 'wins' only if it beats BOTH. Restrict window START to [lo,hi]."""
    r = np.asarray(r, float)
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    n = len(r)
    o = {}
    for H in (12, 36, 60, 120):
        wb, ws, wq, rs, rq = [], [], [], [], []
        for s in range(0, n - H + 1):
            if lo is not None and d[s] < np.datetime64(lo):
                continue
            if hi is not None and d[s] > np.datetime64(hi):
                continue
            a, b, c = dca_t(r[s:s + H]), dca_t(spv[s:s + H]), dca_t(qqv[s:s + H])
            ws.append(a > b)
            wq.append(a > c)
            wb.append(a > b and a > c)
            rs.append(a / b if b > 0 else np.nan)
            rq.append(a / c if c > 0 else np.nan)
        o[H] = dict(
            n=len(wb),
            win_spy=float(np.mean(ws)) if ws else None,
            win_qqq=float(np.mean(wq)) if wq else None,
            win_both=float(np.mean(wb)) if wb else None,
            worst_vs_spy=float(np.nanmin(rs)) if rs else None,
            worst_vs_qqq=float(np.nanmin(rq)) if rq else None,
        )
    return o


def lump_metrics(r, dates):
    r = np.asarray(r, float)
    n = len(r)
    d = pd.to_datetime(pd.Series(dates))
    yrs = (d.iloc[-1] - d.iloc[0]).days / 365.25
    e = np.cumprod(1 + r)
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    sh = r.mean() / max(r.std(), 1e-9) * np.sqrt(12)
    return dict(cagr=float(cagr), sharpe=float(sh), max_dd=float(dd))


# --------------------------------------------------------------------------- #
#  Causal off-switch overlays                                                  #
# --------------------------------------------------------------------------- #
def spy_state(spyf, dates):
    """Per-month SPY features aligned to the strategy's month-end dates
    (known at month-end t -> used to set exposure for the t->t+1 return)."""
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        if d in spyf.index:
            rows.append(spyf.loc[d].to_dict())
        else:
            p = spyf.index.searchsorted(d)
            p = min(max(p - 1, 0), len(spyf) - 1)
            rows.append(spyf.iloc[p].to_dict())
    return rows


def overlay_none(r, **kw):
    return np.asarray(r, float).copy()


def overlay_spy_offswitch(r, spy_rows, ret21_cut=-0.08, reenter_21=0.0):
    """Fast crash off-switch on SPY price state (causal).
    Go to CASH for the t->t+1 month when, as of month-end t, SPY is in a
    confirmed downtrend (below 200dma) AND its 21-day return <= ret21_cut.
    Re-enter as soon as SPY 21d return turns up past reenter_21 -- fast,
    asymmetric, so the post-crash V-recovery is NOT missed."""
    r = np.asarray(r, float)
    out = r.copy()
    for t in range(len(r)):
        s = spy_rows[t]
        below = s.get("spy_dsma200", 0.0) < 0.0
        r21 = s.get("spy_ret_21d", 0.0)
        if below and r21 <= ret21_cut and r21 < reenter_21:
            out[t] = 0.0  # park in cash this month
    return out


def overlay_basket_ddbreak(r, dd_cut=-0.20, mom_reenter=0.0, mom_win=2, park=None):
    """Basket-level fast breaker: when the strategy's OWN trailing equity
    drawdown (known through t-1) breaches dd_cut, switch to `park` (cash=0s
    if None, else a market-beta return stream) until its own short-term
    momentum (sum of last mom_win realised months) turns up past mom_reenter.
    Catches the bounce instead of waiting for full DD recovery."""
    r = np.asarray(r, float)
    out = r.copy()
    parkv = np.zeros_like(r) if park is None else np.asarray(park, float)
    eq, peak, off = 1.0, 1.0, False
    for t in range(len(r)):
        dd = eq / peak - 1.0
        if not off and dd <= dd_cut:
            off = True
        elif off:
            recent = r[max(0, t - mom_win):t].sum()
            if recent > mom_reenter:
                off = False
        if off:
            out[t] = parkv[t]
        eq *= (1 + out[t])
        peak = max(peak, eq)
    return out


def overlay_voltarget(r, target_ann=0.30, win=6):
    """Reference (prior agent's rejected lever) -- de-risk to a vol target,
    park excess in cash. Included to confirm it on the win-rate objective."""
    r = np.asarray(r, float)
    out = r.copy()
    tgt_m = target_ann / np.sqrt(12)
    for t in range(len(r)):
        if t >= win:
            v = r[t - win:t].std()
            e = 1.0 if v <= 1e-6 else min(1.0, tgt_m / v)
            out[t] = e * r[t]
    return out


# --------------------------------------------------------------------------- #
def build_e2(inp):
    members_g, preds, spyf, mr, mp, chronos = inp
    rlA = run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                     cost_bps=10.0, K=2, trigger_mode="blend",
                     select_mode="ml_3plus6")
    rlB = run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos,
                     cost_bps=10.0, K=2, trigger_mode="ml_3plus6",
                     select_mode="blend", adaptive_k=True, conv_lo=0.08,
                     conv_hi=0.18, k_lo=2, k_mid=3, k_hi=3,
                     regime_w={"bull": 0.30, "normal": 0.60, "recovery": 0.60})
    dates = [x["date"] for x in rlA]
    rA = np.array([x["ret_m"] for x in rlA])
    rB = np.array([x["ret_m"] for x in rlB])
    return dates, 0.5 * rA + 0.5 * rB


def report(nm, r, dates, spv, qqv, out):
    full = lump_metrics(r, dates)
    des = dual_dca(r, spv, qqv, dates, hi=DESIGN_END)
    hol = dual_dca(r, spv, qqv, dates, lo=HOLDOUT_START)
    allw = dual_dca(r, spv, qqv, dates)
    out[nm] = dict(full=full, design=des, holdout=hol, all=allw)
    wb = {H: allw[H]["win_both"] for H in (12, 36, 60, 120)}
    print(f"{nm:<26} CAGR {full['cagr']*100:5.1f}% DD {full['max_dd']*100:6.1f}% "
          f"Sh {full['sharpe']:.2f} | winBOTH 1y/3y/5y/10y "
          f"{wb[12]*100:4.0f}/{wb[36]*100:4.0f}/{wb[60]*100:4.0f}/{wb[120]*100:4.0f}  "
          f"worst1y vsSPY/QQQ {allw[12]['worst_vs_spy']:.2f}/{allw[12]['worst_vs_qqq']:.2f}")
    return out


def main():
    inp = load_inputs()
    _, _, spyf, mr, mp, _ = inp
    dates, rE2 = build_e2(inp)
    spv = bench_aligned(dates, mr, "SPY")
    qqv = bench_aligned(dates, mr, "QQQ")
    rows = spy_state(spyf, dates)
    out = {}

    print("=== ALL-PERIOD dual-benchmark (winBOTH = beats SPY-DCA AND QQQ-DCA) ===")
    report("E2 baseline (deployed)", rE2, dates, spv, qqv, out)
    # pre-registered off-switch menu
    report("E2 + SPY-offswitch", overlay_spy_offswitch(rE2, rows), dates, spv, qqv, out)
    report("E2 + basketDD -20%", overlay_basket_ddbreak(rE2, dd_cut=-0.20), dates, spv, qqv, out)
    report("E2 + basketDD -25%", overlay_basket_ddbreak(rE2, dd_cut=-0.25), dates, spv, qqv, out)
    report("E2 + basketDD -30%", overlay_basket_ddbreak(rE2, dd_cut=-0.30), dates, spv, qqv, out)
    report("E2 + voltgt 30%", overlay_voltarget(rE2, 0.30), dates, spv, qqv, out)
    # DIAGNOSTIC: same breaker but park in market-beta (SPY) instead of cash.
    # Isolates how much of the win-rate damage is cash-drag (missing recovery)
    # vs genuine tail-risk. NOT a deployment proposal (uses an ETF proxy).
    report("[diag] basketDD-25 park=beta", overlay_basket_ddbreak(rE2, dd_cut=-0.25, park=spv),
           dates, spv, qqv, out)
    report("[diag] basketDD-30 park=beta", overlay_basket_ddbreak(rE2, dd_cut=-0.30, park=spv),
           dates, spv, qqv, out)

    p = bw.CACHE / "v2" / "sp500_pit" / "augmented" / "improve_consistency_dca.json"
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
