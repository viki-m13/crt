"""Regime-timing R&D: can a LEADING de-risk tier lower E2's -56% GFC
floor without the documented CAGR bleed of reactive overlays?

The deployed crash gate (classify_regime_tight) fires on a 21-day SPY
shock — it is LATE (only ~3 months in all of 2008; DCA_INVESTOR_EVAL).
Reactive vol-target / DD-breaker bleed 9-27pp CAGR (IMPROVE_FINDINGS
Phase 11 / improve_main_strategy). Untried: a TREND-ROLLOVER pre-crash
tier (SPY below 200dma AND 200dma slope negative) that fires EARLIER
and routes the book to the repo's validated market-neutral sleeve
(augmented/v5_mn_sleeve_returns.csv, ~-18% maxDD, rho~0) instead of
cash — keeping a positive-carry, low-vol stream on while the systemic
drawdown plays out, then back to E2 when the trend repairs.

Evaluated on the metric that IS the floor: the accumulating monthly-DCA
max value drawdown (not just lump-sum), plus CAGR / Sharpe / WF /
worst-rolling-5y, vs deployed E2. Causal: the gate at month t uses only
SPY data through t-1.
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

import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa
from improve_main_strategy import load_inputs, evaluate, spy_aligned  # noqa
from improve_phase4 import consistency  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def dca_maxdd(r):
    """Max drawdown of the ACCUMULATING $1/mo portfolio value (the
    number the user calls 'the floor')."""
    val, contrib, peak, mdd = 0.0, 0.0, 0.0, 0.0
    for x in r:
        val = (val + 1.0) * (1.0 + x)
        contrib += 1.0
        peak = max(peak, val)
        if peak > 0:
            mdd = min(mdd, val / peak - 1.0)
    return float(mdd)


def sma_slope_flags(months):
    """Per-month SPY trend state from DAILY SPY, strictly causal (data
    through the prior month-end only): returns dict month-> 'crash' |
    'predeteriorate' | 'ok'.
      crash         : SPY < 200dma AND 200dma slope<0 AND 21d ret <= -5%
      predeteriorate: SPY < 200dma AND 200dma slope < 0   (leading)
      ok            : otherwise
    """
    daily = pd.read_parquet(bw.CACHE / "prices_extended.parquet")
    s = daily["SPY"].dropna()
    sma = s.rolling(200, min_periods=200).mean()
    out = {}
    for m in months:
        asof = m - pd.Timedelta(days=1)  # causal: through prior month
        ss = s.loc[:asof]
        if len(ss) < 230:
            out[m] = "ok"
            continue
        px = float(ss.iloc[-1])
        sm = sma.loc[:asof].dropna()
        sma_now = float(sm.iloc[-1])
        sma_21ago = float(sm.iloc[-22]) if len(sm) > 22 else sma_now
        slope = sma_now - sma_21ago
        ret21 = px / float(ss.iloc[-22]) - 1.0 if len(ss) > 22 else 0.0
        below = px < sma_now
        if below and slope < 0 and ret21 <= -0.05:
            out[m] = "crash"
        elif below and slope < 0:
            out[m] = "predeteriorate"
        else:
            out[m] = "ok"
    return out


def main():
    inp = load_inputs()
    mg, preds, spyf, mr, mp, chronos = inp

    e2_rl, _, _ = bw.run_e2_blend(mg, preds, preds, spyf, mr, mp,
                                  chronos_preds=chronos, cost_bps=10.0,
                                  hold_months=bw.HOLD_MONTHS, K=2)
    dates = [r["date"] for r in e2_rl]
    dts = pd.to_datetime(dates)
    spv = spy_aligned(dts, mr)
    e2 = stream(e2_rl)

    # market-neutral sleeve stream, aligned to the sim months
    mn = np.zeros(len(e2))
    mn_csv = bw.CACHE / "v2" / "sp500_pit" / "augmented" / \
        "v5_mn_sleeve_returns.csv"
    if mn_csv.exists():
        m = pd.read_csv(mn_csv, index_col=0, parse_dates=True).iloc[:, 0]
        m.index = pd.to_datetime(m.index).to_period("M")
        mser = pd.Series(e2, index=dts.to_period("M"))
        al = m.reindex(mser.index).astype(float)
        mn = al.fillna(0.0).to_numpy()

    flags = sma_slope_flags([pd.Timestamp(d) for d in dts])
    fl = np.array([flags[pd.Timestamp(d)] for d in dts])

    def overlay(park="mn", predérisk=1.0):
        """predérisk = fraction of book moved off E2 into `park` while
        the leading 'predeteriorate' tier is on (1.0 = full).
        'crash' tier always -> cash (0 return)."""
        out = e2.copy()
        for i, f in enumerate(fl):
            if f == "crash":
                out[i] = 0.0
            elif f == "predeteriorate":
                carry = mn[i] if park == "mn" else (
                    spv[i] if park == "spy" else 0.0)
                out[i] = (1 - predérisk) * e2[i] + predérisk * carry
        return out

    rows = {}

    def rep(nm, r):
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        dd = dca_maxdd(r)
        rows[nm] = {**m, **c, "dca_maxdd": round(dd, 4)}
        print(f"{nm:<26} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  lumpDD {m['max_dd']*100:6.1f}%  **accumDCA-DD "
              f"{dd*100:6.1f}%**  WF {m['wf_beats']}/{m['wf_n']}  era "
              f"{m['eras_beat']}/4  wrst5y {c['roll60_min']*100:5.1f}%")

    n_pre = int((fl == "predeteriorate").sum())
    n_cr = int((fl == "crash").sum())
    print(f"leading tier months: predeteriorate={n_pre}  crash={n_cr}  "
          f"(of {len(fl)})\n")
    rep("E2 (deployed)", e2)
    rep("E2 + lead->MN (full)", overlay("mn", 1.0))
    rep("E2 + lead->MN (0.6)", overlay("mn", 0.6))
    rep("E2 + lead->cash (full)", overlay("cash", 1.0))
    rep("E2 + lead->SPY (full)", overlay("spy", 1.0))

    out_p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
             / "regime_timing_lead.json")
    out_p.write_text(json.dumps(rows, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
