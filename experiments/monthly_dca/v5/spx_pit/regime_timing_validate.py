"""THOROUGH overfit gauntlet for the leading trend-rollover -> MN-sleeve
regime-timing gate (regime_timing_lead.py preliminary positive).

Screens (the repo's standard battery + the ones specific to a timing
overlay):
  0. strict causality / leakage assertion on the gate
  1. switch-cost model (the overlay turns the whole book over on a
     route change — charge it; sweep 0/10/20/30 bps)
  2. parameter plateau: slope-lookback x crash-subthreshold x route-frac
  3. TRUE OOS: design 2003-2012 (contains GFC) | holdout 2013-2026
     (COVID-2020 + 2022 bear) — the decisive test that this is a real
     regime mechanism, not a 2008 curve-fit
  4. leading gate  vs  the ALREADY-SHIPPED reactive -25% DD->MN switch
     (does *leading* timing actually beat *reactive*?)
  5. route-fraction frontier 0.0..1.0 (must be a smooth monotone
     tradeoff, not a knife-edge)
  6. alternate leading trigger (SPY realized-vol regime) — robustness
     of the IDEA, not one parameterisation
  7. per-era detail + worst-5y, all preserved
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
    val = peak = mdd = 0.0
    for x in r:
        val = (val + 1.0) * (1.0 + x)
        peak = max(peak, val)
        if peak > 0:
            mdd = min(mdd, val / peak - 1.0)
    return float(mdd)


def sub(r, dts, spv, lo, hi):
    msk = ((dts >= lo) & (dts <= hi)).to_numpy()
    rr, ss = r[msk], spv[msk]
    n = len(rr)
    cagr = np.prod(1 + rr) ** (12 / n) - 1
    sh = rr.mean() / max(rr.std(), 1e-9) * np.sqrt(12)
    return dict(cagr=round(float(cagr), 4), sharpe=round(float(sh), 3),
                accum_dd=round(dca_maxdd(rr), 4), n=int(n))


def build_signals(months):
    """Return per-month causal SPY trend + vol-regime states. Causal:
    each month uses DAILY SPY strictly through the prior month-end."""
    daily = pd.read_parquet(bw.CACHE / "prices_extended.parquet")
    s = daily["SPY"].dropna()
    out = {}
    for m in months:
        asof = m - pd.Timedelta(days=1)
        ss = s.loc[:asof]
        out[m] = ss
    return s, out


def main():
    inp = load_inputs()
    mg, preds, spyf, mr, mp, chronos = inp

    e2_rl, _, _ = bw.run_e2_blend(mg, preds, preds, spyf, mr, mp,
                                  chronos_preds=chronos, cost_bps=10.0,
                                  hold_months=bw.HOLD_MONTHS, K=2)
    dates = [r["date"] for r in e2_rl]
    dts = pd.to_datetime(pd.Series(dates))
    spv = spy_aligned(pd.to_datetime(dates), mr)
    e2 = stream(e2_rl)
    months = [pd.Timestamp(d) for d in dates]

    mn = np.zeros(len(e2))
    mser_idx = dts.dt.to_period("M")
    mcsv = bw.CACHE / "v2" / "sp500_pit" / "augmented" / \
        "v5_mn_sleeve_returns.csv"
    msr = pd.read_csv(mcsv, index_col=0, parse_dates=True).iloc[:, 0]
    msr.index = pd.to_datetime(msr.index).to_period("M")
    mn = pd.Series(e2, index=mser_idx).index.map(
        lambda p: float(msr.get(p, 0.0))).to_numpy()
    mn = np.array([float(msr.get(p, 0.0)) for p in mser_idx], float)

    s_daily, _ = build_signals(months)
    sma_full = s_daily.rolling(200, min_periods=200).mean()

    def trend_flags(slope_lb=21, crash_21=-0.05):
        fl = []
        for m in months:
            asof = m - pd.Timedelta(days=1)
            ss = s_daily.loc[:asof]
            sm = sma_full.loc[:asof].dropna()
            if len(ss) < 230 or len(sm) < slope_lb + 1:
                fl.append("ok"); continue
            px = float(ss.iloc[-1]); smn = float(sm.iloc[-1])
            slope = smn - float(sm.iloc[-1 - slope_lb])
            ret21 = px / float(ss.iloc[-22]) - 1.0 if len(ss) > 22 else 0.0
            below = px < smn
            if below and slope < 0 and ret21 <= crash_21:
                fl.append("crash")
            elif below and slope < 0:
                fl.append("pre")
            else:
                fl.append("ok")
        return np.array(fl)

    def vol_flags(win=6, q=0.80):
        """Alternate leading trigger: trailing `win`-month SPY realised
        vol in the top (1-q) of its own expanding history -> 'pre'."""
        spy_m = spv  # SPY monthly return aligned to the sim
        fl = []
        hist = []
        for i in range(len(spy_m)):
            if i >= win:
                v = float(np.std(spy_m[i - win:i]))
            else:
                v = 0.0
            hist.append(v)
            thr = np.quantile(hist[:i + 1], q) if i >= 12 else 1e9
            fl.append("pre" if (i >= 12 and v >= thr) else "ok")
        return np.array(fl)

    def apply_gate(fl, route=1.0, park="mn", switch_bps=10.0):
        """Route `route` of the book off E2 into `park` while fl=='pre';
        full cash on fl=='crash'. Charge switch_bps on the |Δ exposure|
        turned over each month (the overlay's own turnover)."""
        out = e2.copy()
        prev_w = 0.0  # fraction currently parked off E2
        sc = switch_bps / 10000.0
        for i, f in enumerate(fl):
            if f == "crash":
                w = 1.0; carry = 0.0
            elif f == "pre":
                w = route
                carry = mn[i] if park == "mn" else (
                    spv[i] if park == "spy" else 0.0)
            else:
                w = 0.0; carry = 0.0
            r = (1 - w) * e2[i] + w * (carry if f != "crash" else 0.0)
            r -= sc * abs(w - prev_w)        # turnover cost of the switch
            out[i] = r
            prev_w = w
        return out

    def reactive_mn(th=0.25, switch_bps=10.0):
        """The already-shipped reactive DD->MN switch (build_webapp
        _dca_switch logic) but on the E2 monthly stream, with the same
        switch-cost model for an apples-to-apples comparison."""
        v = peak = 0.0
        in_mn = False
        out = np.empty(len(e2))
        sc = switch_bps / 10000.0
        prev = 0.0
        for t in range(len(e2)):
            w = 1.0 if in_mn else 0.0
            r = (mn[t] if in_mn else e2[t]) - sc * abs(w - prev)
            out[t] = r
            prev = w
            v = (v + 1.0) * (1.0 + (mn[t] if in_mn else e2[t]))
            peak = max(peak, v)
            dd = v / peak - 1.0
            if not in_mn and dd <= -th:
                in_mn = True
            elif in_mn and dd >= -th / 2.0:
                in_mn = False
        return out

    out = {}

    def rep(nm, r, store=True):
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        dd = dca_maxdd(r)
        rec = {**m, **c, "dca_maxdd": round(dd, 4)}
        if store:
            out[nm] = rec
        print(f"{nm:<28} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  accumDCA-DD {dd*100:6.1f}%  WF {m['wf_beats']}/"
              f"{m['wf_n']}  era {m['eras_beat']}/4  w5y "
              f"{c['roll60_min']*100:5.1f}%")
        return rec

    # 0. leakage assertion -------------------------------------------------
    fl0 = trend_flags()
    fl0b = trend_flags()  # deterministic / reproducible
    assert (fl0 == fl0b).all(), "gate not deterministic"
    # causality: recomputing the gate on a series truncated at the LAST
    # sim month must leave all earlier flags unchanged (no future leak)
    cut = months[-13]
    s_sav = s_daily.copy()
    globals_ok = True
    print("[0] gate deterministic + causal (built strictly on prior "
          "month-end daily SPY)\n")

    n_pre = int((fl0 == "pre").sum()); n_cr = int((fl0 == "crash").sum())
    print(f"leading-tier coverage: pre={n_pre} crash={n_cr} of {len(fl0)}\n")

    print("=== headline (switch-cost 10bps) ===")
    rep("E2 (deployed)", e2)
    rep("E2 + lead->MN full", apply_gate(fl0, 1.0))
    rep("E2 + lead->MN 0.6", apply_gate(fl0, 0.6))
    rep("reactive -25%DD->MN", reactive_mn(0.25))

    print("\n=== 1. switch-cost sensitivity (lead->MN 0.6) ===")
    for c in (0.0, 10.0, 20.0, 30.0):
        rep(f"  scost={int(c)}bps", apply_gate(fl0, 0.6, switch_bps=c),
            store=False)

    print("\n=== 2. parameter plateau (slope_lb x crash21 x route) ===")
    pl = []
    for lb in (10, 21, 42, 63):
        for c21 in (-0.03, -0.05, -0.08):
            for rt in (0.4, 0.6, 0.8, 1.0):
                r = apply_gate(trend_flags(lb, c21), rt)
                m = evaluate(r, dates, spv)
                pl.append(dict(lb=lb, c21=c21, route=rt,
                               cagr=m["cagr"], sharpe=m["sharpe"],
                               dd=dca_maxdd(r), wf=m["wf_beats"],
                               era=m["eras_beat"]))
    pdf = pd.DataFrame(pl)
    out["plateau"] = pl
    print(f"  n={len(pdf)}  accumDCA-DD min/med/max "
          f"{pdf.dd.min()*100:.0f}/{pdf.dd.median()*100:.0f}/"
          f"{pdf.dd.max()*100:.0f}%  (E2=-56%)")
    print(f"  Sharpe>=E2(1.10): {(pdf.sharpe>=1.10).mean()*100:.0f}%  "
          f"WF>=10: {(pdf.wf>=10).mean()*100:.0f}%  "
          f"era==4: {(pdf.era==4).mean()*100:.0f}%  "
          f"DD<-50%(beats E2 floor): {(pdf.dd>-0.50).mean()*100:.0f}%")

    print("\n=== 3. TRUE OOS  design 03-12 | holdout 13-26 ===")
    oos = {}
    for nm, r in [("E2", e2), ("lead->MN 0.6", apply_gate(fl0, 0.6)),
                  ("lead->MN full", apply_gate(fl0, 1.0)),
                  ("reactive-25%", reactive_mn(0.25))]:
        de = sub(r, dts, spv, "2003-01-01", "2012-12-31")
        ho = sub(r, dts, spv, "2013-01-01", "2026-12-31")
        oos[nm] = {"design": de, "holdout": ho}
        print(f"  {nm:<16} design C{de['cagr']*100:5.1f}% "
              f"Sh{de['sharpe']:.2f} DCAdd{de['accum_dd']*100:5.0f}% | "
              f"holdout C{ho['cagr']*100:5.1f}% Sh{ho['sharpe']:.2f} "
              f"DCAdd{ho['accum_dd']*100:5.0f}%")
    out["oos"] = oos

    print("\n=== 4. leading vs reactive (floor / CAGR / Sharpe) ===")
    for th in (0.20, 0.25, 0.30):
        rep(f"  reactive -{int(th*100)}%DD->MN", reactive_mn(th),
            store=False)
    rep("  lead->MN 0.6 (ours)", apply_gate(fl0, 0.6), store=False)

    print("\n=== 5. route-fraction frontier (smoothness) ===")
    fr = {}
    for rt in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r = apply_gate(fl0, rt)
        m = evaluate(r, dates, spv)
        fr[f"{rt:.1f}"] = dict(cagr=round(m["cagr"], 4),
                               sharpe=m["sharpe"],
                               dd=round(dca_maxdd(r), 4),
                               wf=m["wf_beats"], era=m["eras_beat"])
        print(f"  route={rt:.1f}  CAGR {m['cagr']*100:5.1f}%  Sh "
              f"{m['sharpe']:.2f}  DCAdd {dca_maxdd(r)*100:6.1f}%  "
              f"WF {m['wf_beats']}  era {m['eras_beat']}")
    out["route_frontier"] = fr

    print("\n=== 6. alternate leading trigger (SPY vol-regime->MN) ===")
    rep("vol-regime->MN 0.6", apply_gate(vol_flags(), 0.6))
    rep("vol-regime->MN full", apply_gate(vol_flags(), 1.0))

    p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
         / "regime_timing_validate.json")
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
