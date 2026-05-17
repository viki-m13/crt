"""Overfit gauntlet for the conviction-adaptive-breadth finding, plus a
new regime/quality-defensive selection lever, plus the adaptiveK x
blend-select combo (blend-select is the repo's lowest-DD known lever).

Screens (the repo's standard anti-overfit battery):
  1. parameter plateau (conv thresholds, k_hi) — must be a plateau
  2. cost insensitivity (0/10/20/30 bps)
  3. TRUE OOS  design 2003-2012 | untouched holdout 2013-2026
  4. DCA-investor view (the actual product): rolling 1y/3y/5y/10y win
     vs SPY-DCA + worst window — lump-sum CAGR understates a
     consistency lever for a monthly contributor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from improve_main_strategy import load_inputs, evaluate, spy_aligned  # noqa
from improve_sim_v2 import run_sim_v2  # noqa
from improve_pick_v3 import run_sim_v3  # noqa
from improve_pick_run import yearly  # noqa
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def dca_t(r):
    v = 0.0
    for x in r:
        v = (v + 1) * (1 + x)
    return v


def dca_view(r, spv):
    """Rolling monthly-DCA win-rate vs SPY-DCA and worst MOIC ratio."""
    n = len(r)
    o = {}
    for H in (12, 36, 60, 120):
        wins, ratios = [], []
        for s in range(0, n - H + 1):
            a, b = dca_t(r[s:s + H]), dca_t(spv[s:s + H])
            wins.append(a > b)
            ratios.append(a / b if b > 0 else np.nan)
        o[f"win_{H}"] = round(float(np.mean(wins)), 3) if wins else None
        o[f"worst_ratio_{H}"] = round(float(np.nanmin(ratios)), 3) if ratios else None
    return o


def submetrics(r, dates, spv, lo, hi):
    d = pd.Series(pd.to_datetime(dates))
    msk = ((d >= lo) & (d <= hi)).to_numpy()
    rr, ss = r[msk], spv[msk]
    n = len(rr)
    cagr = np.prod(1 + rr) ** (12 / n) - 1
    sh = rr.mean() / max(rr.std(), 1e-9) * np.sqrt(12)
    e = np.cumprod(1 + rr)
    mdd = float(((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e)).min())
    return dict(cagr=round(float(cagr), 4), sharpe=round(float(sh), 3),
                max_dd=round(mdd, 4), n=int(n))


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp

    dates_ref = None

    def G(rl):
        nonlocal dates_ref
        dates_ref = [r["date"] for r in rl]
        spv = spy_aligned(pd.to_datetime(dates_ref), mr)
        r = stream(rl)
        return r, spv, evaluate(r, dates_ref, spv), yearly(r, dates_ref, spv)

    def line(nm, r, spv, m, y):
        dv = dca_view(r, spv)
        print(f"{nm:<32} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}  "
              f"DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}  "
              f"era {m['eras_beat']}/4  yStd {y['yr_cagr_std']:.2f}  "
              f"wYr {y['worst_year']*100:5.1f}%  "
              f"DCA1y {dv['win_12']} 3y {dv['win_36']} 10y {dv['win_120']}  "
              f"wMOIC1y {dv['worst_ratio_12']}")
        return {**m, **y, **dv}

    out = {}

    rl = run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                    cost_bps=10.0, K=2, trigger_mode="blend",
                    select_mode="ml_3plus6")
    r0, spv, m0, y0 = G(rl)
    out["WIN1"] = line("WIN1 (deployed)", r0, spv, m0, y0)

    def run(nm, **kw):
        rl = run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos,
                        cost_bps=kw.pop("cost_bps", 10.0), K=2, **kw)
        r, sp, m, y = G(rl)
        out[nm] = line(nm, r, sp, m, y)
        return rl

    AK = dict(trigger_mode="blend", select_mode="ml_3plus6",
              adaptive_k=True, conv_lo=0.08, conv_hi=0.18, k_lo=2,
              k_mid=3, k_hi=3)

    print("\n--- 1. adaptiveK parameter plateau ---")
    for lo, hi, kmid, khi in [(0.06, 0.15, 3, 3), (0.07, 0.16, 3, 3),
                              (0.08, 0.18, 3, 3), (0.09, 0.19, 3, 3),
                              (0.10, 0.20, 3, 3), (0.08, 0.18, 3, 4),
                              (0.08, 0.16, 4, 4), (0.07, 0.20, 3, 3)]:
        run(f"AK lo{lo} hi{hi} mid{kmid} hi{khi}", trigger_mode="blend",
            select_mode="ml_3plus6", adaptive_k=True, conv_lo=lo,
            conv_hi=hi, k_lo=2, k_mid=kmid, k_hi=khi)

    print("\n--- 2. cost insensitivity (AK .08/.18/3) ---")
    for c in (0.0, 10.0, 20.0, 30.0):
        run(f"AK cost={int(c)}bps", cost_bps=c, **AK)

    print("\n--- 3. adaptiveK x blend-select (DD lever) ---")
    run("AK + sel=blend", trigger_mode="blend", select_mode="blend",
        adaptive_k=True, conv_lo=0.08, conv_hi=0.18, k_lo=2, k_mid=3,
        k_hi=3)
    run("sel=blend only (WIN2)", trigger_mode="ml_3plus6",
        select_mode="blend")

    print("\n--- 4. TRUE OOS (design 03-12 | holdout 13-26) ---")
    cfgs = {
        "WIN1": ("v2", dict(trigger_mode="blend", select_mode="ml_3plus6")),
        "AK .08/.18/3": ("v3", AK),
        "AK + sel=blend": ("v3", dict(trigger_mode="blend",
                                      select_mode="blend", adaptive_k=True,
                                      conv_lo=0.08, conv_hi=0.18, k_lo=2,
                                      k_mid=3, k_hi=3)),
    }
    oos = {}
    for nm, (eng, kw) in cfgs.items():
        if eng == "v2":
            rl = run_sim_v2(members_g, preds, preds, spyf, mr, mp,
                            chronos, cost_bps=10.0, K=2, **kw)
        else:
            rl = run_sim_v3(members_g, preds, preds, spyf, mr, mp,
                            chronos, cost_bps=10.0, K=2, **kw)
        dts = [x["date"] for x in rl]
        spv2 = spy_aligned(pd.to_datetime(dts), mr)
        rr = stream(rl)
        des = submetrics(rr, dts, spv2, "2003-01-01", "2012-12-31")
        hol = submetrics(rr, dts, spv2, "2013-01-01", "2026-12-31")
        oos[nm] = {"design": des, "holdout": hol}
        print(f"  {nm:<18} design CAGR {des['cagr']*100:5.1f}% "
              f"Sh {des['sharpe']:.2f} DD {des['max_dd']*100:5.1f}% | "
              f"holdout CAGR {hol['cagr']*100:5.1f}% Sh {hol['sharpe']:.2f} "
              f"DD {hol['max_dd']*100:5.1f}%")
    out["oos"] = oos

    p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
         / "improve_pick_validate.json")
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
