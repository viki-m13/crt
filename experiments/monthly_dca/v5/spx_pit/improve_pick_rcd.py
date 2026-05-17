"""Decisive overfit gauntlet for RC D (regime-conditional select-blend:
momentum-tilt in bull, consensus in normal/recovery) vs WIN2.

RC D headline beat WIN2 on every axis at SAME DD. Is it a plateau or a
lucky cell? Screens: dense 3-D plateau (bull x norm x rec), cost
sensitivity (regime-w changes the basket -> can change turnover),
TRUE OOS, MC synthetic-delisting, and the RC B+adaptiveK consistency
variant.
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
from improve_pick_validate import dca_view, submetrics  # noqa
from improve_phase3 import mc_delist  # noqa
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp
    out = {}

    def G(rl):
        d = [r["date"] for r in rl]
        sp = spy_aligned(pd.to_datetime(d), mr)
        r = stream(rl)
        return r, sp, d, evaluate(r, d, sp), yearly(r, d, sp)

    def line(nm, r, sp, m, y, terse=False):
        dv = dca_view(r, sp)
        rec = {**m, **y, **dv}
        if terse:
            print(f"{nm:<26} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
                  f"  DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/"
                  f"{m['wf_n']}  era {m['eras_beat']}/4  yStd "
                  f"{y['yr_cagr_std']:.2f}")
        else:
            print(f"{nm:<26} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
                  f"  DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/"
                  f"{m['wf_n']}  era {m['eras_beat']}/4  yStd "
                  f"{y['yr_cagr_std']:.2f}  D1y {dv['win_12']} 3y "
                  f"{dv['win_36']} 10y {dv['win_120']}")
        return rec

    rl_w2 = run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                       cost_bps=10.0, K=2, trigger_mode="ml_3plus6",
                       select_mode="blend")
    r, sp, d, m, y = G(rl_w2)
    out["WIN2 (deployed)"] = line("WIN2 (deployed)", r, sp, m, y)

    def rc(bull, norm, rec, **extra):
        rw = {"bull": bull, "recovery": rec, "normal": norm, "crash": 0.5}
        return run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos,
                          cost_bps=extra.pop("cost_bps", 10.0), K=2,
                          trigger_mode="ml_3plus6", select_mode="blend",
                          regime_w=rw, **extra)

    print("\n--- 1. dense plateau: bull x norm x rec (terse) ---")
    plateau = []
    for b in (0.20, 0.25, 0.30, 0.35, 0.40):
        for nrm in (0.55, 0.60, 0.65):
            for rcv in (0.50, 0.60, 0.70):
                r, sp, d, m, yy = G(rc(b, nrm, rcv))
                rec = {"bull": b, "norm": nrm, "rec": rcv,
                       "cagr": m["cagr"], "sharpe": m["sharpe"],
                       "max_dd": m["max_dd"], "wf": m["wf_beats"],
                       "era": m["eras_beat"], "ystd": yy["yr_cagr_std"]}
                plateau.append(rec)
    out["plateau"] = plateau
    arr = pd.DataFrame(plateau)
    print(f"  n={len(arr)}  CAGR min/med/max "
          f"{arr.cagr.min()*100:.1f}/{arr.cagr.median()*100:.1f}/"
          f"{arr.cagr.max()*100:.1f}%   "
          f"WF>=9: {(arr.wf>=9).mean()*100:.0f}%  "
          f"era==4: {(arr.era==4).mean()*100:.0f}%  "
          f"CAGR>WIN2(48.8): {(arr.cagr>0.488).mean()*100:.0f}%  "
          f"DD all -54.5%: {(arr.max_dd.round(3)==-0.545).all()}")
    # neighbourhood of RC D (bull .25-.35, norm .55-.65, rec .5-.7)
    nb = arr[(arr.bull.between(0.25, 0.35)) & (arr.norm.between(0.55, 0.65))
             & (arr.rec.between(0.50, 0.70))]
    print(f"  RC-D neighbourhood (n={len(nb)}): CAGR "
          f"{nb.cagr.min()*100:.1f}-{nb.cagr.max()*100:.1f}%  "
          f"all beat WIN2: {(nb.cagr>0.488).all()}  "
          f"WF>=9 all: {(nb.wf>=9).all()}  era4 frac "
          f"{(nb.era==4).mean()*100:.0f}%")

    print("\n--- 2. RC D + neighbours full metrics ---")
    for (b, n, rv, nm) in [(0.30, 0.60, 0.60, "RC D  .30/.60/.60"),
                           (0.25, 0.60, 0.60, "     .25/.60/.60"),
                           (0.35, 0.60, 0.60, "     .35/.60/.60"),
                           (0.30, 0.55, 0.60, "     .30/.55/.60"),
                           (0.30, 0.65, 0.60, "     .30/.65/.60"),
                           (0.30, 0.60, 0.50, "     .30/.60/.50"),
                           (0.30, 0.60, 0.70, "     .30/.60/.70")]:
        r, sp, d, m, yy = G(rc(b, n, rv))
        out[nm.strip()] = line(nm, r, sp, m, yy)

    print("\n--- 3. cost sensitivity (RC D) ---")
    for c in (0.0, 10.0, 20.0, 30.0):
        r, sp, d, m, yy = G(rc(0.30, 0.60, 0.60, cost_bps=c))
        out[f"RC D cost={int(c)}"] = line(f"RC D cost={int(c)}bps",
                                          r, sp, m, yy, terse=True)

    print("\n--- 4. RC D + conviction-adaptive breadth ---")
    r, sp, d, m, yy = G(rc(0.30, 0.60, 0.60, adaptive_k=True,
                           conv_lo=0.08, conv_hi=0.18, k_lo=2, k_mid=3,
                           k_hi=3))
    out["RC D + adaptK"] = line("RC D + adaptK .08/.18/3", r, sp, m, yy)
    r, sp, d, m, yy = G(rc(0.20, 0.60, 0.70, adaptive_k=True,
                           conv_lo=0.08, conv_hi=0.18, k_lo=2, k_mid=3,
                           k_hi=3))
    out["RC B + adaptK"] = line("RC B + adaptK .08/.18/3", r, sp, m, yy)

    print("\n--- 5. TRUE OOS design 03-12 | holdout 13-26 ---")
    oos = {}
    for nm, rl in [("WIN2", rl_w2),
                   ("RC D", rc(0.30, 0.60, 0.60)),
                   ("RC D+adaptK", rc(0.30, 0.60, 0.60, adaptive_k=True,
                                      conv_lo=0.08, conv_hi=0.18, k_lo=2,
                                      k_mid=3, k_hi=3))]:
        dd = [x["date"] for x in rl]
        sp = spy_aligned(pd.to_datetime(dd), mr)
        rr = stream(rl)
        de = submetrics(rr, dd, sp, "2003-01-01", "2012-12-31")
        ho = submetrics(rr, dd, sp, "2013-01-01", "2026-12-31")
        oos[nm] = {"design": de, "holdout": ho}
        print(f"  {nm:<12} design CAGR {de['cagr']*100:5.1f}% Sh "
              f"{de['sharpe']:.2f} DD {de['max_dd']*100:5.1f}% | "
              f"holdout CAGR {ho['cagr']*100:5.1f}% Sh {ho['sharpe']:.2f} "
              f"DD {ho['max_dd']*100:5.1f}%")
    out["oos"] = oos

    print("\n--- 6. MC synthetic-delisting median CAGR ---")
    mcd = {}
    for a in (0.0, 0.04, 0.08):
        bv = mc_delist(rl_w2, mr, a)
        wv = mc_delist(rc(0.30, 0.60, 0.60), mr, a)
        mcd[f"a{int(a*100)}"] = {"WIN2": bv, "RC D": wv}
        print(f"  alpha {int(a*100):>2}%  WIN2 {bv*100:6.1f}%   "
              f"RC D {wv*100:6.1f}%")
    out["mc_delist"] = mcd

    p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
         / "improve_pick_rcd.json")
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
