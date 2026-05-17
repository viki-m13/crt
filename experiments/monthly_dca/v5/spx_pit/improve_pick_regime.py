"""Beat WIN2 decisively: regime-conditional select-blend weight.

WIN2 = trigger=ml_3plus6, select=blend(0.5). Its weakness vs WIN1 is
CAGR (48.8 vs 51.9) + worst-year. Hypothesis: lean momentum (low
blend_w) when the market regime is healthy (capture WIN1 CAGR), lean
consensus (high blend_w) when shaky (keep WIN2 DD). One causal lever,
the regime label is the already-audited classify_regime_tight output.

Gauntlet vs both WIN1 and WIN2 + per-year consistency + DCA view +
plateau + cost + TRUE OOS.
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
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp
    out = {}

    def G(rl):
        dts = [r["date"] for r in rl]
        spv = spy_aligned(pd.to_datetime(dts), mr)
        r = stream(rl)
        return r, spv, dts, evaluate(r, dts, spv), yearly(r, dts, spv)

    def line(nm, r, spv, m, y):
        dv = dca_view(r, spv)
        print(f"{nm:<30} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}  "
              f"DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}  "
              f"era {m['eras_beat']}/4  yStd {y['yr_cagr_std']:.2f}  "
              f"wYr {y['worst_year']*100:5.1f}%  "
              f"D1y {dv['win_12']} 3y {dv['win_36']} 5y {dv['win_60']} "
              f"10y {dv['win_120']}  wM1y {dv['worst_ratio_12']}")
        return {**m, **y, **dv}

    # baselines
    r, spv, dts, m, y = G(run_sim_v2(members_g, preds, preds, spyf, mr, mp,
                                     chronos, cost_bps=10.0, K=2,
                                     trigger_mode="blend",
                                     select_mode="ml_3plus6"))
    out["WIN1"] = line("WIN1 (trig=blend,sel=ml)", r, spv, m, y)
    r, spv, dts, m, y = G(run_sim_v2(members_g, preds, preds, spyf, mr, mp,
                                     chronos, cost_bps=10.0, K=2,
                                     trigger_mode="ml_3plus6",
                                     select_mode="blend"))
    out["WIN2"] = line("WIN2 (trig=ml,sel=blend)*", r, spv, m, y)

    def run(nm, **kw):
        rl = run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos,
                        cost_bps=kw.pop("cost_bps", 10.0), K=2, **kw)
        r, spv, dts, m, y = G(rl)
        out[nm] = line(nm, r, spv, m, y)
        return rl

    print("\n--- 1. static select blend_w sweep (trig=ml, sel=blend) ---")
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        run(f"sel=blend w={w}", trigger_mode="ml_3plus6",
            select_mode="blend", blend_w=w)

    # Regime label set from classify_regime_tight: bull/recovery/normal/crash
    # (crash -> cash, never reaches select). Lean momentum (low w) in bull,
    # consensus (high w) in normal/recovery.
    print("\n--- 2. regime-conditional blend_w (trig=ml, sel=blend) ---")
    grids = {
        "RC A bull.2 rec.6 norm.5": {"bull": 0.2, "recovery": 0.6,
                                     "normal": 0.5, "crash": 0.5},
        "RC B bull.2 rec.7 norm.6": {"bull": 0.2, "recovery": 0.7,
                                     "normal": 0.6, "crash": 0.5},
        "RC C bull.1 rec.7 norm.6": {"bull": 0.1, "recovery": 0.7,
                                     "normal": 0.6, "crash": 0.5},
        "RC D bull.3 rec.6 norm.6": {"bull": 0.3, "recovery": 0.6,
                                     "normal": 0.6, "crash": 0.5},
        "RC E bull.2 rec.8 norm.7": {"bull": 0.2, "recovery": 0.8,
                                     "normal": 0.7, "crash": 0.5},
        "RC F bull.0 rec.6 norm.5": {"bull": 0.0, "recovery": 0.6,
                                     "normal": 0.5, "crash": 0.5},
    }
    for nm, rw in grids.items():
        run(nm, trigger_mode="ml_3plus6", select_mode="blend",
            regime_w=rw)

    print("\n--- 3. + conviction-adaptive breadth on best RC ---")
    bestrw = grids["RC B bull.2 rec.7 norm.6"]
    run("RC B + adaptK .08/.18/3", trigger_mode="ml_3plus6",
        select_mode="blend", regime_w=bestrw, adaptive_k=True,
        conv_lo=0.08, conv_hi=0.18, k_lo=2, k_mid=3, k_hi=3)

    print("\n--- 4. cost insensitivity (RC B) ---")
    for c in (0.0, 10.0, 20.0, 30.0):
        run(f"RC B cost={int(c)}", trigger_mode="ml_3plus6",
            select_mode="blend", regime_w=bestrw, cost_bps=c)

    print("\n--- 5. TRUE OOS design 03-12 | holdout 13-26 ---")
    oos = {}
    cfg = {
        "WIN2": ("v2", dict(trigger_mode="ml_3plus6",
                            select_mode="blend")),
        "RC B": ("v3", dict(trigger_mode="ml_3plus6",
                            select_mode="blend", regime_w=bestrw)),
    }
    for nm, (eng, kw) in cfg.items():
        rl = (run_sim_v2 if eng == "v2" else run_sim_v3)(
            members_g, preds, preds, spyf, mr, mp, chronos,
            cost_bps=10.0, K=2, **kw)
        d = [x["date"] for x in rl]
        sp = spy_aligned(pd.to_datetime(d), mr)
        rr = stream(rl)
        de = submetrics(rr, d, sp, "2003-01-01", "2012-12-31")
        ho = submetrics(rr, d, sp, "2013-01-01", "2026-12-31")
        oos[nm] = {"design": de, "holdout": ho}
        print(f"  {nm:<6} design CAGR {de['cagr']*100:5.1f}% Sh "
              f"{de['sharpe']:.2f} DD {de['max_dd']*100:5.1f}% | "
              f"holdout CAGR {ho['cagr']*100:5.1f}% Sh {ho['sharpe']:.2f} "
              f"DD {ho['max_dd']*100:5.1f}%")
    out["oos"] = oos

    p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
         / "improve_pick_regime.json")
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
