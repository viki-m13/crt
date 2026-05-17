"""Phase 6b: finish the E1-grade overfit gauntlet for the two RC-E1
leaders — mix-weight plateau + MC synthetic-delisting (cost & TRUE-OOS
already passed in phase6).

  LEAD-CAGR : 0.5*WIN1 + 0.5*(RC D + adaptK)        [strict Pareto > E1]
  LEAD-CONS : 0.5*rcA_k + 0.5*rcB_k (pure RC-E1)    [max consistency/OOS]
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
from improve_sim_v2 import run_sim_v2  # noqa
from improve_pick_v3 import run_sim_v3  # noqa
from improve_phase4 import consistency  # noqa
from improve_phase3 import mc_delist  # noqa

RW_D = {"bull": 0.30, "recovery": 0.60, "normal": 0.60, "crash": 0.5}
AK = dict(adaptive_k=True, conv_lo=0.08, conv_hi=0.18, k_lo=2,
          k_mid=3, k_hi=3)


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def main():
    inp = load_inputs()
    mg, preds, spyf, mr, mp, chronos = inp

    bw.SCORER_MODE = "blend"
    base_rl, _, _ = bw.run_full_sim(mg, preds, preds, spyf, mr, mp,
                                    chronos_preds=chronos, cost_bps=10.0,
                                    hold_months=bw.HOLD_MONTHS,
                                    K=bw.K_PICKS)
    dates = [r["date"] for r in base_rl]
    spv = spy_aligned(pd.to_datetime(dates), mr)
    win1 = stream(base_rl)
    win2_rl = run_sim_v2(mg, preds, preds, spyf, mr, mp, chronos,
                         cost_bps=10.0, K=2, trigger_mode="ml_3plus6",
                         select_mode="blend")
    win2 = stream(win2_rl)
    e1 = 0.5 * win1 + 0.5 * win2
    e1_rl = [{"date": d, "ret_m": float(x), "picks": p["picks"],
              "basket_id": p["basket_id"]}
             for d, x, p in zip(dates, e1, base_rl)]

    rcA_k_rl = run_sim_v3(mg, preds, preds, spyf, mr, mp, chronos,
                          cost_bps=10.0, K=2, trigger_mode="blend",
                          select_mode="ml_3plus6", regime_w=RW_D, **AK)
    rcB_k_rl = run_sim_v3(mg, preds, preds, spyf, mr, mp, chronos,
                          cost_bps=10.0, K=2, trigger_mode="ml_3plus6",
                          select_mode="blend", regime_w=RW_D, **AK)
    rcA_k, rcB_k = stream(rcA_k_rl), stream(rcB_k_rl)

    def report(nm, r):
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        print(f"  {nm:<26} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  DD {m['max_dd']*100:5.1f}%  WF {m['wf_beats']}/"
              f"{m['wf_n']}  era {m['eras_beat']}/4  wrst5y "
              f"{c['roll60_min']*100:5.1f}%  CGx0309 "
              f"{c['cagr_ex0309']*100:.1f}%")
        return {**m, **c}

    out = {}
    print("=== baselines ===")
    out["E1"] = report("E1 (deployed)", e1)
    out["LEAD-CAGR"] = report("0.5 WIN1 + 0.5 rcB_k", 0.5 * win1 + 0.5 * rcB_k)
    out["LEAD-CONS"] = report("0.5 rcA_k + 0.5 rcB_k", 0.5 * rcA_k + 0.5 * rcB_k)

    print("\n=== mix-weight plateau (w on first sleeve) ===")
    pl = {}
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        a = report(f"LEAD-CAGR w={w}", w * win1 + (1 - w) * rcB_k)
        b = report(f"LEAD-CONS w={w}", w * rcA_k + (1 - w) * rcB_k)
        pl[f"cagr_w{w}"] = a
        pl[f"cons_w{w}"] = b
    out["plateau"] = pl

    print("\n=== MC synthetic-delisting median CAGR ===")
    mcd = {}
    for a in (0.0, 0.04, 0.08):
        e1v = mc_delist(e1_rl, mr, a)
        # blend two sleeves' delisting paths: average the two rl streams
        cv = 0.5 * mc_delist(base_rl, mr, a) + 0.5 * mc_delist(rcB_k_rl, mr, a)
        kv = 0.5 * mc_delist(rcA_k_rl, mr, a) + 0.5 * mc_delist(rcB_k_rl, mr, a)
        mcd[f"a{int(a*100)}"] = {"E1": e1v, "LEAD-CAGR": round(cv, 4),
                                 "LEAD-CONS": round(kv, 4)}
        print(f"  alpha {int(a*100):>2}%  E1 {e1v*100:6.1f}%   "
              f"LEAD-CAGR {cv*100:6.1f}%   LEAD-CONS {kv*100:6.1f}%")
    out["mc_delist"] = mcd

    p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
         / "improve_phase6b_rce1.json")
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
