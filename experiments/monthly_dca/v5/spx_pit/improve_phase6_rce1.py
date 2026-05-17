"""Phase 6: stack E1's free-consistency lever ON the stronger RC-D sleeve.

E1 (deployed) = 0.5*WIN1 + 0.5*WIN2 : decorrelating two role-swapped
implementations of the SAME alpha buys consistency for free (worst-5y
+2.5 -> +11.9 %/yr, DD -66 -> -56 %).

RC D (this branch) = a STRICTLY STRONGER single sleeve than WIN1/WIN2
(regime-conditional select-blend: momentum-lean in bull, consensus in
normal/recovery). E1's decorrelation lever is ORTHOGONAL to RC D's
regime lever, so they should stack: build E1 out of RC-D-grade sleeves
instead of plain WIN1/WIN2.

Sleeves (role-swap pair, same alpha, decorrelated rebalance/selection):
  RC-A : trigger=regime_blend , select=ml_3plus6   (WIN1-role + RC)
  RC-B : trigger=ml_3plus6    , select=regime_blend (WIN2-role + RC = RC D)
  +adaptK option = conviction-adaptive breadth on the sleeve.

Compared apples-to-apples vs the DEPLOYED E1 on the exact phase-4
gauntlet (same evaluate() + consistency()).
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
import improve_sim_v2 as v2  # noqa
from improve_main_strategy import load_inputs, evaluate, spy_aligned  # noqa
from improve_sim_v2 import run_sim_v2  # noqa
from improve_pick_v3 import run_sim_v3  # noqa
from improve_phase4 import consistency  # noqa
from improve_pick_validate import dca_view, submetrics  # noqa

RW_D = {"bull": 0.30, "recovery": 0.60, "normal": 0.60, "crash": 0.5}
AK = dict(adaptive_k=True, conv_lo=0.08, conv_hi=0.18, k_lo=2,
          k_mid=3, k_hi=3)


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def main():
    inp = load_inputs()
    mg, preds, spyf, mr, mp, chronos = inp

    # --- deployed E1 baseline, EXACT phase-4 construction ---
    bw.SCORER_MODE = "blend"
    base_rl, _, _ = bw.run_full_sim(mg, preds, preds, spyf, mr, mp,
                                    chronos_preds=chronos, cost_bps=10.0,
                                    hold_months=bw.HOLD_MONTHS,
                                    K=bw.K_PICKS)
    dates = [r["date"] for r in base_rl]
    spv = spy_aligned(pd.to_datetime(dates), mr)
    win1 = stream(base_rl)
    win2 = stream(run_sim_v2(mg, preds, preds, spyf, mr, mp, chronos,
                             cost_bps=10.0, K=2,
                             trigger_mode="ml_3plus6",
                             select_mode="blend"))
    e1 = 0.5 * win1 + 0.5 * win2

    def v3(trig, sel, **kw):
        return stream(run_sim_v3(mg, preds, preds, spyf, mr, mp, chronos,
                                 cost_bps=10.0, K=2, trigger_mode=trig,
                                 select_mode=sel, regime_w=RW_D, **kw))

    # RC sleeves (regime-conditional blend in either role)
    rcA = v3("blend", "ml_3plus6")            # WIN1-role + RC trigger
    rcB = v3("ml_3plus6", "blend")            # WIN2-role + RC select = RC D
    rcA_k = v3("blend", "ml_3plus6", **AK)
    rcB_k = v3("ml_3plus6", "blend", **AK)

    variants = {
        "WIN1": win1,
        "WIN2": win2,
        "E1 (deployed)": e1,
        "RC D (rcB) solo": rcB,
        "RC D + adaptK solo": rcB_k,
        "RC-E1 = .5 rcA + .5 rcB": 0.5 * rcA + 0.5 * rcB,
        "RC-E1 + adaptK (both)": 0.5 * rcA_k + 0.5 * rcB_k,
        "RC-E1 adaptK on rcB only": 0.5 * rcA + 0.5 * rcB_k,
        "hyb .5 WIN1 + .5 rcB": 0.5 * win1 + 0.5 * rcB,
        "hyb .5 WIN1 + .5 rcB_k": 0.5 * win1 + 0.5 * rcB_k,
        "hyb .5 WIN2 + .5 rcB_k": 0.5 * win2 + 0.5 * rcB_k,
    }

    rows = {}
    for nm, r in variants.items():
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        dv = dca_view(r, spv)
        rows[nm] = {**m, **c, **dv}

    print(f"\n{'variant':<28}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}{'WF':>5}"
          f"{'era':>4}{'wrst5y':>8}{'yrStd':>7}{'CGx0309':>8}"
          f"{'D3y':>6}{'D5y':>6}{'D10y':>6}")
    print("-" * 96)
    for nm, m in rows.items():
        print(f"{nm:<28}{m['cagr']*100:>6.1f}%{m['sharpe']:>6.2f}"
              f"{m['max_dd']*100:>6.0f}%{m['wf_beats']:>3}/{m['wf_n']}"
              f"{m['eras_beat']:>3}/4{m['roll60_min']*100:>7.1f}%"
              f"{m['yearly_std']*100:>6.0f}%{m['cagr_ex0309']*100:>7.1f}%"
              f"{str(m['win_36']):>6}{str(m['win_60']):>6}"
              f"{str(m['win_120']):>6}")

    # --- overfit checks on the leading RC-E1 variant ---
    print("\n=== leading variant: TRUE OOS + cost ===")
    lead = "RC-E1 + adaptK (both)"

    def lead_stream(cost):
        a = stream(run_sim_v3(mg, preds, preds, spyf, mr, mp, chronos,
                              cost_bps=cost, K=2, trigger_mode="blend",
                              select_mode="ml_3plus6", regime_w=RW_D, **AK))
        b = stream(run_sim_v3(mg, preds, preds, spyf, mr, mp, chronos,
                              cost_bps=cost, K=2,
                              trigger_mode="ml_3plus6",
                              select_mode="blend", regime_w=RW_D, **AK))
        return 0.5 * a + 0.5 * b

    for c in (0.0, 10.0, 20.0, 30.0):
        r = lead_stream(c)
        m = evaluate(r, dates, spv)
        print(f"  cost {int(c):>2}bps  CAGR {m['cagr']*100:5.1f}%  "
              f"Sh {m['sharpe']:.2f}  DD {m['max_dd']*100:5.1f}%  "
              f"WF {m['wf_beats']}/{m['wf_n']}")

    for nm, r in [("E1", e1), (lead, variants[lead])]:
        de = submetrics(r, dates, spv, "2003-01-01", "2012-12-31")
        ho = submetrics(r, dates, spv, "2013-01-01", "2026-12-31")
        print(f"  {nm:<22} design {de['cagr']*100:5.1f}%/Sh{de['sharpe']:.2f}"
              f" DD{de['max_dd']*100:5.1f}% | holdout "
              f"{ho['cagr']*100:5.1f}%/Sh{ho['sharpe']:.2f} "
              f"DD{ho['max_dd']*100:5.1f}%")

    out_p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
             / "improve_phase6_rce1.json")
    out_p.write_text(json.dumps(rows, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
