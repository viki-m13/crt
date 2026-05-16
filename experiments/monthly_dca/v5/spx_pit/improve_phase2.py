"""Phase 2: pick-level scorer changes + best overlays, on the real pipeline.

Validates run_sim_v2 == production (trigger=consensus, select=ml_3plus6),
then sweeps:
  - select_mode in {ml_3plus6, consensus, blend}, trigger likewise
  - book-blend of the two production scorer streams (weighted)
  - light catastrophic DD-breaker on the best consistent book
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
from improve_main_strategy import (load_inputs, evaluate, spy_aligned,  # noqa
                                   overlay_ddbreak, overlay_voltarget)
from improve_sim_v2 import run_sim_v2  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp

    def sim(trig, sel):
        rl = run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                        cost_bps=10.0, K=bw.K_PICKS,
                        trigger_mode=trig, select_mode=sel)
        return rl

    # ---- production baselines via bw.run_full_sim ------------------------ #
    bw.SCORER_MODE = "consensus"
    base_rl, _, _ = bw.run_full_sim(members_g, preds, preds, spyf, mr, mp,
                                    chronos_preds=chronos, cost_bps=10.0,
                                    hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
    bw.SCORER_MODE = "ml_3plus6"
    ml_rl, _, _ = bw.run_full_sim(members_g, preds, preds, spyf, mr, mp,
                                  chronos_preds=chronos, cost_bps=10.0,
                                  hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
    bw.SCORER_MODE = "consensus"

    dates = [r["date"] for r in base_rl]
    spv = spy_aligned(pd.to_datetime(dates), mr)
    rcons = stream(base_rl)
    rml = stream(ml_rl)

    # ---- fidelity check: v2(trigger=consensus, select=ml_3plus6) == prod -- #
    repro = stream(sim("consensus", "ml_3plus6"))
    max_abs = float(np.max(np.abs(repro - rcons)))
    print(f"[fidelity] v2(cons,ml) vs prod consensus  max|Δ ret_m| = {max_abs:.2e}")
    repro_ml = stream(sim("ml_3plus6", "ml_3plus6"))
    print(f"[fidelity] v2(ml,ml)  vs prod ml_3plus6   max|Δ ret_m| = "
          f"{float(np.max(np.abs(repro_ml - rml))):.2e}")

    res = {}
    res["BASE consensus (deployed)"] = evaluate(rcons, dates, spv)
    res["ml_3plus6 (alt deployed)"] = evaluate(rml, dates, spv)

    # ---- pick-level scorer matrix --------------------------------------- #
    for trig in ("consensus", "ml_3plus6", "blend"):
        for sel in ("ml_3plus6", "consensus", "blend"):
            if (trig, sel) in (("consensus", "ml_3plus6"),
                               ("ml_3plus6", "ml_3plus6")):
                continue  # already covered as baselines
            r = stream(sim(trig, sel))
            res[f"v2 trig={trig[:4]} sel={sel[:4]}"] = evaluate(r, dates, spv)

    # ---- weighted book-blend of the two production streams -------------- #
    for w in (0.4, 0.5, 0.6):
        rb = w * rcons + (1 - w) * rml
        res[f"book {int(w*100)}/{int((1-w)*100)} cons/ml"] = evaluate(
            rb, dates, spv)

    # best book-blend = 50/50 (from phase1); apply light cat DD-breakers
    rb = 0.5 * rcons + 0.5 * rml
    for cut in (-0.45, -0.50, -0.55):
        rr = overlay_ddbreak(rb, spv, dd_cut=cut, e_low=0.35,
                             dd_resume=cut * 0.5, park="spy")
        res[f"book50 + ddbrk{int(cut*100)} spy"] = evaluate(rr, dates, spv)

    # blend-select book + DD-breaker (if blend-select is the best picker)
    rbs = stream(sim("blend", "blend"))
    for cut in (-0.45, -0.50):
        rr = overlay_ddbreak(rbs, spv, dd_cut=cut, e_low=0.35,
                             dd_resume=cut * 0.5, park="spy")
        res[f"v2blend + ddbrk{int(cut*100)} spy"] = evaluate(rr, dates, spv)

    # ---- print ---------------------------------------------------------- #
    print(f"\n{'variant':<30}{'CAGR':>8}{'Shrp':>7}{'MaxDD':>8}"
          f"{'WF':>6}{'D3y':>7}{'D5y':>7}{'D10y':>7}{'era':>6}")
    print("-" * 86)
    for nm, m in res.items():
        print(f"{nm:<30}{m['cagr']*100:>7.1f}%{m['sharpe']:>7.2f}"
              f"{m['max_dd']*100:>7.1f}%{m['wf_beats']:>3}/{m['wf_n']}"
              f"{m['dca_H36']:>7}{m['dca_H60']:>7}"
              f"{str(m['dca_H120']):>7}{m['eras_beat']:>4}/4")

    out_p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
             / "improve_phase2.json")
    out_p.write_text(json.dumps(res, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
