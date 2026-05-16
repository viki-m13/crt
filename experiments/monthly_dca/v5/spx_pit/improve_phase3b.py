"""Phase 3b: same gauntlet on the second candidate
`trigger=ml_3plus6, select=blend` (the best DRAWDOWN variant from phase 2),
side-by-side with the phase-3 winner `trigger=blend, select=ml_3plus6`
and the deployed consensus baseline.
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
from improve_phase3 import stream, submetrics, mc_delist  # noqa


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp

    def sim(trig, sel, cost=10.0):
        return run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                          cost_bps=cost, K=bw.K_PICKS,
                          trigger_mode=trig, select_mode=sel)

    bw.SCORER_MODE = "consensus"
    base_rl, _, _ = bw.run_full_sim(members_g, preds, preds, spyf, mr, mp,
                                    chronos_preds=chronos, cost_bps=10.0,
                                    hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
    dates = [r["date"] for r in base_rl]
    spv = spy_aligned(pd.to_datetime(dates), mr)

    cands = {
        "deployed consensus": base_rl,
        "WIN1 trig=blend/sel=ml": sim("blend", "ml_3plus6"),
        "WIN2 trig=ml/sel=blend": sim("ml_3plus6", "blend"),
    }
    out = {}
    print(f"{'variant':<26}{'CAGR':>7}{'Sh':>6}{'MaxDD':>8}{'WF':>6}"
          f"{'D3y':>7}{'D5y':>7}{'D10y':>7}{'era':>5}")
    print("-" * 78)
    for nm, rl in cands.items():
        r = stream(rl)
        m = evaluate(r, dates, spv)
        out[nm] = m
        print(f"{nm:<26}{m['cagr']*100:>6.1f}%{m['sharpe']:>6.2f}"
              f"{m['max_dd']*100:>7.1f}%{m['wf_beats']:>3}/{m['wf_n']}"
              f"{m['dca_H36']:>7}{m['dca_H60']:>7}{str(m['dca_H120']):>7}"
              f"{m['eras_beat']:>4}/4")

    print("\n=== TRUE OOS (design 2003-12 | holdout 2013-26) ===")
    for nm, rl in cands.items():
        r = stream(rl)
        d = submetrics(r, dates, spv, "2003-01-01", "2012-12-31")
        h = submetrics(r, dates, spv, "2013-01-01", "2026-12-31")
        out.setdefault("oos", {})[nm] = {"design": d, "holdout": h}
        print(f"  {nm:<26} design CAGR {d['cagr']*100:5.1f}% Sh {d['sharpe']:.2f}"
              f" | holdout CAGR {h['cagr']*100:5.1f}% Sh {h['sharpe']:.2f}"
              f" DD {h['max_dd']*100:.0f}%")

    print("\n=== MC synthetic-delisting median CAGR ===")
    for a in (0.0, 0.04, 0.08):
        row = {nm: mc_delist(rl, mr, a) for nm, rl in cands.items()}
        out.setdefault("mc_delist", {})[f"a{int(a*100)}"] = row
        print(f"  a{int(a*100):>2}%  " + "  ".join(
            f"{nm.split()[0]} {v*100:5.1f}%" for nm, v in row.items()))

    print("\n=== WIN2 blend-weight plateau ===")
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        v2.BLEND_W = w
        m = evaluate(stream(sim("ml_3plus6", "blend")), dates, spv)
        print(f"  w={w:.1f}  CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}"
              f"  era {m['eras_beat']}/4")
    v2.BLEND_W = 0.5

    out_p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
             / "improve_phase3b.json")
    out_p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
