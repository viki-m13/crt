"""Phase 4b: overfit gauntlet on E1 = WIN1 (+) WIN2 portfolio.

  1. mix-weight plateau (w_win1 0.3..0.7) — must be a plateau
  2. TRUE OOS design 2003-2012 | untouched holdout 2013-2026
  3. cost sensitivity (0/10/20/30 bps, both sleeves rebuilt)
  4. MC synthetic-delisting on the COMBINED two-sleeve book
  5. consistency metrics on design vs holdout separately
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
from improve_phase3 import submetrics  # noqa
from improve_phase4 import consistency  # noqa


def rl_stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def mc_delist_blend(rl1, rl2, mr, alpha, w=0.5, n_iter=25, seed=0):
    """Per-iteration: apply independent per-pick monthly hazard to BOTH
    sleeves, blend their equity curves w/(1-w), take terminal CAGR.
    Median over iters."""
    rng = np.random.default_rng(seed)
    mr_idx = mr.index

    def one(rl, rng):
        eq, dead, cur = 1.0, set(), None
        for row in rl:
            bid = row.get("basket_id")
            if bid != cur:
                dead, cur = set(), bid
            picks = row.get("picks") or []
            if not picks:
                continue
            m = pd.Timestamp(row["date"])
            pos = mr_idx.searchsorted(m)
            if pos + 1 >= len(mr_idx):
                continue
            nxt = mr_idx[pos + 1]
            rr = []
            for tk in picks:
                if tk in dead:
                    rr.append(-1.0); continue
                if rng.random() < alpha / 12.0:
                    dead.add(tk); rr.append(-1.0); continue
                v = mr.at[nxt, tk] if (tk in mr.columns and nxt in mr_idx) else 0.0
                rr.append(0.0 if pd.isna(v) else float(v))
            eq *= (1 + float(np.mean(rr)))
        return eq

    out = []
    n = len(rl1)
    for _ in range(n_iter):
        e1 = one(rl1, rng)
        e2 = one(rl2, rng)
        eb = w * e1 + (1 - w) * e2
        out.append(eb ** (12.0 / n) - 1.0)
    return round(float(np.median(out)), 4)


def main():
    inp = load_inputs()
    mg, preds, spyf, mr, mp, chronos = inp

    def simrl(trig, sel, cost=10.0):
        v2.BLEND_W = 0.5
        return run_sim_v2(mg, preds, preds, spyf, mr, mp, chronos,
                          cost_bps=cost, K=bw.K_PICKS,
                          trigger_mode=trig, select_mode=sel)

    rl1 = simrl("blend", "ml_3plus6")     # WIN1
    rl2 = simrl("ml_3plus6", "blend")     # WIN2
    dates = [r["date"] for r in rl1]
    spv = spy_aligned(pd.to_datetime(dates), mr)
    w1, w2 = rl_stream(rl1), rl_stream(rl2)
    out = {}

    def report(tag, r):
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        out[tag] = {**m, **c}
        print(f"  {tag:<22} CAGR {m['cagr']*100:5.1f}% Sh {m['sharpe']:.2f} "
              f"DD {m['max_dd']*100:5.0f}% WF {m['wf_beats']}/{m['wf_n']} "
              f"era {m['eras_beat']}/4 r60min {c['roll60_min']*100:5.1f}% "
              f"r36beat {c['roll36_beat']*100:.0f}% CAGRx {c['cagr_ex0309']*100:.0f}%")

    print("=== baselines + E1 mix plateau ===")
    report("WIN1", w1)
    report("WIN2", w2)
    for wa in (0.3, 0.4, 0.5, 0.6, 0.7):
        report(f"E1 w1={wa:.1f}", wa * w1 + (1 - wa) * w2)

    print("\n=== TRUE OOS (design 2003-12 | holdout 2013-26) ===")
    e1 = 0.5 * w1 + 0.5 * w2
    for nm, r in (("WIN1", w1), ("WIN2", w2), ("E1 50/50", e1)):
        d = submetrics(r, dates, spv, "2003-01-01", "2012-12-31")
        h = submetrics(r, dates, spv, "2013-01-01", "2026-12-31")
        out.setdefault("oos", {})[nm] = {"design": d, "holdout": h}
        print(f"  {nm:<9} design CAGR {d['cagr']*100:5.1f}% Sh {d['sharpe']:.2f}"
              f" | holdout CAGR {h['cagr']*100:5.1f}% Sh {h['sharpe']:.2f}"
              f" DD {h['max_dd']*100:.0f}%")

    print("\n=== cost sensitivity (E1 50/50, both sleeves rebuilt) ===")
    for c in (0.0, 10.0, 20.0, 30.0):
        a = rl_stream(simrl("blend", "ml_3plus6", cost=c))
        b = rl_stream(simrl("ml_3plus6", "blend", cost=c))
        m = evaluate(0.5 * a + 0.5 * b, dates, spv)
        out.setdefault("cost", {})[f"{int(c)}bps"] = m
        print(f"  {int(c):>3} bps  CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  DD {m['max_dd']*100:5.0f}%  WF {m['wf_beats']}/{m['wf_n']}")

    print("\n=== MC synthetic-delisting (combined two-sleeve book) ===")
    for a in (0.0, 0.04, 0.08):
        wmc = mc_delist_blend(rl1, rl2, mr, a)
        out.setdefault("mc_delist", {})[f"a{int(a*100)}"] = wmc
        print(f"  alpha {int(a*100):>2}%  E1 median CAGR {wmc*100:6.1f}%")

    out_p = bw.CACHE / "v2" / "sp500_pit" / "augmented" / "improve_phase4b.json"
    out_p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
