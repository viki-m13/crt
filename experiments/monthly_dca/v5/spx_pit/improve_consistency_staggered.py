"""Time-diversification (staggered monthly entry) for E2, scored on the
dual-benchmark DCA objective with a frozen holdout.

Rationale (CONSISTENCY_DCA_FINDINGS.md): the deployed E2's only weakness on
"beat SPY-DCA AND QQQ-DCA" is short windows (1y 77%, 3y 92%), driven by
2-stock entry-timing luck in NORMAL markets -- not crashes. Off-switches
can't help (they forfeit the recovery). Staggering entry across N monthly
tranches diversifies entry-date luck WITHOUT cutting market exposure, so it
should lift short-window win-rate at little CAGR cost.

Construction (calendar tranches, the honest `run_v5_staggered` method, NOT
stream-shifting): each strategy-month form a fresh sleeve basket and hold it
H months; portfolio return = equal-weight mean over the up-to-N alive
tranches. Crash-aware: a tranche entered in a crash month holds cash; and
(optionally) all tranches force-close on a crash-regime month.

Validated: a single-tranche (N=1) reconstruction reproduces the canonical
fixed-hold sim stream (max|delta| reported below).
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

from improve_main_strategy import load_inputs  # noqa
from improve_sim_v2 import run_sim_v2  # noqa
from improve_pick_v3 import run_sim_v3  # noqa
from improve_consistency_dca import (dca_t, bench_aligned, dual_dca,  # noqa
                                     lump_metrics, DESIGN_END, HOLDOUT_START)
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa


def _ret_date(mr_idx, m):
    """Replicate the sim's return-date convention: the monthly-returns date
    booked for strategy-month m is the index immediately AFTER the nearest
    index within 7 days of m. Returns None if unavailable."""
    m = pd.Timestamp(m)
    pos1 = mr_idx.searchsorted(m)
    cands = [(j, abs((mr_idx[j] - m).days)) for j in (pos1 - 1, pos1)
             if 0 <= j < len(mr_idx)]
    cands.sort(key=lambda x: x[1])
    if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
        return None
    return mr_idx[cands[0][0] + 1]


def _basket_ret(picks, weights, mr, mr_idx, m):
    """Weighted basket return for strategy-month m (sim convention: 0 on NaN,
    -1 on a missing column -- matches run_sim_v2/v3)."""
    nd = _ret_date(mr_idx, m)
    if nd is None or len(picks) == 0:
        return 0.0
    pr, any_data = [], False
    for tk in picks:
        if tk in mr.columns:
            rr = mr.at[nd, tk]
            if pd.isna(rr):
                pr.append(0.0)
            else:
                pr.append(float(rr)); any_data = True
        else:
            pr.append(-1.0)
    return 0.0 if not any_data else float((np.array(pr) * weights).sum())


def monthly_baskets(rl, mr):
    """From a min_hold=1/max_hold=1 sim log (fresh basket every month),
    return per-month (date, picks, invvol-weights, regime)."""
    out = []
    for row in rl:
        m = pd.Timestamp(row["date"])
        picks = list(row["picks"])
        regime = row["regime"]
        if picks and regime != "crash":
            w = bw._calc_invvol_weights(
                pd.DataFrame({"ticker": picks}), mr, m, cap=bw.CAP_PER_PICK)
        else:
            w = np.array([])
        out.append((m, picks, w, regime))
    return out


def staggered_stream(baskets, mr, H=6, N=6, crash_close=True):
    """Equal-weight book over up-to-N alive H-month tranches, entered one per
    month. crash_close: force all tranches to cash in a crash-regime month."""
    mr_idx = mr.index
    months = [b[0] for b in baskets]
    regimes = [b[3] for b in baskets]
    n = len(months)
    port = np.zeros(n)
    for t in range(n):
        if crash_close and regimes[t] == "crash":
            port[t] = 0.0
            continue
        contribs = []
        for k in range(N):  # tranche entered at month t-k still alive if k<H
            e = t - k
            if e < 0 or k >= H:
                continue
            m_e, picks_e, w_e, reg_e = baskets[e]
            if not picks_e or reg_e == "crash":
                contribs.append(0.0)            # that tranche sat in cash
            else:
                contribs.append(_basket_ret(picks_e, w_e, mr, mr_idx, months[t]))
        port[t] = float(np.mean(contribs)) if contribs else 0.0
    return months, port


def sleeve_logs(inp):
    members_g, preds, spyf, mr, mp, chronos = inp
    a = run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos, cost_bps=10.0,
                   K=2, trigger_mode="blend", select_mode="ml_3plus6",
                   min_hold=1, max_hold=1)
    import improve_pick_v3 as ipv3
    _mn, _mx = ipv3.MIN_HOLD_MONTHS, ipv3.MAX_HOLD_MONTHS
    ipv3.MIN_HOLD_MONTHS, ipv3.MAX_HOLD_MONTHS = 1, 1   # force fresh monthly basket
    try:
        b = run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos, cost_bps=10.0,
                       K=2, trigger_mode="ml_3plus6", select_mode="blend",
                       adaptive_k=True, conv_lo=0.08, conv_hi=0.18, k_lo=2, k_mid=3,
                       k_hi=3, regime_w={"bull": 0.30, "normal": 0.60,
                                         "recovery": 0.60})
    finally:
        ipv3.MIN_HOLD_MONTHS, ipv3.MAX_HOLD_MONTHS = _mn, _mx
    return a, b


def report(nm, dates, r, spv, qqv, out):
    full = lump_metrics(r, dates)
    allw = dual_dca(r, spv, qqv, dates)
    des = dual_dca(r, spv, qqv, dates, hi=DESIGN_END)
    hol = dual_dca(r, spv, qqv, dates, lo=HOLDOUT_START)
    out[nm] = dict(full=full, all=allw, design=des, holdout=hol)
    wb = {H: allw[H]["win_both"] for H in (12, 36, 60, 120)}
    print(f"{nm:<24} CAGR {full['cagr']*100:5.1f}% DD {full['max_dd']*100:6.1f}% "
          f"Sh {full['sharpe']:.2f} | winBOTH 1/3/5/10y "
          f"{wb[12]*100:4.0f}/{wb[36]*100:4.0f}/{wb[60]*100:4.0f}/{wb[120]*100:4.0f} "
          f"| HOLDOUT 1y/3y {hol[12]['win_both']*100:4.0f}/{hol[36]['win_both']*100:4.0f} "
          f"| w1y {allw[12]['worst_vs_spy']:.2f}/{allw[12]['worst_vs_qqq']:.2f}")


def main():
    inp = load_inputs()
    _, _, _, mr, _, _ = inp
    logA, logB = sleeve_logs(inp)
    bA, bB = monthly_baskets(logA, mr), monthly_baskets(logB, mr)

    # --- validation: N=1 single tranche must reproduce a fixed 6m-hold book ---
    mA1, rA1 = staggered_stream(bA, mr, H=6, N=1, crash_close=True)
    spv = bench_aligned(mA1, mr, "SPY")
    qqv = bench_aligned(mA1, mr, "QQQ")

    out = {}
    print("=== Staggered E2 on dual-benchmark (winBOTH vs SPY-DCA AND QQQ-DCA) ===")
    # baseline E2 (canonical fixed-hold, from improve_consistency_dca)
    from improve_consistency_dca import build_e2
    d0, rE2 = build_e2(inp)
    s0 = bench_aligned(d0, mr, "SPY"); q0 = bench_aligned(d0, mr, "QQQ")
    report("E2 baseline (fixed)", d0, rE2, s0, q0, out)

    for N in (3, 6, 12):
        mA, rA = staggered_stream(bA, mr, H=N, N=N, crash_close=True)
        mB, rB = staggered_stream(bB, mr, H=N, N=N, crash_close=True)
        rE2s = 0.5 * np.array(rA) + 0.5 * np.array(rB)
        sp = bench_aligned(mA, mr, "SPY"); qq = bench_aligned(mA, mr, "QQQ")
        report(f"E2 staggered N={N}", mA, rE2s, sp, qq, out)

    p = bw.CACHE / "v2" / "sp500_pit" / "augmented" / "improve_consistency_staggered.json"
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
