"""Quality-sleeve blend for E2 -- frozen-holdout + weight-plateau validation
on the dual-benchmark DCA objective.

Finding (CONSISTENCY_DCA_FINDINGS.md): off-switches and staggering both fail
to raise the dual-benchmark win-rate (they forfeit the recovery / dilute the
alpha). Blending E2 with the price-only QUALITY sleeve (S&P-500, no ETF --
respects constraints) is the only within-constraint lever that keeps win-rate
flat while robustly improving the worst-case window and drawdown.

Quality sleeve stream is the audited `quality` column of
`augmented/second_sleeve_streams.csv` (probe_second_sleeve.py: quality_score_5y
rank, K=2, invvol cap 0.40, rule-based rebalance, no Chronos, walk-forward).
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
from improve_consistency_dca import (build_e2, bench_aligned, dual_dca,  # noqa
                                     lump_metrics, DESIGN_END, HOLDOUT_START)
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa


def main():
    inp = load_inputs()
    _, _, _, mr, _, _ = inp
    d0, rE2 = build_e2(inp)
    base = bw.CACHE / "v2" / "sp500_pit" / "augmented"
    ss = pd.read_csv(base / "second_sleeve_streams.csv", parse_dates=["date"])
    m = pd.DataFrame({"date": pd.to_datetime(d0), "rE2": rE2}).merge(
        ss, on="date", how="inner")
    dts = list(m.date)
    spv = bench_aligned(dts, mr, "SPY")
    qqv = bench_aligned(dts, mr, "QQQ")

    def block(r, lo=None, hi=None):
        a = dual_dca(r, spv, qqv, dts, lo=lo, hi=hi)
        return {H: dict(win_both=a[H]["win_both"],
                        worst_vs_spy=a[H]["worst_vs_spy"],
                        worst_vs_qqq=a[H]["worst_vs_qqq"]) for H in (12, 36, 60, 120)}

    out = {}
    print("=== QUALITY-BLEND plateau + FROZEN HOLDOUT (design<=2015 | holdout>=2016) ===")
    print(f"{'w':>5} {'CAGR':>6} {'DD':>7} {'Sh':>5} | "
          f"{'ALL 1/3/5/10':>14} | {'HOLD 1/3/5':>11} | {'HOLD worst1y':>12}")
    for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        r = (1 - w) * m.rE2.values + w * m["quality"].values
        f = lump_metrics(r, dts)
        A, H = block(r), block(r, lo=HOLDOUT_START)
        out[f"w={w:.2f}"] = dict(full=f, all=A, holdout=H,
                                 design=block(r, hi=DESIGN_END))
        print(f"{w:>5.2f} {f['cagr']*100:5.0f}% {f['max_dd']*100:6.1f}% "
              f"{f['sharpe']:>5.2f} | "
              f"{A[12]['win_both']*100:3.0f}/{A[36]['win_both']*100:3.0f}/"
              f"{A[60]['win_both']*100:3.0f}/{A[120]['win_both']*100:3.0f} | "
              f"{H[12]['win_both']*100:3.0f}/{H[36]['win_both']*100:3.0f}/"
              f"{H[60]['win_both']*100:3.0f} | "
              f"{H[12]['worst_vs_spy']:.2f}/{H[12]['worst_vs_qqq']:.2f}")

    p = base / "improve_consistency_qualityblend.json"
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
