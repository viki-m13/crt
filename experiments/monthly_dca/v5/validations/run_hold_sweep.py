"""Rebalance-frequency sweep: hold periods H = 1, 2, 3, 4, 6, 9, 12 months.

Tests the hypothesis that 2024's lump-sum lag is rebalance-date luck — if so,
shorter holds (more frequent rebalance dates per year) should reduce
single-date risk and recover the 2024 picking-quality edge.

All other strategy parameters identical to v5 production: K=3 by GBM
(pred_3m + pred_6m)/2 scorer + Chronos p70 q=0.45 gate, inv-vol cap=0.40,
PIT S&P 500 universe, tight regime gate, 10 bps cost per rebalance.

Output: results/hold_sweep_summary.csv + per-H equity logs.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    load_all, evaluate, pick_v5_baseline,
)

RES = Path(__file__).resolve().parent / "results"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"Loaded. asofs {data.asofs[0].date()} -> {data.asofs[-1].date()}")

    summary = []
    for h in (1, 2, 3, 4, 6, 9, 12):
        name = f"H{h}m"
        print(f"\n{'='*60}\n  HOLD = {h} months "
              f"(rebalance ~{12/h:.0f}x/year, ~{12/h*23:.0f} entries over 23y)"
              f"\n{'='*60}")
        res = evaluate(data, pick_v5_baseline, name, hold_months=h)
        log = res.pop("log")
        with open(RES / f"{name}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        pd.DataFrame(log).to_csv(RES / f"{name}_equity.csv", index=False)
        print(f"  Lump-sum CAGR: {res['cagr_lump_sum_pct']:7.2f}%  "
              f"DCA: {res['cagr_dca_pct']:7.2f}%")
        print(f"  WF: mean={res['wf_mean_pct']:.2f}% "
              f"min={res['wf_min_pct']:.2f}% "
              f"edge={res['wf_mean_edge_pp']:+.2f}pp  "
              f"beat_spy={res['wf_n_beat_spy']}/10")
        print(f"  Sharpe {res['sharpe']:.2f}  MaxDD {res['max_dd_pct']:.1f}%")
        lag = {y["year"]: y["edge_pp"] for y in res["year_by_year"]
                if y["year"] in (2014, 2018, 2024, 2025)}
        print(f"  Lagging-year edges (pp):  "
              f"2014={lag.get(2014, 0):+5.1f}  "
              f"2018={lag.get(2018, 0):+5.1f}  "
              f"2024={lag.get(2024, 0):+5.1f}  "
              f"2025={lag.get(2025, 0):+5.1f}")
        summary.append({
            "hold_months": h,
            "rebals_per_year": 12 / h,
            "cagr_lump_sum_pct": res["cagr_lump_sum_pct"],
            "cagr_dca_pct": res["cagr_dca_pct"],
            "wf_mean_pct": res["wf_mean_pct"],
            "wf_min_pct": res["wf_min_pct"],
            "wf_mean_edge_pp": res["wf_mean_edge_pp"],
            "wf_n_beat_spy": res["wf_n_beat_spy"],
            "wf_n_positive": res["wf_n_positive"],
            "sharpe": res["sharpe"],
            "max_dd_pct": res["max_dd_pct"],
            "y2014_edge_pp": lag.get(2014, 0),
            "y2018_edge_pp": lag.get(2018, 0),
            "y2024_edge_pp": lag.get(2024, 0),
            "y2025_edge_pp": lag.get(2025, 0),
        })

    df = pd.DataFrame(summary)
    df.to_csv(RES / "hold_sweep_summary.csv", index=False)
    print("\n\n=== HOLD-PERIOD SWEEP SUMMARY (K=3, fixed scorer) ===")
    cols = ["hold_months", "rebals_per_year", "cagr_lump_sum_pct",
            "wf_mean_pct", "wf_min_pct", "wf_n_beat_spy",
            "sharpe", "max_dd_pct",
            "y2014_edge_pp", "y2018_edge_pp",
            "y2024_edge_pp", "y2025_edge_pp"]
    print(df[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
