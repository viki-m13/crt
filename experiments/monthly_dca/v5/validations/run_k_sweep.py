"""K-sweep on v5: hold K=1..5 picks for 6 months, semi-annual rebalance.

Same Chronos-gated ml_3plus6 scorer, same inv-vol cap-0.40 weighting, same
PIT S&P 500 universe — only K varies. Run on the look-ahead-fixed harness.

Output: results/k_sweep_summary.csv + per-K equity logs.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    CHRONOS_FILTER_Q, CAP_PER_PICK, HarnessData,
    load_all, evaluate, invvol_weights,
)

RES = Path(__file__).resolve().parent / "results"


def make_kpick_picker(k: int):
    """Return a pick_fn that selects top-K by GBM score from Chronos-gated cohort."""
    def pick(asof, eligible, data: HarnessData, regime):
        sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
        sub = sub[sub["ticker"].isin(eligible)]
        if len(sub) == 0:
            return [], []
        sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
        ch = data.chronos.get(asof, {})
        if ch:
            sub["chr"] = sub["ticker"].map(ch)
            sub["chr_rk"] = sub["chr"].rank(pct=True)
            sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
        if len(sub) < k:
            return [], []
        top = sub.sort_values("score", ascending=False).head(k)
        picks = top["ticker"].tolist()
        weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
        return picks, list(weights)
    return pick


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"Loaded. asofs {data.asofs[0].date()} -> {data.asofs[-1].date()}")

    summary = []
    for k in (1, 2, 3, 4, 5):
        name = f"k{k}_lump"
        print(f"\n{'='*60}\n  K = {k}  (top-{k} GBM+Chronos, hold 6m, "
              f"inv-vol cap {CAP_PER_PICK})\n{'='*60}")
        res = evaluate(data, make_kpick_picker(k), name)
        log = res.pop("log")
        with open(RES / f"{name}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        pd.DataFrame(log).to_csv(RES / f"{name}_equity.csv", index=False)
        print(f"  Lump-sum CAGR: {res['cagr_lump_sum_pct']:7.2f}%  "
              f"DCA: {res['cagr_dca_pct']:7.2f}%")
        print(f"  WF: mean={res['wf_mean_pct']:.2f}% "
              f"min={res['wf_min_pct']:.2f}% max={res['wf_max_pct']:.2f}% "
              f"edge={res['wf_mean_edge_pp']:+.2f}pp  "
              f"beat_spy={res['wf_n_beat_spy']}/10  "
              f"positive={res['wf_n_positive']}/10")
        print(f"  Sharpe {res['sharpe']:.2f}  MaxDD {res['max_dd_pct']:.1f}%")
        lag = {y["year"]: y["edge_pp"] for y in res["year_by_year"]
                if y["year"] in (2014, 2018, 2024, 2025)}
        print(f"  Lagging-year edges (pp):  "
              f"2014={lag.get(2014, 0):+5.1f}  "
              f"2018={lag.get(2018, 0):+5.1f}  "
              f"2024={lag.get(2024, 0):+5.1f}  "
              f"2025={lag.get(2025, 0):+5.1f}")
        summary.append({
            "k": k,
            "cagr_lump_sum_pct": res["cagr_lump_sum_pct"],
            "cagr_dca_pct": res["cagr_dca_pct"],
            "edge_lump_sum_pp": res["edge_lump_sum_pp"],
            "wf_mean_pct": res["wf_mean_pct"],
            "wf_min_pct": res["wf_min_pct"],
            "wf_max_pct": res["wf_max_pct"],
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
    df.to_csv(RES / "k_sweep_summary.csv", index=False)
    print("\n\n=== K-SWEEP SUMMARY ===")
    cols = ["k", "cagr_lump_sum_pct", "cagr_dca_pct", "wf_mean_pct",
            "wf_min_pct", "wf_n_beat_spy", "wf_n_positive",
            "sharpe", "max_dd_pct",
            "y2014_edge_pp", "y2018_edge_pp", "y2024_edge_pp", "y2025_edge_pp"]
    print(df[cols].round(2).to_string(index=False))
    print(f"\nSaved -> {RES / 'k_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
