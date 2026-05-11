"""K-sweep on the STAGGERED monthly-tranche DCA simulator.

Each calendar month we deposit $1, run the picker, and form a fresh K-pick
tranche on a 6-month hold. At steady state, ~6 overlapping tranches are
active. We sweep K=1..4 to find the per-tranche basket size that maximises
the money-weighted CAGR while keeping the per-tranche worst-case bounded.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, CHRONOS_FILTER_Q, CAP_PER_PICK,
    invvol_weights,
)
from experiments.monthly_dca.v5.validations.run_staggered_dca_all import (
    _run_staggered_with_picker,
)
from experiments.monthly_dca.v5.validations.run_staggered_dca import (
    xirr_monthly_deposits,
)
from experiments.monthly_dca.v5.validations.run_k_sweep import (
    make_kpick_picker,
)

RES = Path(__file__).resolve().parent / "results"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"Loaded. asofs {data.asofs[0].date()} -> {data.asofs[-1].date()}")

    rows = []
    for k in (1, 2, 3, 4):
        name = f"k{k}_stag"
        print(f"\n{'='*60}\n  STAGGERED DCA  K={k}\n{'='*60}")
        res = _run_staggered_with_picker(data, make_kpick_picker(k))
        log = res["log"]
        tranches = res["tranches"]
        n = len(log)
        cum = res["cum_deposits"]
        final = log[-1]["total_nav"]
        irr_strat = xirr_monthly_deposits(log, deposit=1.0)

        # SPY DCA
        cum_spy = 0.0
        nav_spy = 0.0
        spy_log = []
        for r in log:
            d = pd.Timestamp(r["date"])
            ret = (float(data.mret.at[d, "SPY"])
                    if (d in data.mret.index and "SPY" in data.mret.columns
                        and pd.notna(data.mret.at[d, "SPY"]))
                    else 0.0)
            nav_spy = nav_spy * (1 + ret) + 1.0
            cum_spy += 1.0
            spy_log.append({"total_nav": nav_spy})
        irr_spy = xirr_monthly_deposits(spy_log, deposit=1.0)

        closed = [t for t in tranches if t.get("status") == "exited"]
        rets = np.array([t["return_pct"] for t in closed]) if closed else np.array([])
        win_rate = float((rets > 0).mean() * 100) if len(rets) else float("nan")
        mean_ret = float(rets.mean()) if len(rets) else float("nan")
        med_ret = float(np.median(rets)) if len(rets) else float("nan")
        p10 = float(np.percentile(rets, 10)) if len(rets) else float("nan")
        p90 = float(np.percentile(rets, 90)) if len(rets) else float("nan")
        worst = float(rets.min()) if len(rets) else float("nan")

        # Compute 2024 monthly DCA edge
        tr_df = pd.DataFrame([{k_: v for k_, v in t.items()
                                 if k_ in ("entry_date", "return_pct", "status")}
                                for t in tranches])
        if len(tr_df):
            tr_df["entry_date"] = pd.to_datetime(tr_df["entry_date"])
            spy_mret = data.mret["SPY"]
            for yr in (2014, 2018, 2024, 2025):
                t_y = tr_df[(tr_df["entry_date"].dt.year == yr)
                            & (tr_df["status"] == "exited")].copy()
                spy_6m = []
                for d in t_y["entry_date"]:
                    fwd = spy_mret.loc[d:].iloc[1:7].fillna(0)
                    spy_6m.append((1 + fwd).prod() - 1)
                if len(t_y):
                    edge = t_y["return_pct"].values - np.array(spy_6m) * 100
                    edge_mean = float(np.mean(edge))
                else:
                    edge_mean = float("nan")
                rows.append if False else None  # noop guard
                # store later
                if not hasattr(main, f"_e{yr}"):
                    setattr(main, f"_e{yr}", {})
                getattr(main, f"_e{yr}")[k] = edge_mean

        print(f"  Money-weighted CAGR (strategy): {irr_strat*100:7.2f}%")
        print(f"  Money-weighted CAGR (SPY DCA):  {irr_spy*100:7.2f}%   "
              f"edge {(irr_strat-irr_spy)*100:+.2f}pp")
        print(f"  Tranche stats: n={len(rets)}, win {win_rate:.1f}%, "
              f"mean {mean_ret:.2f}%, median {med_ret:.2f}%, "
              f"p10 {p10:.2f}%, p90 {p90:.2f}%, worst {worst:.2f}%")

        rows.append({
            "k": k,
            "n_months": n,
            "deposits": cum,
            "final_nav_strategy": final,
            "multiple_strategy": final / cum,
            "mwcagr_strategy_pct": irr_strat * 100,
            "mwcagr_spy_dca_pct": irr_spy * 100,
            "edge_pp": (irr_strat - irr_spy) * 100,
            "n_tranches_closed": len(rets),
            "tranche_win_rate_pct": win_rate,
            "tranche_mean_pct": mean_ret,
            "tranche_median_pct": med_ret,
            "tranche_p10_pct": p10,
            "tranche_p90_pct": p90,
            "tranche_worst_pct": worst,
            "y2014_dca_edge_pp": getattr(main, "_e2014", {}).get(k, float("nan")),
            "y2018_dca_edge_pp": getattr(main, "_e2018", {}).get(k, float("nan")),
            "y2024_dca_edge_pp": getattr(main, "_e2024", {}).get(k, float("nan")),
            "y2025_dca_edge_pp": getattr(main, "_e2025", {}).get(k, float("nan")),
        })

        pd.DataFrame(log).to_csv(RES / f"{name}_equity.csv", index=False)
        tr_save = [{kk: vv for kk, vv in t.items()
                       if kk not in ("units", "last_marks")} for t in tranches]
        pd.DataFrame(tr_save).to_csv(RES / f"{name}_tranches.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "k_sweep_staggered_summary.csv", index=False)
    print("\n\n=== K-SWEEP STAGGERED DCA SUMMARY ===")
    cols = ["k", "mwcagr_strategy_pct", "mwcagr_spy_dca_pct", "edge_pp",
            "multiple_strategy",
            "tranche_win_rate_pct", "tranche_mean_pct", "tranche_p10_pct",
            "tranche_worst_pct",
            "y2014_dca_edge_pp", "y2018_dca_edge_pp",
            "y2024_dca_edge_pp", "y2025_dca_edge_pp"]
    print(df[cols].round(2).to_string(index=False))
    print(f"\nSaved -> {RES / 'k_sweep_staggered_summary.csv'}")


if __name__ == "__main__":
    main()
