"""Run the staggered monthly-tranche DCA simulator across the top
momentum-anchor variants AND the original baseline. This validates
that the anchor improvements show up in the real-investor flow (deposit
$1 monthly, hold each tranche 6 months) as well as in lump-sum.

Variants tested:
  - baseline_v5
  - anchor_idio              (winner of run_momentum_variants.py)
  - anchor_chronos_gated     (close second)
  - anchor_sharpe_5y
  - anchor_uptrend
  - anchor_rs_spy
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, classify_regime_tight,
    invvol_weights, pick_v5_baseline,
    HOLD_MONTHS, K_PICKS, COST_BPS,
)
from experiments.monthly_dca.v5.validations.run_momentum_variants import (
    make_anchor_picker, _alpha_picks,
)
from experiments.monthly_dca.v5.validations.run_staggered_dca import (
    run_staggered_dca, xirr_monthly_deposits,
)

RES = Path(__file__).resolve().parent / "results"


def _run_staggered_with_picker(data: HarnessData, pick_fn,
                                 start=pd.Timestamp("2003-09-30"),
                                 end=None) -> dict:
    """Adapter: run the staggered DCA loop using an arbitrary pick_fn
    instead of the hard-coded v5 baseline picker. Mirrors run_staggered_dca
    but parameterised."""
    cf = COST_BPS / 1e4
    if end is None:
        end = data.spy_features.index.max()
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    active: list[dict] = []
    tranche_log: list[dict] = []
    cash_pool = 0.0
    cum_deposits = 0.0
    log = []
    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)

        # mark active tranches to month-end
        nav_active = 0.0
        for t in active:
            ticker_values = 0.0
            for tk, units in t["units"].items():
                px = (float(data.mp.at[m, tk])
                       if (tk in data.mp.columns and m in data.mp.index
                           and pd.notna(data.mp.at[m, tk]))
                       else None)
                if px is not None:
                    ticker_values += units * px
                    t["last_marks"][tk] = units * px
                else:
                    ticker_values += t["last_marks"].get(tk, 0.0)
            t["nav"] = ticker_values
            nav_active += ticker_values

        matured = [t for t in active if (i - t["entry_idx"]) >= HOLD_MONTHS]
        proceeds = 0.0
        for t in matured:
            t["exit_date"] = str(m.date())
            t["exit_nav"] = t["nav"]
            t["return_pct"] = (t["nav"] / t["initial_capital"] - 1.0) * 100
            t["status"] = "exited"
            tranche_log.append(t)
            proceeds += t["nav"]
        active = [t for t in active if (i - t["entry_idx"]) < HOLD_MONTHS]

        capital = 1.0 + proceeds + cash_pool
        cum_deposits += 1.0
        cash_pool = 0.0

        eligible = data.members_g.get(m, set()) - {"SPY", "QQQ", "IWM",
                                                      "VTI", "RSP", "DIA"}
        if regime == "crash":
            cash_pool = capital
            picks, weights = [], []
        else:
            picks, weights = pick_fn(m, eligible, data, regime)
        if not picks:
            cash_pool = capital
        else:
            cap_after_cost = capital * (1 - cf)
            units = {}
            marks = {}
            for tk, w in zip(picks, weights):
                px = (float(data.mp.at[m, tk])
                       if (tk in data.mp.columns and m in data.mp.index
                           and pd.notna(data.mp.at[m, tk]))
                       else None)
                if px is None or px <= 0:
                    units[tk] = 0.0
                else:
                    units[tk] = (cap_after_cost * w) / px
                marks[tk] = cap_after_cost * w
            active.append({
                "entry_idx": i,
                "entry_date": str(m.date()),
                "regime": regime,
                "picks": ",".join(picks),
                "weights": ",".join(f"{w:.3f}" for w in weights),
                "initial_capital": capital,
                "units": units,
                "last_marks": marks,
                "nav": cap_after_cost,
                "status": "open",
            })

        total_active_nav = sum(t["nav"] for t in active)
        total_nav = total_active_nav + cash_pool
        log.append({"date": str(m.date()), "regime": regime,
                     "cum_deposits": cum_deposits, "active_tranches": len(active),
                     "cash_pool": cash_pool, "total_nav": total_nav,
                     "current_picks": ",".join(picks),
                     "current_weights": ",".join(f"{w:.3f}" for w in weights)})

    final_m = asofs[-1]
    for t in active:
        t["exit_date"] = str(final_m.date()) + " (open)"
        t["exit_nav"] = t["nav"]
        t["return_pct"] = (t["nav"] / t["initial_capital"] - 1.0) * 100
        t["status"] = "open"
        tranche_log.append(t)
    return {"log": log, "tranches": tranche_log, "n_months": len(asofs),
             "cum_deposits": cum_deposits}


VARIANTS = [
    ("baseline_v5",            pick_v5_baseline),
    ("anchor_idio",            make_anchor_picker("idio_mom_12_1")),
    ("anchor_chronos_gated",   make_anchor_picker("mom_12_1", gate_by_chronos=True)),
    ("anchor_sharpe_5y",       make_anchor_picker("sharpe_5y")),
    ("anchor_uptrend",         make_anchor_picker(
                                    "mom_12_1",
                                    filter_fn=lambda r: r.get("d_sma200", -1) > 0
                                                          and r.get("mom_3y", -1) > 0)),
    ("anchor_rs_spy",          make_anchor_picker("rs_12m_spy")),
    ("anchor_mom12",           make_anchor_picker("mom_12_1")),
]


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"\nLoaded. asofs {data.asofs[0].date()} -> {data.asofs[-1].date()}")

    rows = []
    for name, pick_fn in VARIANTS:
        print(f"\n{'='*60}\n  STAGGERED DCA: {name}\n{'='*60}")
        res = _run_staggered_with_picker(data, pick_fn)
        log = res["log"]
        tranches = res["tranches"]
        n = len(log)
        cum = res["cum_deposits"]
        final = log[-1]["total_nav"]
        irr_strat = xirr_monthly_deposits(log, deposit=1.0)
        # SPY DCA (same as before)
        spy_log = []
        cum_spy = 0.0
        nav_spy = 0.0
        for r in log:
            d = pd.Timestamp(r["date"])
            ret = (float(data.mret.at[d, "SPY"])
                    if (d in data.mret.index and "SPY" in data.mret.columns
                        and pd.notna(data.mret.at[d, "SPY"]))
                    else 0.0)
            nav_spy = nav_spy * (1 + ret) + 1.0
            cum_spy += 1.0
            spy_log.append({"total_nav": nav_spy})
        spy_final = spy_log[-1]["total_nav"]
        irr_spy = xirr_monthly_deposits(spy_log, deposit=1.0)

        closed = [t for t in tranches if t.get("status") == "exited"]
        rets = np.array([t["return_pct"] for t in closed]) if closed else np.array([])
        win_rate = float((rets > 0).mean() * 100) if len(rets) else float("nan")
        mean_ret = float(rets.mean()) if len(rets) else float("nan")
        med_ret = float(np.median(rets)) if len(rets) else float("nan")
        p10 = float(np.percentile(rets, 10)) if len(rets) else float("nan")
        p90 = float(np.percentile(rets, 90)) if len(rets) else float("nan")
        worst = float(rets.min()) if len(rets) else float("nan")

        print(f"  n_months: {n}  deposits: ${cum:.0f}  final NAV: ${final:,.0f}  "
              f"multiple {final/cum:.2f}x")
        print(f"  Money-weighted CAGR (strategy): {irr_strat*100:7.2f}%")
        print(f"  Money-weighted CAGR (SPY DCA):  {irr_spy*100:7.2f}%   "
              f"edge {(irr_strat-irr_spy)*100:+.2f}pp")
        print(f"  Tranche stats: n={len(rets)}, win {win_rate:.1f}%, "
              f"mean {mean_ret:.2f}%, median {med_ret:.2f}%, "
              f"p10 {p10:.2f}%, p90 {p90:.2f}%, worst {worst:.2f}%")

        # Save per-variant files
        pd.DataFrame(log).to_csv(RES / f"staggered_{name}_equity.csv", index=False)
        tr_save = [{k: v for k, v in t.items() if k not in ("units", "last_marks")}
                    for t in tranches]
        pd.DataFrame(tr_save).to_csv(RES / f"staggered_{name}_tranches.csv", index=False)

        rows.append({
            "variant": name,
            "n_months": n,
            "deposits": cum,
            "final_nav_strategy": final,
            "final_nav_spy_dca": spy_final,
            "multiple_strategy": final / cum,
            "multiple_spy_dca": spy_final / cum,
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
        })

    summary = pd.DataFrame(rows).sort_values("mwcagr_strategy_pct", ascending=False)
    summary.to_csv(RES / "SUMMARY_staggered_dca.csv", index=False)
    print(f"\n\nSUMMARY (sorted by mw-CAGR):")
    print(summary.to_string(index=False))
    print(f"\nSaved → {RES / 'SUMMARY_staggered_dca.csv'}")


if __name__ == "__main__":
    main()
