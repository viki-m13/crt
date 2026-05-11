"""Staggered monthly-tranche DCA.

Concept: every month-end the user contributes fresh cash $C and the strategy
forms a NEW 3-stock basket using that month's v5 picks (GBM + Chronos +
inv-vol cap). That tranche is held for exactly 6 months, then sold and the
proceeds are recycled into the current month's tranche.

So at any time the user has up to 6 active tranches (one started each of the
last 6 months). This is the "proper DCA" approach a retail investor would use
to deploy fresh paycheck cash each month while still getting 6-month holds.

Outputs:
  experiments/monthly_dca/v5/validations/results/staggered_dca.json
  experiments/monthly_dca/v5/validations/results/staggered_dca_equity.csv
  experiments/monthly_dca/v5/validations/results/staggered_dca_tranches.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all,
    pick_v5_baseline, classify_regime_tight,
    invvol_weights, lump_sum_cagr, dca_cagr, sharpe_ann, max_dd,
    wf_aggregate, year_by_year,
    CHRONOS_FILTER_Q, CAP_PER_PICK, HOLD_MONTHS, K_PICKS, COST_BPS,
    WALK_FORWARD_SPLITS,
)

RES = Path(__file__).resolve().parent / "results"


def run_staggered_dca(data: HarnessData,
                       start: pd.Timestamp = pd.Timestamp("2003-09-30"),
                       end: pd.Timestamp = None,
                       hold_months: int = HOLD_MONTHS,
                       deposit: float = 1.0,
                       cost_bps: float = COST_BPS) -> dict:
    """Each month-end, deposit `deposit` dollars into a NEW v5 basket
    (3 picks, inv-vol weighted). Hold each tranche for `hold_months`,
    then realize the value and recycle into the current-month tranche.

    Returns dict with per-month aggregate equity, list of tranche records,
    and a notional "monthly return" series defined as
      ret_m = (NAV_t - NAV_{t-1} - deposit_t) / max(NAV_{t-1} + deposit_t, eps)
    so it represents the period-over-period return on the deployed capital.
    """
    cf = cost_bps / 1e4
    if end is None:
        end = data.spy_features.index.max()
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    # Active tranches: list of dicts with entry_date, picks, weights, units (per ticker)
    active: list[dict] = []
    tranche_log: list[dict] = []
    total_invested = 0.0   # cumulative deposits
    cash_pool = 0.0        # uninvested cash (only used if a tranche fails to form, deposit stays in cash)
    cum_deposits = 0.0

    log = []
    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)

        # 1) Mark active tranches to current month's prices to get NAV_before_action
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
                else:
                    # missing price → use previous mark (no change for this ticker)
                    ticker_values += t["last_marks"].get(tk, 0.0)
                t["last_marks"][tk] = (units * px if px is not None else
                                         t["last_marks"].get(tk, 0.0))
            t["nav"] = ticker_values
            nav_active += ticker_values

        # 2) Mature any tranche that has reached hold_months and book exit
        matured = [t for t in active if (i - t["entry_idx"]) >= hold_months]
        proceeds = 0.0
        for t in matured:
            t["exit_date"] = str(m.date())
            t["exit_nav"] = t["nav"]
            t["return_pct"] = (t["nav"] / t["initial_capital"] - 1.0) * 100
            t["status"] = "exited"
            tranche_log.append(t)
            proceeds += t["nav"]
        active = [t for t in active if (i - t["entry_idx"]) < hold_months]

        # 3) New deposit + proceeds form the new tranche's capital
        capital_to_deploy = deposit + proceeds + cash_pool
        cum_deposits += deposit
        cash_pool = 0.0

        # 4) Form new tranche this month
        eligible = data.members_g.get(m, set()) - {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA"}
        if regime == "crash":
            # cash this month: keep capital_to_deploy in cash for next month
            cash_pool = capital_to_deploy
            picks, weights = [], []
        else:
            picks, weights = pick_v5_baseline(m, eligible, data, regime)
        if not picks:
            cash_pool = capital_to_deploy
        else:
            # Convert capital to units at this month's prices
            # capital after cost: capital_to_deploy * (1 - cf)
            cap_after_cost = capital_to_deploy * (1 - cf)
            units: dict = {}
            for tk, w in zip(picks, weights):
                px = (float(data.mp.at[m, tk])
                       if (tk in data.mp.columns and m in data.mp.index
                           and pd.notna(data.mp.at[m, tk]))
                       else None)
                if px is None or px <= 0:
                    # fallback: pretend we bought $X of this ticker at entry,
                    # tracked separately with last_marks
                    units[tk] = 0.0
                else:
                    units[tk] = (cap_after_cost * w) / px
            tranche = {
                "entry_idx": i,
                "entry_date": str(m.date()),
                "regime": regime,
                "picks": ",".join(picks),
                "weights": ",".join(f"{w:.3f}" for w in weights),
                "initial_capital": capital_to_deploy,
                "units": units,
                "last_marks": {tk: capital_to_deploy * w
                                for tk, w in zip(picks, weights)},
                "nav": capital_to_deploy * (1 - cf),
                "status": "open",
            }
            active.append(tranche)

        # 5) Total NAV: sum of all active tranche values + cash_pool
        # (We already marked active tranches at start of this loop; mature ones
        # were converted to cash and recycled into the new tranche.)
        total_active_nav = sum(t["nav"] for t in active)
        total_nav = total_active_nav + cash_pool

        log.append({
            "date": str(m.date()),
            "regime": regime,
            "cum_deposits": cum_deposits,
            "active_tranches": len(active),
            "cash_pool": cash_pool,
            "total_nav": total_nav,
            "current_picks": ",".join(picks),
            "current_weights": ",".join(f"{w:.3f}" for w in weights),
        })

    # Close any still-open tranches at the end
    final_m = asofs[-1]
    for t in active:
        t["exit_date"] = str(final_m.date()) + " (open)"
        t["exit_nav"] = t["nav"]
        t["return_pct"] = (t["nav"] / t["initial_capital"] - 1.0) * 100
        t["status"] = "open"
        tranche_log.append(t)

    return {"log": log, "tranches": tranche_log, "n_months": len(asofs),
             "cum_deposits": cum_deposits}


def xirr_monthly_deposits(log: list, deposit: float = 1.0) -> float:
    """Money-weighted CAGR for staggered DCA.

    Cashflows: −deposit each month, +final_nav at the end. Solve for monthly
    rate r such that NPV = 0; annualise."""
    n = len(log)
    final_nav = log[-1]["total_nav"]

    def npv(r_m):
        s = 0.0
        for i in range(n):
            s += -deposit * (1 + r_m) ** (n - 1 - i)
        return s + final_nav
    lo, hi = -0.5, 0.5
    if npv(lo) < 0 or npv(hi) > 0:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2
        v = npv(mid)
        if abs(v) < 1e-8:
            break
        if v > 0:
            lo = mid
        else:
            hi = mid
    r_m = (lo + hi) / 2
    return (1 + r_m) ** 12 - 1


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"\n{'='*60}\n  STAGGERED MONTHLY-TRANCHE DCA\n{'='*60}")

    res = run_staggered_dca(data)
    log = res["log"]
    tranches = res["tranches"]
    n = len(log)
    cum = res["cum_deposits"]
    final = log[-1]["total_nav"]

    # Money-weighted IRR
    irr_annual = xirr_monthly_deposits(log, deposit=1.0)

    # For comparison: SPY same approach (each month deposit $1 in SPY, hold to today)
    spy_log = []
    cum_spy = 0.0
    nav_spy = 0.0
    for r in log:
        d = pd.Timestamp(r["date"])
        ret = (float(data.mret.at[d, "SPY"])
                if (d in data.mret.index and "SPY" in data.mret.columns
                    and pd.notna(data.mret.at[d, "SPY"]))
                else 0.0)
        nav_spy = nav_spy * (1 + ret) + 1.0   # deposit $1 at month-end then grow
        cum_spy += 1.0
        spy_log.append({"date": r["date"], "nav": nav_spy, "cum": cum_spy})
    spy_final = spy_log[-1]["nav"]
    irr_spy = xirr_monthly_deposits(
        [{"total_nav": s["nav"]} for s in spy_log], deposit=1.0)

    print(f"  n_months: {n}, total deposits: ${cum:.0f}")
    print(f"  final NAV (strategy): ${final:,.0f}  (multiple {final/cum:.2f}×)")
    print(f"  final NAV (SPY DCA):  ${spy_final:,.0f}  (multiple {spy_final/cum:.2f}×)")
    print(f"  Money-weighted CAGR (strategy): {irr_annual*100:7.2f}%")
    print(f"  Money-weighted CAGR (SPY DCA):  {irr_spy*100:7.2f}%")
    print(f"  Edge: {(irr_annual - irr_spy)*100:+.2f}pp")

    # Distribution of tranche returns
    rets = [t["return_pct"] for t in tranches if t.get("status") != "open"]
    if rets:
        rets_arr = np.array(rets)
        print(f"\n  Tranche stats: n={len(rets)}, mean 6m return={rets_arr.mean():.2f}%, "
              f"median={np.median(rets_arr):.2f}%, win rate={(rets_arr > 0).mean()*100:.1f}%")
        print(f"  Best 5 tranches:")
        top5 = sorted(tranches, key=lambda x: x.get("return_pct", -1e9), reverse=True)[:5]
        for t in top5:
            print(f"    {t['entry_date']} → {t['exit_date']}  {t['picks']:<25s} {t['return_pct']:+7.2f}%")
        print(f"  Worst 5 tranches:")
        worst5 = sorted(tranches, key=lambda x: x.get("return_pct", 1e9))[:5]
        for t in worst5:
            print(f"    {t['entry_date']} → {t['exit_date']}  {t['picks']:<25s} {t['return_pct']:+7.2f}%")

    # Save
    pd.DataFrame(log).to_csv(RES / "staggered_dca_equity.csv", index=False)
    # Strip non-serializable bits before saving tranches
    tr_save = []
    for t in tranches:
        tr_save.append({k: v for k, v in t.items()
                          if k not in ("units", "last_marks")})
    pd.DataFrame(tr_save).to_csv(RES / "staggered_dca_tranches.csv", index=False)

    report = {
        "approach": "staggered_monthly_tranche_dca",
        "description": ("Each month-end deposit $1 into a new 3-stock v5 "
                         "basket. Hold each tranche 6 months. Up to 6 "
                         "overlapping tranches active at any time. Mature "
                         "tranches' proceeds plus the new $1 deposit form "
                         "the current month's tranche."),
        "n_months": n,
        "cum_deposits": cum,
        "final_nav_strategy": final,
        "final_nav_spy_dca": spy_final,
        "multiple_strategy": final / cum,
        "multiple_spy_dca": spy_final / cum,
        "money_weighted_cagr_strategy_pct": irr_annual * 100,
        "money_weighted_cagr_spy_dca_pct": irr_spy * 100,
        "edge_pp": (irr_annual - irr_spy) * 100,
        "n_tranches": len(tranches),
        "tranche_mean_6m_return_pct": float(np.mean(rets)) if rets else None,
        "tranche_median_6m_return_pct": float(np.median(rets)) if rets else None,
        "tranche_win_rate_pct": float((np.array(rets) > 0).mean() * 100) if rets else None,
        "tranche_p10_pct": float(np.percentile(rets, 10)) if rets else None,
        "tranche_p90_pct": float(np.percentile(rets, 90)) if rets else None,
    }
    with open(RES / "staggered_dca.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved:")
    print(f"  {RES / 'staggered_dca.json'}")
    print(f"  {RES / 'staggered_dca_equity.csv'}")
    print(f"  {RES / 'staggered_dca_tranches.csv'}")


if __name__ == "__main__":
    main()
