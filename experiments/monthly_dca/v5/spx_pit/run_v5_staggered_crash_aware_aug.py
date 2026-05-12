"""Phase 6b: crash-aware staggered v5 on augmented PIT.

Same 6-tranche stagger as run_v5_staggered_aug.py, with one addition:
when SPY enters the v5 'crash' regime, ALL active tranches force-close
to cash (matching lump-sum v5's crash protection). When crash exits,
fresh monthly deposits restart the stagger from zero — so it takes up
to 6 months to be fully deployed again after a crash.

This is the natural "best of both" between lump-sum (perfect crash
gating, but bad timing-luck) and basic-staggered (good timing-luck,
but no crash gating on legacy tranches).

Inputs / outputs analogous to run_v5_staggered_aug.py; suffix `_ca`
on output files:
  augmented/v5_staggered_ca_summary.json
  augmented/v5_staggered_ca_walkforward.csv
  augmented/v5_staggered_ca_yearly.csv
  augmented/v5_staggered_ca_equity.csv
  augmented/v5_staggered_ca_tranches.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_v5_staggered_aug import (  # noqa: E402
    AUG, PIT, CACHE,
    CHRONOS_FILTER_Q, CAP_PER_PICK, HOLD_MONTHS, K_PICKS, COST_BPS,
    WF_SPLITS, classify_regime_tight, load_spy_features,
    calc_invvol_weights, pick_v5, _tranche_value,
    compute_returns_from_equity, yearly_table, walkforward_table,
)


def run_staggered_crash_aware(panel, ml, chr_, monthly_returns, monthly_prices,
                               members_g, spy):
    """Crash-aware 6-tranche staggered v5.

    Difference from run_staggered: on crash regime, ALL active tranches
    force-close to cash, not just new deposits.
    """
    cf = COST_BPS / 1e4
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}

    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]
    print(f"  simulation months: {len(months)} ({months[0].date()}..{months[-1].date()})")

    active: list[dict] = []
    tranche_log: list[dict] = []
    equity_log: list[dict] = []
    deposit = 1.0
    cash_pool = 0.0
    cum_deposits = 0.0

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})

        # Mark active tranches
        for t in active:
            t["value"] = _tranche_value(t, monthly_prices, m)

        # Crash gate: force-close ALL active tranches into cash.
        # (basic-staggered behavior keeps them open; this variant doesn't.)
        keep = []
        proceeds = 0.0
        if regime == "crash":
            for t in active:
                realised = t["value"] * (1 - cf)
                proceeds += realised
                tranche_log.append({
                    "entry_date": t["entry_date"], "exit_date": m,
                    "entered_with": t["notional_at_entry"], "exited_with": realised,
                    "return_pct": 100.0 * (realised / t["notional_at_entry"] - 1.0),
                    "picks": ",".join(t["picks"]),
                    "regime_at_entry": t["regime_at_entry"],
                    "exit_reason": "crash_force_close",
                })
            active = []
        else:
            # Normal close-at-maturity logic
            for t in active:
                months_held = (m.year - t["entry_date"].year) * 12 + (m.month - t["entry_date"].month)
                if months_held >= HOLD_MONTHS:
                    realised = t["value"] * (1 - cf)
                    proceeds += realised
                    tranche_log.append({
                        "entry_date": t["entry_date"], "exit_date": m,
                        "entered_with": t["notional_at_entry"], "exited_with": realised,
                        "return_pct": 100.0 * (realised / t["notional_at_entry"] - 1.0),
                        "picks": ",".join(t["picks"]),
                        "regime_at_entry": t["regime_at_entry"],
                        "exit_reason": "matured_6m",
                    })
                else:
                    keep.append(t)
            active = keep

        # Form deploy capital
        capital_to_deploy = deposit + proceeds + cash_pool
        cum_deposits += deposit
        cash_pool = 0.0

        # Try to form a new tranche (no new tranche on crash months)
        new_capital_invested = False
        if regime == "crash":
            cash_pool = capital_to_deploy
        else:
            picks, weights = pick_v5(m, panel_by_asof, ml_by_asof, chr_by_asof,
                                     members_g, monthly_returns)
            if not picks:
                cash_pool = capital_to_deploy
            else:
                spent = capital_to_deploy * (1 - cf)
                units = {}
                for tk, w in zip(picks, weights):
                    if tk in monthly_prices.columns and m in monthly_prices.index:
                        px = float(monthly_prices.at[m, tk])
                        if not pd.isna(px) and px > 0:
                            units[tk] = (spent * w) / px
                if units:
                    active.append({
                        "entry_date": m, "picks": picks,
                        "weights": weights, "units": units,
                        "notional_at_entry": capital_to_deploy,
                        "value": spent,
                        "regime_at_entry": regime,
                    })
                    new_capital_invested = True
                else:
                    cash_pool = capital_to_deploy

        nav_invested = sum(t["value"] for t in active)
        nav_total = nav_invested + cash_pool

        equity_log.append({
            "date": m, "regime": regime,
            "n_active_tranches": len(active),
            "nav_invested": nav_invested,
            "cash_pool": cash_pool,
            "nav_total": nav_total,
            "cum_deposits": cum_deposits,
            "new_capital_invested": new_capital_invested,
        })

    return pd.DataFrame(equity_log), pd.DataFrame(tranche_log)


def main():
    print("=" * 64)
    print("Phase 6b: crash-aware staggered v5 on augmented PIT")
    print("=" * 64)

    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])

    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2

    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])

    spy_features = load_spy_features()
    monthly_returns = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    monthly_prices = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    if not isinstance(monthly_returns.index, pd.DatetimeIndex):
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_prices.index = pd.to_datetime(monthly_prices.index)
    monthly_returns = monthly_returns.fillna(0.0)

    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    eq, tranches = run_staggered_crash_aware(panel, ml, chr_, monthly_returns,
                                              monthly_prices, members_g, spy_features)

    eq.to_csv(AUG / "v5_staggered_ca_equity.csv", index=False)
    if len(tranches):
        tranches.to_csv(AUG / "v5_staggered_ca_tranches.csv", index=False)

    yr = yearly_table(eq, monthly_returns)
    yr.to_csv(AUG / "v5_staggered_ca_yearly.csv", index=False)

    wf = walkforward_table(eq, monthly_returns)
    wf.to_csv(AUG / "v5_staggered_ca_walkforward.csv", index=False)

    rets = compute_returns_from_equity(eq, monthly_returns)
    n_months = len(rets)
    cagr_full = (1 + rets).prod() ** (12.0 / n_months) - 1
    spy_full = (1 + monthly_returns["SPY"].loc[rets.index[0]:rets.index[-1]].dropna()).prod() ** (12.0 / n_months) - 1
    sharpe = (rets.mean() / max(rets.std(), 1e-9)) * np.sqrt(12)
    ec = (1 + rets).cumprod()
    peak = ec.cummax()
    mdd = float(((ec - peak) / peak).min())

    print(f"\n[crash-aware staggered]")
    print(f"  n_months: {n_months}")
    print(f"  cagr_full: {cagr_full:.4f}  (SPY: {spy_full:.4f},  edge: {(cagr_full - spy_full)*100:+.2f}pp)")
    print(f"  sharpe: {sharpe:.4f}")
    print(f"  max_dd: {mdd:.4f}")
    print(f"  n_crash_months: {int((eq['regime'] == 'crash').sum())}")
    print(f"  n_tranches_closed: {len(tranches)}")
    print(f"\n[WF]")
    print(wf.to_string(index=False))

    print(f"\n[yearly] (focus on 2024)")
    print(yr.assign(strategy_pct=(yr['strategy_ret']*100).round(1),
                    spy_pct=(yr['spy_ret']*100).round(1),
                    edge_pp_r=yr['edge_pp'].round(1)
                    )[["year","strategy_pct","spy_pct","edge_pp_r"]].to_string(index=False))

    summary = {
        "variant_name": "v5_staggered_6tranche_crash_aware",
        "panel": "augmented_PIT",
        "n_months": int(n_months),
        "cagr_full": float(cagr_full),
        "spy_cagr_full": float(spy_full),
        "edge_full_pp": float((cagr_full - spy_full) * 100),
        "sharpe": float(sharpe),
        "max_dd": float(mdd),
        "n_crash_months": int((eq["regime"] == "crash").sum()),
        "n_tranches_closed": int(len(tranches)),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else None,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else None,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else None,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else None,
        "wf_n_positive": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }
    (AUG / "v5_staggered_ca_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {AUG / 'v5_staggered_ca_summary.json'}")


if __name__ == "__main__":
    main()
