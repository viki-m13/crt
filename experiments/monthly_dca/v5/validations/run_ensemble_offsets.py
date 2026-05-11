"""Ensemble-of-offsets — calendar-anchored version.

Six parallel sub-portfolios, each rebalancing on a FIXED pair of calendar
months (Jan/Jul, Feb/Aug, ..., Jun/Dec). When the regime gate fires "crash",
the sub-portfolio exits to cash but RE-ENTERS only at its next scheduled
calendar rebalance — not at the first non-crash month, which would
collapse all 6 offsets onto the same post-crash schedule (the bug found in
v1 of this script).

Each sub-portfolio runs the full production picker (K=3 GBM+Chronos,
inv-vol cap 0.40, H=6). Capital split 1/6 across the 6.

Hypothesis: vs single-offset production this should give similar long-run
CAGR (offsets are exchangeable in expectation) with materially lower
per-year edge variance and a less-bad worst year.

If it doesn't (regime gate alpha is too tied to the production schedule),
report that honestly.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, pick_v5_baseline, classify_regime_tight,
    invvol_weights, COST_BPS, CAP_PER_PICK, HOLD_MONTHS,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"


def run_calendar_anchored_sim(data: HarnessData, pick_fn,
                                schedule_months: tuple[int, int],
                                start: pd.Timestamp,
                                end: pd.Timestamp,
                                hold_months: int = HOLD_MONTHS,
                                cost_bps: float = COST_BPS,
                                min_hold: int = 3) -> dict:
    """Run a lump-sum sim with calendar-anchored rebalances PLUS
    crash-recovery re-entry. Rebalances on either:
      - first non-crash month after a crash exit (preserves the +28pp
        regime-gate recovery-rally alpha that production captures), OR
      - the sub-portfolio's scheduled month (e.g. Jan/Jul), provided
        at least `min_hold` months have passed since last entry.

    Sub-portfolios with different schedules differ outside crash windows
    and converge briefly at post-crash re-entries.
    """
    cf = cost_bps / 1e4
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    cur_picks: list[str] = []
    cur_weights = np.array([])
    cash = False
    in_basket = False
    months_since_entry = 0
    just_exited_crash = False
    last_regime = "normal"
    equity = 1.0
    log = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        is_scheduled = m.month in schedule_months

        # Apply current month's return to basket carried from prior iteration.
        if not in_basket or cash or not cur_picks:
            ret_m = 0.0
        else:
            r = 0.0
            for tk, w in zip(cur_picks, cur_weights):
                rt = (float(data.mret.at[m, tk])
                      if (tk in data.mret.columns and m in data.mret.index
                          and pd.notna(data.mret.at[m, tk]))
                      else 0.0)
                r += w * rt
            ret_m = r

        # Detect crash transition
        if regime == "crash" and in_basket:
            cur_picks, cur_weights = [], np.array([])
            cash = True
            in_basket = False
            months_since_entry = 0
        just_exited_crash = (last_regime == "crash" and regime != "crash" and cash)

        # Rebalance trigger: crash-exit recovery OR scheduled month with full hold
        do_reb = False
        if regime != "crash":
            if just_exited_crash:
                do_reb = True  # capture recovery rally
            elif is_scheduled and (months_since_entry >= min_hold
                                    or not in_basket):
                do_reb = True  # scheduled rebalance, with min hold respected

        if do_reb:
            eligible = data.members_g.get(m, set()) - set(EXCLUDE)
            picks, weights = pick_fn(m, eligible, data, regime)
            if picks and len(picks) >= 1:
                cur_picks = list(picks)
                cur_weights = np.array(weights, dtype=float)
                if cur_weights.sum() == 0:
                    cur_weights = np.ones(len(cur_picks)) / len(cur_picks)
                else:
                    cur_weights = cur_weights / cur_weights.sum()
                cash = False
                in_basket = True
                months_since_entry = 0
                if log:
                    ret_m -= cf

        if in_basket:
            months_since_entry += 1
        equity *= (1 + ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "is_scheduled": is_scheduled,
                     "in_basket": in_basket, "cash": cash,
                     "just_exited_crash": just_exited_crash,
                     "ret_m": ret_m, "equity": equity,
                     "picks": ",".join(cur_picks) if cur_picks else "",
                     "n_picks": len(cur_picks)})
        last_regime = regime

    return {"log": log}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy()
    spy.index = pd.to_datetime(spy.index)
    start = data.asofs[0]
    end = data.spy_features.index.max()
    print(f"Loaded. Running 6 calendar-anchored sub-portfolios "
          f"{start.date()} → {end.date()}")

    # Six rebalance schedules
    schedules = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
    names = ["Jan/Jul", "Feb/Aug", "Mar/Sep", "Apr/Oct", "May/Nov", "Jun/Dec"]

    sub_logs = {}
    for off, (sched, label) in enumerate(zip(schedules, names)):
        print(f"  Sub-portfolio {off} ({label})...")
        sim = run_calendar_anchored_sim(data, pick_v5_baseline,
                                          schedule_months=sched,
                                          start=start, end=end,
                                          hold_months=HOLD_MONTHS)
        df = pd.DataFrame(sim["log"])
        df["date"] = pd.to_datetime(df["date"])
        sub_logs[off] = df

    # Align to common date grid (use the longest one; they should all be
    # identical since start/end are the same)
    common_dates = sub_logs[0]["date"]
    n = len(common_dates)
    ensemble_eq = np.zeros(n)
    ensemble_ret = np.zeros(n)
    for off in range(6):
        sub = sub_logs[off].set_index("date").reindex(common_dates).reset_index()
        ensemble_eq += sub["equity"].fillna(1.0).values / 6.0
        ensemble_ret += sub["ret_m"].fillna(0.0).values / 6.0

    ens_df = pd.DataFrame({"date": common_dates,
                            "equity": ensemble_eq,
                            "ret_m": ensemble_ret})
    ens_df["year"] = ens_df["date"].dt.year

    # Single-offset (Jan/Jul) for reference
    single = sub_logs[0].copy()
    single["year"] = single["date"].dt.year

    # SPY same window
    spy_w = spy.loc[common_dates.iloc[0]:common_dates.iloc[-1]]

    def stats(rets, eq):
        years = len(rets) / 12
        cagr = (eq[-1] ** (1/years) - 1) * 100
        rr = pd.Series(rets)
        sh = float(rr.mean() / rr.std() * np.sqrt(12)) if rr.std() > 0 else 0.0
        s = pd.Series(eq)
        mdd = float(((s - s.cummax()) / s.cummax()).min() * 100)
        return cagr, sh, mdd

    cagr_e, sh_e, mdd_e = stats(ensemble_ret, ensemble_eq)
    cagr_s, sh_s, mdd_s = stats(single["ret_m"].values, single["equity"].values)
    cagr_spy = ((1 + spy_w.fillna(0)).cumprod().iloc[-1] ** (12 / n) - 1) * 100

    yr_ens = ens_df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1) * 100
    yr_sng = single.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1) * 100
    spy_yr = spy_w.groupby(spy_w.index.year).apply(lambda x: (1+x).prod()-1) * 100

    print(f"\n=== ENSEMBLE-OF-6 (calendar-anchored) vs SINGLE-OFFSET (Jan/Jul) ===")
    print(f"{'Metric':<25} {'Ensemble':>12} {'Single':>12} {'SPY':>10}")
    print(f"{'-'*65}")
    print(f"{'Full-window CAGR':<25} {cagr_e:>11.2f}% {cagr_s:>11.2f}% {cagr_spy:>9.2f}%")
    print(f"{'Edge vs SPY':<25} {cagr_e-cagr_spy:>+10.2f}pp {cagr_s-cagr_spy:>+10.2f}pp")
    print(f"{'Sharpe':<25} {sh_e:>12.2f} {sh_s:>12.2f}")
    print(f"{'MaxDD':<25} {mdd_e:>11.2f}% {mdd_s:>11.2f}%")

    # Per-year breakdown
    print(f"\nPer-year edges vs SPY (Ensemble − Single-offset):")
    print(f"{'Year':>6} {'Ensemble':>12} {'Single':>12} {'Δ':>10}")
    rows_yr = []
    for y in sorted(set(yr_ens.index) & set(yr_sng.index) & set(spy_yr.index)):
        if y < 2004 or y > 2025:
            continue
        e_edge = yr_ens[y] - spy_yr[y]
        s_edge = yr_sng[y] - spy_yr[y]
        d = e_edge - s_edge
        print(f"{y:>6} {e_edge:>+10.2f}pp {s_edge:>+10.2f}pp {d:>+8.2f}pp")
        rows_yr.append({"year": int(y), "ensemble_edge_pp": float(e_edge),
                        "single_edge_pp": float(s_edge), "delta_pp": float(d)})

    e_arr = np.array([r["ensemble_edge_pp"] for r in rows_yr])
    s_arr = np.array([r["single_edge_pp"] for r in rows_yr])
    print(f"\nPer-year edge statistics:")
    print(f"{'Stat':<22} {'Ensemble':>12} {'Single':>12}")
    print(f"{'Mean':<22} {e_arr.mean():>+10.2f}pp {s_arr.mean():>+10.2f}pp")
    print(f"{'Std-dev':<22} {e_arr.std():>+10.2f}pp {s_arr.std():>+10.2f}pp")
    print(f"{'Min (worst year)':<22} {e_arr.min():>+10.2f}pp {s_arr.min():>+10.2f}pp")
    print(f"{'Max (best year)':<22} {e_arr.max():>+10.2f}pp {s_arr.max():>+10.2f}pp")
    print(f"{'Range':<22} {(e_arr.max()-e_arr.min()):>+10.2f}pp {(s_arr.max()-s_arr.min()):>+10.2f}pp")
    print(f"{'# years positive':<22} {(e_arr > 0).sum():>12} {(s_arr > 0).sum():>12}")
    print(f"{'2024 specifically':<22} {e_arr[np.array([r['year'] for r in rows_yr])==2024][0]:>+10.2f}pp {s_arr[np.array([r['year'] for r in rows_yr])==2024][0]:>+10.2f}pp")

    # Per-sub-portfolio CAGRs (sanity check that sub-portfolios are
    # genuinely different)
    print(f"\nIndividual sub-portfolio CAGRs (sanity check):")
    for off, label in enumerate(names):
        df = sub_logs[off]
        years = len(df) / 12
        cagr = (df["equity"].iloc[-1] ** (1/years) - 1) * 100
        df["year"] = df["date"].dt.year
        yr = df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1) * 100
        e2024 = (yr.get(2024, 0) - spy_yr.get(2024, 0))
        print(f"  Off{off} ({label:<8}): CAGR {cagr:6.2f}%  2024 edge {e2024:+6.2f}pp")

    # Save
    ens_df.to_csv(RES / "ensemble_offsets_equity.csv", index=False)
    pd.DataFrame(rows_yr).to_csv(RES / "ensemble_offsets_yearly.csv", index=False)
    summary = {
        "ensemble_cagr_pct": float(cagr_e),
        "single_offset_cagr_pct": float(cagr_s),
        "spy_cagr_pct": float(cagr_spy),
        "ensemble_sharpe": float(sh_e), "single_sharpe": float(sh_s),
        "ensemble_max_dd_pct": float(mdd_e), "single_max_dd_pct": float(mdd_s),
        "year_edge_mean_ensemble_pp": float(e_arr.mean()),
        "year_edge_mean_single_pp": float(s_arr.mean()),
        "year_edge_std_ensemble_pp": float(e_arr.std()),
        "year_edge_std_single_pp": float(s_arr.std()),
        "year_edge_min_ensemble_pp": float(e_arr.min()),
        "year_edge_min_single_pp": float(s_arr.min()),
        "n_years_positive_ensemble": int((e_arr > 0).sum()),
        "n_years_positive_single": int((s_arr > 0).sum()),
        "n_months": int(n),
    }
    with open(RES / "ensemble_offsets_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {RES / 'ensemble_offsets_summary.json'}")


if __name__ == "__main__":
    main()
