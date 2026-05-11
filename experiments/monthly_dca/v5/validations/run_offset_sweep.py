"""Rebalance-OFFSET sweep: H=6 fixed, vary which 2 months of the year hold
the rebalances. Directly isolates the "rebalance-date luck" hypothesis.

6 possible offsets (Jan/Jul, Feb/Aug, Mar/Sep, Apr/Oct, May/Nov, Jun/Dec).
If 2024's lump-sum lag is purely date luck, different offsets should give
materially different 2024 outcomes — but the cross-year mean should converge.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    load_all, evaluate, pick_v5_baseline, HOLD_MONTHS,
)
from experiments.monthly_dca.v5.validations.harness import run_sim as _run_sim


RES = Path(__file__).resolve().parent / "results"


def _run_offset_sim(data, pick_fn, offset_months: int, hold_months: int = 6):
    """Run the simulator starting at asofs[offset_months]. This shifts the
    Jan/Jul rebalance schedule to Feb/Aug (offset=1), Mar/Sep (offset=2), etc.
    """
    # The harness's first asof is the start of regime/picks. By default it
    # uses data.asofs from the earliest available. To shift the rebalance
    # schedule, advance the start by `offset_months`.
    start = data.asofs[offset_months] if offset_months < len(data.asofs) else data.asofs[0]
    end = data.spy_features.index.max()
    return _run_sim(data, pick_fn, start=start, end=end, hold_months=hold_months)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"Loaded. asofs {data.asofs[0].date()} -> {data.asofs[-1].date()}")

    summary = []
    months = {0: "Sep/Mar (default)", 1: "Oct/Apr", 2: "Nov/May",
              3: "Dec/Jun", 4: "Jan/Jul", 5: "Feb/Aug"}
    # The first asof is 2003-01-31; offset 0 starts there (Jan/Jul rebals
    # over the long run because of how do_reb fires on i==0 then every 6 i).
    # Offsets 1..5 shift the entire rebalance schedule by 1..5 months.
    for off in (0, 1, 2, 3, 4, 5):
        name = f"offset{off}"
        first_month = data.asofs[off].strftime("%b %Y")
        print(f"\n{'='*60}\n  Offset = {off}  (first rebalance: {first_month})"
              f"\n{'='*60}")
        sim = _run_offset_sim(data, pick_v5_baseline, off, HOLD_MONTHS)
        log = sim["log"]
        # Pull metrics manually since evaluate() uses default start
        df = pd.DataFrame(log)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        # CAGR
        final = 1.0
        for r in log:
            final *= (1 + r["ret_m"])
        n_months = len(log)
        cagr = (final ** (12 / n_months) - 1) * 100 if n_months > 0 else 0.0
        # Per-year
        yr = df.groupby("year")["ret_m"].apply(lambda x: float((1+x).prod()-1))
        # SPY same window
        spy = data.mret["SPY"].copy()
        spy.index = pd.to_datetime(spy.index)
        spy_yr = spy.groupby(spy.index.year).apply(lambda x: float((1+x).prod()-1))
        # Pull lagging years
        lag = {}
        for y in (2014, 2018, 2024, 2025):
            if y in yr.index and y in spy_yr.index:
                lag[y] = (yr[y] - spy_yr[y]) * 100
        # MaxDD
        eq = df["equity"].values
        peak = pd.Series(eq).cummax().values
        mdd = float(((eq - peak) / peak).min() * 100)
        # Sharpe
        rets = df["ret_m"].values
        sh = float(rets.mean() / rets.std() * 12**0.5) if rets.std() > 0 else 0.0
        # SPY-DCA edge over window
        df["spy_ret"] = spy.reindex(df["date"]).fillna(0).values
        spy_yr_total = spy.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
        spy_cagr = ((1+spy_yr_total).prod() ** (12/len(spy_yr_total)) - 1) * 100
        print(f"  Window: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}  "
              f"({n_months} months)")
        print(f"  CAGR: {cagr:.2f}%  vs SPY {spy_cagr:.2f}%  "
              f"(edge {cagr - spy_cagr:+.2f}pp)")
        print(f"  Sharpe {sh:.2f}  MaxDD {mdd:.1f}%")
        print(f"  Year edges:  2014={lag.get(2014, 0):+5.1f}  "
              f"2018={lag.get(2018, 0):+5.1f}  "
              f"2024={lag.get(2024, 0):+5.1f}  "
              f"2025={lag.get(2025, 0):+5.1f}")
        summary.append({
            "offset": off,
            "first_month": first_month,
            "n_months": n_months,
            "cagr_pct": cagr,
            "spy_cagr_pct": spy_cagr,
            "edge_pp": cagr - spy_cagr,
            "sharpe": sh,
            "max_dd_pct": mdd,
            "y2014_edge_pp": lag.get(2014, 0),
            "y2018_edge_pp": lag.get(2018, 0),
            "y2024_edge_pp": lag.get(2024, 0),
            "y2025_edge_pp": lag.get(2025, 0),
        })

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(RES / "offset_sweep_summary.csv", index=False)
    print("\n\n=== OFFSET SWEEP SUMMARY (K=3, H=6, fixed scorer) ===")
    cols = ["offset", "first_month", "cagr_pct", "edge_pp",
            "sharpe", "max_dd_pct",
            "y2014_edge_pp", "y2018_edge_pp",
            "y2024_edge_pp", "y2025_edge_pp"]
    print(df_sum[cols].round(2).to_string(index=False))
    print(f"\n2024 edge by offset: range "
          f"{df_sum['y2024_edge_pp'].min():.1f}pp to "
          f"{df_sum['y2024_edge_pp'].max():.1f}pp")
    print(f"Full-window CAGR by offset: range "
          f"{df_sum['cagr_pct'].min():.1f}% to "
          f"{df_sum['cagr_pct'].max():.1f}%")


if __name__ == "__main__":
    main()
