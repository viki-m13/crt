"""Phase 5: re-run the canonical v2 sp500_pit_filter_backtest on the
augmented inputs. Produces the same set of summary metrics
(WF mean CAGR, full CAGR, MaxDD, splits beating SPY, etc.) so we can
compare directly to reports/final_validation.md.

This is identical to experiments/monthly_dca/v2/sp500_pit_filter_backtest.py
except every input path is redirected to the augmented files:

  ml_preds.parquet            <- augmented/ml_preds.parquet
  monthly_returns_clean       <- augmented/monthly_returns_clean.parquet
  SPY regime features         <- augmented/features/*.parquet
  PIT membership              <- unchanged (it's the truth source)

Outputs (parallel to PIT/sp500_pit_filter_*):
  augmented/sp500_pit_filter_equity.csv
  augmented/sp500_pit_filter_yearly.csv
  augmented/sp500_pit_filter_walkforward.csv
  augmented/sp500_pit_filter_summary.json
  augmented/sp500_pit_filter_coverage.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
AUG = PIT / "augmented"
AUG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# Borrow the validated, regression-tested helpers from the original script.
from experiments.monthly_dca.v2.sp500_pit_filter_backtest import (  # noqa: E402
    classify_regime,
    build_outputs,
    simulate,
    cagr_from,
    sharpe_monthly,
    max_drawdown,
    yearly_returns,
    spy_yearly,
    walk_forward,
)


def load_spy_features_from(features_dir: Path) -> pd.DataFrame:
    """Same as the original load_spy_features but configurable directory."""
    rows = []
    for f in sorted(features_dir.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def main():
    print("=" * 64)
    print("Phase 5: sp500_pit_filter_backtest on augmented inputs")
    print("=" * 64)

    print("[1] loading augmented predictions ...")
    preds = pd.read_parquet(AUG / "ml_preds.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])
    print(f"    preds: {len(preds)} rows, {preds['asof'].nunique()} months, "
          f"{preds['ticker'].nunique()} tickers")

    print("[2] loading PIT membership (unchanged) ...")
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    print(f"    members: {len(members)}, {members['asof'].nunique()} months")

    print("[3] loading augmented SPY features ...")
    spy = load_spy_features_from(AUG / "features")
    print(f"    SPY features: {spy.shape}")
    if spy.empty:
        # Fallback: regime features for SPY are the same as the original since
        # the SPY series itself didn't change. Use the original features dir.
        print("    (no SPY in augmented features; falling back to original)")
        spy = load_spy_features_from(CACHE / "features")
        print(f"    SPY features (fallback): {spy.shape}")

    print("[4] loading augmented monthly returns ...")
    monthly_returns = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    print(f"    monthly returns: {monthly_returns.shape}")

    print("\n[5] building strategy outputs (regime gate + SP500 PIT filter) ...")
    outs = build_outputs(preds, spy, members)
    print(f"    {len(outs)} months")
    regimes = pd.Series([o.regime for o in outs]).value_counts()
    print(f"    regime distribution:\n{regimes.to_string()}")

    print("\n[6] simulating ...")
    eq = simulate(outs, monthly_returns, cost_bps=10.0, starting_cash=1.0)
    eq.to_csv(AUG / "sp500_pit_filter_equity.csv", index=False)

    cgr = cagr_from(eq["equity"])
    sh = sharpe_monthly(eq["ret_m"])
    mdd, dd_s, dd_e = max_drawdown(eq["equity"],
                                   pd.to_datetime(eq["date"]))
    print(f"    months: {len(eq)}, final ${eq['equity'].iloc[-1]:.2f}, "
          f"cagr {cgr:.2%}, sharpe {sh:.2f}, maxDD {mdd:.2%}")

    # SPY benchmark
    if "SPY" in monthly_returns.columns:
        spy_ret = monthly_returns["SPY"].dropna()
        spy_yr = spy_yearly(spy_ret, eq)
        spy_eq_cum = (1 + spy_ret.loc[pd.to_datetime(eq["date"]).iloc[0]:]).cumprod()
    else:
        spy_yr = pd.Series(dtype=float)

    # Yearly
    yr = yearly_returns(eq)
    yr.to_csv(AUG / "sp500_pit_filter_yearly.csv")

    # SPY-aligned yearly: per-month next-month SPY return aligned to o.asof
    eq_dates = pd.to_datetime(eq["date"])
    next_month = eq_dates + pd.offsets.MonthEnd(1)
    aligned = []
    spy_ret_idx = monthly_returns["SPY"].dropna()
    for i, d in enumerate(eq_dates):
        nxt = next_month.iloc[i]
        if nxt in spy_ret_idx.index:
            aligned.append({"date": d, "spy_ret_m": float(spy_ret_idx.loc[nxt])})
        else:
            aligned.append({"date": d, "spy_ret_m": 0.0})
    spy_aligned = pd.DataFrame(aligned)

    wf = walk_forward(eq, spy_aligned)
    wf.to_csv(AUG / "sp500_pit_filter_walkforward.csv", index=False)

    # Coverage
    aug_cov = members.merge(
        preds.assign(in_panel=1)[["asof", "ticker", "in_panel"]],
        on=["asof", "ticker"], how="left",
    )
    aug_cov["in_panel"] = aug_cov["in_panel"].fillna(0).astype(int)
    coverage_yr = aug_cov.groupby(aug_cov["asof"].dt.year)["in_panel"].mean()
    coverage_yr.to_csv(AUG / "sp500_pit_filter_coverage.csv")
    print(f"\n[7] coverage by year (augmented):")
    print(coverage_yr.to_string())

    summary = {
        "n_months": int(len(eq)),
        "final_equity": float(eq["equity"].iloc[-1]),
        "cagr_full": float(cgr),
        "sharpe": float(sh),
        "max_dd": float(mdd),
        "n_cash_months": int((eq["regime"] == "crash").sum()),
        "wf_mean_cagr": float(wf["cagr"].mean()),
        "wf_median_cagr": float(wf["cagr"].median()),
        "wf_min_cagr": float(wf["cagr"].min()),
        "wf_max_cagr": float(wf["cagr"].max()),
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()),
        "wf_n_positive": int((wf["cagr"] > 0).sum()),
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()),
        "wf_n_splits": int(len(wf)),
        "panel_coverage_2003_first": float(coverage_yr.iloc[0]),
        "panel_coverage_2025_last": float(coverage_yr.iloc[-1]),
    }
    with open(AUG / "sp500_pit_filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[8] saved summary -> {AUG / 'sp500_pit_filter_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
