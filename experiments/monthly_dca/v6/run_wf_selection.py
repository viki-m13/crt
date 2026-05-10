"""Walk-forward parameter selection for Options B (kb) and C (chronos q).

For each test split, the parameter is chosen on the training window only
(splits with end-date strictly before the test split's start). The chosen
parameter is then applied to the test split. This eliminates the in-sample
parameter-selection leak that the published agents had.

Outputs:
  results/wf_B_selection.csv   — per-test-split chosen kb + perf
  results/wf_C_selection.csv   — per-test-split chosen q + perf
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate, WF_SPLITS, cagr_monthly, sharpe_monthly, maxdd_monthly,
)

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def measure_split(eq: pd.DataFrame, spy_aln: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
    sa = spy_aln[(spy_aln["date"] >= lo) & (spy_aln["date"] <= hi)]
    r = e["ret_m"].astype(float); sr = sa["spy_ret_m"].astype(float)
    return {
        "cagr": cagr_monthly(r), "sharpe": sharpe_monthly(r),
        "max_dd": maxdd_monthly(r), "spy_cagr": cagr_monthly(sr),
        "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100,
    }


def run_with_kb(kb: int, panel, mr, spy):
    cfg = V6Config(name=f"kb{kb}", scorer="ml_3plus6", regime_gate="tight",
                   k_normal=3, k_recovery=3, k_bull=kb, weighting="invvol",
                   hold_months=6, cost_bps=10.0, cash_yield_yr=0.03)
    eq = simulate(cfg, panel, mr, spy)
    spy_aln = build_spy_aligned(eq, mr)
    return eq, spy_aln


def run_with_q(q: float, panel, chr_df, mr, spy):
    if q <= 0:
        filt = panel
    else:
        m = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]],
                        on=["asof", "ticker"], how="left").copy()
        m["chr_p70_rk"] = m.groupby("asof")["chronos_p70_3m"].rank(pct=True)
        filt = m[m["chr_p70_rk"].fillna(0.0) >= q][["asof", "ticker", "score", "vol_1y"]].copy()
    cfg = V6Config(name=f"q{q}", scorer="ml_3plus6", regime_gate="tight",
                   k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                   hold_months=6, cost_bps=10.0)
    eq = simulate(cfg, filt, mr, spy)
    spy_aln = build_spy_aligned(eq, mr)
    return eq, spy_aln


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    panel["asof"] = pd.to_datetime(panel["asof"])
    chr_df = pd.read_parquet(PIT / "ml_preds_chronos.parquet")
    chr_df["asof"] = pd.to_datetime(chr_df["asof"])
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    # ------------------------------------------------------------------ Option B WF
    print("\n=== WF kb selection for Option B ===")
    kbs = [1, 2, 3, 4, 5]
    # Pre-compute equity for every kb (cheap since panel small)
    kb_eq = {}
    for kb in kbs:
        eq, spy_aln = run_with_kb(kb, panel, mr, spy)
        kb_eq[kb] = (eq, spy_aln)

    # For each split, choose kb on training data (all splits ending before split start)
    rows_B = []
    for split, lo, hi in WF_SPLITS:
        lo_t = pd.Timestamp(lo); hi_t = pd.Timestamp(hi)
        # Training perf for each kb: aggregate over months [first_asof, lo_t - 1d)
        train_perf = {}
        for kb in kbs:
            eq, _ = kb_eq[kb]
            train_eq = eq[eq["date"] < lo_t]
            r = train_eq["ret_m"].astype(float)
            train_perf[kb] = cagr_monthly(r)
        best_kb = max(train_perf, key=train_perf.get)
        # Test perf on chosen kb
        eq, spy_aln = kb_eq[best_kb]
        test = measure_split(eq, spy_aln, lo_t, hi_t)
        # Also v3 baseline test perf
        cfg_v3 = V6Config(name="v3", scorer="ml_3plus6", regime_gate="tight",
                          k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                          hold_months=6, cost_bps=10.0)
        eq_v3 = simulate(cfg_v3, panel, mr, spy)
        sa_v3 = build_spy_aligned(eq_v3, mr)
        test_v3 = measure_split(eq_v3, sa_v3, lo_t, hi_t)
        rows_B.append({
            "split": split, "best_kb": best_kb,
            "train_perf_by_kb": train_perf,
            **{f"test_{k}": v for k, v in test.items()},
            "test_v3_cagr": test_v3["cagr"], "test_v3_edge_pp": test_v3["edge_pp"],
            "lift_pp_vs_v3": (test["cagr"] - test_v3["cagr"]) * 100,
        })
        print(f"  [{split:9s}] best_kb={best_kb}  test_cagr={test['cagr']*100:6.2f}%  "
              f"v3={test_v3['cagr']*100:6.2f}%  lift={rows_B[-1]['lift_pp_vs_v3']:+5.2f}pp")
    df_B = pd.DataFrame(rows_B)
    df_B.to_csv(OUT / "wf_B_selection.csv", index=False)

    # Aggregate
    mean_lift = float(df_B["lift_pp_vs_v3"].mean())
    median_lift = float(df_B["lift_pp_vs_v3"].median())
    n_beats_v3 = int((df_B["lift_pp_vs_v3"] > 0).sum())
    print(f"\n  WF mean test CAGR (B with WF-kb): {df_B['test_cagr'].mean()*100:.2f}%")
    print(f"  WF mean test CAGR (v3):           {df_B['test_v3_cagr'].mean()*100:.2f}%")
    print(f"  Mean lift vs v3 (WF-honest):      {mean_lift:+.2f}pp")
    print(f"  Splits where B beats v3:          {n_beats_v3}/10")

    # ------------------------------------------------------------------ Option C WF
    print("\n=== WF q selection for Option C ===")
    qs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
    q_eq = {}
    for q in qs:
        eq, spy_aln = run_with_q(q, panel, chr_df, mr, spy)
        q_eq[q] = (eq, spy_aln)

    rows_C = []
    for split, lo, hi in WF_SPLITS:
        lo_t = pd.Timestamp(lo); hi_t = pd.Timestamp(hi)
        train_perf = {}
        for q in qs:
            eq, _ = q_eq[q]
            train_eq = eq[eq["date"] < lo_t]
            r = train_eq["ret_m"].astype(float)
            train_perf[q] = cagr_monthly(r)
        best_q = max(train_perf, key=train_perf.get)
        eq, spy_aln = q_eq[best_q]
        test = measure_split(eq, spy_aln, lo_t, hi_t)
        eq_v3, sa_v3 = q_eq[0.0]
        test_v3 = measure_split(eq_v3, sa_v3, lo_t, hi_t)
        rows_C.append({
            "split": split, "best_q": best_q,
            **{f"test_{k}": v for k, v in test.items()},
            "test_v3_cagr": test_v3["cagr"], "test_v3_edge_pp": test_v3["edge_pp"],
            "lift_pp_vs_v3": (test["cagr"] - test_v3["cagr"]) * 100,
        })
        print(f"  [{split:9s}] best_q={best_q}  test_cagr={test['cagr']*100:6.2f}%  "
              f"v3={test_v3['cagr']*100:6.2f}%  lift={rows_C[-1]['lift_pp_vs_v3']:+5.2f}pp")
    df_C = pd.DataFrame(rows_C)
    df_C.to_csv(OUT / "wf_C_selection.csv", index=False)

    mean_lift = float(df_C["lift_pp_vs_v3"].mean())
    n_beats_v3 = int((df_C["lift_pp_vs_v3"] > 0).sum())
    print(f"\n  WF mean test CAGR (C with WF-q): {df_C['test_cagr'].mean()*100:.2f}%")
    print(f"  WF mean test CAGR (v3):          {df_C['test_v3_cagr'].mean()*100:.2f}%")
    print(f"  Mean lift vs v3 (WF-honest):     {mean_lift:+.2f}pp")
    print(f"  Splits where C beats v3:         {n_beats_v3}/10")


if __name__ == "__main__":
    main()
