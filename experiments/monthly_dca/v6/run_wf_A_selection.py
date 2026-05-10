"""Walk-forward weighting selection for Option A.

A has no continuous parameter — the choice is binary (ew vs invvol).
For each test split, choose the weighting that performed best on training
data (months strictly before the split's start), apply to test.
This is the apples-to-apples WF-honest version of the A vs v3 comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate, WF_SPLITS, cagr_monthly, sharpe_monthly, maxdd_monthly,
)

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def run(weighting: str, cy: float, panel, mr, spy):
    cfg = V6Config(name=f"{weighting}_cy{cy}", scorer="ml_3plus6",
                   regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                   weighting=weighting, hold_months=6, cost_bps=10.0,
                   cash_yield_yr=cy)
    eq = simulate(cfg, panel, mr, spy)
    spy_aln = build_spy_aligned(eq, mr)
    return eq, spy_aln


def measure(eq, spy_aln, lo, hi):
    e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
    sa = spy_aln[(spy_aln["date"] >= lo) & (spy_aln["date"] <= hi)]
    r = e["ret_m"].astype(float); sr = sa["spy_ret_m"].astype(float)
    return {"cagr": cagr_monthly(r), "sharpe": sharpe_monthly(r),
            "max_dd": maxdd_monthly(r), "spy_cagr": cagr_monthly(sr),
            "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100}


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    # Candidate variants (cash yield kept at 0.03 always — non-controversial)
    variants = {
        "v3_ew": ("ew", 0.0),
        "ew_cy3": ("ew", 0.03),
        "invvol_cy3": ("invvol", 0.03),
    }
    eqs = {}
    for label, (w, cy) in variants.items():
        eqs[label] = run(w, cy, panel, mr, spy)

    print("\n=== Option A WF weighting selection (CAGR objective) ===")
    rows_cagr = []
    for split, lo, hi in WF_SPLITS:
        lo_t = pd.Timestamp(lo); hi_t = pd.Timestamp(hi)
        train_perf = {}
        for label in variants:
            eq, _ = eqs[label]
            r = eq[eq["date"] < lo_t]["ret_m"].astype(float)
            train_perf[label] = cagr_monthly(r)
        best = max(train_perf, key=train_perf.get)
        eq, sa = eqs[best]
        t = measure(eq, sa, lo_t, hi_t)
        t_v3 = measure(eqs["v3_ew"][0], eqs["v3_ew"][1], lo_t, hi_t)
        rows_cagr.append({
            "split": split, "selector": "train_cagr",
            "best_variant": best, **{f"test_{k}": v for k, v in t.items()},
            "test_v3_cagr": t_v3["cagr"],
            "lift_pp_vs_v3": (t["cagr"] - t_v3["cagr"]) * 100,
        })
        print(f"  [{split:9s}] best={best:10s} test_cagr={t['cagr']*100:6.2f}%  v3={t_v3['cagr']*100:6.2f}%  lift={rows_cagr[-1]['lift_pp_vs_v3']:+5.2f}pp")

    print("\n=== Option A WF weighting selection (Sharpe objective) ===")
    rows_sharpe = []
    for split, lo, hi in WF_SPLITS:
        lo_t = pd.Timestamp(lo); hi_t = pd.Timestamp(hi)
        train_perf = {}
        for label in variants:
            eq, _ = eqs[label]
            r = eq[eq["date"] < lo_t]["ret_m"].astype(float)
            train_perf[label] = sharpe_monthly(r)
        best = max(train_perf, key=train_perf.get)
        eq, sa = eqs[best]
        t = measure(eq, sa, lo_t, hi_t)
        t_v3 = measure(eqs["v3_ew"][0], eqs["v3_ew"][1], lo_t, hi_t)
        rows_sharpe.append({
            "split": split, "selector": "train_sharpe",
            "best_variant": best, **{f"test_{k}": v for k, v in t.items()},
            "test_v3_sharpe": t_v3["sharpe"], "test_v3_cagr": t_v3["cagr"],
            "lift_sharpe_vs_v3": t["sharpe"] - t_v3["sharpe"],
            "lift_cagr_pp_vs_v3": (t["cagr"] - t_v3["cagr"]) * 100,
        })
        print(f"  [{split:9s}] best={best:10s} test_sh={t['sharpe']:.3f} test_cagr={t['cagr']*100:6.2f}% "
              f" v3_sh={t_v3['sharpe']:.3f} v3_cagr={t_v3['cagr']*100:6.2f}% "
              f"liftSh={t['sharpe']-t_v3['sharpe']:+.3f}")

    pd.DataFrame(rows_cagr + rows_sharpe).to_csv(OUT / "wf_A_selection.csv", index=False)

    df_c = pd.DataFrame(rows_cagr)
    df_s = pd.DataFrame(rows_sharpe)
    print(f"\n[CAGR objective] mean test_cagr={df_c['test_cagr'].mean()*100:.2f}% "
          f"v3 mean={df_c['test_v3_cagr'].mean()*100:.2f}% "
          f"lift={(df_c['test_cagr'].mean()-df_c['test_v3_cagr'].mean())*100:+.2f}pp  "
          f"beats_v3={int((df_c['lift_pp_vs_v3']>0).sum())}/10")
    print(f"[Sharpe obj]    mean test_sharpe={df_s['test_sharpe'].mean():.3f} "
          f"v3 mean={df_s['test_v3_sharpe'].mean():.3f} "
          f"lift={df_s['test_sharpe'].mean()-df_s['test_v3_sharpe'].mean():+.3f}  "
          f"beats_v3={int((df_s['lift_sharpe_vs_v3']>0).sum())}/10")


if __name__ == "__main__":
    main()
