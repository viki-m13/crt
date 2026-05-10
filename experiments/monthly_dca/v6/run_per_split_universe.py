"""Per-split TEST CAGR for v3/A/B/C/A+C/B+C on each universe.

Especially for the tech universes - verify that the lift survives across
all 10 walk-forward splits, not just driven by the recent holdout window.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate, WF_SPLITS, cagr_monthly, sharpe_monthly, maxdd_monthly,
)
from universes import QQQ_TECH, IYW, TECH_BROAD  # noqa: E402

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"


def filter_universe(panel, tickers):
    return panel[panel["ticker"].isin(set(tickers))].copy()


def apply_chronos_filter(panel, chr_df, q):
    m = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]],
                    on=["asof", "ticker"], how="left").copy()
    m["chr_p70_rk"] = m.groupby("asof")["chronos_p70_3m"].rank(pct=True)
    return m[m["chr_p70_rk"].fillna(0.0) >= q][["asof", "ticker", "score", "vol_1y"]]


def main():
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    chr_sp = pd.read_parquet(PIT / "ml_preds_chronos.parquet")
    chr_broad = pd.read_parquet(V2 / "ml_preds_chronos_broader.parquet")
    chr_sp["asof"] = pd.to_datetime(chr_sp["asof"])
    chr_broad["asof"] = pd.to_datetime(chr_broad["asof"])

    panel_sp = load_score_panel("ml_3plus6", "sp500_pit")
    panel_broad = load_score_panel("ml_3plus6", "broader")
    panel_iyw = filter_universe(panel_broad, IYW)
    panel_tech = filter_universe(panel_broad, TECH_BROAD)

    universes = [
        ("sp500_pit", panel_sp, chr_sp),
        ("iyw_tech", panel_iyw, chr_broad),
        ("tech_broad", panel_tech, chr_broad),
    ]
    strats = [
        ("v3", "ew", 3, 0.0),
        ("A", "invvol", 3, 0.0),
        ("B", "invvol", 2, 0.0),
        ("C", "ew", 3, 0.4),
        ("A_plus_C", "invvol", 3, 0.4),
        ("B_plus_C", "invvol", 2, 0.4),
    ]

    rows = []
    for u, panel, chr_df in universes:
        for label, w, kb, q in strats:
            p = apply_chronos_filter(panel, chr_df, q) if q > 0 else panel
            cfg = V6Config(name=label, scorer="ml_3plus6", regime_gate="tight",
                           k_normal=3, k_recovery=3, k_bull=kb, weighting=w,
                           hold_months=6, cost_bps=10.0,
                           cash_yield_yr=(0.03 if w == "invvol" else 0.0))
            eq = simulate(cfg, p, mr, spy)
            spy_aln = build_spy_aligned(eq, mr)
            for split, lo, hi in WF_SPLITS:
                lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
                e = eq[(eq["date"] >= lo_t) & (eq["date"] <= hi_t)]
                sa = spy_aln[(spy_aln["date"] >= lo_t) & (spy_aln["date"] <= hi_t)]
                r = e["ret_m"].astype(float); sr = sa["spy_ret_m"].astype(float)
                rows.append({
                    "universe": u, "strategy": label, "split": split,
                    "cagr": cagr_monthly(r), "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100,
                    "sharpe": sharpe_monthly(r), "max_dd": maxdd_monthly(r),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_split_universe.csv", index=False)

    for u in ["sp500_pit", "iyw_tech", "tech_broad"]:
        sub = df[df["universe"] == u]
        print(f"\n=== {u} edge pp per split ===")
        print(sub.pivot_table(index="split", columns="strategy", values="edge_pp").round(2).to_string())


if __name__ == "__main__":
    main()
