"""Test A+C on the 4 universes the user actually wants to choose between:
sp500_pit, IXN (global tech, US-listed portion), Russell-1000-ish, qqq_tech.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate, WF_SPLITS, cagr_monthly,
)
from universes import IXN_US, R1000, QQQ_TECH, IYW, TECH_BROAD

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"


def filter_universe(panel, tickers):
    return panel[panel["ticker"].isin(set(tickers))].copy()


def apply_chronos(panel, chr_df, q):
    m = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]],
                    on=["asof", "ticker"], how="left").copy()
    m["chr_p70_rk"] = m.groupby("asof")["chronos_p70_3m"].rank(pct=True)
    return m[m["chr_p70_rk"].fillna(0.0) >= q][["asof", "ticker", "score", "vol_1y"]]


def main():
    print("[load]")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    chr_sp = pd.read_parquet(PIT / "ml_preds_chronos.parquet")
    chr_broad = pd.read_parquet(V2 / "ml_preds_chronos_broader.parquet")
    chr_sp["asof"] = pd.to_datetime(chr_sp["asof"])
    chr_broad["asof"] = pd.to_datetime(chr_broad["asof"])

    panel_sp = load_score_panel("ml_3plus6", "sp500_pit")
    panel_broad = load_score_panel("ml_3plus6", "broader")

    universes = {
        "sp500_pit (PIT, ~500)": (panel_sp, chr_sp),
        "ixn_us (~73, US-listed only)": (filter_universe(panel_broad, IXN_US), chr_broad),
        "russell1000_core (~280)": (filter_universe(panel_broad, R1000), chr_broad),
        "qqq_tech (~92)": (filter_universe(panel_broad, QQQ_TECH), chr_broad),
        "iyw_tech (~127, reference)": (filter_universe(panel_broad, IYW), chr_broad),
        "tech_broad (~212, reference)": (filter_universe(panel_broad, TECH_BROAD), chr_broad),
    }

    strats = [
        ("v3", "ew", 3, 0.0),
        ("A", "invvol", 3, 0.0),
        ("C", "ew", 3, 0.4),
        ("A+C", "invvol", 3, 0.4),
    ]

    rows = []
    for u_name, (panel, chr_df) in universes.items():
        n_tickers = panel["ticker"].nunique()
        print(f"\n=== {u_name}  (panel: {n_tickers} tickers) ===")
        for label, w, kb, q in strats:
            p = apply_chronos(panel, chr_df, q) if q > 0 else panel
            cfg = V6Config(name=label, scorer="ml_3plus6", regime_gate="tight",
                           k_normal=3, k_recovery=3, k_bull=kb, weighting=w,
                           hold_months=6, cost_bps=10.0,
                           cash_yield_yr=(0.03 if w == "invvol" else 0))
            try:
                eq = simulate(cfg, p, mr, spy)
                spy_aln = build_spy_aligned(eq, mr)
                m = evaluate(eq, spy_aln, label)
                # Holdout 2024-05 -> 2025-12
                ho = eq[eq["date"] >= "2024-05-01"].reset_index(drop=True)
                ho_spy = build_spy_aligned(ho, mr)
                hm = evaluate(ho, ho_spy, "ho")
                rows.append({
                    "universe": u_name, "strategy": label, "n_tickers": n_tickers,
                    "cagr_full": m["cagr_full"], "sharpe": m["sharpe"],
                    "max_dd": m["max_dd"], "wf_mean": m["wf_mean_cagr"],
                    "wf_min": m["wf_min_cagr"], "wf_beats_spy": m["wf_n_beats_spy"],
                    "ho_cagr": hm["cagr_full"], "ho_edge_pp": hm["edge_full_pp"],
                    "ho_sharpe": hm["sharpe"], "ho_mdd": hm["max_dd"],
                })
                print(f"  [{label:5s}] CAGR={m['cagr_full']*100:5.1f}% Sh={m['sharpe']:.3f} "
                      f"MDD={m['max_dd']*100:5.1f}% WFmean={m['wf_mean_cagr']*100:5.1f}% "
                      f"WFmin={m['wf_min_cagr']*100:5.1f}% beats={m['wf_n_beats_spy']}/10 "
                      f" | HO: edge={hm['edge_full_pp']:+5.1f}pp Sh={hm['sharpe']:.2f}")
            except Exception as e:
                print(f"  [{label}] ERROR: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "deployment_universe_choice.csv", index=False)

    print("\n=== Sharpe summary ===")
    print(df.pivot_table(index="universe", columns="strategy", values="sharpe").round(3).to_string())
    print("\n=== WF mean CAGR ===")
    print(df.pivot_table(index="universe", columns="strategy", values="wf_mean").round(3).to_string())
    print("\n=== Holdout edge vs SPY (pp) ===")
    print(df.pivot_table(index="universe", columns="strategy", values="ho_edge_pp").round(2).to_string())
    print("\n=== MaxDD ===")
    print(df.pivot_table(index="universe", columns="strategy", values="max_dd").round(3).to_string())


if __name__ == "__main__":
    main()
