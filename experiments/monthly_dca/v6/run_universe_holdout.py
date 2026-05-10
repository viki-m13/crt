"""2024-05 → 2025-12 holdout per universe — see what survives the freshest year."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)
from universes import QQQ_TECH, IYW, TECH_BROAD  # noqa: E402

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"

HOLDOUT_START = pd.Timestamp("2024-05-01")


def filter_universe(panel, tickers):
    return panel[panel["ticker"].isin(set(tickers))].copy()


def apply_chronos_filter(panel, chr_df, q):
    m = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]],
                    on=["asof", "ticker"], how="left").copy()
    m["chr_p70_rk"] = m.groupby("asof")["chronos_p70_3m"].rank(pct=True)
    out = m[m["chr_p70_rk"].fillna(0.0) >= q].copy()
    return out[["asof", "ticker", "score", "vol_1y"]]


def main():
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    chr_sp_path = PIT / "ml_preds_chronos.parquet"
    chr_broad_path = V2 / "ml_preds_chronos_broader.parquet"
    chr_sp = pd.read_parquet(chr_sp_path) if chr_sp_path.exists() else None
    chr_broad = pd.read_parquet(chr_broad_path) if chr_broad_path.exists() else None
    if chr_sp is not None: chr_sp["asof"] = pd.to_datetime(chr_sp["asof"])
    if chr_broad is not None: chr_broad["asof"] = pd.to_datetime(chr_broad["asof"])

    panel_sp = load_score_panel("ml_3plus6", "sp500_pit")
    panel_broad = load_score_panel("ml_3plus6", "broader")
    panel_nonsp = load_score_panel("ml_3plus6", "non_sp500")
    panel_qqq = filter_universe(panel_broad, QQQ_TECH)
    panel_iyw = filter_universe(panel_broad, IYW)
    panel_tech = filter_universe(panel_broad, TECH_BROAD)

    universes = [
        ("sp500_pit", panel_sp, chr_sp),
        ("broader", panel_broad, chr_broad),
        ("non_sp500", panel_nonsp, chr_broad),
        ("qqq_tech", panel_qqq, chr_broad),
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
    for u_name, panel, chr_df in universes:
        for label, w, kb, q in strats:
            if q > 0 and chr_df is None:
                continue
            p = apply_chronos_filter(panel, chr_df, q) if q > 0 else panel
            cfg = V6Config(name=label, scorer="ml_3plus6", regime_gate="tight",
                           k_normal=3, k_recovery=3, k_bull=kb, weighting=w,
                           hold_months=6, cost_bps=10.0,
                           cash_yield_yr=(0.03 if w == "invvol" else 0.0))
            try:
                eq = simulate(cfg, p, mr, spy)
                # 2024-05 → end metrics
                ho = eq[eq["date"] >= HOLDOUT_START].reset_index(drop=True)
                if len(ho) == 0:
                    continue
                spy_aln = build_spy_aligned(ho, mr)
                m = evaluate(ho, spy_aln, f"{u_name}|{label}")
                m["universe"] = u_name
                m["strategy"] = label
                m["n_months"] = len(ho)
                rows.append(m)
                print(f"[{u_name:11s}|{label:9s}] {m['n_months']}m CAGR={m['cagr_full']*100:6.2f}% "
                      f"SPY={m['spy_cagr_full']*100:6.2f}% edge={m['edge_full_pp']:+5.2f}pp "
                      f"MDD={m['max_dd']*100:6.2f}% Sh={m['sharpe']:.3f}")
            except Exception as e:
                print(f"[{u_name}|{label}] ERROR: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "universe_holdout.csv", index=False)

    print("\n=== HOLDOUT 2024-05 → present CAGR ===")
    print(df.pivot_table(index="universe", columns="strategy", values="cagr_full").round(4).to_string())
    print("\n=== HOLDOUT Edge vs SPY (pp) ===")
    print(df.pivot_table(index="universe", columns="strategy", values="edge_full_pp").round(2).to_string())
    print("\n=== HOLDOUT MaxDD ===")
    print(df.pivot_table(index="universe", columns="strategy", values="max_dd").round(3).to_string())


if __name__ == "__main__":
    main()
