"""Final comprehensive universe-matrix validation.

For each universe × {v3, A, B, C, A+B, A+C, A+B+C}, measure full-window
CAGR, Sharpe, MaxDD, WF mean CAGR, WF min, beats-SPY, etc.

Universes:
  sp500_pit        — deployed home (PIT)
  broader_1811     — Russell 3000 proxy (no PIT — survivor bias)
  non_sp500        — never in S&P 500 at asof
  qqq_tech         — Nasdaq-100 representative
  iyw_tech         — iShares US tech
  tech_broad       — union of QQQ + IYW + IGM extras
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
    load_spy_features, simulate,
)
from universes import QQQ_TECH, IYW, TECH_BROAD  # noqa: E402

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def filter_universe(panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    s = set(tickers)
    return panel[panel["ticker"].isin(s)].copy()


def apply_chronos_filter(panel: pd.DataFrame, chr_df: pd.DataFrame, q: float) -> pd.DataFrame:
    m = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]],
                    on=["asof", "ticker"], how="left").copy()
    m["chr_p70_rk"] = m.groupby("asof")["chronos_p70_3m"].rank(pct=True)
    out = m[m["chr_p70_rk"].fillna(0.0) >= q].copy()
    return out[["asof", "ticker", "score", "vol_1y"]]


def run_strategy(label, weighting, kb, q, panel, chr_df, mr, spy):
    """Apply optional chronos filter then simulate."""
    if q > 0 and chr_df is not None:
        p = apply_chronos_filter(panel, chr_df, q)
    else:
        p = panel
    cfg = V6Config(
        name=label, scorer="ml_3plus6", regime_gate="tight",
        k_normal=3, k_recovery=3, k_bull=kb, weighting=weighting,
        hold_months=6, cost_bps=10.0,
        cash_yield_yr=(0.03 if weighting == "invvol" else 0.0),
    )
    eq = simulate(cfg, p, mr, spy)
    spy_aln = build_spy_aligned(eq, mr)
    m = evaluate(eq, spy_aln, label)
    return m, eq


def main():
    print("[load core]")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    chr_sp_path = PIT / "ml_preds_chronos.parquet"
    chr_broad_path = V2 / "ml_preds_chronos_broader.parquet"
    chr_sp = pd.read_parquet(chr_sp_path) if chr_sp_path.exists() else None
    chr_broad = pd.read_parquet(chr_broad_path) if chr_broad_path.exists() else None
    if chr_sp is not None: chr_sp["asof"] = pd.to_datetime(chr_sp["asof"])
    if chr_broad is not None: chr_broad["asof"] = pd.to_datetime(chr_broad["asof"])

    print(f"  chronos sp500_pit: {chr_sp.shape if chr_sp is not None else 'MISSING'}")
    print(f"  chronos broader:   {chr_broad.shape if chr_broad is not None else 'MISSING'}")

    panel_sp = load_score_panel("ml_3plus6", "sp500_pit")
    panel_broad = load_score_panel("ml_3plus6", "broader")
    panel_nonsp = load_score_panel("ml_3plus6", "non_sp500")
    panel_qqq = filter_universe(panel_broad, QQQ_TECH)
    panel_iyw = filter_universe(panel_broad, IYW)
    panel_tech = filter_universe(panel_broad, TECH_BROAD)

    print(f"  panel sp500_pit: {len(panel_sp):,} rows, {panel_sp['ticker'].nunique()} tickers")
    print(f"  panel broader:   {len(panel_broad):,} rows, {panel_broad['ticker'].nunique()} tickers")
    print(f"  panel non_sp500: {len(panel_nonsp):,} rows, {panel_nonsp['ticker'].nunique()} tickers")
    print(f"  panel qqq_tech:  {len(panel_qqq):,} rows, {panel_qqq['ticker'].nunique()} tickers")
    print(f"  panel iyw_tech:  {len(panel_iyw):,} rows, {panel_iyw['ticker'].nunique()} tickers")
    print(f"  panel tech_broad:{len(panel_tech):,} rows, {panel_tech['ticker'].nunique()} tickers")

    universes = [
        ("sp500_pit",   panel_sp,    chr_sp),
        ("broader",     panel_broad, chr_broad),
        ("non_sp500",   panel_nonsp, chr_broad),
        ("qqq_tech",    panel_qqq,   chr_broad),
        ("iyw_tech",    panel_iyw,   chr_broad),
        ("tech_broad",  panel_tech,  chr_broad),
    ]

    # Strategies: (label, weighting, kb, q)
    strats = [
        ("v3",       "ew",     3, 0.0),
        ("A",        "invvol", 3, 0.0),
        ("B",        "invvol", 2, 0.0),
        ("C",        "ew",     3, 0.4),
        ("A_plus_C", "invvol", 3, 0.4),
        ("B_plus_C", "invvol", 2, 0.4),
    ]

    rows = []
    for u_name, panel, chr_df in universes:
        for label, w, kb, q in strats:
            if q > 0 and chr_df is None:
                print(f"[{u_name}|{label}] SKIP (no chronos)")
                continue
            t0 = time.time()
            try:
                m, _ = run_strategy(label, w, kb, q, panel, chr_df, mr, spy)
                m["universe"] = u_name
                m["strategy"] = label
                rows.append(m)
                print(f"[{u_name:11s}|{label:9s}] {time.time()-t0:.1f}s  "
                      f"CAGR={m['cagr_full']*100:6.2f}% Sh={m['sharpe']:.3f} "
                      f"MDD={m['max_dd']*100:6.2f}% WFmean={m['wf_mean_cagr']*100:6.2f}% "
                      f"WFmin={m['wf_min_cagr']*100:6.2f}% beats={m['wf_n_beats_spy']}/{m['wf_n_splits']}")
            except Exception as e:
                print(f"[{u_name}|{label}] ERROR: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "universe_matrix.csv", index=False)

    print("\n=== Full CAGR ===")
    print(df.pivot_table(index="universe", columns="strategy", values="cagr_full").round(4).to_string())
    print("\n=== Sharpe ===")
    print(df.pivot_table(index="universe", columns="strategy", values="sharpe").round(3).to_string())
    print("\n=== MaxDD ===")
    print(df.pivot_table(index="universe", columns="strategy", values="max_dd").round(3).to_string())
    print("\n=== WF mean CAGR ===")
    print(df.pivot_table(index="universe", columns="strategy", values="wf_mean_cagr").round(4).to_string())
    print("\n=== Beats SPY (count) ===")
    print(df.pivot_table(index="universe", columns="strategy", values="wf_n_beats_spy").to_string())


if __name__ == "__main__":
    main()
