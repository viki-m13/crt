"""Validate Option C: v3 + Chronos-bolt-tiny p70 confidence filter.

Reproduces the v5 zc4cv winner end-to-end using ml_preds_chronos.parquet
once that file has been built by score_chronos_v2.py.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V6DIR = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6DIR))

from lib_engine import (  # noqa: E402
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = ROOT / "experiments" / "monthly_dca" / "v5" / "cache"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("[load v3 panel & chronos]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    panel["asof"] = pd.to_datetime(panel["asof"])

    chronos_path = PIT / "ml_preds_chronos.parquet"
    if not chronos_path.exists():
        raise SystemExit(f"missing: {chronos_path}")
    chr_df = pd.read_parquet(chronos_path)
    chr_df["asof"] = pd.to_datetime(chr_df["asof"])

    merged = panel.merge(chr_df, on=["asof", "ticker"], how="left")
    print(f"  panel rows={len(panel):,}  chronos rows={len(chr_df):,}  merged={len(merged):,}")
    print(f"  chronos coverage: {merged['chronos_p70_3m'].notna().mean()*100:.1f}%")

    # Cross-sectional rank of chronos_p70_3m per asof
    merged["chr_p70_rk"] = merged.groupby("asof")["chronos_p70_3m"].rank(pct=True)

    # Filter: chr_p70_rk >= q (keep top (1-q) fraction by Chronos confidence)
    quantiles = [0.3, 0.4, 0.5]
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    rows = []
    # v3 baseline (no filter)
    cfg0 = V6Config(name="v3", scorer="ml_3plus6", regime_gate="tight",
                    k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                    hold_months=6, cost_bps=10.0)
    eq = simulate(cfg0, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    rows.append(evaluate(eq, spy_aln, "v3"))
    print(f"[v3] CAGR={rows[-1]['cagr_full']*100:.2f}% Sh={rows[-1]['sharpe']:.3f} "
          f"WFmean={rows[-1]['wf_mean_cagr']*100:.2f}% beats={rows[-1]['wf_n_beats_spy']}/{rows[-1]['wf_n_splits']}")

    for q in quantiles:
        # Filter to picks with Chronos rank >= q
        filt = merged[merged["chr_p70_rk"].fillna(0.0) >= q].copy()
        # If filtered too thin, fall back to v3 (skip months)
        cfg = V6Config(name=f"C_chr_q{q}_k3_h6_ew", scorer="ml_3plus6",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                       weighting="ew", hold_months=6, cost_bps=10.0)
        eq = simulate(cfg, filt[["asof", "ticker", "score", "vol_1y"]], monthly_returns, spy_feats)
        spy_aln = build_spy_aligned(eq, monthly_returns)
        m = evaluate(eq, spy_aln, cfg.name)
        m["q"] = q
        rows.append(m)
        print(f"[C_q{q}] CAGR={m['cagr_full']*100:.2f}% Sh={m['sharpe']:.3f} "
              f"WFmean={m['wf_mean_cagr']*100:.2f}% beats={m['wf_n_beats_spy']}/{m['wf_n_splits']} MDD={m['max_dd']*100:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "optionC_chronos_summary.csv", index=False)
    print("\n=== Summary ===")
    print(df[["name", "cagr_full", "sharpe", "max_dd", "wf_mean_cagr",
              "wf_min_cagr", "wf_n_beats_spy", "wf_mean_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
