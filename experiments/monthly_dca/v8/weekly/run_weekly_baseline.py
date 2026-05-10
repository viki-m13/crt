"""Run the weekly simulator with various configs and report metrics.

This mirrors the structure of v6/run_baseline.py + tier{1,2,3} sweeps but
at weekly cadence using weekly_preds.parquet.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys
from dataclasses import replace as _replace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_weekly import (  # noqa: E402
    WConfig, simulate, evaluate, build_spy_aligned,
    _build_weekly_returns, _load_score_panel,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
WEEKLY_CACHE = Path(__file__).resolve().parent / "cache"


def main():
    print("[load] weekly returns + SPY features + score panel")
    weekly_px, weekly_ret = _build_weekly_returns()
    spy_feats = pd.read_parquet(WEEKLY_CACHE / "spy_features_weekly.parquet")
    panel = _load_score_panel("sp500_pit")
    print(f"  weekly_ret shape={weekly_ret.shape}")
    print(f"  panel rows={len(panel)} weeks={panel['asof'].nunique()} tickers={panel['ticker'].nunique()}")

    # Variants to test
    variants = [
        ("01_k1_h1_safer_tlt", WConfig(name="01", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=1, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("02_k1_h1_safer_cash", WConfig(name="02", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=1, regime_gate="safer", crash_fallback="cash")),
        ("03_k1_h2_safer_tlt", WConfig(name="03", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=2, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("04_k1_h4_safer_tlt", WConfig(name="04", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=4, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("05_k2_h1_safer_tlt", WConfig(name="05", k_normal=2, k_recovery=2, k_bull=2, hold_weeks=1, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("06_k2_h2_safer_tlt", WConfig(name="06", k_normal=2, k_recovery=2, k_bull=2, hold_weeks=2, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("07_k3_h1_safer_tlt", WConfig(name="07", k_normal=3, k_recovery=3, k_bull=3, hold_weeks=1, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("08_k3_h4_safer_tlt", WConfig(name="08", k_normal=3, k_recovery=3, k_bull=3, hold_weeks=4, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT")),
        ("09_k1_h1_tight_tlt", WConfig(name="09", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=1, regime_gate="tight", crash_fallback="tlt", fallback_ticker="TLT")),
        ("10_k1_h1_strict_tlt", WConfig(name="10", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=1, regime_gate="strict_dd", crash_fallback="tlt", fallback_ticker="TLT")),
        ("11_k1_h2_strict_tlt", WConfig(name="11", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=2, regime_gate="strict_dd", crash_fallback="tlt", fallback_ticker="TLT")),
        ("12_k1_h1_safer_tlt_halfwarn", WConfig(name="12", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=1, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT", half_cash_warning=True)),
        ("13_k1_h2_safer_tlt_halfwarn", WConfig(name="13", k_normal=1, k_recovery=1, k_bull=1, hold_weeks=2, regime_gate="safer", crash_fallback="tlt", fallback_ticker="TLT", half_cash_warning=True)),
    ]

    print(f"[run] {len(variants)} weekly variants")
    rows = []
    t0 = time.time()
    for name, cfg in variants:
        cfg2 = _replace(cfg, name=name)
        eq = simulate(cfg2, panel, weekly_ret, spy_feats)
        spy_aln = build_spy_aligned(eq, weekly_ret)
        m = evaluate(eq, spy_aln, name)
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(OUT / "weekly_baseline.csv", index=False)
    cols = ["name", "wf_mean_cagr", "wf_min_cagr", "wf_mean_sharpe", "max_dd",
            "wf_n_pos", "wf_n_beats_spy", "cagr_full", "spy_cagr_full"]
    print()
    print("All variants by WF mean CAGR (weekly, PIT S&P 500, GBM weekly preds):")
    print(df[cols].to_string(index=False))
    print()
    print(f"[done] in {time.time()-t0:.0f}s")

    # Floor pass
    floors = (
        (df["wf_min_cagr"] >= 0.0)
        & (df["wf_mean_sharpe"] >= 1.0)
        & (df["max_dd"] >= -0.50)
        & (df["wf_n_beats_spy"] >= 8)
    )
    df_pass = df[floors].sort_values("wf_mean_cagr", ascending=False)
    print(f"\n{len(df_pass)} pass floors. Top 5:")
    print(df_pass.head(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
