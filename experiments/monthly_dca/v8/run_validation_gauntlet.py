"""Phase 3+4 final validation gauntlet for the exp_02 winner:

  ml_3plus6plus1, k=1, hold=1, regime=safer, weighting=invvol,
  crash_fallback=tlt, fallback_ticker=TLT, cost_bps=10.

Tests:
  (a) Robustness — nudge each hyperparameter ±20% (where meaningful).
  (b) Sub-period stability — yearly and per-decade CAGR.
  (c) Generalisation — replay on broader and non-S&P 500 universes.
  (d) Bias sensitivity — Monte-Carlo synthetic delisting at α∈{0..20}%/yr.
  (e) Frozen holdout — last 18 months (2024-11 → 2026-04) as a final
      single-shot OOS test that was never used for selection (the
      sweep only used 10 splits ending 2024-12).
  (f) Live-degradation forecast: 30%, 50% haircut on edge.

Outputs:
  experiments/monthly_dca/v8/results/validation_*.csv
  experiments/monthly_dca/v8/results/validation_summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys
from dataclasses import replace as _replace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (  # noqa: E402
    V2, V6Config, build_spy_aligned, evaluate,
    load_score_panel, load_spy_features, simulate, cagr_monthly,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def winner_cfg(**kwargs) -> V6Config:
    cfg = V6Config(
        name="exp_02_winner",
        scorer="ml_3plus6plus1", universe="sp500_pit",
        regime_gate="safer",
        k_normal=1, k_recovery=1, k_bull=1,
        weighting="invvol", hold_months=1, cost_bps=10.0,
        crash_fallback="tlt", fallback_ticker="TLT",
    )
    return _replace(cfg, **kwargs) if kwargs else cfg


def main():
    print("[load]")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    panel = load_score_panel("ml_3plus6plus1", "sp500_pit", attach_pullback=True)
    print(f"  panel rows={len(panel)} weeks={panel['asof'].nunique()}")

    cfg = winner_cfg()

    # ---- (a) Robustness sweep — nudge cost ±20%, scoring choice, regime ----
    print("\n[a] Robustness sweep (winner ± nudges)")
    rob_cfgs = [
        ("00_winner",          cfg),
        ("a1_cost_8",          _replace(cfg, cost_bps=8.0)),
        ("a2_cost_12",         _replace(cfg, cost_bps=12.0)),
        ("a3_cost_5",          _replace(cfg, cost_bps=5.0)),
        ("a4_cost_15",         _replace(cfg, cost_bps=15.0)),
        ("a5_scorer_3plus6",   _replace(cfg, scorer="ml_3plus6")),
        ("a6_scorer_h3",       _replace(cfg, scorer="ml_h3")),
        ("a7_scorer_h6",       _replace(cfg, scorer="ml_h6")),
        ("a8_regime_tight",    _replace(cfg, regime_gate="tight")),
        ("a9_regime_strict",   _replace(cfg, regime_gate="strict_dd")),
        ("a10_regime_combo",   _replace(cfg, regime_gate="combo")),
        ("a11_k2",             _replace(cfg, k_normal=2, k_recovery=2, k_bull=2)),
        ("a12_k3",             _replace(cfg, k_normal=3, k_recovery=3, k_bull=3)),
        ("a13_h2",             _replace(cfg, hold_months=2)),
        ("a14_h3",             _replace(cfg, hold_months=3)),
        ("a15_h6",             _replace(cfg, hold_months=6)),
        ("a16_fallback_cash",  _replace(cfg, crash_fallback="cash")),
        ("a17_fallback_spy",   _replace(cfg, crash_fallback="spy", fallback_ticker="SPY")),
        ("a18_weight_ew",      _replace(cfg, weighting="ew")),
    ]
    rob_rows = []
    for name, c in rob_cfgs:
        c2 = _replace(c, name=name)
        sc = c2.scorer
        p = panel if sc == "ml_3plus6plus1" else load_score_panel(sc, "sp500_pit")
        eq = simulate(c2, p, monthly_returns, spy_feats)
        spy_aln = build_spy_aligned(eq, monthly_returns)
        m = evaluate(eq, spy_aln, name)
        rob_rows.append(m)
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(OUT / "validation_robustness.csv", index=False)
    cols = ["name", "wf_mean_cagr", "wf_min_cagr", "wf_mean_sharpe",
            "max_dd", "wf_n_pos", "wf_n_beats_spy", "cagr_full"]
    print(rob_df[cols].to_string(index=False))

    # ---- (b) Sub-period stability ----
    print("\n[b] Sub-period stability (yearly CAGR of winner)")
    eq_w = simulate(cfg, panel, monthly_returns, spy_feats)
    eq_w["year"] = pd.DatetimeIndex(eq_w["date"]).year
    spy_aln_w = build_spy_aligned(eq_w, monthly_returns)
    spy_aln_w["year"] = pd.DatetimeIndex(spy_aln_w["date"]).year

    yearly_rows = []
    for y, sub in eq_w.groupby("year"):
        spy_sub = spy_aln_w[spy_aln_w["year"] == y]
        yearly_rows.append({
            "year": int(y),
            "ret_year": float((1 + sub["ret_m"].fillna(0)).prod() - 1),
            "cagr": cagr_monthly(sub["ret_m"]),
            "spy_ret_year": float((1 + spy_sub["spy_ret_m"].fillna(0)).prod() - 1),
            "n_months": int(len(sub)),
            "n_cash_months": int((sub["regime"] == "cash").sum()),
        })
    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(OUT / "validation_yearly.csv", index=False)
    print(yearly_df.to_string(index=False))
    print(f"  positive years: {int((yearly_df['ret_year'] > 0).sum())}/{len(yearly_df)}")
    print(f"  beat SPY years: {int((yearly_df['ret_year'] > yearly_df['spy_ret_year']).sum())}/{len(yearly_df)}")
    yearly_df["decade"] = (yearly_df["year"] // 10) * 10
    decade_df = yearly_df.groupby("decade").agg(
        years=("year", "count"),
        mean_ret=("ret_year", "mean"),
        median_ret=("ret_year", "median"),
        mean_spy_ret=("spy_ret_year", "mean"),
        positive_years=("ret_year", lambda x: int((x > 0).sum())),
    ).reset_index()
    decade_df.to_csv(OUT / "validation_decade.csv", index=False)
    print()
    print(decade_df.to_string(index=False))

    # ---- (c) Generalisation — broader / non-SP500 universes ----
    print("\n[c] Generalisation universes (winner config, different universe)")
    gen_rows = []
    for u in ("sp500_pit", "broader", "non_sp500"):
        try:
            p = load_score_panel("ml_3plus6plus1", u, attach_pullback=False)
            c2 = _replace(cfg, universe=u, name=f"gen_{u}")
            eq = simulate(c2, p, monthly_returns, spy_feats)
            spy_aln = build_spy_aligned(eq, monthly_returns)
            m = evaluate(eq, spy_aln, f"gen_{u}")
            m["universe"] = u
            gen_rows.append(m)
        except Exception as e:  # noqa: BLE001
            print(f"  {u}: SKIP ({e})")
    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(OUT / "validation_generalisation.csv", index=False)
    print(gen_df[["universe"] + cols[1:]].to_string(index=False))

    # ---- (d) Bias sensitivity (synthetic delisting at α∈{0..20}%/yr) ----
    print("\n[d] Survivorship bias sensitivity (Monte-Carlo synthetic delist)")
    bias_rows = []
    rng = np.random.default_rng(42)
    eq_base = simulate(cfg, panel, monthly_returns, spy_feats).reset_index(drop=True)
    for alpha in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20):
        # Per-month per-pick prob of synthetic delisting (geometric)
        p_m = 1 - (1 - alpha) ** (1 / 12)
        cagrs = []
        for trial in range(20):
            ret = np.array(eq_base["ret_m"].to_list(), dtype=float)
            for i, picks_str in enumerate(eq_base["picks"].fillna("")):
                if not picks_str:
                    continue
                k = picks_str.count(",") + 1
                # Each held pick has prob p_m of being synthetically delisted.
                # If a pick wipes (-100%), that 1/k slice realises -1; the
                # remaining k-1 picks realise their original return; we recompute
                # the basket return as the equal-weight average of {wiped, kept}.
                # Original basket return = ret[i] (already the EW average).
                # If `n_wipe` picks wipe out of k:
                #   kept_ret_avg = (k * ret[i] - n_wipe * 0) / (k - n_wipe)  (approx)
                # but EW of n_wipe (-1) and (k-n_wipe) of (kept_ret_avg) = ret_new
                # We approximate: each wipe subtracts 1/k from the basket return.
                n_wipe = sum(1 for _ in range(k) if rng.random() < p_m)
                if n_wipe > 0:
                    # Replace n_wipe slices of ret[i] with -1.0
                    ret[i] = ((k - n_wipe) * ret[i] + n_wipe * (-1.0)) / k
                    # Floor: monthly return cannot be worse than -100%
                    if ret[i] < -1.0:
                        ret[i] = -1.0
            ret_s = pd.Series(ret)
            cagrs.append(cagr_monthly(ret_s))
        cagrs = np.array(cagrs)
        bias_rows.append({
            "alpha_yr": alpha,
            "cagr_p10": float(np.percentile(cagrs, 10)),
            "cagr_median": float(np.percentile(cagrs, 50)),
            "cagr_p90": float(np.percentile(cagrs, 90)),
            "cagr_mean": float(np.mean(cagrs)),
        })
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(OUT / "validation_bias_sensitivity.csv", index=False)
    print(bias_df.to_string(index=False))

    # ---- (e) Frozen-holdout final shot ----
    print("\n[e] Frozen-holdout (2025-01 -> 2026-04) — strictly after the STRICT split end")
    eq_w = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln_w = build_spy_aligned(eq_w, monthly_returns)
    holdout_lo = pd.Timestamp("2025-01-01")
    holdout_hi = pd.Timestamp("2026-04-30")
    e = eq_w[(eq_w["date"] >= holdout_lo) & (eq_w["date"] <= holdout_hi)]
    s = spy_aln_w[(spy_aln_w["date"] >= holdout_lo) & (spy_aln_w["date"] <= holdout_hi)]
    print(f"  holdout months: {len(e)}")
    if len(e) > 0:
        ret_h = e["ret_m"]
        spy_h = s["spy_ret_m"]
        cagr_h = cagr_monthly(ret_h)
        spy_cagr_h = cagr_monthly(spy_h)
        sh_h = float(ret_h.mean() / max(ret_h.std(), 1e-9) * np.sqrt(12)) if ret_h.std() > 0 else 0.0
        eq_curve = (1 + ret_h.fillna(0)).cumprod()
        max_dd_h = float(((eq_curve - eq_curve.cummax()) / eq_curve.cummax()).min())
        holdout = {
            "n_months": int(len(e)),
            "cagr": cagr_h,
            "spy_cagr": spy_cagr_h,
            "edge_pp": (cagr_h - spy_cagr_h) * 100,
            "sharpe": sh_h,
            "max_dd": max_dd_h,
            "n_cash_months": int((e["regime"] == "cash").sum()),
            "monthly_returns": [float(r) for r in ret_h],
            "spy_monthly_returns": [float(r) for r in spy_h],
        }
    else:
        holdout = {"n_months": 0}
    print(json.dumps({k: v for k, v in holdout.items() if k not in ("monthly_returns","spy_monthly_returns")}, indent=2))
    (OUT / "validation_holdout.json").write_text(json.dumps(holdout, indent=2))

    # ---- (f) Live-degradation forecast ----
    print("\n[f] Live-degradation forecast")
    spy_full_cagr = cagr_monthly(spy_aln_w["spy_ret_m"])
    base_edge = (cagr_monthly(eq_w["ret_m"]) - spy_full_cagr) * 100
    deg = {
        "spy_cagr": float(spy_full_cagr),
        "raw_strategy_cagr": float(cagr_monthly(eq_w["ret_m"])),
        "raw_edge_pp": float(base_edge),
        "haircut_30pct_edge": float(spy_full_cagr + base_edge * 0.7 / 100),
        "haircut_50pct_edge": float(spy_full_cagr + base_edge * 0.5 / 100),
        "still_beats_spy_at_50pct": bool(spy_full_cagr + base_edge * 0.5 / 100 > spy_full_cagr),
    }
    print(json.dumps(deg, indent=2))

    # ---- Summary ----
    summary = {
        "winner_config": {
            "scorer": cfg.scorer,
            "universe": cfg.universe,
            "regime_gate": cfg.regime_gate,
            "k": cfg.k_normal,
            "hold_months": cfg.hold_months,
            "weighting": cfg.weighting,
            "cost_bps": cfg.cost_bps,
            "crash_fallback": cfg.crash_fallback,
            "fallback_ticker": cfg.fallback_ticker,
        },
        "winner_metrics": rob_df[rob_df["name"] == "00_winner"].iloc[0].to_dict(),
        "robustness_min_wf_mean_cagr": float(rob_df["wf_mean_cagr"].min()),
        "robustness_max_wf_mean_cagr": float(rob_df["wf_mean_cagr"].max()),
        "yearly_positive_count": int((yearly_df["ret_year"] > 0).sum()),
        "yearly_beat_spy_count": int((yearly_df["ret_year"] > yearly_df["spy_ret_year"]).sum()),
        "yearly_total": int(len(yearly_df)),
        "generalisation": gen_df[["name", "wf_mean_cagr", "wf_n_beats_spy"]].to_dict("records") if len(gen_df) else [],
        "bias_sensitivity": bias_df.to_dict("records"),
        "holdout": {k: v for k, v in holdout.items() if k not in ("monthly_returns","spy_monthly_returns")},
        "live_deg_forecast": deg,
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] saved -> {OUT}/validation_*")


if __name__ == "__main__":
    main()
