"""Phase 2 + 3 main runner.

Runs the v6 simulator under multiple configurations:
  baseline_v3      = ml_3plus6, K=3 EW, tight regime gate (V3 deployed)
  baseline_v6      = ml_3plus6, K=3 invvol, tight regime gate, cash 3% (V6 winner)
  pillar1_only     = baseline + drop bottom 30% by failure_score
  pillar2_only     = baseline + stock-level trend gate
  pillar4_only     = baseline + composite that adds archetype score
  pillars_1_2      = baseline + failure filter + trend gate
  pillars_1_2_4    = + archetype score
  pillars_1_2_3_4  = + novel-math features
  multi_pillar_full = pillars_1_2_3_4 + new market regime + concentration scaling

Each configuration emits metrics + walk-forward + drawdowns to
backtests/runs/<ts>_<name>/ and a summary row to backtests/experiment_log.csv.

Decomposition: subtracting each variant's headline metric from the
"baseline_v6" yields the marginal contribution of that pillar.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))

from lib_engine import (  # noqa: E402
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = ROOT / "experiments" / "multi_pillar_43Agh" / "backtests"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

from experiments.multi_pillar_43Agh.strategy import selection  # noqa: E402


def run_one(name: str, panel: pd.DataFrame, monthly_returns: pd.DataFrame,
            spy_feats: pd.DataFrame, weighting: str = "ew",
            cash_yield_yr: float = 0.0, k_normal: int = 3, k_recovery: int = 3,
            k_bull: int = 3, hold_months: int = 6,
            regime_gate: str = "tight") -> dict:
    cfg = V6Config(
        name=name,
        scorer="ml_3plus6",  # ignored when panel is pre-built
        universe="sp500_pit",
        regime_gate=regime_gate,
        k_normal=k_normal, k_recovery=k_recovery, k_bull=k_bull,
        weighting=weighting,
        hold_months=hold_months, cost_bps=10.0,
        cash_yield_yr=cash_yield_yr,
    )
    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    metrics = evaluate(eq, spy_aln, name)
    return {"cfg": cfg, "eq": eq, "metrics": metrics, "spy_aln": spy_aln}


def save_run(name: str, result: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder = RUNS / f"{ts}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    result["eq"].to_csv(folder / "equity.csv", index=False)
    (folder / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    return folder


def append_log(name: str, metrics: dict, notes: str = "") -> None:
    log_f = OUT / "experiment_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "cagr_full": metrics["cagr_full"],
        "sharpe": metrics["sharpe"],
        "max_dd": metrics["max_dd"],
        "wf_mean_cagr": metrics["wf_mean_cagr"],
        "wf_mean_sharpe": metrics["wf_mean_sharpe"],
        "wf_min_cagr": metrics["wf_min_cagr"],
        "wf_n_pos": metrics["wf_n_pos"],
        "wf_n_beats_spy": metrics["wf_n_beats_spy"],
        "n_cash": metrics["n_cash"],
        "notes": notes,
    }
    df = pd.DataFrame([row])
    if log_f.exists():
        df.to_csv(log_f, mode="a", header=False, index=False)
    else:
        df.to_csv(log_f, index=False)


def fmt_metrics(m: dict) -> str:
    return (f"CAGR={m['cagr_full']*100:6.2f}%  "
            f"Sharpe={m['sharpe']:5.3f}  "
            f"MaxDD={m['max_dd']*100:6.2f}%  "
            f"WFmeanCAGR={m['wf_mean_cagr']*100:6.2f}%  "
            f"WFmeanSh={m['wf_mean_sharpe']:5.3f}  "
            f"npos={m['wf_n_pos']}/10  beats={m['wf_n_beats_spy']}/10")


def main():
    print("[load] base data ...")
    t0 = time.time()
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    base_panel = load_score_panel("ml_3plus6", "sp500_pit")
    print(f"  base panel rows={len(base_panel)} ({time.time()-t0:.1f}s)")

    results = {}

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\n[run] baseline_v3 (ml_3plus6, K=3 EW, tight) ...")
    r = run_one("baseline_v3", base_panel, monthly_returns, spy_feats, weighting="ew", cash_yield_yr=0.0)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("baseline_v3", r); append_log("baseline_v3", r["metrics"])
    results["baseline_v3"] = r["metrics"]

    print("\n[run] baseline_v6 (ml_3plus6, K=3 invvol, tight, cash 3%) ...")
    r = run_one("baseline_v6", base_panel, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("baseline_v6", r); append_log("baseline_v6", r["metrics"])
    results["baseline_v6"] = r["metrics"]

    # ------------------------------------------------------------------
    # Pillar standalone tests
    # ------------------------------------------------------------------
    print("\n[panel] pillar1_only (failure filter, drop bottom 30%) ...")
    p_p1 = selection.build_composite_panel(
        drop_failure_pct=0.30, apply_trend_gate=False,
        w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.0,
    )
    print(f"  panel shape={p_p1.shape}")
    r = run_one("pillar1_only", p_p1, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillar1_only", r); append_log("pillar1_only", r["metrics"])
    results["pillar1_only"] = r["metrics"]

    print("\n[panel] pillar2_only (trend gate) ...")
    p_p2 = selection.build_composite_panel(
        drop_failure_pct=0.0, apply_trend_gate=True,
        w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.0,
    )
    print(f"  panel shape={p_p2.shape}")
    r = run_one("pillar2_only", p_p2, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillar2_only", r); append_log("pillar2_only", r["metrics"])
    results["pillar2_only"] = r["metrics"]

    print("\n[panel] pillar4_only (archetype 0.5 weight on score) ...")
    p_p4 = selection.build_composite_panel(
        drop_failure_pct=0.0, apply_trend_gate=False,
        w_ml=1.0, w_archetype=0.5, w_novel=0.0, w_classic=0.0, w_failure=0.0,
    )
    print(f"  panel shape={p_p4.shape}")
    r = run_one("pillar4_only", p_p4, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillar4_only", r); append_log("pillar4_only", r["metrics"])
    results["pillar4_only"] = r["metrics"]

    print("\n[panel] pillar3_only (novel features 0.5 weight) ...")
    p_p3 = selection.build_composite_panel(
        drop_failure_pct=0.0, apply_trend_gate=False,
        w_ml=1.0, w_archetype=0.0, w_novel=0.5, w_classic=0.0, w_failure=0.0,
    )
    print(f"  panel shape={p_p3.shape}")
    r = run_one("pillar3_only", p_p3, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillar3_only", r); append_log("pillar3_only", r["metrics"])
    results["pillar3_only"] = r["metrics"]

    # ------------------------------------------------------------------
    # Combined pillars
    # ------------------------------------------------------------------
    print("\n[panel] pillars_1_2 (failure filter + trend gate) ...")
    p_12 = selection.build_composite_panel(
        drop_failure_pct=0.30, apply_trend_gate=True,
        w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.0,
    )
    print(f"  panel shape={p_12.shape}")
    r = run_one("pillars_1_2", p_12, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillars_1_2", r); append_log("pillars_1_2", r["metrics"])
    results["pillars_1_2"] = r["metrics"]

    print("\n[panel] pillars_1_2_4 (+ archetype) ...")
    p_124 = selection.build_composite_panel(
        drop_failure_pct=0.30, apply_trend_gate=True,
        w_ml=1.0, w_archetype=0.4, w_novel=0.0, w_classic=0.2, w_failure=0.0,
    )
    print(f"  panel shape={p_124.shape}")
    r = run_one("pillars_1_2_4", p_124, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillars_1_2_4", r); append_log("pillars_1_2_4", r["metrics"])
    results["pillars_1_2_4"] = r["metrics"]

    print("\n[panel] pillars_1_2_3_4 (+ novel) ...")
    p_1234 = selection.build_composite_panel(
        drop_failure_pct=0.30, apply_trend_gate=True,
        w_ml=1.0, w_archetype=0.4, w_novel=0.3, w_classic=0.2, w_failure=0.0,
    )
    print(f"  panel shape={p_1234.shape}")
    r = run_one("pillars_1_2_3_4", p_1234, monthly_returns, spy_feats, weighting="invvol", cash_yield_yr=0.03)
    print(f"  {fmt_metrics(r['metrics'])}")
    save_run("pillars_1_2_3_4", r); append_log("pillars_1_2_3_4", r["metrics"])
    results["pillars_1_2_3_4"] = r["metrics"]

    # ------------------------------------------------------------------
    # Save decomposition table
    # ------------------------------------------------------------------
    print("\n[decomp] writing pillar decomposition ...")
    base = results["baseline_v6"]
    rows = []
    for name, m in results.items():
        rows.append({
            "config": name,
            "cagr_full": m["cagr_full"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "wf_mean_cagr": m["wf_mean_cagr"],
            "wf_mean_sharpe": m["wf_mean_sharpe"],
            "wf_n_pos": m["wf_n_pos"],
            "wf_n_beats_spy": m["wf_n_beats_spy"],
            "delta_cagr_vs_v6": m["cagr_full"] - base["cagr_full"],
            "delta_sharpe_vs_v6": m["sharpe"] - base["sharpe"],
            "delta_dd_vs_v6": m["max_dd"] - base["max_dd"],
        })
    decomp = pd.DataFrame(rows)
    decomp.to_csv(OUT / "pillar_decomposition.csv", index=False)
    decomp.to_parquet(OUT / "pillar_decomposition.parquet")
    print(decomp.to_string(index=False))


if __name__ == "__main__":
    main()
