"""
Phase 2 Baseline Ladder — honest walk-forward 2003-09 → 2024-04.

Data: pit_panel_with_scores.parquet (PIT-filtered, v3 ML predictions)
      pit_panel_full.parquet (same universe + 47 features)
Regime gate: 'tight' = exact YLOka/v3 crash detector (SPY 21d + 6m signals).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import (
    BacktestConfig, run_backtest, summary_str,
    load_pit_scores_panel, load_pit_full_panel,
)

REPO = Path("/home/user/crt")
QR   = REPO / "quant_research"
JOURNAL  = QR / "state" / "journal.jsonl"
HYP_LOG  = QR / "state" / "hypotheses_tested.jsonl"


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def score_mom_12_1(grp: pd.DataFrame) -> pd.Series:
    df = grp.set_index("ticker")
    return df["mom_12_1"] if "mom_12_1" in df.columns else pd.Series(dtype=float)


def score_lowvol_mom(grp: pd.DataFrame) -> pd.Series:
    df = grp.set_index("ticker")
    if "mom_12_1" not in df.columns or "vol_1y" not in df.columns:
        return pd.Series(dtype=float)
    return df.loc[df["vol_1y"] <= df["vol_1y"].median(), "mom_12_1"]


def score_quality_lowvol_mom(grp: pd.DataFrame) -> pd.Series:
    df = grp.set_index("ticker")
    if "mom_12_1" not in df.columns:
        return pd.Series(dtype=float)
    mask = pd.Series(True, index=df.index)
    if "vol_1y" in df.columns:
        mask &= df["vol_1y"] <= df["vol_1y"].median()
    if "sharpe_1y" in df.columns:
        mask &= df["sharpe_1y"] > 0
    return df.loc[mask, "mom_12_1"]


def score_ml_3plus6(grp: pd.DataFrame) -> pd.Series:
    """v3 production signal: raw average of pred_3m and pred_6m."""
    df = grp.set_index("ticker")
    if "pred_3m" not in df.columns or "pred_6m" not in df.columns:
        return pd.Series(dtype=float)
    return (df["pred_3m"] + df["pred_6m"]) / 2.0


def score_ml_136(grp: pd.DataFrame) -> pd.Series:
    df = grp.set_index("ticker")
    cols = [c for c in ["pred_1m", "pred_3m", "pred_6m"] if c in df.columns]
    return df[cols].mean(axis=1) if cols else pd.Series(dtype=float)


def score_ml_quality_lowvol(grp: pd.DataFrame) -> pd.Series:
    """ML 3plus6 + quality (sharpe_1y > 0) + low-vol filter."""
    df = grp.set_index("ticker")
    if "pred_3m" not in df.columns or "pred_6m" not in df.columns:
        return pd.Series(dtype=float)
    base = (df["pred_3m"] + df["pred_6m"]) / 2.0
    if "sharpe_1y" in df.columns:
        base = base[df["sharpe_1y"] > 0]
    if "vol_1y" in df.columns:
        base = base[df["vol_1y"] <= df["vol_1y"].median()]
    return base


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def log_journal(entry: dict):
    _log(JOURNAL, entry)


def log_hypotheses(n: int, description: str):
    _log(HYP_LOG, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_hparams": n,
        "description": description,
    })


# ---------------------------------------------------------------------------
# Baseline configs
# ---------------------------------------------------------------------------

CONFIGS = [
    # --- JT momentum rungs (no ML, using full feature panel) ---
    BacktestConfig(
        name="rung1_jt_K10_monthly",
        score_fn=score_mom_12_1,
        K=10, weighting="ew", regime_gate="none", hold_months=1,
    ),
    BacktestConfig(
        name="rung2_jt_lowvol_K10",
        score_fn=score_lowvol_mom,
        K=10, weighting="ew", regime_gate="none", hold_months=1,
    ),
    BacktestConfig(
        name="rung3_jt_lvq_K10",
        score_fn=score_quality_lowvol_mom,
        K=10, weighting="ew", regime_gate="none", hold_months=1,
    ),
    BacktestConfig(
        name="rung4_jt_lvq_regime_K10",
        score_fn=score_quality_lowvol_mom,
        K=10, weighting="ew", regime_gate="tight", hold_months=1,
    ),
    # --- v3 ML signal rungs ---
    BacktestConfig(
        name="rung5a_mlv3_K3_h6_tight",    # YLOka v3 baseline replica
        score_fn=score_ml_3plus6,
        K=3, weighting="ew", regime_gate="tight", hold_months=6,
    ),
    BacktestConfig(
        name="rung5b_mlv3_K5_h6_tight",
        score_fn=score_ml_3plus6,
        K=5, weighting="ew", regime_gate="tight", hold_months=6,
    ),
    BacktestConfig(
        name="rung5c_mlv3_K10_h6_tight",
        score_fn=score_ml_3plus6,
        K=10, weighting="ew", regime_gate="tight", hold_months=6,
    ),
    BacktestConfig(
        name="rung5d_mlv3_K3_h3_tight",
        score_fn=score_ml_3plus6,
        K=3, weighting="ew", regime_gate="tight", hold_months=3,
    ),
    BacktestConfig(
        name="rung5e_mlv3_K3_h1_tight",    # monthly rebalance, concentrated
        score_fn=score_ml_3plus6,
        K=3, weighting="ew", regime_gate="tight", hold_months=1,
    ),
    # --- 1m+3m+6m ensemble ---
    BacktestConfig(
        name="rung6_ml136_K3_h6_tight",
        score_fn=score_ml_136,
        K=3, weighting="ew", regime_gate="tight", hold_months=6,
    ),
    # --- ML + quality/lowvol filter ---
    BacktestConfig(
        name="rung7_ml_qlv_K3_h6_tight",
        score_fn=score_ml_quality_lowvol,
        K=3, weighting="ew", regime_gate="tight", hold_months=6,
    ),
    # --- invvol weighting ---
    BacktestConfig(
        name="rung8_ml_invvol_K5_h6_tight",
        score_fn=score_ml_3plus6,
        K=5, weighting="invvol", regime_gate="tight", hold_months=6,
    ),
]


def run_all_baselines() -> list[dict]:
    print("Loading PIT scores panel...")
    pit_scores = load_pit_scores_panel()
    print(f"  pit_panel_with_scores: {pit_scores.shape}, "
          f"asofs={pit_scores['asof'].nunique()}")

    print("Loading PIT full feature panel...")
    pit_full = load_pit_full_panel()
    print(f"  pit_panel_full: {pit_full.shape}")

    # Which panel to use per config
    ML_CONFIGS = {"ml", "136", "qlv", "invvol"}

    results = []
    print(f"\nRunning {len(CONFIGS)} baseline configs on research window 2003-09 → 2024-04...")
    for cfg in CONFIGS:
        uses_ml = any(s in cfg.name for s in ML_CONFIGS)
        panel = pit_scores if uses_ml else pit_full
        print(f"  {cfg.name}...")
        r = run_backtest(cfg, panel, verbose=False)
        print("    " + summary_str(r))
        results.append(r)

    log_hypotheses(len(CONFIGS), "Phase 2 baseline ladder: JT + ML + K/h/regime grid")
    log_journal({
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": "exp_001_baseline_ladder",
        "hypothesis": "Phase 2 baseline ladder: JT → lowvol+quality → regime gate → ML v3",
        "what_i_did": (f"Ran {len(CONFIGS)} baseline configs, research window 2003-09 → 2024-04; "
                       "added regime_tight gate from YLOka harness (SPY 21d+6m signals)"),
        "result": [{"name": r["name"], "cagr": round(r["cagr"], 4),
                    "sharpe": round(r["sharpe"], 4), "maxdd": round(r["maxdd"], 4)}
                   for r in results],
        "hparams_tried": len(CONFIGS),
        "next_action": "Identify best rung as benchmark; proceed to Phase 3 ideas",
    })

    return results


if __name__ == "__main__":
    results = run_all_baselines()
    print("\n" + "=" * 70)
    print("BASELINE LADDER SUMMARY (Phase 2)")
    print("=" * 70)
    for r in results:
        print(summary_str(r))

    # Find best by Sharpe
    best = max(results, key=lambda r: r["sharpe"])
    print(f"\nBest by Sharpe: {summary_str(best)}")
    best_cagr = max(results, key=lambda r: r["cagr"])
    print(f"Best by CAGR: {summary_str(best_cagr)}")
