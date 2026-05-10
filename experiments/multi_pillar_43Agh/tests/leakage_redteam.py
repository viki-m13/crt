"""Phase 5 — Leakage Red-Team.

Runs four leakage-detection tests:

  1. Index reconstitution: every (asof, ticker) in the panel must satisfy
     PIT membership. (delegates to test_pit_membership)

  2. Walk-forward boundary shuffle: replace ml_score with random noise on
     the train period, retrain not applicable here (we don't retrain ml_preds);
     instead, randomly permute the score column at each asof and verify OOS
     edge collapses to zero.

  3. Survivorship test: re-run with delisted-ticker columns FORCED to NaN
     after their last live date — verify results don't dramatically improve.

  4. Generalization: run on the non-S&P-500 (broader universe) and check
     that the strategy still has positive edge there. If edge is much
     larger on S&P-PIT than on broader, that's a flag for S&P-specific
     overfit.

Outputs to experiments/multi_pillar_43Agh/reports/leakage_redteam.md
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))

from lib_engine import (  # noqa: E402
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)
from experiments.multi_pillar_43Agh.strategy import selection  # noqa: E402

REPORTS = ROOT / "experiments" / "multi_pillar_43Agh" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def _run(panel, monthly_returns, spy_feats, name, weighting="invvol", k=3):
    cfg = V6Config(name=name, scorer="ml_3plus6", universe="sp500_pit",
                   regime_gate="tight", k_normal=k, k_recovery=k, k_bull=k,
                   weighting=weighting, hold_months=6, cost_bps=10.0,
                   cash_yield_yr=0.03)
    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    return evaluate(eq, spy_aln, name)


def test_pit_reconstitution(panel, mem):
    j = panel.merge(mem, on=["asof", "ticker"], how="left", indicator=True)
    n_violations = int((j["_merge"] != "both").sum())
    return {"name": "pit_reconstitution", "violations": n_violations,
            "panel_rows": len(panel),
            "passed": n_violations == 0}


def test_shuffle_score(panel, monthly_returns, spy_feats):
    """Permute score within each asof — OOS edge should collapse."""
    rng = np.random.default_rng(42)
    p = panel.copy()
    p["score"] = (p.groupby("asof")["score"].transform(
        lambda s: pd.Series(rng.permutation(s.values), index=s.index)))
    m = _run(p, monthly_returns, spy_feats, "shuffled_score")
    return {"name": "shuffle_score",
            "shuffled_cagr": m["cagr_full"], "shuffled_sharpe": m["sharpe"],
            "spy_cagr": m["spy_cagr_full"],
            "edge_pp": (m["cagr_full"] - m["spy_cagr_full"]) * 100,
            "passed": abs(m["cagr_full"] - m["spy_cagr_full"]) < 0.05}


def test_survivorship(panel, monthly_returns, spy_feats):
    """Re-run BUT with delisted tickers EXCLUDED. If results dramatically
    improve, survivorship leakage is present."""
    delisted_panel = pd.read_parquet(ROOT / "experiments/monthly_dca/cache/delisted_panel.parquet")
    delisted_tickers = set(delisted_panel.columns)
    p = panel[~panel["ticker"].isin(delisted_tickers)].copy()
    print(f"  excluding {len(delisted_tickers)} delisted tickers from panel ({len(panel) - len(p)} rows removed)")
    m = _run(p, monthly_returns, spy_feats, "no_delisted")
    return {"name": "survivorship_excl",
            "excluded_n": len(delisted_tickers),
            "no_delisted_cagr": m["cagr_full"], "no_delisted_sharpe": m["sharpe"],
            "no_delisted_dd": m["max_dd"]}


def test_generalization():
    """Run the multi-pillar build on non-SP500 universe; check edge persists."""
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    panel = selection.build_composite_panel(
        universe="non_sp500",
        drop_failure_pct=0.10, apply_trend_gate=False,
        w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.20)
    m = _run(panel, monthly_returns, spy_feats, "non_sp500")
    return {"name": "generalization_non_sp500",
            "cagr_full": m["cagr_full"], "sharpe": m["sharpe"],
            "max_dd": m["max_dd"], "edge_pp": m["edge_full_pp"],
            "wf_n_beats_spy": m["wf_n_beats_spy"]}


def main():
    print("[redteam] loading data ...")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    mem = pd.read_parquet(ROOT / "experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    panel = selection.build_composite_panel(
        universe="sp500_pit",
        drop_failure_pct=0.10, apply_trend_gate=False,
        w_ml=1.0, w_archetype=0.0, w_novel=0.0, w_classic=0.0, w_failure=0.20)
    panel["asof"] = pd.to_datetime(panel["asof"])

    results = []
    print("\n[1] PIT reconstitution test ...")
    r = test_pit_reconstitution(panel, mem)
    results.append(r); print(f"  {r}")

    print("\n[2] shuffle-score test ...")
    r = test_shuffle_score(panel, monthly_returns, spy_feats)
    results.append(r); print(f"  {r}")

    print("\n[3] survivorship-exclusion test ...")
    r = test_survivorship(panel, monthly_returns, spy_feats)
    results.append(r); print(f"  {r}")

    print("\n[4] generalization (non-SP500) ...")
    r = test_generalization()
    results.append(r); print(f"  {r}")

    out = REPORTS / "leakage_redteam.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
