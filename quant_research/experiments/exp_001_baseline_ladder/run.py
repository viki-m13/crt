"""
Experiment 001 — Baseline Ladder
Five rungs of increasing complexity, each walked forward honestly.
No hyperparameter tuning in this script — design choices only.
"""
from __future__ import annotations
import sys, json, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from backtest.engine import (
    BacktestConfig, simulate, run_walk_forward, compute_metrics,
    _load_features, _load_monthly_prices, _load_pit,
)

OUT = pathlib.Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# OOS window: 2008-01 → 2024-01 (2024-02 onward is lockbox territory)
OOS_START = "2008-01-31"
OOS_END   = "2024-01-31"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _rank_score(series: pd.Series) -> pd.Series:
    """Cross-sectional rank (0→1, higher = better)."""
    return series.rank(pct=True)


def _zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-9:
        return series * 0
    return (series - mu) / sigma


# ── RUNG 1: 12-1 Momentum ────────────────────────────────────────────────────

def score_rung1(feats: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Pure 12-1 momentum (Jegadeesh-Titman)."""
    if "mom_12_1" not in feats.columns:
        return pd.Series(dtype=float)
    return _rank_score(feats["mom_12_1"])


# ── RUNG 2: Momentum + Low-Vol filter ────────────────────────────────────────

def score_rung2(feats: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """12-1 momentum, filtered to bottom 2/3 by 1y vol."""
    if "mom_12_1" not in feats.columns or "vol_1y" not in feats.columns:
        return score_rung1(feats, asof)
    vol_rank = feats["vol_1y"].rank(pct=True)
    eligible = vol_rank[vol_rank <= 0.67].index
    sub = feats.loc[eligible]
    if len(sub) == 0:
        return score_rung1(feats, asof)
    return _rank_score(sub["mom_12_1"])


# ── RUNG 3: Momentum + Low-Vol + Quality ─────────────────────────────────────

def score_rung3(feats: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """12-1 mom + low-vol filter + quality score (composite)."""
    req = ["mom_12_1", "vol_1y", "sharpe_1y", "trend_health_5y", "quality_score_5y"]
    avail = [c for c in req if c in feats.columns]
    if "mom_12_1" not in avail:
        return score_rung2(feats, asof)

    # Low-vol filter: remove top third by vol
    vol_rank = feats["vol_1y"].rank(pct=True) if "vol_1y" in feats.columns else pd.Series(0.5, index=feats.index)
    eligible = vol_rank[vol_rank <= 0.67].index
    sub = feats.loc[eligible]
    if len(sub) == 0:
        return score_rung1(feats, asof)

    # Composite: 50% momentum + 50% quality
    mom = _rank_score(sub["mom_12_1"])
    qual_parts = []
    for col in ["sharpe_1y", "trend_health_5y", "quality_score_5y"]:
        if col in sub.columns:
            qual_parts.append(_rank_score(sub[col]))
    if qual_parts:
        qual = pd.concat(qual_parts, axis=1).mean(axis=1)
        return 0.5 * mom + 0.5 * qual
    return mom


# ── RUNG 4: Momentum + Low-Vol + Quality + Regime Gate ───────────────────────
# Regime gate is handled by the engine (BacktestConfig.regime_gate=True)
# Score function is identical to rung3; engine switches to cash when SPY < 200dma

score_rung4 = score_rung3   # engine applies regime; config uses regime_gate=True


# ── RUNG 5: Cross-Sectional OLS composite ─────────────────────────────────────

_ols_weights: dict[str, float] = {}  # trained in-sample, updated at each rebalance

def score_rung5(feats: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """
    Walk-forward cross-sectional OLS: regress 1m fwd return on factor ranks,
    then use the fitted coefficients to score at asof.

    Training uses all PIT data strictly BEFORE asof (no embargo needed for
    cross-sectional OLS on non-overlapping monthly windows).
    """
    global _ols_weights

    FACTORS = [
        "mom_12_1", "mom_6_1", "mom_3", "vol_1y",
        "sharpe_1y", "trend_health_5y", "d_sma200", "rsi_14",
    ]
    avail_factors = [f for f in FACTORS if f in feats.columns]
    if not avail_factors:
        return score_rung3(feats, asof)

    # Load training data: all feature snapshots strictly before asof
    feat_dir = pathlib.Path(__file__).parents[2].parent / "experiments/monthly_dca/cache/features"
    feat_files = sorted(feat_dir.glob("*.parquet"))
    train_records = []
    px = _load_monthly_prices()
    monthly_ret = px.pct_change()

    for ff in feat_files:
        t = pd.Timestamp(ff.stem)
        if t >= asof:
            break
        if t < asof - pd.DateOffset(years=5):
            continue  # use 5-year rolling window
        # Forward return for t→next month
        # Find next month-end
        idx = monthly_ret.index.searchsorted(t)
        if idx + 1 >= len(monthly_ret):
            continue
        next_t = monthly_ret.index[idx + 1]
        fwd_sub = monthly_ret.loc[next_t]

        f = pd.read_parquet(ff)
        f = f.dropna(subset=avail_factors)
        # Rank-normalize features
        for col in avail_factors:
            f[col] = f[col].rank(pct=True)
        f["fwd"] = fwd_sub.reindex(f.index)
        f = f.dropna(subset=["fwd"])
        train_records.append(f[avail_factors + ["fwd"]])

    if len(train_records) < 12:
        return score_rung3(feats, asof)

    train = pd.concat(train_records, ignore_index=True)
    X = train[avail_factors].values
    y = train["fwd"].values

    # OLS via normal equations (ridge for stability)
    ridge = 1e-3
    A = X.T @ X + ridge * np.eye(len(avail_factors))
    b = X.T @ y
    try:
        coefs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return score_rung3(feats, asof)

    _ols_weights[asof.isoformat()] = dict(zip(avail_factors, coefs.tolist()))

    # Score at asof
    f_now = feats[avail_factors].copy()
    for col in avail_factors:
        f_now[col] = f_now[col].rank(pct=True)
    f_now = f_now.dropna()
    scores = f_now.values @ coefs
    return pd.Series(scores, index=f_now.index)


# ── Run all rungs ────────────────────────────────────────────────────────────

def run_rung(name: str, score_fn, cfg: BacktestConfig) -> dict:
    print(f"\n  Running {name}...")
    t0 = time.time()
    result = run_walk_forward(
        score_fn, cfg,
        oos_start=OOS_START, oos_end=OOS_END,
        window_years=5, step_years=2,
    )
    elapsed = time.time() - t0
    result["name"] = name
    result["elapsed_s"] = round(elapsed, 1)
    print(f"    CAGR={result.get('full',{}).get('cagr',0):.1%}  "
          f"WF_mean_CAGR={result.get('wf_mean_cagr',0):.1%}  "
          f"Sharpe={result.get('full',{}).get('sharpe',0):.3f}  "
          f"WF_Sharpe={result.get('wf_mean_sharpe',0):.3f}  "
          f"({elapsed:.1f}s)")
    return result


if __name__ == "__main__":
    results = []

    # Rung 1: pure momentum, no regime, EW
    cfg1 = BacktestConfig(k=10, weighting="ew", regime_gate=False, cost_bps=5.0)
    results.append(run_rung("R1_momentum_12_1", score_rung1, cfg1))

    # Rung 2: momentum + low-vol filter, no regime
    cfg2 = BacktestConfig(k=10, weighting="ew", regime_gate=False, cost_bps=5.0)
    results.append(run_rung("R2_mom_lowvol", score_rung2, cfg2))

    # Rung 3: momentum + low-vol + quality, no regime
    cfg3 = BacktestConfig(k=10, weighting="ew", regime_gate=False, cost_bps=5.0)
    results.append(run_rung("R3_mom_lowvol_quality", score_rung3, cfg3))

    # Rung 4: same + regime gate (invvol)
    cfg4 = BacktestConfig(k=10, weighting="invvol", regime_gate=True, cost_bps=5.0)
    results.append(run_rung("R4_regime_gate", score_rung4, cfg4))

    # Rung 4b: concentration K=5
    cfg4b = BacktestConfig(k=5, weighting="invvol", regime_gate=True, cost_bps=5.0)
    results.append(run_rung("R4b_k5_regime", score_rung4, cfg4b))

    # Rung 5: cross-sectional OLS composite (slow — does WF refit)
    cfg5 = BacktestConfig(k=10, weighting="invvol", regime_gate=True, cost_bps=5.0)
    results.append(run_rung("R5_ols_composite", score_rung5, cfg5))

    # Save
    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print("\n── Baseline Ladder Summary ──")
    print(f"{'Name':<30} {'CAGR':>7} {'WF_CAGR':>8} {'Sharpe':>7} {'WF_Sharpe':>10} {'MaxDD':>7}")
    for r in results:
        f = r.get("full", {})
        print(f"{r['name']:<30} {f.get('cagr',0):>7.1%} {r.get('wf_mean_cagr',0):>8.1%} "
              f"{f.get('sharpe',0):>7.3f} {r.get('wf_mean_sharpe',0):>10.3f} "
              f"{f.get('max_dd',0):>7.1%}")
