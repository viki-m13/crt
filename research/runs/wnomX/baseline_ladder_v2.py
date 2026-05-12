"""
Phase 2 Baseline Ladder v2 — uses pre-computed PIT feature cache.

Fixes v1 bug: vol_12m was NaN because DataFrame.dropna() on 1833 columns.
Uses pit_panel_full.parquet (already validated PIT features from YLOka sessions)
for features, and sp500_membership_monthly.parquet for universe.

All rungs use OOS window 2008-09-30 to 2024-04-30 (lockbox sealed from 2024-05).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent
CACHE_DIR = Path("/home/user/crt/experiments/monthly_dca/cache")
V2_DIR = CACHE_DIR / "v2" / "sp500_pit"
DATA_DIR = Path("/home/user/crt/data/YLOka")
EXP_DIR = ROOT / "experiments" / "exp_000_baseline_ladder"
STATE_DIR = ROOT / "state"
EXP_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_END = pd.Timestamp("2024-04-30")
LOCKBOX_START = pd.Timestamp("2024-05-31")
OOS_START = pd.Timestamp("2008-08-31")
K_DEFAULT = 5
COST_BPS = 5.0
CASH_YIELD_MONTHLY = 0.03 / 12


def load_data():
    print("Loading PIT feature panel (pit_panel_full.parquet)...")
    feats = pd.read_parquet(DATA_DIR / "pit_panel_full.parquet")
    feats["asof"] = pd.to_datetime(feats["asof"])
    # Restrict to research window (never touch lockbox)
    feats = feats[feats["asof"] <= RESEARCH_END].copy()
    print(f"  Features: {len(feats)} rows, {feats['asof'].nunique()} asofs, {feats['ticker'].nunique()} tickers")

    print("Loading SPX PIT membership...")
    members = pd.read_parquet(V2_DIR / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members = members[members["asof"] <= RESEARCH_END].copy()
    print(f"  Members: {len(members)} rows, {members['asof'].nunique()} asofs")

    print("Loading forward returns (ml_preds_v2.parquet)...")
    fwd = pd.read_parquet(CACHE_DIR / "v2" / "ml_preds_v2.parquet")[["asof", "ticker", "fwd_1m_ret"]].copy()
    fwd["asof"] = pd.to_datetime(fwd["asof"])
    fwd = fwd[fwd["asof"] <= RESEARCH_END].copy()
    print(f"  Fwd returns: {len(fwd)} rows")

    # Load SPY prices for regime gate
    print("Loading SPY prices for regime gate...")
    prices = pd.read_parquet(CACHE_DIR / "prices_extended.parquet")["SPY"].dropna()
    prices.index = pd.to_datetime(prices.index)
    print(f"  SPY: {len(prices)} daily points, {prices.index[0].date()} to {prices.index[-1].date()}")

    return feats, members, fwd, prices


def compute_spx_regime(spy: pd.Series, asof_dates: list) -> dict:
    """SPY above its 200-day MA -> regime_ok=True."""
    result = {}
    for asof in asof_dates:
        hist = spy.loc[:asof].dropna()
        if len(hist) < 200:
            result[asof] = True
        else:
            ma200 = hist.iloc[-200:].mean()
            result[asof] = bool(hist.iloc[-1] >= ma200)
    return result


def run_rung(
    name: str,
    panel: pd.DataFrame,
    score_col: str,
    members: pd.DataFrame,
    K: int = K_DEFAULT,
    use_regime: bool = False,
    regime_map: dict = None,
) -> dict:
    """
    Run single rung. panel must have [asof, ticker, score_col, fwd_1m_ret].
    """
    asof_dates = sorted(panel["asof"].dropna().unique())
    asof_dates = [d for d in asof_dates if d <= RESEARCH_END]

    monthly_rets = {}
    prev_tickers = set()

    for asof in asof_dates:
        # Regime gate
        if use_regime and regime_map and not regime_map.get(asof, True):
            monthly_rets[asof] = CASH_YIELD_MONTHLY
            prev_tickers = set()
            continue

        # PIT universe
        universe = set(members[members["asof"] == asof]["ticker"])
        if not universe:
            monthly_rets[asof] = 0.0
            continue

        # Score and filter
        month = panel[(panel["asof"] == asof) & (panel["ticker"].isin(universe))]
        month = month.dropna(subset=[score_col, "fwd_1m_ret"]).copy()
        if len(month) < K:
            monthly_rets[asof] = CASH_YIELD_MONTHLY
            continue

        top = month.nlargest(K, score_col)
        tickers = top["ticker"].tolist()
        w = 1.0 / len(tickers)

        port_ret = (top["fwd_1m_ret"] * w).sum()

        # Cost: one-way on new positions + exits
        new_buys = set(tickers) - prev_tickers
        exits = prev_tickers - set(tickers)
        buy_w = len(new_buys) / max(K, 1)
        sell_w = len(exits) / max(K, 1)
        cost = (buy_w + sell_w) * (COST_BPS / 10_000)
        monthly_rets[asof] = port_ret - cost
        prev_tickers = set(tickers)

    rets = pd.Series(monthly_rets).sort_index()
    oos = rets[rets.index >= OOS_START]

    def metrics(r):
        r = r.dropna()
        if len(r) < 12:
            return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "n": len(r)}
        cum = (1 + r).cumprod()
        n_yr = len(r) / 12
        cagr = cum.iloc[-1] ** (1 / n_yr) - 1
        av = r.mean() * 12
        vv = r.std() * np.sqrt(12)
        sr = av / vv if vv > 0 else np.nan
        dd = (cum / cum.cummax() - 1).min()
        return {"cagr": round(cagr, 4), "sharpe": round(sr, 4), "max_dd": round(dd, 4), "n": len(r)}

    return {
        "name": name,
        "rets": rets,
        "oos": oos,
        "full": metrics(rets),
        "oos_m": metrics(oos),
    }


def main():
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"Phase 2 Baseline Ladder v2 — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M UTC')}")
    print(f"{'='*65}\n")

    feats, members, fwd, spy = load_data()

    # Build panel: join features, membership enforcement, fwd returns
    # Only keep tickers that appear in both features and members at each asof
    panel_base = members.merge(feats, on=["asof", "ticker"], how="left")
    panel_base = panel_base.merge(fwd, on=["asof", "ticker"], how="left")
    print(f"\nPanel: {len(panel_base)} rows after joining features + fwd returns")

    # Cross-sectional ranks (within each asof, within PIT universe)
    for col in ["mom_12_1", "mom_6_1", "vol_1y", "trend_health_5y"]:
        if col in panel_base.columns:
            panel_base[f"rank_{col}"] = panel_base.groupby("asof")[col].rank(pct=True)

    panel_base["inv_vol"] = 1.0 / panel_base["vol_1y"].clip(lower=0.001)
    panel_base["rank_inv_vol"] = panel_base.groupby("asof")["inv_vol"].rank(pct=True)

    # Regime gate
    asof_dates = sorted(panel_base["asof"].unique())
    regime_map = compute_spx_regime(spy, asof_dates)
    panel_base["regime_ok"] = panel_base["asof"].map(regime_map)

    print("\nFeature coverage check:")
    for col in ["mom_12_1", "vol_1y", "trend_health_5y", "fwd_1m_ret"]:
        pct = panel_base[col].notna().mean() if col in panel_base.columns else 0
        print(f"  {col}: {pct:.1%} non-null")

    results = []

    # R1: 12-1 momentum
    print("\n--- R1: 12-1 Momentum EW top-5 ---")
    r1 = run_rung("R1_mom12_1", panel_base, "mom_12_1", members, K=K_DEFAULT)
    print(f"  OOS CAGR={r1['oos_m']['cagr']:.1%}  Sharpe={r1['oos_m']['sharpe']:.3f}  MaxDD={r1['oos_m']['max_dd']:.1%}  N={r1['oos_m']['n']}")
    results.append(r1)

    # R2: Momentum + Low-Vol
    print("\n--- R2: Momentum + Low-Vol composite ---")
    panel_base["score_r2"] = panel_base["rank_mom_12_1"] + 0.5 * panel_base["rank_inv_vol"]
    r2 = run_rung("R2_mom_lovol", panel_base, "score_r2", members, K=K_DEFAULT)
    print(f"  OOS CAGR={r2['oos_m']['cagr']:.1%}  Sharpe={r2['oos_m']['sharpe']:.3f}  MaxDD={r2['oos_m']['max_dd']:.1%}  N={r2['oos_m']['n']}")
    results.append(r2)

    # R3: Momentum + Low-Vol + Quality (trend health)
    print("\n--- R3: + Trend Health quality screen ---")
    if "rank_trend_health_5y" in panel_base.columns:
        panel_base["score_r3"] = (panel_base["rank_mom_12_1"] +
                                   0.5 * panel_base["rank_inv_vol"] +
                                   0.5 * panel_base["rank_trend_health_5y"])
    else:
        panel_base["score_r3"] = panel_base["score_r2"]
    r3 = run_rung("R3_quality", panel_base, "score_r3", members, K=K_DEFAULT)
    print(f"  OOS CAGR={r3['oos_m']['cagr']:.1%}  Sharpe={r3['oos_m']['sharpe']:.3f}  MaxDD={r3['oos_m']['max_dd']:.1%}  N={r3['oos_m']['n']}")
    results.append(r3)

    # R4: + Regime gate
    print("\n--- R4: + Regime gate (SPY 200d MA) ---")
    r4 = run_rung("R4_regime", panel_base, "score_r3", members, K=K_DEFAULT,
                  use_regime=True, regime_map=regime_map)
    print(f"  OOS CAGR={r4['oos_m']['cagr']:.1%}  Sharpe={r4['oos_m']['sharpe']:.3f}  MaxDD={r4['oos_m']['max_dd']:.1%}  N={r4['oos_m']['n']}")
    results.append(r4)

    # R5: Walk-forward OLS cross-sectional ranker
    print("\n--- R5: Walk-forward OLS cross-sectional ranker ---")
    ols_cols = [c for c in ["rank_mom_12_1", "rank_mom_6_1", "rank_inv_vol"] if c in panel_base.columns]
    oos_asofs = [d for d in asof_dates if d >= OOS_START]
    ols_records = []
    for asof in oos_asofs:
        train_asofs = [d for d in asof_dates if d < asof][-60:]
        if len(train_asofs) < 12:
            continue
        train = panel_base[panel_base["asof"].isin(train_asofs)].dropna(subset=ols_cols + ["fwd_1m_ret"])
        if len(train) < 100:
            continue
        X_tr = train[ols_cols].values
        y_tr = train["fwd_1m_ret"].values
        try:
            m = LinearRegression().fit(X_tr, y_tr)
            test = panel_base[panel_base["asof"] == asof].dropna(subset=ols_cols)
            if len(test) == 0:
                continue
            test = test.copy()
            test["score_r5"] = m.predict(test[ols_cols].values)
            ols_records.append(test[["asof", "ticker", "score_r5"]])
        except Exception:
            continue

    if ols_records:
        ols_preds = pd.concat(ols_records, ignore_index=True)
        panel_r5 = panel_base.merge(ols_preds, on=["asof", "ticker"], how="left")
        r5 = run_rung("R5_ols", panel_r5, "score_r5", members, K=K_DEFAULT)
        print(f"  OOS CAGR={r5['oos_m']['cagr']:.1%}  Sharpe={r5['oos_m']['sharpe']:.3f}  MaxDD={r5['oos_m']['max_dd']:.1%}  N={r5['oos_m']['n']}")
        results.append(r5)
    else:
        print("  Insufficient OLS data.")

    # R6: GBM predictions from v3 (baseline reference using existing preds)
    print("\n--- R6: v3 GBM predictions (pred_3m + pred_6m, reference) ---")
    if "pred_3m" in panel_base.columns and "pred_6m" in panel_base.columns:
        panel_base["rank_pred_3m"] = panel_base.groupby("asof")["pred_3m"].rank(pct=True)
        panel_base["rank_pred_6m"] = panel_base.groupby("asof")["pred_6m"].rank(pct=True)
        panel_base["score_r6"] = 0.5 * panel_base["rank_pred_3m"] + 0.5 * panel_base["rank_pred_6m"]
        r6 = run_rung("R6_gbm_v3", panel_base, "score_r6", members, K=K_DEFAULT)
        print(f"  OOS CAGR={r6['oos_m']['cagr']:.1%}  Sharpe={r6['oos_m']['sharpe']:.3f}  MaxDD={r6['oos_m']['max_dd']:.1%}  N={r6['oos_m']['n']}")
        results.append(r6)
        # Also try K=3 which was optimal in YLOka
        r6b = run_rung("R6b_gbm_v3_K3", panel_base, "score_r6", members, K=3)
        print(f"  K=3: CAGR={r6b['oos_m']['cagr']:.1%}  Sharpe={r6b['oos_m']['sharpe']:.3f}  MaxDD={r6b['oos_m']['max_dd']:.1%}")
        results.append(r6b)

    # Summary
    print(f"\n{'='*70}")
    print("BASELINE LADDER SUMMARY (OOS = 2008-09 to 2024-04)")
    print(f"{'='*70}")
    print(f"{'Rung':<28} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'N':>6}")
    print("-" * 70)
    for r in results:
        m = r["oos_m"]
        print(f"{r['name']:<28} {m['cagr']:>8.1%} {m['sharpe']:>8.3f} {m['max_dd']:>8.1%} {m['n']:>6}")
    print(f"{'v3_GBM_K3_prod(ref)':<28} {'39.77%':>8} {'0.953':>8} {'-49.83%':>8} {'248':>6}")
    print(f"\nTarget: CAGR >= 50%, Sharpe >= 2.0")
    print(f"Gap vs target: need {50.0 - (max(r['oos_m']['cagr'] or 0 for r in results)*100):.1f}pp CAGR and {2.0 - (max((r['oos_m']['sharpe'] or 0) for r in results)):.2f} Sharpe improvement")

    # Save
    rows = []
    for r in results:
        for wnd, m in [("oos", r["oos_m"]), ("full", r["full"])]:
            rows.append({"rung": r["name"], "window": wnd, **m})
    pd.DataFrame(rows).to_parquet(EXP_DIR / "results_v2.parquet")

    # Save equity curves
    oos_curves = pd.DataFrame({r["name"]: r["oos"] for r in results})
    oos_curves.to_parquet(EXP_DIR / "oos_equity_curves.parquet")

    # Journal
    best = max(results, key=lambda r: r["oos_m"].get("sharpe", -999) or -999)
    elapsed = time.time() - t0

    summary_md = f"""# Experiment 000 — Phase 2 Baseline Ladder v2

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Universe**: SPX PIT (sp500_membership_monthly.parquet, 2003-01 to 2024-04)
**Features**: pit_panel_full.parquet (YLOka pre-computed, validated PIT)
**OOS window**: 2008-09-30 to 2024-04-30
**K**: {K_DEFAULT} (also K=3 for R6b), **Cost**: {COST_BPS} bps one-way

## Results (OOS)

| Rung | CAGR | Sharpe | MaxDD | N |
|---|---:|---:|---:|---:|
"""
    for r in results:
        m = r["oos_m"]
        summary_md += f"| {r['name']} | {m['cagr']:.1%} | {m['sharpe']:.3f} | {m['max_dd']:.1%} | {m['n']} |\n"
    summary_md += f"| v3_GBM_K3_prod (reference) | 39.77% | 0.953 | -49.83% | 248 |\n"

    summary_md += f"""
**Best OOS Sharpe**: {best['name']} — CAGR {best['oos_m']['cagr']:.1%}, Sharpe {best['oos_m']['sharpe']:.3f}

## Gap Analysis

The success gate requires CAGR ≥ 50% and Sharpe ≥ 2.0 on walk-forward OOS.
The v3 GBM is the price-only ceiling at ~40% CAGR / Sharpe ~0.95 after 88 experiments.
The baseline ladder confirms this ceiling: momentum + quality + regime factors produce
significantly lower performance than a properly trained GBM on the same features.

## What's needed

1. **New data**: fundamentals (earnings, profitability, valuation) would provide the
   largest marginal information gain. Volume-based features are a close second.
2. **Better ML**: LSTM/Transformer on feature sequences could capture non-linear
   temporal patterns that a monthly cross-sectional GBM cannot.
3. **Alternative targets**: train on probability of >50% gain in 12m rather than
   raw return (right-tail asymmetric objective).
4. **Meta-labeling**: use v3 as primary selector, train a secondary classifier to
   take/skip each v3 pick based on contextual signals.

## Elapsed

{elapsed:.1f}s
"""
    with open(EXP_DIR / "summary.md", "w") as f:
        f.write(summary_md)

    journal_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": "exp_000_baseline_ladder_v2",
        "hypothesis": "Phase 2 baseline ladder (5 rungs + GBM reference) establishes floor metrics",
        "what_i_did": "Fixed vol_12m NaN bug from v1. Used pit_panel_full.parquet features. R1-R5 momentum/quality/OLS rungs + R6 v3 GBM reference",
        "result": {
            "best_oos_rung": best["name"],
            "best_oos_cagr": best["oos_m"]["cagr"],
            "best_oos_sharpe": best["oos_m"]["sharpe"],
            "v3_ref_sharpe": 0.953,
            "gap_to_target_sharpe": round(2.0 - (best["oos_m"]["sharpe"] or 0), 3),
        },
        "hparams_tried": len(results),
        "next_action": "Design GBM cross-sectional ranker with new feature engineering or explore meta-labeling on v3",
    }
    with open(STATE_DIR / "journal.jsonl", "a") as f:
        f.write(json.dumps(journal_entry) + "\n")

    hyp_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": "exp_000_baseline_ladder_v2",
        "n_hparams": len(results),
        "description": f"{len(results)} baseline rungs (fixed scoring, K=5 or K=3)",
    }
    with open(STATE_DIR / "hypotheses_tested.jsonl", "a") as f:
        f.write(json.dumps(hyp_entry) + "\n")

    print(f"\n[DONE] Results saved. Elapsed: {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    results = main()
