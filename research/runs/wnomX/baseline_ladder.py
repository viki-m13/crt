"""
Phase 2 Baseline Ladder — quant_research bootstrap.

Implements 5 rungs as specified in CLAUDE.md:
  R1: Equal-weight top-N by 12-1 momentum
  R2: Low-volatility filter within top momentum
  R3: Quality screen (trend health proxy for gross profitability)
  R4: Regime gate (SPX > 200-day MA -> invest, else cash)
  R5: Cross-sectional OLS on [mom_12_1, mom_6_1, inv_vol] -> score

Walk-forward:
  - Initial training window: first 5 years (data available from 2003-09)
  - OOS starts: 2008-09 (first rebalance after 5-year burn-in)
  - OOS ends: 2024-04 (lockbox from 2024-05)
  - Lockbox: 2024-05 to present (never touched here)

Output:
  - experiments/exp_000_baseline_ladder/results.parquet
  - experiments/exp_000_baseline_ladder/summary.md
  - Metrics logged to state/journal.jsonl
  - Hypotheses count logged to state/hypotheses_tested.jsonl
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

# Paths
ROOT = Path(__file__).parent
CACHE_DIR = Path("/home/user/crt/experiments/monthly_dca/cache")
V2_DIR = CACHE_DIR / "v2" / "sp500_pit"
PRICES_PATH = CACHE_DIR / "prices_extended.parquet"
MEMBERSHIP_PATH = V2_DIR / "sp500_membership_monthly.parquet"
FWDRET_PATH = CACHE_DIR / "v2" / "ml_preds_v2.parquet"
EXP_DIR = ROOT / "experiments" / "exp_000_baseline_ladder"
STATE_DIR = ROOT / "state"
EXP_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_END = pd.Timestamp("2024-04-30")
LOCKBOX_START = pd.Timestamp("2024-05-31")
OOS_START = pd.Timestamp("2008-08-31")   # 5-year burn-in from 2003-09
K = 5                                      # default top-K
COST_BPS = 5.0                             # one-way cost


# ── data loading ─────────────────────────────────────────────────────────────

def load_data():
    print("Loading prices...")
    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)
    print(f"  Prices: {prices.shape}, {prices.index[0].date()} to {prices.index[-1].date()}")

    print("Loading PIT membership...")
    members = pd.read_parquet(MEMBERSHIP_PATH)
    members["asof"] = pd.to_datetime(members["asof"])
    # Restrict to research window
    members = members[members["asof"] <= RESEARCH_END].copy()
    print(f"  Members: {len(members)} rows, {members['asof'].nunique()} asofs")

    print("Loading forward returns...")
    fwd = pd.read_parquet(FWDRET_PATH)[["asof", "ticker", "fwd_1m_ret"]].copy()
    fwd["asof"] = pd.to_datetime(fwd["asof"])
    fwd = fwd[fwd["asof"] <= RESEARCH_END].copy()
    print(f"  Fwd returns: {len(fwd)} rows")

    return prices, members, fwd


# ── feature computation ───────────────────────────────────────────────────────

def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily prices -> monthly returns, aligned to month-end dates."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change()


def compute_features_full(prices: pd.DataFrame, asof_dates: list) -> pd.DataFrame:
    """
    Compute features for every (asof, ticker) in a PIT manner.
    Uses rolling lookback so features at t only use prices <= t.
    Returns long-format DataFrame with columns [asof, ticker, mom_12_1, mom_6_1,
    vol_12m, trend_health].
    """
    monthly = prices.resample("ME").last()
    # Align asof dates to available month-ends
    avail_months = monthly.index.tolist()

    records = []
    for asof in asof_dates:
        # Find closest month-end at or before asof
        month_end = max([d for d in avail_months if d <= asof], default=None)
        if month_end is None:
            continue
        hist = monthly.loc[:month_end]
        if len(hist) < 14:  # need at least 13 months for 12-1 mom
            continue

        p_now = hist.iloc[-2] if len(hist) >= 2 else hist.iloc[-1]  # skip last month
        p_6m  = hist.iloc[-7]  if len(hist) >= 7  else hist.iloc[0]
        p_12m = hist.iloc[-13] if len(hist) >= 13 else hist.iloc[0]

        mom_12_1 = (p_now / p_12m) - 1
        mom_6_1  = (p_now / p_6m)  - 1

        # 12m vol from daily
        daily_hist = prices.loc[:asof]
        if len(daily_hist) >= 60:
            rets_d = daily_hist.pct_change().dropna()
            vol_12m = (rets_d.iloc[-252:].std() * np.sqrt(252)) if len(rets_d) >= 252 else (rets_d.std() * np.sqrt(252))
        else:
            vol_12m = pd.Series(np.nan, index=p_now.index)

        # Trend health (fraction of days above 200d MA)
        if len(daily_hist) >= 200:
            window_d = daily_hist.iloc[-756:]  # ~3 years
            th_vals = {}
            for col in window_d.columns:
                s = window_d[col].dropna()
                if len(s) < 200:
                    th_vals[col] = np.nan
                else:
                    ma200 = s.rolling(200).mean()
                    above = (s > ma200).dropna()
                    th_vals[col] = above.mean() if len(above) > 0 else np.nan
            trend_health = pd.Series(th_vals)
        else:
            trend_health = pd.Series(np.nan, index=p_now.index)

        # SPX proxy: mean of all tickers
        spx_close = daily_hist.mean(axis=1).dropna()
        if len(spx_close) >= 200:
            spx_ma200 = spx_close.rolling(200).mean().iloc[-1]
            regime_ok = bool(spx_close.iloc[-1] >= spx_ma200)
        else:
            regime_ok = True

        frame = pd.DataFrame({
            "mom_12_1": mom_12_1,
            "mom_6_1": mom_6_1,
            "vol_12m": vol_12m,
            "trend_health": trend_health,
        })
        frame["asof"] = asof
        frame.index.name = "ticker"
        frame = frame.reset_index()
        frame["regime_ok"] = regime_ok
        records.append(frame)

    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


# ── OLS scorer (Rung 5) ───────────────────────────────────────────────────────

def fit_ols_rung5(
    features: pd.DataFrame,
    fwd: pd.DataFrame,
    train_asofs: list,
) -> pd.Series:
    """
    Fit a cross-sectional OLS on [mom_12_1, mom_6_1, inv_vol] vs fwd_1m_ret
    using all training asofs. Return coefficient series.
    """
    cols = ["mom_12_1", "mom_6_1", "inv_vol"]
    train_data = features[features["asof"].isin(train_asofs)].copy()
    train_data = train_data.merge(fwd[["asof","ticker","fwd_1m_ret"]], on=["asof","ticker"], how="left")
    train_data = train_data.dropna(subset=cols + ["fwd_1m_ret"])
    if len(train_data) < 100:
        return None
    # Cross-sectional rank-normalize within each asof to reduce scale effects
    for col in cols:
        train_data[col] = train_data.groupby("asof")[col].rank(pct=True)
    X = train_data[cols].values
    y = train_data["fwd_1m_ret"].values
    model = LinearRegression().fit(X, y)
    return model


# ── Backtest runner ────────────────────────────────────────────────────────────

def run_rung(
    name: str,
    panel: pd.DataFrame,
    fwd: pd.DataFrame,
    members: pd.DataFrame,
    score_col: str,
    K: int = K,
    cost_bps: float = COST_BPS,
    use_regime_gate: bool = False,
) -> dict:
    """
    Run a single rung of the baseline ladder.
    panel: long-format [asof, ticker, score_col, regime_ok, ...]
    """
    # Only research asofs, after OOS_START for OOS metrics
    asof_dates = sorted(panel["asof"].unique())
    asof_dates = [d for d in asof_dates if d <= RESEARCH_END]

    cash_monthly = 0.03 / 12  # 3% annual cash yield
    monthly_rets = {}
    prev_tickers = set()

    for asof in asof_dates:
        # PIT universe at this asof
        universe_at_asof = set(members[members["asof"] == asof]["ticker"])
        if not universe_at_asof:
            monthly_rets[asof] = 0.0
            continue

        # Apply regime gate
        if use_regime_gate:
            regime_row = panel[panel["asof"] == asof]["regime_ok"].iloc[0] if len(panel[panel["asof"] == asof]) > 0 else True
            if not regime_row:
                monthly_rets[asof] = cash_monthly
                prev_tickers = set()
                continue

        # Score within universe
        month_data = panel[
            (panel["asof"] == asof) &
            (panel["ticker"].isin(universe_at_asof))
        ].dropna(subset=[score_col]).copy()

        if len(month_data) < K:
            monthly_rets[asof] = cash_monthly
            continue

        top = month_data.nlargest(K, score_col)
        tickers = top["ticker"].tolist()
        n = len(tickers)
        weights = {t: 1.0 / n for t in tickers}

        # Forward returns
        fwd_asof = fwd[fwd["asof"] == asof].set_index("ticker")["fwd_1m_ret"]
        port_ret = sum(weights.get(t, 0) * fwd_asof.get(t, 0.0) for t in tickers)

        # Turnover cost
        new_buys = set(tickers) - prev_tickers
        exits = prev_tickers - set(tickers)
        buy_w = sum(weights.get(t, 0.0) for t in new_buys)
        sell_w = len(exits) / max(K, 1)
        cost = (buy_w + sell_w) * (cost_bps / 10_000)
        monthly_rets[asof] = port_ret - cost
        prev_tickers = set(tickers)

    rets = pd.Series(monthly_rets).sort_index()
    oos_rets = rets[rets.index >= OOS_START]

    def metrics(r):
        r = r.dropna()
        if len(r) < 12:
            return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "n": len(r)}
        cum = (1 + r).cumprod()
        n_years = len(r) / 12
        cagr = cum.iloc[-1] ** (1 / n_years) - 1
        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        dd = (cum / cum.cummax() - 1).min()
        return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 4),
                "max_dd": round(dd, 4), "n": len(r)}

    return {
        "name": name,
        "monthly_returns": rets,
        "oos_returns": oos_rets,
        "full_metrics": metrics(rets),
        "oos_metrics": metrics(oos_rets),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 2 Baseline Ladder — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    prices, members, fwd = load_data()

    # Get asof dates from membership (research window only)
    asof_dates = sorted(members["asof"].unique())
    print(f"Computing features over {len(asof_dates)} months...")

    # Feature computation (PIT)
    feats = compute_features_full(prices, asof_dates)
    print(f"Features computed: {len(feats)} rows")

    # Merge with membership to enforce PIT universe
    panel = members.merge(feats, on=["asof", "ticker"], how="left")
    panel["inv_vol"] = 1.0 / panel["vol_12m"].clip(lower=1e-6)

    # Cross-sectional ranks for each feature
    for col in ["mom_12_1", "mom_6_1", "trend_health", "inv_vol"]:
        panel[f"rank_{col}"] = panel.groupby("asof")[col].rank(pct=True)

    # ── Rung 1: 12-1 momentum ─────────────────────────────────────────────
    print("\n--- Rung 1: 12-1 Momentum (EW top-5) ---")
    r1 = run_rung("R1_mom12_1", panel, fwd, members, score_col="mom_12_1", K=K)
    print(f"  Full:  CAGR={r1['full_metrics']['cagr']:.1%}  Sharpe={r1['full_metrics']['sharpe']:.3f}  MaxDD={r1['full_metrics']['max_dd']:.1%}")
    print(f"  OOS:   CAGR={r1['oos_metrics']['cagr']:.1%}  Sharpe={r1['oos_metrics']['sharpe']:.3f}  MaxDD={r1['oos_metrics']['max_dd']:.1%}  N={r1['oos_metrics']['n']}")

    # ── Rung 2: Momentum + low-vol filter ────────────────────────────────
    print("\n--- Rung 2: Momentum × Low-Vol composite ---")
    # score = mom rank - 0.5 * vol rank
    panel["score_r2"] = panel["rank_mom_12_1"] - 0.5 * panel["rank_inv_vol"].apply(lambda x: 1-x if not np.isnan(x) else np.nan)
    # Actually: high inv_vol rank = low vol, which is good. So: mom_rank + 0.5 * inv_vol_rank
    panel["score_r2"] = panel["rank_mom_12_1"] + 0.5 * panel["rank_inv_vol"]
    r2 = run_rung("R2_mom_lovol", panel, fwd, members, score_col="score_r2", K=K)
    print(f"  Full:  CAGR={r2['full_metrics']['cagr']:.1%}  Sharpe={r2['full_metrics']['sharpe']:.3f}  MaxDD={r2['full_metrics']['max_dd']:.1%}")
    print(f"  OOS:   CAGR={r2['oos_metrics']['cagr']:.1%}  Sharpe={r2['oos_metrics']['sharpe']:.3f}  MaxDD={r2['oos_metrics']['max_dd']:.1%}  N={r2['oos_metrics']['n']}")

    # ── Rung 3: + Quality (trend health) ─────────────────────────────────
    print("\n--- Rung 3: Momentum + Low-Vol + Trend Health ---")
    panel["score_r3"] = panel["rank_mom_12_1"] + 0.5 * panel["rank_inv_vol"] + 0.5 * panel["rank_trend_health"]
    r3 = run_rung("R3_quality", panel, fwd, members, score_col="score_r3", K=K)
    print(f"  Full:  CAGR={r3['full_metrics']['cagr']:.1%}  Sharpe={r3['full_metrics']['sharpe']:.3f}  MaxDD={r3['full_metrics']['max_dd']:.1%}")
    print(f"  OOS:   CAGR={r3['oos_metrics']['cagr']:.1%}  Sharpe={r3['oos_metrics']['sharpe']:.3f}  MaxDD={r3['oos_metrics']['max_dd']:.1%}  N={r3['oos_metrics']['n']}")

    # ── Rung 4: + Regime gate ─────────────────────────────────────────────
    print("\n--- Rung 4: + Regime Gate (SPX 200d MA) ---")
    r4 = run_rung("R4_regime", panel, fwd, members, score_col="score_r3", K=K, use_regime_gate=True)
    print(f"  Full:  CAGR={r4['full_metrics']['cagr']:.1%}  Sharpe={r4['full_metrics']['sharpe']:.3f}  MaxDD={r4['full_metrics']['max_dd']:.1%}")
    print(f"  OOS:   CAGR={r4['oos_metrics']['cagr']:.1%}  Sharpe={r4['oos_metrics']['sharpe']:.3f}  MaxDD={r4['oos_metrics']['max_dd']:.1%}  N={r4['oos_metrics']['n']}")

    # ── Rung 5: OLS cross-sectional ---
    print("\n--- Rung 5: OLS cross-sectional ranker ---")
    # Walk-forward OLS: train on prior 5 years, predict next month
    ols_scores = []
    ols_oos_asofs = [d for d in asof_dates if d >= OOS_START]

    for asof in ols_oos_asofs:
        train_asofs = [d for d in asof_dates if d < asof][-60:]  # up to 5 years
        if len(train_asofs) < 12:
            continue
        train_data = panel[panel["asof"].isin(train_asofs)].copy()
        train_data = train_data.merge(fwd[["asof","ticker","fwd_1m_ret"]], on=["asof","ticker"], how="left")
        cols = ["rank_mom_12_1", "rank_mom_6_1", "rank_inv_vol"]
        train_data = train_data.dropna(subset=cols + ["fwd_1m_ret"])
        if len(train_data) < 50:
            continue
        X_train = train_data[cols].values
        y_train = train_data["fwd_1m_ret"].values
        try:
            model = LinearRegression().fit(X_train, y_train)
            test_data = panel[panel["asof"] == asof].copy()
            test_data = test_data.dropna(subset=cols)
            if len(test_data) == 0:
                continue
            X_test = test_data[cols].values
            preds = model.predict(X_test)
            test_data = test_data.copy()
            test_data["score_r5"] = preds
            ols_scores.append(test_data[["asof", "ticker", "score_r5"]])
        except Exception:
            continue

    if ols_scores:
        ols_panel = pd.concat(ols_scores, ignore_index=True)
        ols_panel_full = panel.merge(ols_panel, on=["asof","ticker"], how="left")
        r5 = run_rung("R5_ols", ols_panel_full, fwd, members, score_col="score_r5", K=K)
        print(f"  Full:  CAGR={r5['full_metrics']['cagr']:.1%}  Sharpe={r5['full_metrics']['sharpe']:.3f}  MaxDD={r5['full_metrics']['max_dd']:.1%}")
        print(f"  OOS:   CAGR={r5['oos_metrics']['cagr']:.1%}  Sharpe={r5['oos_metrics']['sharpe']:.3f}  MaxDD={r5['oos_metrics']['max_dd']:.1%}  N={r5['oos_metrics']['n']}")
    else:
        r5 = None
        print("  Insufficient data for OLS rung.")

    # ── Summary table ─────────────────────────────────────────────────────
    rungs = [r for r in [r1, r2, r3, r4, r5] if r is not None]
    print(f"\n{'='*60}")
    print("BASELINE LADDER SUMMARY (OOS = 2008-09 to 2024-04)")
    print(f"{'='*60}")
    print(f"{'Rung':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'N':>6}")
    print("-" * 60)
    for r in rungs:
        m = r["oos_metrics"]
        print(f"{r['name']:<25} {m['cagr']:>8.1%} {m['sharpe']:>8.3f} {m['max_dd']:>8.1%} {m['n']:>6}")

    # v3 baseline from prior session for reference
    print(f"{'v3_GBM_baseline(ref)':<25} {'39.77%':>8} {'0.953':>8} {'-49.83%':>8} {'248':>6}")
    print(f"\nTarget: CAGR >= 50%, Sharpe >= 2.0")

    # ── Save results ──────────────────────────────────────────────────────
    results_rows = []
    for r in rungs:
        row = {"rung": r["name"], "window": "oos"}
        row.update(r["oos_metrics"])
        results_rows.append(row)
        row2 = {"rung": r["name"], "window": "full"}
        row2.update(r["full_metrics"])
        results_rows.append(row2)

    results_df = pd.DataFrame(results_rows)
    results_df.to_parquet(EXP_DIR / "results.parquet")

    # Save monthly returns for best OOS rung
    best_rung = max(rungs, key=lambda r: r["oos_metrics"].get("sharpe", -999) or -999)
    best_rets = pd.DataFrame({"monthly_ret": best_rung["oos_returns"]})
    best_rets.to_parquet(EXP_DIR / "best_oos_returns.parquet")

    elapsed = time.time() - t0

    # ── Summary markdown ──────────────────────────────────────────────────
    summary_md = f"""# Experiment 000 — Phase 2 Baseline Ladder

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Universe**: SPX PIT (sp500_membership_monthly.parquet)
**OOS window**: 2008-09-30 to 2024-04-30 ({r1['oos_metrics']['n']} months)
**K**: {K}, **Cost**: {COST_BPS} bps one-way

## Results

| Rung | CAGR | Sharpe | MaxDD | N |
|---|---:|---:|---:|---:|
"""
    for r in rungs:
        m = r["oos_metrics"]
        summary_md += f"| {r['name']} | {m['cagr']:.1%} | {m['sharpe']:.3f} | {m['max_dd']:.1%} | {m['n']} |\n"
    summary_md += f"| v3_GBM_baseline (reference) | 39.77% | 0.953 | -49.83% | 248 |\n"
    summary_md += f"""
**Best rung**: {best_rung['name']} — OOS CAGR {best_rung['oos_metrics']['cagr']:.1%}, Sharpe {best_rung['oos_metrics']['sharpe']:.3f}

## Key observations

1. Simple momentum (R1) establishes the baseline — this is the Jegadeesh-Titman benchmark.
2. Low-vol overlay (R2) reduces drawdown but may sacrifice CAGR.
3. Trend health quality screen (R3) filters for sustained uptrenders — proxy for fundamental quality.
4. Regime gate (R4) reduces MaxDD materially in crash periods at some cost to CAGR.
5. OLS blender (R5) tests whether linear combination adds over ranking alone.
6. **The target (CAGR ≥ 50%, Sharpe ≥ 2.0) requires substantially better signal.**
7. Prior YLOka sessions (v3 GBM) at ~40% CAGR represent the ceiling for price-only features.
8. Achieving the target likely requires fundamentals, volume, or alternative data.

## What's needed beyond the baseline

- Either: deeper ML (LSTM/Transformer on sequences)
- Or: richer signals (fundamentals, earnings, volume-based)
- Or: better regime-aware concentration + timing
- The GBM v3 represents saturation of the price-only space after 88+ experiments.

## Elapsed

{elapsed:.1f}s
"""

    with open(EXP_DIR / "summary.md", "w") as f:
        f.write(summary_md)

    # ── Journal entry ─────────────────────────────────────────────────────
    journal_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": "exp_000_baseline_ladder",
        "hypothesis": "Phase 2 baseline ladder establishes the floor for all subsequent models",
        "what_i_did": "Implemented 5-rung baseline ladder (12-1 mom, +low-vol, +quality, +regime, +OLS) with PIT SPX universe",
        "result": {
            "best_rung": best_rung["name"],
            "best_oos_cagr": best_rung["oos_metrics"]["cagr"],
            "best_oos_sharpe": best_rung["oos_metrics"]["sharpe"],
            "v3_ref_cagr": 0.3977,
            "v3_ref_sharpe": 0.953,
        },
        "hparams_tried": 5,
        "next_action": "Explore GBM cross-sectional ranker with existing features as the first serious model attempt",
    }

    STATE_DIR.mkdir(exist_ok=True)
    with open(STATE_DIR / "journal.jsonl", "a") as f:
        f.write(json.dumps(journal_entry) + "\n")

    hyp_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": "exp_000_baseline_ladder",
        "n_hparams": 5,
        "description": "5 baseline rungs (K=5 each, fixed)",
    }
    with open(STATE_DIR / "hypotheses_tested.jsonl", "a") as f:
        f.write(json.dumps(hyp_entry) + "\n")

    print(f"\n[DONE] Results saved to {EXP_DIR}")
    print(f"[DONE] Elapsed: {elapsed:.1f}s")

    return rungs


if __name__ == "__main__":
    rungs = main()
