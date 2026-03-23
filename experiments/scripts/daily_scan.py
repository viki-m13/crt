#!/usr/bin/env python3
"""
TMD-ARC Daily Scanner
======================
Runs the TMD-ARC strategy against current market data and outputs
JSON files for the experiments web frontend.

Mirrors the main CRT daily_scan.py but uses the TMD-ARC strategy.

Usage:
    python experiments/scripts/daily_scan.py

Output:
    experiments/docs/data/full.json     — top signals + full ranking
    experiments/docs/data/tickers/*.json — per-ticker detail
    experiments/docs/data/last_run.txt  — timestamp
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

# Add experiments root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from prepare import load_data, compute_features, UNIVERSE, TRANSACTION_COST_BPS

# Output directories
DOCS_DIR = os.path.join(EXPERIMENTS_DIR, "docs")
DATA_OUT_DIR = os.path.join(DOCS_DIR, "data")
TICKERS_DIR = os.path.join(DATA_OUT_DIR, "tickers")

# Strategy config (best from experiment loop)
MTMDI_ZSCORE_ENTRY = 1.5
CACS_ENTRY_THRESHOLD = 0.02
MPR_THRESHOLD = 0.5
STOP_LOSS = -0.07
TAKE_PROFIT = 0.39
MAX_HOLD_DAYS = 21
VOL_TARGET = 0.15


def detect_regime(vol_ratio_5_21, vol_21d):
    if vol_ratio_5_21 > 1.5 or vol_21d > 0.30:
        return "high"
    elif vol_ratio_5_21 < 0.7 and vol_21d < 0.12:
        return "low"
    return "normal"


def score_stock(ticker, features, close_series):
    """Score a single stock based on TMD-ARC signals. Returns dict or None."""
    if features is None or len(features) == 0:
        return None

    # Get latest row
    latest = features.iloc[-1]
    if latest.isna().all():
        return None

    mtmdi_z = latest.get("mtmdi_zscore", np.nan)
    mtmdi_dir = latest.get("mtmdi_direction", np.nan)
    mtmdi_raw = latest.get("mtmdi", np.nan)
    cacs = latest.get("cacs", np.nan)
    mpr_z = latest.get("mpr_zscore", np.nan)
    mpr = latest.get("mpr", np.nan)
    vol_21d = latest.get("vol_21d", np.nan)
    vol_ratio = latest.get("vol_ratio_5_21", np.nan)
    dd = latest.get("drawdown_252d", np.nan)
    pos_range = latest.get("position_in_52w_range", np.nan)

    # Skip if core features are missing
    if any(np.isnan(v) for v in [mtmdi_z, vol_21d]):
        return None

    # Current price
    if len(close_series) == 0:
        return None
    current_price = close_series.iloc[-1]

    # --- Signal Scoring ---
    # MTMDI signal strength (0-100)
    mtmdi_signal = min(abs(mtmdi_z) / 3.0, 1.0) * 100

    # Direction signal
    direction_positive = mtmdi_dir > 0 if not np.isnan(mtmdi_dir) else False

    # Cascade signal (0-100)
    cacs_val = cacs if not np.isnan(cacs) else 0
    cascade_signal = min(abs(cacs_val) / 0.05, 1.0) * 100

    # MPR signal (0-100)
    mpr_val = mpr_z if not np.isnan(mpr_z) else 0
    momentum_signal = min(max(mpr_val, 0) / 2.0, 1.0) * 100

    # Check entry conditions
    has_mtmdi = abs(mtmdi_z) >= MTMDI_ZSCORE_ENTRY
    has_direction = direction_positive
    has_cascade = cacs_val > CACS_ENTRY_THRESHOLD
    has_momentum = mpr_val > MPR_THRESHOLD

    # Conviction score (0-100)
    conviction = (
        mtmdi_signal * 0.50 +
        cascade_signal * 0.30 +
        momentum_signal * 0.20
    )

    # Is this an actionable signal?
    is_signal = bool(has_mtmdi and has_direction and (has_cascade or has_momentum))

    # Regime
    regime = detect_regime(
        vol_ratio if not np.isnan(vol_ratio) else 1.0,
        vol_21d
    )

    # Historical returns for context
    rets = {}
    for w in [5, 10, 21, 63, 126, 252]:
        key = f"ret_{w}d"
        val = latest.get(key, np.nan)
        if not np.isnan(val):
            rets[f"{w}d"] = round(float(val) * 100, 2)

    # Build price chart data (last 252 days)
    chart_data = []
    chart_close = close_series.iloc[-252:] if len(close_series) >= 252 else close_series
    for date, price in chart_close.items():
        chart_data.append({
            "date": str(date.date()),
            "price": round(float(price), 2)
        })

    # MTMDI history for chart (last 252 days)
    mtmdi_history = []
    mtmdi_series = features["mtmdi_zscore"].iloc[-252:] if len(features) >= 252 else features["mtmdi_zscore"]
    for date, val in mtmdi_series.items():
        if not np.isnan(val):
            mtmdi_history.append({
                "date": str(date.date()),
                "value": round(float(val), 3)
            })

    return {
        "ticker": ticker,
        "price": round(float(current_price), 2),
        "conviction": round(float(conviction), 1),
        "is_signal": is_signal,
        "mtmdi_zscore": round(float(mtmdi_z), 3) if not np.isnan(mtmdi_z) else None,
        "mtmdi_raw": round(float(mtmdi_raw), 3) if not np.isnan(mtmdi_raw) else None,
        "mtmdi_direction": round(float(mtmdi_dir), 3) if not np.isnan(mtmdi_dir) else None,
        "cascade_score": round(float(cacs_val), 4),
        "momentum_persistence": round(float(mpr_val), 3),
        "vol_21d": round(float(vol_21d) * 100, 1),  # as percentage
        "vol_regime": regime,
        "drawdown": round(float(dd) * 100, 1) if not np.isnan(dd) else None,
        "position_in_range": round(float(pos_range) * 100, 1) if not np.isnan(pos_range) else None,
        "returns": rets,
        "signals": {
            "mtmdi": bool(has_mtmdi),
            "direction": bool(has_direction),
            "cascade": bool(has_cascade),
            "momentum": bool(has_momentum),
        },
        "chart": chart_data,
        "mtmdi_history": mtmdi_history,
    }


def run_scan():
    """Run the full daily scan."""
    print(f"TMD-ARC Daily Scan — {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    # Load data
    print("Loading market data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    # Get market benchmark
    market_close = data["SPY"]["Close"] if "SPY" in data else None

    # Score all stocks
    print("Scoring stocks...")
    all_scores = []
    featured_tickers = [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
        "TSLA", "JPM", "V", "UNH", "BRK-B", "XOM", "LLY", "COST"
    ]

    for ticker, df in data.items():
        if "Close" not in df.columns:
            continue
        try:
            volume = df.get("Volume")
            features = compute_features(df["Close"], volume, market_close)
            score = score_stock(ticker, features, df["Close"])
            if score is not None:
                all_scores.append(score)
        except Exception as e:
            print(f"  Warning: {ticker} failed: {e}")

    print(f"  Scored {len(all_scores)} tickers")

    # Sort by conviction (highest first)
    all_scores.sort(key=lambda s: s["conviction"], reverse=True)

    # Separate active signals from non-signals
    active_signals = [s for s in all_scores if s["is_signal"]]
    all_ranked = all_scores

    print(f"  Active signals: {len(active_signals)}")
    print(f"  Top 5 by conviction:")
    for s in all_scores[:5]:
        sig = "SIGNAL" if s["is_signal"] else "      "
        print(f"    {sig} {s['ticker']}: conviction={s['conviction']:.1f}, "
              f"MTMDI z={s['mtmdi_zscore']}, cascade={s['cascade_score']:.4f}")

    # Create output directories
    os.makedirs(TICKERS_DIR, exist_ok=True)

    # Get top 10 for embedding in full.json
    top_10 = all_scores[:10]

    # Backtest summary (from saved results)
    results_dir = os.path.join(EXPERIMENTS_DIR, "results")
    backtest_summary = {}
    for fname in ["baseline_metrics.json", "test_results.json", "validation_report.json"]:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                backtest_summary[fname.replace(".json", "")] = json.load(f)

    # Build full.json
    full_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "TMD-ARC",
        "strategy_full_name": "Temporal Momentum Dispersion with Adaptive Regime Cascade",
        "n_tickers": len(all_scores),
        "n_active_signals": len(active_signals),
        "config": {
            "mtmdi_zscore_entry": MTMDI_ZSCORE_ENTRY,
            "cacs_entry_threshold": CACS_ENTRY_THRESHOLD,
            "mpr_threshold": MPR_THRESHOLD,
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "max_hold_days": MAX_HOLD_DAYS,
        },
        "performance": {
            "train": {"period": "2010-2019", "sharpe": 2.839, "cagr": "32.68%", "max_dd": "-7.52%"},
            "validation": {"period": "2020-2022", "sharpe": 2.242, "cagr": "34.03%", "max_dd": "-7.69%"},
            "test": {"period": "2023-2026", "sharpe": 3.825, "cagr": "43.15%", "max_dd": "-3.77%"},
            "walk_forward": {
                "avg_sharpe": 2.825,
                "folds_positive": "5/5",
                "fold_sharpes": [2.565, 4.007, 3.013, 2.633, 1.908],
            },
            "bootstrap_ci": {"sharpe_low": 1.458, "sharpe_high": 3.549},
            "vs_spy": {"strategy_cagr": "43.15%", "spy_cagr": "19.11%"},
            "vs_random": {"percentile": "100%"},
        },
        "top_10": [{
            "ticker": s["ticker"],
            "price": s["price"],
            "conviction": s["conviction"],
            "is_signal": s["is_signal"],
            "mtmdi_zscore": s["mtmdi_zscore"],
            "cascade_score": s["cascade_score"],
            "momentum_persistence": s["momentum_persistence"],
            "vol_regime": s["vol_regime"],
            "drawdown": s["drawdown"],
            "returns": s["returns"],
        } for s in top_10],
        "all_stocks": [{
            "ticker": s["ticker"],
            "price": s["price"],
            "conviction": s["conviction"],
            "is_signal": s["is_signal"],
            "mtmdi_zscore": s["mtmdi_zscore"],
            "cascade_score": s["cascade_score"],
            "momentum_persistence": s["momentum_persistence"],
            "vol_regime": s["vol_regime"],
            "drawdown": s["drawdown"],
            "returns": s.get("returns", {}),
            "signals": s.get("signals"),
        } for s in all_ranked],
        "backtest_summary": backtest_summary,
    }

    # Write full.json
    full_path = os.path.join(DATA_OUT_DIR, "full.json")
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2, default=str)
    print(f"\n  Wrote {full_path} ({os.path.getsize(full_path) / 1024:.0f} KB)")

    # Write per-ticker JSON files
    for score in all_scores:
        ticker_path = os.path.join(TICKERS_DIR, f"{score['ticker']}.json")
        with open(ticker_path, "w") as f:
            json.dump(score, f, indent=2, default=str)

    print(f"  Wrote {len(all_scores)} ticker files")

    # Write timestamp
    ts_path = os.path.join(DATA_OUT_DIR, "last_run.txt")
    with open(ts_path, "w") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d"))

    print(f"\nDone. {len(active_signals)} active signals generated.")
    return full_data


if __name__ == "__main__":
    run_scan()
