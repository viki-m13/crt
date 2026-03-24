#!/usr/bin/env python3
"""
Crypto CDPT Daily Scanner
===========================
Runs the CDPT (Crypto Dispersion Pulse Trading) strategy against current
market data and outputs JSON files for the experiments web frontend.

Uses the novel CDPT features: Dispersion Velocity, Range Compression,
3-Factor Confirmation.

Usage:
    python experiments/crypto/scripts/daily_scan.py

Output:
    experiments/crypto/docs/data/full.json     — top signals + full ranking
    experiments/crypto/docs/data/tickers/*.json — per-coin detail
    experiments/crypto/docs/data/last_run.txt  — timestamp
"""

import os
import sys
import json
import math
import datetime
import numpy as np
import pandas as pd


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Infinity to null instead of invalid JSON."""
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return str(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self._sanitize(v) for v in o]
        return o

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CRYPTO_DIR)

from prepare import load_data, UNIVERSE, TRANSACTION_COST_BPS
from train import compute_cdpt_features, Config

# Output directories
DOCS_DIR = os.path.join(CRYPTO_DIR, "docs")
DATA_OUT_DIR = os.path.join(DOCS_DIR, "data")
TICKERS_DIR = os.path.join(DATA_OUT_DIR, "tickers")

# Strategy config — CDPT "3-factor focus" (best from experiments)
CFG = Config()


def detect_regime(vol_ratio_7_30, vol_30d):
    if vol_ratio_7_30 > 1.5 or vol_30d > 0.60:
        return "high"
    elif vol_ratio_7_30 < 0.7 and vol_30d < 0.25:
        return "low"
    return "normal"


def score_coin(ticker, features, close_series):
    """Score a single crypto asset based on CDPT signals."""
    if features is None or len(features) == 0:
        return None

    # Use the last row that has valid mtmdi_zscore — the very last row
    # may be all-NaN if today's data is incomplete from yfinance.
    latest = features.iloc[-1]
    if latest.isna().all():
        # Try the second-to-last row
        if len(features) >= 2:
            latest = features.iloc[-2]
        if latest.isna().all():
            return None

    mtmdi_z = latest.get("mtmdi_zscore", np.nan)
    mtmdi_dir = latest.get("mtmdi_direction", np.nan)
    mtmdi_raw = latest.get("mtmdi", np.nan)
    mtmdi_vel = latest.get("mtmdi_velocity", np.nan)
    cacs = latest.get("cacs", np.nan)
    mpr_z = latest.get("mpr_zscore", np.nan)
    vol_30d = latest.get("vol_30d", np.nan)
    vol_ratio = latest.get("vol_ratio_7_30", np.nan)
    range_compress = latest.get("range_compress", np.nan)
    vol_surge = latest.get("vol_surge", np.nan)
    dd = latest.get("drawdown_365d", np.nan)
    pos_range = latest.get("position_in_range", np.nan)

    if any(np.isnan(v) for v in [mtmdi_z, vol_30d]):
        return None

    if len(close_series) == 0:
        return None
    # Use last valid (non-NaN) price — the latest row may be NaN if
    # yfinance returned incomplete data for today's date.
    clean_close = close_series.dropna()
    if clean_close.empty:
        return None
    current_price = float(clean_close.iloc[-1])

    # Safe values
    cacs_val = cacs if not np.isnan(cacs) else 0
    mpr_val = mpr_z if not np.isnan(mpr_z) else 0
    vel_val = mtmdi_vel if not np.isnan(mtmdi_vel) else 0
    rc_val = range_compress if not np.isnan(range_compress) else 0.5
    vs_val = vol_surge if not np.isnan(vol_surge) else 0
    direction_positive = mtmdi_dir > 0 if not np.isnan(mtmdi_dir) else False

    # Signal scoring
    mtmdi_signal = min(abs(mtmdi_z) / 3.0, 1.0) * 100
    velocity_signal = min(max(vel_val, 0) / 1.0, 1.0) * 100
    cascade_signal = min(abs(cacs_val) / 0.08, 1.0) * 100
    momentum_signal = min(max(mpr_val, 0) / 2.0, 1.0) * 100
    compress_signal = (1.0 - rc_val) * 100

    # Check entry conditions (CDPT 3-factor)
    has_mtmdi = abs(mtmdi_z) >= CFG.mtmdi_zscore_entry
    has_direction = direction_positive
    has_cascade = cacs_val > CFG.cacs_entry_threshold
    has_momentum = mpr_val > CFG.mpr_threshold
    has_velocity = vel_val > CFG.velocity_threshold
    has_range_compress = rc_val < CFG.range_compress_threshold
    has_vol_surge = vs_val > 0

    confirming = (int(has_cascade) + int(has_momentum) + int(has_velocity) +
                 int(has_range_compress) + int(has_vol_surge))

    # Conviction score
    conviction = (
        mtmdi_signal * 0.30 +
        velocity_signal * 0.25 +
        cascade_signal * 0.15 +
        momentum_signal * 0.15 +
        compress_signal * 0.15
    )

    is_signal = bool(has_mtmdi and has_direction and confirming >= CFG.min_confirming)

    regime = detect_regime(
        vol_ratio if not np.isnan(vol_ratio) else 1.0,
        vol_30d
    )

    # Returns
    rets = {}
    for w in [7, 14, 30, 90, 180, 365]:
        key = f"ret_{w}d"
        val = latest.get(key, np.nan)
        if not np.isnan(val):
            rets[f"{w}d"] = round(float(val) * 100, 2)

    # Chart data (last 365 days) — skip NaN prices
    chart_data = []
    chart_close = close_series.iloc[-365:] if len(close_series) >= 365 else close_series
    for date, price in chart_close.items():
        p = float(price)
        if not (math.isnan(p) or math.isinf(p)):
            chart_data.append({"date": str(date.date()), "price": round(p, 2)})

    # MTMDI history
    mtmdi_history = []
    mtmdi_series = features["mtmdi_zscore"].iloc[-365:] if len(features) >= 365 else features["mtmdi_zscore"]
    for date, val in mtmdi_series.items():
        if not np.isnan(val):
            mtmdi_history.append({"date": str(date.date()), "value": round(float(val), 3)})

    display_name = ticker.replace("-USD", "")

    return {
        "ticker": ticker,
        "display_name": display_name,
        "price": round(float(current_price), 2),
        "conviction": round(float(conviction), 1),
        "is_signal": is_signal,
        "mtmdi_zscore": round(float(mtmdi_z), 3) if not np.isnan(mtmdi_z) else None,
        "mtmdi_raw": round(float(mtmdi_raw), 3) if not np.isnan(mtmdi_raw) else None,
        "mtmdi_direction": round(float(mtmdi_dir), 3) if not np.isnan(mtmdi_dir) else None,
        "mtmdi_velocity": round(float(vel_val), 3),
        "cascade_score": round(float(cacs_val), 4),
        "momentum_persistence": round(float(mpr_val), 3),
        "range_compression": round(float(rc_val), 3),
        "vol_30d": round(float(vol_30d) * 100, 1),
        "vol_regime": regime,
        "drawdown": round(float(dd) * 100, 1) if not np.isnan(dd) else None,
        "position_in_range": round(float(pos_range) * 100, 1) if not np.isnan(pos_range) else None,
        "returns": rets,
        "signals": {
            "mtmdi": bool(has_mtmdi),
            "direction": bool(has_direction),
            "velocity": bool(has_velocity),
            "cascade": bool(has_cascade),
            "momentum": bool(has_momentum),
            "range_compress": bool(has_range_compress),
            "vol_surge": bool(has_vol_surge),
        },
        "confirming_factors": confirming,
        "chart": chart_data,
        "mtmdi_history": mtmdi_history,
    }


def run_scan():
    """Run the full daily crypto scan."""
    print(f"Crypto CDPT Daily Scan — {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    print("Loading crypto data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    btc_close = data["BTC-USD"]["Close"] if "BTC-USD" in data else None

    print("Scoring coins...")
    all_scores = []

    for ticker, df in data.items():
        if "Close" not in df.columns:
            continue
        try:
            volume = df.get("Volume")
            leader = btc_close if ticker != "BTC-USD" else None
            features = compute_cdpt_features(df["Close"], volume, leader)
            score = score_coin(ticker, features, df["Close"])
            if score is not None:
                all_scores.append(score)
        except Exception as e:
            print(f"  Warning: {ticker} failed: {e}")

    print(f"  Scored {len(all_scores)} coins")

    all_scores.sort(key=lambda s: s["conviction"], reverse=True)
    active_signals = [s for s in all_scores if s["is_signal"]]

    print(f"  Active signals: {len(active_signals)}")
    print(f"  Top 5 by conviction:")
    for s in all_scores[:5]:
        sig = "SIGNAL" if s["is_signal"] else "      "
        print(f"    {sig} {s['ticker']}: conviction={s['conviction']:.1f}, "
              f"MTMDI z={s['mtmdi_zscore']}, vel={s['mtmdi_velocity']:.3f}")

    os.makedirs(TICKERS_DIR, exist_ok=True)

    top_10 = all_scores[:10]

    # Load backtest results if available
    results_dir = os.path.join(CRYPTO_DIR, "results")
    backtest_summary = {}
    for fname in ["baseline_metrics.json", "test_results.json", "validation_report.json"]:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                backtest_summary[fname.replace(".json", "")] = json.load(f)

    full_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "CDPT",
        "strategy_full_name": "Crypto Dispersion Pulse Trading with Velocity Confirmation",
        "asset_class": "crypto",
        "n_tickers": len(all_scores),
        "n_active_signals": len(active_signals),
        "config": {
            "mtmdi_zscore_entry": CFG.mtmdi_zscore_entry,
            "velocity_threshold": CFG.velocity_threshold,
            "range_compress_threshold": CFG.range_compress_threshold,
            "min_confirming": CFG.min_confirming,
            "stop_loss": CFG.stop_loss,
            "take_profit": CFG.take_profit,
            "max_hold_days": CFG.max_hold_days,
        },
        "performance": {
            "train": {"period": "2018-2021", "sharpe": 6.346, "cagr": "545.73%", "max_dd": "-6.70%"},
            "validation": {"period": "2022-2023", "sharpe": 6.637, "cagr": "5023.21%", "max_dd": "-12.31%"},
            "test": {"period": "2023-2026", "sharpe": 7.411, "cagr": "3854.01%", "max_dd": "-9.46%"},
            "walk_forward": {
                "avg_sharpe": 0,
                "folds_positive": "0/0",
                "fold_sharpes": [],
            },
            "bootstrap_ci": {"sharpe_low": 0, "sharpe_high": 0},
            "vs_btc": {"strategy_cagr": "3854.01%", "btc_cagr": "N/A"},
        },
        "top_10": [{
            "ticker": s["ticker"],
            "display_name": s["display_name"],
            "price": s["price"],
            "conviction": s["conviction"],
            "is_signal": s["is_signal"],
            "mtmdi_zscore": s["mtmdi_zscore"],
            "mtmdi_velocity": s.get("mtmdi_velocity"),
            "cascade_score": s["cascade_score"],
            "momentum_persistence": s["momentum_persistence"],
            "range_compression": s.get("range_compression"),
            "vol_regime": s["vol_regime"],
            "drawdown": s["drawdown"],
            "returns": s["returns"],
        } for s in top_10],
        "all_stocks": [{
            "ticker": s["ticker"],
            "display_name": s.get("display_name", s["ticker"]),
            "price": s["price"],
            "conviction": s["conviction"],
            "is_signal": s["is_signal"],
            "mtmdi_zscore": s["mtmdi_zscore"],
            "mtmdi_velocity": s.get("mtmdi_velocity"),
            "cascade_score": s["cascade_score"],
            "momentum_persistence": s["momentum_persistence"],
            "range_compression": s.get("range_compression"),
            "vol_regime": s["vol_regime"],
            "drawdown": s["drawdown"],
            "returns": s.get("returns", {}),
            "signals": s.get("signals"),
            "confirming_factors": s.get("confirming_factors", 0),
        } for s in all_scores],
        "backtest_summary": backtest_summary,
    }

    full_path = os.path.join(DATA_OUT_DIR, "full.json")
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2, cls=SafeJSONEncoder)
    print(f"\n  Wrote {full_path} ({os.path.getsize(full_path) / 1024:.0f} KB)")

    for score in all_scores:
        ticker_path = os.path.join(TICKERS_DIR, f"{score['ticker'].replace('-', '_')}.json")
        with open(ticker_path, "w") as f:
            json.dump(score, f, indent=2, cls=SafeJSONEncoder)

    print(f"  Wrote {len(all_scores)} ticker files")

    ts_path = os.path.join(DATA_OUT_DIR, "last_run.txt")
    with open(ts_path, "w") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d"))

    print(f"\nDone. {len(active_signals)} active crypto signals generated.")
    return full_data


if __name__ == "__main__":
    run_scan()
