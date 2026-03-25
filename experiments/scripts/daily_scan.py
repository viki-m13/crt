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

    # Get latest row — fall back to second-to-last if the last row is all NaN
    # (can happen when yfinance returns incomplete data for today's date)
    latest = features.iloc[-1]
    if latest.isna().all():
        if len(features) >= 2:
            latest = features.iloc[-2]
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

    # Current price — use last valid (non-NaN) value
    if len(close_series) == 0:
        return None
    clean_close = close_series.dropna()
    if clean_close.empty:
        return None
    current_price = float(clean_close.iloc[-1])

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

    # Build price chart data (last 252 days) — skip NaN prices
    chart_data = []
    chart_close = close_series.iloc[-252:] if len(close_series) >= 252 else close_series
    for date, price in chart_close.items():
        p = float(price)
        if not (math.isnan(p) or math.isinf(p)):
            chart_data.append({
                "date": str(date.date()),
                "price": round(p, 2)
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


def load_trades_state():
    """Load persisted trade tracking state from disk."""
    trades_path = os.path.join(DATA_OUT_DIR, "trades.json")
    if os.path.exists(trades_path):
        with open(trades_path) as f:
            return json.load(f)
    return {"open_positions": [], "closed_trades": [], "equity_curve": []}


def save_trades_state(state):
    """Persist trade tracking state to disk."""
    trades_path = os.path.join(DATA_OUT_DIR, "trades.json")
    with open(trades_path, "w") as f:
        json.dump(state, f, indent=2, cls=SafeJSONEncoder)


def update_trades(all_scores, data):
    """
    Update trade tracking: close expired/stopped positions, open new ones from signals.
    Returns updated trades state dict.
    """
    state = load_trades_state()
    today = datetime.date.today().isoformat()

    # Build price lookup from scores
    price_map = {s["ticker"]: s["price"] for s in all_scores}

    # --- Close positions that hit exit conditions ---
    still_open = []
    for pos in state["open_positions"]:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        entry_date = pos["entry_date"]

        current_price = price_map.get(ticker)
        if current_price is None:
            # Ticker no longer in scan; try to get price from data
            if ticker in data and "Close" in data[ticker].columns:
                clean = data[ticker]["Close"].dropna()
                current_price = round(float(clean.iloc[-1]), 2) if len(clean) > 0 else None
            if current_price is None:
                still_open.append(pos)
                continue

        pnl_pct = (current_price / entry_price - 1)
        days_held = (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(entry_date)).days
        # Approximate trading days as ~70% of calendar days
        trading_days_approx = int(days_held * 5 / 7)

        exit_reason = None
        if pnl_pct <= STOP_LOSS:
            exit_reason = "stop_loss"
        elif pnl_pct >= TAKE_PROFIT:
            exit_reason = "take_profit"
        elif trading_days_approx >= MAX_HOLD_DAYS:
            exit_reason = "time_exit"

        if exit_reason:
            state["closed_trades"].append({
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": today,
                "exit_price": current_price,
                "pnl_pct": round(pnl_pct * 100, 2),
                "days_held": days_held,
                "exit_reason": exit_reason,
                "conviction": pos.get("conviction", 0),
            })
        else:
            # Update current price and unrealized P&L
            pos["current_price"] = current_price
            pos["unrealized_pnl_pct"] = round(pnl_pct * 100, 2)
            pos["days_held"] = days_held
            still_open.append(pos)

    # --- Open new positions from active signals ---
    open_tickers = {p["ticker"] for p in still_open}
    closed_recently = {
        t["ticker"] for t in state["closed_trades"]
        if t.get("exit_date", "") >= (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    }

    active_signals = [s for s in all_scores if s["is_signal"]]
    for s in active_signals:
        if s["ticker"] not in open_tickers and s["ticker"] not in closed_recently:
            still_open.append({
                "ticker": s["ticker"],
                "entry_date": today,
                "entry_price": s["price"],
                "current_price": s["price"],
                "unrealized_pnl_pct": 0.0,
                "days_held": 0,
                "conviction": round(s["conviction"], 1),
                "stop_loss_price": round(s["price"] * (1 + STOP_LOSS), 2),
                "take_profit_price": round(s["price"] * (1 + TAKE_PROFIT), 2),
            })

    state["open_positions"] = still_open

    # --- Update equity curve ---
    # Compute portfolio value: start at 10000, equal weight per position
    if not state["equity_curve"]:
        prev_value = 10000.0
    else:
        prev_value = state["equity_curve"][-1]["value"]

    # Daily return from open positions (equal-weighted)
    n_open = len(still_open)
    if n_open > 0:
        avg_unrealized = sum(p.get("unrealized_pnl_pct", 0) for p in still_open) / n_open
        # If this is not the first data point, compute incremental change
        if state["equity_curve"]:
            prev_avg = state["equity_curve"][-1].get("avg_unrealized", 0)
            daily_change_pct = (avg_unrealized - prev_avg) / 100
        else:
            daily_change_pct = 0
    else:
        avg_unrealized = 0
        daily_change_pct = 0

    # Also account for closed trades today
    closed_today = [t for t in state["closed_trades"] if t.get("exit_date") == today]
    for ct in closed_today:
        # Add realized P&L to equity
        daily_change_pct += (ct["pnl_pct"] / 100) / max(n_open + len(closed_today), 1)

    new_value = round(prev_value * (1 + daily_change_pct), 2)

    # Don't add duplicate entry for same date
    if state["equity_curve"] and state["equity_curve"][-1]["date"] == today:
        state["equity_curve"][-1] = {
            "date": today,
            "value": new_value,
            "n_open": n_open,
            "avg_unrealized": round(avg_unrealized, 2),
        }
    else:
        state["equity_curve"].append({
            "date": today,
            "value": new_value,
            "n_open": n_open,
            "avg_unrealized": round(avg_unrealized, 2),
        })

    save_trades_state(state)
    return state


def compute_trades_summary(state):
    """Compute summary statistics for the trades section."""
    closed = state["closed_trades"]
    open_pos = state["open_positions"]

    if not closed:
        return {
            "total_closed": 0,
            "total_open": len(open_pos),
            "win_rate": 0,
            "avg_pnl_pct": 0,
            "best_trade_pct": 0,
            "worst_trade_pct": 0,
            "avg_hold_days": 0,
            "total_realized_pnl_pct": 0,
            "wins": 0,
            "losses": 0,
        }

    pnls = [t["pnl_pct"] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        "total_closed": len(closed),
        "total_open": len(open_pos),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 2) if pnls else 0,
        "best_trade_pct": round(max(pnls), 2) if pnls else 0,
        "worst_trade_pct": round(min(pnls), 2) if pnls else 0,
        "avg_hold_days": round(sum(t["days_held"] for t in closed) / len(closed), 1),
        "total_realized_pnl_pct": round(sum(pnls), 2),
        "wins": len(wins),
        "losses": len(losses),
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

    # --- Update trade tracking ---
    print("Updating trade tracking...")
    trades_state = update_trades(all_scores, data)
    trades_summary = compute_trades_summary(trades_state)
    print(f"  Open positions: {trades_summary['total_open']}")
    print(f"  Closed trades: {trades_summary['total_closed']}")
    if trades_summary['total_closed'] > 0:
        print(f"  Win rate: {trades_summary['win_rate']}%")
        print(f"  Avg P&L: {trades_summary['avg_pnl_pct']}%")

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
        "trades": {
            "summary": trades_summary,
            "open_positions": trades_state["open_positions"],
            "closed_trades": trades_state["closed_trades"],
            "equity_curve": trades_state["equity_curve"],
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
        json.dump(full_data, f, indent=2, cls=SafeJSONEncoder)
    print(f"\n  Wrote {full_path} ({os.path.getsize(full_path) / 1024:.0f} KB)")

    # Write per-ticker JSON files
    for score in all_scores:
        ticker_path = os.path.join(TICKERS_DIR, f"{score['ticker']}.json")
        with open(ticker_path, "w") as f:
            json.dump(score, f, indent=2, cls=SafeJSONEncoder)

    print(f"  Wrote {len(all_scores)} ticker files")

    # Write timestamp
    ts_path = os.path.join(DATA_OUT_DIR, "last_run.txt")
    with open(ts_path, "w") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d"))

    print(f"\nDone. {len(active_signals)} active signals generated.")
    return full_data


if __name__ == "__main__":
    run_scan()
