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

# Strategy config (best from experiment loop — must match train.py Config)
MTMDI_ZSCORE_ENTRY = 1.5
CACS_ENTRY_THRESHOLD = 0.02
MPR_THRESHOLD = 0.5
MTMDI_ZSCORE_EXIT = 0.5    # Exit when dispersion resolves (matches backtest)
STOP_LOSS = -0.08
TAKE_PROFIT = 0.20
MAX_HOLD_DAYS = 63
VOL_TARGET = 0.15
MAX_POSITION_PCT = 0.05    # 5% per position (matches backtest)
MAX_TOTAL_EXPOSURE = 0.80  # 80% total exposure (matches backtest)
TRANSACTION_COST_BPS = 10  # 10bps per trade each way (matches backtest)


def compute_position_size(strength, vol_21d, vol_regime):
    """Volatility-targeted position sizing matching backtest strategy.py logic."""
    if vol_21d < 0.01:
        vol_21d = 0.01
    base_size = VOL_TARGET / vol_21d
    size = base_size * strength
    if vol_regime == "high":
        size *= 0.5
    elif vol_regime == "low":
        size *= 1.0
    return min(size, MAX_POSITION_PCT)


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

    # Signal strength (0-1) for position sizing — matches backtest
    strength = conviction / 100.0

    # Regime
    regime = detect_regime(
        vol_ratio if not np.isnan(vol_ratio) else 1.0,
        vol_21d
    )

    # Compute position size matching backtest
    pos_size = compute_position_size(strength, vol_21d, regime) if is_signal else 0.0

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
        "strength": round(float(strength), 4),
        "vol_21d_raw": float(vol_21d),  # raw decimal for position sizing
        "vol_21d": round(float(vol_21d) * 100, 1),  # as percentage for display
        "vol_regime": regime,
        "position_size": round(float(pos_size), 4),
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


def update_trades(all_scores, data, features_cache=None):
    """
    Update trade tracking with realistic execution matching the backtest:
    - Execute pending signals from previous day at today's open (next-day execution)
    - Check exits at close (stop loss, take profit, time, MTMDI resolution)
    - Apply transaction costs (10bps each way)
    - Store new signals as pending for next-day open execution
    Returns updated trades state dict.
    """
    state = load_trades_state()
    today = datetime.date.today().isoformat()
    tc = TRANSACTION_COST_BPS / 10000

    # Build price lookups
    close_price_map = {s["ticker"]: s["price"] for s in all_scores}
    open_price_map = {}
    for ticker, df in data.items():
        if "Open" in df.columns:
            clean = df["Open"].dropna()
            if len(clean) > 0:
                open_price_map[ticker] = round(float(clean.iloc[-1]), 2)

    # Build MTMDI z-score lookup for resolution exits
    mtmdi_map = {}
    for s in all_scores:
        if s.get("mtmdi_zscore") is not None:
            mtmdi_map[s["ticker"]] = abs(s["mtmdi_zscore"])

    # --- Execute pending signals from previous day at today's open ---
    pending = state.get("pending_signals", [])
    still_open = list(state["open_positions"])
    open_tickers = {p["ticker"] for p in still_open}

    for sig in pending:
        ticker = sig["ticker"]
        if ticker in open_tickers:
            continue
        open_price = open_price_map.get(ticker)
        signal_close = sig.get("signal_close_price", 0)
        if open_price is None:
            continue
        # Gap-down protection: skip if open gaps past stop loss level
        if signal_close > 0:
            gap_pct = (open_price / signal_close) - 1
            if gap_pct <= STOP_LOSS:
                continue
        sig_size = sig.get("position_size", MAX_POSITION_PCT)
        still_open.append({
            "ticker": ticker,
            "entry_date": today,
            "entry_price": open_price,  # Enter at today's open
            "current_price": open_price,
            "unrealized_pnl_pct": 0.0,
            "days_held": 0,
            "conviction": sig.get("conviction", 0),
            "position_size": round(sig_size, 4),
            "stop_loss_price": round(open_price * (1 + STOP_LOSS), 2),
            "take_profit_price": round(open_price * (1 + TAKE_PROFIT), 2),
        })
        open_tickers.add(ticker)

    # --- Check exits for all open positions at close ---
    remaining_open = []
    for pos in still_open:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        entry_date = pos["entry_date"]

        current_price = close_price_map.get(ticker)
        if current_price is None:
            if ticker in data and "Close" in data[ticker].columns:
                clean = data[ticker]["Close"].dropna()
                current_price = round(float(clean.iloc[-1]), 2) if len(clean) > 0 else None
            if current_price is None:
                remaining_open.append(pos)
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
        elif ticker in mtmdi_map and mtmdi_map[ticker] < MTMDI_ZSCORE_EXIT:
            exit_reason = "mtmdi_resolved"

        if exit_reason:
            # Apply transaction costs (10bps entry + 10bps exit)
            net_pnl_pct = pnl_pct - 2 * tc
            state["closed_trades"].append({
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": today,
                "exit_price": current_price,
                "pnl_pct": round(net_pnl_pct * 100, 2),
                "days_held": days_held,
                "exit_reason": exit_reason,
                "conviction": pos.get("conviction", 0),
                "position_size": pos.get("position_size", MAX_POSITION_PCT),
            })
        else:
            pos["current_price"] = current_price
            pos["unrealized_pnl_pct"] = round(pnl_pct * 100, 2)
            pos["days_held"] = days_held
            remaining_open.append(pos)

    state["open_positions"] = remaining_open

    # --- Store new active signals as pending for next-day open ---
    active_signals = [s for s in all_scores if s["is_signal"]]
    closed_recently = {
        t["ticker"] for t in state["closed_trades"]
        if t.get("exit_date", "") >= (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    }
    current_open_tickers = {p["ticker"] for p in remaining_open}

    new_pending = []
    # Respect max total exposure when queuing new signals
    current_exposure = sum(p.get("position_size", MAX_POSITION_PCT) for p in remaining_open)
    for s in sorted(active_signals, key=lambda x: x["conviction"], reverse=True):
        if s["ticker"] not in current_open_tickers and s["ticker"] not in closed_recently:
            sig_size = s.get("position_size", MAX_POSITION_PCT)
            if current_exposure + sig_size > MAX_TOTAL_EXPOSURE:
                continue
            new_pending.append({
                "ticker": s["ticker"],
                "conviction": round(s["conviction"], 1),
                "signal_close_price": s["price"],  # Today's close for gap-down check
                "position_size": round(sig_size, 4),
                "vol_21d_raw": s.get("vol_21d_raw", 0.15),
                "vol_regime": s.get("vol_regime", "normal"),
            })
            current_exposure += sig_size
    state["pending_signals"] = new_pending

    # --- Update equity curve (size-weighted returns matching backtest) ---
    if not state["equity_curve"]:
        prev_value = 10000.0
    else:
        prev_value = state["equity_curve"][-1]["value"]

    n_open = len(remaining_open)

    # Compute size-weighted daily return like backtest:
    # daily_ret = sum(stock_return * position_size) for all open positions
    daily_ret = 0.0
    prev_prices = state.get("_prev_prices", {})

    for pos in remaining_open:
        ticker = pos["ticker"]
        current_price = pos.get("current_price", pos["entry_price"])
        size = pos.get("position_size", MAX_POSITION_PCT)

        if pos["entry_date"] == today:
            # Entry day: return from open (entry price) to close
            if pos["entry_price"] > 0:
                stock_ret = (current_price / pos["entry_price"]) - 1
                daily_ret += stock_ret * size
        else:
            # Subsequent days: return from previous close to current close
            prev_price = prev_prices.get(ticker, pos["entry_price"])
            if prev_price > 0:
                stock_ret = (current_price / prev_price) - 1
                daily_ret += stock_ret * size

    # Account for closed trades today (realized P&L weighted by position size)
    closed_today = [t for t in state["closed_trades"] if t.get("exit_date") == today]
    for ct in closed_today:
        ct_ticker = ct["ticker"]
        ct_size = ct.get("position_size", MAX_POSITION_PCT)
        prev_price = prev_prices.get(ct_ticker, ct["entry_price"])
        if prev_price > 0:
            stock_ret = (ct["exit_price"] / prev_price) - 1
            daily_ret += stock_ret * ct_size

    new_value = round(prev_value * (1 + daily_ret), 2)

    # Store current prices for next day's return calculation
    new_prev_prices = {}
    for pos in remaining_open:
        new_prev_prices[pos["ticker"]] = pos.get("current_price", pos["entry_price"])
    state["_prev_prices"] = new_prev_prices

    avg_unrealized = 0
    if n_open > 0:
        avg_unrealized = sum(p.get("unrealized_pnl_pct", 0) for p in remaining_open) / n_open

    eq_entry = {
        "date": today,
        "value": new_value,
        "n_open": n_open,
        "avg_unrealized": round(avg_unrealized, 2),
    }
    if state["equity_curve"] and state["equity_curve"][-1]["date"] == today:
        state["equity_curve"][-1] = eq_entry
    else:
        state["equity_curve"].append(eq_entry)

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


def backfill_trades(data, market_close, lookback_days=63):
    """
    Run the strategy over recent history to seed trade tracking with past trades.
    Uses the same execution model as the backtest: next-day open entry, gap-down
    protection, MTMDI resolution exit, and transaction costs.
    Returns a trades state dict.
    """
    print("  Backfilling historical trades...")
    tc = TRANSACTION_COST_BPS / 10000

    # Pre-compute features
    features_cache = {}
    for ticker, df in data.items():
        if "Close" not in df.columns:
            continue
        try:
            volume = df.get("Volume")
            features_cache[ticker] = compute_features(df["Close"], volume, market_close)
        except Exception:
            pass

    # Get trading dates from SPY
    spy_df = data.get("SPY")
    if spy_df is None or len(spy_df) < lookback_days:
        return {"open_positions": [], "closed_trades": [], "equity_curve": [], "pending_signals": []}

    all_dates = spy_df.index[-lookback_days:]
    positions = {}  # ticker -> dict
    closed_trades = []
    pending_signals = []  # (ticker, strength, conviction, signal_close, vol_21d, vol_regime)
    equity_value = 10000.0
    equity_curve = []
    prev_close_prices = {}  # for daily return calculation

    for date in all_dates:
        date_str = str(date.date())

        # Get prices
        open_prices = {}
        close_prices = {}
        for ticker, df in data.items():
            if date in df.index:
                if "Open" in df.columns:
                    val = df.loc[date, "Open"]
                    if not np.isnan(val):
                        open_prices[ticker] = float(val)
                if "Close" in df.columns:
                    val = df.loc[date, "Close"]
                    if not np.isnan(val):
                        close_prices[ticker] = float(val)

        # Execute pending from yesterday at today's open
        current_exposure = sum(
            pos.get("position_size", MAX_POSITION_PCT) for pos in positions.values()
        )
        for sig_ticker, sig_strength, sig_conv, sig_close, sig_vol, sig_regime in pending_signals:
            if sig_ticker in positions:
                continue
            op = open_prices.get(sig_ticker)
            if op is None:
                continue
            if sig_close > 0:
                gap = (op / sig_close) - 1
                if gap <= STOP_LOSS:
                    continue
            pos_size = compute_position_size(sig_strength, sig_vol, sig_regime)
            if current_exposure + pos_size > MAX_TOTAL_EXPOSURE:
                continue
            positions[sig_ticker] = {
                "entry_date": date_str,
                "entry_price": round(op, 2),
                "conviction": sig_conv,
                "position_size": round(pos_size, 4),
                "days_held": 0,
            }
            current_exposure += pos_size
        pending_signals = []

        # Check exits at close
        exited_today = []
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1
            cp = close_prices.get(ticker)
            if cp is None:
                continue

            pnl = (cp / pos["entry_price"]) - 1
            cal_days = (date.date() - datetime.date.fromisoformat(pos["entry_date"])).days
            trading_days_approx = int(cal_days * 5 / 7)

            # Get MTMDI for resolution check
            mtmdi_z = None
            feat = features_cache.get(ticker)
            if feat is not None and date in feat.index:
                mz = feat.loc[date].get("mtmdi_zscore", np.nan)
                if not np.isnan(mz):
                    mtmdi_z = abs(mz)

            exit_reason = None
            if pnl <= STOP_LOSS:
                exit_reason = "stop_loss"
            elif pnl >= TAKE_PROFIT:
                exit_reason = "take_profit"
            elif trading_days_approx >= MAX_HOLD_DAYS:
                exit_reason = "time_exit"
            elif mtmdi_z is not None and mtmdi_z < MTMDI_ZSCORE_EXIT:
                exit_reason = "mtmdi_resolved"

            if exit_reason:
                net_pnl = pnl - 2 * tc
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": date_str,
                    "exit_price": round(cp, 2),
                    "pnl_pct": round(net_pnl * 100, 2),
                    "days_held": cal_days,
                    "exit_reason": exit_reason,
                    "conviction": pos.get("conviction", 0),
                    "position_size": pos.get("position_size", MAX_POSITION_PCT),
                })
                exited_today.append(ticker)

        for ticker in exited_today:
            del positions[ticker]

        # Generate new signals
        features_dict = {}
        for ticker, feats in features_cache.items():
            if date in feats.index:
                features_dict[ticker] = feats.loc[date]

        for ticker, feat_row in features_dict.items():
            if ticker in positions:
                continue
            mtmdi_z = feat_row.get("mtmdi_zscore", 0)
            mtmdi_dir = feat_row.get("mtmdi_direction", 0)
            cacs = feat_row.get("cacs", 0)
            mpr = feat_row.get("mpr_zscore", 0)
            vol_21d = feat_row.get("vol_21d", 0.15)
            vol_ratio = feat_row.get("vol_ratio_5_21", 1.0)

            if any(np.isnan(v) for v in [mtmdi_z, mtmdi_dir]):
                continue
            if abs(mtmdi_z) < MTMDI_ZSCORE_ENTRY:
                continue
            if mtmdi_dir <= 0:
                continue

            cacs_val = cacs if not np.isnan(cacs) else 0
            mpr_val = mpr if not np.isnan(mpr) else 0
            if not (cacs_val > CACS_ENTRY_THRESHOLD or mpr_val > MPR_THRESHOLD):
                continue

            strength = (
                min(abs(mtmdi_z) / 3.0, 1.0) * 0.5 +
                min(abs(cacs_val) / 0.05, 1.0) * 0.3 +
                min(max(mpr_val, 0) / 2.0, 1.0) * 0.2
            )
            conviction = round(strength * 100, 1)
            close_p = close_prices.get(ticker, 0)
            if np.isnan(vol_21d):
                vol_21d = 0.15
            if np.isnan(vol_ratio):
                vol_ratio = 1.0
            vol_regime = detect_regime(vol_ratio, vol_21d)
            pending_signals.append((ticker, strength, conviction, close_p, vol_21d, vol_regime))

        # Update equity curve (size-weighted returns matching backtest)
        daily_ret = 0.0
        for ticker, pos in positions.items():
            cp = close_prices.get(ticker)
            if cp is None:
                continue
            size = pos.get("position_size", MAX_POSITION_PCT)
            if pos["entry_date"] == date_str:
                # Entry day: return from open (entry price) to close
                if pos["entry_price"] > 0:
                    stock_ret = (cp / pos["entry_price"]) - 1
                    daily_ret += stock_ret * size
            else:
                # Subsequent days: return from previous close to current close
                prev_price = prev_close_prices.get(ticker, pos["entry_price"])
                if prev_price > 0:
                    stock_ret = (cp / prev_price) - 1
                    daily_ret += stock_ret * size

        # Account for exited positions' returns today
        closed_today = [t for t in closed_trades if t["exit_date"] == date_str]
        for ct in closed_today:
            ct_size = ct.get("position_size", MAX_POSITION_PCT)
            prev_price = prev_close_prices.get(ct["ticker"], ct["entry_price"])
            if prev_price > 0:
                stock_ret = (ct["exit_price"] / prev_price) - 1
                daily_ret += stock_ret * ct_size

        equity_value = round(equity_value * (1 + daily_ret), 2)

        n_pos = len(positions)
        avg_unrealized_pct = 0
        if n_pos > 0:
            total_unrealized = sum(
                ((close_prices.get(t, positions[t]["entry_price"]) / positions[t]["entry_price"]) - 1)
                for t in positions if positions[t]["entry_price"] > 0
            )
            avg_unrealized_pct = (total_unrealized / n_pos) * 100

        equity_curve.append({
            "date": date_str,
            "value": equity_value,
            "n_open": n_pos,
            "avg_unrealized": round(avg_unrealized_pct, 2),
        })

        # Store close prices for next day's return calculation
        prev_close_prices = dict(close_prices)

    # Convert remaining positions to open_positions format
    open_positions = []
    for ticker, pos in positions.items():
        cp = close_prices.get(ticker, pos["entry_price"])
        pnl = (cp / pos["entry_price"] - 1) if pos["entry_price"] > 0 else 0
        cal_days = (datetime.date.today() - datetime.date.fromisoformat(pos["entry_date"])).days
        open_positions.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "current_price": round(cp, 2),
            "unrealized_pnl_pct": round(pnl * 100, 2),
            "days_held": cal_days,
            "conviction": pos.get("conviction", 0),
            "position_size": pos.get("position_size", MAX_POSITION_PCT),
            "stop_loss_price": round(pos["entry_price"] * (1 + STOP_LOSS), 2),
            "take_profit_price": round(pos["entry_price"] * (1 + TAKE_PROFIT), 2),
        })

    pending_out = [
        {"ticker": t, "conviction": c, "signal_close_price": sc,
         "position_size": round(compute_position_size(s, v, vr), 4)}
        for t, s, c, sc, v, vr in pending_signals
    ]

    print(f"  Backfill complete: {len(closed_trades)} closed trades, "
          f"{len(open_positions)} open positions")

    return {
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "equity_curve": equity_curve,
        "pending_signals": pending_out,
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

    # --- Backfill historical trades on first run ---
    trades_path = os.path.join(DATA_OUT_DIR, "trades.json")
    existing_state = load_trades_state()
    needs_backfill = (
        not os.path.exists(trades_path) or
        (not existing_state.get("closed_trades") and not existing_state.get("open_positions"))
    )
    if needs_backfill:
        print("No trade history found — running historical backfill (last 63 trading days)...")
        backfill_state = backfill_trades(data, market_close, lookback_days=63)
        save_trades_state(backfill_state)

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
            "mtmdi_zscore_exit": MTMDI_ZSCORE_EXIT,
            "cacs_entry_threshold": CACS_ENTRY_THRESHOLD,
            "mpr_threshold": MPR_THRESHOLD,
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "max_hold_days": MAX_HOLD_DAYS,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "execution": "next_day_open",
        },
        # Performance metrics from backtest with next-day-open execution model
        "performance": {
            "train": {"period": "2010-2019", "sharpe": 1.624, "cagr": "15.15%", "max_dd": "-6.90%"},
            "validation": {"period": "2020-2022", "sharpe": 1.562, "cagr": "18.53%", "max_dd": "-7.64%"},
            "test": {"period": "2023-2026", "sharpe": 2.354, "cagr": "21.00%", "max_dd": "-3.91%"},
            "walk_forward": {
                "avg_sharpe": 0,
                "folds_positive": "0/0",
                "fold_sharpes": [],
            },
            "bootstrap_ci": {"sharpe_low": 0, "sharpe_high": 0},
            "vs_spy": {"strategy_cagr": "21.00%", "spy_cagr": "19.11%"},
            "vs_random": {"percentile": "N/A"},
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
            "position_size": s.get("position_size", 0),
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
            "position_size": s.get("position_size", 0),
            "vol_21d_raw": s.get("vol_21d_raw", 0.15),
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
