#!/usr/bin/env python3
"""
Crypto CDPT Daily Scanner — Comprehensive Edition
====================================================
Produces detailed backtest data, trade logs, equity curves,
open positions, and explicit trading instructions.

Output: crypto/docs/data/full.json
"""

import os
import sys
import json
import math
import datetime
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(CRYPTO_DIR, "src"))

from prepare import load_data, UNIVERSE, TRANSACTION_COST_BPS
from strategy import compute_cdpt_features, Config, generate_signals, detect_regime

# Output directories
DOCS_DIR = os.path.join(CRYPTO_DIR, "docs")
DATA_OUT_DIR = os.path.join(DOCS_DIR, "data")
TICKERS_DIR = os.path.join(DATA_OUT_DIR, "tickers")

# Position tracking file (persists between runs)
POSITIONS_FILE = os.path.join(DATA_OUT_DIR, "positions.json")
TRADE_LOG_FILE = os.path.join(DATA_OUT_DIR, "trade_log.json")

CFG = Config()


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
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


def safe_float(v, default=None):
    if v is None:
        return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (ValueError, TypeError):
        return default


def load_positions():
    """Load persisted open positions from previous run."""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return {}


def save_positions(positions):
    """Save open positions for next run."""
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, cls=SafeJSONEncoder)


def load_trade_log():
    """Load historical trade log."""
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f:
            return json.load(f)
    return []


def save_trade_log(trades):
    """Save trade log."""
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(trades, f, indent=2, cls=SafeJSONEncoder)




def run_full_backtest(data_dict, start_date, end_date, cfg=None):
    """
    Run comprehensive backtest returning every trade with full details.
    Returns (trades_list, daily_equity_series).
    """
    if cfg is None:
        cfg = Config()

    btc_close = data_dict.get("BTC-USD", pd.DataFrame()).get("Close")

    # Pre-compute features
    features_cache = {}
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            continue
        try:
            volume = df.get("Volume")
            leader = btc_close if ticker != "BTC-USD" else None
            features_cache[ticker] = compute_cdpt_features(df["Close"], volume, leader)
        except Exception:
            pass

    market_df = data_dict["BTC-USD"]
    dates = market_df.loc[start_date:end_date].index

    positions = {}
    closed_trades = []
    equity = 10000.0
    equity_curve = []
    tc = TRANSACTION_COST_BPS / 10000

    for date in dates:
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index and "Close" in df.columns:
                prices[ticker] = float(df.loc[date, "Close"])

        # Check exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1
            price = prices.get(ticker)
            if price is None or np.isnan(price):
                continue

            pnl_pct = (price / pos["entry_price"] - 1)
            should_exit = False
            exit_reason = ""

            if pnl_pct <= cfg.stop_loss:
                should_exit, exit_reason = True, "stop_loss"
            elif pnl_pct >= cfg.take_profit:
                should_exit, exit_reason = True, "take_profit"
            elif pos["days_held"] >= cfg.max_hold_days:
                should_exit, exit_reason = True, "max_hold"
            elif pos["days_held"] >= 2:
                feat = features_cache.get(ticker)
                if feat is not None and date in feat.index:
                    mz = feat.loc[date].get("mtmdi_zscore", 0)
                    mv = feat.loc[date].get("mtmdi_velocity", 0)
                    if not np.isnan(mz) and not np.isnan(mv):
                        if abs(mz) < cfg.mtmdi_zscore_exit or mv < cfg.mtmdi_velocity_exit:
                            should_exit, exit_reason = True, "signal_resolved"

            if should_exit:
                net_pnl = pnl_pct - 2 * tc
                dollar_pnl = equity * pos["size"] * net_pnl
                equity += dollar_pnl
                closed_trades.append({
                    "ticker": ticker,
                    "display_name": ticker.replace("-USD", ""),
                    "entry_date": str(pos["entry_date"])[:10],
                    "entry_time": "23:59 UTC (daily close)",
                    "exit_date": str(date)[:10],
                    "exit_time": "23:59 UTC (daily close)",
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(price, 4),
                    "position_size_pct": round(pos["size"] * 100, 2),
                    "gross_pnl_pct": round(pnl_pct * 100, 2),
                    "net_pnl_pct": round(net_pnl * 100, 2),
                    "dollar_pnl": round(dollar_pnl, 2),
                    "days_held": pos["days_held"],
                    "exit_reason": exit_reason,
                    "stop_loss_price": round(pos["entry_price"] * (1 + cfg.stop_loss), 4),
                    "take_profit_price": round(pos["entry_price"] * (1 + cfg.take_profit), 4),
                    "signal_strength": round(pos.get("strength", 0), 3),
                    "regime_at_entry": pos.get("regime", "unknown"),
                })
                del positions[ticker]

        # Generate new signals
        features_dict = {}
        for ticker, feats in features_cache.items():
            if date in feats.index:
                features_dict[ticker] = feats.loc[date].to_dict()

        signals = generate_signals(date, features_dict, positions, cfg)
        total_exposure = sum(p["size"] for p in positions.values())

        for ticker, strength, size, regime in signals:
            if total_exposure >= cfg.max_total_exposure:
                break
            price = prices.get(ticker)
            if price is None or np.isnan(price):
                continue
            remaining = cfg.max_total_exposure - total_exposure
            size = min(size, remaining)
            if size < 0.005:
                continue
            positions[ticker] = {
                "entry_date": date,
                "entry_price": price,
                "direction": 1,
                "size": size,
                "strength": strength,
                "days_held": 0,
                "regime": regime,
            }
            total_exposure += size

        # Daily portfolio return
        daily_ret = 0.0
        for ticker, pos in positions.items():
            if pos["days_held"] < 1:
                continue
            if ticker in data_dict:
                df = data_dict[ticker]
                if date in df.index:
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev = df.iloc[idx - 1]["Close"]
                        curr = df.iloc[idx]["Close"]
                        daily_ret += (float(curr) / float(prev) - 1) * pos["size"]

        equity *= (1 + daily_ret)
        equity_curve.append({
            "date": str(date)[:10],
            "equity": round(equity, 2),
            "daily_return_pct": round(daily_ret * 100, 4),
            "positions_open": len(positions),
            "total_exposure_pct": round(sum(p["size"] for p in positions.values()) * 100, 1),
        })

    return closed_trades, equity_curve




def score_coin(ticker, features, close_series, btc_close=None):
    """Score a single crypto asset. Returns dict with all signal details."""
    if features is None or len(features) == 0:
        return None

    latest = features.iloc[-1]
    if latest.isna().all() and len(features) >= 2:
        latest = features.iloc[-2]
    if latest.isna().all():
        return None

    mtmdi_z = latest.get("mtmdi_zscore", np.nan)
    mtmdi_dir = latest.get("mtmdi_direction", np.nan)
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

    clean_close = close_series.dropna()
    if clean_close.empty:
        return None
    current_price = float(clean_close.iloc[-1])
    price_date = str(clean_close.index[-1])[:10]

    # Safe values
    cacs_val = cacs if not np.isnan(cacs) else 0
    mpr_val = mpr_z if not np.isnan(mpr_z) else 0
    vel_val = mtmdi_vel if not np.isnan(mtmdi_vel) else 0
    rc_val = range_compress if not np.isnan(range_compress) else 0.5
    vs_val = vol_surge if not np.isnan(vol_surge) else 0
    direction_positive = mtmdi_dir > 0 if not np.isnan(mtmdi_dir) else False

    # Entry condition checks
    has_mtmdi = abs(mtmdi_z) >= CFG.mtmdi_zscore_entry
    has_direction = direction_positive
    has_cascade = cacs_val > CFG.cacs_entry_threshold
    has_momentum = mpr_val > CFG.mpr_threshold
    has_velocity = vel_val > CFG.velocity_threshold
    has_range_compress = rc_val < CFG.range_compress_threshold
    has_vol_surge = vs_val > 0

    confirming = (int(has_cascade) + int(has_momentum) + int(has_velocity) +
                 int(has_range_compress) + int(has_vol_surge))

    conviction = (
        min(abs(mtmdi_z) / 3.0, 1.0) * 30 +
        min(max(vel_val, 0) / 1.0, 1.0) * 25 +
        min(abs(cacs_val) / 0.08, 1.0) * 15 +
        min(max(mpr_val, 0) / 2.0, 1.0) * 15 +
        (1.0 - rc_val) * 15
    )

    is_signal = bool(has_mtmdi and has_direction and confirming >= CFG.min_confirming)

    regime = detect_regime(
        vol_ratio if not np.isnan(vol_ratio) else 1.0,
        vol_30d
    )

    # Position size calculation
    v = max(float(vol_30d), 0.05)
    strength = conviction / 100.0
    raw_size = CFG.vol_target / v * strength
    if regime == "high":
        raw_size *= CFG.high_vol_reduction
    position_size = min(raw_size, CFG.max_position_pct)

    # Price levels for trading
    stop_loss_price = round(current_price * (1 + CFG.stop_loss), 4)
    take_profit_price = round(current_price * (1 + CFG.take_profit), 4)

    # Returns
    rets = {}
    for w in [7, 14, 30, 90, 180, 365]:
        key = f"ret_{w}d"
        val = latest.get(key, np.nan)
        if not np.isnan(val):
            rets[f"{w}d"] = round(float(val) * 100, 2)

    # Chart data (last 365 days)
    chart_data = []
    chart_close = close_series.iloc[-365:] if len(close_series) >= 365 else close_series
    for date, price in chart_close.items():
        p = float(price)
        if not (math.isnan(p) or math.isinf(p)):
            chart_data.append({"date": str(date)[:10], "price": round(p, 2)})

    display_name = ticker.replace("-USD", "")

    return {
        "ticker": ticker,
        "display_name": display_name,
        "price": round(current_price, 4),
        "price_date": price_date,
        "conviction": round(conviction, 1),
        "is_signal": is_signal,
        "position_size_pct": round(position_size * 100, 2) if is_signal else 0,
        "stop_loss_price": stop_loss_price if is_signal else None,
        "take_profit_price": take_profit_price if is_signal else None,
        "max_hold_days": CFG.max_hold_days,
        "mtmdi_zscore": safe_float(mtmdi_z),
        "mtmdi_direction": safe_float(mtmdi_dir),
        "mtmdi_velocity": safe_float(vel_val),
        "cascade_score": safe_float(cacs_val),
        "momentum_persistence": safe_float(mpr_val),
        "range_compression": safe_float(rc_val),
        "vol_30d": safe_float(vol_30d * 100, 0),
        "vol_regime": regime,
        "drawdown": safe_float(dd * 100) if not np.isnan(dd) else None,
        "position_in_range": safe_float(pos_range * 100) if not np.isnan(pos_range) else None,
        "returns": rets,
        "signals": {
            "mtmdi_spike": bool(has_mtmdi),
            "direction_positive": bool(has_direction),
            "velocity_accelerating": bool(has_velocity),
            "cascade_divergence": bool(has_cascade),
            "momentum_persistent": bool(has_momentum),
            "range_compressed": bool(has_range_compress),
            "volume_surge": bool(has_vol_surge),
        },
        "confirming_factors": confirming,
        "chart": chart_data,
    }




def update_live_positions(all_scores, data_dict):
    """
    Track live positions: close expired ones, open new signals.
    Returns (open_positions, new_trades_closed, new_signals).
    """
    today = datetime.date.today().isoformat()
    positions = load_positions()
    trade_log = load_trade_log()

    # Build price lookup
    prices = {}
    for s in all_scores:
        prices[s["ticker"]] = s["price"]

    # Check exits on existing positions
    newly_closed = []
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        pos["days_held"] = pos.get("days_held", 0) + 1
        price = prices.get(ticker)
        if price is None:
            continue

        pnl_pct = (price / pos["entry_price"] - 1)
        should_exit = False
        exit_reason = ""

        if pnl_pct <= CFG.stop_loss:
            should_exit, exit_reason = True, "STOP LOSS HIT"
        elif pnl_pct >= CFG.take_profit:
            should_exit, exit_reason = True, "TAKE PROFIT HIT"
        elif pos["days_held"] >= CFG.max_hold_days:
            should_exit, exit_reason = True, "MAX HOLD REACHED"
        else:
            # Check signal resolution
            score = next((s for s in all_scores if s["ticker"] == ticker), None)
            if score:
                mz = score.get("mtmdi_zscore", 0) or 0
                mv = score.get("mtmdi_velocity", 0) or 0
                if pos["days_held"] >= 2:
                    if abs(mz) < CFG.mtmdi_zscore_exit or mv < CFG.mtmdi_velocity_exit:
                        should_exit, exit_reason = True, "SIGNAL RESOLVED"

        if should_exit:
            tc = TRANSACTION_COST_BPS / 10000
            net_pnl = pnl_pct - 2 * tc
            trade = {
                "ticker": ticker,
                "display_name": ticker.replace("-USD", ""),
                "entry_date": pos["entry_date"],
                "exit_date": today,
                "entry_price": pos["entry_price"],
                "exit_price": round(price, 4),
                "position_size_pct": round(pos["size"] * 100, 2),
                "gross_pnl_pct": round(pnl_pct * 100, 2),
                "net_pnl_pct": round(net_pnl * 100, 2),
                "days_held": pos["days_held"],
                "exit_reason": exit_reason,
                "stop_loss_price": round(pos["entry_price"] * (1 + CFG.stop_loss), 4),
                "take_profit_price": round(pos["entry_price"] * (1 + CFG.take_profit), 4),
            }
            newly_closed.append(trade)
            trade_log.append(trade)
            del positions[ticker]

    # Open new positions from today's signals
    new_signals = []
    active_signals = [s for s in all_scores if s["is_signal"]]
    total_exposure = sum(p["size"] for p in positions.values())

    for s in active_signals:
        if s["ticker"] in positions:
            continue
        if total_exposure >= CFG.max_total_exposure:
            break
        size = s["position_size_pct"] / 100.0
        remaining = CFG.max_total_exposure - total_exposure
        size = min(size, remaining)
        if size < 0.005:
            continue

        positions[s["ticker"]] = {
            "entry_date": today,
            "entry_price": s["price"],
            "size": size,
            "strength": s["conviction"] / 100.0,
            "days_held": 0,
            "stop_loss_price": s["stop_loss_price"],
            "take_profit_price": s["take_profit_price"],
        }
        total_exposure += size
        new_signals.append({
            "ticker": s["ticker"],
            "display_name": s["display_name"],
            "action": "BUY",
            "entry_price": s["price"],
            "position_size_pct": round(size * 100, 2),
            "stop_loss_price": s["stop_loss_price"],
            "take_profit_price": s["take_profit_price"],
            "max_hold_days": CFG.max_hold_days,
            "conviction": s["conviction"],
            "confirming_factors": s["confirming_factors"],
        })

    # Format open positions for display
    open_positions = []
    for ticker, pos in positions.items():
        current_price = prices.get(ticker, pos["entry_price"])
        pnl_pct = (current_price / pos["entry_price"] - 1) * 100
        days_remaining = max(0, CFG.max_hold_days - pos.get("days_held", 0))
        open_positions.append({
            "ticker": ticker,
            "display_name": ticker.replace("-USD", ""),
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "current_price": round(current_price, 4),
            "pnl_pct": round(pnl_pct, 2),
            "position_size_pct": round(pos["size"] * 100, 2),
            "days_held": pos.get("days_held", 0),
            "days_remaining": days_remaining,
            "stop_loss_price": pos.get("stop_loss_price", round(pos["entry_price"] * (1 + CFG.stop_loss), 4)),
            "take_profit_price": pos.get("take_profit_price", round(pos["entry_price"] * (1 + CFG.take_profit), 4)),
            "status": "HOLD" if days_remaining > 0 else "EXIT TOMORROW",
        })

    save_positions(positions)
    save_trade_log(trade_log)

    return open_positions, newly_closed, new_signals, trade_log




def generate_trading_instructions(new_signals, open_positions, newly_closed):
    """
    Generate explicit, unambiguous trading instructions.
    This is DAILY trading — scan runs once per day after market close.
    """
    today = datetime.date.today()
    instructions = {
        "generated": datetime.datetime.now().isoformat(),
        "trading_frequency": "DAILY",
        "scan_time": "23:00 UTC (6:00 PM ET) — after daily candle close",
        "execution_window": "Execute within 1 hour of scan. Use LIMIT orders at scan price or better.",
        "summary": "",
        "actions_today": [],
        "open_position_management": [],
        "rules_reference": {
            "entry_rule": "BUY at current price when ALL conditions met: MTMDI z-score >= 1.0, direction positive, and 3+ of 5 confirming factors fire.",
            "exit_rules": [
                f"SELL if price drops {abs(CFG.stop_loss)*100:.0f}% from entry (STOP LOSS at entry_price * {1+CFG.stop_loss})",
                f"SELL if price rises {CFG.take_profit*100:.0f}% from entry (TAKE PROFIT at entry_price * {1+CFG.take_profit})",
                f"SELL after {CFG.max_hold_days} calendar days regardless of P&L (MAX HOLD)",
                "SELL if MTMDI z-score drops below 0.4 OR velocity reverses below -0.3 (SIGNAL RESOLVED)",
            ],
            "position_sizing": f"Size = {CFG.vol_target*100:.0f}% / coin_30d_vol * signal_strength. Max {CFG.max_position_pct*100:.0f}% per coin, {CFG.max_total_exposure*100:.0f}% total.",
            "order_type": "LIMIT order at current close price. If not filled within 1 hour, use MARKET order.",
            "exchange": "Any major CEX (Binance, Coinbase, Kraken). Use spot, NOT futures.",
        },
    }

    actions = []

    # New buys
    for sig in new_signals:
        actions.append({
            "action": "BUY",
            "ticker": sig["ticker"],
            "display_name": sig["display_name"],
            "order_type": "LIMIT",
            "limit_price": sig["entry_price"],
            "position_size_pct": sig["position_size_pct"],
            "stop_loss_price": sig["stop_loss_price"],
            "take_profit_price": sig["take_profit_price"],
            "max_hold_until": str(today + datetime.timedelta(days=CFG.max_hold_days)),
            "instruction": (
                f"BUY {sig['display_name']} — Place LIMIT BUY at ${sig['entry_price']:.4f}. "
                f"Position size: {sig['position_size_pct']:.1f}% of portfolio. "
                f"Immediately set STOP LOSS at ${sig['stop_loss_price']:.4f} "
                f"and TAKE PROFIT at ${sig['take_profit_price']:.4f}. "
                f"If not filled in 1hr, use MARKET order. "
                f"Max hold: exit by {today + datetime.timedelta(days=CFG.max_hold_days)} regardless."
            ),
            "conviction": sig["conviction"],
        })

    # Exits
    for trade in newly_closed:
        actions.append({
            "action": "SELL",
            "ticker": trade["ticker"],
            "display_name": trade["display_name"],
            "order_type": "MARKET",
            "exit_price": trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct": trade["net_pnl_pct"],
            "instruction": (
                f"SELL {trade['display_name']} — {trade['exit_reason']}. "
                f"Place MARKET SELL immediately. "
                f"Entry was ${trade['entry_price']:.4f}, current ${trade['exit_price']:.4f}, "
                f"P&L: {trade['net_pnl_pct']:+.2f}%."
            ),
        })

    # No action
    if not actions:
        actions.append({
            "action": "NO_ACTION",
            "instruction": "No new trades today. Hold existing positions and monitor stop losses.",
        })

    instructions["actions_today"] = actions

    # Open position management
    for pos in open_positions:
        instructions["open_position_management"].append({
            "ticker": pos["ticker"],
            "display_name": pos["display_name"],
            "action": "HOLD" if pos["status"] == "HOLD" else "PREPARE TO EXIT",
            "entry_price": pos["entry_price"],
            "current_price": pos["current_price"],
            "pnl_pct": pos["pnl_pct"],
            "days_held": pos["days_held"],
            "days_remaining": pos["days_remaining"],
            "stop_loss_price": pos["stop_loss_price"],
            "take_profit_price": pos["take_profit_price"],
            "instruction": (
                f"{'HOLD' if pos['status']=='HOLD' else 'EXIT TOMORROW'} {pos['display_name']} — "
                f"Entry ${pos['entry_price']:.4f}, Now ${pos['current_price']:.4f} "
                f"({pos['pnl_pct']:+.2f}%). "
                f"Day {pos['days_held']}/{CFG.max_hold_days}. "
                f"SL: ${pos['stop_loss_price']:.4f}, TP: ${pos['take_profit_price']:.4f}. "
                f"{'Exit at next scan if still open.' if pos['status']!='HOLD' else 'Keep stop loss and take profit orders active.'}"
            ),
        })

    n_buys = sum(1 for a in actions if a["action"] == "BUY")
    n_sells = sum(1 for a in actions if a["action"] == "SELL")
    instructions["summary"] = (
        f"{n_buys} new BUY{'s' if n_buys!=1 else ''}, "
        f"{n_sells} SELL{'s' if n_sells!=1 else ''}, "
        f"{len(open_positions)} open position{'s' if len(open_positions)!=1 else ''}."
    )

    return instructions




def compute_backtest_stats(trades, equity_curve):
    """Compute comprehensive backtest statistics."""
    if not trades or not equity_curve:
        return {}

    df = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_curve)

    total_trades = len(df)
    winners = df[df["net_pnl_pct"] > 0]
    losers = df[df["net_pnl_pct"] <= 0]

    win_rate = len(winners) / total_trades if total_trades > 0 else 0
    avg_win = winners["net_pnl_pct"].mean() if len(winners) > 0 else 0
    avg_loss = losers["net_pnl_pct"].mean() if len(losers) > 0 else 0
    best_trade = df["net_pnl_pct"].max() if total_trades > 0 else 0
    worst_trade = df["net_pnl_pct"].min() if total_trades > 0 else 0
    avg_hold = df["days_held"].mean() if total_trades > 0 else 0

    gross_wins = winners["net_pnl_pct"].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers["net_pnl_pct"].sum()) if len(losers) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (999 if gross_wins > 0 else 0)

    # Equity curve stats
    final_equity = eq["equity"].iloc[-1] if len(eq) > 0 else 10000
    total_return = (final_equity / 10000 - 1) * 100
    n_days = len(eq)
    n_years = n_days / 365
    cagr = ((final_equity / 10000) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    daily_rets = eq["daily_return_pct"] / 100
    sharpe = 0
    if daily_rets.std() > 0:
        excess = daily_rets - 0.02 / 365
        sharpe = excess.mean() / excess.std() * np.sqrt(365)

    # Max drawdown
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    # Exit reason breakdown
    exit_reasons = {}
    if total_trades > 0:
        for reason in df["exit_reason"].unique():
            subset = df[df["exit_reason"] == reason]
            exit_reasons[reason] = {
                "count": len(subset),
                "pct": round(len(subset) / total_trades * 100, 1),
                "avg_pnl": round(subset["net_pnl_pct"].mean(), 2),
            }

    # Monthly breakdown
    monthly = []
    if total_trades > 0:
        df["exit_month"] = pd.to_datetime(df["exit_date"]).dt.to_period("M")
        for month, group in df.groupby("exit_month"):
            monthly.append({
                "month": str(month),
                "trades": len(group),
                "wins": len(group[group["net_pnl_pct"] > 0]),
                "total_pnl_pct": round(group["net_pnl_pct"].sum(), 2),
                "avg_pnl_pct": round(group["net_pnl_pct"].mean(), 2),
                "best": round(group["net_pnl_pct"].max(), 2),
                "worst": round(group["net_pnl_pct"].min(), 2),
            })

    return {
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "best_trade_pct": round(best_trade, 2),
        "worst_trade_pct": round(worst_trade, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_hold_days": round(avg_hold, 1),
        "sharpe": round(sharpe, 2),
        "cagr_pct": round(cagr, 1),
        "total_return_pct": round(total_return, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "final_equity": round(final_equity, 2),
        "start_equity": 10000,
        "n_days": n_days,
        "exit_reasons": exit_reasons,
        "monthly_breakdown": monthly,
    }




def run_scan():
    """Run the full daily crypto scan with comprehensive output."""
    now = datetime.datetime.now()
    print(f"Crypto CDPT Comprehensive Daily Scan — {now.isoformat()}")
    print("=" * 60)

    print("Loading crypto data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    btc_close = data.get("BTC-USD", pd.DataFrame()).get("Close")
    btc_price = float(btc_close.dropna().iloc[-1]) if btc_close is not None and not btc_close.dropna().empty else 0
    btc_sma50 = float(btc_close.rolling(50).mean().dropna().iloc[-1]) if btc_close is not None else 0
    btc_regime = "BULL" if btc_price > btc_sma50 else "BEAR"

    # Score all coins
    print("Scoring coins...")
    all_scores = []
    for ticker, df in data.items():
        if "Close" not in df.columns:
            continue
        try:
            volume = df.get("Volume")
            leader = btc_close if ticker != "BTC-USD" else None
            features = compute_cdpt_features(df["Close"], volume, leader)
            score = score_coin(ticker, features, df["Close"], btc_close)
            if score is not None:
                all_scores.append(score)
        except Exception as e:
            print(f"  Warning: {ticker}: {e}")

    all_scores.sort(key=lambda s: s["conviction"], reverse=True)
    print(f"  Scored {len(all_scores)} coins, {sum(1 for s in all_scores if s['is_signal'])} active signals")

    # Update live positions
    print("Updating positions...")
    open_positions, newly_closed, new_signals, full_trade_log = update_live_positions(all_scores, data)
    print(f"  Open: {len(open_positions)}, Closed today: {len(newly_closed)}, New buys: {len(new_signals)}")

    # Generate trading instructions
    print("Generating trading instructions...")
    instructions = generate_trading_instructions(new_signals, open_positions, newly_closed)

    # Run backtests on multiple periods
    print("Running backtests (this may take a minute)...")

    # Last 90 days backtest (detailed)
    end_date = now.strftime("%Y-%m-%d")
    start_90d = (now - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
    trades_90d, equity_90d = run_full_backtest(data, start_90d, end_date, CFG)
    stats_90d = compute_backtest_stats(trades_90d, equity_90d)
    print(f"  90-day backtest: {stats_90d.get('total_trades', 0)} trades, Sharpe {stats_90d.get('sharpe', 0)}")

    # Full test period (2023-2026)
    trades_test, equity_test = run_full_backtest(data, "2023-10-01", end_date, CFG)
    stats_test = compute_backtest_stats(trades_test, equity_test)
    print(f"  Test period: {stats_test.get('total_trades', 0)} trades, Sharpe {stats_test.get('sharpe', 0)}")

    # Train period
    trades_train, equity_train = run_full_backtest(data, "2018-01-01", "2021-12-31", CFG)
    stats_train = compute_backtest_stats(trades_train, equity_train)

    # Validation period
    trades_valid, equity_valid = run_full_backtest(data, "2022-04-01", "2023-06-30", CFG)
    stats_valid = compute_backtest_stats(trades_valid, equity_valid)

    # Build daily top-1 pick history (last 90+ days from test backtest)
    daily_top1_history = []
    if trades_test:
        # Group trades by entry date and take the highest strength one
        trades_by_date = {}
        for t in trades_test:
            d = t["entry_date"]
            if d not in trades_by_date or t.get("signal_strength", 0) > trades_by_date[d].get("signal_strength", 0):
                trades_by_date[d] = t
        for d in sorted(trades_by_date.keys())[-120:]:  # last 120 entries
            t = trades_by_date[d]
            daily_top1_history.append({
                "date": t["entry_date"],
                "ticker": t["ticker"],
                "display_name": t["display_name"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "exit_date": t["exit_date"],
                "pnl_pct": t["net_pnl_pct"],
                "days_held": t["days_held"],
                "exit_reason": t["exit_reason"],
            })

    # Assemble output
    os.makedirs(TICKERS_DIR, exist_ok=True)

    full_data = {
        "generated": now.isoformat(),
        "strategy": "CDPT",
        "strategy_full_name": "Crypto Dispersion Pulse Trading",
        "version": "3-factor-focus",
        "trading_frequency": "DAILY",
        "asset_class": "crypto",
        "n_tickers_scored": len(all_scores),
        "n_active_signals": sum(1 for s in all_scores if s["is_signal"]),

        "btc": {
            "price": round(btc_price, 2),
            "sma50": round(btc_sma50, 2),
            "regime": btc_regime,
        },

        "config": {
            "mtmdi_zscore_entry": CFG.mtmdi_zscore_entry,
            "velocity_threshold": CFG.velocity_threshold,
            "range_compress_threshold": CFG.range_compress_threshold,
            "min_confirming_factors": CFG.min_confirming,
            "stop_loss_pct": CFG.stop_loss * 100,
            "take_profit_pct": CFG.take_profit * 100,
            "max_hold_days": CFG.max_hold_days,
            "max_position_pct": CFG.max_position_pct * 100,
            "max_total_exposure_pct": CFG.max_total_exposure * 100,
            "vol_target_pct": CFG.vol_target * 100,
        },

        "trading_instructions": instructions,

        "open_positions": open_positions,
        "newly_closed_today": newly_closed,
        "new_signals_today": new_signals,

        "backtest": {
            "last_90_days": {
                "stats": stats_90d,
                "trades": trades_90d[-200:],  # cap for JSON size
                "equity_curve": equity_90d,
            },
            "test_period": {
                "period": "2023-10-01 to today",
                "stats": stats_test,
                "trades": trades_test[-500:],
                "equity_curve": equity_test[-365:],  # last year of equity
            },
            "train_period": {
                "period": "2018-01-01 to 2021-12-31",
                "stats": stats_train,
            },
            "validation_period": {
                "period": "2022-04-01 to 2023-06-30",
                "stats": stats_valid,
            },
        },

        "daily_top1_history": daily_top1_history,

        "historical_picks": trades_test[-300:] if trades_test else [],

        "live_trade_log": full_trade_log[-100:],

        "top_10": [{
            "ticker": s["ticker"],
            "display_name": s["display_name"],
            "price": s["price"],
            "conviction": s["conviction"],
            "is_signal": s["is_signal"],
            "position_size_pct": s["position_size_pct"],
            "stop_loss_price": s["stop_loss_price"],
            "take_profit_price": s["take_profit_price"],
            "mtmdi_zscore": s["mtmdi_zscore"],
            "mtmdi_velocity": s["mtmdi_velocity"],
            "cascade_score": s["cascade_score"],
            "range_compression": s["range_compression"],
            "vol_regime": s["vol_regime"],
            "returns": s["returns"],
            "signals": s["signals"],
            "confirming_factors": s["confirming_factors"],
        } for s in all_scores[:10]],

        "all_coins": [{
            "ticker": s["ticker"],
            "display_name": s["display_name"],
            "price": s["price"],
            "conviction": s["conviction"],
            "is_signal": s["is_signal"],
            "mtmdi_zscore": s["mtmdi_zscore"],
            "mtmdi_velocity": s["mtmdi_velocity"],
            "vol_regime": s["vol_regime"],
            "returns": s.get("returns", {}),
            "confirming_factors": s["confirming_factors"],
        } for s in all_scores],
    }

    # Write main JSON
    full_path = os.path.join(DATA_OUT_DIR, "full.json")
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2, cls=SafeJSONEncoder)
    print(f"\n  Wrote {full_path} ({os.path.getsize(full_path) / 1024:.0f} KB)")

    # Write per-ticker files
    for score in all_scores:
        tp = os.path.join(TICKERS_DIR, f"{score['ticker'].replace('-', '_')}.json")
        with open(tp, "w") as f:
            json.dump(score, f, indent=2, cls=SafeJSONEncoder)

    # Write timestamp
    with open(os.path.join(DATA_OUT_DIR, "last_run.txt"), "w") as f:
        f.write(now.strftime("%Y-%m-%d %H:%M:%S UTC"))

    print(f"\nDone. {len(new_signals)} new signals, {len(open_positions)} open, {len(newly_closed)} closed.")
    return full_data


if __name__ == "__main__":
    run_scan()
