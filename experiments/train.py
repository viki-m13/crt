#!/usr/bin/env python3
"""
train.py — Simplified momentum-pullback strategy with vol-adjusted exits.
==========================================================================
Designed so backtest matches live execution exactly.

CRITICAL EXECUTION REALISM:
- Signals at close → execute next day open + slippage
- Stop loss uses intraday LOW (not close) — like a real stop order
- Take profit uses intraday HIGH — like a real limit order
- Gap-through: if open gaps past stop, exit at OPEN (real gap risk)
- Trailing peak uses intraday HIGH, checked against LOW
- Vol-adjusted stops: each stock gets stops calibrated to its daily range
  (a 15% vol stock gets ~2.4% stop; a 30% vol stock gets ~4.7%)

Run: python train.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from prepare import (
    load_data, compute_features, evaluate_strategy,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TEST_START, TEST_END,
    TRANSACTION_COST_BPS,
)


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

@dataclass
class Config:
    # --- Market regime filter ---
    market_ret_63d_min: float = 0.01
    market_pos_range_min: float = 0.55

    # --- Entry filters ---
    min_ret_126d: float = 0.08      # Strong 6-month momentum
    min_ret_63d: float = 0.02       # Positive quarter
    max_ret_5d: float = 0.005       # Recent pullback
    min_pos_range: float = 0.55     # Upper 45% of 52w range
    min_vol_21d: float = 0.06       # Not dead
    max_vol_21d: float = 0.30       # Not too volatile
    max_drawdown_252d: float = -0.08 # Not in drawdown

    # --- Vol-adjusted exits (simple: stop + TP + time) ---
    # Stops wide enough to survive intraday noise
    stop_atr_mult: float = 3.0      # Stop at 3x daily vol (checked vs intraday LOW)
    take_profit: float = 0.15       # 15% TP (checked vs intraday HIGH)
    max_hold_days: int = 30         # 6 weeks

    # --- Position sizing ---
    position_size: float = 0.020    # 2% per position
    max_total_exposure: float = 0.85

    # --- Execution realism ---
    entry_slippage_bps: float = 10
    exit_slippage_bps: float = 5


EXCLUDED_TICKERS = {"SPY", "VIX"}


# ============================================================
# STRATEGY LOGIC
# ============================================================

def check_market_regime(features_dict, cfg):
    """Only trade when SPY is in favorable regime."""
    spy = features_dict.get("SPY")
    if spy is None:
        return False
    r = spy.get("ret_63d", np.nan)
    p = spy.get("position_in_52w_range", np.nan)
    if np.isnan(r) or np.isnan(p):
        return False
    return r >= cfg.market_ret_63d_min and p >= cfg.market_pos_range_min


def generate_signals(date, features_dict, positions, cfg):
    """Momentum + pullback entry signal."""
    signals = []

    for ticker, feat in features_dict.items():
        if ticker in positions or ticker in EXCLUDED_TICKERS:
            continue

        ret_126d = feat.get("ret_126d", np.nan)
        ret_63d = feat.get("ret_63d", np.nan)
        ret_5d = feat.get("ret_5d", np.nan)
        pos_range = feat.get("position_in_52w_range", np.nan)
        vol_21d = feat.get("vol_21d", np.nan)
        dd_252d = feat.get("drawdown_252d", np.nan)

        vals = [ret_126d, ret_63d, ret_5d, pos_range, vol_21d, dd_252d]
        if any(np.isnan(v) for v in vals):
            continue

        if ret_126d < cfg.min_ret_126d:
            continue
        if ret_63d < cfg.min_ret_63d:
            continue
        if ret_5d > cfg.max_ret_5d:
            continue
        if pos_range < cfg.min_pos_range:
            continue
        if vol_21d < cfg.min_vol_21d or vol_21d > cfg.max_vol_21d:
            continue
        if dd_252d < cfg.max_drawdown_252d:
            continue

        strength = (
            min(ret_126d / 0.25, 1.0) * 0.4 +
            min(ret_63d / 0.10, 1.0) * 0.2 +
            min(max(-ret_5d, 0) / 0.03, 1.0) * 0.2 +
            min(pos_range, 1.0) * 0.2
        )
        signals.append((ticker, strength, vol_21d))

    signals.sort(key=lambda s: s[1], reverse=True)
    return signals


def check_exits_realistic(pos, day_open, day_high, day_low, day_close, cfg,
                          exit_slippage):
    """
    Check exit conditions using INTRADAY OHLC for realistic execution.

    Simple 3-way exit: stop, TP, or time. No trailing stop.
    - STOP ORDER: checked vs intraday LOW (real stop order behavior)
    - LIMIT ORDER (TP): checked vs intraday HIGH (real limit order behavior)
    - MAX HOLD: exit at CLOSE with slippage (market-on-close order)
    - Gap-through: if open gaps past stop/TP, exit at OPEN (real gap risk)
    """
    entry = pos["entry_price"]
    stop_level = pos["stop_level"]

    pnl_at_open = (day_open / entry) - 1
    pnl_at_low = (day_low / entry) - 1
    pnl_at_high = (day_high / entry) - 1

    # --- 1. GAP-THROUGH STOP ---
    if pnl_at_open <= stop_level:
        actual_exit = day_open * (1 - exit_slippage)
        return True, "stop_loss_gap", actual_exit

    # --- 2. INTRADAY STOP (vs LOW) ---
    if pnl_at_low <= stop_level:
        stop_price = entry * (1 + stop_level)
        return True, "stop_loss", stop_price

    # --- 3. GAP-THROUGH TP ---
    if pnl_at_open >= cfg.take_profit:
        actual_exit = day_open * (1 - exit_slippage)
        return True, "take_profit_gap", actual_exit

    # --- 4. INTRADAY TP (vs HIGH) ---
    if pnl_at_high >= cfg.take_profit:
        tp_price = entry * (1 + cfg.take_profit)
        return True, "take_profit", tp_price

    # --- 5. MAX HOLD ---
    if pos["days_held"] >= cfg.max_hold_days:
        actual_exit = day_close * (1 - exit_slippage)
        return True, "max_hold", actual_exit

    return False, "", 0


def run_backtest(data_dict, start_date, end_date, cfg=None):
    """Run backtest with vol-adjusted stops and intraday execution."""
    if cfg is None:
        cfg = Config()

    market_close = None
    if "SPY" in data_dict and "Close" in data_dict["SPY"].columns:
        market_close = data_dict["SPY"]["Close"]

    features_cache = {}
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            continue
        try:
            features_cache[ticker] = compute_features(
                df["Close"], df.get("Volume"), market_close
            )
        except Exception:
            pass

    dates = data_dict["SPY"].loc[start_date:end_date].index
    positions = {}
    closed_trades = []
    daily_returns = []
    tc = TRANSACTION_COST_BPS / 10000
    entry_slip = cfg.entry_slippage_bps / 10000
    exit_slip = cfg.exit_slippage_bps / 10000
    pending_signals = []

    for date in dates:
        open_prices = {}
        high_prices = {}
        low_prices = {}
        close_prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                row = df.loc[date]
                if "Open" in df.columns:
                    open_prices[ticker] = row["Open"]
                if "High" in df.columns:
                    high_prices[ticker] = row["High"]
                if "Low" in df.columns:
                    low_prices[ticker] = row["Low"]
                if "Close" in df.columns:
                    close_prices[ticker] = row["Close"]

        # === EXECUTE PENDING SIGNALS AT OPEN ===
        total_exposure = sum(p["size"] for p in positions.values())
        for ticker, strength, vol_21d, signal_close in pending_signals:
            if total_exposure >= cfg.max_total_exposure:
                break
            if ticker in positions:
                continue
            op = open_prices.get(ticker)
            if op is None or np.isnan(op):
                continue

            # Compute vol-adjusted stop level
            daily_vol = vol_21d / np.sqrt(252)
            stop_pct = -cfg.stop_atr_mult * daily_vol

            # Gap-down protection
            if signal_close > 0:
                gap = (op / signal_close) - 1
                if gap <= stop_pct:
                    continue

            entry_price = op * (1 + entry_slip)
            size = min(cfg.position_size, cfg.max_total_exposure - total_exposure)
            if size < 0.005:
                continue

            positions[ticker] = {
                "entry_date": date,
                "entry_price": entry_price,
                "direction": 1,
                "size": size,
                "days_held": 0,
                "stop_level": stop_pct,
                "vol_21d": vol_21d,
            }
            total_exposure += size
        pending_signals = []

        # === CHECK EXITS ===
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1

            day_open = open_prices.get(ticker)
            day_high = high_prices.get(ticker)
            day_low = low_prices.get(ticker)
            day_close = close_prices.get(ticker)

            if any(v is None or np.isnan(v)
                   for v in [day_open, day_high, day_low, day_close]):
                continue

            should_exit, exit_reason, exit_price = check_exits_realistic(
                pos, day_open, day_high, day_low, day_close, cfg, exit_slip
            )

            if should_exit:
                pnl = (exit_price / pos["entry_price"]) - 1
                net_pnl = pnl - 2 * tc
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "direction": 1,
                    "size": pos["size"],
                    "gross_pnl": pnl,
                    "net_pnl": net_pnl,
                    "days_held": pos["days_held"],
                    "exit_reason": exit_reason,
                    "stop_level": pos["stop_level"],
                })
                del positions[ticker]

        # === GENERATE SIGNALS ===
        features_dict = {}
        for ticker, feats in features_cache.items():
            if date in feats.index:
                features_dict[ticker] = feats.loc[date].to_dict()

        if check_market_regime(features_dict, cfg):
            signals = generate_signals(date, features_dict, positions, cfg)
        else:
            signals = []

        pending_signals = [
            (t, s, v, close_prices.get(t, 0)) for t, s, v in signals
        ]

        # === DAILY PORTFOLIO RETURN ===
        daily_ret = 0.0
        # Open positions: close-to-close or entry-to-close
        for ticker, pos in positions.items():
            if pos["entry_date"] == date:
                cp = close_prices.get(ticker)
                if cp and pos["entry_price"] > 0:
                    daily_ret += (cp / pos["entry_price"] - 1) * pos["size"]
            elif ticker in data_dict:
                df = data_dict[ticker]
                if date in df.index:
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev = df.iloc[idx - 1]["Close"]
                        curr = df.iloc[idx]["Close"]
                        daily_ret += (curr / prev - 1) * pos["size"]

        # Closed positions today: use actual exit price
        for trade in closed_trades:
            if trade["exit_date"] == date:
                t_ticker = trade["ticker"]
                t_entry = trade["entry_price"]
                t_exit = trade["exit_price"]
                t_size = trade["size"]
                t_entry_date = trade["entry_date"]

                if t_entry_date == date:
                    daily_ret += (t_exit / t_entry - 1) * t_size
                else:
                    df = data_dict.get(t_ticker)
                    if df is not None and date in df.index:
                        idx = df.index.get_loc(date)
                        if idx > 0:
                            prev_close = df.iloc[idx - 1]["Close"]
                            daily_ret += (t_exit / prev_close - 1) * t_size

        daily_returns.append(daily_ret)

    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    return trades_df, daily_returns


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    cfg = Config()
    print(f"\nMomentum-Pullback + Vol-Adjusted Stops (realistic execution)")
    print(f"  Market:  SPY ret_63d>{cfg.market_ret_63d_min}, range>{cfg.market_pos_range_min}")
    print(f"  Entry:   ret_126d>{cfg.min_ret_126d}, ret_63d>{cfg.min_ret_63d}, "
          f"ret_5d<{cfg.max_ret_5d}")
    print(f"  Stop:    {cfg.stop_atr_mult}x daily vol (e.g. 20% ann vol → "
          f"{cfg.stop_atr_mult * 0.20 / np.sqrt(252):.1%} stop)")
    print(f"  TP/Hold: TP={cfg.take_profit}, MaxHold={cfg.max_hold_days}")
    print(f"  Exit:    Simple 3-way: vol-adjusted stop, TP, or time")
    print(f"  Size:    {cfg.position_size} per pos, {cfg.max_total_exposure} max")
    print(f"  Costs:   {TRANSACTION_COST_BPS}bps comm + "
          f"{cfg.entry_slippage_bps}bps entry + {cfg.exit_slippage_bps}bps exit slip")
    print(f"  Realism: stops vs LOW, TP vs HIGH, gaps at OPEN, vol-adjusted")

    all_metrics = {}
    all_trades = {}
    for period_name, start, end in [
        ("TRAINING", TRAIN_START, TRAIN_END),
        ("VALIDATION", VALID_START, VALID_END),
        ("TEST (OOS)", TEST_START, TEST_END),
    ]:
        print(f"\n{'='*60}")
        print(f"{period_name}: {start} to {end}")
        print(f"{'='*60}")
        t, r = run_backtest(data, start, end, cfg)
        m = evaluate_strategy(t, r, period_name)
        all_metrics[period_name] = m
        all_trades[period_name] = t

    print(f"\n{'='*60}")
    print(f"CROSS-PERIOD CONSISTENCY")
    print(f"{'='*60}")
    print(f"{'Period':<12} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>8}")
    print(f"-" * 62)
    for name, key in [("Train", "TRAINING"), ("Valid", "VALIDATION"), ("Test", "TEST (OOS)")]:
        m = all_metrics[key]
        print(f"{name:<12} {m['sharpe']:>8.3f} {m['cagr']:>7.1%} "
              f"{m['max_drawdown']:>7.1%} {m['win_rate']:>7.1%} {m['profit_factor']:>5.2f} {m['n_trades']:>8}")

    for name, key in [("Train", "TRAINING"), ("Valid", "VALIDATION"), ("Test", "TEST (OOS)")]:
        tdf = all_trades[key]
        if len(tdf) > 0:
            print(f"\n{name} exits:")
            for reason, cnt in tdf["exit_reason"].value_counts().items():
                pnl = tdf.loc[tdf["exit_reason"] == reason, "net_pnl"].mean()
                print(f"  {reason}: {cnt} ({cnt/len(tdf):.0%}) avg_pnl={pnl:.3f}")
            # Show average stop level used
            if "stop_level" in tdf.columns:
                avg_stop = tdf["stop_level"].mean()
                print(f"  Avg stop level: {avg_stop:.2%}")
