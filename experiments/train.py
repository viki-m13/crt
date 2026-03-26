#!/usr/bin/env python3
"""
train.py — The strategy file that the agent modifies.
=======================================================
This is the equivalent of autoresearch's train.py.
The agent experiments by modifying this file.

Current best: Experiment 12 — Sharpe 3.072
Config: SL=-0.07, TP=0.39, MaxHold=21

Run: python train.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# Import fixed evaluation harness (DO NOT MODIFY prepare.py)
from prepare import (
    load_data, compute_features, evaluate_strategy,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TRANSACTION_COST_BPS,
)


# ============================================================
# STRATEGY PARAMETERS (MODIFY THESE)
# ============================================================

@dataclass
class Config:
    # Entry signals
    mtmdi_zscore_entry: float = 1.5
    cacs_entry_threshold: float = 0.02
    mpr_threshold: float = 0.5

    # Exit signals
    mtmdi_zscore_exit: float = 0.5
    max_hold_days: int = 21       # Experiment 12 best
    stop_loss: float = -0.07      # Experiment 12 best
    take_profit: float = 0.39     # Experiment 12 best

    # Position sizing
    max_position_pct: float = 0.05
    max_total_exposure: float = 0.80
    vol_target: float = 0.15

    # Regime adaptation
    high_vol_reduction: float = 0.5
    low_vol_boost: float = 1.0


# ============================================================
# STRATEGY LOGIC (MODIFY THIS)
# ============================================================

def detect_regime(vol_ratio_5_21, vol_21d):
    """Classify volatility regime."""
    if vol_ratio_5_21 > 1.5 or vol_21d > 0.30:
        return "high"
    elif vol_ratio_5_21 < 0.7 and vol_21d < 0.12:
        return "low"
    return "normal"


def compute_position_size(strength, vol_21d, regime, cfg):
    """Volatility-targeted position sizing with regime adaptation."""
    vol_21d = max(vol_21d, 0.01)
    base = cfg.vol_target / vol_21d * strength
    if regime == "high":
        base *= cfg.high_vol_reduction
    elif regime == "low":
        base *= cfg.low_vol_boost
    return min(base, cfg.max_position_pct)


def generate_signals(date, features_dict, positions, cfg):
    """Generate entry signals for all stocks."""
    signals = []

    for ticker, feat in features_dict.items():
        if ticker in positions:
            continue

        mtmdi_z = feat.get("mtmdi_zscore", 0)
        mtmdi_dir = feat.get("mtmdi_direction", 0)
        cascade = feat.get("cacs", 0)
        mpr = feat.get("mpr_zscore", 0)
        vol_ratio = feat.get("vol_ratio_5_21", 1.0)
        vol_21d = feat.get("vol_21d", 0.15)

        if any(np.isnan(v) for v in [mtmdi_z, mtmdi_dir, vol_21d]):
            continue

        # Entry conditions
        if abs(mtmdi_z) < cfg.mtmdi_zscore_entry:
            continue
        if mtmdi_dir <= 0:
            continue

        cascade_val = cascade if not np.isnan(cascade) else 0
        mpr_val = mpr if not np.isnan(mpr) else 0

        has_cascade = cascade_val > cfg.cacs_entry_threshold
        has_momentum = mpr_val > cfg.mpr_threshold

        if not (has_cascade or has_momentum):
            continue

        # Signal strength
        strength = (
            min(abs(mtmdi_z) / 3.0, 1.0) * 0.5 +
            min(abs(cascade_val) / 0.05, 1.0) * 0.3 +
            min(max(mpr_val, 0) / 2.0, 1.0) * 0.2
        )

        regime = detect_regime(vol_ratio, vol_21d)
        signals.append((ticker, strength, vol_21d, regime))

    signals.sort(key=lambda s: s[1], reverse=True)
    return signals


def run_backtest(data_dict, start_date, end_date, cfg=None):
    """Run the full backtest. Returns (trades_df, daily_returns)."""
    if cfg is None:
        cfg = Config()

    # Get market benchmark
    market_close = data_dict.get("SPY", {}).get("Close") if isinstance(data_dict.get("SPY"), pd.DataFrame) else None
    if market_close is None and "SPY" in data_dict:
        market_close = data_dict["SPY"]["Close"]

    # Pre-compute features
    features_cache = {}
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            continue
        try:
            volume = df.get("Volume")
            features_cache[ticker] = compute_features(
                df["Close"], volume, market_close
            )
        except Exception:
            pass

    # Get trading dates from SPY
    market_df = data_dict["SPY"]
    dates = market_df.loc[start_date:end_date].index

    positions = {}  # ticker -> {entry_date, entry_price, direction, size, days_held}
    closed_trades = []
    daily_returns = []
    tc = TRANSACTION_COST_BPS / 10000
    pending_signals = []  # Signals awaiting next-day open execution

    for date in dates:
        # Get today's open and close prices
        open_prices = {}
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                if "Open" in df.columns:
                    open_prices[ticker] = df.loc[date, "Open"]
                if "Close" in df.columns:
                    prices[ticker] = df.loc[date, "Close"]

        # === Execute pending signals from yesterday at today's open ===
        total_exposure = sum(p["size"] for p in positions.values())
        for ticker, strength, vol_21d, regime, signal_close in pending_signals:
            if total_exposure >= cfg.max_total_exposure:
                break
            if ticker in positions:
                continue
            open_price = open_prices.get(ticker)
            if open_price is None or np.isnan(open_price):
                continue
            # Gap-down protection: skip if open gaps past stop loss level
            if signal_close > 0:
                gap_pct = (open_price / signal_close) - 1
                if gap_pct <= cfg.stop_loss:
                    continue
            size = compute_position_size(strength, vol_21d, regime, cfg)
            remaining = cfg.max_total_exposure - total_exposure
            size = min(size, remaining)
            if size < 0.005:
                continue
            positions[ticker] = {
                "entry_date": date,
                "entry_price": open_price,  # Enter at next-day open
                "direction": 1,
                "size": size,
                "strength": strength,
                "days_held": 0,
            }
            total_exposure += size
        pending_signals = []

        # Update positions (check exits at close)
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1
            price = prices.get(ticker)
            if price is None or np.isnan(price):
                continue

            pnl = (price / pos["entry_price"] - 1) * pos["direction"]
            should_exit = False
            exit_reason = ""

            if pnl <= cfg.stop_loss:
                should_exit, exit_reason = True, "stop_loss"
            elif pnl >= cfg.take_profit:
                should_exit, exit_reason = True, "take_profit"
            elif pos["days_held"] >= cfg.max_hold_days:
                should_exit, exit_reason = True, "max_hold"
            else:
                feat = features_cache.get(ticker)
                if feat is not None and date in feat.index:
                    mz = feat.loc[date].get("mtmdi_zscore", 0)
                    if not np.isnan(mz) and abs(mz) < cfg.mtmdi_zscore_exit:
                        should_exit, exit_reason = True, "mtmdi_resolved"

            if should_exit:
                net_pnl = pnl - 2 * tc
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "direction": pos["direction"],
                    "size": pos["size"],
                    "gross_pnl": pnl,
                    "net_pnl": net_pnl,
                    "days_held": pos["days_held"],
                    "exit_reason": exit_reason,
                })
                del positions[ticker]

        # Get features for today
        features_dict = {}
        for ticker, feats in features_cache.items():
            if date in feats.index:
                features_dict[ticker] = feats.loc[date].to_dict()

        # Generate signals (pending for next-day open execution)
        signals = generate_signals(date, features_dict, positions, cfg)
        pending_signals = [
            (ticker, strength, vol_21d, regime, prices.get(ticker, 0))
            for ticker, strength, vol_21d, regime in signals
        ]

        # Compute daily portfolio return
        daily_ret = 0.0
        for ticker, pos in positions.items():
            if pos["entry_date"] == date:
                # Entry day: return from open (entry price) to close
                close_price = prices.get(ticker)
                if close_price and pos["entry_price"] > 0:
                    stock_ret = (close_price / pos["entry_price"] - 1) * pos["direction"]
                    daily_ret += stock_ret * pos["size"]
            elif ticker in data_dict:
                df = data_dict[ticker]
                if date in df.index:
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev = df.iloc[idx - 1]["Close"]
                        curr = df.iloc[idx]["Close"]
                        stock_ret = (curr / prev - 1) * pos["direction"]
                        daily_ret += stock_ret * pos["size"]
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
    print(f"\nConfig: SL={cfg.stop_loss}, TP={cfg.take_profit}, "
          f"MaxHold={cfg.max_hold_days}")
    print(f"        MTMDI_entry={cfg.mtmdi_zscore_entry}, "
          f"CACS={cfg.cacs_entry_threshold}, MPR={cfg.mpr_threshold}")

    print(f"\nRunning backtest: {TRAIN_START} to {TRAIN_END}")
    trades, daily_rets = run_backtest(data, TRAIN_START, TRAIN_END, cfg)

    print(f"\nResults on training period:")
    metrics = evaluate_strategy(trades, daily_rets, "Training")

    # Also run on validation (for comparison, but NOT used for optimization)
    print(f"\nRunning validation: {VALID_START} to {VALID_END}")
    v_trades, v_rets = run_backtest(data, VALID_START, VALID_END, cfg)
    print(f"\nResults on validation period:")
    v_metrics = evaluate_strategy(v_trades, v_rets, "Validation")
