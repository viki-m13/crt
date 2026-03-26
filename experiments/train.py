#!/usr/bin/env python3
"""
train.py — Simplified momentum-pullback strategy with progressive trailing.
=============================================================================
Designed so backtest matches live execution exactly.

Key insight: cast a wide net for entries (many small positions), then let
progressive trailing stops do the heavy lifting. Losers get cut, winners
get locked in. With 30-50 concurrent positions, portfolio-level consistency
is extremely high even if per-trade edge is modest.

Design principles:
1. Broad entry filter: any stock in uptrend with a pullback
2. Many small positions (1.5% each, up to 90%)
3. Progressive trailing stop ladder: gradually lock in gains
4. Market regime filter: skip corrections
5. Entry slippage: 10 bps to match real fills
6. No complex derived features — only raw momentum, vol, drawdown

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
    # --- Market regime filter (only trade in clear uptrends) ---
    market_ret_63d_min: float = 0.01
    market_pos_range_min: float = 0.55

    # --- Entry filters (deliberately broad — trailing stops do the filtering) ---
    min_ret_126d: float = 0.05      # Positive 6-month momentum
    min_ret_63d: float = 0.01       # Slightly positive quarter
    max_ret_5d: float = 0.005       # Recent pause/pullback
    min_pos_range: float = 0.50     # Upper half of 52w range
    min_vol_21d: float = 0.06       # Not dead
    max_vol_21d: float = 0.32       # Not exploding
    max_drawdown_252d: float = -0.10 # Not in deep drawdown

    # --- Hard exits ---
    stop_loss: float = -0.015       # 1.5% hard stop (very tight — trailing does the rest)
    take_profit: float = 0.14       # 14% target
    max_hold_days: int = 25         # 5 weeks
    # Percentage trailing: once up trail_min_gain, keep trail_keep_pct of gains
    trail_min_gain: float = 0.012   # Start trailing after +1.2% gain
    trail_keep_pct: float = 0.50    # Keep 50% of peak gain

    # --- Position sizing ---
    position_size: float = 0.015    # 1.5% per position
    max_total_exposure: float = 0.90

    # --- Execution ---
    slippage_bps: float = 10


EXCLUDED_TICKERS = {"SPY", "VIX"}

# No more fixed ladder — using percentage-based trailing instead
# When max_pnl >= trail_min_gain, stop at max_pnl * trail_keep_pct
# Example: peak +4% -> stop at +2% (keep 50%)


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


def get_trail_stop(max_pnl, cfg):
    """Percentage trailing: keep trail_keep_pct of peak gain."""
    if max_pnl >= cfg.trail_min_gain:
        return max_pnl * cfg.trail_keep_pct
    return None


def generate_signals(date, features_dict, positions, cfg):
    """
    Broad momentum + pullback signal.
    Intentionally cast a wide net — the trailing stops do the filtering.
    """
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

        # Ranking: prefer stronger momentum with deeper pullback
        strength = (
            min(ret_126d / 0.25, 1.0) * 0.4 +
            min(ret_63d / 0.10, 1.0) * 0.2 +
            min(max(-ret_5d, 0) / 0.03, 1.0) * 0.2 +
            min(pos_range, 1.0) * 0.2
        )
        signals.append((ticker, strength, vol_21d))

    signals.sort(key=lambda s: s[1], reverse=True)
    return signals


def run_backtest(data_dict, start_date, end_date, cfg=None):
    """Run backtest with progressive trailing stops and slippage."""
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
    slippage = cfg.slippage_bps / 10000
    pending_signals = []

    for date in dates:
        open_prices = {}
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                if "Open" in df.columns:
                    open_prices[ticker] = df.loc[date, "Open"]
                if "Close" in df.columns:
                    prices[ticker] = df.loc[date, "Close"]

        # Execute pending signals at open
        total_exposure = sum(p["size"] for p in positions.values())
        for ticker, strength, vol_21d, signal_close in pending_signals:
            if total_exposure >= cfg.max_total_exposure:
                break
            if ticker in positions:
                continue
            op = open_prices.get(ticker)
            if op is None or np.isnan(op):
                continue
            if signal_close > 0:
                gap = (op / signal_close) - 1
                if gap <= cfg.stop_loss:
                    continue
            entry_price = op * (1 + slippage)
            size = min(cfg.position_size, cfg.max_total_exposure - total_exposure)
            if size < 0.005:
                continue
            positions[ticker] = {
                "entry_date": date,
                "entry_price": entry_price,
                "direction": 1,
                "size": size,
                "days_held": 0,
                "max_pnl": 0.0,
            }
            total_exposure += size
        pending_signals = []

        # Check exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1
            price = prices.get(ticker)
            if price is None or np.isnan(price):
                continue

            pnl = (price / pos["entry_price"] - 1)
            pos["max_pnl"] = max(pos["max_pnl"], pnl)

            should_exit = False
            exit_reason = ""

            if pnl <= cfg.stop_loss:
                should_exit, exit_reason = True, "stop_loss"
            elif pnl >= cfg.take_profit:
                should_exit, exit_reason = True, "take_profit"
            else:
                trail = get_trail_stop(pos["max_pnl"], cfg)
                if trail is not None and pnl <= trail:
                    should_exit, exit_reason = True, "trailing_stop"
                elif pos["days_held"] >= cfg.max_hold_days:
                    should_exit, exit_reason = True, "max_hold"

            if should_exit:
                net_pnl = pnl - 2 * tc
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "direction": 1,
                    "size": pos["size"],
                    "gross_pnl": pnl,
                    "net_pnl": net_pnl,
                    "days_held": pos["days_held"],
                    "exit_reason": exit_reason,
                })
                del positions[ticker]

        # Generate signals
        features_dict = {}
        for ticker, feats in features_cache.items():
            if date in feats.index:
                features_dict[ticker] = feats.loc[date].to_dict()

        if check_market_regime(features_dict, cfg):
            signals = generate_signals(date, features_dict, positions, cfg)
        else:
            signals = []

        pending_signals = [
            (t, s, v, prices.get(t, 0)) for t, s, v in signals
        ]

        # Daily return
        daily_ret = 0.0
        for ticker, pos in positions.items():
            if pos["entry_date"] == date:
                cp = prices.get(ticker)
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
    print(f"\nMomentum-Pullback + Progressive Trail (v4)")
    print(f"  Market:  SPY ret_63d>{cfg.market_ret_63d_min}, range>{cfg.market_pos_range_min}")
    print(f"  Entry:   ret_126d>{cfg.min_ret_126d}, ret_63d>{cfg.min_ret_63d}, "
          f"ret_5d<{cfg.max_ret_5d}")
    print(f"  Exit:    SL={cfg.stop_loss}, TP={cfg.take_profit}, MaxHold={cfg.max_hold_days}")
    print(f"  Trail:   after +{cfg.trail_min_gain:.1%}, keep {cfg.trail_keep_pct:.0%} of peak")
    print(f"  Size:    {cfg.position_size} per pos, {cfg.max_total_exposure} max")
    print(f"  Costs:   {TRANSACTION_COST_BPS}bps + {cfg.slippage_bps}bps slippage")

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

        if period_name == "TRAINING":
            trades, daily_rets, metrics = t, r, m
        elif period_name == "VALIDATION":
            v_trades, v_rets, v_metrics = t, r, m
        else:
            t_trades, t_rets, t_metrics = t, r, m

    print(f"\n{'='*60}")
    print(f"CROSS-PERIOD CONSISTENCY")
    print(f"{'='*60}")
    print(f"{'Period':<12} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>8}")
    print(f"-" * 62)
    for name, m in [("Train", metrics), ("Valid", v_metrics), ("Test", t_metrics)]:
        print(f"{name:<12} {m['sharpe']:>8.3f} {m['cagr']:>7.1%} "
              f"{m['max_drawdown']:>7.1%} {m['win_rate']:>7.1%} {m['profit_factor']:>5.2f} {m['n_trades']:>8}")

    for name, tdf in [("Train", trades), ("Valid", v_trades), ("Test", t_trades)]:
        if len(tdf) > 0:
            print(f"\n{name} exits:")
            for reason, cnt in tdf["exit_reason"].value_counts().items():
                pnl = tdf.loc[tdf["exit_reason"] == reason, "net_pnl"].mean()
                print(f"  {reason}: {cnt} ({cnt/len(tdf):.0%}) avg_pnl={pnl:.3f}")
