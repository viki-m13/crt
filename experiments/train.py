#!/usr/bin/env python3
"""
train.py — Quiet Breakout with Adaptive Hold (QBAH) Strategy
==============================================================
PATENTABLE NOVEL ELEMENTS:
1. "Quiet Breakout" entry: vol compressed + stock near 52-week high
   (institutional accumulation before next leg — most breakouts are LOUD,
   but quiet ones persist because they haven't attracted momentum chasers yet)
2. NO STOP LOSS ORDERS — eliminates backtest-vs-live gap entirely
3. Adaptive hold via day-5 close checkpoint:
   - If day 5 close > entry → extend hold to day 15 (ride the winner)
   - If day 5 close <= entry → sell at day 6 open (cut the loser)
   This is NOT an intraday stop — it's a daily close check, trivially
   replicated in live by looking at your portfolio at 4pm on day 5.
4. Concentrated portfolio (max 2 positions at 25% each)

EXECUTION MODEL (identical backtest ↔ live):
- Signal: at market close
- Entry: next day OPEN + 10bps slippage
- Day-5 check: compare day-5 CLOSE to entry price. Decision made after hours.
- Loser exit: day 6 OPEN + 10bps slippage (sell at open)
- Winner exit: day 15 CLOSE - 5bps slippage (sell at close)
- No intraday orders of any kind. Just buy, check, sell.

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


@dataclass
class Config:
    # --- Market regime ---
    market_pos_range_min: float = 0.50

    # --- "Quiet Breakout" entry ---
    max_vol_ratio_5_21: float = 0.65     # Vol compressed (5d < 65% of 21d)
    min_pos_range: float = 0.85          # Near 52-week high (breakout zone)
    min_ret_63d: float = 0.04            # Strong quarterly trend
    min_ret_5d: float = -0.01            # Not pulling back sharply
    min_vol_21d: float = 0.10            # Enough base vol
    max_vol_21d: float = 0.30            # Not chaotic
    max_drawdown_252d: float = -0.05     # Near highs = minimal drawdown

    # --- Adaptive hold (NO stops, NO intraday orders) ---
    checkpoint_day: int = 5              # Check at close of day 5
    winner_hold_days: int = 15           # Winners held to day 15
    gap_down_skip: float = -0.04         # Skip entry if open gaps > 4%

    # --- Position sizing ---
    max_positions: int = 2
    position_size: float = 0.25          # 25% per position

    # --- Execution ---
    entry_slippage_bps: float = 10
    exit_slippage_bps: float = 5         # MOC exit
    loser_exit_slippage_bps: float = 10  # Sell at open (worse fill)


EXCLUDED_TICKERS = {"SPY", "VIX", "TLT", "IEF", "HYG", "GLD", "SLV", "USO"}


def check_market_regime(features_dict, cfg):
    spy = features_dict.get("SPY")
    if spy is None:
        return False
    p = spy.get("position_in_52w_range", np.nan)
    return not np.isnan(p) and p >= cfg.market_pos_range_min


def generate_signals(date, features_dict, positions, cfg):
    """
    Quiet Breakout: vol compressed + near 52-week high.
    The rarest and highest-conviction setup: the stock is coiling
    at the top of its range. Institutional accumulation without fanfare.
    """
    signals = []

    for ticker, feat in features_dict.items():
        if ticker in EXCLUDED_TICKERS or ticker in positions:
            continue

        vol_ratio = feat.get("vol_ratio_5_21", np.nan)
        pos_range = feat.get("position_in_52w_range", np.nan)
        ret_63d = feat.get("ret_63d", np.nan)
        ret_5d = feat.get("ret_5d", np.nan)
        vol_21d = feat.get("vol_21d", np.nan)
        dd_252d = feat.get("drawdown_252d", np.nan)

        vals = [vol_ratio, pos_range, ret_63d, ret_5d, vol_21d, dd_252d]
        if any(np.isnan(v) for v in vals):
            continue

        if vol_ratio >= cfg.max_vol_ratio_5_21:
            continue
        if pos_range < cfg.min_pos_range:
            continue
        if ret_63d < cfg.min_ret_63d:
            continue
        if ret_5d < cfg.min_ret_5d:
            continue
        if vol_21d < cfg.min_vol_21d or vol_21d > cfg.max_vol_21d:
            continue
        if dd_252d < cfg.max_drawdown_252d:
            continue

        score = (1.0 - vol_ratio) * pos_range * (1 + ret_63d)
        signals.append((ticker, score))

    signals.sort(key=lambda s: s[1], reverse=True)
    return signals


def run_backtest(data_dict, start_date, end_date, cfg=None):
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
    loser_exit_slip = cfg.loser_exit_slippage_bps / 10000
    pending_signals = []
    # Positions flagged for loser exit at next open
    pending_loser_exits = []

    for i, date in enumerate(dates):
        open_prices = {}
        close_prices = {}
        for ticker, df in data_dict.items():
            if date in df.index:
                if "Open" in df.columns:
                    open_prices[ticker] = df.loc[date, "Open"]
                if "Close" in df.columns:
                    close_prices[ticker] = df.loc[date, "Close"]

        # === EXIT LOSERS AT OPEN (from yesterday's day-5 check) ===
        for ticker in pending_loser_exits:
            if ticker not in positions:
                continue
            pos = positions[ticker]
            op = open_prices.get(ticker)
            if op is None or np.isnan(op):
                continue
            exit_price = op * (1 - loser_exit_slip)
            pnl = (exit_price / pos["entry_price"]) - 1
            net_pnl = pnl - 2 * tc
            closed_trades.append({
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": date,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "size": pos["size"],
                "gross_pnl": pnl,
                "net_pnl": net_pnl,
                "days_held": pos["days_held"],
                "exit_reason": "loser_cut",
            })
            del positions[ticker]
        pending_loser_exits = []

        # === EXECUTE PENDING ENTRIES AT OPEN ===
        for ticker, score, signal_close in pending_signals:
            if len(positions) >= cfg.max_positions:
                break
            if ticker in positions:
                continue
            op = open_prices.get(ticker)
            if op is None or np.isnan(op):
                continue
            if signal_close > 0:
                gap = (op / signal_close) - 1
                if gap <= cfg.gap_down_skip:
                    continue
            entry_price = op * (1 + entry_slip)
            positions[ticker] = {
                "entry_date": date,
                "entry_price": entry_price,
                "size": cfg.position_size,
                "days_held": 0,
                "extended": False,
            }
        pending_signals = []

        # === UPDATE POSITIONS ===
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1

            cp = close_prices.get(ticker)
            if cp is None or np.isnan(cp):
                continue

            # --- Day-5 checkpoint: decide at close ---
            if pos["days_held"] == cfg.checkpoint_day and not pos["extended"]:
                pnl_at_close = (cp / pos["entry_price"]) - 1
                if pnl_at_close > 0:
                    # Winner: extend to day 15
                    pos["extended"] = True
                else:
                    # Loser: flag for exit at NEXT day's open
                    pending_loser_exits.append(ticker)

            # --- Winner exit at day 15 close ---
            if pos["days_held"] >= cfg.winner_hold_days and pos["extended"]:
                exit_price = cp * (1 - exit_slip)
                pnl = (exit_price / pos["entry_price"]) - 1
                net_pnl = pnl - 2 * tc
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "size": pos["size"],
                    "gross_pnl": pnl,
                    "net_pnl": net_pnl,
                    "days_held": pos["days_held"],
                    "exit_reason": "winner_exit",
                })
                del positions[ticker]

            # --- Backstop: if not extended and past checkpoint+1, exit ---
            # (handles edge case where loser exit didn't execute)
            elif pos["days_held"] > cfg.checkpoint_day + 2 and not pos["extended"]:
                exit_price = cp * (1 - exit_slip)
                pnl = (exit_price / pos["entry_price"]) - 1
                net_pnl = pnl - 2 * tc
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "size": pos["size"],
                    "gross_pnl": pnl,
                    "net_pnl": net_pnl,
                    "days_held": pos["days_held"],
                    "exit_reason": "loser_backstop",
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

        n_open = cfg.max_positions - len(positions)
        if n_open > 0 and signals:
            pending_signals = [
                (t, s, close_prices.get(t, 0))
                for t, s in signals[:n_open]
            ]

        # === DAILY RETURN ===
        daily_ret = 0.0
        for ticker, pos in positions.items():
            cp = close_prices.get(ticker)
            if cp is None or np.isnan(cp):
                continue
            if pos["entry_date"] == date:
                daily_ret += (cp / pos["entry_price"] - 1) * pos["size"]
            elif ticker in data_dict:
                df = data_dict[ticker]
                if date in df.index:
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev = df.iloc[idx - 1]["Close"]
                        daily_ret += (cp / prev - 1) * pos["size"]

        # Closed positions today
        for trade in closed_trades:
            if trade["exit_date"] == date:
                tk = trade["ticker"]
                if trade["entry_date"] == date:
                    daily_ret += (trade["exit_price"] / trade["entry_price"] - 1) * trade["size"]
                else:
                    df = data_dict.get(tk)
                    if df is not None and date in df.index:
                        idx = df.index.get_loc(date)
                        if idx > 0:
                            prev = df.iloc[idx - 1]["Close"]
                            daily_ret += (trade["exit_price"] / prev - 1) * trade["size"]

        daily_returns.append(daily_ret)

    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    return trades_df, daily_returns


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    cfg = Config()
    print(f"\nQuiet Breakout + Adaptive Hold (QBAH)")
    print(f"  NOVEL: No stops. Day-{cfg.checkpoint_day} checkpoint. Backtest = Live.")
    print(f"  Market:  SPY pos_range > {cfg.market_pos_range_min}")
    print(f"  Entry:   vol_ratio < {cfg.max_vol_ratio_5_21}, pos_range > {cfg.min_pos_range}")
    print(f"           ret_63d > {cfg.min_ret_63d}, ret_5d > {cfg.min_ret_5d}")
    print(f"  Exit:    Day {cfg.checkpoint_day}: losers cut at next open. "
          f"Winners held to day {cfg.winner_hold_days}.")
    print(f"  Size:    {cfg.position_size:.0%} per stock, max {cfg.max_positions}")
    print(f"  Costs:   {TRANSACTION_COST_BPS}bps + {cfg.entry_slippage_bps}bps/{cfg.exit_slippage_bps}bps slip")

    all_m = {}
    all_t = {}
    for pname, s, e in [
        ("TRAINING", TRAIN_START, TRAIN_END),
        ("VALIDATION", VALID_START, VALID_END),
        ("TEST (OOS)", TEST_START, TEST_END),
    ]:
        print(f"\n{'='*60}")
        print(f"{pname}: {s} to {e}")
        print(f"{'='*60}")
        t, r = run_backtest(data, s, e, cfg)
        m = evaluate_strategy(t, r, pname)
        all_m[pname] = m
        all_t[pname] = t

    print(f"\n{'='*60}")
    print(f"CROSS-PERIOD CONSISTENCY")
    print(f"{'='*60}")
    print(f"{'Period':<12} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>8}")
    print("-" * 62)
    for n, k in [("Train", "TRAINING"), ("Valid", "VALIDATION"), ("Test", "TEST (OOS)")]:
        m = all_m[k]
        print(f"{n:<12} {m['sharpe']:>8.3f} {m['cagr']:>7.1%} "
              f"{m['max_drawdown']:>7.1%} {m['win_rate']:>7.1%} {m['profit_factor']:>5.2f} {m['n_trades']:>8}")

    for n, k in [("Train", "TRAINING"), ("Valid", "VALIDATION"), ("Test", "TEST (OOS)")]:
        tdf = all_t[k]
        if len(tdf) > 0:
            print(f"\n{n} exit breakdown:")
            for reason, cnt in tdf["exit_reason"].value_counts().items():
                avg = tdf.loc[tdf["exit_reason"] == reason, "net_pnl"].mean()
                wr = (tdf.loc[tdf["exit_reason"] == reason, "net_pnl"] > 0).mean()
                print(f"  {reason}: {cnt} ({cnt/len(tdf):.0%}), avg_pnl={avg:.3f}, win_rate={wr:.0%}")
