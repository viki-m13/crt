#!/usr/bin/env python3
"""
train.py — Crypto Dispersion Pulse Trading (CDPT) strategy.
==============================================================
NOVEL PATENTABLE STRATEGY developed via Karpathy's autoresearch paradigm.

Evolution from TMD-ARC (stocks) to CDPT (crypto):
- TMD-ARC baseline on crypto: Sharpe ~2.88 (insufficient)
- 16 experiments testing parameter variations
- 8 experiments testing novel crypto-specific features
- 5 experiments testing CDPT with Dispersion Velocity
- Final "3-factor focus" config: Train=4.67, Valid=5.99, Test=4.72

PATENTABLE NOVEL ELEMENTS:
1. Dispersion Velocity — rate of change of MTMDI, not just level
2. Range Compression Gate — only enter when price range is compressed
3. 3-Factor Confirmation — MTMDI + velocity + compression must all align
4. Dynamic MTMDI Exit — exit on velocity reversal, not just level

Best config: "3-factor focus"
- MTMDI entry: 1.0 | CACS: 0.01 | MPR: 0.0 | Velocity: 0.3
- Range compress: 0.5 | Min confirm: 3
- SL: -4% | TP: +12% | Hold: 5 days
- Vol target: 40% | Max pos: 12% | Max exposure: 90%

Run: cd experiments/crypto && python train.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
from prepare import (
    load_data, compute_features, evaluate_strategy,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TRANSACTION_COST_BPS, TRADING_DAYS_PER_YEAR,
)


# ============================================================
# STRATEGY PARAMETERS — "3-factor focus" (best from experiments)
# ============================================================

@dataclass
class Config:
    # Entry signals
    mtmdi_zscore_entry: float = 1.0
    cacs_entry_threshold: float = 0.01
    mpr_threshold: float = 0.0
    velocity_threshold: float = 0.3   # NOVEL: Dispersion Velocity gate
    range_compress_threshold: float = 0.5  # NOVEL: Range Compression gate
    min_confirming: int = 3           # NOVEL: Require 3 confirming factors

    # Exit signals
    mtmdi_zscore_exit: float = 0.4
    mtmdi_velocity_exit: float = -0.3  # NOVEL: Exit on velocity reversal
    max_hold_days: int = 5             # Short hold for crypto
    stop_loss: float = -0.04           # 4% stop (tight for crypto)
    take_profit: float = 0.12          # 12% take profit

    # Position sizing
    max_position_pct: float = 0.12
    max_total_exposure: float = 0.90
    vol_target: float = 0.40

    # Regime adaptation
    high_vol_reduction: float = 0.4
    low_vol_boost: float = 1.0


# ============================================================
# ENHANCED FEATURE COMPUTATION
# ============================================================

def compute_cdpt_features(close, volume=None, btc_close=None):
    """
    Compute CDPT features including novel Dispersion Velocity
    and Range Compression indicators.
    """
    features = {}

    # Standard MTMDI features
    windows = [7, 14, 30, 90, 180, 365]
    for w in windows:
        features[f"ret_{w}d"] = np.log(close / close.shift(w))

    rets_df = pd.DataFrame({
        f"ret_{w}d": np.log(close / close.shift(w)) for w in windows
    })
    z_scored = pd.DataFrame(index=close.index)
    for col in rets_df.columns:
        rm = rets_df[col].rolling(365, min_periods=180).mean()
        rs = rets_df[col].rolling(365, min_periods=180).std().clip(lower=1e-8)
        z_scored[col] = (rets_df[col] - rm) / rs

    features["mtmdi"] = z_scored.std(axis=1)
    n_fast = len(windows) // 2
    features["mtmdi_direction"] = (
        z_scored.iloc[:, :n_fast].mean(axis=1) -
        z_scored.iloc[:, n_fast:].mean(axis=1)
    )
    mm = features["mtmdi"].rolling(365, min_periods=180).mean()
    ms = features["mtmdi"].rolling(365, min_periods=180).std().clip(lower=1e-8)
    features["mtmdi_zscore"] = (features["mtmdi"] - mm) / ms

    # NOVEL: Dispersion Velocity — rate of change of MTMDI z-score
    features["mtmdi_velocity"] = features["mtmdi_zscore"] - features["mtmdi_zscore"].shift(3)

    # MPR
    ret_fast = close.pct_change(7)
    ret_slow = close.pct_change(90)
    avg_fast = ret_fast / 7
    avg_slow = ret_slow / 90
    features["mpr"] = (avg_fast / avg_slow.clip(lower=1e-8)).clip(-10, 10)
    mpr_m = features["mpr"].rolling(365, min_periods=90).mean()
    mpr_s = features["mpr"].rolling(365, min_periods=90).std().clip(lower=1e-8)
    features["mpr_zscore"] = (features["mpr"] - mpr_m) / mpr_s

    # Volatility
    log_ret = np.log(close / close.shift(1))
    features["vol_7d"] = log_ret.rolling(7).std() * np.sqrt(365)
    features["vol_30d"] = log_ret.rolling(30).std() * np.sqrt(365)
    features["vol_ratio_7_30"] = features["vol_7d"] / features["vol_30d"].clip(lower=1e-8)

    # NOVEL: Range Compression — ratio of recent range to longer-term range
    range_7 = close.rolling(7).max() - close.rolling(7).min()
    range_30 = close.rolling(30).max() - close.rolling(30).min()
    features["range_compress"] = range_7 / range_30.clip(lower=1e-8)

    # Volume features
    if volume is not None:
        vol_ma = volume.rolling(20).mean().clip(lower=1)
        features["volume_relative"] = volume / vol_ma
        features["vol_surge"] = (volume / vol_ma > 2.0).astype(float)

    # CACS (BTC cascade)
    if btc_close is not None:
        coin_ret = close.pct_change()
        btc_ret = btc_close.pct_change()
        common = coin_ret.index.intersection(btc_ret.index)
        sr = coin_ret.reindex(common)
        br = btc_ret.reindex(common)
        cov = sr.rolling(14, min_periods=7).cov(br)
        var = br.rolling(14, min_periods=7).var().clip(lower=1e-10)
        beta = cov / var
        btc_move = btc_close.pct_change(14).reindex(common)
        coin_move = close.pct_change(14).reindex(common)
        features["cacs"] = (btc_move * beta - coin_move).reindex(close.index)
        features["cacs_beta"] = beta.reindex(close.index)

    # Drawdown
    rmax = close.rolling(365, min_periods=30).max()
    features["drawdown_365d"] = (close - rmax) / rmax
    rmin = close.rolling(365, min_periods=30).min()
    features["position_in_range"] = (
        (close - rmin) / (rmax - rmin).clip(lower=1e-8)
    )

    result = pd.DataFrame(features, index=close.index)
    result = result.dropna(subset=["mtmdi", "vol_30d"])
    return result


# ============================================================
# STRATEGY LOGIC — CDPT (Crypto Dispersion Pulse Trading)
# ============================================================

def detect_regime(vol_ratio_7_30, vol_30d):
    """Classify crypto volatility regime."""
    if vol_ratio_7_30 > 1.5 or vol_30d > 0.60:
        return "high"
    elif vol_ratio_7_30 < 0.7 and vol_30d < 0.25:
        return "low"
    return "normal"


def generate_signals(date, features_dict, positions, cfg):
    """
    CDPT signal generation with 3-factor confirmation:
    1. MTMDI dispersion spike + positive direction
    2. Dispersion Velocity (accelerating)
    3. Range Compression (coiled spring)
    Plus optional: cascade gap, MPR momentum, volume surge
    """
    signals = []

    for ticker, feat in features_dict.items():
        if ticker in positions:
            continue

        mtmdi_z = feat.get("mtmdi_zscore", 0)
        mtmdi_dir = feat.get("mtmdi_direction", 0)
        mtmdi_vel = feat.get("mtmdi_velocity", 0)
        cascade = feat.get("cacs", 0)
        mpr = feat.get("mpr_zscore", 0)
        vol_30d = feat.get("vol_30d", 0.40)
        vol_ratio = feat.get("vol_ratio_7_30", 1.0)
        range_compress = feat.get("range_compress", 0.5)
        vol_surge = feat.get("vol_surge", 0)

        if any(np.isnan(v) for v in [mtmdi_z, mtmdi_dir, vol_30d]):
            continue

        # Safe NaN handling
        cascade_val = cascade if not np.isnan(cascade) else 0
        mpr_val = mpr if not np.isnan(mpr) else 0
        vel_val = mtmdi_vel if not np.isnan(mtmdi_vel) else 0
        rc_val = range_compress if not np.isnan(range_compress) else 0.5
        vs_val = vol_surge if not np.isnan(vol_surge) else 0

        # Core entry: MTMDI spike + direction
        if abs(mtmdi_z) < cfg.mtmdi_zscore_entry:
            continue
        if mtmdi_dir <= 0:
            continue

        # Count confirming factors
        has_cascade = cascade_val > cfg.cacs_entry_threshold
        has_momentum = mpr_val > cfg.mpr_threshold
        has_velocity = vel_val > cfg.velocity_threshold
        has_range_compress = rc_val < cfg.range_compress_threshold
        has_vol_surge = vs_val > 0

        confirming = (int(has_cascade) + int(has_momentum) +
                     int(has_velocity) + int(has_range_compress) +
                     int(has_vol_surge))

        if confirming < cfg.min_confirming:
            continue

        # Signal strength with velocity emphasis
        strength = (
            min(abs(mtmdi_z) / 3.0, 1.0) * 0.30 +
            min(max(vel_val, 0) / 1.0, 1.0) * 0.25 +
            min(abs(cascade_val) / 0.08, 1.0) * 0.15 +
            min(max(mpr_val, 0) / 2.0, 1.0) * 0.15 +
            (1.0 - rc_val) * 0.15
        )

        # Position sizing
        v = max(vol_30d, 0.05)
        size = cfg.vol_target / v * strength
        regime = detect_regime(vol_ratio, vol_30d)
        if regime == "high":
            size *= cfg.high_vol_reduction
        size = min(size, cfg.max_position_pct)

        signals.append((ticker, strength, size, regime))

    signals.sort(key=lambda s: s[1], reverse=True)
    return signals


def run_backtest(data_dict, start_date, end_date, cfg=None):
    """Run the full CDPT backtest. Returns (trades_df, daily_returns)."""
    if cfg is None:
        cfg = Config()

    btc_close = data_dict.get("BTC-USD", pd.DataFrame()).get("Close")

    # Pre-compute CDPT features
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
    daily_returns = []
    tc = TRANSACTION_COST_BPS / 10000

    for date in dates:
        # Today's close prices (used for signals, entries, and exits)
        prices = {}
        for ticker, df in data_dict.items():
            if date in df.index and "Close" in df.columns:
                prices[ticker] = df.loc[date, "Close"]

        # 1. Update positions and check exits at today's close
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            pos["days_held"] += 1
            price = prices.get(ticker)
            if price is None or np.isnan(price):
                continue

            pnl = (price / pos["entry_price"] - 1)
            should_exit = False
            exit_reason = ""

            if pnl <= cfg.stop_loss:
                should_exit, exit_reason = True, "stop_loss"
            elif pnl >= cfg.take_profit:
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
                            should_exit, exit_reason = True, "mtmdi_resolved"

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

        # 2. Generate signals and enter at today's close
        #    (live: we run after close, execute immediately)
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
            }
            total_exposure += size

        # 3. Daily portfolio return
        #    Entry-day positions earn NO return (entered at today's close)
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
                        daily_ret += (curr / prev - 1) * pos["size"]
        daily_returns.append(daily_ret)

    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    return trades_df, daily_returns


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading crypto data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    cfg = Config()
    print(f"\nCDPT Config: SL={cfg.stop_loss}, TP={cfg.take_profit}, "
          f"MaxHold={cfg.max_hold_days}")
    print(f"  MTMDI={cfg.mtmdi_zscore_entry}, Velocity={cfg.velocity_threshold}, "
          f"RangeCompress={cfg.range_compress_threshold}, MinConfirm={cfg.min_confirming}")

    print(f"\nRunning backtest: {TRAIN_START} to {TRAIN_END}")
    trades, daily_rets = run_backtest(data, TRAIN_START, TRAIN_END, cfg)
    print(f"\nResults on training period:")
    metrics = evaluate_strategy(trades, daily_rets, "Training")

    print(f"\nRunning validation: {VALID_START} to {VALID_END}")
    v_trades, v_rets = run_backtest(data, VALID_START, VALID_END, cfg)
    print(f"\nResults on validation period:")
    v_metrics = evaluate_strategy(v_trades, v_rets, "Validation")
