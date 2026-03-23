#!/usr/bin/env python3
"""
prepare.py — Fixed constants, data prep, and evaluation.
==========================================================
DO NOT MODIFY THIS FILE. This is the equivalent of autoresearch's prepare.py.

This file contains:
1. Fixed data split constants
2. Data download and caching
3. Feature computation
4. Strategy evaluation harness

The agent modifies train.py, not this file.
"""

import os
import sys
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# CONSTANTS (DO NOT MODIFY)
# ============================================================

# Data splits with buffer zones to prevent leakage
TRAIN_START = "2010-01-01"
TRAIN_END = "2019-12-31"
VALID_START = "2020-04-01"
VALID_END = "2022-12-31"
TEST_START = "2023-04-01"
TEST_END = "2026-03-15"

# Quality filters
MIN_HISTORY_DAYS = 1000
MIN_MEDIAN_DOLLAR_VOLUME = 5_000_000

# Transaction costs (per trade, one way)
TRANSACTION_COST_BPS = 10

# Evaluation
RISK_FREE_RATE = 0.02

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Universe: ETFs + top 100 liquid stocks
UNIVERSE = [
    # ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "XLB", "XLRE", "XLC",
    "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
    # Stocks
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM",
    "CSCO", "VZ", "ADBE", "CRM", "CMCSA", "PFE", "NFLX", "INTC",
    "ABT", "KO", "PEP", "TMO", "MRK", "ABBV", "COST", "AVGO", "ACN",
    "CVX", "LLY", "MCD", "WMT", "DHR", "TXN", "NEE", "BMY", "QCOM",
    "UNP", "HON", "LOW", "AMGN", "LIN", "RTX",
    "ORCL", "PM", "UPS", "CAT", "GS", "MS", "BLK", "ISRG", "MDT",
    "DE", "ADP", "GILD", "BKNG", "SYK", "MMM", "GE", "CB", "CI",
    "SO", "DUK", "MO", "CL", "ITW", "FIS", "USB", "SCHW", "PNC",
    "CME", "AON", "ICE", "NSC", "EMR", "APD", "SHW", "ETN", "ECL",
    "WM", "ROP", "LRCX", "KLAC", "AMAT", "MCHP", "SNPS", "CDNS",
    "FTNT", "PANW", "NOW", "WDAY",
]


# ============================================================
# DATA LOADING
# ============================================================

def download_data(force=False):
    """Download OHLCV data from Yahoo Finance, cache as CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "download_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    results = {}
    today = datetime.date.today().isoformat()

    for ticker in UNIVERSE:
        cache_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not force and os.path.exists(cache_path):
            if manifest.get(ticker, {}).get("date") == today:
                try:
                    results[ticker] = pd.read_csv(
                        cache_path, index_col=0, parse_dates=True
                    )
                    continue
                except Exception:
                    pass

        try:
            print(f"  Downloading {ticker}...")
            data = yf.download(
                ticker, start="2008-01-01", end=today,
                progress=False, auto_adjust=True
            )
            if data is None or len(data) < 100:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.to_csv(cache_path)
            manifest[ticker] = {"date": today, "rows": len(data)}
            results[ticker] = data
        except Exception as e:
            print(f"  ERROR: {ticker}: {e}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Loaded {len(results)} tickers.")
    return results


def load_data():
    """Load cached data or download if needed."""
    # Try to load from cache first
    if os.path.exists(DATA_DIR):
        results = {}
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                ticker = f.replace(".csv", "")
                try:
                    results[ticker] = pd.read_csv(
                        os.path.join(DATA_DIR, f),
                        index_col=0, parse_dates=True
                    )
                except Exception:
                    pass
        if len(results) > 50:
            return results

    return download_data()


# ============================================================
# FEATURE COMPUTATION
# ============================================================

MOMENTUM_WINDOWS = [5, 10, 21, 63, 126, 252]


def compute_features(close, volume=None, market_close=None):
    """
    Compute the full feature set for one stock.
    All features use ONLY past data (no lookahead).
    """
    features = {}

    # --- Momentum returns ---
    for w in MOMENTUM_WINDOWS:
        features[f"ret_{w}d"] = np.log(close / close.shift(w))

    # --- MTMDI (Multi-Timeframe Momentum Dispersion Index) ---
    rets_df = pd.DataFrame({
        f"ret_{w}d": np.log(close / close.shift(w)) for w in MOMENTUM_WINDOWS
    })
    z_scored = pd.DataFrame(index=close.index)
    for col in rets_df.columns:
        rm = rets_df[col].rolling(252, min_periods=126).mean()
        rs = rets_df[col].rolling(252, min_periods=126).std().clip(lower=1e-8)
        z_scored[col] = (rets_df[col] - rm) / rs

    features["mtmdi"] = z_scored.std(axis=1)
    n_fast = len(MOMENTUM_WINDOWS) // 2
    features["mtmdi_direction"] = (
        z_scored.iloc[:, :n_fast].mean(axis=1) -
        z_scored.iloc[:, n_fast:].mean(axis=1)
    )
    mm = features["mtmdi"].rolling(252, min_periods=126).mean()
    ms = features["mtmdi"].rolling(252, min_periods=126).std().clip(lower=1e-8)
    features["mtmdi_zscore"] = (features["mtmdi"] - mm) / ms

    # --- MPR (Momentum Persistence Ratio) ---
    ret_fast = close.pct_change(5)
    ret_slow = close.pct_change(63)
    avg_fast = ret_fast / 5
    avg_slow = ret_slow / 63
    features["mpr"] = (avg_fast / avg_slow.clip(lower=1e-8)).clip(-10, 10)
    mpr_m = features["mpr"].rolling(252, min_periods=63).mean()
    mpr_s = features["mpr"].rolling(252, min_periods=63).std().clip(lower=1e-8)
    features["mpr_zscore"] = (features["mpr"] - mpr_m) / mpr_s

    # --- Volatility ---
    log_ret = np.log(close / close.shift(1))
    features["vol_5d"] = log_ret.rolling(5).std() * np.sqrt(252)
    features["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
    features["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)
    features["vol_ratio_5_21"] = features["vol_5d"] / features["vol_21d"].clip(lower=1e-8)
    features["vol_ratio_21_63"] = features["vol_21d"] / features["vol_63d"].clip(lower=1e-8)

    # --- Volume ---
    if volume is not None:
        vol_ma20 = volume.rolling(20).mean().clip(lower=1)
        features["volume_relative"] = volume / vol_ma20
        features["volume_trend"] = volume.rolling(5).mean() / vol_ma20

    # --- Drawdown ---
    rmax = close.rolling(252, min_periods=21).max()
    features["drawdown_252d"] = (close - rmax) / rmax
    features["dd_change_5d"] = features["drawdown_252d"] - features["drawdown_252d"].shift(5)
    rmin = close.rolling(252, min_periods=21).min()
    features["position_in_52w_range"] = (
        (close - rmin) / (rmax - rmin).clip(lower=1e-8)
    )

    # --- CACS (Cross-Asset Cascade Score) ---
    if market_close is not None:
        stock_ret = close.pct_change()
        market_ret = market_close.pct_change()
        common = stock_ret.index.intersection(market_ret.index)
        sr = stock_ret.reindex(common)
        mr = market_ret.reindex(common)
        cov = sr.rolling(21, min_periods=10).cov(mr)
        var = mr.rolling(21, min_periods=10).var().clip(lower=1e-10)
        beta = cov / var
        leader_move = market_close.pct_change(21).reindex(common)
        stock_move = close.pct_change(21).reindex(common)
        features["cacs"] = (leader_move * beta - stock_move).reindex(close.index)
        features["cacs_beta"] = beta.reindex(close.index)

    result = pd.DataFrame(features, index=close.index)
    result = result.dropna(subset=["mtmdi", "vol_21d"])
    return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate_strategy(trades_df, daily_returns, period_name=""):
    """
    Evaluate strategy performance. This is the ground truth metric.
    DO NOT MODIFY.
    """
    if len(daily_returns) == 0:
        print("---")
        print("sharpe:          0.000")
        print("cagr:            0.00%")
        print("max_drawdown:    0.00%")
        print("n_trades:        0")
        print("win_rate:        0.00%")
        print("profit_factor:   0.00")
        print("avg_hold_days:   0.0")
        return {"sharpe": 0}

    rets = pd.Series(daily_returns)
    n_years = len(rets) / 252

    # Sharpe
    excess = rets - RISK_FREE_RATE / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # CAGR
    cum = (1 + rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Trade stats
    n_trades = len(trades_df) if trades_df is not None and len(trades_df) > 0 else 0
    win_rate = 0
    profit_factor = 0
    avg_hold = 0

    if n_trades > 0:
        win_rate = (trades_df["net_pnl"] > 0).mean()
        wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].sum()
        losses = abs(trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].sum())
        profit_factor = wins / losses if losses > 0 else (999 if wins > 0 else 0)
        avg_hold = trades_df["days_held"].mean()

    print("---")
    print(f"sharpe:          {sharpe:.3f}")
    print(f"cagr:            {cagr:.2%}")
    print(f"max_drawdown:    {max_dd:.2%}")
    print(f"n_trades:        {n_trades}")
    print(f"win_rate:        {win_rate:.2%}")
    print(f"profit_factor:   {profit_factor:.2f}")
    print(f"avg_hold_days:   {avg_hold:.1f}")

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold,
    }


# ============================================================
# MAIN (data prep only)
# ============================================================

if __name__ == "__main__":
    print("Downloading market data...")
    data = download_data()
    print(f"Done. {len(data)} tickers saved to {DATA_DIR}/")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Valid: {VALID_START} to {VALID_END}")
    print(f"Test:  {TEST_START} to {TEST_END}")
