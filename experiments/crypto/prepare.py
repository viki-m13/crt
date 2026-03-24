#!/usr/bin/env python3
"""
prepare.py — Fixed constants, data prep, and evaluation for Crypto TMD-ARC.
=============================================================================
DO NOT MODIFY THIS FILE. This is the evaluation harness.
The agent modifies train.py, not this file.

Crypto-specific: BTC benchmark, 365-day annualization, crypto universe.
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

TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VALID_START = "2022-04-01"
VALID_END = "2023-06-30"
TEST_START = "2023-10-01"
TEST_END = "2026-03-15"

MIN_HISTORY_DAYS = 800
MIN_MEDIAN_DOLLAR_VOLUME = 1_000_000

TRANSACTION_COST_BPS = 15  # Crypto exchanges charge more
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 365

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Crypto universe
UNIVERSE = [
    # === Top 20 by Market Cap ===
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD",
    "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD",
    "BCH-USD", "XLM-USD", "ALGO-USD", "FIL-USD", "NEAR-USD",
    # === DeFi ===
    "AAVE-USD", "MKR-USD", "SNX-USD", "SUSHI-USD", "YFI-USD",
    "CRV-USD", "BAL-USD", "UMA-USD", "1INCH-USD", "DYDX-USD",
    "LDO-USD", "RPL-USD",
    # === Layer 1 / Layer 2 ===
    "ICP-USD", "FTM-USD", "HBAR-USD", "THETA-USD", "VET-USD",
    "NEO-USD", "EOS-USD", "TRX-USD", "EGLD-USD", "FLOW-USD",
    "MINA-USD", "KAVA-USD", "ONE-USD", "CELO-USD", "KSM-USD",
    "ZIL-USD", "ICX-USD", "QTUM-USD", "WAVES-USD", "XTZ-USD",
    "FET-USD", "ROSE-USD", "AR-USD", "STX-USD",
    "SUI-USD", "APT-USD", "SEI-USD", "INJ-USD", "TIA-USD",
    "OP-USD", "ARB-USD",
    # === Gaming / Metaverse ===
    "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "GALA-USD",
    "ILV-USD", "RNDR-USD",
    # === Privacy ===
    "ZEC-USD", "DASH-USD", "XMR-USD",
    # === Exchange Tokens ===
    "CRO-USD", "OKB-USD", "LEO-USD",
    # === Infrastructure / Oracle ===
    "CHZ-USD", "LRC-USD", "BAT-USD", "ZRX-USD",
    "STORJ-USD", "ANKR-USD", "BAND-USD", "API3-USD",
    # === Meme / Community ===
    "SHIB-USD", "PEPE24478-USD", "FLOKI-USD", "BONK-USD",
    # === Newer Large Caps ===
    "WLD-USD", "JUP-USD", "PENDLE-USD",
    # === AI / Data ===
    "OCEAN-USD", "AGIX-USD",
    # === Additional Liquid Alts ===
    "SKL-USD", "COTI-USD", "RLC-USD", "NMR-USD",
    "REQ-USD", "OMG-USD", "CELR-USD",
    "DENT-USD", "HOT-USD", "WIN-USD", "SC-USD",
    "AUDIO-USD", "PERP-USD", "RUNE-USD", "OSMO-USD",
]


# ============================================================
# DATA LOADING
# ============================================================

def download_data(force=False):
    """Download crypto OHLCV data from Yahoo Finance."""
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "download_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    results = {}
    today = datetime.date.today().isoformat()

    for ticker in UNIVERSE:
        cache_path = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.csv")
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
                ticker, start="2017-01-01", end=today,
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

    print(f"Loaded {len(results)} crypto tickers.")
    return results


def load_data():
    """Load cached data or download if needed."""
    if os.path.exists(DATA_DIR):
        results = {}
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                ticker = f.replace(".csv", "").replace("_", "-")
                try:
                    results[ticker] = pd.read_csv(
                        os.path.join(DATA_DIR, f),
                        index_col=0, parse_dates=True
                    )
                except Exception:
                    pass
        if len(results) > 10:
            return results
    return download_data()


# ============================================================
# FEATURE COMPUTATION
# ============================================================

MOMENTUM_WINDOWS = [7, 14, 30, 90, 180, 365]


def compute_features(close, volume=None, btc_close=None):
    """
    Compute complete feature set for one crypto asset.
    All features use ONLY past data (no lookahead).
    """
    features = {}

    # Momentum returns
    for w in MOMENTUM_WINDOWS:
        features[f"ret_{w}d"] = np.log(close / close.shift(w))

    # MTMDI
    rets_df = pd.DataFrame({
        f"ret_{w}d": np.log(close / close.shift(w)) for w in MOMENTUM_WINDOWS
    })
    z_scored = pd.DataFrame(index=close.index)
    for col in rets_df.columns:
        rm = rets_df[col].rolling(365, min_periods=180).mean()
        rs = rets_df[col].rolling(365, min_periods=180).std().clip(lower=1e-8)
        z_scored[col] = (rets_df[col] - rm) / rs

    features["mtmdi"] = z_scored.std(axis=1)
    n_fast = len(MOMENTUM_WINDOWS) // 2
    features["mtmdi_direction"] = (
        z_scored.iloc[:, :n_fast].mean(axis=1) -
        z_scored.iloc[:, n_fast:].mean(axis=1)
    )
    mm = features["mtmdi"].rolling(365, min_periods=180).mean()
    ms = features["mtmdi"].rolling(365, min_periods=180).std().clip(lower=1e-8)
    features["mtmdi_zscore"] = (features["mtmdi"] - mm) / ms

    # MPR
    ret_fast = close.pct_change(7)
    ret_slow = close.pct_change(90)
    avg_fast = ret_fast / 7
    avg_slow = ret_slow / 90
    features["mpr"] = (avg_fast / avg_slow.clip(lower=1e-8)).clip(-10, 10)
    mpr_m = features["mpr"].rolling(365, min_periods=90).mean()
    mpr_s = features["mpr"].rolling(365, min_periods=90).std().clip(lower=1e-8)
    features["mpr_zscore"] = (features["mpr"] - mpr_m) / mpr_s

    # Volatility (annualized with sqrt(365))
    log_ret = np.log(close / close.shift(1))
    features["vol_7d"] = log_ret.rolling(7).std() * np.sqrt(365)
    features["vol_30d"] = log_ret.rolling(30).std() * np.sqrt(365)
    features["vol_90d"] = log_ret.rolling(90).std() * np.sqrt(365)
    features["vol_ratio_7_30"] = features["vol_7d"] / features["vol_30d"].clip(lower=1e-8)
    features["vol_ratio_30_90"] = features["vol_30d"] / features["vol_90d"].clip(lower=1e-8)

    # Volume
    if volume is not None:
        vol_ma20 = volume.rolling(20).mean().clip(lower=1)
        features["volume_relative"] = volume / vol_ma20
        features["volume_trend"] = volume.rolling(7).mean() / vol_ma20

    # Drawdown
    rmax = close.rolling(365, min_periods=30).max()
    features["drawdown_365d"] = (close - rmax) / rmax
    features["dd_change_7d"] = features["drawdown_365d"] - features["drawdown_365d"].shift(7)
    rmin = close.rolling(365, min_periods=30).min()
    features["position_in_range"] = (
        (close - rmin) / (rmax - rmin).clip(lower=1e-8)
    )

    # CACS (vs BTC)
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

    result = pd.DataFrame(features, index=close.index)
    result = result.dropna(subset=["mtmdi", "vol_30d"])
    return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate_strategy(trades_df, daily_returns, period_name=""):
    """
    Evaluate crypto strategy performance. Ground truth metric.
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
    n_years = len(rets) / TRADING_DAYS_PER_YEAR

    excess = rets - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if excess.std() > 0 else 0

    cum = (1 + rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

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


if __name__ == "__main__":
    print("Downloading crypto market data...")
    data = download_data()
    print(f"Done. {len(data)} tickers saved to {DATA_DIR}/")
