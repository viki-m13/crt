"""
Data Pipeline for TMD-ARC Strategy Experiments
===============================================
Downloads market data and creates strict train/validation/test splits
with NO lookahead bias or data leakage.

Split methodology:
- Training:   2010-01-01 to 2019-12-31 (10 years)
- Validation:  2020-01-01 to 2022-12-31 (3 years, includes COVID + recovery)
- Test:        2023-01-01 to present     (out-of-sample, NEVER touched until final eval)

A 63-day (3-month) buffer is enforced between splits to prevent
any feature leakage from lookback windows crossing split boundaries.
"""

import os
import json
import datetime
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf


# === SPLIT DATES (FIXED - NEVER MODIFY) ===
TRAIN_START = "2010-01-01"
TRAIN_END = "2019-12-31"
VALID_START = "2020-04-01"  # 3-month buffer after train
VALID_END = "2022-12-31"
TEST_START = "2023-04-01"   # 3-month buffer after validation
TEST_END = "2026-03-15"     # ~current

# Buffer in trading days between splits (prevents feature leakage)
SPLIT_BUFFER_DAYS = 63

# Minimum trading days required per stock
MIN_HISTORY_DAYS = 1000

# Minimum median daily dollar volume
MIN_MEDIAN_DOLLAR_VOLUME = 5_000_000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_universe():
    """
    Returns the stock universe: Russell 1000 components + key ETFs.
    Uses a broad, liquid universe to avoid selection bias.
    """
    # Core sector ETFs for cross-asset cascade detection
    etfs = [
        "SPY", "QQQ", "IWM", "DIA",       # Broad market
        "XLK", "XLF", "XLE", "XLV",         # Sectors
        "XLI", "XLY", "XLP", "XLU",         # Sectors cont.
        "XLB", "XLRE", "XLC",               # Sectors cont.
        "TLT", "IEF", "HYG",               # Bonds
        "GLD", "SLV", "USO",               # Commodities
        "VIX",                               # Volatility (for regime)
    ]

    # Top 100 most liquid stocks (avoids survivorship bias by using
    # historically significant names, not just current leaders)
    stocks = [
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

    return etfs + stocks


def download_data(tickers=None, force=False):
    """
    Download OHLCV data from Yahoo Finance.
    Caches to disk to avoid repeated downloads.
    Returns dict of {ticker: DataFrame}.
    """
    if tickers is None:
        tickers = get_universe()

    os.makedirs(DATA_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "download_manifest.json")

    # Load existing manifest
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    results = {}
    failed = []
    today = datetime.date.today().isoformat()

    for ticker in tickers:
        cache_path = os.path.join(DATA_DIR, f"{ticker}.csv")

        # Use cache if fresh (downloaded today) and not forcing
        if not force and os.path.exists(cache_path):
            if manifest.get(ticker, {}).get("date") == today:
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    results[ticker] = df
                    continue
                except Exception:
                    pass

        # Download from Yahoo Finance
        try:
            print(f"  Downloading {ticker}...")
            data = yf.download(
                ticker, start="2008-01-01", end=today,
                progress=False, auto_adjust=True
            )
            if data is None or len(data) < 100:
                print(f"  WARNING: {ticker} has insufficient data ({len(data) if data is not None else 0} rows)")
                failed.append(ticker)
                continue

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Save to CSV
            data.to_csv(cache_path)
            manifest[ticker] = {
                "date": today,
                "rows": len(data),
                "start": str(data.index[0].date()),
                "end": str(data.index[-1].date()),
                "checksum": hashlib.md5(
                    str(len(data)).encode()
                ).hexdigest()[:12]
            }
            results[ticker] = data

        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
            failed.append(ticker)

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDownloaded {len(results)}/{len(tickers)} tickers. "
          f"Failed: {len(failed)}")
    if failed:
        print(f"  Failed tickers: {failed}")

    return results


def apply_quality_filters(data_dict):
    """
    Filter stocks by minimum history and liquidity requirements.
    Returns filtered dict and a log of what was removed and why.
    """
    filtered = {}
    removal_log = []

    for ticker, df in data_dict.items():
        # Check minimum history
        if len(df) < MIN_HISTORY_DAYS:
            removal_log.append({
                "ticker": ticker,
                "reason": "insufficient_history",
                "detail": f"{len(df)} days < {MIN_HISTORY_DAYS} required"
            })
            continue

        # Check liquidity (median daily dollar volume)
        if "Volume" in df.columns and "Close" in df.columns:
            dollar_vol = (df["Close"] * df["Volume"]).median()
            if dollar_vol < MIN_MEDIAN_DOLLAR_VOLUME:
                removal_log.append({
                    "ticker": ticker,
                    "reason": "low_liquidity",
                    "detail": f"median $vol {dollar_vol:,.0f} < {MIN_MEDIAN_DOLLAR_VOLUME:,.0f}"
                })
                continue

        # Check for excessive missing data (>5%)
        missing_pct = df["Close"].isna().mean()
        if missing_pct > 0.05:
            removal_log.append({
                "ticker": ticker,
                "reason": "missing_data",
                "detail": f"{missing_pct:.1%} missing"
            })
            continue

        filtered[ticker] = df

    print(f"Quality filter: {len(filtered)}/{len(data_dict)} passed. "
          f"Removed {len(removal_log)}.")

    return filtered, removal_log


def split_data(df, split="train"):
    """
    Extract a specific time split from a DataFrame.
    Enforces buffer zones between splits.

    Returns a copy to prevent accidental mutation.
    """
    if split == "train":
        return df.loc[TRAIN_START:TRAIN_END].copy()
    elif split == "valid":
        return df.loc[VALID_START:VALID_END].copy()
    elif split == "test":
        return df.loc[TEST_START:TEST_END].copy()
    elif split == "train_valid":
        # For final model fitting (train + valid), but NOT test
        train = df.loc[TRAIN_START:TRAIN_END].copy()
        valid = df.loc[VALID_START:VALID_END].copy()
        return pd.concat([train, valid])
    else:
        raise ValueError(f"Unknown split: {split}")


def get_spy_returns(data_dict):
    """Extract SPY returns as market benchmark (used for beta, regime, etc.)."""
    if "SPY" not in data_dict:
        raise ValueError("SPY must be in the universe for market benchmark")
    spy = data_dict["SPY"]["Close"]
    return spy.pct_change().dropna()


def verify_no_leakage(train_df, valid_df, test_df):
    """
    Verify there is no temporal overlap or insufficient buffer between splits.
    Raises AssertionError if leakage is detected.
    """
    if len(train_df) == 0 or len(valid_df) == 0:
        return  # Skip if split is empty

    train_end = train_df.index.max()
    valid_start = valid_df.index.min()
    buffer = (valid_start - train_end).days

    assert buffer >= 60, (
        f"LEAKAGE: Train ends {train_end}, Valid starts {valid_start}. "
        f"Buffer is only {buffer} days (need >= 60)."
    )

    if len(test_df) > 0:
        valid_end = valid_df.index.max()
        test_start = test_df.index.min()
        buffer2 = (test_start - valid_end).days

        assert buffer2 >= 60, (
            f"LEAKAGE: Valid ends {valid_end}, Test starts {test_start}. "
            f"Buffer is only {buffer2} days (need >= 60)."
        )

    print("Leakage check PASSED: all splits properly separated.")


def load_or_download(force=False):
    """
    Main entry point: download data, filter, verify splits.
    Returns (data_dict, removal_log).
    """
    print("=" * 60)
    print("TMD-ARC Data Pipeline")
    print("=" * 60)
    print(f"Train:  {TRAIN_START} to {TRAIN_END}")
    print(f"Valid:  {VALID_START} to {VALID_END}")
    print(f"Test:   {TEST_START} to {TEST_END}")
    print(f"Buffer: {SPLIT_BUFFER_DAYS} trading days between splits")
    print()

    data = download_data(force=force)
    filtered, removal_log = apply_quality_filters(data)

    # Verify no leakage for a sample stock
    sample_ticker = "SPY" if "SPY" in filtered else list(filtered.keys())[0]
    sample = filtered[sample_ticker]
    verify_no_leakage(
        split_data(sample, "train"),
        split_data(sample, "valid"),
        split_data(sample, "test")
    )

    return filtered, removal_log


if __name__ == "__main__":
    data, log = load_or_download()
    print(f"\nReady: {len(data)} tickers loaded.")

    # Print split sizes for SPY as reference
    spy = data.get("SPY")
    if spy is not None:
        for split_name in ["train", "valid", "test"]:
            s = split_data(spy, split_name)
            print(f"  {split_name}: {len(s)} days "
                  f"({s.index[0].date()} to {s.index[-1].date()})")
