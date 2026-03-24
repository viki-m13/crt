"""
Crypto Data Pipeline for TMD-ARC Strategy
==========================================
Downloads cryptocurrency OHLCV data and creates strict train/validation/test splits.

Crypto-specific adaptations:
- 24/7 markets → 365 trading days/year (vs 252 for stocks)
- Higher volatility baseline → adjusted quality filters
- BTC as market benchmark (instead of SPY)
- Crypto universe: top liquid coins with sufficient history

Split methodology (same temporal discipline as stocks):
- Training:   2018-01-01 to 2021-12-31 (4 years — crypto data sparser pre-2018)
- Validation:  2022-04-01 to 2023-06-30 (bear market + recovery)
- Test:        2023-10-01 to present     (out-of-sample)

A 90-day buffer between splits prevents feature leakage.
"""

import os
import json
import datetime
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf


# === SPLIT DATES (FIXED - NEVER MODIFY) ===
TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VALID_START = "2022-04-01"  # 90-day buffer after train
VALID_END = "2023-06-30"
TEST_START = "2023-10-01"   # 90-day buffer after validation
TEST_END = "2026-03-15"     # ~current

# Buffer in calendar days between splits
SPLIT_BUFFER_DAYS = 90

# Minimum trading days required per coin
MIN_HISTORY_DAYS = 800

# Minimum median daily dollar volume for crypto
MIN_MEDIAN_DOLLAR_VOLUME = 1_000_000

# Trading days per year for crypto (24/7 markets)
TRADING_DAYS_PER_YEAR = 365

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_universe():
    """
    Returns the crypto universe: top liquid cryptocurrencies available on Yahoo Finance.
    Focuses on coins with sufficient history (pre-2018) and liquidity.
    """
    # Extensive crypto universe — 108 liquid cryptocurrencies
    crypto = [
        # Top 20 by Market Cap
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD",
        "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD",
        "BCH-USD", "XLM-USD", "ALGO-USD", "FIL-USD", "NEAR-USD",
        # DeFi
        "AAVE-USD", "MKR-USD", "SNX-USD", "SUSHI-USD", "YFI-USD",
        "CRV-USD", "BAL-USD", "UMA-USD", "1INCH-USD", "DYDX-USD",
        "LDO-USD", "RPL-USD",
        # Layer 1 / Layer 2
        "ICP-USD", "FTM-USD", "HBAR-USD", "THETA-USD", "VET-USD",
        "NEO-USD", "EOS-USD", "TRX-USD", "EGLD-USD", "FLOW-USD",
        "MINA-USD", "KAVA-USD", "ONE-USD", "CELO-USD", "KSM-USD",
        "ZIL-USD", "ICX-USD", "QTUM-USD", "WAVES-USD", "XTZ-USD",
        "FET-USD", "ROSE-USD", "AR-USD", "STX-USD",
        "SUI-USD", "APT-USD", "SEI-USD", "INJ-USD", "TIA-USD",
        "OP-USD", "ARB-USD",
        # Gaming / Metaverse
        "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "GALA-USD",
        "ILV-USD", "RNDR-USD",
        # Privacy
        "ZEC-USD", "DASH-USD", "XMR-USD",
        # Exchange Tokens
        "CRO-USD", "OKB-USD", "LEO-USD",
        # Infrastructure / Oracle
        "CHZ-USD", "LRC-USD", "BAT-USD", "ZRX-USD",
        "STORJ-USD", "ANKR-USD", "BAND-USD", "API3-USD",
        # Meme / Community
        "SHIB-USD", "PEPE24478-USD", "FLOKI-USD", "BONK-USD",
        # Newer Large Caps
        "WLD-USD", "JUP-USD", "PENDLE-USD",
        # AI / Data
        "OCEAN-USD", "AGIX-USD",
        # Additional Liquid Alts
        "SKL-USD", "COTI-USD", "RLC-USD", "NMR-USD",
        "REQ-USD", "OMG-USD", "CELR-USD",
        "DENT-USD", "HOT-USD", "WIN-USD", "SC-USD",
        "AUDIO-USD", "PERP-USD", "RUNE-USD", "OSMO-USD",
    ]
    return crypto


def download_data(tickers=None, force=False):
    """
    Download OHLCV data from Yahoo Finance for crypto.
    Caches to disk to avoid repeated downloads.
    Returns dict of {ticker: DataFrame}.
    """
    if tickers is None:
        tickers = get_universe()

    os.makedirs(DATA_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "download_manifest.json")

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    results = {}
    failed = []
    today = datetime.date.today().isoformat()

    for ticker in tickers:
        cache_path = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.csv")

        # Use cache if fresh
        if not force and os.path.exists(cache_path):
            if manifest.get(ticker, {}).get("date") == today:
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    results[ticker] = df
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
                print(f"  WARNING: {ticker} has insufficient data ({len(data) if data is not None else 0} rows)")
                failed.append(ticker)
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.to_csv(cache_path)
            manifest[ticker] = {
                "date": today,
                "rows": len(data),
                "start": str(data.index[0].date()),
                "end": str(data.index[-1].date()),
                "checksum": hashlib.md5(str(len(data)).encode()).hexdigest()[:12]
            }
            results[ticker] = data
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
            failed.append(ticker)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDownloaded {len(results)}/{len(tickers)} tickers. Failed: {len(failed)}")
    if failed:
        print(f"  Failed tickers: {failed}")

    return results


def apply_quality_filters(data_dict):
    """Filter crypto by minimum history and liquidity."""
    filtered = {}
    removal_log = []

    for ticker, df in data_dict.items():
        if len(df) < MIN_HISTORY_DAYS:
            removal_log.append({
                "ticker": ticker,
                "reason": "insufficient_history",
                "detail": f"{len(df)} days < {MIN_HISTORY_DAYS} required"
            })
            continue

        if "Volume" in df.columns and "Close" in df.columns:
            dollar_vol = (df["Close"] * df["Volume"]).median()
            if dollar_vol < MIN_MEDIAN_DOLLAR_VOLUME:
                removal_log.append({
                    "ticker": ticker,
                    "reason": "low_liquidity",
                    "detail": f"median $vol {dollar_vol:,.0f} < {MIN_MEDIAN_DOLLAR_VOLUME:,.0f}"
                })
                continue

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
    """Extract a specific time split from a DataFrame."""
    if split == "train":
        return df.loc[TRAIN_START:TRAIN_END].copy()
    elif split == "valid":
        return df.loc[VALID_START:VALID_END].copy()
    elif split == "test":
        return df.loc[TEST_START:TEST_END].copy()
    elif split == "train_valid":
        train = df.loc[TRAIN_START:TRAIN_END].copy()
        valid = df.loc[VALID_START:VALID_END].copy()
        return pd.concat([train, valid])
    else:
        raise ValueError(f"Unknown split: {split}")


def verify_no_leakage(train_df, valid_df, test_df):
    """Verify there is no temporal overlap between splits."""
    if len(train_df) == 0 or len(valid_df) == 0:
        return

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
    """Main entry point: download data, filter, verify splits."""
    print("=" * 60)
    print("Crypto TMD-ARC Data Pipeline")
    print("=" * 60)
    print(f"Train:  {TRAIN_START} to {TRAIN_END}")
    print(f"Valid:  {VALID_START} to {VALID_END}")
    print(f"Test:   {TEST_START} to {TEST_END}")
    print(f"Buffer: {SPLIT_BUFFER_DAYS} calendar days between splits")
    print()

    data = download_data(force=force)
    filtered, removal_log = apply_quality_filters(data)

    sample_ticker = "BTC-USD" if "BTC-USD" in filtered else list(filtered.keys())[0]
    sample = filtered[sample_ticker]
    verify_no_leakage(
        split_data(sample, "train"),
        split_data(sample, "valid"),
        split_data(sample, "test")
    )

    return filtered, removal_log


if __name__ == "__main__":
    data, log = load_or_download()
    print(f"\nReady: {len(data)} crypto tickers loaded.")

    btc = data.get("BTC-USD")
    if btc is not None:
        for split_name in ["train", "valid", "test"]:
            s = split_data(btc, split_name)
            if len(s) > 0:
                print(f"  {split_name}: {len(s)} days "
                      f"({s.index[0].date()} to {s.index[-1].date()})")
