"""Load iShares Core S&P 500 ETF (IVV) holdings to define current S&P 500 universe.

Downloads the holdings CSV/XLS from iShares:
  https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf

Extracts tickers and saves to:
  experiments/monthly_dca/v5/ivv_holdings_latest.csv

Then refreshes yfinance data for any tickers missing from
docs/data/tickers/*.json so the panel covers the current S&P 500.

Run from repo root:
    python3 -m experiments.monthly_dca.v5.load_ivv_holdings
"""
from __future__ import annotations
import json
import io
import sys
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TICKER_DIR = ROOT / "docs" / "data" / "tickers"
V5_DIR = ROOT / "experiments" / "monthly_dca" / "v5"
IVV_HOLDINGS = V5_DIR / "ivv_holdings_latest.csv"

# iShares public-data CSV endpoint (no auth needed)
IVV_URL = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"

# Some IVV listings include cash / non-equity entries we filter out.
EXCLUDE_SECTORS = {"Cash and/or Derivatives", "Money Market", "Cash"}
EXCLUDE_TICKERS = {"USD", "MARGIN_USD", "BLK CSH FND TREASURY SL AGNCY",
                   "BLK CSH FND TREASURY SL AGENCY", "BLACKROCK CASH",
                   "GOLDMAN FS TREASURY"}


def _normalize_ticker(t: str) -> str:
    """Standardise iShares ticker format to yfinance / our panel format.

    iShares uses concatenated tickers (BRKB, BFB) while yfinance and our
    panel use the hyphenated form (BRK-B, BF-B).  Apply known mappings.
    """
    if not t or not isinstance(t, str):
        return ""
    t = t.strip().upper()
    SHARE_CLASS_MAP = {
        "BRKB": "BRK-B",
        "BFB": "BF-B",
        "BRKA": "BRK-A",
    }
    if t in SHARE_CLASS_MAP:
        return SHARE_CLASS_MAP[t]
    return t


def fetch_ivv_holdings() -> pd.DataFrame:
    """Fetch the latest IVV holdings CSV from iShares. Returns a DataFrame
    with columns ticker, name, weight_pct, sector."""
    import urllib.request, ssl
    print(f"Fetching IVV holdings from iShares...", flush=True)
    # SSL context that's lenient on cert validity for environments with
    # skewed clocks; iShares CSV is public data so this is acceptable.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(IVV_URL,
                                  headers={"User-Agent": "Mozilla/5.0 (compatible; dailystockguide)"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    # iShares CSV has a header section (about 9 lines), then the holdings table.
    # Find the line that starts with "Ticker," (the column header row).
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"Ticker"') or line.startswith("Ticker,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not locate header row 'Ticker,' in IVV CSV")
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_ivv_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Return clean ticker-level holdings (one row per stock)."""
    # Look for expected columns
    expected = ["Ticker", "Name", "Weight (%)", "Sector", "Asset Class"]
    avail = [c for c in df.columns if c in expected]
    if "Ticker" not in df.columns:
        raise RuntimeError(f"Ticker column missing. Columns: {df.columns.tolist()[:20]}")
    out = pd.DataFrame()
    out["ticker"] = df["Ticker"].astype(str).map(_normalize_ticker)
    out["name"] = df.get("Name", "").astype(str)
    out["sector"] = df.get("Sector", "").astype(str)
    out["asset_class"] = df.get("Asset Class", "").astype(str)
    w_col = next((c for c in df.columns if "weight" in c.lower()), None)
    out["weight_pct"] = pd.to_numeric(df[w_col], errors="coerce") if w_col else None
    # Filter out cash / non-equity
    out = out[~out["sector"].isin(EXCLUDE_SECTORS)]
    out = out[~out["ticker"].isin(EXCLUDE_TICKERS)]
    out = out[out["ticker"].str.match(r"^[A-Z][A-Z\.\-]{0,5}$", na=False)]
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"] != ""]
    out = out.drop_duplicates(subset=["ticker"])
    out = out.sort_values("weight_pct", ascending=False).reset_index(drop=True)
    return out


def refresh_yfinance(missing: list[str], start: str = "2014-01-01") -> None:
    """Backfill any IVV constituents not yet in docs/data/tickers/."""
    if not missing:
        print("No new tickers to fetch.")
        return
    print(f"Fetching {len(missing)} missing tickers via yfinance...", flush=True)
    try:
        import yfinance as yf
    except ImportError:
        print("WARNING: yfinance not installed; skipping fetch")
        return
    df = yf.download(missing, start=start, auto_adjust=True, progress=False, threads=True)
    if df is None or df.empty:
        print("WARNING: yfinance returned empty")
        return
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else df.xs("Adj Close", axis=1, level=0)
    else:
        close = df[["Close"]].rename(columns={"Close": missing[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)

    for tk in missing:
        if tk not in close.columns:
            continue
        ts = close[tk].dropna()
        if len(ts) < 252:
            print(f"  {tk}: only {len(ts)} days; skipping")
            continue
        json_path = TICKER_DIR / f"{tk}.json"
        data = {
            "ticker": tk,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "source": "yfinance",
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in ts.index],
                "prices": [float(p) for p in ts.values],
            },
        }
        json_path.write_text(json.dumps(data))
        print(f"  saved {tk}.json ({len(ts)} days)")


def main():
    raw = fetch_ivv_holdings()
    holdings = parse_ivv_holdings(raw)
    print(f"Parsed {len(holdings)} IVV constituents")
    V5_DIR.mkdir(parents=True, exist_ok=True)
    holdings.to_csv(IVV_HOLDINGS, index=False)
    print(f"Saved to {IVV_HOLDINGS}")

    # Identify missing tickers
    existing = {p.stem for p in TICKER_DIR.glob("*.json")}
    missing = sorted([t for t in holdings["ticker"] if t and t not in existing])
    print(f"\n{len(missing)} IVV constituents missing from docs/data/tickers/:")
    if missing:
        for t in missing[:20]:
            print(f"  {t}")
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")
        refresh_yfinance(missing)
    # Summary
    print(f"\nTop 10 by weight:")
    print(holdings.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
