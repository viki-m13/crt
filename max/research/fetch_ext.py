"""Fetch full-history OHLCV for the 97 stocks + SPY benchmark via yfinance.

Saves one parquet per field (Open/High/Low/Close/Volume/AdjClose) to
max/research/data/raw/. The scanner uses auto_adjust=False and takes both
Close and AdjClose, so we match that.

This is the ONLY script that touches yfinance directly — everything
downstream reads from the saved parquets so the validation is reproducible
without re-downloading.
"""
import os, sys, time
import pandas as pd
import yfinance as yf

# The 97 stocks from SHORT_UNIVERSE minus crypto (BTC/ETH) and SQ (delisted).
# Kept SPY as explicit benchmark — scanner includes it already.
# Fetched via period="max" so each ticker starts from its own IPO/listing.
STOCKS = [
    "AAPL","MSFT","NVDA","AVGO","ADBE","CRM","AMD","INTC","CSCO","TXN",
    "AMAT","MU","NOW","PANW","CDNS",
    "GOOGL","META","NFLX","DIS","T","VZ",
    "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","GM",
    "PG","KO","PEP","COST","WMT","PM","CL",
    "XOM","CVX","COP","SLB","EOG","MPC",
    "JPM","BAC","WFC","GS","MS","BLK","AXP","C","USB","PNC",
    "UNH","JNJ","LLY","PFE","ABBV","TMO","ABT","MRK","AMGN","GILD",
    "CAT","HON","UPS","BA","RTX","DE","GE","LMT","UNP","FDX",
    "LIN","APD","FCX","NEM","NUE",
    "AMT","PLD","CCI","EQIX","SPG",
    "NEE","DUK","SO","D","AEP",
    "SPY","QQQ","IWM","DIA",
    "ARM","SMCI","COIN","MARA",
]

OUT_DIR = "/home/user/crt/max/research/data/raw"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for lvl in (1, -1, 0):
            try:
                out = raw.xs(field, axis=1, level=lvl)
                if isinstance(out, pd.Series):
                    out = out.to_frame()
                out.columns = out.columns.astype(str)
                return out
            except Exception:
                pass
    if field in raw.columns:
        s = raw[field].copy()
        return s.to_frame() if isinstance(s, pd.Series) else s
    return pd.DataFrame()


def fetch(tickers, chunk=60):
    fields = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    bufs = {f: [] for f in fields}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        print(f"  [{i+1:3d}..{min(i+chunk, len(tickers)):3d}] fetching {len(batch)} tickers", flush=True)
        raw = yf.download(batch, period="max", interval="1d",
                          auto_adjust=False, progress=False,
                          group_by="ticker", threads=True)
        for f in fields:
            df = extract_field(raw, f)
            if not df.empty:
                bufs[f].append(df)
        time.sleep(1)
    out = {}
    for f in fields:
        xs = [x for x in bufs[f] if x is not None and not x.empty]
        if not xs:
            out[f.replace(" ", "")] = pd.DataFrame()
            continue
        x = pd.concat(xs, axis=1)
        x = x.loc[:, ~x.columns.duplicated()].sort_index()
        x.index = pd.to_datetime(x.index, utc=True).tz_localize(None)
        out[f.replace(" ", "")] = x
    return out


if __name__ == "__main__":
    print(f"Fetching {len(STOCKS)} tickers from yfinance (period=max)...")
    data = fetch(sorted(set(STOCKS)))
    for f, df in data.items():
        path = os.path.join(OUT_DIR, f"{f}.parquet")
        df.to_parquet(path, compression="zstd")
        size_mb = os.path.getsize(path) / 1e6
        first = df.index.min() if len(df) else "-"
        last  = df.index.max() if len(df) else "-"
        print(f"  {f:10s}: {df.shape}  {first} → {last}  {size_mb:.2f} MB")
    # Sanity: AdjClose start dates per ticker
    ac = data["AdjClose"]
    starts = {c: ac[c].first_valid_index() for c in ac.columns}
    starts = sorted(starts.items(), key=lambda x: (x[1] or pd.Timestamp.max))
    print("\nPer-ticker first valid date:")
    for t, d in starts:
        print(f"  {t:6s}  {d}")
