"""Retry single-ticker pulls that failed. Patches the existing parquets."""
import os, time
import pandas as pd
import yfinance as yf

RAW = "/home/user/crt/max/research/data/raw"
MISSING = ["JNJ"]

fields = [("Open", "Open"), ("High", "High"), ("Low", "Low"),
          ("Close", "Close"), ("Volume", "Volume"), ("AdjClose", "Adj Close")]

for tk in MISSING:
    print(f"Fetching {tk}...")
    # Exponential backoff retry
    df = None
    for attempt in range(5):
        try:
            df = yf.download(tk, period="max", interval="1d",
                             auto_adjust=False, progress=False)
            if df is not None and len(df) > 0:
                break
        except Exception as e:
            print(f"  attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)
    if df is None or len(df) == 0:
        print(f"  FAILED: {tk}")
        continue
    # yfinance single-ticker can return MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    for fname, yfcol in fields:
        path = os.path.join(RAW, f"{fname}.parquet")
        existing = pd.read_parquet(path)
        col = df[yfcol] if yfcol in df.columns else None
        if col is None:
            print(f"  {fname}: source column '{yfcol}' missing"); continue
        existing[tk] = col.reindex(existing.index)
        existing = existing.sort_index(axis=1)
        existing.to_parquet(path, compression="zstd")
        print(f"  {fname}: added column {tk} → shape {existing.shape}")

print("\nSanity:")
ac = pd.read_parquet(os.path.join(RAW, "AdjClose.parquet"))
for tk in MISSING:
    if tk in ac.columns:
        fv = ac[tk].first_valid_index()
        lv = ac[tk].last_valid_index()
        print(f"  {tk}: {fv} → {lv}  nonnull={ac[tk].notna().sum()}")
