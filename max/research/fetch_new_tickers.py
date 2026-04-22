"""Fetch OHLCV for NEW tickers and merge into existing raw parquets.

Unlike fetch_ext.py which rebuilds everything from scratch, this script:
  1. Reads existing raw/{Open,High,Low,Close,Volume,AdjClose}.parquet
  2. Identifies which of the proposed new tickers aren't already present
  3. Fetches period="max" OHLCV for the missing ones via yfinance
  4. Merges into the existing parquets, saves back

Used for universe expansion (step35 setup).
"""
import os, sys, time
import pandas as pd
import yfinance as yf

# Proposed additions from universe expansion research (step33 agent).
# All have >= 15Y history and fit CAP5's growth/volatility/recovery profile.
NEW_TICKERS = [
    # Technology additions
    "ORCL", "QCOM", "KLAC", "LRCX", "ASML", "TER", "NTAP", "GRMN", "SWKS", "INFY",
    # Financial Services
    "MA", "CME", "BX", "ARCC",
    # Healthcare
    "VRTX",
    # Consumer / Industrials / Materials
    "AZO", "EXPE", "STZ", "PSA", "JCI", "EMR", "SCCO", "CPRT", "LUV",
    # Additional quality/liquid names (step20 / strategy fit)
    "SHW",   # Sherwin Williams — industrial defensive
    "ADP",   # payroll — recurring revenue
    "ACN",   # Accenture — consulting growth
    "ISRG",  # Intuitive Surgical — medical devices growth
    "REGN",  # Regeneron — biotech growth
    "CHTR",  # Charter — telecom cyclical
    "GIS",   # General Mills — consumer defensive
]

OUT_DIR = "/home/user/crt/max/research/data/raw"


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


def fetch_chunk(tickers, chunk=30):
    fields = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    bufs = {f: [] for f in fields}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        print(f"  [{i+1:3d}..{min(i+chunk, len(tickers)):3d}] fetching {len(batch)} tickers", flush=True)
        try:
            raw = yf.download(batch, period="max", interval="1d",
                              auto_adjust=False, progress=False,
                              group_by="ticker", threads=True)
        except Exception as e:
            print(f"    ERROR: {e}; retrying single-ticker")
            raw = None
        if raw is None or raw.empty:
            # Fallback: single-ticker fetch
            for t in batch:
                try:
                    r1 = yf.download(t, period="max", interval="1d",
                                     auto_adjust=False, progress=False)
                    if r1 is not None and not r1.empty:
                        for f in fields:
                            df = r1[[f]].copy()
                            df.columns = [t]
                            bufs[f].append(df)
                except Exception:
                    pass
            time.sleep(2)
            continue
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


def merge_and_save(new_data):
    """Merge new-ticker data into existing parquets."""
    fields = ["Open", "High", "Low", "Close", "Volume", "AdjClose"]
    summary = {}
    for f in fields:
        path = os.path.join(OUT_DIR, f"{f}.parquet")
        existing = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
        existing.index = pd.to_datetime(existing.index, utc=True).tz_localize(None) \
            if hasattr(existing.index, 'tz') and existing.index.tz is not None \
            else pd.to_datetime(existing.index)

        nd = new_data.get(f, pd.DataFrame())
        if nd.empty:
            print(f"  {f}: no new data, skipping")
            continue

        # Union of columns (preserve existing if duplicate)
        new_cols = [c for c in nd.columns if c not in existing.columns]
        if not new_cols:
            print(f"  {f}: all new tickers already present, skipping")
            continue

        nd_new = nd[new_cols]
        # Align indices
        combined_idx = existing.index.union(nd_new.index).sort_values()
        existing_r = existing.reindex(combined_idx)
        nd_r = nd_new.reindex(combined_idx)
        merged = pd.concat([existing_r, nd_r], axis=1)

        # Backup and save
        bak = path + ".prev.bak"
        if not os.path.exists(bak):
            os.rename(path, bak) if os.path.exists(path) else None
            existing.to_parquet(bak, compression="zstd") if os.path.exists(bak) is False else None
        merged.to_parquet(path, compression="zstd")
        summary[f] = dict(added=new_cols, shape=merged.shape)
        print(f"  {f}: added {len(new_cols)} tickers, shape now {merged.shape}")
    return summary


if __name__ == "__main__":
    # Load existing AdjClose to see which are new
    ac_path = os.path.join(OUT_DIR, "AdjClose.parquet")
    existing = pd.read_parquet(ac_path)
    existing_tks = set(existing.columns)

    to_fetch = sorted([t for t in NEW_TICKERS if t not in existing_tks])
    print(f"Proposed new tickers: {len(NEW_TICKERS)}")
    print(f"Already in parquet  : {len(NEW_TICKERS) - len(to_fetch)}")
    print(f"Will fetch          : {len(to_fetch)}")
    print(f"  {to_fetch}")

    if not to_fetch:
        print("Nothing to fetch. Exiting.")
        sys.exit(0)

    data = fetch_chunk(to_fetch)

    # Report per-ticker history coverage before merging
    ac_new = data.get("AdjClose")
    if ac_new is not None and not ac_new.empty:
        print("\nPer-new-ticker first valid date:")
        for t in ac_new.columns:
            first = ac_new[t].first_valid_index()
            last = ac_new[t].last_valid_index()
            print(f"  {t:6s}  {first} → {last}")

    summary = merge_and_save(data)
    print(f"\nDone. Summary: {summary}")
