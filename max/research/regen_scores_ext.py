"""Regenerate point-in-time final_score series over extended history.

Loads OHLCV parquets produced by fetch_ext.py, then invokes the live
scanner's score_one_ticker() with PLOT_LAST_DAYS monkey-patched to a
long window (default 20 years). Extracts the per-ticker historical
series (dates, prices, final) and saves everything to
max/research/data/bt_ext.parquet files.

This faithfully uses the same scoring machinery that powers production
(analog matching is point-in-time within score_one_ticker), with the
single exception that `quality` remains today's value — documented in
the scanner's own docstring and accepted here as a minor known
concession (same as production).
"""
import os, sys, time, json
import numpy as np
import pandas as pd

# Import the scanner's scoring module
sys.path.insert(0, "/home/user/crt/max/scripts")
import daily_scan_max as ds

# Extend the backtest plot window to ~20 years
EXT_YEARS = 20
ds.PLOT_LAST_DAYS = 365 * EXT_YEARS
print(f"PLOT_LAST_DAYS overridden to {ds.PLOT_LAST_DAYS} ({EXT_YEARS}y)")

RAW = "/home/user/crt/max/research/data/raw"
OUT = "/home/user/crt/max/research/data"

print("Loading parquets...")
O = pd.read_parquet(os.path.join(RAW, "Open.parquet"))
H = pd.read_parquet(os.path.join(RAW, "High.parquet"))
L = pd.read_parquet(os.path.join(RAW, "Low.parquet"))
C = pd.read_parquet(os.path.join(RAW, "Close.parquet"))
V = pd.read_parquet(os.path.join(RAW, "Volume.parquet"))
A = pd.read_parquet(os.path.join(RAW, "AdjClose.parquet"))
print(f"  shape: {O.shape}, date range {O.index.min()} → {O.index.max()}")

# Normalize index to UTC-naive → scanner uses UTC tz-aware; match it.
for df in (O, H, L, C, V, A):
    df.index = pd.to_datetime(df.index, utc=True)

PX = A if not A.empty else C

BENCH = ds.BENCH  # "SPY"
if BENCH not in PX.columns:
    raise RuntimeError(f"{BENCH} missing from AdjClose parquet")

spy_px = PX[BENCH].dropna()
spy_h = H[BENCH].reindex(spy_px.index).dropna()
spy_l = L[BENCH].reindex(spy_px.index).dropna()
spy_px = spy_px.reindex(spy_h.index).reindex(spy_l.index).dropna()
print(f"SPY series length: {len(spy_px)}  {spy_px.index.min()} → {spy_px.index.max()}")

print("Computing market regime features on SPY...")
mkt = ds.compute_market_regime(spy_px, spy_h, spy_l)
print(f"  mkt shape: {mkt.shape}")

feature_cols = [
    "dd_lt","pos_lt","dd_st","pos_st","atr_pct","volu_z","gap","trend_st",
    "idio_dd_lt","idio_pos_lt","idio_dd_st","idio_pos_st",
    "mkt_trend","mkt_vol","mkt_dd","mkt_atr_pct",
]
zwin = max(63, ds.LB_ST)

tickers = [c for c in C.columns if c in O.columns and c in H.columns
           and c in L.columns and c in V.columns and c in PX.columns]
print(f"Scoring {len(tickers)} tickers...")

bt_records = {}  # ticker -> {"dates": [...], "prices": [...], "final": [...]}
today_rows = []  # today's summary per ticker

t0 = time.time()
for i, t in enumerate(sorted(tickers), 1):
    try:
        out = ds.score_one_ticker(t, O, H, L, C, V, PX, spy_px, mkt, feature_cols, zwin)
    except Exception as e:
        print(f"  [{i:3d}/{len(tickers)}] {t}: ERROR {type(e).__name__}: {e}")
        continue
    if out is None:
        print(f"  [{i:3d}/{len(tickers)}] {t}: skipped")
        continue
    row, det = out
    series = det.get("series", {})
    if not series.get("dates"):
        continue
    bt_records[t] = {
        "dates": series["dates"],
        "prices": series["prices"],
        "final": series["final"],
    }
    today_rows.append(row)
    if i % 10 == 0 or i == len(tickers):
        elapsed = time.time() - t0
        print(f"  [{i:3d}/{len(tickers)}] scored {len(bt_records)}  {elapsed:.0f}s")

print(f"\nTotal scored: {len(bt_records)}")

# Save: one long-form parquet with (ticker, date, price, final)
rows = []
for tk, s in bt_records.items():
    for d, p, f in zip(s["dates"], s["prices"], s["final"]):
        rows.append({"ticker": tk, "date": d, "price": p, "final": f})
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
out_path = os.path.join(OUT, "bt_ext.parquet")
df.to_parquet(out_path, compression="zstd")
print(f"\nWrote {len(df):,} rows → {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")

# Per-ticker summary of coverage
cov = df.groupby("ticker").agg(
    n=("date", "size"),
    start=("date", "min"),
    end=("date", "max"),
    first_final=("final", lambda s: s.dropna().iloc[0] if s.dropna().size else None),
    last_final=("final", lambda s: s.dropna().iloc[-1] if s.dropna().size else None),
).reset_index().sort_values("start")
print("\nCoverage by ticker (head/tail):")
print(cov.head(10).to_string(index=False))
print("...")
print(cov.tail(10).to_string(index=False))

# Today's summary
tdf = pd.DataFrame(today_rows)
tdf.to_parquet(os.path.join(OUT, "today_ext.parquet"), compression="zstd")
print(f"\nToday summary: {tdf.shape} → today_ext.parquet")
