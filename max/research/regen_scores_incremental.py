"""Incremental regen: only score tickers missing from bt_ext.parquet.

Loads the existing bt_ext.parquet and only runs score_one_ticker for
tickers NOT yet present. Used after fetch_new_tickers.py to score only
the new additions (much faster than a full regen).

IMPORTANT: The market regime features are still recomputed, since SPY's
OHLCV may have been extended.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/crt/max/scripts")
import daily_scan_max as ds

EXT_YEARS = 20
ds.PLOT_LAST_DAYS = 365 * EXT_YEARS
print(f"PLOT_LAST_DAYS = {ds.PLOT_LAST_DAYS}")

RAW = "/home/user/crt/max/research/data/raw"
OUT = "/home/user/crt/max/research/data"

print("Loading parquets...")
O = pd.read_parquet(os.path.join(RAW, "Open.parquet"))
H = pd.read_parquet(os.path.join(RAW, "High.parquet"))
L = pd.read_parquet(os.path.join(RAW, "Low.parquet"))
C = pd.read_parquet(os.path.join(RAW, "Close.parquet"))
V = pd.read_parquet(os.path.join(RAW, "Volume.parquet"))
A = pd.read_parquet(os.path.join(RAW, "AdjClose.parquet"))
print(f"  raw parquet shape: {O.shape}")

for df in (O, H, L, C, V, A):
    df.index = pd.to_datetime(df.index, utc=True)

PX = A if not A.empty else C
BENCH = ds.BENCH
spy_px = PX[BENCH].dropna()
spy_h = H[BENCH].reindex(spy_px.index).dropna()
spy_l = L[BENCH].reindex(spy_px.index).dropna()
spy_px = spy_px.reindex(spy_h.index).reindex(spy_l.index).dropna()

print("Computing market regime on SPY...")
mkt = ds.compute_market_regime(spy_px, spy_h, spy_l)

feature_cols = [
    "dd_lt","pos_lt","dd_st","pos_st","atr_pct","volu_z","gap","trend_st",
    "idio_dd_lt","idio_pos_lt","idio_dd_st","idio_pos_st",
    "mkt_trend","mkt_vol","mkt_dd","mkt_atr_pct",
]
zwin = max(63, ds.LB_ST)

all_tickers = [c for c in C.columns if c in O.columns and c in H.columns
               and c in L.columns and c in V.columns and c in PX.columns]

# Load existing bt_ext
existing_path = os.path.join(OUT, "bt_ext.parquet")
existing_tks = set()
existing_df = None
if os.path.exists(existing_path):
    existing_df = pd.read_parquet(existing_path)
    existing_tks = set(existing_df["ticker"].unique())
    print(f"Existing bt_ext has {len(existing_tks)} tickers")
else:
    print("No existing bt_ext.parquet; full regen required")
    sys.exit(1)

todo = sorted([t for t in all_tickers if t not in existing_tks])
print(f"Tickers to score incrementally: {len(todo)}")
print(f"  {todo}")
if not todo:
    print("Nothing to do.")
    sys.exit(0)

today_path = os.path.join(OUT, "today_ext.parquet")
today_df = pd.read_parquet(today_path) if os.path.exists(today_path) else pd.DataFrame()

bt_rows = []
today_rows = []

t0 = time.time()
scored = 0
for i, t in enumerate(todo, 1):
    try:
        out = ds.score_one_ticker(t, O, H, L, C, V, PX, spy_px, mkt, feature_cols, zwin)
    except Exception as e:
        print(f"  [{i}/{len(todo)}] {t}: ERROR {type(e).__name__}: {e}", flush=True)
        continue
    if out is None:
        print(f"  [{i}/{len(todo)}] {t}: skipped", flush=True)
        continue
    row, det = out
    series = det.get("series", {})
    if not series.get("dates"):
        continue
    q = row.get("quality")
    for d, p, f, w, fr in zip(series["dates"], series["prices"], series["final"],
                               series.get("wash", [None]*len(series["dates"])),
                               series.get("final_raw", [None]*len(series["dates"]))):
        bt_rows.append({"ticker": t, "date": d, "price": p, "final": f,
                        "wash": w, "final_raw": fr, "quality": q})
    today_rows.append(row)
    scored += 1
    if i % 5 == 0 or i == len(todo):
        elapsed = time.time() - t0
        print(f"  [{i}/{len(todo)}] scored {scored}  {elapsed:.0f}s", flush=True)

print(f"\nScored {scored} new tickers")

# Merge into existing
new_df = pd.DataFrame(bt_rows)
new_df["date"] = pd.to_datetime(new_df["date"])

combined = pd.concat([existing_df, new_df], ignore_index=True)
bak = existing_path + ".pre_incremental.bak"
if not os.path.exists(bak):
    os.rename(existing_path, bak)
combined.to_parquet(existing_path, compression="zstd")
print(f"\nMerged: {len(existing_df):,} existing + {len(new_df):,} new = {len(combined):,} rows")
print(f"  Tickers: {combined['ticker'].nunique()}")
print(f"  → {existing_path}")

# Update today_ext.parquet
if today_rows:
    new_today = pd.DataFrame(today_rows)
    if not today_df.empty:
        # Drop any same-ticker stale rows and append
        today_df = today_df[~today_df["ticker"].isin(new_today["ticker"])]
        combined_today = pd.concat([today_df, new_today], ignore_index=True)
    else:
        combined_today = new_today
    tbak = today_path + ".pre_incremental.bak"
    if not os.path.exists(tbak):
        os.rename(today_path, tbak)
    combined_today.to_parquet(today_path, compression="zstd")
    print(f"  today_ext.parquet: {len(combined_today)} rows")

print("\n## DONE")
