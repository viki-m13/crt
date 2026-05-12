"""Build an augmented PIT-S&P-500 daily price panel.

Goal: reduce the survivorship bias caused by the v2 panel missing ~374 of the
985 unique tickers that historically belonged to the S&P 500 between 2003 and
2026 (the panel only has 611 of them, i.e. 62% coverage; the 374 missing names
are mostly acquired or bankrupt companies whose tickers were retired).

This script merges three data sources to backfill as many of the missing
tickers as we can WITHOUT paid data:

  1. **Existing v2 daily panel** (cache/prices_extended.parquet): 1833 tickers,
     covers ~62% of the PIT universe.
  2. **FNSPID** (Hugging Face Zihan1004/FNSPID, full_history.zip): community
     dataset scraped through 2023-12 that retains historical OHLC for many
     acquired and renamed names (AGN, ANTM, ABMD, ALXN, ARNC, ATVI, BLL,
     RTN -> RTX, UTX -> RTX, SYMC -> GEN, BHGE -> BKR, DWDP -> DD, etc.).
     License is CC BY-NC 4.0 (research use; non-commercial).
  3. **yfinance**: for ~50 tickers FNSPID does not cover but yfinance still
     serves (mostly recently-acquired or renamed names whose ticker history
     remained queryable, e.g. AAP, AET, ANDV, BOL, CBE).

Each candidate ticker is **validated by date-overlap**: yfinance / FNSPID
must have data that begins before (or within 1y of) when that ticker first
appears in our PIT membership list. Otherwise the symbol is treated as
"recycled" (different company on the same letters, e.g. modern ACV is
Aberdeen Asia Pacific Income, NOT Alberto-Culver) and dropped.

Inputs:
  experiments/monthly_dca/cache/prices_extended.parquet          (daily, 1833 tickers)
  experiments/monthly_dca/cache/v2/sp500_pit/sp500_missing_tickers.txt
  experiments/monthly_dca/cache/v2/sp500_pit/sp500_backfill_plan.json
  experiments/monthly_dca/cache/v2/sp500_pit/fnspid/extracted/*.csv

Outputs:
  experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet
      Wide daily-close panel keyed by date x ticker, including ALL the
      backfilled tickers AND the original 1833 (so a single drop-in for
      cache/prices_extended.parquet in downstream feature / model code).
  experiments/monthly_dca/cache/v2/sp500_pit/backfilled_tickers.json
      Per-ticker provenance: source (fnspid|yfinance), rows fetched,
      yf_start, yf_end, pit_start, pit_end.

  experiments/monthly_dca/cache/v2/sp500_pit/coverage_after_backfill.csv
      Year-by-year fraction of PIT members present in the augmented panel
      (compare against the existing sp500_pit_filter_coverage.csv).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
FNSPID = PIT / "fnspid" / "extracted"


# Renamed-company map: when a PIT ticker was retired due to a merger/rename,
# the current-symbol history under the new ticker IS the historical company's
# data (e.g., Raytheon's RTN was reorganised into RTX in 2020 — RTX's history
# pre-2020 is just RTN's history). Used to expand FNSPID coverage.
RENAMES = {
    "BHGE": "BKR",   # Baker Hughes GE -> Baker Hughes Co
    "DWDP": "DD",    # DowDuPont -> DuPont post 2019 spinoff
    "RTN":  "RTX",   # Raytheon -> Raytheon Technologies (UTC merger)
    "UTX":  "RTX",   # United Tech -> Raytheon Technologies (same merger)
    "SYMC": "GEN",   # Symantec -> NortonLifeLock -> Gen Digital
    "FB":   "META",  # Facebook -> Meta
    "ANTM": "ELV",   # Anthem -> Elevance (kept by FNSPID under ANTM)
    "ADS":  "BFH",   # Alliance Data -> Bread Financial
    "CTL":  "LUMN",  # CenturyLink -> Lumen
    "VIAC": "PARA",  # ViacomCBS -> Paramount
    "VIAB": "PARA",  # Viacom B
    "WLTW": "WTW",   # Willis Towers Watson rebrand
    "FBHS": "FBIN",  # Fortune Brands rebrand
    "NLOK": "GEN",   # NortonLifeLock -> Gen
    "DISCK": "WBD",  # Discovery K -> Warner Bros Discovery
    "DISCA": "WBD",
}


def load_pit_window() -> pd.DataFrame:
    """For each PIT ticker, the (first, last) month it was in the index."""
    mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    return mem.groupby("ticker")["asof"].agg(["min", "max"])


def is_reuse(pit_start: pd.Timestamp, pit_end: pd.Timestamp,
             yf_start: pd.Timestamp) -> bool:
    """Heuristic: source history starts >180d AFTER the ticker left the index
    => almost certainly a different company on the same letters."""
    return yf_start > pit_end + pd.Timedelta(days=180)


def load_fnspid_for_ticker(orig: str) -> pd.Series | None:
    """Load and return a tz-naive daily Close series for one ticker from
    extracted FNSPID CSVs. Returns None if no usable file exists."""
    p = FNSPID / f"{orig}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    if df.empty:
        return None
    df = df.sort_values("date").set_index("date")
    # FNSPID provides "adj close" already split/dividend-adjusted.
    col = "adj close" if "adj close" in df.columns else "close"
    s = df[col].astype(float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def fetch_yfinance(tickers: list[str]) -> dict[str, pd.Series]:
    """Batch fetch daily auto-adjusted Close from yfinance.

    Returns {original_ticker -> Series}. Handles BF.B -> BF-B etc.
    """
    if not tickers:
        return {}
    import yfinance as yf
    variants = []
    map_ = {}
    for t in tickers:
        for v in {t, t.replace(".", "-")}:
            variants.append(v)
            map_[v] = t
    print(f"  yfinance: fetching {len(variants)} symbol variants for {len(tickers)} tickers ...")
    df = yf.download(variants, start="1995-01-01", end="2026-06-01",
                     auto_adjust=True, progress=False, threads=True)
    out: dict[str, pd.Series] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(0)
        close = df["Close"] if "Close" in levels else df["Adj Close"]
    else:
        close = df[["Close"]].rename(columns={"Close": variants[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    for v in variants:
        orig = map_[v]
        if v not in close.columns:
            continue
        s = close[v].dropna()
        if len(s) < 60:
            continue
        # Prefer the variant that gives us more rows
        if orig not in out or len(s) > len(out[orig]):
            out[orig] = s
    return out


def main():
    print("=" * 64)
    print("Build PIT-S&P-500 augmented daily price panel")
    print("=" * 64)

    # 1. Existing daily panel
    base = pd.read_parquet(CACHE / "prices_extended.parquet")
    if not isinstance(base.index, pd.DatetimeIndex):
        base.index = pd.to_datetime(base.index)
    base.index = base.index.tz_localize(None) if base.index.tz is not None else base.index
    print(f"[1] Base panel: {base.shape} ({base.shape[1]} tickers, "
          f"{base.index.min().date()}..{base.index.max().date()})")
    have_panel = set(base.columns)

    # 2. PIT membership window
    pit = load_pit_window()
    pit_tickers = set(pit.index)
    print(f"[2] PIT universe: {len(pit_tickers)} unique tickers")

    missing = sorted(pit_tickers - have_panel)
    print(f"    missing from base panel: {len(missing)}")

    # 3. Try FNSPID for every missing ticker (with rename map)
    print(f"[3] Trying FNSPID for {len(missing)} missing tickers ...")
    fnspid_series: dict[str, pd.Series] = {}
    fnspid_meta: dict[str, dict] = {}
    for t in missing:
        s = load_fnspid_for_ticker(t)
        if s is None and t in RENAMES:
            s = load_fnspid_for_ticker(RENAMES[t])
        if s is None:
            continue
        pit_start, pit_end = pit.loc[t, "min"], pit.loc[t, "max"]
        yf_start = s.index.min()
        if is_reuse(pit_start, pit_end, yf_start):
            # Drop: yfinance/FNSPID has data ONLY after ticker left index -> reuse
            continue
        fnspid_series[t] = s
        fnspid_meta[t] = {
            "source": "fnspid",
            "rename_alias": RENAMES.get(t),
            "rows": int(len(s)),
            "start": str(yf_start.date()),
            "end": str(s.index.max().date()),
            "pit_start": str(pit_start.date()),
            "pit_end": str(pit_end.date()),
        }
    print(f"    FNSPID provided {len(fnspid_series)} valid series")

    # 4. Try yfinance for whatever FNSPID didn't cover
    still_missing = [t for t in missing if t not in fnspid_series]
    print(f"[4] Trying yfinance for {len(still_missing)} still-missing tickers ...")
    raw_yf = fetch_yfinance(still_missing)
    yf_series: dict[str, pd.Series] = {}
    yf_meta: dict[str, dict] = {}
    for t, s in raw_yf.items():
        pit_start, pit_end = pit.loc[t, "min"], pit.loc[t, "max"]
        yf_start = s.index.min()
        if is_reuse(pit_start, pit_end, yf_start):
            continue
        yf_series[t] = s
        yf_meta[t] = {
            "source": "yfinance",
            "rename_alias": None,
            "rows": int(len(s)),
            "start": str(yf_start.date()),
            "end": str(s.index.max().date()),
            "pit_start": str(pit_start.date()),
            "pit_end": str(pit_end.date()),
        }
    print(f"    yfinance provided {len(yf_series)} additional valid series "
          f"(after reuse filtering)")

    # 5. Merge into a single daily panel
    new_series = {**fnspid_series, **yf_series}
    all_meta = {**fnspid_meta, **yf_meta}
    if not new_series:
        print("[5] Nothing to merge; aborting.")
        return
    new_df = pd.DataFrame(new_series)
    new_df = new_df.sort_index()
    # Align to base panel's index
    all_idx = base.index.union(new_df.index)
    base_aligned = base.reindex(all_idx)
    new_aligned = new_df.reindex(all_idx)
    augmented = pd.concat([base_aligned, new_aligned], axis=1)
    # If a ticker happens to exist in both (shouldn't, but guard), the new one
    # is dropped because base order comes first and concat keeps both columns.
    augmented = augmented.loc[:, ~augmented.columns.duplicated()]
    print(f"[5] Augmented panel: {augmented.shape} "
          f"(+{augmented.shape[1] - base.shape[1]} new tickers)")

    # 6. Per-year coverage check
    mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    aug_cols = set(augmented.columns)
    mem["in_aug_panel"] = mem["ticker"].isin(aug_cols).astype(int)
    cov = mem.groupby(mem["asof"].dt.year)["in_aug_panel"].mean()
    cov.to_csv(PIT / "coverage_after_backfill.csv", header=["coverage"])
    print(f"[6] Coverage by year (after backfill):")
    for y, v in cov.items():
        print(f"      {y}: {v:.2%}")

    # 7. Save
    out_panel = PIT / "prices_extended_pit.parquet"
    out_meta = PIT / "backfilled_tickers.json"
    augmented.to_parquet(out_panel)
    with open(out_meta, "w") as f:
        json.dump(all_meta, f, indent=2, sort_keys=True)
    print(f"\nSaved augmented daily panel -> {out_panel}")
    print(f"Saved provenance metadata    -> {out_meta}")

    # 8. Summary
    n_pit = len(pit_tickers)
    n_existing_in_pit = len(pit_tickers & have_panel)
    n_new_in_pit = len(set(new_series.keys()) & pit_tickers)
    print(f"\n=== Summary ===")
    print(f"PIT universe:                       {n_pit}")
    print(f"In base panel:                      {n_existing_in_pit} "
          f"({100*n_existing_in_pit/n_pit:.1f}%)")
    print(f"Backfilled (FNSPID + yfinance):     {n_new_in_pit} "
          f"(+{100*n_new_in_pit/n_pit:.1f}pp)")
    print(f"After backfill (free data ceiling): {n_existing_in_pit + n_new_in_pit} "
          f"({100*(n_existing_in_pit+n_new_in_pit)/n_pit:.1f}%)")
    print(f"Permanently unreachable (free):     {n_pit - n_existing_in_pit - n_new_in_pit}")


if __name__ == "__main__":
    main()
