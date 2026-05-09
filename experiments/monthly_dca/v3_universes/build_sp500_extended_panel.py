"""
Build the SP500 PIT cross-section by EXTENDING the v2 cross-section with
features for the 194 delisted-S&P-500 tickers.

This avoids re-computing the 67 features for the 629 tickers already in v2
cache (which would take ~30 min). We only compute features for the 194
"new" delisted tickers.

Saves: cache/v3_universes/sp500_pit/extended_cross_section.parquet

Strategy:
  1. Load v2 cross-section (1,833 tickers × 352 months × 67 features)
  2. Filter v2 cross-section to (asof, ticker) where ticker in S&P 500 PIT membership
  3. For the 194 delisted-only tickers, compute features on the COMBINED panel
     (v2 panel + delisted backfill, so SPY-relative features compute correctly)
  4. Append to cross-section
  5. Apply PIT membership filter

Run:
    python3 -m experiments.monthly_dca.v3_universes.build_sp500_extended_panel
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.monthly_dca.backtester import compute_features
from experiments.monthly_dca.extra_features import compute_extras
from experiments.monthly_dca.alpha_features import compute_alpha_features
from experiments.monthly_dca.alpha2_features import compute_alpha2

CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
DATA = ROOT / "experiments" / "monthly_dca" / "v3_universes" / "data"
OUT = CACHE / "v3_universes" / "sp500_pit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load existing v2 cross-section
    print("Loading v2 cross-section...")
    v2_big = pd.read_parquet(CACHE / "v2" / "panel_cross_section_v3.parquet")
    print(f"  v2 cross-section: {v2_big.shape}")
    v2_tickers = set(v2_big.reset_index()["ticker"].unique())
    print(f"  v2 has {len(v2_tickers)} unique tickers")

    # Load combined SP500 panel
    print("Loading SP500 combined panel...")
    sp_panel = pd.read_parquet(OUT / "prices.parquet")
    print(f"  SP500 panel: {sp_panel.shape}")
    sp_tickers = set(sp_panel.columns)
    delisted_only = sorted(sp_tickers - v2_tickers)
    print(f"  Delisted-only (need features): {len(delisted_only)}")

    # Load membership
    mem = pd.read_parquet(DATA / "sp500_pit_membership.parquet")
    mem["date"] = pd.to_datetime(mem["date"])
    sp500_ever = set(mem["ticker"].unique())

    # Compute features for delisted-only tickers, using the FULL combined panel
    # (so that SPY-relative features have SPY available)
    months_with_features = sorted(v2_big.reset_index()["asof"].unique())
    print(f"  Months to compute: {len(months_with_features)}")

    # Build sub-panel: combined panel with all SP500 tickers + SPY (for features)
    print("Building sub-panel for feature compute (all SP500 tickers + SPY)...")
    sub_panel = sp_panel  # already has all sp500 + SPY

    # For each month, compute features and keep only delisted-only tickers
    rows_for_delisted = []
    n = len(months_with_features)
    for i, m in enumerate(months_with_features):
        m_ts = pd.Timestamp(m)
        try:
            pack = compute_features(sub_panel, m_ts, min_history=504)
            df = pack.df()
            try:
                df = df.join(compute_extras(sub_panel, m_ts), how="left")
            except Exception:
                pass
            try:
                af = compute_alpha_features(sub_panel, m_ts)
                for c in af.columns:
                    if c not in df.columns:
                        df[c] = af[c]
            except Exception:
                pass
            try:
                a2 = compute_alpha2(sub_panel, m_ts)
                for c in a2.columns:
                    if c not in df.columns:
                        df[c] = a2[c]
            except Exception:
                pass
            df.index.name = "ticker"
            df = df.loc[df.index.intersection(delisted_only)]  # keep only delisted-only
            df["asof"] = m_ts
            rows_for_delisted.append(df)
            if (i + 1) % 12 == 0:
                print(f"  [{i+1}/{n}] {m_ts.date()}: {df.shape}", flush=True)
        except Exception as e:
            print(f"  skip {m_ts.date()}: {e}", flush=True)

    if rows_for_delisted:
        delisted_features = pd.concat(rows_for_delisted, axis=0)
        delisted_features = delisted_features.reset_index().set_index(["asof", "ticker"])
        # Append fwd returns from the SP500 monthly panel
        sp_monthly = sp_panel.resample("ME").last()
        sp_monthly_ret = sp_monthly.pct_change().clip(lower=-1.0, upper=2.0)
        # For each (asof, ticker), compute fwd returns
        # (mirroring the logic in v2 build_dataset / run_universe)
        rows_with_fwd = []
        # Get sorted month list and panel index
        for tname, idx in [
            ("fwd_1m_ret", 1), ("fwd_3m_ret", 3), ("fwd_6m_ret", 6), ("fwd_12m_ret", 12),
        ]:
            pass  # we'll compute below
        # Simpler: for each asof, compute fwd returns at horizons
        delisted_features = delisted_features.reset_index()
        # Build a fast asof->panel pos lookup
        panel_idx = sp_monthly.index
        for h in (1, 3, 6, 12):
            fwd_col = f"fwd_{h}m_ret"
            delisted_features[fwd_col] = np.nan
        for asof, gd in delisted_features.groupby("asof"):
            asof_t = pd.Timestamp(asof)
            pos = panel_idx.searchsorted(asof_t)
            candidates = []
            for j in (pos - 1, pos):
                if 0 <= j < len(panel_idx):
                    candidates.append((j, abs((panel_idx[j] - asof_t).days)))
            candidates.sort(key=lambda x: x[1])
            if not candidates or candidates[0][1] > 7:
                continue
            pos1 = candidates[0][0]
            p1 = sp_monthly.iloc[pos1]
            for h in (1, 3, 6, 12):
                pos_h = pos1 + h
                if pos_h >= len(panel_idx):
                    continue
                ph = sp_monthly.iloc[pos_h]
                fwd = (ph / p1 - 1).clip(lower=-1.0, upper=2.0 * h)
                # Delist
                end_pos = min(pos1 + h + 6, len(panel_idx) - 1)
                fut = sp_monthly.iloc[pos1 + h: end_pos + 1]
                any_fut = fut.notna().any()
                p1v = p1.notna()
                phn = ph.isna()
                delist_mask = p1v & phn & ~any_fut.reindex(sp_monthly.columns, fill_value=False)
                fwd[delist_mask] = -1.0
                # Apply to delisted_features rows for this asof
                mask = delisted_features["asof"] == asof
                tickers_here = delisted_features.loc[mask, "ticker"]
                fwd_series = fwd.reindex(tickers_here.values).values
                delisted_features.loc[mask, f"fwd_{h}m_ret"] = fwd_series
        delisted_features = delisted_features.set_index(["asof", "ticker"])
        print(f"  Delisted-only cross-section: {delisted_features.shape}")
        delisted_features.to_parquet(OUT / "delisted_only_cross_section.parquet")

    # Now apply membership filter to v2 + concat delisted
    print("Applying PIT membership filter to v2 cross-section...")
    v2_flat = v2_big.reset_index()
    v2_flat["asof"] = pd.to_datetime(v2_flat["asof"])
    mem_by_date = {pd.Timestamp(d): set(gd["ticker"].unique())
                   for d, gd in mem.groupby("date")}
    mem_dates = sorted(mem_by_date.keys())

    keep_rows = []
    for d, gd in v2_flat.groupby("asof"):
        d_t = pd.Timestamp(d)
        pos = np.searchsorted(mem_dates, d_t, side="right") - 1
        if pos < 0:
            continue
        members = mem_by_date[mem_dates[pos]] | {"SPY"}
        keep_rows.append(gd[gd["ticker"].isin(members)])
    v2_filt = pd.concat(keep_rows)
    print(f"  v2 filtered to PIT members: {v2_filt.shape}")

    # Combine v2 filtered + delisted features
    if rows_for_delisted:
        delisted_filt = []
        delisted_features_flat = delisted_features.reset_index()
        for d, gd in delisted_features_flat.groupby("asof"):
            d_t = pd.Timestamp(d)
            pos = np.searchsorted(mem_dates, d_t, side="right") - 1
            if pos < 0:
                continue
            members = mem_by_date[mem_dates[pos]]
            delisted_filt.append(gd[gd["ticker"].isin(members)])
        if delisted_filt:
            delisted_filt = pd.concat(delisted_filt)
            print(f"  Delisted filtered to PIT members: {delisted_filt.shape}")
            combined = pd.concat([v2_filt, delisted_filt], ignore_index=True)
        else:
            combined = v2_filt
    else:
        combined = v2_filt

    combined = combined.set_index(["asof", "ticker"])
    print(f"  Combined cross-section: {combined.shape}")
    combined.to_parquet(OUT / "panel_cross_section_v3.parquet")
    print(f"Saved {OUT / 'panel_cross_section_v3.parquet'}")


if __name__ == "__main__":
    main()
