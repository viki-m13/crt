"""
Build the v2 dataset:
- panel_cross_section_v3.parquet: ticker × month × features + multi-horizon fwd returns
- monthly_prices_clean.parquet: clean month-end prices (data-error filtered)
- monthly_returns_clean.parquet: clean month-over-month returns (capped to [-100%, +200%])
- bad_data_tickers.json: blacklist of tickers with data quality issues

Run from the repo root:
    python3 -m experiments.monthly_dca.v2.build_dataset

This is the single source of truth for the v2 strategy. Persists everything
to experiments/monthly_dca/cache/v2/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
OUT = CACHE / "v2"
FEATURES_DIR = CACHE / "features"


def _detect_bad_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Detect ticker-reuse / data-error months.

    Signature: 80%+ drop followed by 200%+ surge within 3 months,
    or 200%+ surge followed by 80%+ drop within 3 months.

    Returns a boolean mask (same shape as monthly) where True = bad cell.
    Around each detected bad signal, a 6-month window (±3 months) is masked.
    """
    mret = monthly.pct_change()

    extreme_drops = mret < -0.80
    mret_max_3m_ahead = mret.rolling(3).max().shift(-3)
    sig1 = extreme_drops & (mret_max_3m_ahead > 2.0)

    extreme_surge = mret > 2.0
    mret_min_3m_ahead = mret.rolling(3).min().shift(-3)
    sig2 = extreme_surge & (mret_min_3m_ahead < -0.8)

    reuse = sig1 | sig2
    counts = reuse.sum().sort_values(ascending=False)

    mask_bad = pd.DataFrame(False, index=monthly.index, columns=monthly.columns)
    for tk in counts[counts > 0].index:
        sig = reuse[tk]
        bad_dates = sig[sig].index
        for d in bad_dates:
            i = monthly.index.get_loc(d)
            lo = max(0, i - 3)
            hi = min(len(monthly.index) - 1, i + 3)
            for j in range(lo, hi + 1):
                mask_bad.iat[j, monthly.columns.get_loc(tk)] = True
    return mask_bad


def build_clean_monthly() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    panel = pd.read_parquet(CACHE / "prices_extended.parquet")
    monthly = panel.resample("ME").last()
    mask_bad = _detect_bad_months(monthly)
    monthly_clean = monthly.where(~mask_bad)
    mret_clean = monthly_clean.pct_change().clip(lower=-1.0, upper=2.0)
    bad_list = sorted(mask_bad.any(axis=0)[mask_bad.any(axis=0)].index.tolist())
    return monthly_clean, mret_clean, bad_list


def build_cross_section(monthly_clean: pd.DataFrame) -> pd.DataFrame:
    """Build the cross-section: per-(month, ticker) features + multi-horizon fwd returns."""
    months = sorted(pd.Timestamp(p.stem) for p in FEATURES_DIR.glob("*.parquet"))
    rows = []
    for k, m in enumerate(months[:-1]):
        pos = monthly_clean.index.searchsorted(m)
        if pos >= len(monthly_clean.index):
            continue
        # Pick the panel-month-end nearest to feature month (within ±7d)
        candidates = []
        for j in (pos - 1, pos):
            if 0 <= j < len(monthly_clean.index):
                d = monthly_clean.index[j]
                candidates.append((j, abs((d - m).days)))
        candidates.sort(key=lambda x: x[1])
        if not candidates or candidates[0][1] > 7:
            continue
        pos1 = candidates[0][0]
        if pos1 + 1 >= len(monthly_clean.index):
            continue
        p1 = monthly_clean.iloc[pos1]
        targets = {}
        for horizon in (1, 3, 6, 12):
            pos_h = pos1 + horizon
            if pos_h >= len(monthly_clean.index):
                break
            ph = monthly_clean.iloc[pos_h]
            cap = 2.0 * horizon
            fwd = (ph / p1 - 1).clip(lower=-1.0, upper=cap)
            # Delist: p1 valid, ph NaN, no recovery in 6m -> -1
            end_pos = min(pos1 + horizon + 6, len(monthly_clean.index) - 1)
            future_window = monthly_clean.iloc[pos1 + horizon: end_pos + 1]
            any_future = future_window.notna().any()
            p1_valid = p1.notna()
            ph_nan = ph.isna()
            delist = p1_valid & ph_nan & ~any_future.reindex(monthly_clean.columns, fill_value=False)
            fwd[delist] = -1.0
            targets[f"fwd_{horizon}m_ret"] = fwd
        feats = pd.read_parquet(FEATURES_DIR / f"{m.date()}.parquet")
        if isinstance(feats.index, pd.MultiIndex):
            feats = feats.reset_index().set_index("ticker")
        out = feats.copy()
        for tname, tval in targets.items():
            out = out.join(tval.rename(tname), how="left")
        out["asof"] = m
        rows.append(out)
    big = pd.concat(rows, axis=0, ignore_index=False)
    big.index.name = "ticker"
    big = big.reset_index().set_index(["asof", "ticker"])
    return big


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Building clean monthly panel ===")
    monthly_clean, mret_clean, bad_list = build_clean_monthly()
    monthly_clean.to_parquet(OUT / "monthly_prices_clean.parquet")
    mret_clean.to_parquet(OUT / "monthly_returns_clean.parquet")
    with open(OUT / "bad_data_tickers.json", "w") as f:
        json.dump(bad_list, f, indent=2)
    print(f"  Bad-data tickers: {len(bad_list)} -> {bad_list}")

    print("=== Building cross-section ===")
    big = build_cross_section(monthly_clean)
    print(f"  Shape: {big.shape}")
    big.to_parquet(OUT / "panel_cross_section_v3.parquet")
    print("Saved.")


if __name__ == "__main__":
    main()
