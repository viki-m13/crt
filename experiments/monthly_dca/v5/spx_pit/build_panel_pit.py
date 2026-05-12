"""Phase 3.5: rebuild panel_cross_section_v3 from the augmented features.

Mirror of experiments/monthly_dca/v2/build_dataset.py:build_cross_section,
but reads features from the augmented features dir and joins against
the augmented monthly_returns_clean. Output is the input to ml_strategy
walk-forward training (Phase 4).

Inputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/features/*.parquet
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/monthly_prices_clean.parquet

Outputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/panel_cross_section_v3.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG = PIT / "augmented"
FEATURES = AUG / "features"


def build_cross_section(monthly_clean: pd.DataFrame) -> pd.DataFrame:
    months = sorted(pd.Timestamp(p.stem) for p in FEATURES.glob("*.parquet"))
    rows = []
    for k, m in enumerate(months[:-1]):
        pos = monthly_clean.index.searchsorted(m)
        if pos >= len(monthly_clean.index):
            continue
        # nearest month-end within ±7 days
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
            delist = (p1_valid & ph_nan
                      & ~any_future.reindex(monthly_clean.columns, fill_value=False))
            fwd[delist] = -1.0
            targets[f"fwd_{horizon}m_ret"] = fwd
        feats = pd.read_parquet(FEATURES / f"{m.date()}.parquet")
        if isinstance(feats.index, pd.MultiIndex):
            feats = feats.reset_index().set_index("ticker")
        out = feats.copy()
        for tname, tval in targets.items():
            out = out.join(tval.rename(tname), how="left")
        out["asof"] = m
        rows.append(out)
        if (k + 1) % 24 == 0:
            print(f"    [{k+1}/{len(months)-1}] {m.date()} ({len(out)} tickers)")
    big = pd.concat(rows, axis=0, ignore_index=False)
    big.index.name = "ticker"
    big = big.reset_index().set_index(["asof", "ticker"])
    return big


def main():
    t0 = time.time()
    print("=" * 64)
    print("Phase 3.5: rebuild panel_cross_section_v3 (augmented)")
    print("=" * 64)

    monthly_clean = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    print(f"[1] monthly_clean: {monthly_clean.shape}")

    n_features = len(list(FEATURES.glob("*.parquet")))
    print(f"[2] augmented feature files: {n_features}")

    big = build_cross_section(monthly_clean)
    print(f"[3] cross-section: {big.shape}")

    out_path = AUG / "panel_cross_section_v3.parquet"
    big.to_parquet(out_path)
    print(f"[4] saved -> {out_path}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
