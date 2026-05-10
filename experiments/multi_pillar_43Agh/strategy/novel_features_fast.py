"""Pillar 3 — Novel features (fast/vectorised).

Computes 3 fast features per (asof, ticker):
  spy_corr_60d        : 60-day rolling correlation with SPY (vectorised)
  price_persistence   : 60-day lag-1 autocorrelation, mapped to [0,1]
  abs_skew_60d        : abs of 60-day return skew (proxy for tail asymmetry)

All vectorised over the panel. ~100x faster than the per-ticker GPD version.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"
OUT = ROOT / "experiments" / "multi_pillar_43Agh" / "data" / "novel_features"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("[load] daily prices ...")
    prices = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"  shape={prices.shape}")
    rets = prices.pct_change()

    asofs = sorted(pd.Timestamp(p.stem) for p in FEATURES_DIR.glob("*.parquet"))
    print(f"computing for {len(asofs)} asofs (vectorised) ...")

    # Pre-compute SPY returns
    spy_ret = rets.get("SPY")
    if spy_ret is None:
        print("ERROR: no SPY column in prices_extended")
        return

    n_built = 0
    for ao in asofs:
        out_f = OUT / f"{pd.Timestamp(ao).date()}.parquet"
        if out_f.exists():
            continue
        # Find nearest <= ao trading day in rets
        avail = rets.index[rets.index <= ao]
        if len(avail) == 0:
            continue
        last = avail.max()
        pos = int(rets.index.get_loc(last))
        lo = max(0, pos - 60 + 1)
        block = rets.iloc[lo: pos + 1]
        if len(block) < 30:
            continue
        block_clean = block.dropna(axis=1, thresh=int(0.7 * len(block)))
        if len(block_clean.columns) == 0:
            continue
        # SPY corr — vectorised
        spy_block = block_clean["SPY"] if "SPY" in block_clean.columns else block["SPY"].reindex(block_clean.index)
        # corr of each col with spy_block
        spy_centered = spy_block - spy_block.mean()
        col_centered = block_clean.subtract(block_clean.mean(), axis=1)
        cov = (col_centered.multiply(spy_centered, axis=0)).sum() / (len(block_clean) - 1)
        denom = (block_clean.std() * spy_block.std() + 1e-12)
        corr = (cov / denom).rename("spy_corr_60d")
        # Lag-1 autocorr — vectorised
        a = block_clean.iloc[:-1].reset_index(drop=True)
        b = block_clean.iloc[1:].reset_index(drop=True)
        a_c = a.subtract(a.mean(), axis=1)
        b_c = b.subtract(b.mean(), axis=1)
        cov_ab = (a_c * b_c).sum() / (len(a) - 1)
        denom_ab = (a.std() * b.std() + 1e-12)
        ac = (cov_ab / denom_ab).clip(-1, 1)
        persistence = (0.5 + 0.5 * ac).rename("price_persistence")
        # 60-day return skew (abs) — pandas builtin
        sk = block_clean.skew().abs().rename("abs_skew_60d")
        out = pd.concat([corr, persistence, sk], axis=1)
        out.index.name = "ticker"
        out.to_parquet(out_f)
        n_built += 1
        if n_built % 50 == 0:
            print(f"  built {n_built}/{len(asofs)}")
    print(f"done; built {n_built} parquets total")


if __name__ == "__main__":
    main()
