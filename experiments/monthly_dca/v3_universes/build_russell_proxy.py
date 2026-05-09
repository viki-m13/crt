"""
Build a Russell 1000 PROXY using existing US panel.

We don't have actual Russell 1000 historical membership. Approximation:
at each month-end T, restrict to the top-N tickers by a market-size proxy.

Since we have no volume data, we use a composite proxy:
  size_proxy(t, T) = log1p(price(t, T)) * sqrt(history_length(t, T))

This biases toward tickers that:
  1. Have higher absolute price levels (large companies tend to have $20-$500
     prices, microcaps have $1-$10)
  2. Have longer trading histories (established companies)

This is imperfect but gives a Russell 1000-like cohort that excludes the
microcap tail of our 1,833-ticker universe. Documented as a proxy in REPORT.

Saves a PIT membership table at cache/v3_universes/russell1000_proxy/membership.parquet
plus the (link to existing) panel.

Run: python3 -m experiments.monthly_dca.v3_universes.build_russell_proxy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
OUT = CACHE / "v3_universes" / "russell1000_proxy"
OUT.mkdir(parents=True, exist_ok=True)


def main(top_n: int = 1000):
    panel = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"Panel: {panel.shape}")

    # Resample to month-end for membership snapshots
    monthly = panel.resample("ME").last()
    print(f"Monthly: {monthly.shape}")

    rows = []
    for d in monthly.index:
        # Compute size proxy at this date
        # 1. Last-N-day average price (smooth out single-day noise)
        end_pos = monthly.index.get_loc(d)
        # Use last 6 months of price for stability
        lookback_months = 6
        start_pos = max(0, end_pos - lookback_months + 1)
        recent_window = monthly.iloc[start_pos: end_pos + 1]
        if len(recent_window) < 1:
            continue
        recent_mean_px = recent_window.mean()
        # 2. History length: count of valid prices for each ticker up to date d
        history_len = panel.loc[:d].notna().sum()
        # Composite proxy
        valid = recent_mean_px.notna() & (history_len >= 252)  # need ≥1y history
        proxy = np.log1p(recent_mean_px[valid]) * np.sqrt(history_len[valid])
        if len(proxy) == 0:
            continue
        # Top-N
        top = proxy.nlargest(top_n).index
        for t in top:
            rows.append({"date": d.date(), "ticker": t})

    mem = pd.DataFrame(rows)
    print(f"Membership rows: {len(mem)}")
    print(f"  Per-month size: mean {mem.groupby('date').size().mean():.0f}, "
          f"min {mem.groupby('date').size().min()}, "
          f"max {mem.groupby('date').size().max()}")
    print(f"  Unique tickers ever in: {mem['ticker'].nunique()}")
    mem.to_parquet(OUT / "membership.parquet", index=False)

    # We can re-use the existing prices_extended.parquet directly via a symlink/copy
    # For consistency with run_universe, save a thin reference panel
    # (just copy or symlink to avoid duplication)
    src = CACHE / "prices_extended.parquet"
    dst = OUT / "prices.parquet"
    if dst.exists():
        dst.unlink()
    # Symlink to save space
    import os
    try:
        os.symlink(src, dst)
        print(f"Symlinked prices.parquet -> {src}")
    except OSError:
        # Fallback: copy (small enough to be OK)
        import shutil
        shutil.copy(src, dst)
        print(f"Copied prices.parquet from {src}")

    coverage = {
        "top_n_per_month": top_n,
        "n_membership_rows": len(mem),
        "unique_tickers_ever": int(mem['ticker'].nunique()),
        "n_months": int(mem.groupby('date').ngroups),
        "date_range": f"{mem['date'].min()} - {mem['date'].max()}",
    }
    with open(OUT / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"\n=== Coverage ===")
    for k, v in coverage.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main(top_n=1000)
