"""Precompute & cache forward returns for every (asof, ticker) pair under
multiple exit rules. Once cached, oracle/strategy backtests become near-instant.

Output: cache/fwd_returns.parquet with multi-index (asof, ticker) and one
column per exit rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import (
    CACHE,
    DEFAULT_RULES,
    ExitRule,
    load_features,
    load_panel,
    simulate_exit,
)
from experiments.monthly_dca.backtester import month_end_dates


OUT = CACHE / "fwd_returns.parquet"


def main(start: str = "2017-12-31", end: str = "2025-12-31") -> None:
    panel = load_panel()
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    rules = list(DEFAULT_RULES)
    eval_pos = len(panel.index) - 1

    rows: list[dict] = []
    for i, asof in enumerate(months):
        try:
            feats = load_features(asof)
        except FileNotFoundError:
            continue
        pos = panel.index.searchsorted(asof)
        if pos >= len(panel.index):
            continue
        for tkr in feats.index:
            if tkr not in panel.columns:
                continue
            arr = panel[tkr].iloc[pos: eval_pos + 1].to_numpy(dtype=float)
            if len(arr) == 0 or not np.isfinite(arr[0]):
                continue
            row = {"asof": asof, "ticker": tkr, "entry_px": float(arr[0])}
            for r in rules:
                ret, days, _ = simulate_exit(arr, r)
                row[f"ret__{r.name}"] = ret
                row[f"days__{r.name}"] = int(days)
            rows.append(row)
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  [{i+1}/{len(months)}] {asof.date()} -> rows so far: {len(rows)}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, compression="zstd")
    print(f"Wrote {OUT}: {df.shape}")


if __name__ == "__main__":
    main()
