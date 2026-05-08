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


def main(start: str = "2002-01-01", end: str = "2099-01-01") -> None:
    panel = load_panel()
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    rules = list(DEFAULT_RULES)
    eval_pos = len(panel.index) - 1

    rows: list[dict] = []
    # Checkpoint: if a partial parquet exists, load it and skip already-done months
    ckpt = OUT.with_suffix(".ckpt.parquet")
    done_months: set[pd.Timestamp] = set()
    if ckpt.exists():
        prior = pd.read_parquet(ckpt)
        done_months = set(pd.to_datetime(prior["asof"].unique()))
        rows = prior.to_dict(orient="records")
        print(f"  Resuming from checkpoint: {len(rows)} rows, {len(done_months)} months already done")

    for i, asof in enumerate(months):
        if asof in done_months:
            continue
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
            # Checkpoint every 12 months so partial work isn't lost
            try:
                pd.DataFrame(rows).to_parquet(ckpt, compression="zstd")
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, compression="zstd")
    if ckpt.exists():
        ckpt.unlink()
    print(f"Wrote {OUT}: {df.shape}")


if __name__ == "__main__":
    main()
