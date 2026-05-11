"""Phase 5c: score Chronos-Bolt-Tiny on the augmented PIT panel.

Identical model + windowing as experiments/monthly_dca/v5/score_chronos_bolt.py
but reads the augmented sp500_pit_panel and augmented daily prices, writes
to augmented/ml_preds_chronos.parquet.

Inputs:
  augmented/sp500_pit_panel.parquet      (joined panel — defines asofs & tickers)
  prices_extended_pit.parquet            (augmented daily prices)

Output:
  augmented/ml_preds_chronos.parquet     (asof, ticker, chronos_p50/p70/p90_3m, chronos_p50_peak)

Wallclock: ~60-90 min on CPU for ~280 asofs x ~500 tickers each.
"""
from __future__ import annotations
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG = PIT / "augmented"

WINDOW_DAYS = 252       # 1y daily context
PREDICTION_LENGTH = 64  # ~3 months ahead


def main():
    out_path = AUG / "ml_preds_chronos.parquet"
    if out_path.exists():
        print(f"  already exists; skip ({out_path})")
        return

    from chronos import BaseChronosPipeline
    print("loading chronos-bolt-tiny...", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("loaded.", flush=True)

    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet")
    print(f"daily: {daily.shape}", flush=True)

    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())
    print(f"asofs in augmented panel: {len(asofs)}", flush=True)

    rows = []
    t0 = time.time()
    for i, d in enumerate(asofs):
        sub = panel[panel["asof"] == d]
        contexts = []
        tickers = []
        for tk in sub["ticker"]:
            if tk not in daily.columns:
                continue
            ts = daily[tk].dropna()
            pos = ts.index.searchsorted(d, side="right") - 1
            if pos < WINDOW_DAYS:
                continue
            window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
            if len(window) < WINDOW_DAYS or window.isna().any():
                continue
            arr = window.values.astype(np.float32)
            contexts.append(arr)
            tickers.append(tk)

        if len(contexts) == 0:
            continue
        # Batched inference
        all_preds = []
        for j in range(0, len(contexts), 500):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[j:j + 500]]
            with torch.no_grad():
                out = pipe.predict(batch, prediction_length=PREDICTION_LENGTH)
            all_preds.append(np.asarray(out))
        preds = np.concatenate(all_preds, axis=0)  # (N, 9, 64)
        for j, tk in enumerate(tickers):
            entry_px = contexts[j][-1]
            p50_final = float(preds[j, 4, -1] / entry_px - 1)
            p70_final = float(preds[j, 6, -1] / entry_px - 1)
            p90_final = float(preds[j, 8, -1] / entry_px - 1)
            p50_path = preds[j, 4, :] / entry_px - 1
            p50_max = float(p50_path.max())
            rows.append({
                "asof": d, "ticker": tk,
                "chronos_p50_3m": p50_final,
                "chronos_p70_3m": p70_final,
                "chronos_p90_3m": p90_final,
                "chronos_p50_peak": p50_max,
            })
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(asofs) - i - 1)
        if i % 10 == 0 or i == len(asofs) - 1:
            print(f"  asof {d.date()} ({i+1}/{len(asofs)}): scored {len(tickers)}; "
                  f"cum={len(rows)}; elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
                  flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"saved {out_df.shape} to {out_path}")


if __name__ == "__main__":
    main()
