"""Score the PIT panel using Chronos-Bolt-Tiny zero-shot forecasts.

Chronos-Bolt is Amazon's optimized variant — 250x faster than original
chronos-t5 with comparable accuracy.  Tractable on CPU.

For each (asof, ticker), forecast next 64 trading days (~3 months).
Use the median (p50) and p90 forecast as cross-sectional features.
"""
from __future__ import annotations
import time, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

WINDOW_DAYS = 252  # 1 year of context
PREDICTION_LENGTH = 64  # ~3 months ahead


def main():
    out_path = PIT / "ml_preds_chronos.parquet"
    if out_path.exists():
        print("  already exists; skip"); return

    from chronos import BaseChronosPipeline
    print("loading chronos-bolt-tiny...", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("loaded.", flush=True)

    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())

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
            if pos < WINDOW_DAYS: continue
            window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
            if len(window) < WINDOW_DAYS or window.isna().any():
                continue
            arr = window.values.astype(np.float32)
            contexts.append(arr)
            tickers.append(tk)

        if len(contexts) == 0: continue
        # Run in batches of 500
        all_preds = []
        for j in range(0, len(contexts), 500):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[j:j+500]]
            with torch.no_grad():
                out = pipe.predict(batch, prediction_length=PREDICTION_LENGTH)
            all_preds.append(np.asarray(out))
        preds = np.concatenate(all_preds, axis=0)  # (N, 9, 64)
        # Quantile levels: by default chronos returns 9 quantiles (10%-90%, step 10%)
        # We want median (p50, idx=4) and upper (p70, idx=6 or p80, idx=7)
        for j, tk in enumerate(tickers):
            entry_px = contexts[j][-1]
            p50_final = float(preds[j, 4, -1] / entry_px - 1)  # median 3m return
            p70_final = float(preds[j, 6, -1] / entry_px - 1)  # 70th pct
            p90_final = float(preds[j, 8, -1] / entry_px - 1)  # 90th pct
            # Path metrics: max drawdown along the path, peak-to-end
            p50_path = preds[j, 4, :] / entry_px - 1
            peak = p50_path.max()
            p50_max = float(peak)
            rows.append({"asof": d, "ticker": tk,
                         "chronos_p50_3m": p50_final,
                         "chronos_p70_3m": p70_final,
                         "chronos_p90_3m": p90_final,
                         "chronos_p50_peak": p50_max})
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(asofs) - i - 1)
        if i % 10 == 0 or i == len(asofs) - 1:
            print(f"  asof {d.date()} ({i+1}/{len(asofs)}): scored {len(tickers)}; cum={len(rows)}; "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
