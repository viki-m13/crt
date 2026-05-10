"""Score the BROADER 1833-ticker universe with Chronos-bolt-tiny.

This generates ml_preds_chronos_broader.parquet — used to test that the
Chronos confidence filter generalises beyond PIT S&P 500.
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

WINDOW_DAYS = 252
PREDICTION_LENGTH = 64

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}


def main():
    out_path = PIT / "ml_preds_chronos_broader.parquet"
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
    feature_files = {pd.Timestamp(p.stem): p for p in (CACHE / "features").glob("*.parquet")}
    asofs = sorted(feature_files.keys())

    rows = []
    t0 = time.time()
    for i, d in enumerate(asofs):
        # Get all tickers with features at this asof
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        contexts = []
        tickers = []
        for tk in feat.index:
            if tk not in daily.columns: continue
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
        # Run in batches
        all_preds = []
        for j in range(0, len(contexts), 500):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[j:j+500]]
            with torch.no_grad():
                out = pipe.predict(batch, prediction_length=PREDICTION_LENGTH)
            all_preds.append(np.asarray(out))
        preds = np.concatenate(all_preds, axis=0)
        for j, tk in enumerate(tickers):
            entry_px = contexts[j][-1]
            p70_final = float(preds[j, 6, -1] / entry_px - 1)
            rows.append({"asof": d, "ticker": tk, "chronos_p70_3m": p70_final})
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(asofs) - i - 1)
        if i % 10 == 0 or i == len(asofs) - 1:
            print(f"  {d.date()} ({i+1}/{len(asofs)}): {len(tickers)} tk; cum={len(rows)}; "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
