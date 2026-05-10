"""Score chronos forecasts on a broader set of tickers (used for QQQ, IYW,
tech, broader universe testing). Only scores tickers in the iShares tech ETFs
or Nasdaq-100 plus any extra tickers needed.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))
from universes import QQQ_TECH, IYW, IGM_EXTRA, TECH_BROAD  # noqa: E402

WINDOW_DAYS = 252
PREDICTION_LENGTH = 64


def main():
    out_path = CACHE / "v2" / "ml_preds_chronos_broader.parquet"
    if out_path.exists():
        print(f"already exists: {out_path}")
        return

    from chronos import BaseChronosPipeline
    print("loading chronos-bolt-tiny...", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32)
    print("loaded.", flush=True)

    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"daily shape: {daily.shape}, columns(head): {list(daily.columns)[:5]}", flush=True)

    # Use the same asofs as the SP500 PIT panel; the broader universe tickers
    # are the union of broader 1811 + ETF universes + sp500_pit
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    asofs = sorted(ml["asof"].unique())
    all_tickers = sorted(set(ml["ticker"].unique()) | set(QQQ_TECH) | set(IYW) | set(IGM_EXTRA))
    all_tickers = [t for t in all_tickers if t in daily.columns]
    print(f"asofs: {len(asofs)}  tickers: {len(all_tickers)}", flush=True)

    rows = []
    t0 = time.time()
    for i, d in enumerate(asofs):
        contexts, tickers = [], []
        for tk in all_tickers:
            ts = daily[tk].dropna()
            pos = ts.index.searchsorted(d, side="right") - 1
            if pos < WINDOW_DAYS: continue
            window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
            if len(window) < WINDOW_DAYS or window.isna().any(): continue
            contexts.append(window.values.astype(np.float32))
            tickers.append(tk)

        if not contexts: continue
        all_preds = []
        for j in range(0, len(contexts), 500):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[j:j+500]]
            with torch.no_grad():
                out = pipe.predict(batch, prediction_length=PREDICTION_LENGTH)
            all_preds.append(np.asarray(out))
        preds = np.concatenate(all_preds, axis=0)

        for j, tk in enumerate(tickers):
            entry_px = contexts[j][-1]
            p50 = float(preds[j, 4, -1] / entry_px - 1)
            p70 = float(preds[j, 6, -1] / entry_px - 1)
            p90 = float(preds[j, 8, -1] / entry_px - 1)
            rows.append({"asof": d, "ticker": tk,
                         "chronos_p50_3m": p50, "chronos_p70_3m": p70,
                         "chronos_p90_3m": p90})

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(asofs) - i - 1)
        if i % 5 == 0 or i == len(asofs) - 1:
            print(f"  asof {d.date()} ({i+1}/{len(asofs)}): "
                  f"scored {len(tickers)} cum={len(rows)} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
