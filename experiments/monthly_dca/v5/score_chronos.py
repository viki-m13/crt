"""Score the PIT panel using a zero-shot Chronos forecaster from HuggingFace.

Chronos is Amazon's state-of-the-art time-series foundation model.  We use it
to forecast the next 6 months of returns for each stock and use the median
forecast (or upper quantile) as the score.

Uses the smallest model variant (chronos-t5-small, ~46M params) for CPU speed.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

WINDOW_DAYS = 180  # 9 months of daily data
PREDICTION_LENGTH_DAYS = 126  # ~6 months ahead
NUM_SAMPLES = 20  # number of probabilistic samples


def main():
    out_path = PIT / "ml_preds_chronos.parquet"
    if out_path.exists():
        print("  already exists; skip"); return
    print("loading Chronos model (chronos-t5-small)...", flush=True)
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("loaded.", flush=True)

    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())
    # Use only every 6th month to speed up (panel still has 268 monthly asofs; chronos is slow)
    # Actually still 280 ms; let me sample
    sampled_asofs = asofs[::3]  # every 3 months
    print(f"sampled {len(sampled_asofs)} asofs out of {len(asofs)}", flush=True)

    rows = []
    t0 = time.time()
    for i, d in enumerate(sampled_asofs):
        sub = panel[panel["asof"] == d]
        # Build context series for each ticker
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
            # Normalize to start = 1
            arr = window.values / window.values[0]
            contexts.append(arr)
            tickers.append(tk)

        if len(contexts) == 0:
            continue
        # Batch predict
        # Chronos expects a list of 1D arrays (or torch tensors)
        ctx_tensors = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        try:
            with torch.no_grad():
                samples = pipe.predict(
                    ctx_tensors,
                    prediction_length=PREDICTION_LENGTH_DAYS,
                    num_samples=NUM_SAMPLES,
                )
            # samples: (batch, num_samples, prediction_length)
            arr = np.asarray(samples)
            for j, tk in enumerate(tickers):
                pred = arr[j]  # (num_samples, prediction_length)
                # Final value at end of horizon: pred / start_value (which was 1 due to normalization)
                final_vals = pred[:, -1]  # (num_samples,)
                # Forecast median 6m return
                ret_med = float(np.median(final_vals)) - 1.0
                ret_p75 = float(np.percentile(final_vals, 75)) - 1.0
                ret_p90 = float(np.percentile(final_vals, 90)) - 1.0
                rows.append({"asof": d, "ticker": tk,
                             "chronos_p50": ret_med,
                             "chronos_p75": ret_p75,
                             "chronos_p90": ret_p90})
        except Exception as e:
            print(f"  error at {d}: {e}", flush=True)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(sampled_asofs) - i - 1)
        print(f"  asof {d.date()}: scored {len(tickers)} tickers; cum={len(rows)}; "
              f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
