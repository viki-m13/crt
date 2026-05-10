"""Score the PIT panel using IBM Granite Tiny Time Mixers (TTM).

TTM is a tiny (805K params) time-series foundation model from IBM.
Designed for fast CPU inference. Outputs deterministic point forecasts
(no probabilistic samples).

Output: ml_preds_ttm.parquet — forecasts 96 trading days ahead from 512-day context.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

CONTEXT_LEN = 512
PREDICTION_LEN = 96


def main():
    out_path = PIT / "ml_preds_ttm.parquet"
    if out_path.exists():
        print("  already exists; skip"); return

    from tsfm_public import TinyTimeMixerForPrediction
    print("loading IBM TTM...", flush=True)
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2", num_input_channels=1
    )
    model.eval()
    print(f"loaded; {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())

    rows = []
    t0 = time.time()
    for i, d in enumerate(asofs):
        sub = panel[panel["asof"] == d]
        contexts, tickers, entry_pxs = [], [], []
        for tk in sub["ticker"]:
            if tk not in daily.columns: continue
            ts = daily[tk].dropna()
            pos = ts.index.searchsorted(d, side="right") - 1
            if pos < CONTEXT_LEN: continue
            window = ts.iloc[pos - CONTEXT_LEN + 1: pos + 1]
            if len(window) < CONTEXT_LEN or window.isna().any(): continue
            arr = window.values.astype(np.float32)
            contexts.append(arr)
            tickers.append(tk)
            entry_pxs.append(arr[-1])
        if len(contexts) == 0: continue
        ctx_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            out = model(past_values=ctx_tensor)
        # out.prediction_outputs: (batch, pred_len=96, channels=1)
        preds = out.prediction_outputs.squeeze(-1).cpu().numpy()  # (batch, 96)
        for j, tk in enumerate(tickers):
            entry_px = entry_pxs[j]
            # Final step forecast = end-of-horizon expected price
            final_3m = float(preds[j, 63] / entry_px - 1)  # ~3 months ahead (idx 63 ~ 64 days)
            final_full = float(preds[j, -1] / entry_px - 1)  # ~96 days ahead
            peak_pred = float(np.max(preds[j]) / entry_px - 1)
            rows.append({"asof": d, "ticker": tk,
                         "ttm_3m": final_3m,
                         "ttm_4m": final_full,
                         "ttm_peak": peak_pred})
        elapsed = time.time() - t0
        if i % 20 == 0 or i == len(asofs) - 1:
            eta = elapsed / (i + 1) * (len(asofs) - i - 1)
            print(f"  {d.date()} ({i+1}/{len(asofs)}): {len(tickers)} tk; cum={len(rows)}; "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
