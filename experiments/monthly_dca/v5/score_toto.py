"""Score the PIT panel using DataDog Toto-Open-Base-1.0 (151M params).

Toto is a probabilistic time-series foundation model. Use median forecast.
Output: ml_preds_toto.parquet
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

CONTEXT_LEN = 252
PREDICTION_LEN = 64
NUM_SAMPLES = 20


def main():
    out_path = PIT / "ml_preds_toto.parquet"
    if out_path.exists():
        print("  already exists; skip"); return

    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries

    print("loading Toto-Open-Base-1.0...", flush=True)
    model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    model.eval()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    fc = TotoForecaster(model.model)

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
            contexts.append(arr); tickers.append(tk); entry_pxs.append(arr[-1])
        if len(contexts) == 0: continue
        # Batch in chunks
        batch_size = 50
        for chunk_start in range(0, len(contexts), batch_size):
            chunk = contexts[chunk_start:chunk_start+batch_size]
            chunk_tk = tickers[chunk_start:chunk_start+batch_size]
            chunk_ep = entry_pxs[chunk_start:chunk_start+batch_size]
            series = torch.tensor(np.stack(chunk), dtype=torch.float32)
            timestamps = torch.zeros(len(chunk), CONTEXT_LEN, dtype=torch.long)
            interval = torch.full((len(chunk),), 86400, dtype=torch.long)
            mts = MaskedTimeseries(
                series=series, padding_mask=torch.ones_like(series).bool(),
                id_mask=torch.zeros_like(series).long(),
                timestamp_seconds=timestamps, time_interval_seconds=interval,
            )
            with torch.no_grad():
                f = fc.forecast(mts, prediction_length=PREDICTION_LEN, num_samples=NUM_SAMPLES, samples_per_batch=NUM_SAMPLES)
            # f.median: (1, batch, pred_len)
            med = f.median.squeeze(0).cpu().numpy()  # (batch, pred_len)
            samples_arr = f.samples.cpu().numpy() if hasattr(f, 'samples') else None
            for j, (tk, ep) in enumerate(zip(chunk_tk, chunk_ep)):
                p50_3m = float(med[j, -1] / ep - 1)
                rec = {"asof": d, "ticker": tk, "toto_p50_3m": p50_3m}
                if samples_arr is not None:
                    p_arr = samples_arr[:, j, -1]  # (num_samples,)
                    p70 = float(np.percentile(p_arr, 70) / ep - 1)
                    p90 = float(np.percentile(p_arr, 90) / ep - 1)
                    rec["toto_p70_3m"] = p70
                    rec["toto_p90_3m"] = p90
                rows.append(rec)
        elapsed = time.time() - t0
        if i % 5 == 0 or i == len(asofs) - 1:
            eta = elapsed / (i + 1) * (len(asofs) - i - 1)
            print(f"  {d.date()} ({i+1}/{len(asofs)}): {len(tickers)} tk; cum={len(rows)}; "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
