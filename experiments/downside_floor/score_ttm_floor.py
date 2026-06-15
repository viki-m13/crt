"""Second, INDEPENDENT HF foundation model for the ensemble: IBM Granite
Tiny Time Mixer (ibm-granite/granite-timeseries-ttm-r2).

TTM is an 805K-param MLP-Mixer — architecturally orthogonal to Chronos's
T5 encoder-decoder — and emits a deterministic POINT forecast. We use it for
what it is good at: a smooth multi-step trend extrapolation (a "momentum/
trend" read), to be blended correlation-neutrally with Chronos's risk read.

Output: ttm_floor_preds.parquet  (asof, ticker, ttm_trend_1m, ttm_trend_3m)
Runtime: ~3-5 min on CPU.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
OUT = Path(__file__).resolve().parent / "ttm_floor_preds.parquet"

CONTEXT_LEN = 512


def main():
    if OUT.exists():
        print(f"  already exists; skip ({OUT})")
        return
    from tsfm_public import TinyTimeMixerForPrediction
    print("loading IBM Granite TTM-r2 ...", flush=True)
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2", num_input_channels=1)
    model.eval()
    print(f"loaded; {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet", columns=["asof", "ticker"])
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())

    rows, t0 = [], time.time()
    for i, d in enumerate(asofs):
        sub = panel[panel["asof"] == d]
        contexts, tickers, entries = [], [], []
        for tk in sub["ticker"]:
            if tk not in daily.columns:
                continue
            ts = daily[tk].dropna()
            pos = ts.index.searchsorted(d, side="right") - 1
            if pos < CONTEXT_LEN - 1:
                continue
            window = ts.iloc[pos - CONTEXT_LEN + 1: pos + 1]
            if len(window) < CONTEXT_LEN or window.isna().any():
                continue
            contexts.append(window.values.astype(np.float32))
            tickers.append(tk)
            entries.append(float(window.values[-1]))
        if not contexts:
            continue
        preds = []
        for cs in range(0, len(contexts), 256):
            ct = torch.tensor(np.stack(contexts[cs:cs + 256]), dtype=torch.float32).unsqueeze(-1)
            with torch.no_grad():
                out = model(past_values=ct)
            preds.append(out.prediction_outputs.squeeze(-1).cpu().numpy())  # (B,96)
        preds = np.concatenate(preds, axis=0)
        entries = np.asarray(entries)
        for j, tk in enumerate(tickers):
            rows.append({"asof": d, "ticker": tk,
                         "ttm_trend_1m": float(preds[j, 20] / entries[j] - 1),
                         "ttm_trend_3m": float(preds[j, 62] / entries[j] - 1)})
        if i % 20 == 0 or i == len(asofs) - 1:
            el = time.time() - t0
            print(f"  {d.date()} ({i+1}/{len(asofs)}) tk={len(tickers)} cum={len(rows)} "
                  f"el={el:.0f}s eta={el/(i+1)*(len(asofs)-i-1):.0f}s", flush=True)

    pd.DataFrame(rows).to_parquet(OUT, index=False)
    print(f"saved {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
