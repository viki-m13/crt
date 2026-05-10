"""Score the PIT panel using Salesforce Moirai-1.0-R-small (14M params).

Moirai is a probabilistic time-series foundation model.
Output: ml_preds_moirai.parquet
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

CONTEXT_LEN = 256
PREDICTION_LEN = 64
PATCH_SIZE = 32
NUM_SAMPLES = 20


def main():
    out_path = PIT / "ml_preds_moirai.parquet"
    if out_path.exists():
        print("  already exists; skip"); return

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    print("loading moirai-1.0-R-small...", flush=True)
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
    model = MoiraiForecast(module=module, prediction_length=PREDICTION_LEN,
                           context_length=CONTEXT_LEN, patch_size=PATCH_SIZE,
                           num_samples=NUM_SAMPLES, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    model.eval()
    print(f"params: {sum(p.numel() for p in module.parameters()):,}", flush=True)

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
        # Batch
        B = 200
        for cs in range(0, len(contexts), B):
            chunk = contexts[cs:cs+B]
            chunk_tk = tickers[cs:cs+B]
            chunk_ep = entry_pxs[cs:cs+B]
            target = torch.tensor(np.stack(chunk), dtype=torch.float32).unsqueeze(-1)
            mask = torch.ones_like(target).bool()
            is_pad = torch.zeros(target.size(0), CONTEXT_LEN).bool()
            with torch.no_grad():
                pred = model(past_target=target, past_observed_target=mask, past_is_pad=is_pad)
            # pred: (batch, num_samples, pred_len)
            pred_arr = pred.cpu().numpy()
            for j, (tk, ep) in enumerate(zip(chunk_tk, chunk_ep)):
                p_arr = pred_arr[j, :, -1]  # samples at end of horizon
                p50 = float(np.median(p_arr) / ep - 1)
                p70 = float(np.percentile(p_arr, 70) / ep - 1)
                p90 = float(np.percentile(p_arr, 90) / ep - 1)
                rows.append({"asof": d, "ticker": tk,
                             "moirai_p50_3m": p50,
                             "moirai_p70_3m": p70,
                             "moirai_p90_3m": p90})
        elapsed = time.time() - t0
        if i % 10 == 0 or i == len(asofs) - 1:
            eta = elapsed / (i + 1) * (len(asofs) - i - 1)
            print(f"  {d.date()} ({i+1}/{len(asofs)}): {len(tickers)} tk; cum={len(rows)}; "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"saved {out.shape} to {out_path}")


if __name__ == "__main__":
    main()
