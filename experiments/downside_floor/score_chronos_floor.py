"""Chronos-Bolt DOWNSIDE forecaster for the Floor picker.

The deployed v5 strategy already uses amazon/chronos-bolt-tiny, but only
keeps the *final* 3-month return quantiles (p50/p70/p90) and a p50 peak.
That throws away exactly the information the "don't go below my buy price"
objective needs: the LOWER quantiles of the forward PATH.

This script re-scores the PIT panel with the same SOTA HuggingFace
time-series foundation model (amazon/chronos-bolt-tiny, Apache-2.0) but
keeps the full 9-quantile forecast path, and converts it into
downside-risk forecasts relative to the entry (purchase) price:

  chr_exp_uw_frac_H   model's expected fraction of the next H days spent
                      below the entry price  = mean_t P(price_t < entry)
  chr_p_below_end_H   P(price at +H < entry)
  chr_trough_q10_H    worst 10th-percentile path level / entry - 1
  chr_trough_q30_H    worst 30th-percentile path level / entry - 1
  chr_p50_end_H       median forecast return at +H (context/sanity)

P(price_t < entry) is read off the 9 forecast quantiles by linear
interpolation of the implied CDF (with linear tail extrapolation).

Zero-shot foundation model => no training, no look-ahead leakage: each
forecast sees only the trailing 252 closes up to asof.

Runtime: ~5-8 min on CPU for the full 254-asof panel.
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
OUT = Path(__file__).resolve().parent / "chronos_floor_preds.parquet"

WINDOW_DAYS = 252
PRED_LEN = 126                       # 6 months ahead
CUTS = {"1m": 21, "3m": 63, "6m": 126}
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def prob_below(Q: np.ndarray, entry: np.ndarray) -> np.ndarray:
    """P(price < entry) from quantile prices.

    Q:     (..., 9) quantile prices (monotone increasing in last axis)
    entry: (...,)    threshold price, broadcastable to Q[..., 0]
    Returns P in [0, 1] with shape Q.shape[:-1], via linear CDF interp
    and linear tail extrapolation off the nearest quantile gap.
    """
    e = np.asarray(entry)[..., None]                  # (..., 1)
    k = (Q < e).sum(axis=-1)                           # # quantiles below entry, 0..9
    hi = np.clip(k, 1, len(LEVELS) - 1)
    lo = hi - 1
    qlo = np.take_along_axis(Q, lo[..., None], axis=-1)[..., 0]
    qhi = np.take_along_axis(Q, hi[..., None], axis=-1)[..., 0]
    plo = LEVELS[lo]
    phi = LEVELS[hi]
    frac = (e[..., 0] - qlo) / (qhi - qlo + 1e-12)     # <0 or >1 in the tails -> extrapolates
    p = plo + frac * (phi - plo)
    return np.clip(p, 0.0, 1.0)


def main():
    if OUT.exists():
        print(f"  already exists; skip ({OUT})")
        return
    from chronos import BaseChronosPipeline
    print("loading amazon/chronos-bolt-tiny ...", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", device_map="cpu", dtype=torch.float32)

    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet", columns=["asof", "ticker"])
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())
    print(f"asofs: {len(asofs)}", flush=True)

    rows = []
    t0 = time.time()
    for i, d in enumerate(asofs):
        sub = panel[panel["asof"] == d]
        contexts, tickers, entries = [], [], []
        for tk in sub["ticker"]:
            if tk not in daily.columns:
                continue
            ts = daily[tk].dropna()
            pos = ts.index.searchsorted(d, side="right") - 1
            if pos < WINDOW_DAYS - 1:
                continue
            window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
            if len(window) < WINDOW_DAYS or window.isna().any():
                continue
            contexts.append(torch.tensor(window.values.astype(np.float32)))
            tickers.append(tk)
            entries.append(float(window.values[-1]))
        if not contexts:
            continue
        entries = np.asarray(entries)

        # batched quantile forecast -> (N, PRED_LEN, 9)
        qall = []
        for j in range(0, len(contexts), 400):
            with torch.no_grad():
                q, _ = pipe.predict_quantiles(
                    contexts[j:j + 400], prediction_length=PRED_LEN,
                    quantile_levels=list(LEVELS))
            qall.append(np.asarray(q, dtype=np.float64))
        Q = np.concatenate(qall, axis=0)                    # (N, PRED_LEN, 9)

        # P(price_t < entry) for every forecast day
        pbelow = prob_below(Q, entries[:, None])            # (N, PRED_LEN)
        ret_q10 = Q[:, :, 0] / entries[:, None] - 1.0        # 10th pct path return
        ret_q30 = Q[:, :, 2] / entries[:, None] - 1.0
        ret_q50 = Q[:, :, 4] / entries[:, None] - 1.0

        for n, tk in enumerate(tickers):
            rec = {"asof": d, "ticker": tk}
            for name, c in CUTS.items():
                rec[f"chr_exp_uw_frac_{name}"] = float(pbelow[n, :c].mean())
                rec[f"chr_p_below_end_{name}"] = float(pbelow[n, c - 1])
                rec[f"chr_trough_q10_{name}"] = float(ret_q10[n, :c].min())
                rec[f"chr_trough_q30_{name}"] = float(ret_q30[n, :c].min())
                rec[f"chr_p50_end_{name}"] = float(ret_q50[n, c - 1])
            rows.append(rec)

        if i % 20 == 0 or i == len(asofs) - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (len(asofs) - i - 1)
            print(f"  {d.date()} ({i+1}/{len(asofs)}) names={len(tickers)} "
                  f"cum={len(rows)} el={el:.0f}s eta={eta:.0f}s", flush=True)

    out = pd.DataFrame(rows).sort_values(["asof", "ticker"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {out.shape} -> {OUT}")


if __name__ == "__main__":
    main()
