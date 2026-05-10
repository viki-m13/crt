"""Cron daily refresh for v5 strategy (v3 + Chronos filter).

Runs after the daily v3 cron — appends Chronos predictions for the latest
month to the live data.

Steps:
  1. Compute v3 ml_3plus6 score (from existing ml_preds_live.parquet via cron)
  2. Run Chronos-bolt-tiny on the LATEST month for current S&P 500 members
  3. Build v5 picks = top-3 by ml_3plus6 score among stocks with Chronos rank >= 0.4
  4. Update live_state in webapp data.json

Designed to be additive on top of cron_daily_refresh.py (v3).  The Chronos
score is only computed for ~500 stocks at the latest asof, runs in <30s on CPU.
"""
from __future__ import annotations
import json, sys, time
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
CHRONOS_MODEL = "amazon/chronos-bolt-tiny"
CHRONOS_FILTER_QUANTILE = 0.4

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}


def main():
    print("=== v5 daily refresh ===", flush=True)
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    last_asof = panel["asof"].max()
    print(f"latest asof: {last_asof.date()}", flush=True)
    sub = panel[panel["asof"] == last_asof].copy()

    # v3 score from live preds
    live_preds = pd.read_parquet(V2 / "ml_preds_live.parquet")
    live_preds["asof"] = pd.to_datetime(live_preds["asof"])
    live_at_asof = live_preds[live_preds["asof"] == last_asof]
    if len(live_at_asof) == 0:
        # fall back to wf preds
        wf = pd.read_parquet(V2 / "ml_preds_v2.parquet")
        wf["asof"] = pd.to_datetime(wf["asof"])
        live_at_asof = wf[wf["asof"] == last_asof]
    sub = sub.merge(live_at_asof[["asof","ticker","pred_3m","pred_6m"]], on=["asof","ticker"], how="left")
    sub["v3_score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2

    # Chronos forecast
    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"loading {CHRONOS_MODEL}...", flush=True)
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)

    contexts, tickers = [], []
    for tk in sub["ticker"]:
        if tk in EXCLUDE or tk not in daily.columns: continue
        ts = daily[tk].dropna()
        pos = ts.index.searchsorted(last_asof, side="right") - 1
        if pos < WINDOW_DAYS: continue
        window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
        if len(window) < WINDOW_DAYS or window.isna().any(): continue
        contexts.append(torch.tensor(window.values.astype(np.float32)))
        tickers.append(tk)
    print(f"  forecasting {len(contexts)} tickers...", flush=True)
    with torch.no_grad():
        out = pipe.predict(contexts, prediction_length=PREDICTION_LENGTH)
    preds = np.asarray(out)
    p70_df = pd.DataFrame({"ticker": tickers,
                           "chronos_p70_3m": [float(preds[j, 6, -1] / contexts[j].numpy()[-1] - 1)
                                              for j in range(len(tickers))]})
    sub = sub.merge(p70_df, on="ticker", how="left")
    sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)

    # Apply filter and pick top-3
    eligible = sub.dropna(subset=["v3_score", "chronos_p70_3m"])
    eligible = eligible[eligible["chr_p70_rk"] >= CHRONOS_FILTER_QUANTILE]
    eligible = eligible[~eligible["ticker"].isin(EXCLUDE)]
    top = eligible.sort_values("v3_score", ascending=False).head(3)
    print(f"  picks: {top['ticker'].tolist()}", flush=True)

    # Cache scores for webapp builder
    sub.to_parquet(PIT / "v5_today_scored.parquet", index=False)
    print(f"saved scored panel for {last_asof.date()}")


if __name__ == "__main__":
    main()
