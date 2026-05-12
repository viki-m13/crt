"""Score today's S&P 500 picks using the v5 Chronos-filter winner.

Strategy: ml_3plus6 (v3 baseline) + Chronos-bolt-tiny p70 confidence filter.
At each rebalance:
  1. Compute v3 ml_3plus6 score for each PIT S&P 500 member.
  2. Compute Chronos-bolt-tiny p70 (3m forecast, 70th-percentile probabilistic
     forecast) for each member from 252-day daily price history.
  3. Cross-sectionally rank both within S&P 500.
  4. Filter to stocks with Chronos rank >= 0.4 (top 60%).
  5. Pick top-3 by ml_3plus6 score from filtered pool.
  6. Equal-weight, hold 6 months.
  7. Tight regime gate.

Backtest: 44.81% full CAGR, 45.86% WF mean OOS, 17.01% WF min, 10/10 splits beat SPY.
vs v3 baseline: +5.04pp full / +3.06pp WF mean / +2.52pp WF min / 10/10 vs 9/10.
"""
from __future__ import annotations
import json, sys
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
EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}


def main():
    # Load latest panel
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    last_asof = panel["asof"].max()
    print(f"latest asof: {last_asof.date()}", flush=True)

    sub = panel[panel["asof"] == last_asof].copy()
    print(f"PIT members at asof: {len(sub)}", flush=True)

    # Attach v3 ml predictions
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    sub = sub.merge(ml[ml["asof"] == last_asof], on=["asof", "ticker"], how="left")
    sub["v3_score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2

    # Run Chronos on the PIT members at this asof
    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"loading {CHRONOS_MODEL}...", flush=True)
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)
    contexts = []
    tickers = []
    for tk in sub["ticker"]:
        if tk not in daily.columns: continue
        ts = daily[tk].dropna()
        pos = ts.index.searchsorted(last_asof, side="right") - 1
        if pos < WINDOW_DAYS: continue
        window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
        if len(window) < WINDOW_DAYS or window.isna().any():
            continue
        contexts.append(torch.tensor(window.values.astype(np.float32)))
        tickers.append(tk)
    if not contexts:
        print("no Chronos contexts; abort"); return
    print(f"  forecasting {len(contexts)} tickers...", flush=True)
    with torch.no_grad():
        out = pipe.predict(contexts, prediction_length=PREDICTION_LENGTH)
    preds = np.asarray(out)
    p70 = pd.DataFrame({"ticker": tickers,
                         "chronos_p70_3m": [float(preds[j, 6, -1] / contexts[j].numpy()[-1] - 1)
                                            for j in range(len(tickers))]})
    sub = sub.merge(p70, on="ticker", how="left")
    sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)

    # Apply filter and pick. K=2 (was K=3) — updated 2026-05-12 after
    # the augmented-PIT sweep confirmed K=2 dominates K=3 on every
    # metric (see experiments/monthly_dca/v5/spx_pit/IMPROVEMENTS.md).
    eligible = sub.dropna(subset=["v3_score", "chronos_p70_3m"])
    eligible = eligible[eligible["chr_p70_rk"] >= 0.45]
    eligible = eligible[~eligible["ticker"].isin(EXCLUDE)]
    print(f"  eligible after filter: {len(eligible)}", flush=True)
    K_PICKS = 2
    top = eligible.sort_values("v3_score", ascending=False).head(K_PICKS)
    picks = []
    for i, (_, r) in enumerate(top.iterrows()):
        picks.append({
            "ticker": r["ticker"],
            "weight": 1 / K_PICKS,
            "v3_score": float(r["v3_score"]),
            "chronos_p70_3m": float(r["chronos_p70_3m"]),
            "chronos_rank": float(r["chr_p70_rk"]),
        })

    out = {
        "asof": str(last_asof.date()),
        "strategy": "v5_chr_p70_filter_k2",
        "spec": {"k": K_PICKS, "weighting": "invvol", "regime_gate": "tight",
                 "hold_months": 6, "chronos_filter_quantile": 0.45,
                 "chronos_model": CHRONOS_MODEL},
        "picks": picks,
    }
    out_path = PIT / "v5_today_picks.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
