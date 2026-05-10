"""Walk-forward HistGBM at WEEKLY cadence.

Honest setup:
- Target: rank of 4-week forward return per asof (cross-sectional).
- Embargo: 6 weeks (>= 4w target horizon, with safety buffer).
- Retrain frequency: every 13 weeks (quarterly).
- Train cutoff at retrain time T: asof < T - 6 weeks (no future leakage).
- Cross-sectional rank features per asof.

Output:
  experiments/monthly_dca/v8/weekly/cache/weekly_preds.parquet
  columns: asof, ticker, fwd_4w_ret, pred
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[4]
WEEKLY_CACHE = Path(__file__).resolve().parent / "cache"

EXCLUDE = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}

EMBARGO_WEEKS = 6
RETRAIN_EVERY_WEEKS = 13
TRAIN_START = pd.Timestamp("2003-01-01")
TRAIN_END = pd.Timestamp("2026-04-30")


def main():
    print("[load] weekly features + membership")
    feat = pd.read_parquet(WEEKLY_CACHE / "features_weekly.parquet")
    feat["asof"] = pd.to_datetime(feat["asof"])
    feat = feat[~feat["ticker"].isin(EXCLUDE)].copy()
    print(f"  rows={len(feat)} weeks={feat['asof'].nunique()} tickers={feat['ticker'].nunique()}")

    # Restrict to PIT S&P 500
    mem = pd.read_parquet(WEEKLY_CACHE / "sp500_membership_weekly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    feat = feat.merge(mem, on=["asof", "ticker"], how="inner")
    print(f"  PIT-restricted rows={len(feat)}")

    feature_cols = [c for c in feat.columns
                    if c not in ("asof", "ticker") and not c.startswith("fwd_")]

    # Cross-sectional rank features (PIT — per asof only)
    print("[xs] cross-sectional rank features")
    t0 = time.time()
    feat = feat.sort_values(["asof", "ticker"]).reset_index(drop=True)
    for c in feature_cols:
        feat[c + "_xs"] = feat.groupby("asof")[c].transform(
            lambda x: (x.rank(pct=True) - 0.5) * 2)
    print(f"  done in {time.time()-t0:.1f}s")

    xs_cols = [c + "_xs" for c in feature_cols]

    # Per-asof rank target
    feat["rank_target_4w"] = feat.groupby("asof")["fwd_4w_ret"].rank(pct=True)

    weeks = sorted(feat["asof"].unique())
    weeks_in_range = [w for w in weeks if TRAIN_START <= w <= TRAIN_END]
    print(f"[wf] retraining every {RETRAIN_EVERY_WEEKS} weeks across "
          f"{len(weeks_in_range)} test weeks ({weeks_in_range[0].date()} -> "
          f"{weeks_in_range[-1].date()})")

    weeks_arr = pd.DatetimeIndex(weeks)
    model = None
    last_retrain_idx = -10**6
    all_preds = []
    n_retrains = 0
    t0 = time.time()

    for i, tm in enumerate(weeks_in_range):
        # When to retrain?
        wpos = weeks_arr.get_loc(tm)
        do_retrain = (model is None) or (wpos - last_retrain_idx >= RETRAIN_EVERY_WEEKS)
        if do_retrain:
            cutoff = tm - pd.Timedelta(weeks=EMBARGO_WEEKS)
            train = feat[(feat["asof"] < cutoff) & feat["rank_target_4w"].notna()]
            if len(train) < 30000:
                continue
            Xt = train[xs_cols].values
            yt = train["rank_target_4w"].values
            mask = ~np.isnan(yt)
            Xt = Xt[mask]
            yt = yt[mask]
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.04, max_depth=6,
                min_samples_leaf=300, l2_regularization=1.0,
                random_state=42,
            )
            model.fit(Xt, yt)
            last_retrain_idx = wpos
            n_retrains += 1
            if n_retrains % 8 == 0 or i == 0:
                print(f"  retrain @ {tm.date()}  train_rows={len(train)}  "
                      f"({time.time()-t0:.0f}s, n_retrains={n_retrains})")

        test = feat[feat["asof"] == tm]
        if len(test) == 0 or model is None:
            continue
        Xtest = test[xs_cols].values
        pred = model.predict(Xtest)
        all_preds.append(test[["asof", "ticker", "fwd_4w_ret"]].assign(pred=pred))

    out = pd.concat(all_preds, ignore_index=True)
    out_path = WEEKLY_CACHE / "weekly_preds.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n[done] {len(out)} preds across {out['asof'].nunique()} weeks "
          f"({n_retrains} retrains, {time.time()-t0:.0f}s)")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
