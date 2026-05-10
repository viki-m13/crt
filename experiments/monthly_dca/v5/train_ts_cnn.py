"""Train a 1D CNN on raw 252-day price-return paths to predict 6m forward
cross-sectional rank.

Approach:
- Input: 252-day normalized log returns
- Hidden: small 1D CNN (3 conv layers + global avg pool + 2 FC)
- Output: scalar (predict cross-sectional rank in [0, 1])
- Loss: MSE on rank target
- Walk-forward annual retrain with 7-month embargo, 10y rolling window

This is a tractable CPU-friendly architecture (~10K params).
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS"}

WINDOW_DAYS = 252


def load_daily():
    return pd.read_parquet(CACHE / "prices_extended.parquet")


def normalized_log_path(prices: pd.Series, end_date: pd.Timestamp, window=WINDOW_DAYS):
    pos = prices.index.searchsorted(end_date, side="right") - 1
    if pos < 0 or pos < window:
        return None
    sub = prices.iloc[pos - window + 1: pos + 1]
    if sub.isna().any() or len(sub) < window:
        return None
    log_ret = np.log(sub.values / sub.values[0])
    return log_ret.astype(np.float32)


def build_dataset(daily, monthly_returns):
    """Build (X, y) dataset where:
       X = 252-day log-return path
       y = 6m forward rank (cross-sectional within asof)
    """
    feature_files = {pd.Timestamp(p.stem): p for p in (CACHE / "features").glob("*.parquet")}
    asofs = sorted(feature_files.keys())
    log_mr = np.log1p(monthly_returns.fillna(0)).cumsum()
    mr_dates = monthly_returns.index.sort_values()
    asof_to_pos = {}
    for d in pd.DatetimeIndex(asofs):
        pos = mr_dates.searchsorted(d)
        cand = []
        for j in (pos - 1, pos):
            if 0 <= j < len(mr_dates):
                cand.append((j, abs((mr_dates[j] - d).days)))
        cand.sort(key=lambda x: x[1])
        if cand and cand[0][1] <= 7:
            asof_to_pos[d] = cand[0][0]

    rows = []
    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        if len(feat) < 100:
            continue
        if d not in asof_to_pos:
            continue
        pos = asof_to_pos[d]
        if pos + 6 >= len(mr_dates):
            continue
        d0 = mr_dates[pos]; dh = mr_dates[pos + 6]

        # Compute 6m forward returns and ranks
        fwd_rets = {}
        for tk in feat.index:
            if tk not in monthly_returns.columns:
                continue
            try:
                lr0 = log_mr.at[d0, tk]; lrh = log_mr.at[dh, tk]
            except KeyError:
                continue
            if pd.isna(lr0) or pd.isna(lrh):
                continue
            fwd_rets[tk] = float(np.expm1(lrh - lr0))
        if len(fwd_rets) < 50:
            continue
        # Cross-sectional rank
        ranks = pd.Series(fwd_rets).rank(pct=True)
        # Build paths
        for tk, r in ranks.items():
            if tk not in daily.columns:
                continue
            path = normalized_log_path(daily[tk].dropna(), d)
            if path is None:
                continue
            # Skip paths with NaN/inf (bad price data)
            if not np.all(np.isfinite(path)):
                continue
            rows.append({"asof": d, "ticker": tk, "path": path, "rank": float(r)})
    return rows


def fit_cnn_walkforward(rows):
    """Walk-forward fit a 1D CNN. Annual retrain, 7-month embargo, 10y window."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    df = pd.DataFrame(rows)
    df["year"] = df["asof"].dt.year
    years = sorted(df["year"].unique())

    class TSCNN(nn.Module):
        def __init__(self, window=WINDOW_DAYS):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 16, 7, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 32, 5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Sequential(
                nn.Linear(32 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            x = x.unsqueeze(1)  # (B, 1, T)
            x = self.conv(x)
            x = x.flatten(1)
            return self.fc(x).squeeze(-1)

    device = "cpu"
    preds_rows = []
    for y in years:
        if y < 2005: continue
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.DateOffset(months=7)
        train_lo = pd.Timestamp(year=max(2003, y - 10), month=1, day=1)
        train_data = df[(df["asof"] >= train_lo) & (df["asof"] < cutoff)]
        if len(train_data) < 1000: continue
        X_tr = np.stack(train_data["path"].values).astype(np.float32)
        y_tr = train_data["rank"].values.astype(np.float32)

        model = TSCNN().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        loader = DataLoader(ds, batch_size=512, shuffle=True)
        for epoch in range(3):
            for X, y2 in loader:
                X, y2 = X.to(device), y2.to(device)
                optim.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y2)
                loss.backward()
                optim.step()

        # Predict for test asofs in year y
        te = df[df["year"] == y]
        if len(te) == 0: continue
        X_te = torch.from_numpy(np.stack(te["path"].values).astype(np.float32)).to(device)
        with torch.no_grad():
            pred = model(X_te).cpu().numpy()
        for (asof, ticker), p in zip(zip(te["asof"], te["ticker"]), pred):
            preds_rows.append({"asof": asof, "ticker": ticker, "p_cnn": float(p)})
        print(f"  year {y}: n_train={len(train_data)}, cum preds={len(preds_rows)}", flush=True)

    return pd.DataFrame(preds_rows)


def main():
    print("=== TS-CNN training ===", flush=True)
    out_path = PIT / "ml_preds_cnn.parquet"
    if out_path.exists():
        print("  already exists, skip"); return
    daily = load_daily()
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    print("  building dataset...", flush=True)
    rows = build_dataset(daily, monthly_returns)
    print(f"  dataset: {len(rows)} (asof,ticker,path,rank) rows", flush=True)
    if len(rows) < 1000:
        print("  too few rows; abort"); return
    preds = fit_cnn_walkforward(rows)
    preds.to_parquet(out_path, index=False)
    print(f"  saved {preds.shape} to {out_path}", flush=True)


if __name__ == "__main__":
    main()
