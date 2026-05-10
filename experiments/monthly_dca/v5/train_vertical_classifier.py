"""Train a "vertical-winner" classifier — predicts whether a stock is at the
START of an NVDA/TSLA-style multi-bagger run.

Target: a stock has gone up >100% in the next 12 months (true multibagger).

Training data: full broader 1833-ticker panel, 67 features per (asof, ticker).
Walk-forward annual retrain with 7-month embargo (matching v2/v4).

The classifier predicts P(stock has 100%+ 12m forward return).  We can use this
score directly to pick stocks in the PIT S&P 500 universe.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
FEATURES_DIR = CACHE / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}


def build_panel():
    """Build the broader panel with 12m forward returns + verticality target."""
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    feature_files = {pd.Timestamp(p.stem): p for p in FEATURES_DIR.glob("*.parquet")}
    asofs = sorted(feature_files.keys())
    ref_d = pd.Timestamp("2010-12-31")
    if ref_d not in feature_files: ref_d = asofs[len(asofs)//2]
    feature_cols = list(pd.read_parquet(feature_files[ref_d]).columns)
    print(f"  reference cols: {len(feature_cols)}")

    chunks = []
    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        if not set(feature_cols).issubset(feat.columns):
            continue
        feat = feat[feature_cols]
        if len(feat) < 100:
            continue
        # Cross-sectional ranks
        for c in feature_cols:
            r = feat[c].rank(pct=True)
            feat[c + "_xs"] = (r - 0.5) * 2
        feat = feat.reset_index().rename(columns={"index": "ticker"})
        feat["asof"] = d
        chunks.append(feat)
    panel = pd.concat(chunks, ignore_index=True)
    panel = panel[["asof", "ticker"] + [c + "_xs" for c in feature_cols] + feature_cols]
    print(f"  panel: {panel.shape}")

    # 12m forward returns
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

    fwd_12m = []
    for _, row in panel[["asof", "ticker"]].iterrows():
        d = row["asof"]; tk = row["ticker"]
        pos = asof_to_pos.get(d, None)
        if pos is None or pos + 12 >= len(mr_dates) or tk not in monthly_returns.columns:
            fwd_12m.append(np.nan); continue
        d0 = mr_dates[pos]; dh = mr_dates[pos + 12]
        try:
            lr0 = log_mr.at[d0, tk]; lrh = log_mr.at[dh, tk]
        except KeyError:
            fwd_12m.append(np.nan); continue
        if pd.isna(lr0) or pd.isna(lrh):
            fwd_12m.append(np.nan); continue
        fwd_12m.append(np.expm1(lrh - lr0))
    panel["fwd_12m_ret"] = fwd_12m
    # Verticality target: 12m forward return > 100%
    panel["target_vertical"] = (panel["fwd_12m_ret"] > 1.0).astype(float)
    pos_rate = panel["target_vertical"].dropna().mean()
    print(f"  vertical positive rate: {pos_rate:.4f}")
    return panel, feature_cols


def fit_classifier(panel, feature_cols, embargo_months=7, n_seeds=3,
                   rolling_years=10, train_start=pd.Timestamp("2003-01-01")):
    xs_cols = [c + "_xs" for c in feature_cols]
    months = sorted(panel["asof"].unique())
    asof_by_year = {}
    for m in months:
        y = pd.Timestamp(m).year
        asof_by_year.setdefault(y, []).append(pd.Timestamp(m))
    years = sorted(asof_by_year.keys())

    preds_rows = []
    for y in years:
        if y < train_start.year + 2: continue
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.DateOffset(months=embargo_months)
        train_lo = pd.Timestamp(year=max(train_start.year, y - rolling_years), month=1, day=1)
        train_data = panel[(panel["asof"] >= train_lo) & (panel["asof"] < cutoff)].copy()
        train_data = train_data.dropna(subset=["target_vertical"] + xs_cols)
        if len(train_data) < 1000: continue
        X_tr = train_data[xs_cols].values.astype(np.float32)
        y_tr = train_data["target_vertical"].values.astype(np.float32)
        # use scale_pos_weight to handle imbalance
        spw = (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6)
        seed_models = []
        for seed in range(n_seeds):
            m = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=47,
                min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.8,
                bagging_freq=5, seed=seed*13+1, verbose=-1, n_jobs=-1,
                scale_pos_weight=spw,
            )
            m.fit(X_tr, y_tr)
            seed_models.append(m)

        for ta in asof_by_year[y]:
            te = panel[panel["asof"] == ta].dropna(subset=xs_cols)
            if len(te) < 50: continue
            X_te = te[xs_cols].values.astype(np.float32)
            preds_seeds = np.column_stack([mm.predict_proba(X_te)[:, 1] for mm in seed_models])
            pred = preds_seeds.mean(axis=1)
            for tk, p in zip(te["ticker"].values, pred):
                preds_rows.append({"asof": pd.Timestamp(ta), "ticker": tk, "p_vertical": float(p)})
        print(f"  year {y}: spw={spw:.2f}, n_train={len(train_data)}; cum preds={len(preds_rows)}", flush=True)

    return pd.DataFrame(preds_rows)


def main():
    print("=== Vertical-winner classifier training ===", flush=True)
    out_path = PIT / "ml_preds_vertical.parquet"
    if out_path.exists():
        print("  already exists, skip"); return
    panel, fc = build_panel()
    preds = fit_classifier(panel, fc)
    preds.to_parquet(out_path, index=False)
    print(f"  saved {preds.shape} to {out_path}")


if __name__ == "__main__":
    main()
