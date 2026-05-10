"""Train a v6 ML model: v3's 67 features + 11 new proprietary features.

Annual retrain, 7-month embargo, full-history training (no rolling window).
LightGBM 5-seed ensemble for robustness.  Predicts 3m and 6m forward rank
(matching v2's multi-horizon approach).
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
FEATURES_DIR = CACHE / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}


def build_panel():
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    feature_files = {pd.Timestamp(p.stem): p for p in FEATURES_DIR.glob("*.parquet")}
    asofs = sorted(feature_files.keys())
    ref_d = pd.Timestamp("2010-12-31")
    if ref_d not in feature_files: ref_d = asofs[len(asofs)//2]
    feature_cols = list(pd.read_parquet(feature_files[ref_d]).columns)

    chunks = []
    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        if not set(feature_cols).issubset(feat.columns): continue
        feat = feat[feature_cols]
        if len(feat) < 100: continue
        for c in feature_cols:
            r = feat[c].rank(pct=True)
            feat[c + "_xs"] = (r - 0.5) * 2
        feat = feat.reset_index().rename(columns={"index": "ticker"})
        feat["asof"] = d
        chunks.append(feat)
    panel = pd.concat(chunks, ignore_index=True)

    # Attach proprietary features
    props = pd.read_parquet(PIT / "proprietary_features.parquet")
    props["asof"] = pd.to_datetime(props["asof"])
    panel = panel.merge(props, on=["asof", "ticker"], how="left")

    # Cross-sectional rank for new features that aren't already 0-1 ranked
    new_xs_cols = ["mtf_alignment", "coiling_strength", "reversal_mom",
                   "power_consolidation", "vertical_index", "quality_compounder",
                   "recovery_setup", "rank_mom_change_12", "rank_mom_change_3"]
    for c in new_xs_cols:
        if c in panel.columns:
            r = panel.groupby("asof")[c].rank(pct=True)
            panel[c + "_xs"] = (r - 0.5) * 2

    # Forward returns
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
        if cand and cand[0][1] <= 7: asof_to_pos[d] = cand[0][0]

    fwd = {3: [], 6: []}
    for h in (3, 6):
        for _, row in panel[["asof", "ticker"]].iterrows():
            d = row["asof"]; tk = row["ticker"]
            pos = asof_to_pos.get(d, None)
            if pos is None or pos + h >= len(mr_dates) or tk not in monthly_returns.columns:
                fwd[h].append(np.nan); continue
            d0 = mr_dates[pos]; dh = mr_dates[pos + h]
            try:
                lr0 = log_mr.at[d0, tk]; lrh = log_mr.at[dh, tk]
            except KeyError:
                fwd[h].append(np.nan); continue
            if pd.isna(lr0) or pd.isna(lrh):
                fwd[h].append(np.nan); continue
            fwd[h].append(np.expm1(lrh - lr0))
    for h in (3, 6):
        panel[f"fwd_{h}m_ret"] = fwd[h]
        panel[f"rank_target_{h}m"] = panel.groupby("asof")[f"fwd_{h}m_ret"].rank(pct=True)

    return panel, feature_cols, new_xs_cols


def fit_walkforward(panel, feature_cols, new_xs_cols, embargo_months=7, n_seeds=5,
                    train_start=pd.Timestamp("2003-01-01")):
    import lightgbm as lgb
    xs_cols = [c + "_xs" for c in feature_cols] + [c + "_xs" for c in new_xs_cols if c + "_xs" in panel.columns]
    print(f"  using {len(xs_cols)} feature columns", flush=True)
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
        train_data = panel[panel["asof"] < cutoff].copy()
        if len(train_data) < 1000: continue
        for h in (3, 6):
            tgt = f"rank_target_{h}m"
            tr = train_data.dropna(subset=[tgt] + xs_cols)
            if len(tr) < 1000: continue
            X_tr = tr[xs_cols].values.astype(np.float32)
            y_tr = tr[tgt].values.astype(np.float32)
            seeds_models = []
            for seed in range(n_seeds):
                m = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.04, num_leaves=63,
                    min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                    bagging_freq=5, seed=seed*13+1, verbose=-1, n_jobs=-1,
                )
                m.fit(X_tr, y_tr)
                seeds_models.append(m)
            for ta in asof_by_year[y]:
                te = panel[panel["asof"] == ta].dropna(subset=xs_cols)
                if len(te) < 50: continue
                X_te = te[xs_cols].values.astype(np.float32)
                seed_preds = np.column_stack([mm.predict(X_te) for mm in seeds_models])
                pred = seed_preds.mean(axis=1)
                for tk, p in zip(te["ticker"].values, pred):
                    preds_rows.append({"asof": pd.Timestamp(ta), "ticker": tk,
                                       f"pred_v6_{h}m": float(p)})
        print(f"  year {y}: trained {n_seeds} seeds × 2 horizons (n_train={len(tr)}); cum preds={len(preds_rows)}", flush=True)

    df = pd.DataFrame(preds_rows)
    df3 = df.dropna(subset=["pred_v6_3m"]).groupby(["asof","ticker"], as_index=False)["pred_v6_3m"].mean()
    df6 = df.dropna(subset=["pred_v6_6m"]).groupby(["asof","ticker"], as_index=False)["pred_v6_6m"].mean()
    out = df3.merge(df6, on=["asof", "ticker"], how="outer")
    out["pred_v6"] = (out["pred_v6_3m"].fillna(0.5) + out["pred_v6_6m"].fillna(0.5)) / 2
    return out


def main():
    out_path = PIT / "ml_preds_v6.parquet"
    if out_path.exists():
        print("  already exists; skip"); return
    print("=== v6 ML training (v3 features + 11 proprietary) ===", flush=True)
    panel, fc, new_xs = build_panel()
    print(f"  panel: {panel.shape}", flush=True)
    preds = fit_walkforward(panel, fc, new_xs)
    preds.to_parquet(out_path, index=False)
    print(f"  saved {preds.shape} to {out_path}")


if __name__ == "__main__":
    main()
