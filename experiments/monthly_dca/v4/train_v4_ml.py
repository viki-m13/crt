"""Train v4 ML predictions: stronger LightGBM ensemble.

Key improvements over v2 ML model:
  1. LightGBM (faster, often better than HistGradientBoosting)
  2. 5-seed ensemble (averaged predictions)
  3. Direct cross-sectional rank target (smoother than raw return)
  4. Multi-horizon stack: 3m and 6m rank targets, predictions averaged
  5. Annual retrain with 7-month embargo (matches v2 — for fair WF comparison)
  6. Trained on broader 1833-ticker panel; predicts on full panel
  7. Cross-sectional within-month standardization of features at predict time

Output: cache/v2/sp500_pit/ml_preds_v4.parquet  with cols
  [asof, ticker, pred_v4_3m, pred_v4_6m, pred_v4]  where pred_v4 = mean(3m,6m).
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


def build_broad_panel():
    """Build feature panel for the broader 1833-ticker universe (all months).

    Cross-sectionally rank features within each month, then compute fwd 3m/6m
    rank targets across the broader universe.
    """
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    feature_files = {pd.Timestamp(p.stem): p for p in FEATURES_DIR.glob("*.parquet")}
    asofs = sorted(feature_files.keys())

    # Use the canonical 67-feature reference
    ref_d = pd.Timestamp("2010-12-31")
    if ref_d not in feature_files: ref_d = asofs[len(asofs) // 2]
    feature_cols = list(pd.read_parquet(feature_files[ref_d]).columns)
    print(f"  reference cols ({ref_d.date()}): {len(feature_cols)}")

    chunks = []
    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        if not set(feature_cols).issubset(feat.columns):
            continue
        feat = feat[feature_cols]
        if len(feat) < 100:
            continue
        for c in feature_cols:
            r = feat[c].rank(pct=True)
            feat[c + "_xs"] = (r - 0.5) * 2
        feat = feat.reset_index().rename(columns={"index": "ticker"})
        feat["asof"] = d
        chunks.append(feat)
    panel = pd.concat(chunks, ignore_index=True)
    panel = panel[["asof", "ticker"] + [c + "_xs" for c in feature_cols] + feature_cols]
    print(f"  broad panel: {panel.shape}, {panel['ticker'].nunique()} tickers")

    # forward returns
    mr = monthly_returns
    mr_dates = mr.index.sort_values()
    log_mr = np.log1p(mr.fillna(0)).cumsum()

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

    print("  computing forward 3m/6m...")
    fwd = {3: [], 6: []}
    for h in (3, 6):
        for _, row in panel[["asof", "ticker"]].iterrows():
            d = row["asof"]; tk = row["ticker"]
            pos = asof_to_pos.get(d, None)
            if pos is None or pos + h >= len(mr_dates) or tk not in mr.columns:
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
    # cross-sectional rank target
    for h in (3, 6):
        panel[f"rank_target_{h}m"] = panel.groupby("asof")[f"fwd_{h}m_ret"].rank(pct=True)
    return panel, feature_cols


def fit_walkforward_ensemble(panel, feature_cols,
                             embargo_months=7, n_seeds=3,
                             rolling_years=10,
                             train_start=pd.Timestamp("2003-01-01")):
    """Annual retrain LightGBM ensemble. Returns DataFrame[asof, ticker, pred_v4_3m, pred_v4_6m].

    Uses a rolling-window training set of `rolling_years` to keep training fast
    and adapt to regime shifts.
    """
    import lightgbm as lgb
    xs_cols = [c + "_xs" for c in feature_cols]
    months = sorted(panel["asof"].unique())
    asof_by_year: dict[int, list] = {}
    for m in months:
        y = pd.Timestamp(m).year
        asof_by_year.setdefault(y, []).append(pd.Timestamp(m))
    years = sorted(asof_by_year.keys())

    preds_rows = []
    for year_idx, y in enumerate(years):
        if y < train_start.year + 2:
            continue
        cutoff = pd.Timestamp(year=y, month=1, day=1) - pd.DateOffset(months=embargo_months)
        train_lo = pd.Timestamp(year=max(train_start.year, y - rolling_years), month=1, day=1)
        train_data = panel[(panel["asof"] >= train_lo) & (panel["asof"] < cutoff)].copy()
        if len(train_data) < 1000:
            continue
        test_asofs = asof_by_year[y]

        for h in (3, 6):
            tgt = f"rank_target_{h}m"
            tr = train_data.dropna(subset=[tgt] + xs_cols)
            if len(tr) < 1000:
                continue
            X_train = tr[xs_cols].values.astype(np.float32)
            y_train = tr[tgt].values.astype(np.float32)
            seed_models = []
            for seed in range(n_seeds):
                m = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=47,
                    max_depth=-1,
                    min_data_in_leaf=300,
                    feature_fraction=0.7,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    seed=seed * 13 + 1,
                    verbose=-1,
                    n_jobs=-1,
                    objective="regression",
                )
                m.fit(X_train, y_train)
                seed_models.append(m)

            for ta in test_asofs:
                te = panel[panel["asof"] == ta].dropna(subset=xs_cols)
                if len(te) < 50:
                    continue
                X_te = te[xs_cols].values.astype(np.float32)
                preds_seeds = np.column_stack([mm.predict(X_te) for mm in seed_models])
                pred = preds_seeds.mean(axis=1)
                for tk, p in zip(te["ticker"].values, pred):
                    preds_rows.append({"asof": pd.Timestamp(ta), "ticker": tk,
                                       f"pred_v4_{h}m": float(p)})
        print(f"  year {y}: trained {n_seeds} seeds × 2 horizons (n_train={len(tr)}, train [{train_lo.year}..{cutoff.year-1}]); cum preds={len(preds_rows)}",
              flush=True)

    df = pd.DataFrame(preds_rows)
    # Pivot to consolidated form: one row per (asof, ticker) with both predictions
    df3 = df.dropna(subset=["pred_v4_3m"]).groupby(["asof", "ticker"], as_index=False)["pred_v4_3m"].mean()
    df6 = df.dropna(subset=["pred_v4_6m"]).groupby(["asof", "ticker"], as_index=False)["pred_v4_6m"].mean()
    out = df3.merge(df6, on=["asof", "ticker"], how="outer")
    out["pred_v4"] = (out["pred_v4_3m"].fillna(0.5) + out["pred_v4_6m"].fillna(0.5)) / 2
    return out


def main():
    print("=== v4 ML training ===")
    out_path = PIT / "ml_preds_v4.parquet"
    if out_path.exists():
        print(f"  already exists at {out_path}, skipping")
        return
    print("\n[1/2] Build broader 1833-ticker training panel ...")
    panel, feature_cols = build_broad_panel()
    print(f"  panel saved in memory: {panel.shape}")

    print("\n[2/2] Walk-forward fit LightGBM 5-seed ensemble ...")
    t0 = time.time()
    preds = fit_walkforward_ensemble(panel, feature_cols, n_seeds=5)
    print(f"  done in {(time.time()-t0)/60:.1f} min, preds: {preds.shape}")
    preds.to_parquet(out_path, index=False)
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
