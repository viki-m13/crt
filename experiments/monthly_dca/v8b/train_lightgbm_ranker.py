"""
Train a strong walk-forward LightGBM ranker on the full 67-feature panel
to rank stocks by their forward 3-6m returns. Walk-forward retrain
every year, with a 7-month embargo (matches v3 protocol).

Output: experiments/monthly_dca/v8b/cache/lgb_ranker_preds.parquet
        columns: asof, ticker, score, score_h3, score_h6, score_h12

This is a STACK / RE-RANK on top of the rich feature panel. We add a few
cross-sectional ranks and interactions to the existing 67 features to
help the model.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit" / "feature_panel_pit.parquet"
MONTHLY = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet"
OUT_DIR = ROOT / "experiments" / "monthly_dca" / "v8b" / "cache"
OUT_DIR.mkdir(exist_ok=True, parents=True)

EXCLUDE_TICKERS = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}


def fwd_ret(monthly_returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """At each row t the value is the realized geometric return over t+1..t+H.
    Delisted rows become NaN -> -1.0 to preserve survivorship-bias-aware target."""
    mr = monthly_returns.copy()
    mr = mr.fillna(-1.0)
    log = np.log1p(mr)
    log = log.replace([-np.inf], -10.0)
    cum = log.rolling(horizon).sum().shift(-horizon)
    return np.expm1(cum)


def load_panel_with_targets() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL)
    panel["asof"] = pd.to_datetime(panel["asof"])
    panel = panel[~panel["ticker"].isin(EXCLUDE_TICKERS)].copy()
    mr = pd.read_parquet(MONTHLY)
    mr.index = pd.to_datetime(mr.index)
    fwd1 = fwd_ret(mr, 1)
    fwd3 = fwd_ret(mr, 3)
    fwd6 = fwd_ret(mr, 6)
    fwd12 = fwd_ret(mr, 12)

    # The fwd_ret matrices index = asof month-end. The panel asof is also
    # month-end. Join by (asof, ticker).
    def _to_long(fwd, name):
        f = fwd.stack().reset_index()
        f.columns = ["asof", "ticker", name]
        return f

    f1 = _to_long(fwd1, "fwd_1m")
    f3 = _to_long(fwd3, "fwd_3m")
    f6 = _to_long(fwd6, "fwd_6m")
    f12 = _to_long(fwd12, "fwd_12m")

    panel = (panel.merge(f1, on=["asof", "ticker"], how="left")
                   .merge(f3, on=["asof", "ticker"], how="left")
                   .merge(f6, on=["asof", "ticker"], how="left")
                   .merge(f12, on=["asof", "ticker"], how="left"))
    return panel


def add_xs_ranks(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in df.columns:
            out[f"{c}_xsrk"] = df.groupby("asof")[c].rank(pct=True)
    return out


def build_design_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_cols = [c for c in panel.columns if c not in
                    ("asof", "ticker", "fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m")]
    X = panel[feature_cols].astype(float).values
    return X, feature_cols


def main():
    print("Loading panel + computing forward returns...")
    panel = load_panel_with_targets()
    rank_features = ["mom_12_1", "mom_6_1", "mom_3", "mom_per_unit_vol_12",
                     "trend_health_5y", "earnings_drift_proxy", "trend_r2_12m",
                     "mom_consistency_12m", "rs_12m_spy", "idio_mom_12_1",
                     "vol_1y", "max_dd_5y", "multibagger_ratio_24m"]
    panel = add_xs_ranks(panel, rank_features)

    feature_cols = [c for c in panel.columns if c not in
                    ("asof", "ticker", "fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m")]
    print(f"Total features: {len(feature_cols)}")

    asofs = sorted(panel["asof"].unique())
    panel = panel.set_index("asof")
    print(f"Total months: {len(asofs)}, range {asofs[0]} -> {asofs[-1]}")

    # Walk-forward: retrain every year, 7-month embargo
    EMBARGO_MONTHS = 7
    RETRAIN_EVERY_MONTHS = 12
    PRED_HORIZONS = ("fwd_3m", "fwd_6m")

    pred_rows = []
    last_retrain = None
    models: dict = {}

    train_start = pd.Timestamp("2003-01-01")  # we have features from 1997, but let's use 2003+ since SP500 PIT starts there

    test_months = [m for m in asofs if m >= pd.Timestamp("2003-09-30")]

    # Pre-compute training row indices by month for speed
    panel_reset = panel.reset_index()
    panel_reset["asof"] = pd.to_datetime(panel_reset["asof"])

    for tm in test_months:
        # Retrain every 12 months
        months_since = 999 if last_retrain is None else (tm.year - last_retrain.year) * 12 + (tm.month - last_retrain.month)
        if last_retrain is None or months_since >= RETRAIN_EVERY_MONTHS:
            cutoff = tm - pd.DateOffset(months=EMBARGO_MONTHS)
            train = panel_reset[panel_reset["asof"] < cutoff].copy()
            # Keep rows with all targets present
            train = train.dropna(subset=list(PRED_HORIZONS))
            # Cap forward returns to avoid extreme outliers driving the model
            for h in PRED_HORIZONS:
                train[h] = train[h].clip(-0.95, 5.0)
            print(f"\n[Retrain @ {tm.date()}] train_rows={len(train)} cutoff<{cutoff.date()}")
            X = train[feature_cols].astype(float).values
            X = np.where(np.isfinite(X), X, 0.0)
            for h in PRED_HORIZONS:
                y = train[h].values
                # NOTE: cross-section rank target gives LightGBM a stable
                # signal across regimes. We rank each asof's universe by fwd_h
                # so the target ∈ [0, 1] is not biased by macro level.
                ym = train.groupby("asof")[h].rank(pct=True).values
                # Train regressor on the rank target
                m = lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=400,
                    learning_rate=0.04,
                    num_leaves=64,
                    min_data_in_leaf=200,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    reg_lambda=1.0,
                    verbose=-1,
                )
                m.fit(X, ym)
                models[h] = m
            last_retrain = tm

        # Predict for tm
        sub = panel_reset[panel_reset["asof"] == tm].copy()
        if len(sub) == 0:
            continue
        Xt = sub[feature_cols].astype(float).values
        Xt = np.where(np.isfinite(Xt), Xt, 0.0)
        preds = {}
        for h, m in models.items():
            preds[h] = m.predict(Xt)
        sub_pred = sub[["asof", "ticker"]].copy()
        sub_pred["pred_3m"] = preds["fwd_3m"]
        sub_pred["pred_6m"] = preds["fwd_6m"]
        sub_pred["score"] = (sub_pred["pred_3m"] + sub_pred["pred_6m"]) / 2
        pred_rows.append(sub_pred)

    out = pd.concat(pred_rows, ignore_index=True)
    out_path = OUT_DIR / "lgb_ranker_preds.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved {len(out)} predictions to {out_path}")
    print(out.head())


if __name__ == "__main__":
    main()
