"""
Walk-forward LightGBM CLASSIFIER predicting probability that a stock will be
in the TOP 10% of forward 6-month returns (a "banger").

Cross-sectional binary target: at each asof, label top 10% of fwd_6m as 1, rest 0.
This focuses the model on the tail of returns rather than the mean.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit" / "feature_panel_pit.parquet"
MONTHLY = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet"
OUT = ROOT / "experiments" / "monthly_dca" / "v8b" / "cache" / "banger_clf_preds.parquet"

EXCLUDE_TICKERS = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}


def fwd_ret(monthly_returns, horizon):
    mr = monthly_returns.copy().fillna(-1.0)
    log = np.log1p(mr).replace([-np.inf], -10.0)
    cum = log.rolling(horizon).sum().shift(-horizon)
    return np.expm1(cum)


def add_xs_ranks(df, cols):
    out = df.copy()
    for c in cols:
        if c in df.columns:
            out[f"{c}_xsrk"] = df.groupby("asof")[c].rank(pct=True)
    return out


def main():
    panel = pd.read_parquet(PANEL)
    panel["asof"] = pd.to_datetime(panel["asof"])
    panel = panel[~panel["ticker"].isin(EXCLUDE_TICKERS)].copy()

    mr = pd.read_parquet(MONTHLY)
    mr.index = pd.to_datetime(mr.index)
    fwd6 = fwd_ret(mr, 6).stack().reset_index()
    fwd6.columns = ["asof", "ticker", "fwd_6m"]
    fwd12 = fwd_ret(mr, 12).stack().reset_index()
    fwd12.columns = ["asof", "ticker", "fwd_12m"]
    panel = panel.merge(fwd6, on=["asof", "ticker"], how="left").merge(fwd12, on=["asof", "ticker"], how="left")

    # Cross-sectional ranks
    rank_features = ["mom_12_1", "mom_6_1", "mom_3", "mom_per_unit_vol_12",
                     "trend_health_5y", "earnings_drift_proxy", "trend_r2_12m",
                     "mom_consistency_12m", "rs_12m_spy", "idio_mom_12_1",
                     "vol_1y", "max_dd_5y", "multibagger_ratio_24m", "near_52wh_60d",
                     "tight_consolidation_60", "breakout_strength_60"]
    panel = add_xs_ranks(panel, rank_features)

    feature_cols = [c for c in panel.columns if c not in
                    ("asof", "ticker", "fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m")]
    print(f"Features: {len(feature_cols)}")

    asofs = sorted(panel["asof"].unique())
    panel = panel.reset_index(drop=True)

    test_months = [m for m in asofs if m >= pd.Timestamp("2003-09-30")]
    pred_rows = []
    last_retrain = None
    models = {}

    for tm in test_months:
        months_since = 999 if last_retrain is None else (tm.year - last_retrain.year) * 12 + (tm.month - last_retrain.month)
        if last_retrain is None or months_since >= 12:
            cutoff = tm - pd.DateOffset(months=7)
            tr = panel[panel["asof"] < cutoff].dropna(subset=["fwd_6m", "fwd_12m"]).copy()
            if len(tr) < 1000:
                continue
            # cross-sectional top decile target for fwd_6m
            tr["rk_6m"] = tr.groupby("asof")["fwd_6m"].rank(pct=True)
            tr["rk_12m"] = tr.groupby("asof")["fwd_12m"].rank(pct=True)
            tr["y_top10_6"] = (tr["rk_6m"] >= 0.90).astype(int)
            tr["y_top10_12"] = (tr["rk_12m"] >= 0.90).astype(int)
            X = tr[feature_cols].astype(float).values
            X = np.where(np.isfinite(X), X, 0.0)
            print(f"[Retrain @ {tm.date()}] rows={len(tr)} pos_rate(6m)={tr['y_top10_6'].mean():.3f}")
            for h, col in [("6", "y_top10_6"), ("12", "y_top10_12")]:
                m = lgb.LGBMClassifier(
                    objective="binary",
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
                m.fit(X, tr[col].values)
                models[h] = m
            last_retrain = tm

        sub = panel[panel["asof"] == tm].copy()
        if len(sub) == 0:
            continue
        Xt = sub[feature_cols].astype(float).values
        Xt = np.where(np.isfinite(Xt), Xt, 0.0)
        p6 = models["6"].predict_proba(Xt)[:, 1]
        p12 = models["12"].predict_proba(Xt)[:, 1]
        sub_pred = sub[["asof", "ticker"]].copy()
        sub_pred["banger_p6"] = p6
        sub_pred["banger_p12"] = p12
        sub_pred["banger_score"] = (p6 + p12) / 2
        pred_rows.append(sub_pred)

    out = pd.concat(pred_rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"Saved {len(out)} predictions to {OUT}")


if __name__ == "__main__":
    main()
