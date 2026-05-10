"""
Train a richer LightGBM regressor on engineered + cross-sectional rank features.
Targets: cross-sectional rank of fwd 3m AND 6m returns (mean of two).
Walk-forward retrain every 6 months (more responsive than 12), 7-month embargo.
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
OUT = ROOT / "experiments" / "monthly_dca" / "v8b" / "cache" / "rich_ml_preds.parquet"

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


def add_engineered(p: pd.DataFrame) -> pd.DataFrame:
    out = p.copy()
    # Cross-sectional ranks for the most predictive raw features
    rank_features = [
        "mom_12_1", "mom_6_1", "mom_3", "mom_per_unit_vol_12",
        "trend_health_5y", "earnings_drift_proxy", "trend_r2_12m",
        "mom_consistency_12m", "rs_12m_spy", "idio_mom_12_1",
        "vol_1y", "vol_3m", "vol_6m",
        "max_dd_5y", "multibagger_ratio_24m",
        "near_52wh_60d", "tight_consolidation_60", "breakout_strength_60",
        "sharpe_12m", "sharpe_5y", "trend_slope_252",
        "fip_score", "rsi_14", "d_sma200", "d_sma50",
        "pullback_1y", "pullback_3y", "dd_from_52wh", "drawdown_age_days",
        "best_month_24m", "worst_month_24m", "tail_ratio_24m", "rsi_zone_score",
    ]
    for c in rank_features:
        if c in out.columns:
            out[f"{c}_xsr"] = out.groupby("asof")[c].rank(pct=True)
    # Engineered interactions
    out["mom_x_health"] = out.get("mom_12_1", 0).fillna(0) * out.get("trend_health_5y", 0).fillna(0)
    out["mom_x_earndrift"] = out.get("mom_12_1", 0).fillna(0) * out.get("earnings_drift_proxy", 0).fillna(0)
    out["mom_x_consistency"] = out.get("mom_12_1", 0).fillna(0) * out.get("mom_consistency_12m", 0).fillna(0)
    out["mom_x_breakout"] = out.get("mom_12_1", 0).fillna(0) * out.get("breakout_strength_60", 0).fillna(0)
    out["acc_x_consistency"] = out.get("mom_accel", 0).fillna(0) * out.get("mom_consistency_12m", 0).fillna(0)
    out["mb_x_mom"] = out.get("multibagger_ratio_24m", 0).fillna(0) * out.get("mom_12_1", 0).fillna(0)
    out["sharpe_x_mom"] = out.get("sharpe_12m", 0).fillna(0) * out.get("mom_12_1", 0).fillna(0)
    out["tight_x_near_high"] = out.get("tight_consolidation_60", 0).fillna(0) * out.get("near_52wh_60d", 0).fillna(0)
    out["edrift_x_health"] = out.get("earnings_drift_proxy", 0).fillna(0) * out.get("trend_health_5y", 0).fillna(0)
    return out


def main():
    panel = pd.read_parquet(PANEL)
    panel["asof"] = pd.to_datetime(panel["asof"])
    panel = panel[~panel["ticker"].isin(EXCLUDE_TICKERS)].copy()

    mr = pd.read_parquet(MONTHLY)
    mr.index = pd.to_datetime(mr.index)
    f3 = fwd_ret(mr, 3).stack().reset_index()
    f3.columns = ["asof", "ticker", "fwd_3m"]
    f6 = fwd_ret(mr, 6).stack().reset_index()
    f6.columns = ["asof", "ticker", "fwd_6m"]
    panel = panel.merge(f3, on=["asof", "ticker"], how="left").merge(f6, on=["asof", "ticker"], how="left")

    panel = add_engineered(panel)
    feature_cols = [c for c in panel.columns if c not in
                    ("asof", "ticker", "fwd_1m", "fwd_3m", "fwd_6m")]
    print(f"Features: {len(feature_cols)}")

    asofs = sorted(panel["asof"].unique())
    panel = panel.reset_index(drop=True)
    test_months = [m for m in asofs if m >= pd.Timestamp("2003-09-30")]
    pred_rows = []
    last_retrain = None
    models = {}

    for tm in test_months:
        months_since = 999 if last_retrain is None else (tm.year - last_retrain.year) * 12 + (tm.month - last_retrain.month)
        # retrain twice a year
        if last_retrain is None or months_since >= 6:
            cutoff = tm - pd.DateOffset(months=7)
            tr = panel[panel["asof"] < cutoff].dropna(subset=["fwd_3m", "fwd_6m"]).copy()
            if len(tr) < 1000:
                continue
            # Targets: cross-sectional rank percentile of forward returns
            tr["y_3m"] = tr.groupby("asof")["fwd_3m"].rank(pct=True)
            tr["y_6m"] = tr.groupby("asof")["fwd_6m"].rank(pct=True)
            X = tr[feature_cols].astype(float).values
            X = np.where(np.isfinite(X), X, 0.0)
            print(f"[Retrain @ {tm.date()}] rows={len(tr)} feats={len(feature_cols)}")
            for h, ycol in [("3m", "y_3m"), ("6m", "y_6m")]:
                m = lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=600,
                    learning_rate=0.03,
                    num_leaves=96,
                    min_data_in_leaf=150,
                    feature_fraction=0.7,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    reg_lambda=2.0,
                    verbose=-1,
                )
                m.fit(X, tr[ycol].values)
                models[h] = m
            last_retrain = tm

        sub = panel[panel["asof"] == tm].copy()
        if len(sub) == 0:
            continue
        Xt = sub[feature_cols].astype(float).values
        Xt = np.where(np.isfinite(Xt), Xt, 0.0)
        p3 = models["3m"].predict(Xt)
        p6 = models["6m"].predict(Xt)
        sub_pred = sub[["asof", "ticker"]].copy()
        sub_pred["pred_3m"] = p3
        sub_pred["pred_6m"] = p6
        sub_pred["score"] = (p3 + p6) / 2
        pred_rows.append(sub_pred)

    out = pd.concat(pred_rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"Saved {len(out)} predictions to {OUT}")


if __name__ == "__main__":
    main()
