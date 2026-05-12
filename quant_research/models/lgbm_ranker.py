"""
Walk-forward LightGBM cross-sectional ranker for monthly stock selection.

Training protocol:
- At each rebalance date t, fit on all available data up to t - embargo
- Embargo: 12 months (to prevent leakage of forward returns)
- Feature: 79-feature snapshot for each (date, ticker) pair
- Label: sign(1m forward return) for ranking, or rank of 1m forward return
- Predict: rank score for next month
- Select top-K

Point-in-time: only uses features/prices available at date t.
"""
from __future__ import annotations
import glob
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

FEAT_DIR = Path("/home/user/crt/experiments/monthly_dca/cache/features")
CACHE_DIR = Path("/home/user/crt/quant_research/data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}

# Features to use (exclude price-level and derived ML predictions)
FEATURE_COLS = [
    "mom_12_1", "mom_6_1", "mom_3", "mom_3y", "mom_5y", "mom_2y",
    "vol_1y", "vol_12m", "vol_3m", "vol_6m",
    "sharpe_1y", "sharpe_12m", "sharpe_5y",
    "trend_health_5y", "trend_r2_12m", "trend_slope_252",
    "frac_above_50dma_1y", "mom_consistency_12m",
    "d_sma200", "d_sma50", "sma50_above_200",
    "rsi_14", "rsi_zone_score",
    "dd_from_52wh", "below_52wh", "near_52wh_60d",
    "recovery_rate",
    "beta_2y", "max_dd_5y",
    "rs_3m_spy", "rs_6m_spy", "rs_12m_spy",
    "mom_accel", "accel",
    "idio_mom_12_1", "mom_per_unit_vol_12",
    "fip_score", "breakout_strength_60",
    "tight_consolidation_60", "vol_asym_60", "vol_asym_126",
    "crt_6m", "crt_3m", "rbi_60", "rbi_120",
    "prerunner_dist",
    "range_pos_1y", "bb_width_pct",
    "multibagger_ratio_24m", "quality_score_5y",
    "dist_from_low_1y", "drawdown_age_days",
    "excess_5y_logret", "acceleration_2y",
]


def load_full_panel(start: str = "2003-01-31", end: str = "2025-12-31") -> pd.DataFrame:
    """Load all feature snapshots into a single panel DataFrame."""
    files = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
    frames = []
    for f in files:
        date = pd.Timestamp(Path(f).stem)
        if date < pd.Timestamp(start) or date > pd.Timestamp(end):
            continue
        df = pd.read_parquet(f)
        df = df[~df.index.isin(EXCLUDE)]
        df["asof"] = date
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames).reset_index()
    panel.rename(columns={"index": "ticker", "ticker": "ticker"}, errors="ignore")
    if panel.index.name == "ticker":
        panel = panel.reset_index()
    return panel


def compute_forward_returns(prices: pd.DataFrame, feat_dates: list) -> pd.DataFrame:
    """Compute 1-month forward returns aligned to feature dates."""
    monthly_px = prices.resample("ME").last().ffill(limit=5)

    records = []
    for i, date in enumerate(feat_dates[:-1]):
        next_date = feat_dates[i + 1]

        d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
        if d0_idx > 0 and monthly_px.index[d0_idx] > date:
            d0_idx -= 1
        d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
        if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
            d1_idx -= 1

        p0 = monthly_px.iloc[d0_idx]
        p1 = monthly_px.iloc[d1_idx]

        fwd_ret = (p1 - p0) / p0
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)
        fwd_df = pd.DataFrame({"asof": date, "fwd_ret_1m": fwd_ret})
        records.append(fwd_df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records)


class WalkForwardLGBM:
    """Walk-forward LightGBM ranker for monthly cross-sectional stock selection."""

    def __init__(
        self,
        train_months: int = 48,  # 4 years training window
        embargo_months: int = 3,   # no-peek buffer
        min_train_months: int = 24,
        top_k: int = 10,
        lgb_params: Optional[dict] = None,
    ):
        self.train_months = train_months
        self.embargo_months = embargo_months
        self.min_train_months = min_train_months
        self.top_k = top_k
        self.lgb_params = lgb_params or {
            "objective": "regression",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        }
        self._models: dict = {}
        self._panel: Optional[pd.DataFrame] = None
        self._fwd_returns: Optional[pd.DataFrame] = None

    def _get_cache_key(self, params_str: str) -> str:
        return hashlib.md5(params_str.encode()).hexdigest()[:12]

    def prepare_data(self, prices: pd.DataFrame, feat_dates: list) -> None:
        """Build feature panel and forward returns. Cached to disk."""
        cache_key = self._get_cache_key(
            f"panel_{feat_dates[0]}_{feat_dates[-1]}_{self.train_months}"
        )
        panel_cache = CACHE_DIR / f"lgbm_panel_{cache_key}.pkl"
        fwd_cache = CACHE_DIR / f"lgbm_fwd_{cache_key}.pkl"

        if panel_cache.exists() and fwd_cache.exists():
            print("  Loading panel from cache...")
            with open(panel_cache, "rb") as f:
                self._panel = pickle.load(f)
            with open(fwd_cache, "rb") as f:
                self._fwd_returns = pickle.load(f)
            print(f"  Panel: {self._panel.shape}, FwdRet: {self._fwd_returns.shape}")
            return

        print("  Building feature panel...")
        panel_frames = []
        for date in feat_dates:
            path = FEAT_DIR / f"{date.strftime('%Y-%m-%d')}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df = df[~df.index.isin(EXCLUDE)]
            df.index.name = "ticker"
            df = df.reset_index()
            df["asof"] = date
            panel_frames.append(df)

        self._panel = pd.concat(panel_frames, ignore_index=True)
        print(f"  Panel shape: {self._panel.shape}")

        print("  Computing forward returns...")
        self._fwd_returns = compute_forward_returns(prices, feat_dates)
        print(f"  FwdRet shape: {self._fwd_returns.shape}")

        with open(panel_cache, "wb") as f:
            pickle.dump(self._panel, f)
        with open(fwd_cache, "wb") as f:
            pickle.dump(self._fwd_returns, f)

    def get_features(self, asof_date: pd.Timestamp) -> pd.DataFrame:
        """Get features for a specific date."""
        subset = self._panel[self._panel["asof"] == asof_date]
        available_cols = [c for c in FEATURE_COLS if c in subset.columns]
        result = subset[["ticker", "asof"] + available_cols].copy()
        return result

    def get_score_fn(
        self,
        predict_date: pd.Timestamp,
        feat_dates: list,
    ):
        """
        Returns a score function for predict_date that predicts
        next-month returns using a model trained on historical data.
        """
        # Find training window
        pred_idx = feat_dates.index(predict_date) if predict_date in feat_dates else -1
        if pred_idx < 0:
            return None

        # Training: up to pred_idx - embargo_months
        embargo_end_idx = pred_idx - self.embargo_months
        if embargo_end_idx < self.min_train_months:
            return None  # Not enough history

        train_start_idx = max(0, embargo_end_idx - self.train_months)
        train_dates = feat_dates[train_start_idx:embargo_end_idx]

        if len(train_dates) < self.min_train_months:
            return None

        # Check model cache
        model_key = (train_dates[0], train_dates[-1], self.train_months)
        if model_key in self._models:
            model, feature_cols = self._models[model_key]
        else:
            # Fit model
            model, feature_cols = self._fit_model(train_dates)
            if model is None:
                return None
            self._models[model_key] = (model, feature_cols)

        # Return score function
        def score_fn(feat_df: pd.DataFrame) -> pd.Series:
            available_cols = [c for c in feature_cols if c in feat_df.columns]
            if not available_cols:
                return pd.Series(dtype=float)
            X = feat_df[available_cols].copy()
            X = X.fillna(X.median())
            pred = model.predict(X)
            return pd.Series(pred, index=feat_df.index)

        return score_fn

    def _fit_model(self, train_dates: list) -> tuple:
        """Fit LightGBM on training dates."""
        if self._panel is None or self._fwd_returns is None:
            return None, []

        train_date_set = set(train_dates)

        # Filter panel to training dates
        X_full = self._panel[self._panel["asof"].isin(train_date_set)].copy()
        if X_full.empty:
            return None, []

        # Filter fwd returns to training dates (fwd has 'asof' column, ticker as index)
        fwd_train = self._fwd_returns[
            self._fwd_returns["asof"].isin(train_date_set)
        ].copy()
        if fwd_train.empty:
            return None, []

        # Merge: panel has 'ticker' column, fwd has ticker as index
        fwd_train_reset = fwd_train.reset_index()
        # After reset_index, old index becomes 'index' column
        if "ticker" not in fwd_train_reset.columns:
            fwd_train_reset = fwd_train_reset.rename(columns={"index": "ticker"})
        fwd_train_reset = fwd_train_reset[["ticker", "asof", "fwd_ret_1m"]]
        fwd_train_reset["ticker"] = fwd_train_reset["ticker"].astype(str)

        merged = X_full.merge(
            fwd_train_reset,
            on=["ticker", "asof"],
            how="inner",
        )

        available_cols = [c for c in FEATURE_COLS if c in merged.columns]
        if not available_cols:
            return None, []

        X_data = merged[available_cols].copy()
        y = merged["fwd_ret_1m"]
        valid = y.notna() & (y.abs() < 3.0)

        if valid.sum() < 200:
            return None, []

        X_train = X_data[valid].fillna(X_data[valid].median())
        y_train = y[valid]

        # Cross-sectional rank normalization (rank within each date)
        y_ranked = y_train.copy()
        merged_valid = merged[valid].copy()
        for date_val, grp in merged_valid.groupby("asof"):
            idx = grp.index
            if len(idx) < 5:
                continue
            y_ranked.loc[idx] = y_train.loc[idx].rank(pct=True)

        model = lgb.LGBMRegressor(**self.lgb_params)
        try:
            model.fit(X_train, y_ranked)
        except Exception as e:
            print(f"    Fit error: {e}")
            return None, []

        return model, available_cols
