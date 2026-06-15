"""Shared data layer for the Floor picker: merge labels + Chronos downside
forecasts + cross-sectional features, and train the walk-forward downside
GBMs (features = cross-sectional panel features + Chronos forecasts).

Cache the fully-scored frame to floor_scored.parquet so signal exploration
and the final backtest don't pay the GBM cost repeatedly.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUG = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit" / "augmented"
SCORED = HERE / "floor_scored.parquet"

HORIZONS = ["1m", "3m", "6m", "12m"]
GBM_EMBARGO_DAYS = 120
GBM_RETRAIN_EVERY = 3
# learned downside targets (all "lower = safer" except maxdd where higher=safer)
GBM_TARGETS = ["uw_frac_1m", "uw_frac_3m", "ever_below_3m", "maxdd_3m"]

CHR_COLS = [
    "chr_exp_uw_frac_1m", "chr_p_below_end_1m", "chr_trough_q10_1m", "chr_trough_q30_1m", "chr_p50_end_1m",
    "chr_exp_uw_frac_3m", "chr_p_below_end_3m", "chr_trough_q10_3m", "chr_trough_q30_3m", "chr_p50_end_3m",
    "chr_exp_uw_frac_6m", "chr_p_below_end_6m", "chr_trough_q10_6m", "chr_trough_q30_6m", "chr_p50_end_6m",
]


def load_merged():
    lab = pd.read_parquet(HERE / "downside_labels.parquet")
    chr_ = pd.read_parquet(HERE / "chronos_floor_preds.parquet")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    xs = [c for c in panel.columns if c.endswith("_xs")]
    panel = panel[["asof", "ticker"] + xs].copy()
    for df in (lab, chr_, panel):
        df["asof"] = pd.to_datetime(df["asof"])
    df = lab.merge(chr_, on=["asof", "ticker"], how="left").merge(
        panel, on=["asof", "ticker"], how="left")
    return df, xs


def add_gbms(df, xs, targets=GBM_TARGETS):
    feats = list(xs) + CHR_COLS
    asofs = np.array(sorted(df["asof"].unique()))
    for tgt in targets:
        df[f"gbm_{tgt}"] = np.nan
    models = {t: None for t in targets}
    for i, t in enumerate(asofs):
        embargo = t - pd.Timedelta(days=GBM_EMBARGO_DAYS)
        if (i % GBM_RETRAIN_EVERY == 0) or any(m is None for m in models.values()):
            tr = df[df["asof"] <= embargo]
            for tgt in targets:
                sub = tr[tr[tgt].notna()]
                if len(sub) >= 5000:
                    m = HistGradientBoostingRegressor(
                        max_iter=250, learning_rate=0.05, max_depth=6,
                        min_samples_leaf=60, l2_regularization=1.0, random_state=0)
                    m.fit(sub[feats].to_numpy(np.float32), sub[tgt].to_numpy())
                    models[tgt] = m
        cur = df["asof"] == t
        Xc = df.loc[cur, feats].to_numpy(np.float32)
        for tgt in targets:
            if models[tgt] is not None:
                df.loc[cur, f"gbm_{tgt}"] = models[tgt].predict(Xc)
    return df


def build(force=False):
    if SCORED.exists() and not force:
        return pd.read_parquet(SCORED)
    df, xs = load_merged()
    df = add_gbms(df, xs)
    df.to_parquet(SCORED, index=False)
    return df


if __name__ == "__main__":
    import time
    t = time.time()
    df = build(force=True)
    print(f"built {df.shape} in {time.time()-t:.0f}s -> {SCORED}")
    for tgt in GBM_TARGETS:
        print(f"  gbm_{tgt} coverage {df[f'gbm_{tgt}'].notna().mean():.2%}")
