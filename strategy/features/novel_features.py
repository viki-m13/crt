"""Novel features for the pre-runner strategy.

All features computed strictly from `panel.loc[panel.index <= asof]`.
PIT slicing is enforced at the feature builder boundary.

Features:
  crt_6m       — Cross-Sectional Rank Trajectory (CRT). Spearman correlation
                 between time and the 21-day-return rank-percentile across the
                 cross-section, computed at last 6 month-ends.
  crt_3m       — Same, last 3 month-ends.
  rank_now     — Current 21-day return rank percentile.
  rank_6m_ago  — 6 month-ends ago rank percentile.
  rbi_60       — Reflexive Bounce Intensity. Of the last 60d's "big down days"
                 (return < -2%), fraction that had at least one >+1% day in
                 the next 5 trading days.
  vol_asym_60  — Sum-of-squared-up-days / sum-of-squared-down-days, last 60d.
  cst_score    — Capitulation-Stabilization Transition: how flat is the
                 60-day OLS slope of log-prices recently? Approaching zero
                 from below = stabilizing. Specifically:
                 cst = max(0, recent_slope - prev_slope) where prev is 60-120d
                 ago and recent is last 60d, on the log-price series.
  vov_asym     — Volatility of volatility (60d std of rolling 5d vol).
                 Lower for stable-high-vol; higher for explosive moves.
  prerunner_dist — Mahalanobis distance to the pre-runner archetype centroid
                 in {vol_3m, dd_from_52wh, accel, drawdown_age, trend_health}
                 5-D space. Lower = closer to the archetype.

These compose into the final selection score in `strategy/selection.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import (
    CACHE,
    FEATURES_DIR,
    load_features,
    load_feature_months,
    load_panel,
)
from experiments.monthly_dca.backtester import month_end_dates


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]. NaNs preserved."""
    return s.rank(pct=True)


def _21d_return(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    sub = panel.loc[panel.index <= asof]
    if len(sub) < 22:
        return pd.Series(dtype=float)
    last = sub.iloc[-1]
    prev = sub.iloc[-22]
    return (last / prev - 1.0)


def compute_crt(panel: pd.DataFrame, asof: pd.Timestamp,
                 lookback_months: int = 6) -> pd.DataFrame:
    """For each ticker, compute CRT = Spearman(time, rank_pct) across last N
    month-ends.  Returns a DataFrame indexed by ticker with columns
    crt_<N>m, rank_now, rank_<N>m_ago.
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) < 22 * (lookback_months + 1):
        return pd.DataFrame(index=panel.columns,
                            columns=[f"crt_{lookback_months}m",
                                     "rank_now",
                                     f"rank_{lookback_months}m_ago"])

    # Build the rank history at each of the last N+1 month-ends.
    # Use month-end approximation: 21 trading days per month.
    snapshots = []  # list of Series of percentile-rank
    for i in range(lookback_months + 1):
        cutoff = sub.iloc[: len(sub) - i * 21] if i > 0 else sub
        if len(cutoff) < 22:
            return pd.DataFrame(index=panel.columns,
                                columns=[f"crt_{lookback_months}m",
                                         "rank_now",
                                         f"rank_{lookback_months}m_ago"])
        last = cutoff.iloc[-1]
        prev = cutoff.iloc[-22]
        ret21 = (last / prev - 1.0)
        snapshots.append(_safe_pct_rank(ret21))

    snapshots = list(reversed(snapshots))  # oldest first
    rank_mat = pd.concat(snapshots, axis=1)
    rank_mat.columns = list(range(len(snapshots)))

    # Spearman corr(time, rank) per ticker
    t = np.arange(len(snapshots), dtype=float)
    t_centered = t - t.mean()
    t_var = float((t_centered ** 2).sum())

    def crt_row(row: np.ndarray) -> float:
        if np.any(~np.isfinite(row)):
            return np.nan
        # Convert to ranks within row to compute Spearman
        # Easier: just compute Pearson on values; the "rank percentiles"
        # are already on a uniform [0,1] scale, so Pearson ≈ Spearman.
        y = row - row.mean()
        cov = float((t_centered * y).sum())
        y_var = float((y ** 2).sum())
        if y_var <= 0 or t_var <= 0:
            return np.nan
        return cov / math.sqrt(t_var * y_var)

    crts = rank_mat.apply(lambda r: crt_row(r.to_numpy()), axis=1)
    out = pd.DataFrame({
        f"crt_{lookback_months}m": crts,
        "rank_now": snapshots[-1],
        f"rank_{lookback_months}m_ago": snapshots[0],
    })
    return out


def compute_rbi(panel: pd.DataFrame, asof: pd.Timestamp,
                 window: int = 60, big_down: float = -0.02,
                 reflex_window: int = 5, reflex_up: float = 0.01) -> pd.Series:
    """Reflexive Bounce Intensity over `window` trading days.

    For each big-down-day (daily return <= big_down), count whether at least
    one >= reflex_up day occurred within next reflex_window trading days.
    RBI = #yes / #big-down-days.  If no big-down-days in window, return 0.5
    (neutral).
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) < window + reflex_window + 2:
        return pd.Series(np.nan, index=panel.columns)
    rets = sub.pct_change().iloc[-(window + reflex_window):]
    out = {}
    arr = rets.to_numpy()
    cols = rets.columns
    n_rows = arr.shape[0]
    rbi_start = n_rows - window  # we count big-down-days only in the last `window` rows
    for j, c in enumerate(cols):
        col = arr[:, j]
        if not np.any(np.isfinite(col)):
            out[c] = np.nan
            continue
        big_down_idx = np.where(col[rbi_start:rbi_start + window] <= big_down)[0]
        if len(big_down_idx) == 0:
            out[c] = 0.5
            continue
        bounce = 0
        for k in big_down_idx:
            abs_k = rbi_start + k
            slc = col[abs_k + 1: abs_k + 1 + reflex_window]
            if len(slc) and np.nanmax(slc) >= reflex_up:
                bounce += 1
        out[c] = bounce / len(big_down_idx)
    return pd.Series(out)


def compute_vol_asym(panel: pd.DataFrame, asof: pd.Timestamp,
                       window: int = 60) -> pd.Series:
    """Sum of squared positive returns / sum of squared negative returns
    over last `window` days.  > 1 = positive variance dominates.
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) < window + 1:
        return pd.Series(np.nan, index=panel.columns)
    rets = sub.pct_change().iloc[-window:]
    pos = rets.where(rets > 0, 0.0)
    neg = rets.where(rets < 0, 0.0)
    pos_ss = (pos ** 2).sum()
    neg_ss = (neg ** 2).sum()
    return (pos_ss / neg_ss.replace(0, np.nan))


def compute_cst(panel: pd.DataFrame, asof: pd.Timestamp,
                 recent: int = 60, prev: int = 60) -> pd.Series:
    """Capitulation–Stabilization Transition.

    Slope of log-price over last `recent` days minus slope over the
    `prev` days before that.  Positive = stabilization (slope flattening
    from negative toward zero or positive).  Higher = stronger
    stabilization.
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) < recent + prev + 2:
        return pd.Series(np.nan, index=panel.columns)
    sub = sub.iloc[-(recent + prev):]
    log_p = np.log(sub.replace(0, np.nan))

    def _slope(arr: np.ndarray) -> float:
        if not np.all(np.isfinite(arr)):
            return np.nan
        n = len(arr)
        x = np.arange(n)
        x_mean = x.mean()
        y_mean = arr.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 0:
            return np.nan
        return float(((x - x_mean) * (arr - y_mean)).sum() / denom)

    out = {}
    for c in log_p.columns:
        col = log_p[c].to_numpy()
        prev_slope = _slope(col[:prev])
        recent_slope = _slope(col[prev:prev + recent])
        if not (np.isfinite(prev_slope) and np.isfinite(recent_slope)):
            out[c] = np.nan
            continue
        # Stabilization: improvement in slope, esp. transitioning from
        # negative to less-negative or zero.  Bonus if prev was negative.
        improvement = recent_slope - prev_slope
        if prev_slope >= 0:
            # Already trending up — not capitulation-stabilization
            out[c] = max(improvement, 0) * 0.5
        else:
            out[c] = improvement
    return pd.Series(out)


def compute_vov(panel: pd.DataFrame, asof: pd.Timestamp,
                 window: int = 60) -> pd.Series:
    """Volatility of volatility.  Std of rolling 5-day realized vol over
    last `window` days, scaled by mean rolling vol (so it's a CV).
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) < window + 6:
        return pd.Series(np.nan, index=panel.columns)
    rets = sub.pct_change().iloc[-(window + 5):]
    rolling_vol = rets.rolling(5).std()
    rolling_vol = rolling_vol.iloc[-window:]
    mean_v = rolling_vol.mean()
    std_v = rolling_vol.std()
    return (std_v / mean_v.replace(0, np.nan))


def compute_prerunner_distance(feats: pd.DataFrame,
                                 archetype: dict[str, float] | None = None
                                 ) -> pd.Series:
    """Z-scaled Euclidean distance to the pre-runner archetype centroid.

    Archetype is set from the forensic medians (research/forensics).
    """
    if archetype is None:
        archetype = {
            "vol_3m": 0.79,
            "dd_from_52wh": 0.55,    # positive number — distance below 52wh
            "accel": 0.13,
            "drawdown_age_days": 169.0,
            "trend_health_5y": 0.55,
        }
    keys = [k for k in archetype.keys() if k in feats.columns]
    if not keys:
        return pd.Series(np.nan, index=feats.index)
    sub = feats[keys].astype(float)
    # Z-scale by cross-sectional median+MAD
    out = pd.DataFrame(index=feats.index)
    for k in keys:
        med = sub[k].median()
        mad = (sub[k] - med).abs().median()
        if not np.isfinite(mad) or mad <= 0:
            mad = 1.0
        out[k] = (sub[k] - archetype[k]) / mad
    dist = np.sqrt((out ** 2).sum(axis=1))
    return dist


def compute_all_novel(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Compute the full novel-feature pack at one asof."""
    out = pd.DataFrame(index=panel.columns)

    # CRT (6m and 3m)
    try:
        crt6 = compute_crt(panel, asof, lookback_months=6)
        out = out.join(crt6, how="left")
    except Exception:
        pass
    try:
        crt3 = compute_crt(panel, asof, lookback_months=3)
        # Avoid duplicate column names
        crt3 = crt3.rename(columns={"rank_now": "rank_now_3"})
        out = out.join(crt3[["crt_3m", "rank_3m_ago"]], how="left")
    except Exception:
        pass

    # RBI
    try:
        out["rbi_60"] = compute_rbi(panel, asof, window=60)
    except Exception:
        pass
    try:
        out["rbi_120"] = compute_rbi(panel, asof, window=120)
    except Exception:
        pass

    # Vol asymmetry
    try:
        out["vol_asym_60"] = compute_vol_asym(panel, asof, window=60)
    except Exception:
        pass
    try:
        out["vol_asym_126"] = compute_vol_asym(panel, asof, window=126)
    except Exception:
        pass

    # CST
    try:
        out["cst_score"] = compute_cst(panel, asof, recent=60, prev=60)
    except Exception:
        pass

    # VoV
    try:
        out["vov_60"] = compute_vov(panel, asof, window=60)
    except Exception:
        pass

    # Pre-runner distance — needs cached features
    try:
        feats = load_features(asof)
        # bring in dd_from_52wh, accel, drawdown_age_days, vol_3m, trend_health_5y
        keep = [c for c in ["vol_3m", "dd_from_52wh", "accel",
                            "drawdown_age_days", "trend_health_5y"]
                 if c in feats.columns]
        if keep:
            out["prerunner_dist"] = compute_prerunner_distance(feats[keep])
    except Exception:
        pass

    return out


def main(start: str = "1997-01-01", end: str = "2099-01-01",
          overwrite: bool = False) -> None:
    """Compute novel features for all month-ends and merge into existing
    cached feature parquets."""
    panel = load_panel()
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    print(f"Computing novel features for {len(months)} months")
    sentinel = "crt_6m"
    n_done = 0
    for i, asof in enumerate(months):
        feat_path = FEATURES_DIR / f"{asof.date()}.parquet"
        if not feat_path.exists():
            continue
        existing = pd.read_parquet(feat_path)
        if (not overwrite) and (sentinel in existing.columns):
            n_done += 1
            continue
        try:
            extras = compute_all_novel(panel, asof)
        except Exception as e:
            print(f"  skip {asof.date()}: {e}")
            continue
        # Drop overlap columns
        for col in list(extras.columns):
            if col in existing.columns:
                if overwrite:
                    existing = existing.drop(columns=[col])
                else:
                    extras = extras.drop(columns=[col])
        merged = existing.join(extras, how="left")
        merged.to_parquet(feat_path)
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  [{i+1}/{len(months)}] {asof.date()}: {merged.shape}")
    print(f"Done. {n_done} months already had features (skipped).")


if __name__ == "__main__":
    main()
