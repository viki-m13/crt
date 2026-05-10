"""Pillar 1 — Failure-Avoidance Filter.

Computes a per-(asof, ticker) failure_score in [0, 1] (higher = more
failure-prone). Drops the bottom X% by failure_score before the
selection stage.

Built from forensic Study B's discriminating features. The score is a
weighted, sign-corrected sum of cross-sectional ranks — fully PIT
(every value at asof T uses only data with index ≤ T, since each
features parquet at T is itself PIT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"

# Failure-discriminating features (signed): positive sign means HIGHER value
# → MORE failure-prone. Derived from forensic_analysis output where the
# event-median is higher (or less negative) than the control-median.
# Magnitudes from KS and median delta in the Phase 1 analysis.
FAILURE_WEIGHTS = {
    "max_dd_5y":         (-1.0, 0.18),  # event_med deeper -> sign negative on the value (more neg = worse)
    "vol_1y":            ( 1.0, 0.18),  # higher vol = more failure-prone
    "best_month_24m":    ( 1.0, 0.18),  # bigger swings up = unstable
    "worst_month_24m":   (-1.0, 0.16),  # more neg worst-month = more failure-prone
    "vol_6m":            ( 1.0, 0.16),
    "pullback_all":      (-1.0, 0.15),  # deeper pullback = more failure-prone
    "pullback_5y":       (-1.0, 0.15),
    "tight_consolidation_60": (-1.0, 0.14),  # less consolidation = unstable
    "trend_health_5y":   (-1.0, 0.14),  # poor trend health → failure-prone
    "recovery_rate":     (-1.0, 0.13),  # poor historical recovery → failure-prone
    "mom_consistency_12m": (-1.0, 0.12),
    "frac_above_50dma_1y": (-1.0, 0.12),
    "rs_12m_spy":        (-1.0, 0.11),
    "d_sma200":          (-1.0, 0.11),
    "fip_score":         (-1.0, 0.10),
    "drawdown_age_days": ( 1.0, 0.10),  # long drawdown age = slow bleed
}


def _load_features_at(asof: pd.Timestamp) -> pd.DataFrame | None:
    f = FEATURES_DIR / f"{pd.Timestamp(asof).date()}.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f)


def compute_failure_score_at(asof: pd.Timestamp,
                             tickers: list[str] | None = None,
                             feat_df: pd.DataFrame | None = None) -> pd.Series:
    """Compute failure_score for the supplied tickers (or all in the panel)."""
    if feat_df is None:
        feat_df = _load_features_at(asof)
    if feat_df is None:
        return pd.Series(dtype=float)
    if tickers is not None:
        feat_df = feat_df.loc[feat_df.index.intersection(tickers)]

    score = pd.Series(0.0, index=feat_df.index)
    weight_sum = 0.0
    for col, (sign, w) in FAILURE_WEIGHTS.items():
        if col not in feat_df.columns:
            continue
        v = feat_df[col]
        # Cross-sectional rank → [0, 1]; missing → 0.5 neutral
        r = v.rank(pct=True)
        r = r.fillna(0.5)
        # If sign > 0, higher rank → higher failure score; if sign < 0, invert
        contrib = r if sign > 0 else (1.0 - r)
        score = score + w * contrib
        weight_sum += w
    if weight_sum > 0:
        score = score / weight_sum
    return score.rename("failure_score")


def build_failure_panel(asofs: list[pd.Timestamp]) -> pd.DataFrame:
    """Build a panel: asof, ticker, failure_score, failure_rank."""
    rows = []
    for ao in asofs:
        s = compute_failure_score_at(ao)
        if len(s) == 0:
            continue
        df = s.to_frame()
        df["asof"] = pd.Timestamp(ao)
        df["ticker"] = df.index
        df["failure_rank"] = df["failure_score"].rank(pct=True)
        rows.append(df.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def filter_universe(panel: pd.DataFrame, drop_bottom_pct: float = 0.30) -> pd.DataFrame:
    """Given a panel with `failure_rank` column, drop the worst drop_bottom_pct
    by rank (i.e. failure_rank > 1 - drop_bottom_pct = bottom of the universe)."""
    keep = panel["failure_rank"] <= (1.0 - drop_bottom_pct)
    return panel.loc[keep].copy()


if __name__ == "__main__":
    # Smoke test: compute failure score at a few asofs.
    import sys
    asofs_all = sorted(p.stem for p in FEATURES_DIR.glob("*.parquet"))
    sample = [pd.Timestamp(a) for a in (asofs_all[100], asofs_all[200], asofs_all[-1])]
    print(f"smoke test on asofs: {sample}")
    for a in sample:
        s = compute_failure_score_at(a)
        if len(s) == 0:
            continue
        print(f"  {a.date()}  n={len(s)}  mean={s.mean():.3f}  std={s.std():.3f}  "
              f"top5={s.nlargest(5).round(3).to_dict()}")
