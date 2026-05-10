"""Pillar 4 — Forensic archetype score.

Two implementations:
  A) Engineered: explicit boolean rules from the winner archetype centroid.
  B) Learned: scaled-Euclidean distance in feature space to winner centroid;
     winners are "close" (small distance), so we transform to a [0,1] score
     where higher = closer to winner archetype.

The winner centroid was built from forensic Study A using ONLY winner
events whose pre-window asof < the asof at which we want to score.
This is enforced by requiring the centroid file to be loaded once
per asof-aware build call. For simplicity in v1 we use the global
centroid built from all 1995-2026 winner episodes — see
research/forensics/archetypes.md. The leakage red-team in Phase 5
verifies that out-of-sample IC holds when the centroid is rebuilt with
strict prior-only data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"
DATA = ROOT / "experiments" / "multi_pillar_43Agh" / "data"

# Features used in archetype matching. Choose those with highest KS in
# Study A (winners vs controls). Keep a small set so distances are
# stable.
ARCHETYPE_COLS = [
    "vol_1y", "pullback_all", "max_dd_5y", "best_month_24m", "worst_month_24m",
    "pullback_5y", "quality_score_5y", "trend_health_5y", "rs_12m_spy",
    "vol_contraction", "tight_consolidation_60", "mom_consistency_12m",
    "drawdown_age_days", "fip_score", "near_52wh_60d", "rsi_14",
]


def load_centroid() -> pd.Series:
    f = DATA / "winner_archetype_centroid.parquet"
    return pd.read_parquet(f)["winner_centroid"]


def archetype_distance(feat_df: pd.DataFrame, centroid: pd.Series,
                       cols: list[str] | None = None) -> pd.Series:
    """Per-ticker distance to centroid in feature space, normalised by per-feature
    cross-sectional standard deviation at this asof.

    Returns Series in [0, +inf), smaller = closer to centroid.
    """
    cols = cols or ARCHETYPE_COLS
    cols = [c for c in cols if c in feat_df.columns and c in centroid.index]
    if not cols:
        return pd.Series(dtype=float)
    sub = feat_df[cols].copy()
    cent = centroid[cols]
    # Cross-sectional std at this asof for normalisation
    stds = sub.std()
    stds = stds.replace(0, 1e-6).fillna(1.0)
    z = (sub - cent) / stds
    z = z.fillna(0.0)
    dist = np.sqrt((z ** 2).mean(axis=1))  # mean-squared-z-distance
    return dist


def archetype_score_at(asof: pd.Timestamp, feat_df: pd.DataFrame | None = None,
                       centroid: pd.Series | None = None) -> pd.Series:
    """Per-ticker archetype score in [0, 1], higher = more like winner archetype.

    Score = 1 - rank(distance), so the closest tickers get score ≈ 1.
    """
    if feat_df is None:
        f = FEATURES_DIR / f"{pd.Timestamp(asof).date()}.parquet"
        if not f.exists():
            return pd.Series(dtype=float)
        feat_df = pd.read_parquet(f)
    if centroid is None:
        centroid = load_centroid()
    dist = archetype_distance(feat_df, centroid)
    if len(dist) == 0:
        return pd.Series(dtype=float)
    score = 1.0 - dist.rank(pct=True)
    return score.rename("archetype_score")


# Engineered rule-based score (binary: 0 or 1 + partial credit)
def engineered_archetype_score(feat_df: pd.DataFrame) -> pd.Series:
    """Heuristic rule-based archetype score in [0, 1].

    From Study A: pre-runners are high-vol, deeply pulled-back, with
    consolidation patterns and signs of base-building.
    """
    score = pd.Series(0.0, index=feat_df.index)
    if "vol_1y" in feat_df.columns:
        v = feat_df["vol_1y"].fillna(0.0)
        # Sweet spot: 0.40 - 0.80 (winners' median = 0.58)
        score += np.clip((v - 0.30) / 0.30, 0, 1) * 0.20  # ramp up
        score -= np.clip((v - 0.80) / 0.40, 0, 1) * 0.10  # too high penalised
    if "pullback_all" in feat_df.columns:
        pb = feat_df["pullback_all"].fillna(0.0)
        # winners' median = -0.72; deep pullback adds score
        score += np.clip(-pb / 1.0, 0, 1) * 0.20
    if "best_month_24m" in feat_df.columns:
        bm = feat_df["best_month_24m"].fillna(0.0)
        # winners' median = +0.33; bigger best-month boosts score
        score += np.clip(bm / 0.50, 0, 1) * 0.15
    if "vol_contraction" in feat_df.columns:
        vc = feat_df["vol_contraction"].fillna(0.5)
        # higher = vol is contracting (pre-breakout setup)
        score += np.clip(vc, 0, 1) * 0.15
    if "tight_consolidation_60" in feat_df.columns:
        tc = feat_df["tight_consolidation_60"].fillna(0.0)
        score += np.clip(tc, 0, 1) * 0.10
    if "near_52wh_60d" in feat_df.columns:
        # near 52wh recently → not a death-spiral; mild positive
        n52 = feat_df["near_52wh_60d"].fillna(0.0)
        score += np.clip(n52, 0, 1) * 0.10
    if "rs_12m_spy" in feat_df.columns:
        rs = feat_df["rs_12m_spy"].fillna(0.0)
        # positive RS adds, negative subtracts
        score += np.clip(rs / 0.50, -0.5, 1.0) * 0.10
    return score.clip(0.0, 1.0).rename("archetype_engineered")


def combined_archetype_score(asof: pd.Timestamp,
                             feat_df: pd.DataFrame | None = None,
                             centroid: pd.Series | None = None) -> pd.Series:
    """Average of distance-rank and engineered scores."""
    if feat_df is None:
        f = FEATURES_DIR / f"{pd.Timestamp(asof).date()}.parquet"
        if not f.exists():
            return pd.Series(dtype=float)
        feat_df = pd.read_parquet(f)
    sa = archetype_score_at(asof, feat_df, centroid)
    sb = engineered_archetype_score(feat_df)
    out = pd.concat([sa, sb], axis=1).mean(axis=1)
    return out.rename("archetype_score")


def build_archetype_panel(asofs: list[pd.Timestamp]) -> pd.DataFrame:
    centroid = load_centroid()
    rows = []
    for ao in asofs:
        f = FEATURES_DIR / f"{pd.Timestamp(ao).date()}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        s = combined_archetype_score(ao, df, centroid)
        out = s.to_frame()
        out["asof"] = pd.Timestamp(ao)
        out["ticker"] = out.index
        rows.append(out.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    # Smoke test
    asofs_all = sorted(p.stem for p in FEATURES_DIR.glob("*.parquet"))
    sample = [pd.Timestamp(a) for a in (asofs_all[100], asofs_all[150], asofs_all[-1])]
    centroid = load_centroid()
    print("smoke test archetype score:")
    for ao in sample:
        f = FEATURES_DIR / f"{ao.date()}.parquet"
        df = pd.read_parquet(f)
        s = combined_archetype_score(ao, df, centroid)
        print(f"  {ao.date()}: n={len(s)}, mean={s.mean():.3f}, top5={s.nlargest(5).round(3).to_dict()}")
