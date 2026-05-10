"""Pillar 5 — Composite selection.

Builds the score_panel that the v6 engine consumes. Composes Pillars 1-4
with the existing ML score:
  - Stage 1: drop bottom X% of universe by failure_score (Pillar 1)
  - Stage 2: keep only trend-eligible names (Pillar 2)
  - Stage 3: composite = w_ml*z(ml) + w_arch*z(arch) + w_novel*z(novel) + w_classic*z(classic)

The composite score replaces the `score` column in `load_score_panel`'s
output. Engine consumes unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"
NOVEL_DIR = ROOT / "experiments" / "multi_pillar_43Agh" / "data" / "novel_features"
DATA = ROOT / "experiments" / "multi_pillar_43Agh" / "data"

# Re-use v6 engine's panel loader
import sys
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))
from lib_engine import load_score_panel as _v6_load_score_panel  # noqa: E402

from . import failure_filter, archetype, trend_regime  # noqa: E402


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.std() == 0 or len(s.dropna()) < 5:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std()


def build_composite_panel(
    universe: str = "sp500_pit",
    drop_failure_pct: float = 0.30,
    apply_trend_gate: bool = True,
    w_ml: float = 1.0,
    w_archetype: float = 0.4,
    w_novel: float = 0.3,
    w_classic: float = 0.3,
    w_failure: float = 0.4,
    novel_features_blend: float = 0.5,
) -> pd.DataFrame:
    """Build a multi-pillar score panel compatible with v6 engine.

    Output columns: asof, ticker, score, vol_1y, [extras]
    """
    # Base ML panel + extras (mom_12_1, pullback_1y, trend_health_5y, vol_rank)
    panel = _v6_load_score_panel("ml_3plus6", universe, attach_pullback=True)
    panel["asof"] = pd.to_datetime(panel["asof"])
    panel = panel.rename(columns={"score": "ml_score"})

    # Compute Pillars 1, 4 per asof
    centroid = archetype.load_centroid()
    asofs = sorted(panel["asof"].unique())

    pillars = []
    for ao in asofs:
        f = FEATURES_DIR / f"{pd.Timestamp(ao).date()}.parquet"
        if not f.exists():
            continue
        feat = pd.read_parquet(f)

        # Pillar 1 — failure score
        fs = failure_filter.compute_failure_score_at(ao, feat_df=feat)

        # Pillar 4 — archetype score
        arch = archetype.combined_archetype_score(ao, feat, centroid)

        # Pillar 3 — novel features
        nf_path = NOVEL_DIR / f"{pd.Timestamp(ao).date()}.parquet"
        if nf_path.exists():
            nfdf = pd.read_parquet(nf_path)
        else:
            nfdf = pd.DataFrame()

        # Pillar 2 — trend eligibility
        if apply_trend_gate:
            elig = trend_regime.compute_trend_eligibility(feat)
        else:
            elig = pd.Series(True, index=feat.index)

        # Classic momentum + quality composite
        classic_cols = []
        if "mom_12_1" in feat.columns:
            classic_cols.append("mom_12_1")
        if "quality_score_5y" in feat.columns:
            classic_cols.append("quality_score_5y")
        if classic_cols:
            cl = feat[classic_cols].apply(_zscore).mean(axis=1)
        else:
            cl = pd.Series(0.0, index=feat.index)

        df = pd.DataFrame({
            "ticker": feat.index,
            "failure_score": fs.reindex(feat.index).values,
            "archetype_score": arch.reindex(feat.index).values,
            "classic_score": cl.reindex(feat.index).values,
            "trend_eligible": elig.reindex(feat.index).fillna(False).values,
        })
        # Novel feature composite from fast panel: spy_corr_60d, price_persistence, abs_skew_60d
        if len(nfdf):
            n_join = nfdf.reindex(feat.index)
            corr = n_join.get("spy_corr_60d", pd.Series(0.5, index=feat.index)).fillna(0.5)
            pp = n_join.get("price_persistence", pd.Series(0.5, index=feat.index)).fillna(0.5)
            sk = n_join.get("abs_skew_60d", pd.Series(0.0, index=feat.index)).fillna(0.0)
            # Lower spy_corr = more idiosyncratic; higher persistence = trending; lower abs_skew = safer
            novel = (-_zscore(corr) * 0.3
                     + _zscore(pp) * 0.5
                     - _zscore(sk) * 0.2)
            df["novel_score"] = novel.values
        else:
            df["novel_score"] = 0.0

        df["asof"] = pd.Timestamp(ao)
        pillars.append(df)

    if not pillars:
        return panel.assign(score=panel["ml_score"])
    pillars_df = pd.concat(pillars, ignore_index=True)

    # Merge into ml panel
    panel = panel.merge(pillars_df, on=["asof", "ticker"], how="left")

    # Apply Stage 1: drop bottom X% by failure_score (per asof)
    if drop_failure_pct > 0:
        panel["failure_rank"] = panel.groupby("asof")["failure_score"].rank(pct=True)
        # failure_rank close to 1 = highest failure score = riskiest → drop
        panel = panel[panel["failure_rank"] <= (1.0 - drop_failure_pct)].copy()

    # Apply Stage 2: trend gate
    if apply_trend_gate:
        panel = panel[panel["trend_eligible"].fillna(False)].copy()

    # Stage 3: composite score
    grp = panel.groupby("asof")
    panel["ml_z"] = grp["ml_score"].transform(_zscore)
    panel["arch_z"] = grp["archetype_score"].transform(_zscore)
    panel["novel_z"] = grp["novel_score"].transform(_zscore) if "novel_score" in panel.columns else 0.0
    panel["classic_z"] = grp["classic_score"].transform(_zscore)
    panel["fail_z"] = grp["failure_score"].transform(_zscore)

    panel["score"] = (
        w_ml * panel["ml_z"].fillna(0.0)
        + w_archetype * panel["arch_z"].fillna(0.0)
        + w_novel * panel["novel_z"].fillna(0.0)
        + w_classic * panel["classic_z"].fillna(0.0)
        - w_failure * panel["fail_z"].fillna(0.0)  # higher failure score → lower composite
    )

    # Required output columns
    out_cols = ["asof", "ticker", "score", "vol_1y"]
    extras = [c for c in ("pullback_1y", "mom_12_1", "trend_health_5y", "vol_rank",
                          "ml_score", "archetype_score", "novel_score",
                          "failure_score", "classic_score") if c in panel.columns]
    return panel[out_cols + extras].copy()


if __name__ == "__main__":
    print("[selection] building composite panel ...")
    panel = build_composite_panel()
    print(f"  shape={panel.shape}")
    print(f"  asof range: {panel['asof'].min()} → {panel['asof'].max()}")
    print(f"  unique asofs: {panel['asof'].nunique()}")
    print(f"  unique tickers: {panel['ticker'].nunique()}")
    print(f"  score quantiles: {panel['score'].quantile([0.1, 0.5, 0.9]).round(3).to_dict()}")
    print("\n  sample top picks at last asof:")
    last_ao = panel["asof"].max()
    last = panel[panel["asof"] == last_ao].nlargest(10, "score")
    print(last[["ticker", "score", "ml_score", "archetype_score", "failure_score"]].to_string(index=False))
