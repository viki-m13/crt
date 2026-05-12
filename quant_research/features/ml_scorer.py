"""
ML scorer: wraps the v3 walk-forward retrain predictions.

Two prediction sources:
1. sp500_pit_retrain_preds.parquet — the actual v3 walk-forward trained model
   covering 2005-11 → 2025-12. This is the TRUE v3 signal to benchmark against.
2. YLOka pit_panel_with_scores.parquet — a different model for 2003-09 → 2025-12.
   Use only after verifying it matches v3 performance.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pathlib, sys

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

_V3_PATH = ROOT / "experiments/monthly_dca/cache/v2/ml_preds_v2.parquet"   # CORRECT v3 source
_V3_RETRAIN_PATH = ROOT / "experiments/monthly_dca/cache/v2/sp500_pit/sp500_pit_retrain_preds.parquet"
_YLOKA_PATH = ROOT / "data/YLOka/pit_panel_with_scores.parquet"

_v3_preds: pd.DataFrame | None = None
_yloka: pd.DataFrame | None = None


def _load_v3_preds() -> pd.DataFrame:
    """Load the actual v3 full-panel predictions (ml_preds_v2)."""
    global _v3_preds
    if _v3_preds is None:
        df = pd.read_parquet(_V3_PATH)
        df["asof"] = pd.to_datetime(df["asof"])
        _v3_preds = df
    return _v3_preds


def _load_yloka() -> pd.DataFrame:
    global _yloka
    if _yloka is None:
        _yloka = pd.read_parquet(_YLOKA_PATH)
        _yloka["asof"] = pd.to_datetime(_yloka["asof"])
    return _yloka


def _snap_from(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    snap = df[df["asof"] == asof]
    if len(snap) == 0:
        prior = df[df["asof"] <= asof]
        if len(prior) == 0:
            return pd.DataFrame()
        snap = df[df["asof"] == prior["asof"].max()]
    return snap.set_index("ticker")


def _rank_in_full_universe(snap: pd.DataFrame, feats_index, col: str) -> pd.Series:
    """Rank col within the FULL snapshot, then filter to feats_index."""
    full_rank = snap[col].rank(pct=True)
    return full_rank.reindex(feats_index)


def make_v3_scorer(blend: str = "3plus6"):
    """
    v3 ML signal using ml_preds_v2 (covers 2003-09 → 2025-12).
    Uses raw score sum (same as v6 lib_engine.py: score = (pred_3m + pred_6m)/2).
    blend: '3plus6' | '3plus6plus1' | 'pred'
    """
    def score_fn(feats: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        df = _load_v3_preds()
        snap = _snap_from(df, asof)
        if len(snap) == 0:
            return pd.Series(dtype=float)
        # Use RAW score (not rank) — same as v6 engine
        if blend == "3plus6":
            scores = (snap["pred_3m"] + snap["pred_6m"]) / 2
        elif blend == "3plus6plus1":
            scores = (snap["pred_1m"] + snap["pred_3m"] + snap["pred_6m"]) / 3
        else:
            scores = snap["pred"]
        # Filter to PIT universe (feats.index already PIT-filtered by caller)
        return scores.reindex(feats.index).dropna()
    score_fn.__name__ = f"v3_{blend}"
    return score_fn


def make_yloka_scorer(blend: str = "3plus6"):
    """
    YLOka model (covers 2003-09 → 2025-12; different training from v3 retrain).
    Ranks within the FULL YLOka universe, then filters to PIT members.
    """
    def score_fn(feats: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        df = _load_yloka()
        snap = _snap_from(df, asof)
        if len(snap) == 0:
            return pd.Series(dtype=float)
        if blend == "3plus6":
            r3 = _rank_in_full_universe(snap, feats.index, "pred_3m")
            r6 = _rank_in_full_universe(snap, feats.index, "pred_6m")
            scores = (0.5 * r3 + 0.5 * r6)
        elif blend == "3plus6plus1":
            r1 = _rank_in_full_universe(snap, feats.index, "pred_1m")
            r3 = _rank_in_full_universe(snap, feats.index, "pred_3m")
            r6 = _rank_in_full_universe(snap, feats.index, "pred_6m")
            scores = (r1 + r3 + r6) / 3
        else:
            scores = _rank_in_full_universe(snap, feats.index, "pred")
        return scores.dropna()
    score_fn.__name__ = f"yloka_{blend}"
    return score_fn


# Convenience aliases
def make_yloka_3plus6_scorer():
    return make_yloka_scorer("3plus6")

def make_yloka_3plus6plus1_scorer():
    return make_yloka_scorer("3plus6plus1")

