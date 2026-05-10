"""
v8 — Score factory for novel & proprietary stock-selection strategies.

Each scorer takes the per-month feature subframe (sub) and assigns a "score"
column. The downstream engine (v6/lib_engine.simulate) sorts top-K by score
and equal/inv-vol weights them. ALL features are point-in-time; no fitted
parameters are used unless the strategy explicitly trains a walk-forward
ML model elsewhere.

Inputs:
- panel: feature_panel_pit.parquet — columns: features... + asof + ticker
- ml_preds: ml_preds_v2.parquet (existing PIT walk-forward ML predictions)

Output:
- DataFrame with columns: asof, ticker, score, vol_1y, mom_12_1, pullback_1y, trend_health_5y
  (the engine pulls the extras for filtering)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
PANEL_PARQUET = PIT / "feature_panel_pit.parquet"
MLPREDS = CACHE / "v2" / "ml_preds_v2.parquet"
LGB_PREDS = ROOT / "experiments" / "monthly_dca" / "v8b" / "cache" / "lgb_ranker_preds.parquet"

EXCLUDE_TICKERS = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL_PARQUET)
    p["asof"] = pd.to_datetime(p["asof"])
    p = p[~p["ticker"].isin(EXCLUDE_TICKERS)].copy()
    return p


def load_mlpreds() -> pd.DataFrame:
    m = pd.read_parquet(MLPREDS)
    m["asof"] = pd.to_datetime(m["asof"])
    m["ml_score"] = (m["pred_3m"] + m["pred_6m"]) / 2
    return m[["asof", "ticker", "ml_score", "pred_1m", "pred_3m", "pred_6m"]]


def load_lgb_preds() -> pd.DataFrame:
    m = pd.read_parquet(LGB_PREDS)
    m["asof"] = pd.to_datetime(m["asof"])
    m = m.rename(columns={"score": "lgb_score", "pred_3m": "lgb_3m", "pred_6m": "lgb_6m"})
    return m[["asof", "ticker", "lgb_score", "lgb_3m", "lgb_6m"]]


def load_banger_clf() -> pd.DataFrame:
    p = ROOT / "experiments" / "monthly_dca" / "v8b" / "cache" / "banger_clf_preds.parquet"
    m = pd.read_parquet(p)
    m["asof"] = pd.to_datetime(m["asof"])
    return m[["asof", "ticker", "banger_p6", "banger_p12", "banger_score"]]


# ---------------------------------------------------------------------------
# Cross-sectional helpers (per-asof rank/zscore)
# ---------------------------------------------------------------------------
def xs_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("asof")[col].rank(pct=True)


def xs_z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("asof")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


# ---------------------------------------------------------------------------
# Atomic factor scorers (each must produce a `score` column on the panel)
# ---------------------------------------------------------------------------
def s_mom_12_1(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "mom_12_1")


def s_idio_mom(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "idio_mom_12_1")


def s_mom_per_vol(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "mom_per_unit_vol_12")


def s_breakout(p: pd.DataFrame) -> pd.Series:
    """Tight consolidation + new 52wh + low BB width = launching pad."""
    a = xs_rank(p, "tight_consolidation_60")
    b = xs_rank(p, "near_52wh_60d")
    c = 1 - xs_rank(p, "bb_width_pct")  # narrow band = good
    return (a + b + c) / 3


def s_breakout_strength(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "breakout_strength_60")


def s_trend_quality(p: pd.DataFrame) -> pd.Series:
    """Stan Weinstein Stage 2: r2 12m × consistency × health."""
    a = xs_rank(p, "trend_r2_12m")
    b = xs_rank(p, "mom_consistency_12m")
    c = xs_rank(p, "trend_health_5y")
    d = xs_rank(p, "frac_above_50dma_1y")
    return (a + b + c + d) / 4


def s_quality_score(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "quality_score_5y")


def s_earnings_drift(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "earnings_drift_proxy")


def s_acceleration(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "mom_accel")


def s_low_vol(p: pd.DataFrame) -> pd.Series:
    return 1 - xs_rank(p, "vol_1y")


def s_sharpe_12m(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "sharpe_12m")


def s_multibagger_24m(p: pd.DataFrame) -> pd.Series:
    return xs_rank(p, "multibagger_ratio_24m")


# ---------------------------------------------------------------------------
# Composite proprietary strategies
# ---------------------------------------------------------------------------
def s_concretum_trend(p: pd.DataFrame) -> pd.Series:
    """
    Trend-following inspired by the Concretum article.
    1) Strong 12-1 momentum
    2) High trend R^2 (smooth uptrend, not gambling)
    3) Idiosyncratic mom (alpha not beta)
    4) Above SMA200 (Stage 2 confirmed)
    5) Earnings drift proxy positive
    """
    mom = xs_rank(p, "mom_12_1")
    r2 = xs_rank(p, "trend_r2_12m")
    idio = xs_rank(p, "idio_mom_12_1")
    above200 = (p["d_sma200"] > 0).astype(float)
    ed = xs_rank(p, "earnings_drift_proxy")
    score = mom * 0.30 + r2 * 0.20 + idio * 0.20 + ed * 0.15 + above200 * 0.15
    return score


def s_alpha_apex(p: pd.DataFrame) -> pd.Series:
    """
    Proprietary blend designed for high CAGR:
    - High momentum (mom_per_unit_vol_12, mom_12_1)
    - Acceleration (mom_accel, accel)
    - Quality of trend (trend_r2_12m, mom_consistency_12m)
    - Multi-bagger track record (24m)
    - Earnings drift positive
    - NOT extreme pullback (filter out -50%+ down stocks)
    """
    mpv = xs_rank(p, "mom_per_unit_vol_12")
    mom = xs_rank(p, "mom_12_1")
    macc = xs_rank(p, "mom_accel")
    r2 = xs_rank(p, "trend_r2_12m")
    cons = xs_rank(p, "mom_consistency_12m")
    mb = xs_rank(p, "multibagger_ratio_24m")
    ed = xs_rank(p, "earnings_drift_proxy")
    score = (mpv * 0.25 + mom * 0.20 + macc * 0.10 + r2 * 0.10
             + cons * 0.10 + mb * 0.10 + ed * 0.15)
    # Penalize stocks deeply below 52w high (cuts falling knives)
    pull = p["pullback_1y"].fillna(0)
    score = score * np.where(pull < -0.5, 0.5, 1.0)
    return score


def s_alpha_apex_v2(p: pd.DataFrame) -> pd.Series:
    """Tighter trend-quality emphasis; tail-aware."""
    mpv = xs_rank(p, "mom_per_unit_vol_12")
    sh = xs_rank(p, "sharpe_12m")
    macc = xs_rank(p, "mom_accel")
    r2 = xs_rank(p, "trend_r2_12m")
    cons = xs_rank(p, "mom_consistency_12m")
    mb = xs_rank(p, "multibagger_ratio_24m")
    ed = xs_rank(p, "earnings_drift_proxy")
    bk = xs_rank(p, "breakout_strength_60")
    score = (mpv * 0.20 + sh * 0.10 + macc * 0.10 + r2 * 0.10
             + cons * 0.10 + mb * 0.15 + ed * 0.15 + bk * 0.10)
    return score


def s_dual_momentum(p: pd.DataFrame) -> pd.Series:
    """Absolute momentum + relative momentum (RS vs SPY)."""
    rs = xs_rank(p, "rs_12m_spy")
    mom = xs_rank(p, "mom_12_1")
    return (rs + mom) / 2


def s_low_dd_winner(p: pd.DataFrame) -> pd.Series:
    """Smooth winners: high mom, low max DD 5y, low recent vol."""
    mom = xs_rank(p, "mom_12_1")
    minimal_dd = 1 - xs_rank(p, "max_dd_5y")  # less negative = better
    lv = 1 - xs_rank(p, "vol_3m")
    cons = xs_rank(p, "mom_consistency_12m")
    return (mom + minimal_dd + lv + cons) / 4


def s_qmom_ml(panel: pd.DataFrame, ml: pd.DataFrame, w_ml: float = 0.5) -> pd.Series:
    """Blend ML pred with trend-quality (Concretum-style)."""
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
    ml_rank = p.groupby("asof")["ml_score"].rank(pct=True).fillna(0.5)
    trend = s_concretum_trend(p)
    return ml_rank * w_ml + trend * (1 - w_ml)


def s_ml_only(panel: pd.DataFrame, ml: pd.DataFrame) -> pd.Series:
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
    return p.groupby("asof")["ml_score"].rank(pct=True)


def s_lgb_only(panel: pd.DataFrame, lgb: pd.DataFrame) -> pd.Series:
    p = panel.merge(lgb[["asof", "ticker", "lgb_score"]], on=["asof", "ticker"], how="left")
    return p.groupby("asof")["lgb_score"].rank(pct=True)


def s_ml_plus_lgb(panel: pd.DataFrame, ml: pd.DataFrame, lgb: pd.DataFrame, w_lgb: float = 0.5) -> pd.Series:
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
    p = p.merge(lgb[["asof", "ticker", "lgb_score"]], on=["asof", "ticker"], how="left")
    ml_rank = p.groupby("asof")["ml_score"].rank(pct=True)
    lgb_rank = p.groupby("asof")["lgb_score"].rank(pct=True)
    return ml_rank * (1 - w_lgb) + lgb_rank * w_lgb


def s_ml_plus_banger(panel: pd.DataFrame, ml: pd.DataFrame, banger: pd.DataFrame,
                     w_b: float = 0.5) -> pd.Series:
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
    p = p.merge(banger[["asof", "ticker", "banger_score"]], on=["asof", "ticker"], how="left")
    ml_rank = p.groupby("asof")["ml_score"].rank(pct=True)
    b_rank = p.groupby("asof")["banger_score"].rank(pct=True)
    return ml_rank * (1 - w_b) + b_rank * w_b


def s_conviction_stack(panel: pd.DataFrame, ml: pd.DataFrame, lgb: pd.DataFrame,
                       banger: pd.DataFrame) -> pd.Series:
    """Conviction = stocks ranked highly by ALL three models simultaneously.
    Uses minimum of three normalized ranks → only 'consensus' picks score high."""
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
    p = p.merge(lgb[["asof", "ticker", "lgb_score"]], on=["asof", "ticker"], how="left")
    p = p.merge(banger[["asof", "ticker", "banger_score"]], on=["asof", "ticker"], how="left")
    r1 = p.groupby("asof")["ml_score"].rank(pct=True).fillna(0.5)
    r2 = p.groupby("asof")["lgb_score"].rank(pct=True).fillna(0.5)
    r3 = p.groupby("asof")["banger_score"].rank(pct=True).fillna(0.5)
    # Conviction = mean rank, but penalise by minimum (any model dislikes → drop)
    return (r1 + r2 + r3) / 3 + np.minimum(np.minimum(r1, r2), r3) * 0.5


# ---------------------------------------------------------------------------
# Public entry point: build a score panel for the v6 engine
# ---------------------------------------------------------------------------
def build_score_panel(strategy: str, weights: dict | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: asof, ticker, score, vol_1y, vol_rank,
    mom_12_1, pullback_1y, trend_health_5y. The v6 engine consumes this directly.
    """
    panel = load_panel()
    ml = load_mlpreds()
    try:
        lgb = load_lgb_preds()
    except Exception:
        lgb = None
    weights = weights or {}

    # Compute the base score depending on strategy
    if strategy == "ml_3plus6":
        scores = s_ml_only(panel, ml)
    elif strategy == "lgb":
        scores = s_lgb_only(panel, lgb)
    elif strategy == "ml_plus_lgb":
        scores = s_ml_plus_lgb(panel, ml, lgb, w_lgb=weights.get("lgb", 0.5))
    elif strategy == "banger":
        banger = load_banger_clf()
        p = panel.merge(banger[["asof", "ticker", "banger_score"]], on=["asof", "ticker"], how="left")
        scores = p.groupby("asof")["banger_score"].rank(pct=True)
    elif strategy == "ml_plus_banger":
        banger = load_banger_clf()
        scores = s_ml_plus_banger(panel, ml, banger, w_b=weights.get("banger", 0.5))
    elif strategy == "conviction_stack":
        banger = load_banger_clf()
        scores = s_conviction_stack(panel, ml, lgb, banger)
    elif strategy == "mom_12_1":
        scores = s_mom_12_1(panel)
    elif strategy == "idio_mom":
        scores = s_idio_mom(panel)
    elif strategy == "mom_per_vol":
        scores = s_mom_per_vol(panel)
    elif strategy == "breakout":
        scores = s_breakout(panel)
    elif strategy == "breakout_strength":
        scores = s_breakout_strength(panel)
    elif strategy == "trend_quality":
        scores = s_trend_quality(panel)
    elif strategy == "concretum_trend":
        scores = s_concretum_trend(panel)
    elif strategy == "alpha_apex":
        scores = s_alpha_apex(panel)
    elif strategy == "alpha_apex_v2":
        scores = s_alpha_apex_v2(panel)
    elif strategy == "dual_momentum":
        scores = s_dual_momentum(panel)
    elif strategy == "low_dd_winner":
        scores = s_low_dd_winner(panel)
    elif strategy == "earnings_drift":
        scores = s_earnings_drift(panel)
    elif strategy == "acceleration":
        scores = s_acceleration(panel)
    elif strategy == "quality_score":
        scores = s_quality_score(panel)
    elif strategy == "multibagger":
        scores = s_multibagger_24m(panel)
    elif strategy == "qmom_ml":
        scores = s_qmom_ml(panel, ml, w_ml=weights.get("ml", 0.5))
    elif strategy == "blend":
        # Custom: weighted blend of named atomic factors
        # weights = {factor_name: w}
        s = pd.Series(0.0, index=panel.index)
        total = 0.0
        for k, w in weights.items():
            if k == "ml":
                p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="left")
                fs = p.groupby("asof")["ml_score"].rank(pct=True).fillna(0.5)
            else:
                fn = globals().get(f"s_{k}")
                if fn is None:
                    raise ValueError(f"Unknown atomic factor: {k}")
                fs = fn(panel)
            s = s + fs.fillna(0.5) * w
            total += w
        if total > 0:
            s = s / total
        scores = s
    else:
        raise ValueError(strategy)

    out = panel[["asof", "ticker", "vol_1y", "mom_12_1", "pullback_1y",
                 "trend_health_5y", "d_sma200"]].copy()
    out["score"] = scores.values
    out = out.dropna(subset=["score"]).copy()
    out["vol_rank"] = out.groupby("asof")["vol_1y"].rank(pct=True)
    return out


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "concretum_trend"
    sp = build_score_panel(s)
    print(f"Strategy={s} rows={len(sp)} asofs={sp['asof'].nunique()}")
    print(sp.head(3))
    print("Mean tickers per asof:", sp.groupby("asof").size().mean())
