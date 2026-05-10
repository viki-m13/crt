"""Pillar 2 — Trend & Regime gate.

Two responsibilities:
1. Market regime classification → cash sleeve, gross exposure, target K.
2. Per-stock trend eligibility → which tickers are admissible for selection.

Both are PIT: features at asof T use only data with index ≤ T.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------
def classify_market_regime(spy_row: dict, breadth: float | None = None,
                           dispersion: float | None = None) -> str:
    """Return one of: hostile | mixed | risk_on | recovery.

    Inputs:
        spy_row: dict with keys spy_dsma200, spy_dd_from_52wh, spy_mom_12_1,
                 spy_mom_6_1, spy_ret_21d, spy_below_200_streak, spy_rsi14
        breadth: fraction of universe above 200dma (0..1) — optional
        dispersion: cross-sectional return std over last 21d — optional
    """
    dsma = spy_row.get("spy_dsma200", 0.0)
    dd52 = spy_row.get("spy_dd_from_52wh", 0.0)
    m12 = spy_row.get("spy_mom_12_1", 0.0)
    m6 = spy_row.get("spy_mom_6_1", 0.0)
    r21 = spy_row.get("spy_ret_21d", 0.0)
    streak = spy_row.get("spy_below_200_streak", 0.0)

    # Hostile: strong bear evidence — go to cash
    if r21 <= -0.08:
        return "hostile"
    if (m6 <= -0.05 and r21 <= -0.03):
        return "hostile"
    if (dsma <= -0.05 and m12 <= 0.0 and dd52 <= -0.15):
        return "hostile"

    # Recovery: post-bear bounce — concentrate
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"

    # Risk-on: confirmed uptrend
    if dsma > 0 and m12 >= 0.05:
        if breadth is None or breadth >= 0.45:
            return "risk_on"

    return "mixed"


def build_breadth_panel() -> pd.DataFrame:
    """For each asof, fraction of all panel tickers with d_sma200 > 0."""
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        ao = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "d_sma200" not in df.columns or len(df) == 0:
            continue
        ds = df["d_sma200"].dropna()
        if len(ds) == 0:
            continue
        breadth = float((ds > 0).mean())
        # Cross-sectional dispersion of 21d returns
        dr = df["ret_21d"].dropna() if "ret_21d" in df.columns else pd.Series([])
        disp = float(dr.std()) if len(dr) >= 30 else np.nan
        rows.append({"asof": ao, "breadth_above_200": breadth, "dispersion_21d": disp})
    return pd.DataFrame(rows).set_index("asof")


# ---------------------------------------------------------------------------
# Per-stock trend eligibility
# ---------------------------------------------------------------------------
TREND_GATE_DEFAULT = {
    # All conditions soft-AND'd; ticker passes if it satisfies ALL.
    "mom_12_1_min": -0.30,         # allow mild downtrend
    "mom_3_min": -0.15,            # not in active freefall
    "d_sma200_min": -0.10,         # within 10% of 200dma OR above
    "dd_from_52wh_min": -0.55,     # not in death-spiral (positive magnitude in feature → < 0.55 means dd magnitude < 0.55)
    "frac_above_50dma_1y_min": 0.20,  # at least 20% of last year above 50dma
}


def compute_trend_eligibility(feat_df: pd.DataFrame,
                              gate: dict | None = None) -> pd.Series:
    """Boolean series: True if ticker is eligible by trend gate."""
    gate = gate or TREND_GATE_DEFAULT
    if feat_df is None or len(feat_df) == 0:
        return pd.Series(dtype=bool)
    elig = pd.Series(True, index=feat_df.index)
    if "mom_12_1" in feat_df.columns:
        elig &= feat_df["mom_12_1"].fillna(0.0) >= gate["mom_12_1_min"]
    if "mom_3" in feat_df.columns:
        elig &= feat_df["mom_3"].fillna(0.0) >= gate["mom_3_min"]
    if "d_sma200" in feat_df.columns:
        elig &= feat_df["d_sma200"].fillna(0.0) >= gate["d_sma200_min"]
    if "dd_from_52wh" in feat_df.columns:
        # dd_from_52wh is stored as positive magnitude; "no death spiral" means dd <= 0.55
        elig &= feat_df["dd_from_52wh"].fillna(0.0) <= -gate["dd_from_52wh_min"]
    if "frac_above_50dma_1y" in feat_df.columns:
        elig &= feat_df["frac_above_50dma_1y"].fillna(0.0) >= gate["frac_above_50dma_1y_min"]
    return elig


def trend_eligibility_panel(asofs: list[pd.Timestamp],
                            gate: dict | None = None) -> pd.DataFrame:
    """Returns long panel: asof, ticker, eligible (bool)."""
    rows = []
    for ao in asofs:
        f = FEATURES_DIR / f"{pd.Timestamp(ao).date()}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        elig = compute_trend_eligibility(df, gate)
        out = elig.to_frame("eligible")
        out["asof"] = pd.Timestamp(ao)
        out["ticker"] = out.index
        rows.append(out.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Concentration / gross from regime
# ---------------------------------------------------------------------------
def regime_to_sizing(regime: str) -> dict:
    """Returns dict: K_target, gross."""
    return {
        "hostile":  {"K": 0,  "gross": 0.0},
        "mixed":    {"K": 5,  "gross": 0.7},
        "risk_on":  {"K": 3,  "gross": 1.0},
        "recovery": {"K": 3,  "gross": 1.0},
    }[regime]


if __name__ == "__main__":
    # Smoke test
    asofs_all = sorted(p.stem for p in FEATURES_DIR.glob("*.parquet"))
    sample = [pd.Timestamp(a) for a in (asofs_all[100], asofs_all[200], asofs_all[-1])]
    print(f"smoke test asofs: {[a.date() for a in sample]}")
    for ao in sample:
        f = FEATURES_DIR / f"{ao.date()}.parquet"
        df = pd.read_parquet(f) if f.exists() else None
        if df is None:
            continue
        elig = compute_trend_eligibility(df)
        print(f"  {ao.date()}: total={len(df)} eligible={int(elig.sum())} "
              f"({elig.sum()/len(df):.1%})")
