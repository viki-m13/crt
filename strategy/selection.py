"""Selection logic for the Pre-Runner Footprint strategy.

The composite score combines:

  S(stock) = w_crt * z(crt_6m) +
            w_arch * (-z(prerunner_dist)) +
            w_filter * archetype_filter +
            w_rbi * z(rbi_60) +
            w_cst * z(cst_score) +
            w_long_term * z(trend_health_5y)

But the primary mechanism is a HARD GATE plus a SOFT RANK.

  HARD GATE (must satisfy):
    - vol_3m > q40    (above-median realized vol)
    - dd_from_52wh > 0.20    (at least 20% off 52wh)
    - drawdown_age_days > 90 (drawdown is at least 4mo old)
    - accel > 0      (selling decelerating)
    - rank_now > rank_6m_ago - 0.10  (rank not catastrophically falling)
    - trend_health_5y > 0.30  (long-term still has at least some health)

  SOFT RANK on the gated subset:
    score = 0.40 * z(crt_6m) +
            0.20 * z(rbi_60) +
            0.20 * (-z(prerunner_dist)) +
            0.10 * z(cst_score) +
            0.10 * z(trend_health_5y)

The hard gate selects "is in pre-runner footprint and stabilized".
The soft rank picks the ones with the best covert-leadership signal.

Then the score is used by the existing compound_engine harness with
top_k = 5 and monthly rebalance.

A DEFENSIVE CAP: if SPY 200dma is < -10% AND SPY rsi_14 < 35 (deep
bear), return all NaN -> hold cash. (Same regime gate as
strategy_rotation in REPORT.md.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        sd = s.std()
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd
    return (s - med) / (mad * 1.4826)


def quantile(s: pd.Series, q: float) -> float:
    s = s.astype(float).dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def prerunner_score(df: pd.DataFrame,
                     w_crt: float = 0.40,
                     w_rbi: float = 0.20,
                     w_arch: float = 0.20,
                     w_cst: float = 0.10,
                     w_qual: float = 0.10,
                     gate_vol_quantile: float = 0.40,
                     gate_dd: float = 0.20,
                     gate_age_days: float = 90.0,
                     gate_trend_health: float = 0.30,
                     defensive_dsma: float = -0.10,
                     defensive_rsi: float = 35.0) -> pd.Series:
    """Score function over a per-asof feature DataFrame.

    Returns a Series of scores indexed by ticker.  NaN means excluded.
    """
    # Defensive regime gate
    if "SPY" in df.index:
        spy = df.loc["SPY"]
        spy_dsma = spy.get("d_sma200", np.nan)
        spy_rsi = spy.get("rsi_14", np.nan)
        if (np.isfinite(spy_dsma) and spy_dsma < defensive_dsma and
            np.isfinite(spy_rsi) and spy_rsi < defensive_rsi):
            return pd.Series(np.nan, index=df.index)

    out = pd.Series(np.nan, index=df.index)

    needed = ["vol_3m", "dd_from_52wh", "drawdown_age_days", "accel",
              "trend_health_5y", "crt_6m", "rbi_60", "prerunner_dist",
              "cst_score", "rank_now", "rank_6m_ago"]
    for c in needed:
        if c not in df.columns:
            return out

    # Compute hard-gate mask
    vol_q = quantile(df["vol_3m"], gate_vol_quantile)

    gate = (
        (df["vol_3m"] > vol_q) &
        (df["dd_from_52wh"] > gate_dd) &
        (df["drawdown_age_days"] > gate_age_days) &
        (df["accel"] > 0) &
        (df["trend_health_5y"] > gate_trend_health) &
        (df["rank_now"] > df["rank_6m_ago"] - 0.10)
    )

    # Compute soft scores (z-score within universe, not just gated set —
    # so the score is comparable across rebalances)
    z_crt = zscore(df["crt_6m"])
    z_rbi = zscore(df["rbi_60"])
    # prerunner_dist: lower is better, so negate
    z_arch = -zscore(df["prerunner_dist"])
    z_cst = zscore(df["cst_score"])
    z_qual = zscore(df["trend_health_5y"])

    composite = (w_crt * z_crt + w_rbi * z_rbi + w_arch * z_arch +
                 w_cst * z_cst + w_qual * z_qual)

    out.loc[gate.fillna(False)] = composite.loc[gate.fillna(False)]
    # Exclude benchmarks from picks
    for ex in ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"):
        if ex in out.index:
            out.loc[ex] = np.nan
    return out


def make_strategy(top_k: int = 5, **kwargs):
    """Wrapper that returns a Strategy compatible with compound_engine."""
    from experiments.monthly_dca.compound_engine import Strategy
    return Strategy(
        name="prerunner_v1",
        score_fn=lambda df: prerunner_score(df, **kwargs),
        top_k=top_k,
        description="Pre-Runner Footprint with Cross-Sectional Rank Trajectory",
    )


def make_no_gate_strategy(top_k: int = 5):
    """Diagnostic: same composite, no hard gate."""
    from experiments.monthly_dca.compound_engine import Strategy
    def _score(df):
        out = pd.Series(np.nan, index=df.index)
        needed = ["crt_6m", "rbi_60", "prerunner_dist", "cst_score",
                  "trend_health_5y"]
        if not all(c in df.columns for c in needed):
            return out
        z_crt = zscore(df["crt_6m"])
        z_rbi = zscore(df["rbi_60"])
        z_arch = -zscore(df["prerunner_dist"])
        z_cst = zscore(df["cst_score"])
        z_qual = zscore(df["trend_health_5y"])
        composite = 0.40*z_crt + 0.20*z_rbi + 0.20*z_arch + 0.10*z_cst + 0.10*z_qual
        for ex in ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"):
            if ex in composite.index:
                composite.loc[ex] = np.nan
        return composite
    return Strategy(name="prerunner_no_gate", score_fn=_score, top_k=top_k,
                    description="Pre-Runner composite, no hard gate")


def make_crt_only_strategy(top_k: int = 5):
    """Diagnostic: pure CRT with archetype filter only."""
    from experiments.monthly_dca.compound_engine import Strategy
    def _score(df):
        out = pd.Series(np.nan, index=df.index)
        if "crt_6m" not in df.columns or "vol_3m" not in df.columns:
            return out
        # Loose footprint filter
        gate = (
            (df["vol_3m"] > df["vol_3m"].quantile(0.40)) &
            (df["dd_from_52wh"] > 0.20)
        )
        out.loc[gate.fillna(False)] = df.loc[gate.fillna(False), "crt_6m"]
        for ex in ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"):
            if ex in out.index:
                out.loc[ex] = np.nan
        return out
    return Strategy(name="crt_only", score_fn=_score, top_k=top_k,
                    description="Pure CRT score with vol+dd gate")
