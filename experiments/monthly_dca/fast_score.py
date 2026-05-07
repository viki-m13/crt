"""Once forward returns are cached, evaluating a strategy is just:
  1. score every (asof, ticker) using cached features
  2. pick top-K per asof
  3. join cached forward returns
  4. compute summary metrics

This module exposes `evaluate_strategy(score_fn, top_k=..., rule_name=...)` that
runs in ~1s for 7y of monthly data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import (
    CACHE,
    FEATURES_DIR,
    load_panel,
    xirr,
)


BENCH_EXCLUDED = ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA",
                  "BTC-USD", "ETH-USD")  # ETFs and crypto, not equities


@lru_cache(maxsize=1)
def load_features_long() -> pd.DataFrame:
    """Concat all per-month feature parquets into one long DataFrame.

    Index: MultiIndex (asof, ticker). Columns: feature names.
    """
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        df = df.copy()
        df["asof"] = pd.Timestamp(p.stem)
        df["ticker"] = df.index
        frames.append(df.reset_index(drop=True))
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.set_index(["asof", "ticker"]).sort_index()
    return out


@lru_cache(maxsize=1)
def load_fwd() -> pd.DataFrame:
    p = CACHE / "fwd_returns.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Run forward_returns.py first; missing {p}")
    df = pd.read_parquet(p)
    df = df.set_index(["asof", "ticker"]).sort_index()
    return df


def universe_filter(scores: pd.Series) -> pd.Series:
    """Drop benchmarks from candidate set."""
    bad = [t for t in BENCH_EXCLUDED if t in scores.index.get_level_values("ticker")]
    return scores[~scores.index.get_level_values("ticker").isin(bad)]


@dataclass
class EvalResult:
    name: str
    picks: pd.DataFrame             # asof, ticker, score
    summary: pd.DataFrame           # one row per rule
    spy_dca_cagr: float
    eval_at: pd.Timestamp


def evaluate_strategy(
    score_fn: Callable[[pd.DataFrame], pd.Series],
    top_k: int = 5,
    name: str = "strategy",
    start: str | None = None,
    end: str | None = None,
    panel: pd.DataFrame | None = None,
    delist_prob_annual: float = 0.04,
    delist_iters: int = 200,
    delist_wipeout: float = -1.0,
) -> EvalResult:
    feats = load_features_long()
    fwd = load_fwd()

    # Filter by date window
    if start is not None:
        feats = feats.loc[feats.index.get_level_values("asof") >= pd.Timestamp(start)]
        fwd = fwd.loc[fwd.index.get_level_values("asof") >= pd.Timestamp(start)]
    if end is not None:
        feats = feats.loc[feats.index.get_level_values("asof") <= pd.Timestamp(end)]
        fwd = fwd.loc[fwd.index.get_level_values("asof") <= pd.Timestamp(end)]

    # Score per row
    feats_indexed = feats.copy()
    scores = score_fn(feats_indexed)
    scores = scores.dropna()
    scores = universe_filter(scores)

    # Top-K per asof
    df = scores.reset_index()
    df.columns = ["asof", "ticker", "score"]
    df = df.sort_values(["asof", "score"], ascending=[True, False])
    picks = df.groupby("asof", group_keys=False).head(top_k).reset_index(drop=True)

    if picks.empty:
        return EvalResult(name=name, picks=picks, summary=pd.DataFrame(),
                          spy_dca_cagr=float("nan"),
                          eval_at=pd.NaT)

    # Join forward returns
    fwd_reset = fwd.reset_index()
    merged = picks.merge(fwd_reset, on=["asof", "ticker"], how="left")
    if panel is None:
        panel = load_panel()
    eval_at = panel.index.max()

    # Build summary across rules
    rule_cols = [c for c in fwd.columns if c.startswith("ret__")]
    summary_rows = []

    rng = np.random.default_rng(0)

    for rc in rule_cols:
        rule_name = rc[len("ret__"):]
        f = merged[rc].to_numpy(dtype=float)
        days_col = f"days__{rule_name}"
        d = merged[days_col].to_numpy(dtype=float) if days_col in merged.columns else np.full(len(f), np.nan)
        valid = np.isfinite(f)
        if not valid.any():
            continue
        fv = f[valid]
        dv = d[valid]
        ah = pd.to_datetime(merged["asof"].to_numpy()[valid])

        # SPY held to eval_at over same window
        spy = panel["SPY"]
        bv = []
        for asof_t in ah:
            pos = panel.index.searchsorted(asof_t)
            if pos >= len(panel.index):
                bv.append(np.nan)
                continue
            arr = spy.iloc[pos:].to_numpy(dtype=float)
            mask = np.isfinite(arr)
            if mask.any():
                bv.append(arr[mask][-1] / arr[0] - 1.0)
            else:
                bv.append(np.nan)
        bv = np.asarray(bv, dtype=float)

        win = float((fv > 0).mean())
        beat = float((fv > bv).mean())

        # Bias-corrected via synthetic delistings
        if delist_iters > 0:
            days_to_eval = np.asarray([(eval_at - t).days for t in ah], dtype=float)
            years_to_eval = np.maximum(days_to_eval, 1.0) / 365.25
            p_del = 1.0 - (1.0 - delist_prob_annual) ** years_to_eval
            wins = []
            means = []
            for it in range(delist_iters):
                u = rng.random(len(fv))
                fv_mc = np.where(u < p_del, delist_wipeout, fv)
                wins.append(float((fv_mc > 0).mean()))
                means.append(float(fv_mc.mean()))
            win_corr = float(np.median(wins))
            mean_corr = float(np.median(means))
        else:
            win_corr = float("nan")
            mean_corr = float("nan")

        # CAGR DCA portfolio
        cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
        cashflows.append((eval_at, float(np.sum(1 + fv))))
        cagr_dca = xirr(cashflows)
        # SPY DCA same dates
        cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
        cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
        cagr_spy = xirr(cashflows_spy)

        # Per-pick CAGR (median)
        years_held = np.maximum(dv, 1) / 252.0
        per_pick_cagr = (1 + fv) ** (1.0 / years_held) - 1.0
        per_pick_cagr_median = float(np.nanmedian(per_pick_cagr))

        summary_rows.append(
            {
                "strategy": name,
                "exit": rule_name,
                "n_picks": int(len(fv)),
                "win_rate": win,
                "win_rate_bias_corr": win_corr,
                "beat_spy_rate": beat,
                "median_ret": float(np.nanmedian(fv)),
                "mean_ret": float(np.nanmean(fv)),
                "p10_ret": float(np.nanpercentile(fv, 10)),
                "p90_ret": float(np.nanpercentile(fv, 90)),
                "per_pick_cagr_median": per_pick_cagr_median,
                "cagr_dca_portfolio": cagr_dca,
                "cagr_spy_dca": cagr_spy,
                "edge_vs_spy_dca": cagr_dca - cagr_spy,
                "mean_ret_bias_corr": mean_corr,
            }
        )

    return EvalResult(name=name, picks=picks, summary=pd.DataFrame(summary_rows),
                      spy_dca_cagr=summary_rows[0]["cagr_spy_dca"] if summary_rows else float("nan"),
                      eval_at=eval_at)


def oracle_bound(top_k: int = 5, rule_name: str = "hold_forever",
                 start: str = "2017-12-31", end: str = "2024-12-31") -> dict:
    """Theoretical ceiling: pick the top-K stocks at each month-end with the
    largest forward return under `rule_name`.
    """
    fwd = load_fwd().reset_index()
    fwd = fwd[(fwd["asof"] >= pd.Timestamp(start)) & (fwd["asof"] <= pd.Timestamp(end))]
    fwd = fwd[~fwd["ticker"].isin(BENCH_EXCLUDED)]
    rc = f"ret__{rule_name}"
    fwd = fwd.dropna(subset=[rc])
    if fwd.empty:
        return {"cagr_dca": float("nan")}
    df = fwd.sort_values(["asof", rc], ascending=[True, False]).groupby("asof", group_keys=False).head(top_k)
    fv = df[rc].to_numpy(dtype=float)

    panel = load_panel()
    eval_at = panel.index.max()
    cashflows = [(pd.Timestamp(t), -1.0) for t in df["asof"]]
    cashflows.append((eval_at, float(np.sum(1 + fv))))
    cagr = xirr(cashflows)

    spy = panel["SPY"]
    bv = []
    for asof_t in df["asof"]:
        pos = panel.index.searchsorted(pd.Timestamp(asof_t))
        if pos >= len(panel.index):
            bv.append(np.nan)
            continue
        arr = spy.iloc[pos:].to_numpy(dtype=float)
        mask = np.isfinite(arr)
        if mask.any():
            bv.append(arr[mask][-1] / arr[0] - 1.0)
        else:
            bv.append(np.nan)
    bv = np.asarray(bv, dtype=float)
    cashflows_spy = [(pd.Timestamp(t), -1.0) for t in df["asof"]]
    cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
    cagr_spy = xirr(cashflows_spy)
    return {
        "rule": rule_name,
        "top_k": top_k,
        "n_picks": len(fv),
        "median_pick_ret": float(np.nanmedian(fv)),
        "win_rate": float((fv > 0).mean()),
        "cagr_dca": cagr,
        "cagr_spy_dca": cagr_spy,
    }
