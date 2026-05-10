"""Cheap experiment harness for the YLOka session.

Reuses the cached v3 GBM predictions (ml_preds_v2.parquet) and PIT membership
(sp500_membership_monthly.parquet) so we don't have to retrain the model for
every scorer/portfolio variant. The simulator mirrors the production
sp500_pit_extended_sweep.simulate_variant logic but is self-contained so we can
mutate scoring, picking, weighting, regime gate, and exit logic independently.

Conventions:
- One asof = one month-end. The "asof" timestamp comes from ml_preds_v2.
- At month-end T we compute features at T (already done), pick a basket, and
  earn the next-month return (T -> T+1). This matches the production engine.
- Cost is `cost_bps_per_leg / 10000` applied at every rebalance.
- Cash regime earns the configured `cash_yield_apr / 12` per month.
- "Walk-forward OOS" means: research window 2003-09 -> 2024-04. Holdout is
  2024-05 -> 2026-04 and is NEVER queried by this harness.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

CACHE = Path("/home/user/crt/experiments/monthly_dca/cache")
PIT_DIR = CACHE / "v2" / "sp500_pit"

DATA = Path("/home/user/crt/data/YLOka")
RUNS = Path("/home/user/crt/backtests/YLOka/runs")
LOG_PATH = Path("/home/user/crt/backtests/YLOka/experiment_log.csv")

RESEARCH_END = pd.Timestamp("2024-04-30")
HOLDOUT_START = pd.Timestamp("2024-05-01")
HOLDOUT_END = pd.Timestamp("2026-04-30")

EXCLUDE = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}


@dataclass
class StratConfig:
    name: str
    K: int = 3
    hold_months: int = 6
    weighting: str = "ew"  # ew | conv | invvol
    cost_bps_per_leg: float = 5.0
    cash_yield_apr: float = 0.0
    crash_gate: bool = True  # use the v3 'tight' gate
    soft_cash: bool = False  # H4 - smooth de-risk
    # picker hooks
    score_fn_name: str = "ml_3plus6"  # which scorer to use
    pick_filter: str = "none"  # "none" | "donchian130" | "accel"
    conv_lambda: float = 0.0  # H2 - 0=ew; >0 = softmax tilt by score z
    cash_score_floor: float = 0.0  # H2 - if top score < floor, partial cash
    # H7 - dispersion-conditional K. When wide=K_low (concentrate); when narrow=K_high (diversify)
    dispersion_K: bool = False
    K_low_dispersion: int = 5    # when xs mom dispersion is narrow (low) -> diversify
    K_high_dispersion: int = 3   # when xs mom dispersion is wide (high) -> concentrate
    dispersion_threshold_pctile: float = 0.50  # >= median dispersion = wide


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    """Long-format (asof, ticker, pred_1m, pred_3m, pred_6m, pred, score)."""
    p = pd.read_parquet(DATA / "pit_panel_with_scores.parquet")
    p["asof"] = pd.to_datetime(p["asof"])
    return p[~p["ticker"].isin(EXCLUDE)].copy()


def load_panel_ensemble() -> pd.DataFrame:
    """Same as load_panel plus pred_12m + pred_12m_cls (H1 ensemble heads)."""
    p = pd.read_parquet(DATA / "pit_panel_with_12m.parquet")
    p["asof"] = pd.to_datetime(p["asof"])
    return p[~p["ticker"].isin(EXCLUDE)].copy()


def load_xs_dispersion() -> pd.DataFrame:
    """Per-asof XS dispersion of mom_12_1 (and mom_6_1, vol_12m). H7 input."""
    return pd.read_parquet(DATA / "xs_dispersion.parquet")


def load_monthly_returns() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")


def load_prices() -> pd.DataFrame:
    """Wide daily price panel for breakout / Donchian features."""
    p = pd.read_parquet(CACHE / "prices_extended.parquet")
    p.index = pd.to_datetime(p.index)
    return p


def load_spy_features() -> pd.DataFrame:
    """SPY-derived features at each cached month-end. Mirrors production
    sp500_pit_extended_sweep.load_spy_features so the regime classifier
    matches v3 exactly."""
    fdir = CACHE / "features"
    rows = []
    for f in sorted(fdir.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_dd_from_52wh": float(spy.get("dd_from_52wh", 0.0)),
            "spy_vol_12m": float(spy.get("vol_12m", spy.get("vol_1y", 0.15))),
        })
    return pd.DataFrame(rows).set_index("asof")


# ----------------------------------------------------------------------
# Regime gate (matches v3 'tight' from sp500_pit_extended_sweep)
# ----------------------------------------------------------------------

def regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if pd.isna(r21):
        return "normal"
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def soft_cash_weight(s: dict) -> float:
    """H4 - continuous risk_off in [0, 1]; 0 = full equity, 1 = full cash."""
    dd = s.get("spy_dd_from_52wh", 0.0)
    vol = s.get("spy_vol_12m", 0.15)
    if pd.isna(dd) or pd.isna(vol):
        return 0.0
    risk = max(0.0, max(
        max(0.0, -dd - 0.05) / 0.10,   # dd: -5%->0, -15%->1
        max(0.0, vol - 0.20) / 0.15,   # vol: 20%->0, 35%->1
    ))
    return float(min(risk, 1.0))


# ----------------------------------------------------------------------
# Scorers and pickers
# ----------------------------------------------------------------------

def score_ml_3plus6(panel_at: pd.DataFrame) -> pd.Series:
    """v3 baseline: mean of 3m and 6m predictions."""
    return (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2


def score_ml_136(panel_at: pd.DataFrame) -> pd.Series:
    """1m + 3m + 6m equal-weight (toy multi-horizon)."""
    return (panel_at["pred_1m"] + panel_at["pred_3m"] + panel_at["pred_6m"]) / 3


def _xs_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank in [0, 1] within the given panel-at slice."""
    return s.rank(pct=True)


def score_ens_3_6_12(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v1 — equal-weight rank ensemble of 3m, 6m, 12m heads.

    Falls back to ml_3plus6 if pred_12m missing (early-period coverage hole)."""
    if "pred_12m" not in panel_at.columns:
        return score_ml_3plus6(panel_at)
    r3 = _xs_rank(panel_at["pred_3m"])
    r6 = _xs_rank(panel_at["pred_6m"])
    r12 = _xs_rank(panel_at["pred_12m"])
    # When pred_12m is NaN, fall back to (r3 + r6)/2
    base = (r3 + r6) / 2
    full = (r3 + r6 + r12) / 3
    return full.where(panel_at["pred_12m"].notna(), base)


def score_ens_3_6_12_cls(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v2 — rank ensemble of 3m, 6m, 12m + classifier prob bonus.

    score = 0.7 * mean(rank(pred_3m, pred_6m, pred_12m)) + 0.3 * pred_12m_cls
    """
    base = score_ens_3_6_12(panel_at)
    cls = panel_at.get("pred_12m_cls")
    if cls is None or cls.isna().all():
        return base
    cls_rank = _xs_rank(cls.fillna(cls.median()))
    return 0.7 * base + 0.3 * cls_rank


def score_ens_3_6_12_invvol(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v3 — score / vol_1y to penalise high-vol picks within the ensemble."""
    base = score_ens_3_6_12(panel_at)
    vol = panel_at.get("vol_1y")
    if vol is None or vol.isna().all():
        return base
    v = vol.copy()
    v = v.where(v > 0.05, 0.40)
    return base - 0.10 * _xs_rank(v)


def score_ml_3plus6_plus_12m_tilt(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v5 — keep baseline ml_3plus6 magnitudes; add small 12m-rank tilt.

    score = ml_3plus6 + 0.05 * (rank(pred_12m) - 0.5)
    The tilt is small enough to act as a tie-breaker on the baseline top picks
    rather than re-rank the universe.
    """
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    if "pred_12m" not in panel_at.columns:
        return base
    r12 = _xs_rank(panel_at["pred_12m"].fillna(panel_at["pred_12m"].median()))
    return base + 0.05 * (r12 - 0.5)


def score_ml_3plus6_plus_12m_tilt_strong(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v6 — same as v5 but with 0.15 tilt weight."""
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    if "pred_12m" not in panel_at.columns:
        return base
    r12 = _xs_rank(panel_at["pred_12m"].fillna(panel_at["pred_12m"].median()))
    return base + 0.15 * (r12 - 0.5)


def score_ml_3plus6_cls_filter(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v7 — classifier hard filter: drop picks where prob(top-quintile) < 0.18.

    Returns the baseline ml_3plus6 score, but sets score to -1e9 for any
    name where the classifier predicts below-threshold probability of being
    a top-quintile performer. Effectively excludes those names from the basket.
    """
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    cls = panel_at.get("pred_12m_cls")
    if cls is None or cls.isna().all():
        return base
    keep = cls.fillna(1.0) >= 0.18
    return base.where(keep, -1e9)


def score_ml_3plus6_cls_filter_tight(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v8 — tighter classifier filter at 0.25."""
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    cls = panel_at.get("pred_12m_cls")
    if cls is None or cls.isna().all():
        return base
    keep = cls.fillna(1.0) >= 0.25
    return base.where(keep, -1e9)


def score_ml_3plus6_cls_tilt(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v9 — classifier as soft tilt: score = ml_3plus6 + 0.05 * (cls - 0.2)"""
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    cls = panel_at.get("pred_12m_cls")
    if cls is None or cls.isna().all():
        return base
    return base + 0.05 * (cls.fillna(0.2) - 0.2)


def score_ens_36_12wt(panel_at: pd.DataFrame) -> pd.Series:
    """H1 v4 — biased toward longer horizons: 0.2 * 3m + 0.4 * 6m + 0.4 * 12m
    using rank-transformed predictions."""
    if "pred_12m" not in panel_at.columns:
        return score_ml_3plus6(panel_at)
    r3 = _xs_rank(panel_at["pred_3m"])
    r6 = _xs_rank(panel_at["pred_6m"])
    r12 = _xs_rank(panel_at["pred_12m"])
    base = 0.5 * r3 + 0.5 * r6
    full = 0.2 * r3 + 0.4 * r6 + 0.4 * r12
    return full.where(panel_at["pred_12m"].notna(), base)


SCORERS = {
    "ml_3plus6": score_ml_3plus6,
    "ml_136": score_ml_136,
    "ens_3_6_12": score_ens_3_6_12,
    "ens_3_6_12_cls": score_ens_3_6_12_cls,
    "ens_3_6_12_invvol": score_ens_3_6_12_invvol,
    "ens_36_12wt": score_ens_36_12wt,
    "ml_3plus6_plus_12m_tilt": score_ml_3plus6_plus_12m_tilt,
    "ml_3plus6_plus_12m_tilt_strong": score_ml_3plus6_plus_12m_tilt_strong,
    "ml_3plus6_cls_filter": score_ml_3plus6_cls_filter,
    "ml_3plus6_cls_filter_tight": score_ml_3plus6_cls_filter_tight,
    "ml_3plus6_cls_tilt": score_ml_3plus6_cls_tilt,
}


def pick_top_k(scored: pd.DataFrame, K: int) -> pd.DataFrame:
    return scored.sort_values("score", ascending=False).head(K).reset_index(drop=True)


def filter_donchian130(picks: pd.DataFrame, prices: pd.DataFrame, asof: pd.Timestamp,
                        threshold: float = 0.05) -> pd.DataFrame:
    """Keep only picks whose price at asof is within `threshold` of 130d high."""
    if asof not in prices.index:
        # find nearest prior trading day
        idx = prices.index
        pos = idx.searchsorted(asof, side="right") - 1
        if pos < 0:
            return picks
        asof = idx[pos]
    window = prices.loc[:asof].iloc[-130:]
    keep = []
    for _, row in picks.iterrows():
        tk = row["ticker"]
        if tk not in window.columns:
            continue
        s = window[tk].dropna()
        if len(s) < 100:
            continue
        if s.iloc[-1] / s.max() >= 1 - threshold:
            keep.append(row)
    if not keep:
        return picks  # fallback
    return pd.DataFrame(keep).reset_index(drop=True)


def filter_accel(picks: pd.DataFrame, panel_at: pd.DataFrame) -> pd.DataFrame:
    """Keep picks where 1-month pred > average 3m+6m pred (accel)."""
    merged = picks.merge(panel_at[["ticker", "pred_1m", "pred_3m", "pred_6m"]],
                         on="ticker", how="left", suffixes=("", "_p"))
    accel = merged["pred_1m"] >= 0.5 * (merged["pred_3m"] + merged["pred_6m"])
    keep = merged[accel].copy()
    if keep.empty:
        return picks
    cols = picks.columns.tolist()
    return keep[cols].reset_index(drop=True)


# ----------------------------------------------------------------------
# Weight functions
# ----------------------------------------------------------------------

def weights_ew(picks: pd.DataFrame, *_) -> np.ndarray:
    return np.ones(len(picks)) / len(picks)


def weights_conv(picks: pd.DataFrame, panel_at: pd.DataFrame, lam: float) -> np.ndarray:
    """H2 conviction-spread sizing.

    weights ∝ softmax(λ · z), z = (score - median(universe)) / mad(universe)
    """
    if lam <= 0:
        return weights_ew(picks)
    s = picks["score"].values.astype(float)
    u = panel_at["score"].values.astype(float)
    med = np.median(u)
    mad = np.median(np.abs(u - med)) + 1e-9
    z = (s - med) / mad
    w = np.exp(lam * z)
    w = w / w.sum()
    # Cap any single name at 0.6
    w = np.minimum(w, 0.6)
    return w / w.sum()


# ----------------------------------------------------------------------
# Main simulator
# ----------------------------------------------------------------------

def simulate(cfg: StratConfig, panel: pd.DataFrame, mr: pd.DataFrame,
             spy_features: pd.DataFrame, prices: Optional[pd.DataFrame] = None,
             xs_dispersion: Optional[pd.DataFrame] = None,
             start: Optional[pd.Timestamp] = None,
             end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Run the strategy and return per-month equity / regime / picks."""
    months = sorted(panel["asof"].unique())
    if start is not None:
        months = [m for m in months if m >= start]
    if end is not None:
        months = [m for m in months if m <= end]

    by_asof = {pd.Timestamp(d): g for d, g in panel.groupby("asof")}
    mr_idx = mr.index
    cf = cfg.cost_bps_per_leg / 10000.0
    cash_m = (1 + cfg.cash_yield_apr) ** (1 / 12) - 1

    score_fn = SCORERS[cfg.score_fn_name]
    equity = 1.0
    cur_picks: list[str] = []
    cur_weights: np.ndarray = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        m = pd.Timestamp(m)
        do_reb = (i == 0) or (held_for >= cfg.hold_months) or cash

        # Compute regime / soft-cash from SPY features at m
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        if cfg.crash_gate:
            regime = regime_tight(spy_now)
        else:
            regime = "normal"

        soft_cash_w = soft_cash_weight(spy_now) if cfg.soft_cash else 0.0

        if do_reb:
            sub = by_asof.get(m, pd.DataFrame()).copy()
            if not sub.empty:
                sub["score"] = score_fn(sub)
                sub = sub.dropna(subset=["score"])
            if regime == "crash" and cfg.crash_gate:
                cur_picks, cur_weights, cash = [], np.array([]), True
                held_for = 0
            elif sub.empty:
                cur_picks, cur_weights, cash = [], np.array([]), True
                held_for = 0
            else:
                K = cfg.K
                # H7 - dispersion-conditional K via precomputed XS dispersion table
                if cfg.dispersion_K and xs_dispersion is not None:
                    if m in xs_dispersion.index:
                        disp_now = float(xs_dispersion.loc[m, "xs_mom12_std"])
                        # Threshold: historical percentile of dispersion up to and including m
                        hist = xs_dispersion.loc[:m, "xs_mom12_std"]
                        if len(hist) > 12:
                            thr = float(hist.quantile(cfg.dispersion_threshold_pctile))
                            if disp_now >= thr:
                                K = cfg.K_high_dispersion  # wide -> concentrate
                            else:
                                K = cfg.K_low_dispersion   # narrow -> diversify
                picks = pick_top_k(sub, K)
                if cfg.pick_filter == "donchian130":
                    if prices is None:
                        raise RuntimeError("donchian filter needs prices panel")
                    picks = filter_donchian130(picks, prices, m)
                elif cfg.pick_filter == "accel":
                    picks = filter_accel(picks, sub)

                # H2 conviction sizing
                if cfg.weighting == "conv":
                    w = weights_conv(picks, sub, cfg.conv_lambda)
                elif cfg.weighting == "ew":
                    w = weights_ew(picks)
                else:
                    w = weights_ew(picks)

                # Cash floor: if top score < threshold, hold partial cash
                eq_w = 1.0
                if cfg.cash_score_floor > 0 and len(sub) > 0:
                    top_score = float(picks["score"].iloc[0])
                    hist_floor = sub["score"].quantile(cfg.cash_score_floor)
                    if top_score < hist_floor:
                        eq_w = max(0.0, top_score / hist_floor) if hist_floor > 0 else 1.0

                cur_picks = picks["ticker"].tolist()
                cur_weights = w * eq_w
                cash = False
                held_for = 0

        # Realize next-month return
        pos1 = mr_idx.searchsorted(m)
        if cash or len(cur_picks) == 0:
            ret_m = cash_m
        elif pos1 + 1 >= len(mr_idx) or pos1 - 1 < 0:
            ret_m = 0.0
        else:
            cands = []
            for j in (pos1 - 1, pos1):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in mr.columns:
                        r = mr.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                eq_part = float((pick_rets * cur_weights).sum())
                # cash slice gets cash yield
                cash_part_w = max(0.0, 1.0 - cur_weights.sum())
                ret_m = eq_part + cash_part_w * cash_m
                # Soft-cash overlay: blend with cash
                if cfg.soft_cash:
                    ret_m = (1 - soft_cash_w) * ret_m + soft_cash_w * cash_m

        # Apply transaction cost at rebalance
        if do_reb and not cash and len(cur_picks) > 0:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m
        held_for += 1
        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": "cash" if cash else regime,
            "soft_cash_w": soft_cash_w,
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def metrics(eq: pd.DataFrame) -> dict:
    r = eq["ret_m"].astype(float)
    n = len(r)
    if n == 0:
        return {}
    ec = (1 + r).cumprod()
    cagr = float(ec.iloc[-1] ** (12 / n) - 1)
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    peak = ec.cummax()
    mdd = float(((ec - peak) / peak).min())
    sortino = float(r.mean() / r.clip(upper=0).std() * np.sqrt(12)) if r.clip(upper=0).std() > 0 else 0.0
    return dict(
        n_months=n, cagr=cagr, sharpe=sh, sortino=sortino, max_dd=mdd,
        best_m=float(r.max()), worst_m=float(r.min()),
        positive_months=int((r > 0).sum()), total_months=n,
        cash_months=int((eq["regime"] == "cash").sum()),
    )


def rolling_5y_cagr(eq: pd.DataFrame, window_m: int = 60) -> pd.Series:
    """Non-overlapping rolling 5y CAGR distribution."""
    r = eq["ret_m"].astype(float)
    if len(r) < window_m:
        return pd.Series([], dtype=float)
    starts = list(range(0, len(r) - window_m + 1, window_m // 4))  # 75%-overlap rolling
    out = []
    for s in starts:
        sub = r.iloc[s:s + window_m]
        cagr = (1 + sub).prod() ** (12 / len(sub)) - 1
        out.append({"start_idx": s, "start_date": eq["date"].iloc[s],
                     "end_date": eq["date"].iloc[s + window_m - 1], "cagr": cagr})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Run logging
# ----------------------------------------------------------------------

def run_and_log(cfg: StratConfig, panel: pd.DataFrame, mr: pd.DataFrame,
                spy_features: pd.DataFrame, prices: Optional[pd.DataFrame] = None,
                xs_dispersion: Optional[pd.DataFrame] = None,
                window: str = "research") -> dict:
    """Run a config, write manifest, append to log csv. Returns metrics dict."""
    if window == "research":
        start, end = None, RESEARCH_END
    elif window == "holdout":
        start, end = HOLDOUT_START, HOLDOUT_END
    elif window == "full":
        start, end = None, None
    else:
        raise ValueError(window)

    t0 = time.time()
    eq = simulate(cfg, panel, mr, spy_features, prices=prices,
                   xs_dispersion=xs_dispersion, start=start, end=end)
    met = metrics(eq)
    met["window"] = window
    met["wall_time_s"] = round(time.time() - t0, 2)

    cfg_dict = asdict(cfg)
    cfg_hash = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:10]
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{cfg.name}_{cfg_hash}_{window}"
    rdir = RUNS / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    eq.to_parquet(rdir / "equity.parquet")
    with open(rdir / "manifest.json", "w") as f:
        json.dump({"cfg": cfg_dict, "metrics": met, "run_id": run_id,
                    "git_sha": _git_sha()}, f, indent=2)
    _append_log(run_id, cfg_dict, met)
    return met


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", "/home/user/crt", "rev-parse", "HEAD"],
                                          text=True).strip()
    except Exception:
        return "unknown"


def _append_log(run_id: str, cfg: dict, met: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {**{f"cfg_{k}": v for k, v in cfg.items()}, **met, "run_id": run_id}
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    print("Loading panel + monthly returns + SPY features ...")
    panel = load_panel()
    mr = load_monthly_returns()
    spy = load_spy_features()
    prices = None  # only loaded if needed
    print(f"panel: {panel.shape}, mr: {mr.shape}, spy: {spy.shape}")

    base = StratConfig(name="v3_baseline_repro")
    met = run_and_log(base, panel, mr, spy, window="research")
    print("Baseline v3 reproduction (research window):")
    for k, v in met.items():
        print(f"  {k}: {v}")
