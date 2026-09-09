"""Matured-Evidence Adaptive Paths — a 90% precision GATE, not a 90% claim.

The requirement was a method that is over 90% accurate about a stock being
higher at least 30 sessions later. Measured on this repo's survivorship-free
panel, a stock is higher 30 sessions later 51.4% of the time and a year later
57.0% of the time (research/uptrend/baserates.py). Getting to 90% on demand,
for any ticker, on any day, is not a thing that exists.

What can exist is a system that REFUSES TO SPEAK unless the evidence for 90%
is already there. That is what this is:

  - it produces a calibrated probability that a name is higher in h sessions;
  - it picks a threshold using only its OWN previously issued forecasts that
    have since matured — never a number fitted to the outcomes it is judged
    on;
  - a threshold is eligible only if the CONSERVATIVE LOWER BOUND on its
    realised precision clears 90%, where the bound counts non-overlapping
    horizon blocks rather than pretending 6.6m stock-days are independent;
  - the bound is Bonferroni-corrected across every horizon x threshold pair
    considered, because scanning eight thresholds and picking the best is how
    a 90% number gets manufactured;
  - and when nothing qualifies, it emits nothing. Silence is the normal
    output, and an empty day is a correct answer.

DYNAMIC AND ADAPTIVE, specifically: three experts (base rate, regularised
logistic, gradient boosting) are mixed by exponentially-decayed recent loss
with a fixed-share floor so a temporarily-bad expert can recover; the mixture
is then recalibrated online against matured forecasts; and the path band is a
conformal radius re-estimated from matured residuals. Nothing is fitted once
and frozen.

THE PATH is a band, not a line: a centre from a supervised location model and
a radius sized so the WHOLE trajectory — every daily close, not sampled
checkpoints — lands inside at the target rate.

Design credit: the MEAP protocol (evidence gate, matured-history calibration,
whole-path conformal radius, family-wise correction) comes from the engine
supplied by the user. What is different here is the data it is pointed at:
the point-in-time panel in research/uptrend/data.py, which carries the 4,725
tickers that stopped trading. On a survivor-only universe the 504-session
base rate is 9.4 points higher than the truth, and every precision figure
inherits that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (30, 63, 126, 252)
THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
EXPERTS = ("base", "linear", "nonlinear")
FEATURES = ("r5", "r21", "r63", "r126", "r252", "mom12_1", "vol21", "vol63",
            "vol252", "vol_ratio", "drawdown", "ma_gap", "relative63",
            "relative252", "up_fraction", "breadth", "dispersion",
            "market63", "market252", "market_vol", "market_drawdown",
            "market_ma_gap")


@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = HORIZONS
    first_fit_year: int = 2000
    evaluation_year: int = 2005
    train_years: int = 10
    minimum_training_rows: int = 1500
    minimum_calibration_dates: int = 24
    expert_half_life_sessions: int = 756
    path_alpha: float = 0.10
    minimum_gate_rows: int = 200
    minimum_gate_blocks: int = 12
    minimum_gate_coverage: float = 0.01
    precision_target: float = 0.90
    family_alpha: float = 0.05
    refit_every: int = 63
    step: int = 21
    names_per_date: int = 400
    # A date with only a handful of live names is not a cross-section; it is
    # noise that would dominate the equal-weighted date weighting. Configurable
    # rather than hardcoded, or the engine cannot be tested on a small panel.
    min_names_per_date: int = 50
    random_state: int = 731

    def __post_init__(self):
        if not self.horizons or any(h < 30 for h in self.horizons):
            raise ValueError("every horizon must be at least 30 sessions")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be distinct")


# --------------------------------------------------------------- features --

def build_features(px: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features. Every column uses data up to and including i.

    Returned long-form with `i` (row index) and `ticker`, so the leakage rule
    downstream is a comparison on integers rather than on dates.
    """
    lp = np.log(px.to_numpy(dtype=np.float32))
    n, k = lp.shape
    F = pd.DataFrame(lp, index=px.index, columns=px.columns)
    r1 = F.diff()

    def rr(w):
        return (F - F.shift(w))

    def vol(w):
        return r1.rolling(w, min_periods=max(5, w // 2)).std() * np.sqrt(252)

    feats = {
        "r5": rr(5), "r21": rr(21), "r63": rr(63), "r126": rr(126),
        "r252": rr(252),
        "mom12_1": rr(252) - rr(21),
        "vol21": vol(21), "vol63": vol(63), "vol252": vol(252),
        "drawdown": F - F.rolling(252, min_periods=60).max(),
        "ma_gap": F - F.rolling(200, min_periods=100).mean(),
        "up_fraction": (r1 > 0).rolling(63, min_periods=30).mean(),
    }
    feats["vol_ratio"] = feats["vol21"] / feats["vol252"].replace(0, np.nan)

    # market = equal-weight mean log price change across live names
    mkt63 = feats["r63"].mean(axis=1)
    mkt252 = feats["r252"].mean(axis=1)
    feats["relative63"] = feats["r63"].sub(mkt63, axis=0)
    feats["relative252"] = feats["r252"].sub(mkt252, axis=0)

    mret = r1.mean(axis=1)
    mlvl = mret.cumsum()
    mvol = mret.rolling(63, min_periods=30).std() * np.sqrt(252)
    mdd = mlvl - mlvl.rolling(252, min_periods=60).max()
    mgap = mlvl - mlvl.rolling(200, min_periods=100).mean()
    breadth = feats["ma_gap"].gt(0).mean(axis=1)
    disp = r1.std(axis=1)

    out = {}
    for name, df in feats.items():
        out[name] = df.to_numpy(dtype=np.float32)
    # Market features are one value per DATE. Broadcasting them to a full
    # (bars x tickers) matrix each — seven of the twenty-two — was 2.8 GB of
    # duplicated scalars and OOM-killed the full-panel run at 14 GB. They are
    # kept one-dimensional and broadcast per row in rows_for_date instead.
    for name, ser in (("market63", mkt63), ("market252", mkt252),
                      ("market_vol", mvol), ("market_drawdown", mdd),
                      ("market_ma_gap", mgap), ("breadth", breadth),
                      ("dispersion", disp)):
        out[name] = ser.to_numpy(dtype=np.float32)
    return out


MARKET_FEATURES = ("market63", "market252", "market_vol", "market_drawdown",
                   "market_ma_gap", "breadth", "dispersion")


def rows_for_date(fmats, i: int, live: np.ndarray) -> pd.DataFrame:
    """One row per live ticker at bar i, with every feature finite."""
    k = len(live)
    cols = {}
    for f in FEATURES:
        v = fmats[f]
        cols[f] = (np.full(k, v[i], dtype=np.float32) if v.ndim == 1
                   else v[i])
    d = pd.DataFrame(cols)
    d["col"] = np.arange(d.shape[0])
    d = d[live]
    return d.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------- weights --

def date_weights(dates: np.ndarray, age: np.ndarray | None = None,
                 half_life: float = 756.0) -> np.ndarray:
    """Recency weights, equalised so a busy date cannot outvote a quiet one.

    CORRECTION to the supplied protocol. It normalised the decayed weights by
    their per-date SUM. Every row on a given date shares that date's age, so
    the decay factor appeared identically in numerator and denominator and
    cancelled exactly — every date came out weighted 1.0 regardless of age,
    and `expert_half_life_sessions` did nothing at all. The engine was not
    adaptive in the one place it claimed to be.

    Dividing by the per-date COUNT instead keeps the intended property (a
    date with 300 names does not outvote a date with 3) while letting the
    date's own decay survive.
    """
    a = np.zeros(len(dates)) if age is None else np.maximum(np.asarray(age, float), 0)
    w = np.power(0.5, a / max(half_life, 1.0))
    cnt = pd.Series(1.0).repeat(len(dates)).to_numpy()
    cnt = pd.Series(cnt).groupby(pd.Series(dates)).transform("sum").to_numpy()
    w = w / np.where(cnt > 0, cnt, 1.0)
    tot = w.sum()
    return w / tot if tot > 0 else w


def weighted_quantile(x: np.ndarray, q: float, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    x = x[ok]
    w = np.ones_like(x) if w is None else np.asarray(w, float)[ok]
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float("nan")
    return float(np.interp(q * c[-1], c, x))


# ------------------------------------------------------------ the gate ----

def effective_block_wilson(p: float, n_blocks: int, alpha: float = 0.05) -> float:
    """Conservative lower bound on precision, counted in horizon blocks.

    Deliberately NOT an exact interval: overlapping forecasts on correlated
    stocks are nothing like independent Bernoulli trials. Counting blocks
    rather than rows is what stops 6.6m stock-days from certifying anything
    they like. Treat it as a screen.
    """
    if n_blocks <= 0 or not np.isfinite(p):
        return 0.0
    z = float(norm.isf(alpha))
    n = float(n_blocks)
    v = (p + z * z / (2 * n)
         - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return float(np.clip(v, 0.0, 1.0))


def evidence_gate(history: pd.DataFrame, cfg: Config, h: int):
    """Lowest threshold whose matured precision LOWER BOUND clears the target.

    Uses only forecasts this system already issued and which have since
    matured. Nothing here sees an outcome it has not already been scored on.
    """
    best, details = float("inf"), {"reason": "insufficient_matured_evidence",
                                   "candidates": []}
    if history is None or history.empty:
        return best, details
    alpha = cfg.family_alpha / (len(cfg.horizons) * len(THRESHOLDS))
    for t in THRESHOLDS:
        sel = history.loc[history["p_cal"] >= t]
        n = len(sel)
        if n == 0:
            continue
        blocks = int(sel["i"].floordiv(2 * h).nunique())
        precision = float(sel["up"].fillna(0).mean())
        lower = effective_block_wilson(precision, blocks, alpha)
        coverage = n / len(history)
        details["candidates"].append(dict(
            threshold=t, n=n, blocks=blocks, precision=precision,
            conservative_lower=lower, coverage=coverage))
        if (n >= cfg.minimum_gate_rows and blocks >= cfg.minimum_gate_blocks
                and coverage >= cfg.minimum_gate_coverage
                and lower > cfg.precision_target and t < best):
            best = t
            details["reason"] = "screen_passed_not_certified"
    return best, details


# ---------------------------------------------------------------- experts --

def fit_experts(train: pd.DataFrame, boundary: int, h: int, cfg: Config):
    X = train[list(FEATURES)].to_numpy(dtype=float)
    y = train["up"].to_numpy(dtype=int)
    w = date_weights(train["i"].to_numpy(),
                     boundary - train["exit_i"].to_numpy(),
                     cfg.expert_half_life_sessions)
    base = float((np.sum(w * y) + 1) / (w.sum() + 2))
    linear = make_pipeline(StandardScaler(),
                           LogisticRegression(C=0.10, max_iter=300,
                                              random_state=cfg.random_state))
    linear.fit(X, y, logisticregression__sample_weight=w)
    nonlinear = HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=7, learning_rate=0.05,
        l2_regularization=20, min_samples_leaf=200, early_stopping=False,
        random_state=cfg.random_state)
    nonlinear.fit(X, y, sample_weight=w)
    target = np.clip(train["z"].to_numpy(), -5, 5)
    location = make_pipeline(StandardScaler(), Ridge(alpha=500.0))
    location.fit(X, target, ridge__sample_weight=w)
    return base, linear, nonlinear, location


def matured_history(parts: list[pd.DataFrame], asof_i: int) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.loc[out["exit_i"] < asof_i].copy()


def adaptive_probability(raw: np.ndarray, hist: pd.DataFrame, asof: int,
                         cfg: Config):
    """Mix the experts by recent loss, then recalibrate on matured forecasts."""
    weights = np.ones(len(EXPERTS)) / len(EXPERTS)
    resolved = hist.loc[hist["up"].notna()] if (hist is not None and not hist.empty) else pd.DataFrame()
    if not resolved.empty and resolved["i"].nunique() >= cfg.minimum_calibration_dates:
        w = date_weights(resolved["i"].to_numpy(),
                         asof - resolved["exit_i"].to_numpy(),
                         cfg.expert_half_life_sessions)
        old = resolved[[f"p_{e}" for e in EXPERTS]].to_numpy()
        y = resolved["up"].to_numpy(dtype=float)
        loss = np.average((old - y[:, None]) ** 2, axis=0, weights=w)
        ws = np.exp(-12 * (loss - loss.min()))
        # fixed share: no expert is ever driven to zero, so a regime change
        # can bring it back
        weights = 0.10 + 0.70 * ws / ws.sum()
        weights = weights / weights.sum()
    mixed = raw @ weights
    if resolved.empty or resolved["i"].nunique() < cfg.minimum_calibration_dates:
        return mixed, weights
    y = resolved["up"].to_numpy(dtype=int)
    if np.unique(y).size < 2:
        return mixed, weights
    w = date_weights(resolved["i"].to_numpy(),
                     asof - resolved["exit_i"].to_numpy(),
                     cfg.expert_half_life_sessions)
    old = np.clip(resolved["p_mixed"].to_numpy(), 0.001, 0.999)
    cal = LogisticRegression(C=0.05, max_iter=150,
                             random_state=cfg.random_state)
    cal.fit(logit(old)[:, None], y, sample_weight=w)
    out = cal.predict_proba(logit(np.clip(mixed, 0.001, 0.999))[:, None])[:, 1]
    return np.clip(out, 0.001, 0.999), weights


def adaptive_path_radius(hist: pd.DataFrame, asof: int, cfg: Config):
    """Conformal radius covering the WHOLE path, from matured residuals."""
    if hist is None or hist.empty:
        return float("nan"), float("nan")
    c = hist.loc[np.isfinite(hist["path_score"])]
    if c.empty or c["i"].nunique() < cfg.minimum_calibration_dates:
        return float("nan"), float("nan")
    w = date_weights(c["i"].to_numpy(), asof - c["exit_i"].to_numpy(),
                     cfg.expert_half_life_sessions)
    q = weighted_quantile(c["path_score"].to_numpy(), 1 - cfg.path_alpha, w)
    q0 = weighted_quantile(c["flat_path_score"].to_numpy(), 1 - cfg.path_alpha, w)
    return q, q0


# ------------------------------------------------------------ prequential --

def path_scores(lp: np.ndarray, rows: pd.DataFrame, h: int) -> dict:
    """Whole-trajectory residual scores, computed AFTER the forecast is issued.

    The score is the worst standardised deviation over EVERY daily close in
    the horizon, not at sampled checkpoints — a band that only has to contain
    the endpoint is a much weaker promise than it looks, and quoting it as
    "the path" would be the lie.
    """
    n = len(rows)
    out = {k: np.full(n, np.nan) for k in
           ("path_score", "flat_path_score", "max_loss")}
    steps = np.arange(1, h + 1)
    for j, row in enumerate(rows.itertuples()):
        i, c = int(row.i), int(row.col)
        if i + h >= lp.shape[0]:
            continue
        actual = lp[i + steps, c] - lp[i, c]
        if not np.isfinite(actual).all():
            continue
        scale = row.sigma * np.sqrt(steps)
        centre = row.mu * steps / h
        out["path_score"][j] = np.max(np.abs(actual - centre) / scale)
        out["flat_path_score"][j] = np.max(np.abs(actual) / scale)
        out["max_loss"][j] = np.expm1(np.min(actual))
    return out


def run_prequential(px: pd.DataFrame, h: int, cfg: Config,
                    verbose: bool = True) -> pd.DataFrame:
    """Walk forward issuing forecasts, learning only from matured outcomes.

    DELISTING. The supplied protocol assumes a price exists at i+h. On a
    universe that keeps its dead, 4,725 tickers stop. An observation whose
    ticker stops trading inside the horizon is labelled up=False, not dropped:
    dropping it is the survivorship choice, it is the one everybody makes
    silently, and at 504 sessions it is worth 9.4 points of base rate. A stock
    that stopped trading is emphatically not one we were right about.
    """
    fmats = build_features(px)
    arr = px.to_numpy(dtype=np.float64)
    lp = np.log(np.where(arr > 0, arr, np.nan))
    n, k = lp.shape
    finite = np.isfinite(lp)
    last = np.where(finite.any(0), n - 1 - finite[::-1].argmax(0), -1)

    years = px.index.year.to_numpy()
    start_i = int(np.argmax(years >= cfg.first_fit_year))
    eval_i = int(np.argmax(years >= cfg.evaluation_year))
    rng = np.random.default_rng(cfg.random_state)

    parts: list[pd.DataFrame] = []
    experts = None
    fitted_at = -10 ** 9

    for i in range(eval_i, n - h, cfg.step):
        live = finite[i] & finite[i - 252] if i >= 252 else finite[i]
        if live.sum() < cfg.min_names_per_date:
            continue

        # ---- refit on data whose labels had ALL matured by the boundary ---
        if i - fitted_at >= cfg.refit_every:
            boundary = i - h                     # nothing after this is known
            tr = []
            lo = max(start_i, boundary - cfg.train_years * 252)
            for j in range(lo, boundary, cfg.step):
                if j - 252 < 0:
                    continue
                lv = finite[j] & finite[j - 252]
                if lv.sum() < cfg.min_names_per_date:
                    continue
                d = rows_for_date(fmats, j, lv)
                if d.empty:
                    continue
                if len(d) > cfg.names_per_date:
                    d = d.iloc[rng.choice(len(d), cfg.names_per_date, False)]
                cc = d["col"].to_numpy()
                fwd = lp[j + h, cc] - lp[j, cc]
                gone = (j + h) > last[cc]
                d["up"] = np.where(gone, 0, (fwd > 0).astype(int))
                sig = np.nanstd(np.diff(lp[max(0, j - 63):j + 1, cc], axis=0),
                                axis=0)
                d["sigma"] = np.where(sig > 1e-6, sig, np.nan)
                d["z"] = np.clip(np.nan_to_num(fwd, nan=0.0)
                                 / (d["sigma"] * np.sqrt(h)), -5, 5)
                d["i"], d["exit_i"] = j, j + h
                tr.append(d.dropna(subset=["sigma"]))
            train = pd.concat(tr, ignore_index=True) if tr else pd.DataFrame()
            if len(train) >= cfg.minimum_training_rows:
                experts = fit_experts(train, boundary, h, cfg)
                fitted_at = i
        if experts is None:
            continue
        base, linear, nonlinear, location = experts

        # ---- issue forecasts for today ------------------------------------
        d = rows_for_date(fmats, i, live)
        if d.empty:
            continue
        if len(d) > cfg.names_per_date:
            d = d.iloc[rng.choice(len(d), cfg.names_per_date, False)]
        X = d[list(FEATURES)].to_numpy(dtype=float)
        raw = np.column_stack([
            np.full(len(d), base),
            linear.predict_proba(X)[:, 1],
            nonlinear.predict_proba(X)[:, 1]])
        hist = matured_history(parts, i)
        p_cal, wts = adaptive_probability(raw, hist, i, cfg)

        cc = d["col"].to_numpy()
        sig = np.nanstd(np.diff(lp[max(0, i - 63):i + 1, cc], axis=0), axis=0)
        d = d.assign(i=i, exit_i=i + h, date=px.index[i],
                     ticker=px.columns[cc],
                     p_base=raw[:, 0], p_linear=raw[:, 1],
                     p_nonlinear=raw[:, 2],
                     p_mixed=raw @ wts, p_cal=p_cal,
                     sigma=np.where(sig > 1e-6, sig, np.nan))
        d["mu"] = np.clip(location.predict(X), -5, 5) * d["sigma"] * np.sqrt(h)

        # ---- realise the outcome (still invisible until it matures) -------
        fwd = lp[i + h, cc] - lp[i, cc]
        gone = (i + h) > last[cc]
        d["delisted"] = gone
        d["up"] = np.where(gone, 0.0, (fwd > 0).astype(float))
        d["fwd"] = np.where(gone, np.nan, fwd)
        for key, val in path_scores(lp, d, h).items():
            d[key] = val

        gate, details = evidence_gate(hist, cfg, h)
        d["gate"] = gate
        d["gate_reason"] = details["reason"]
        d["emitted"] = d["p_cal"] >= gate
        parts.append(d.dropna(subset=["sigma"]))

        if verbose and len(parts) % 20 == 0:
            print(f"  {px.index[i].date()}  rows={len(d)}  "
                  f"gate={'-' if not np.isfinite(gate) else gate:>4}  "
                  f"emitted={int(d['emitted'].sum())}", flush=True)

    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame())
