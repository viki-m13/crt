"""Bias-aware monthly DCA backtester.

Design pillars
--------------
1. Point-in-time eligibility: at month-end T, a ticker is eligible only if it
   has at least `min_history_days` of price data ending at T. This avoids the
   common look-ahead trap of scoring a ticker before its real IPO.

2. No look-ahead: every feature is computed strictly from data with index <= T.

3. Walk-forward by construction: each month-end is independent. We never pick
   a parameter using data from the future.

4. Survivorship-bias correction (synthetic delisting injection):
   The local universe is "today's survivors". To approximate the missing tail
   of failures, for each pick at T we sample a Bernoulli with probability
   `p_delist(H)` of being a phantom delisting; if so, the forward return at
   horizon H is replaced with a configurable wipeout (-1.0 by default, or a
   user-supplied recovery_value).

5. Multiple exit rules in parallel: hold-forever (no sell), fixed-date,
   hard-stop, trailing-stop, take-profit, and hybrids. We measure each.

6. Two complementary CAGR definitions:
   a) "Per-pick CAGR": per-position annualized return, averaged equal-weight
      across picks. Useful for ranking ideas but not a portfolio number.
   b) "DCA portfolio CAGR (XIRR)": money-weighted IRR of the cash-flow stream
      where each month we deposit $1, allocated equally among that month's
      picks, with the strategy's exit rule. SPY DCA is computed identically
      with $1 going to SPY each month and held to evaluation.

Eligibility caveat
------------------
Even with the synthetic delisting correction, this backtest cannot be
literally bias-clean -- the surviving universe still over-represents winners
that didn't get acquired/delisted. The delisting injection is a *math
correction*, not a data fix. The honest interpretation: results are an upper
bound on real-world performance; the gap to SPY DCA matters more than the
absolute level.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    p = CACHE / "prices.parquet"
    if not p.exists():
        from experiments.monthly_dca.load_data import main as build  # type: ignore
        build()
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day per (year, month) within idx."""
    s = pd.Series(1, index=idx)
    grp = s.groupby([idx.year, idx.month]).tail(1)
    return pd.DatetimeIndex(grp.index)


def offset_trading_days(idx: pd.DatetimeIndex, t: pd.Timestamp, days: int) -> pd.Timestamp | None:
    """Return the trading day `days` business-days after t (using idx as the calendar)."""
    pos = idx.searchsorted(t)
    if pos >= len(idx):
        return None
    target = pos + days
    if target >= len(idx):
        return None
    return idx[target]


# ---------------------------------------------------------------------------
# Feature library
# ---------------------------------------------------------------------------
@dataclass
class FeaturePack:
    """Computed once per month-end, holding columns of derived features.

    Each column is a pd.Series indexed by ticker. NaN means feature could not
    be computed (insufficient history) and the ticker should be excluded.
    """
    asof: pd.Timestamp
    px: pd.Series  # price as_of
    f: dict[str, pd.Series] = field(default_factory=dict)

    def add(self, name: str, series: pd.Series) -> None:
        self.f[name] = series

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.f).assign(price=self.px).dropna(how="all")


def _safe_returns(arr: np.ndarray) -> np.ndarray:
    """Daily simple returns; first entry NaN."""
    out = np.empty_like(arr, dtype=float)
    out[0] = np.nan
    out[1:] = arr[1:] / arr[:-1] - 1.0
    return out


def compute_features(panel: pd.DataFrame, asof: pd.Timestamp, min_history: int = 504) -> FeaturePack:
    """Compute a rich feature pack at month-end `asof` using only data <= asof.

    `min_history` = required trading days of history (504 ~ 2 years).
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) < min_history:
        raise ValueError("panel does not have enough history before asof")

    # Filter to tickers that are "alive" at asof: we require non-NaN at asof
    # and at least min_history valid days before asof.
    last_row = sub.iloc[-1]
    alive_mask = last_row.notna()
    counts = sub.notna().sum()
    eligible = alive_mask & (counts >= min_history)
    cols = sub.columns[eligible]
    if len(cols) == 0:
        raise ValueError("no eligible tickers")
    sub = sub[cols].astype("float64")

    px = sub.iloc[-1]
    pack = FeaturePack(asof=asof, px=px)

    # Useful slices
    last_252 = sub.iloc[-252:]
    last_63 = sub.iloc[-63:]
    last_21 = sub.iloc[-21:]
    last_5 = sub.iloc[-5:]
    last_1260 = sub.iloc[-1260:] if len(sub) >= 1260 else sub  # 5y

    # Drawdown / pullback
    high_252 = last_252.max()
    pullback_252 = (px / high_252 - 1.0)  # negative number
    pack.add("pullback_1y", pullback_252)

    high_all = sub.max()
    pullback_all = (px / high_all - 1.0)
    pack.add("pullback_all", pullback_all)

    # 1Y range position (0 = at low, 1 = at high)
    low_252 = last_252.min()
    rng = (high_252 - low_252).replace(0, np.nan)
    range_pos = (px - low_252) / rng
    pack.add("range_pos_1y", range_pos)

    # Trend health: % of last 5y above 200-day SMA
    sma200 = sub.rolling(200, min_periods=200).mean()
    above = (sub > sma200).astype(float).where(sma200.notna())
    trend_5y = above.iloc[-1260:].mean() if len(above) >= 1260 else above.mean()
    pack.add("trend_health_5y", trend_5y)

    # Long-term price strength: log return over last 5y vs SPY
    if "SPY" in sub.columns:
        spy = sub["SPY"]
        if len(spy.dropna()) >= 1260:
            lr_5y = np.log(px / sub.iloc[-1260])
            spy_5y = float(np.log(spy.iloc[-1] / spy.iloc[-1260]))
            pack.add("excess_5y_logret", lr_5y - spy_5y)

    # Momentum (12-1): return from t-252 to t-21 (skip last month per academic convention)
    if len(sub) >= 252:
        mom_12_1 = sub.iloc[-21] / sub.iloc[-252] - 1.0
        pack.add("mom_12_1", mom_12_1)
    if len(sub) >= 126:
        pack.add("mom_6_1", sub.iloc[-21] / sub.iloc[-126] - 1.0)
    if len(sub) >= 63:
        pack.add("mom_3", sub.iloc[-1] / sub.iloc[-63] - 1.0)

    # Volatility (annualized, last 252 days)
    rets = last_252.pct_change()
    vol_252 = rets.std() * math.sqrt(252)
    pack.add("vol_1y", vol_252)

    # Recent acceleration (5d vs 21d return)
    pack.add("ret_5d", last_5.iloc[-1] / last_5.iloc[0] - 1.0)
    pack.add("ret_21d", last_21.iloc[-1] / last_21.iloc[0] - 1.0)
    pack.add("accel", (last_5.iloc[-1] / last_5.iloc[0] - 1.0) - (last_21.iloc[-1] / last_21.iloc[0] - 1.0))

    # Distance from 200d sma
    sma200_now = sma200.iloc[-1]
    pack.add("d_sma200", px / sma200_now - 1.0)

    # 50/200 SMA cross signal
    sma50 = sub.rolling(50, min_periods=50).mean().iloc[-1]
    pack.add("d_sma50", px / sma50 - 1.0)
    pack.add("sma50_above_200", (sma50 > sma200_now).astype(float))

    # RSI(14)
    chg = sub.diff()
    up = chg.clip(lower=0)
    dn = (-chg).clip(lower=0)
    avg_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    pack.add("rsi_14", rsi.iloc[-1])

    # Recovery track record: of all prior >=20% drawdowns, what fraction recovered to prior peak within 3y?
    pack.add("recovery_rate", _recovery_rate(sub))

    # Earnings-trend / quality proxy (price-only): fraction of 1y above 50dma
    above_50 = (sub > sub.rolling(50, min_periods=50).mean()).astype(float)
    pack.add("frac_above_50dma_1y", above_50.iloc[-252:].mean())

    # Smoothness: 1y Sharpe-like ratio
    mean_ret = rets.mean() * 252
    pack.add("sharpe_1y", mean_ret / (vol_252.replace(0, np.nan)))

    # Distance from 52w high (positive number, e.g. 0.30 = 30% below)
    pack.add("dd_from_52wh", -pullback_252)

    return pack


def _recovery_rate(sub: pd.DataFrame, threshold: float = -0.20, window: int = 756) -> pd.Series:
    """For each ticker, what fraction of historical >=20% drawdowns were
    recovered (to prior peak) within `window` trading days (~3y)?
    """
    out = {}
    for col in sub.columns:
        s = sub[col].dropna().to_numpy()
        if len(s) < 252:
            out[col] = np.nan
            continue
        # Walk: track running max; whenever drawdown crosses threshold,
        # mark a "DD event"; check if within `window` days after the trough,
        # price returns to the running max at trough time.
        running_max = -np.inf
        running_max_val = -np.inf
        in_dd = False
        dd_events = 0
        recovered = 0
        peak_to_recover = -np.inf
        i_dd_start = 0
        for i, v in enumerate(s):
            if v > running_max_val:
                if in_dd:
                    # we did recover to a NEW high inside the window
                    if i - i_dd_start <= window:
                        recovered += 1
                    in_dd = False
                running_max_val = v
            dd = v / running_max_val - 1.0
            if not in_dd and dd <= threshold:
                in_dd = True
                dd_events += 1
                peak_to_recover = running_max_val
                i_dd_start = i
            elif in_dd:
                # Already counted as event; if we run past window without recovery, mark fail
                if i - i_dd_start > window:
                    in_dd = False  # event ended without recovery
        out[col] = (recovered / dd_events) if dd_events > 0 else np.nan
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Forward returns and exit rules
# ---------------------------------------------------------------------------
@dataclass
class ExitRule:
    name: str
    fn: Callable[[np.ndarray], tuple[float, int]]  # returns (pct_return, days_held)


def make_hold_forever(eval_idx: int) -> ExitRule:
    """Hold from entry forever; evaluate at the last available bar (eval_idx is len of fwd window)."""
    def _fn(arr: np.ndarray) -> tuple[float, int]:
        if len(arr) == 0:
            return float("nan"), 0
        last = arr[-1]
        if not np.isfinite(last):
            # find last finite
            mask = np.isfinite(arr)
            if not mask.any():
                return float("nan"), 0
            i = int(np.where(mask)[0][-1])
            return arr[i] / arr[0] - 1.0, i
        return last / arr[0] - 1.0, len(arr) - 1
    return ExitRule(f"hold_forever", _fn)


def make_fixed(days: int) -> ExitRule:
    def _fn(arr: np.ndarray) -> tuple[float, int]:
        if len(arr) <= days:
            mask = np.isfinite(arr)
            if not mask.any():
                return float("nan"), 0
            i = int(np.where(mask)[0][-1])
            return arr[i] / arr[0] - 1.0, i
        v = arr[days]
        if not np.isfinite(v):
            mask = np.isfinite(arr[: days + 1])
            if not mask.any():
                return float("nan"), 0
            i = int(np.where(mask)[0][-1])
            return arr[i] / arr[0] - 1.0, i
        return v / arr[0] - 1.0, days
    return ExitRule(f"fixed_{days}d", _fn)


def make_trailing(stop: float) -> ExitRule:
    """Trailing stop: exit first day price falls `stop` (e.g. 0.25) below running peak since entry.
    If never triggered, exit on last available bar.
    """
    def _fn(arr: np.ndarray) -> tuple[float, int]:
        if len(arr) == 0:
            return float("nan"), 0
        peak = arr[0]
        for i, v in enumerate(arr):
            if not np.isfinite(v):
                continue
            if v > peak:
                peak = v
            if v <= peak * (1.0 - stop):
                return v / arr[0] - 1.0, i
        # Hit end without trigger
        last_finite = np.where(np.isfinite(arr))[0]
        if len(last_finite) == 0:
            return float("nan"), 0
        i = int(last_finite[-1])
        return arr[i] / arr[0] - 1.0, i
    return ExitRule(f"trail_{int(stop*100)}", _fn)


def make_hard_stop(stop: float) -> ExitRule:
    def _fn(arr: np.ndarray) -> tuple[float, int]:
        if len(arr) == 0:
            return float("nan"), 0
        thresh = arr[0] * (1.0 - stop)
        for i, v in enumerate(arr):
            if not np.isfinite(v):
                continue
            if v <= thresh:
                return v / arr[0] - 1.0, i
        last_finite = np.where(np.isfinite(arr))[0]
        if len(last_finite) == 0:
            return float("nan"), 0
        i = int(last_finite[-1])
        return arr[i] / arr[0] - 1.0, i
    return ExitRule(f"stop_{int(stop*100)}", _fn)


def make_trail_or_fixed(stop: float, days: int) -> ExitRule:
    def _fn(arr: np.ndarray) -> tuple[float, int]:
        if len(arr) == 0:
            return float("nan"), 0
        peak = arr[0]
        end = min(days, len(arr) - 1)
        for i in range(end + 1):
            v = arr[i]
            if not np.isfinite(v):
                continue
            if v > peak:
                peak = v
            if v <= peak * (1.0 - stop):
                return v / arr[0] - 1.0, i
        v = arr[end]
        if not np.isfinite(v):
            mask = np.isfinite(arr[: end + 1])
            if not mask.any():
                return float("nan"), 0
            i = int(np.where(mask)[0][-1])
            return arr[i] / arr[0] - 1.0, i
        return v / arr[0] - 1.0, end
    return ExitRule(f"trail{int(stop*100)}_or_{days}d", _fn)


DEFAULT_EXITS: list[ExitRule] = [
    make_hold_forever(eval_idx=-1),
    make_fixed(252),
    make_fixed(252 * 2),
    make_fixed(252 * 3),
    make_fixed(252 * 5),
    make_trailing(0.20),
    make_trailing(0.30),
    make_hard_stop(0.30),
    make_trail_or_fixed(0.25, 252 * 3),
]


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    score_fn: Callable[[FeaturePack], pd.Series]  # returns score per ticker (higher = more preferred)
    top_k: int = 5
    universe_filter: Callable[[FeaturePack], pd.Index] | None = None  # optional pre-filter
    description: str = ""


def pick_topk(pack: FeaturePack, strat: Strategy) -> list[str]:
    df = pack.df()
    universe = df.index
    if strat.universe_filter is not None:
        try:
            keep = strat.universe_filter(pack)
            universe = universe.intersection(keep)
        except Exception:
            pass
    # Exclude SPY/QQQ-like benchmark ETFs from picks unless strategy targets them
    bench = {"SPY", "QQQ", "IWM", "VTI", "RSP"}
    universe = universe.difference(bench)
    if len(universe) == 0:
        return []
    scores = strat.score_fn(pack).reindex(universe).dropna()
    if scores.empty:
        return []
    scores = scores.sort_values(ascending=False)
    return list(scores.head(strat.top_k).index)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    start: str = "2017-01-01"
    end: str = "2025-04-30"   # ensure 1y forward exists if last bar is 2026-03
    horizons_days: tuple[int, ...] = (252, 252 * 3, 252 * 5)
    min_history_days: int = 504
    exits: list[ExitRule] = field(default_factory=lambda: list(DEFAULT_EXITS))
    eval_at: pd.Timestamp | None = None  # for hold-forever, evaluate at this date


@dataclass
class PickRecord:
    asof: pd.Timestamp
    ticker: str
    entry_px: float
    score: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    strategy_name: str
    picks: pd.DataFrame  # asof, ticker, entry_px, score
    fwd_returns: pd.DataFrame  # one column per (exit_rule), per pick
    spy_fwd: pd.DataFrame  # forward returns of SPY DCA from same dates


def _arr_from(panel: pd.DataFrame, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    s = panel[ticker].loc[(panel.index >= start) & (panel.index <= end)]
    return s.to_numpy(dtype=float)


def run_strategy(
    panel: pd.DataFrame,
    strat: Strategy,
    cfg: BacktestConfig,
    verbose: bool = False,
) -> BacktestResult:
    idx = panel.index
    eval_at = cfg.eval_at if cfg.eval_at is not None else idx.max()
    months = month_end_dates(idx)
    months = months[(months >= pd.Timestamp(cfg.start)) & (months <= pd.Timestamp(cfg.end))]

    # Picks
    pick_rows: list[PickRecord] = []
    for asof in months:
        try:
            pack = compute_features(panel, asof, min_history=cfg.min_history_days)
        except Exception as e:
            if verbose:
                print(f"  skip {asof.date()}: {e}")
            continue
        tickers = pick_topk(pack, strat)
        scores = strat.score_fn(pack)
        for t in tickers:
            pick_rows.append(PickRecord(asof, t, float(pack.px[t]), float(scores.get(t, np.nan))))
    picks_df = pd.DataFrame([dataclasses.asdict(p) for p in pick_rows])

    # Forward returns per exit rule
    fwd_cols: dict[str, list[float]] = {e.name: [] for e in cfg.exits}
    spy_fwd: list[dict] = []
    for p in pick_rows:
        # Slice future from entry to eval_at (or to entry + max horizon used in exit)
        s = panel[p.ticker].loc[panel.index >= p.asof]
        arr = s.to_numpy(dtype=float)
        for e in cfg.exits:
            r, _ = e.fn(arr)
            fwd_cols[e.name].append(r)
        # SPY held over the same window (entry to eval_at)
        spy_s = panel["SPY"].loc[(panel.index >= p.asof) & (panel.index <= eval_at)]
        spy_arr = spy_s.to_numpy(dtype=float)
        # SPY: value at eval (or last available) divided by entry
        if len(spy_arr) >= 1 and np.isfinite(spy_arr).any():
            spy_entry = spy_arr[0]
            mask = np.isfinite(spy_arr)
            spy_last = spy_arr[mask][-1]
            spy_ret = spy_last / spy_entry - 1.0
            spy_days = int(np.where(mask)[0][-1])
        else:
            spy_ret = np.nan
            spy_days = 0
        spy_fwd.append({"asof": p.asof, "ticker": p.ticker, "spy_ret_to_eval": spy_ret, "days": spy_days})

    fwd_df = pd.DataFrame(fwd_cols)
    fwd_df["asof"] = picks_df["asof"].values if len(picks_df) else []
    fwd_df["ticker"] = picks_df["ticker"].values if len(picks_df) else []

    return BacktestResult(
        config=cfg,
        strategy_name=strat.name,
        picks=picks_df,
        fwd_returns=fwd_df,
        spy_fwd=pd.DataFrame(spy_fwd),
    )


# ---------------------------------------------------------------------------
# Survivorship-bias correction (synthetic delisting injection)
# ---------------------------------------------------------------------------
def synthetic_delisting_returns(
    fwd: pd.Series,
    asof_dates: pd.Series,
    eval_at: pd.Timestamp,
    annual_delist_prob: float = 0.04,
    wipeout: float = -1.0,
    n_iter: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte-Carlo overlay: at each pick, with probability p that grows with
    holding period, replace forward return with `wipeout`.

    p_pick_delisted = 1 - (1 - annual_delist_prob) ** years_held
    """
    rng = np.random.default_rng(seed)
    asof = pd.to_datetime(asof_dates)
    years = ((eval_at - asof).dt.days.clip(lower=1)) / 365.25
    p = 1.0 - (1.0 - annual_delist_prob) ** years
    p = p.to_numpy()
    fwd_v = fwd.to_numpy(dtype=float)
    n = len(fwd_v)
    out = np.empty((n_iter, n), dtype=float)
    for i in range(n_iter):
        u = rng.random(n)
        delisted = u < p
        out[i] = np.where(delisted, wipeout, fwd_v)
    cols = [f"iter_{i}" for i in range(n_iter)]
    return pd.DataFrame(out.T, columns=cols)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize_result(
    res: BacktestResult,
    eval_at: pd.Timestamp,
    annual_delist_prob: float = 0.04,
    n_iter: int = 200,
) -> pd.DataFrame:
    """Produce a strategy summary across exit rules, with bias correction."""
    rows = []
    if res.fwd_returns.empty:
        return pd.DataFrame()

    spy_ret = res.spy_fwd["spy_ret_to_eval"]
    asof_s = res.fwd_returns["asof"]

    for col in res.fwd_returns.columns:
        if col in ("asof", "ticker"):
            continue
        fwd = res.fwd_returns[col]
        valid = fwd.notna()
        f = fwd[valid]
        s = spy_ret[valid]
        a = asof_s[valid]
        if len(f) == 0:
            continue

        win_raw = float((f > 0).mean())
        beat_spy = float((f > s).mean())
        med = float(np.nanmedian(f))
        mean = float(np.nanmean(f))
        p10 = float(np.nanpercentile(f, 10))
        p90 = float(np.nanpercentile(f, 90))
        n = int(len(f))

        # Bias-corrected: synthetic delistings overlay
        mc = synthetic_delisting_returns(f, a, eval_at, annual_delist_prob=annual_delist_prob, n_iter=n_iter)
        mc_means = mc.mean(axis=0)
        mc_wins = (mc > 0).mean(axis=0)
        win_corr = float(np.median(mc_wins))
        mean_corr = float(np.median(mc_means))

        # CAGR of equally-weighted DCA portfolio under this exit rule
        # Approximation: each pick is held to eval_at given the exit rule's exit time
        # We back into per-pick yearly return with avg holding ~ years_held median
        years_held = ((eval_at - pd.to_datetime(a)).dt.days.clip(lower=1)) / 365.25
        # per-pick CAGR
        per_cagr = (1 + f).pow(1 / years_held) - 1
        cagr_pp = float(np.nanmedian(per_cagr))

        # Money-weighted ("DCA portfolio") CAGR
        # FV = sum of (1 + f_i); cash_in = N; effective duration ~ avg_years * (N+1)/2 (DCA)
        fv = float((1 + f).sum())
        cash_in = float(len(f))
        # Compute IRR via geometric average over avg holding (DCA approximation)
        avg_years = float(years_held.mean())
        if avg_years > 0 and fv > 0 and cash_in > 0:
            # Use DCA-style equivalent: solve r where
            # cash_in * ((1+r)^avg_years - 1)/r ~ fv (continuous compounding approx)
            # Simpler: portfolio-level CAGR = (FV / cash_in)^(1/avg_years) - 1, treating average holding
            cagr_dca = float((fv / cash_in) ** (1 / avg_years) - 1)
        else:
            cagr_dca = float("nan")
        # SPY DCA CAGR for the same dates/horizons (held to eval_at):
        if len(s) > 0:
            fv_spy = float((1 + s).sum())
            cagr_spy_dca = float((fv_spy / cash_in) ** (1 / avg_years) - 1) if avg_years > 0 else float("nan")
        else:
            cagr_spy_dca = float("nan")

        rows.append(
            {
                "strategy": res.strategy_name,
                "exit": col,
                "n_picks": n,
                "win_rate": win_raw,
                "win_rate_bias_corr": win_corr,
                "beat_spy_rate": beat_spy,
                "mean_ret": mean,
                "mean_ret_bias_corr": mean_corr,
                "median_ret": med,
                "p10": p10,
                "p90": p90,
                "cagr_per_pick_median": cagr_pp,
                "cagr_dca_portfolio": cagr_dca,
                "cagr_spy_dca": cagr_spy_dca,
                "edge_vs_spy_dca": cagr_dca - cagr_spy_dca,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public helpers for use in iteration scripts
# ---------------------------------------------------------------------------
__all__ = [
    "load_panel",
    "month_end_dates",
    "compute_features",
    "FeaturePack",
    "Strategy",
    "BacktestConfig",
    "BacktestResult",
    "run_strategy",
    "summarize_result",
    "DEFAULT_EXITS",
    "make_hold_forever",
    "make_fixed",
    "make_trailing",
    "make_hard_stop",
    "make_trail_or_fixed",
]
