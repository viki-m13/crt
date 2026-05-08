"""Fast engine: re-evaluate strategies using cached features.

Adds proper money-weighted CAGR (XIRR), per-pick CAGR, vs-SPY-DCA comparison,
and synthetic-delisting bias correction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    """Load the extended panel (2000-) if available; else the standard one."""
    ext = CACHE / "prices_extended.parquet"
    if ext.exists():
        return pd.read_parquet(ext)
    return pd.read_parquet(CACHE / "prices.parquet")


def load_feature_months() -> list[pd.Timestamp]:
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    return [pd.Timestamp(f.stem) for f in files]


def load_features(asof: pd.Timestamp) -> pd.DataFrame:
    return pd.read_parquet(FEATURES_DIR / f"{asof.date()}.parquet")


# ---------------------------------------------------------------------------
# IRR (Newton-Raphson on monthly cash flows)
# ---------------------------------------------------------------------------
def xirr(cashflows: list[tuple[pd.Timestamp, float]], guess: float = 0.10) -> float:
    """Return the annualized money-weighted IRR for an irregular cash-flow stream.

    cashflows: list of (date, amount) where deposits are negative (outflows)
               and the final valuation is positive (inflow).
    """
    if not cashflows:
        return float("nan")
    dates = [c[0] for c in cashflows]
    amounts = np.asarray([c[1] for c in cashflows], dtype=float)
    t0 = dates[0]
    dt = np.asarray([(d - t0).days / 365.25 for d in dates], dtype=float)

    def npv(r: float) -> float:
        return float(np.sum(amounts / (1.0 + r) ** dt))

    def dnpv(r: float) -> float:
        return float(np.sum(-dt * amounts / (1.0 + r) ** (dt + 1.0)))

    r = guess
    for _ in range(200):
        v = npv(r)
        d = dnpv(r)
        if d == 0 or not np.isfinite(d):
            break
        r_new = r - v / d
        if not np.isfinite(r_new):
            break
        if abs(r_new - r) < 1e-9:
            r = r_new
            break
        # Keep r > -1
        if r_new <= -0.999:
            r_new = -0.999
        r = r_new
    if abs(npv(r)) > 1e-3 * abs(np.sum(np.abs(amounts))):
        # Fallback: bisection in [-0.99, 5.0]
        lo, hi = -0.99, 5.0
        if npv(lo) * npv(hi) > 0:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if npv(lo) * npv(mid) <= 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-9:
                break
        r = 0.5 * (lo + hi)
    return r


# ---------------------------------------------------------------------------
# Exit-rule simulator on a forward price slice
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExitRule:
    name: str
    days: int = -1               # if >=0, force exit on this day (calendar in trading days)
    trail: float = 0.0           # trailing stop fraction; 0 disables
    hard_stop: float = 0.0       # hard stop fraction; 0 disables
    take_profit: float = 0.0     # take-profit; 0 disables


def simulate_exit(prices: np.ndarray, rule: ExitRule) -> tuple[float, int, float]:
    """Return (return, hold_days, exit_price). Hold-forever if days<0 and others 0."""
    if len(prices) == 0:
        return float("nan"), 0, float("nan")
    entry = prices[0]
    peak = entry
    end_day = (rule.days if rule.days >= 0 else len(prices) - 1)
    end_day = min(end_day, len(prices) - 1)
    for i in range(end_day + 1):
        v = prices[i]
        if not np.isfinite(v):
            continue
        if v > peak:
            peak = v
        if rule.trail > 0 and v <= peak * (1.0 - rule.trail):
            return v / entry - 1.0, i, v
        if rule.hard_stop > 0 and v <= entry * (1.0 - rule.hard_stop):
            return v / entry - 1.0, i, v
        if rule.take_profit > 0 and v >= entry * (1.0 + rule.take_profit):
            return v / entry - 1.0, i, v
    # Reached end without trigger
    last_finite = np.where(np.isfinite(prices[: end_day + 1]))[0]
    if len(last_finite) == 0:
        return float("nan"), 0, float("nan")
    i = int(last_finite[-1])
    return prices[i] / entry - 1.0, i, prices[i]


DEFAULT_RULES: list[ExitRule] = [
    ExitRule("hold_forever"),
    ExitRule("fixed_1y", days=252),
    ExitRule("fixed_2y", days=252 * 2),
    ExitRule("fixed_3y", days=252 * 3),
    ExitRule("fixed_5y", days=252 * 5),
    ExitRule("trail_25", trail=0.25),
    ExitRule("trail_35", trail=0.35),
    ExitRule("trail_50", trail=0.50),
    ExitRule("stop_30", hard_stop=0.30),
    ExitRule("trail35_or_3y", trail=0.35, days=252 * 3),
    ExitRule("trail50_or_5y", trail=0.50, days=252 * 5),
    ExitRule("tp100", take_profit=1.00),
    ExitRule("tp200", take_profit=2.00),
]


# ---------------------------------------------------------------------------
# Strategy: a function from feature DataFrame -> Series of scores (higher = better)
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    score_fn: Callable[[pd.DataFrame], pd.Series]
    top_k: int = 5
    description: str = ""


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    start: str = "2017-12-31"
    end: str = "2024-12-31"      # leave a 1y buffer to 2025-12 (and 2026-03 for full)
    rules: list[ExitRule] = field(default_factory=lambda: list(DEFAULT_RULES))
    eval_at: pd.Timestamp | None = None
    bench_ticker: str = "SPY"
    delist_prob_annual: float = 0.04
    delist_wipeout: float = -1.0
    delist_iters: int = 0          # 0 = skip MC; otherwise run MC overlay


def backtest(
    panel: pd.DataFrame,
    strat: Strategy,
    cfg: BacktestConfig,
) -> dict:
    months = load_feature_months()
    months = [m for m in months if pd.Timestamp(cfg.start) <= m <= pd.Timestamp(cfg.end)]
    eval_at = cfg.eval_at if cfg.eval_at is not None else panel.index.max()

    bench_series = panel[cfg.bench_ticker]

    picks_records: list[dict] = []
    for asof in months:
        feats = load_features(asof)
        scores = strat.score_fn(feats)
        scores = scores.dropna()
        # Exclude benchmark/large-ETFs
        scores = scores.drop(labels=[t for t in ("SPY", "QQQ", "IWM", "VTI", "RSP") if t in scores.index], errors="ignore")
        if scores.empty:
            continue
        top = scores.sort_values(ascending=False).head(strat.top_k)
        for tkr, s in top.items():
            picks_records.append({"asof": asof, "ticker": tkr, "score": float(s),
                                  "entry_px": float(feats.loc[tkr, "price"]) if tkr in feats.index else np.nan})
    picks = pd.DataFrame(picks_records)

    if picks.empty:
        return {"strategy": strat.name, "n_picks": 0, "summary": pd.DataFrame()}

    # Forward arrays per pick
    fwd_by_rule = {r.name: np.full(len(picks), np.nan) for r in cfg.rules}
    fwd_days_by_rule = {r.name: np.zeros(len(picks), dtype=int) for r in cfg.rules}
    bench_ret_to_eval = np.full(len(picks), np.nan)
    held_to_eval_days = np.zeros(len(picks), dtype=int)

    panel_idx = panel.index
    eval_pos = panel_idx.searchsorted(eval_at, side="right") - 1
    eval_at_panel = panel_idx[eval_pos]

    for i, row in picks.iterrows():
        asof = row["asof"]
        tkr = row["ticker"]
        # Entry pos
        pos = panel_idx.searchsorted(asof)
        if pos >= len(panel_idx):
            continue
        if panel_idx[pos] != asof:
            # ensure we are at the asof month-end day; align if needed
            pos = max(0, pos - 1)
        prices = panel[tkr].iloc[pos: eval_pos + 1].to_numpy(dtype=float)
        if len(prices) == 0 or not np.isfinite(prices[0]):
            continue
        for r in cfg.rules:
            ret, days, _ = simulate_exit(prices, r)
            fwd_by_rule[r.name][i] = ret
            fwd_days_by_rule[r.name][i] = days
        # Bench: SPY held to eval_at
        bp = bench_series.iloc[pos: eval_pos + 1].to_numpy(dtype=float)
        if len(bp) > 0 and np.isfinite(bp[0]):
            mask = np.isfinite(bp)
            if mask.any():
                bench_ret_to_eval[i] = bp[mask][-1] / bp[0] - 1.0
                held_to_eval_days[i] = int(np.where(mask)[0][-1])

    # Add forward columns to picks
    out = picks.copy()
    out["bench_ret_to_eval"] = bench_ret_to_eval
    out["held_to_eval_days"] = held_to_eval_days

    summaries = []
    for r in cfg.rules:
        f = fwd_by_rule[r.name]
        d = fwd_days_by_rule[r.name]
        valid = np.isfinite(f)
        if not valid.any():
            continue
        fv = f[valid]
        dv = d[valid]
        bv = bench_ret_to_eval[valid]
        ah = picks["asof"].to_numpy()[valid]

        # Per-pick stats
        years_held = np.maximum(dv, 1) / 252.0
        per_pick_cagr = (1 + fv) ** (1.0 / years_held) - 1.0

        # Win rate / beat-spy
        win = float((fv > 0).mean())
        beat = float((fv > bv).mean())

        # Bias-corrected via synthetic delistings
        if cfg.delist_iters > 0:
            rng = np.random.default_rng(0)
            ah_ts = pd.to_datetime(ah)
            days = (eval_at_panel - ah_ts).days
            days_arr = np.asarray(days, dtype=float)
            years_to_eval = np.maximum(days_arr, 1.0) / 365.25
            p_del = 1.0 - (1.0 - cfg.delist_prob_annual) ** years_to_eval
            wins = []
            means = []
            for it in range(cfg.delist_iters):
                u = rng.random(len(fv))
                fv_mc = np.where(u < p_del, cfg.delist_wipeout, fv)
                wins.append(float((fv_mc > 0).mean()))
                means.append(float(fv_mc.mean()))
            win_corr = float(np.median(wins))
            mean_corr = float(np.median(means))
        else:
            win_corr = float("nan")
            mean_corr = float("nan")

        # Money-weighted IRR (DCA portfolio): each pick is $1 deposit on its asof,
        # value grows with that pick's return until exit; on exit, that capital sits
        # at exit value (no reinvestment). Final eval = sum of (1+r_i) at eval_at.
        # Cashflows: -1 on each asof; +sum_pv on eval_at_panel.
        # Need terminal value: for picks held to eval, terminal = (1+r); for those
        # exited earlier, terminal = (1+r) * (1+0) since we hold cash. Equivalent.
        cashflows: list[tuple[pd.Timestamp, float]] = []
        for asof_t in ah:
            cashflows.append((pd.Timestamp(asof_t), -1.0))
        terminal_value = float(np.sum(1 + fv))
        cashflows.append((eval_at_panel, terminal_value))
        cagr_dca = xirr(cashflows)

        # SPY DCA same dates
        spy_terminal = float(np.sum(1 + bv))
        cashflows_spy: list[tuple[pd.Timestamp, float]] = [(pd.Timestamp(asof_t), -1.0) for asof_t in ah]
        cashflows_spy.append((eval_at_panel, spy_terminal))
        cagr_spy = xirr(cashflows_spy)

        # Buy-and-hold of the basket weighted by entry: per-pick CAGR mean
        per_pick_cagr_mean = float(np.nanmean(per_pick_cagr))
        per_pick_cagr_median = float(np.nanmedian(per_pick_cagr))

        summaries.append(
            {
                "strategy": strat.name,
                "exit": r.name,
                "n_picks": int(valid.sum()),
                "win_rate": win,
                "win_rate_bias_corr": win_corr,
                "beat_spy_rate": beat,
                "median_ret": float(np.nanmedian(fv)),
                "mean_ret": float(np.nanmean(fv)),
                "p10_ret": float(np.nanpercentile(fv, 10)),
                "p90_ret": float(np.nanpercentile(fv, 90)),
                "per_pick_cagr_mean": per_pick_cagr_mean,
                "per_pick_cagr_median": per_pick_cagr_median,
                "cagr_dca_portfolio": cagr_dca,
                "cagr_spy_dca": cagr_spy,
                "edge_vs_spy_dca": cagr_dca - cagr_spy,
                "mean_ret_bias_corr": mean_corr,
            }
        )
    return {
        "strategy": strat.name,
        "picks": out,
        "summary": pd.DataFrame(summaries),
    }


# ---------------------------------------------------------------------------
# Persistent registry of cached feature columns we have available
# ---------------------------------------------------------------------------
def feature_columns_at(asof: pd.Timestamp) -> list[str]:
    return list(load_features(asof).columns)
