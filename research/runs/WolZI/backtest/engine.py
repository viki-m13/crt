"""Core walk-forward backtest engine.

Convention:
  - signal_date (asof): month-end T.  Features computed at close of T.
  - return_date: next month-end T+1.  Strategy earns monthly return at T+1.
  - Costs: round-trip 5 bps floor + 0 price-impact (monthly concentration small vs ADV).
  - Regime gate: optional; goes to cash_yield when crash detected.

Walk-forward splits: expanding window, refit every `refit_every` months.
Research window: 2003-09-30 → LOCKBOX_START-1m.
Lockbox: LOCKBOX_START → end of panel. Touch at most once per model family.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def next_month_end(asof: pd.Timestamp) -> pd.Timestamp:
    """Return calendar month-end of the month AFTER asof's month.

    Works correctly for both month-end and non-month-end asof dates.
    E.g. asof=2004-01-30 → 2004-02-29, asof=2004-01-31 → 2004-02-29.
    """
    # Move to first day of next calendar month, then to its end
    if asof.month == 12:
        return pd.Timestamp(asof.year + 1, 1, 31)  # Jan always has 31 days
    else:
        return (pd.Timestamp(asof.year, asof.month + 1, 1)
                + pd.offsets.MonthEnd(0))

REPO = Path("/home/user/crt")
CACHE = REPO / "experiments/monthly_dca/cache"
DATA = REPO / "data/YLOka"
QR = REPO / "quant_research"

# The last 24 months of the panel are the lockbox
LOCKBOX_START = pd.Timestamp("2024-01-31")
RESEARCH_END = pd.Timestamp("2023-12-31")

COST_BPS_FLOOR = 5.0  # per leg (buy or sell)

EXCLUDE = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "pit_panel_full.parquet")
    p["asof"] = pd.to_datetime(p["asof"])
    return p[~p["ticker"].isin(EXCLUDE)].copy()


def load_monthly_returns() -> pd.DataFrame:
    """Wide: index=month_end_date, columns=ticker, values=return that month."""
    mr = pd.read_parquet(CACHE / "v2/monthly_returns_clean.parquet")
    mr.index = pd.to_datetime(mr.index)
    return mr


def load_spy_monthly() -> pd.DataFrame:
    """SPY monthly features (for regime gate). Uses monthly_returns + price panel."""
    prices = pd.read_parquet(CACHE / "prices_extended.parquet")
    prices.index = pd.to_datetime(prices.index)
    spy = prices["SPY"].dropna()

    # Month-end dates
    mr = load_monthly_returns()
    rows = []
    for dt in mr.index:
        px_to = spy.asof(dt)
        px_200 = spy.loc[:dt].iloc[-200:].mean() if len(spy.loc[:dt]) >= 10 else np.nan
        px_50 = spy.loc[:dt].iloc[-50:].mean() if len(spy.loc[:dt]) >= 5 else np.nan
        px_1m = spy.asof(dt - pd.offsets.MonthEnd(1))
        px_6m = spy.asof(dt - pd.offsets.MonthEnd(6))
        px_12m = spy.asof(dt - pd.offsets.MonthEnd(12))
        rows.append({
            "date": dt,
            "px": px_to,
            "d_sma200": (px_to / px_200 - 1) if px_200 > 0 else 0.0,
            "d_sma50": (px_to / px_50 - 1) if px_50 > 0 else 0.0,
            "ret_1m": (px_to / px_1m - 1) if px_1m > 0 else 0.0,
            "ret_6m": (px_to / px_6m - 1) if px_6m > 0 else 0.0,
            "ret_12m": (px_to / px_12m - 1) if px_12m > 0 else 0.0,
        })
    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Regime gate
# ---------------------------------------------------------------------------

def regime_gate(spy_row: dict) -> str:
    """Returns 'equity', 'cash', 'recovery', or 'bull'."""
    r1m = spy_row.get("ret_1m", 0.0)
    r6m = spy_row.get("ret_6m", 0.0)
    d200 = spy_row.get("d_sma200", 0.0)

    if pd.isna(r1m):
        return "equity"
    # Crash: sharp 1m decline or persistent 6m decline below 200 SMA
    if r1m <= -0.08 or (r6m <= -0.10 and d200 < 0):
        return "cash"
    if d200 < -0.05:
        return "cash"
    if d200 > 0 and r6m < 0 and r1m > 0:
        return "recovery"
    if d200 > 0 and r6m >= 0.10:
        return "bull"
    return "equity"


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def compute_costs(prev_weights: dict[str, float], new_weights: dict[str, float],
                  cost_bps: float = COST_BPS_FLOOR) -> float:
    """Return total cost as fraction of portfolio (both legs combined)."""
    all_tickers = set(prev_weights) | set(new_weights)
    turnover = sum(
        abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
        for t in all_tickers
    ) / 2.0  # one-way turnover
    return turnover * (cost_bps / 10000.0) * 2  # round-trip


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def equal_weight(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def inv_vol_weight(tickers: list[str], vols: dict[str, float]) -> dict[str, float]:
    if not tickers:
        return {}
    inv = {t: 1.0 / max(vols.get(t, 0.2), 0.05) for t in tickers}
    total = sum(inv.values())
    return {t: v / total for t, v in inv.items()}


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    name: str
    K: int = 10
    weighting: str = "ew"           # ew | invvol
    score_col: str = "mom_12_1"     # feature to rank by (descending)
    use_regime: bool = True
    cash_yield_apr: float = 0.04    # 4% cash yield (T-bill proxy)
    cost_bps: float = COST_BPS_FLOOR
    # Optional filters
    vol_filter: bool = False        # remove top-30% vol names before ranking
    vol_col: str = "vol_12m"
    quality_filter: bool = False    # keep only above-median trend_health_5y
    research_only: bool = True      # exclude lockbox dates


def run_backtest(config: BacktestConfig,
                 panel: Optional[pd.DataFrame] = None,
                 mr: Optional[pd.DataFrame] = None,
                 spy: Optional[pd.DataFrame] = None) -> dict:
    """Run single-config backtest. Returns summary dict + monthly returns Series."""
    if panel is None:
        panel = load_panel()
    if mr is None:
        mr = load_monthly_returns()
    if spy is None:
        spy = load_spy_monthly()

    # Filter to research window if requested
    if config.research_only:
        panel = panel[panel["asof"] <= RESEARCH_END]

    asof_dates = sorted(panel["asof"].unique())
    if not asof_dates:
        return {"name": config.name, "cagr": np.nan, "sharpe": np.nan}

    equity = 1.0
    monthly_rets = []
    dates_out = []
    prev_weights: dict[str, float] = {}
    cash_months = 0
    equity_months = 0

    for asof in asof_dates:
        snap = panel[panel["asof"] == asof].copy()
        snap = snap.dropna(subset=[config.score_col])

        # Regime check: use spy features at asof month-end
        spy_row = spy.loc[asof].to_dict() if asof in spy.index else {}
        regime = regime_gate(spy_row) if config.use_regime else "equity"

        # Next month return date (calendar month-end of month AFTER asof's month)
        next_date = next_month_end(asof)

        if regime == "cash":
            ret = config.cash_yield_apr / 12.0
            cost = compute_costs(prev_weights, {}, config.cost_bps)
            net_ret = ret - cost
            prev_weights = {}
            equity *= (1 + net_ret)
            monthly_rets.append(net_ret)
            dates_out.append(next_date)
            cash_months += 1
            continue

        # Apply filters
        if config.vol_filter and config.vol_col in snap.columns:
            threshold = snap[config.vol_col].quantile(0.70)
            snap = snap[snap[config.vol_col] <= threshold]

        if config.quality_filter and "trend_health_5y" in snap.columns:
            threshold = snap["trend_health_5y"].median()
            snap = snap[snap["trend_health_5y"] >= threshold]

        # Rank and pick top-K
        snap_sorted = snap.sort_values(config.score_col, ascending=False)
        top_k = snap_sorted.head(config.K)["ticker"].tolist()

        if not top_k:
            ret = config.cash_yield_apr / 12.0
            monthly_rets.append(ret)
            dates_out.append(next_date)
            continue

        # Weights
        if config.weighting == "invvol" and config.vol_col in snap.columns:
            vol_map = snap.set_index("ticker")[config.vol_col].to_dict()
            new_weights = inv_vol_weight(top_k, vol_map)
        else:
            new_weights = equal_weight(top_k)

        # Cost
        cost = compute_costs(prev_weights, new_weights, config.cost_bps)

        # Portfolio return
        if next_date in mr.index:
            port_ret = sum(
                w * mr.loc[next_date, t]
                for t, w in new_weights.items()
                if t in mr.columns and not np.isnan(mr.loc[next_date, t])
            )
        else:
            port_ret = 0.0

        net_ret = port_ret - cost
        equity *= (1 + net_ret)
        monthly_rets.append(net_ret)
        dates_out.append(next_date)
        prev_weights = new_weights
        equity_months += 1

    if not monthly_rets:
        return {"name": config.name, "cagr": np.nan, "sharpe": np.nan}

    rets_series = pd.Series(monthly_rets, index=dates_out)
    n_months = len(rets_series)
    n_years = n_months / 12.0

    cagr = equity ** (1 / n_years) - 1 if n_years > 0 else np.nan
    mean_m = rets_series.mean()
    std_m = rets_series.std()
    sharpe = (mean_m / std_m * np.sqrt(12)) if std_m > 0 else np.nan

    # Max drawdown
    cum = (1 + rets_series).cumprod()
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1)
    max_dd = dd.min()

    return {
        "name": config.name,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_months": n_months,
        "cash_months": cash_months,
        "equity_months": equity_months,
        "final_equity": equity,
        "monthly_returns": rets_series,
    }


# ---------------------------------------------------------------------------
# Sub-period analysis
# ---------------------------------------------------------------------------

def sub_period_sharpes(rets: pd.Series, n_chunks: int = 3) -> list[float]:
    """Split OOS into n equal chunks and compute Sharpe per chunk."""
    n = len(rets)
    chunk_size = n // n_chunks
    sharpes = []
    for i in range(n_chunks):
        chunk = rets.iloc[i * chunk_size:(i + 1) * chunk_size]
        if len(chunk) < 6:
            sharpes.append(np.nan)
            continue
        s = chunk.mean() / chunk.std() * np.sqrt(12) if chunk.std() > 0 else np.nan
        sharpes.append(float(s))
    return sharpes


def summarize(result: dict) -> str:
    if "monthly_returns" not in result:
        return f"{result['name']}: no data"
    mr = result["monthly_returns"]
    chunks = sub_period_sharpes(mr)
    return (
        f"{result['name']:40s} "
        f"CAGR={result['cagr']:6.1%}  "
        f"Sharpe={result['sharpe']:5.2f}  "
        f"MaxDD={result['max_dd']:6.1%}  "
        f"N={result['n_months']}m  "
        f"Cash={result['cash_months']}m  "
        f"SubSharpes={[f'{s:.2f}' for s in chunks]}"
    )
