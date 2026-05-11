"""
Walk-forward backtest engine for monthly-rebalance equity strategies.

Data contract:
  - prices: pd.DataFrame, DatetimeIndex (daily), columns=tickers, values=adj_close
  - pit_membership: pd.DataFrame with columns ['asof', 'ticker']
  - signals: pd.DataFrame with columns ['asof', 'ticker', <score_col>]
  - fwd_returns: pd.DataFrame with columns ['asof', 'ticker', 'fwd_1m_ret']

PIT contract: at month-end date t, universe = tickers with asof == t in pit_membership.
Cost model: round-trip = 2 * cost_bps / 10000. Applied on buys + sells separately.
Slippage: half-spread assumed included in cost_bps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


LOCKBOX_START = pd.Timestamp("2024-05-31")


@dataclass
class BacktestConfig:
    name: str
    K: int = 5
    score_col: str = "score"
    weighting: str = "ew"          # ew | invvol
    cost_bps_one_way: float = 5.0  # one-way; round-trip = 2x
    cash_yield_annual: float = 0.03
    regime_gate: bool = False       # go cash when SPX < 200d MA
    research_end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2024-04-30"))


def compute_metrics(monthly_rets: pd.Series) -> dict:
    """Annualised CAGR, Sharpe, MaxDD from a monthly return series."""
    r = monthly_rets.dropna()
    if len(r) < 12:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "n_months": len(r)}
    cum = (1 + r).cumprod()
    n_years = len(r) / 12
    cagr = cum.iloc[-1] ** (1 / n_years) - 1
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    rolling_max = cum.cummax()
    dd = (cum / rolling_max - 1)
    max_dd = dd.min()
    return {
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "n_months": len(r),
    }


def compute_spx_200ma_gate(prices: pd.DataFrame, asof_dates: list) -> dict:
    """Returns {asof: True if SPX (proxy=equal-weight) above 200-day MA}."""
    # Use SPY as proxy if present, else market-cap-approx via avg
    gate = {}
    spy_col = "SPY" if "SPY" in prices.columns else None
    for asof in asof_dates:
        daily = prices.loc[:asof]
        if spy_col:
            series = daily[spy_col].dropna()
        else:
            series = daily.mean(axis=1).dropna()
        if len(series) < 200:
            gate[asof] = True  # insufficient history -> assume invested
        else:
            ma200 = series.iloc[-200:].mean()
            gate[asof] = bool(series.iloc[-1] >= ma200)
    return gate


def run_backtest(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    pit_membership: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
) -> dict:
    """
    Run a simple monthly-rebalance backtest.

    Returns dict with:
      monthly_returns: pd.Series indexed by asof date
      positions: dict {asof: list of (ticker, weight)}
      metrics: dict (full period and OOS only)
    """
    # Merge signals with PIT membership to enforce universe constraint
    members = pit_membership.copy()
    sig = signals[["asof", "ticker", cfg.score_col]].copy()
    panel = members.merge(sig, on=["asof", "ticker"], how="left")

    # Forward returns
    fwd = fwd_returns[["asof", "ticker", "fwd_1m_ret"]].copy()
    panel = panel.merge(fwd, on=["asof", "ticker"], how="left")

    asof_dates = sorted(panel["asof"].unique())
    asof_dates = [d for d in asof_dates if d <= cfg.research_end]

    # Regime gate
    regime_ok = {}
    if cfg.regime_gate:
        regime_ok = compute_spx_200ma_gate(prices, asof_dates)
    else:
        regime_ok = {d: True for d in asof_dates}

    cash_monthly = cfg.cash_yield_annual / 12
    cost_rt = 2 * cfg.cost_bps_one_way / 10_000  # round-trip cost

    monthly_rets = {}
    positions = {}
    prev_tickers = set()

    for asof in asof_dates:
        if not regime_ok.get(asof, True):
            monthly_rets[asof] = cash_monthly
            positions[asof] = []
            prev_tickers = set()
            continue

        month_data = panel[panel["asof"] == asof].dropna(subset=[cfg.score_col])
        if month_data.empty:
            monthly_rets[asof] = cash_monthly
            positions[asof] = []
            prev_tickers = set()
            continue

        # Pick top-K by score
        top = month_data.nlargest(cfg.K, cfg.score_col)
        tickers = top["ticker"].tolist()

        # Weighting
        if cfg.weighting == "invvol" and len(asof_dates) > 1:
            # Use 12-month vol from prices
            vol_map = {}
            for t in tickers:
                if t in prices.columns:
                    hist = prices.loc[:asof, t].dropna()
                    if len(hist) >= 25:
                        ret = hist.pct_change().dropna()
                        vol_map[t] = ret.iloc[-252:].std() if len(ret) >= 252 else ret.std()
                    else:
                        vol_map[t] = 1.0
                else:
                    vol_map[t] = 1.0
            inv_vol = {t: 1.0 / max(vol_map[t], 1e-6) for t in tickers}
            total = sum(inv_vol.values())
            weights = {t: inv_vol[t] / total for t in tickers}
        else:
            n = len(tickers)
            weights = {t: 1.0 / n for t in tickers}

        # Portfolio return = weighted sum of fwd_1m_ret
        port_ret = 0.0
        for _, row in top.iterrows():
            t = row["ticker"]
            r = row["fwd_1m_ret"]
            w = weights.get(t, 0.0)
            if pd.isna(r):
                r = 0.0  # treat missing as flat
            port_ret += w * r

        # Turnover cost: penalise new buys and exited sells
        new_buys = set(tickers) - prev_tickers
        exits = prev_tickers - set(tickers)
        n_trades = len(new_buys) + len(exits)
        # Cost as a fraction of trades relative to portfolio
        turnover_frac = n_trades / max(cfg.K, 1)
        cost = turnover_frac * (cfg.cost_bps_one_way / 10_000)
        # Add cost on new buys proportionally
        buy_weight = sum(weights.get(t, 0.0) for t in new_buys)
        sell_weight = len(exits) / max(cfg.K, 1)
        # Simplified: cost = (new_buy_weight + exit_weight) * one-way cost
        exact_cost = (buy_weight + sell_weight) * (cfg.cost_bps_one_way / 10_000)

        monthly_rets[asof] = port_ret - exact_cost
        positions[asof] = [(t, weights[t]) for t in tickers]
        prev_tickers = set(tickers)

    rets = pd.Series(monthly_rets).sort_index()
    metrics = compute_metrics(rets)

    return {
        "monthly_returns": rets,
        "positions": positions,
        "metrics": metrics,
        "config": cfg,
    }
