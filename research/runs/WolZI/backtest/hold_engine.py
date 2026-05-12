"""Walk-forward backtest with variable hold periods and correct date arithmetic.

This engine mirrors the YLOka harness logic but in a cleaner form.
The monthly_returns index has calendar month-end dates; we match the
nearest one and advance by 1 to get next-month return.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

REPO = Path("/home/user/crt")
CACHE = REPO / "experiments/monthly_dca/cache"
DATA = REPO / "data/YLOka"

RESEARCH_END = pd.Timestamp("2023-12-31")
LOCKBOX_START = pd.Timestamp("2024-01-31")
COST_BPS = 5.0

EXCLUDE = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS",
}


def load_data():
    """Returns (panel, mr, spy_feat, mr_idx)."""
    panel = pd.read_parquet(DATA / "pit_panel_full.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    panel = panel[~panel["ticker"].isin(EXCLUDE)].copy()

    mr = pd.read_parquet(CACHE / "v2/monthly_returns_clean.parquet")
    mr.index = pd.to_datetime(mr.index)
    mr_idx = mr.index  # calendar month-end dates

    prices = pd.read_parquet(CACHE / "prices_extended.parquet")
    prices.index = pd.to_datetime(prices.index)

    spy = prices["SPY"].dropna()
    rows = []
    for dt in mr.index:
        px = spy.asof(dt)
        sma200_vals = spy.loc[:dt].iloc[-200:] if len(spy.loc[:dt]) >= 5 else spy.loc[:dt]
        sma50_vals = spy.loc[:dt].iloc[-50:] if len(spy.loc[:dt]) >= 5 else spy.loc[:dt]
        px_200 = sma200_vals.mean() if len(sma200_vals) > 0 else np.nan
        px_50 = sma50_vals.mean() if len(sma50_vals) > 0 else np.nan
        px_1m = spy.asof(dt - pd.DateOffset(months=1))
        px_6m = spy.asof(dt - pd.DateOffset(months=6))
        px_12m = spy.asof(dt - pd.DateOffset(months=12))
        rows.append({
            "date": dt,
            "d_sma200": (px / px_200 - 1) if px_200 and not np.isnan(px_200) else 0.0,
            "d_sma50": (px / px_50 - 1) if px_50 and not np.isnan(px_50) else 0.0,
            "ret_1m": (px / px_1m - 1) if px_1m > 0 else 0.0,
            "ret_6m": (px / px_6m - 1) if px_6m > 0 else 0.0,
            "ret_12m": (px / px_12m - 1) if px_12m > 0 else 0.0,
        })
    spy_feat = pd.DataFrame(rows).set_index("date")
    return panel, mr, mr_idx, spy_feat


def _spy_regime(spy_feat: pd.DataFrame, date: pd.Timestamp) -> str:
    """Crash gate matching YLOka 'tight' regime."""
    if date not in spy_feat.index:
        # Find nearest date
        idx = spy_feat.index.searchsorted(date)
        if idx >= len(spy_feat.index):
            return "normal"
        date = spy_feat.index[min(idx, len(spy_feat.index) - 1)]
    row = spy_feat.loc[date]
    r1m = float(row.get("ret_1m", 0))
    r6m = float(row.get("ret_6m", 0))
    d200 = float(row.get("d_sma200", 0))
    if np.isnan(r1m):
        return "normal"
    if r1m <= -0.08 or (r6m <= -0.10 and d200 < 0):
        return "crash"
    if d200 < -0.05:
        return "crash"
    if d200 > 0 and r6m >= 0.10:
        return "bull"
    if d200 > 0 and r6m < 0 and r1m > 0:
        return "recovery"
    return "normal"


def _next_mr_date(asof: pd.Timestamp, mr_idx: pd.Index) -> Optional[pd.Timestamp]:
    """Find the next monthly return date after asof by matching nearest mr_idx entry."""
    pos = mr_idx.searchsorted(asof)
    # Take the nearest entry at or just after asof
    if pos < len(mr_idx) and abs((mr_idx[pos] - asof).days) <= 7:
        nearest_pos = pos
    elif pos > 0 and abs((mr_idx[pos - 1] - asof).days) <= 7:
        nearest_pos = pos - 1
    else:
        # Fall back to pos
        nearest_pos = min(pos, len(mr_idx) - 1)
    next_pos = nearest_pos + 1
    if next_pos >= len(mr_idx):
        return None
    return mr_idx[next_pos]


def simulate(
    score_col: str,
    K: int = 3,
    hold_months: int = 6,
    weighting: str = "ew",
    use_crash_gate: bool = True,
    cash_yield_apr: float = 0.04,
    cost_bps: float = COST_BPS,
    panel: Optional[pd.DataFrame] = None,
    mr: Optional[pd.DataFrame] = None,
    mr_idx: Optional[pd.Index] = None,
    spy_feat: Optional[pd.DataFrame] = None,
    research_only: bool = True,
) -> dict:
    if panel is None or mr is None or mr_idx is None or spy_feat is None:
        panel, mr, mr_idx, spy_feat = load_data()

    if research_only:
        panel = panel[panel["asof"] <= RESEARCH_END]

    months = sorted(panel["asof"].unique())
    by_asof = {pd.Timestamp(d): g for d, g in panel.groupby("asof")}

    cf = cost_bps / 10_000
    cash_m = (1 + cash_yield_apr) ** (1 / 12) - 1

    equity = 1.0
    cur_picks: list[str] = []
    cur_weights: np.ndarray = np.array([])
    held_for = 0
    in_cash = False
    rows = []

    for i, asof in enumerate(months):
        asof = pd.Timestamp(asof)

        # Regime check using CURRENT calendar month-end (no look-ahead)
        # asof is the last trading day of the month; the current calendar month-end
        # is the last calendar day (asof + MonthEnd(0)).
        spy_date = asof + pd.offsets.MonthEnd(0)
        regime = _spy_regime(spy_feat, spy_date) if use_crash_gate else "normal"

        do_reb = (i == 0) or (held_for >= hold_months) or in_cash

        if do_reb:
            sub = by_asof.get(asof, pd.DataFrame()).copy()
            sub = sub.dropna(subset=[score_col]) if not sub.empty else sub

            if regime == "crash" and use_crash_gate:
                cur_picks, cur_weights, in_cash = [], np.array([]), True
                held_for = 0
            elif sub.empty:
                in_cash = True
                held_for = 0
            else:
                top = sub.nlargest(K, score_col)
                cur_picks = top["ticker"].tolist()
                if weighting == "invvol" and "vol_12m" in sub.columns:
                    vols = sub.set_index("ticker")["vol_12m"].to_dict()
                    inv = np.array([1.0 / max(vols.get(t, 0.2), 0.05) for t in cur_picks])
                    cur_weights = inv / inv.sum()
                else:
                    cur_weights = np.ones(len(cur_picks)) / len(cur_picks)
                in_cash = False
                held_for = 0

        # Get next-month return
        next_d = _next_mr_date(asof, mr_idx)
        if in_cash or not cur_picks or next_d is None:
            ret_m = cash_m
        else:
            pick_rets = np.array([
                mr.at[next_d, t] if t in mr.columns and not np.isnan(mr.at[next_d, t]) else -1.0
                for t in cur_picks
            ])
            ret_m = float((pick_rets * cur_weights).sum())
            cash_w = max(0.0, 1.0 - cur_weights.sum())
            ret_m += cash_w * cash_m

        # Apply cost at rebalance
        cost = cf if (do_reb and not in_cash and cur_picks) else 0.0
        net_ret = ret_m - cost
        equity = max(0.0, equity * (1 + net_ret))

        if not in_cash:
            held_for += 1

        rows.append({
            "asof": asof,
            "next_date": next_d,
            "regime": regime,
            "in_cash": in_cash,
            "rebalanced": do_reb,
            "ret": net_ret,
            "equity": equity,
            "picks": "|".join(cur_picks[:3]),
        })

    df = pd.DataFrame(rows)
    rets = df["ret"]
    n = len(rets)
    n_years = n / 12.0
    cagr = equity ** (1 / n_years) - 1 if n_years > 0 else np.nan
    std_m = rets.std()
    sharpe = rets.mean() / std_m * np.sqrt(12) if std_m > 0 else np.nan
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    cash_months = int(df["in_cash"].sum())

    return {
        "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
        "n_months": n, "cash_months": cash_months,
        "final_equity": equity,
        "monthly_rets": rets,
        "detail": df,
    }
