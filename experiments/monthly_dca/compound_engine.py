"""True compounding portfolio backtester.

Key design vs `fast_engine.py`:
  * Tracks an actual portfolio with positions (entry_px, shares, peak)
  * Reinvests cash from exits into NEXT month's picks (compounding)
  * Realistic equity curve, not money-weighted XIRR with no reinvestment
  * Multiple exit rules supported in parallel via independent runs
  * Optional per-trade slippage and survivorship-bias overlay

This unlocks dramatically higher CAGR for strategies that exit (trailing
stop, fixed time, take-profit) and let the freed capital re-invest into the
next month's high-conviction picks.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import (
    CACHE,
    FEATURES_DIR,
    load_features,
    load_feature_months,
    load_panel,
    xirr,
)


BENCH_EXCLUDED = ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD")


# ---------------------------------------------------------------------------
# Exit rules — same shape as fast_engine but applied each day in simulation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExitSpec:
    name: str
    days: int = -1                    # exit after this many trading days (-1 = never by time)
    trail: float = 0.0                # trailing stop fraction
    hard_stop: float = 0.0            # hard stop on entry price
    take_profit: float = 0.0          # exit at +TP
    monthly_rebalance: bool = False   # if True, sell everything every month


REINVEST_RULES = [
    ExitSpec("hold_forever"),
    ExitSpec("trail_25", trail=0.25),
    ExitSpec("trail_35", trail=0.35),
    ExitSpec("trail_50", trail=0.50),
    ExitSpec("fixed_1y", days=252),
    ExitSpec("fixed_3y", days=252 * 3),
    ExitSpec("monthly_rebalance", monthly_rebalance=True),
    ExitSpec("trail35_or_3y", trail=0.35, days=252 * 3),
    ExitSpec("trail50_or_5y", trail=0.50, days=252 * 5),
]


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    shares: float
    peak: float
    exit_date: Optional[pd.Timestamp] = None
    exit_px: Optional[float] = None
    days_held: int = 0
    forced_exit: bool = False


@dataclass
class CompoundResult:
    strategy: str
    exit_rule: str
    top_k: int
    n_months: int
    n_trades: int
    total_deposited: float
    final_equity: float
    cagr_money_weighted: float       # XIRR with deposits + final equity
    cagr_total_money: float          # raw final/deposited^(1/years)
    equity_curve: pd.DataFrame       # date, equity, cash, n_positions
    trades: pd.DataFrame             # entry/exit log


# ---------------------------------------------------------------------------
# Strategy type
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    score_fn: Callable[[pd.DataFrame], pd.Series]
    top_k: int = 5
    description: str = ""


# ---------------------------------------------------------------------------
# Main compound backtest
# ---------------------------------------------------------------------------
def run_compound(
    panel: pd.DataFrame,
    strat: Strategy,
    exit_rule: ExitSpec,
    start: str = "2002-01-31",
    end: str = "2024-12-31",
    monthly_deposit: float = 1.0,
    eval_at: Optional[pd.Timestamp] = None,
    delist_alpha: float = 0.0,            # synthetic delisting prob/yr
    delist_seed: int = 0,
    exclude: tuple[str, ...] = BENCH_EXCLUDED,
    cost_bps: float = 5.0,                # round-trip cost in bps; applied at entry+exit
) -> CompoundResult:
    """Simulate a compounding monthly DCA portfolio.

    Mechanics:
      * On each month-end T:
          1. Mark-to-market existing positions, evaluate exit conditions.
          2. If a position triggers exit -> sell at next-day open (here we use
             same-day close as approximation), credit cash.
          3. If `monthly_rebalance` is set, also sell ALL existing positions.
          4. Add `monthly_deposit` to cash.
          5. Score features at T, pick top-K. If no picks (regime says cash),
             cash sits.
          6. Allocate ALL cash equally across the top-K picks (full deployment).
          7. New positions enter at the SAME-DAY close at T (next-month open
             approximation: we use T's close).
      * Daily loop between month-ends to evaluate exits (trailing stop, hard
        stop, take-profit, fixed-day).
      * Cost: cost_bps charged on every entry and exit.

    Returns:
      CompoundResult with equity curve and trade log.
    """
    months = load_feature_months()
    months = [m for m in months if pd.Timestamp(start) <= m <= pd.Timestamp(end)]
    if not months:
        raise ValueError(f"No feature months in [{start}, {end}]")
    panel_idx = panel.index
    if eval_at is None:
        eval_at = panel_idx[-1]
    eval_pos = panel_idx.searchsorted(eval_at, side="right") - 1
    eval_at_panel = panel_idx[eval_pos]

    # Precompute month-end positions in panel
    month_positions = []
    for m in months:
        pos = panel_idx.searchsorted(m)
        if pos >= len(panel_idx):
            break
        if panel_idx[pos] != m:
            pos = max(0, pos - 1)
        month_positions.append(pos)
    if not month_positions:
        raise ValueError("No valid month positions in panel")

    cost_factor = cost_bps / 10000.0

    # State
    cash = 0.0
    positions: list[Position] = []
    trades: list[dict] = []
    equity_history: list[dict] = []
    deposits: list[tuple[pd.Timestamp, float]] = []
    rng = np.random.default_rng(delist_seed)

    last_panel_pos = month_positions[-1]

    def evaluate_exit_check(pos: Position, day_idx: int, current_px: float) -> tuple[bool, str]:
        days_held = day_idx - panel_idx.get_loc(pos.entry_date)
        if days_held < 1:
            return False, ""
        if exit_rule.monthly_rebalance:
            return False, ""  # handled at month boundary
        if exit_rule.days > 0 and days_held >= exit_rule.days:
            return True, "time_exit"
        if exit_rule.hard_stop > 0 and current_px <= pos.entry_px * (1.0 - exit_rule.hard_stop):
            return True, "hard_stop"
        if exit_rule.trail > 0 and current_px <= pos.peak * (1.0 - exit_rule.trail):
            return True, "trailing_stop"
        if exit_rule.take_profit > 0 and current_px >= pos.entry_px * (1.0 + exit_rule.take_profit):
            return True, "take_profit"
        return False, ""

    def force_close_all(at_pos: int, reason: str):
        nonlocal cash
        date_t = panel_idx[at_pos]
        for p in positions[:]:
            curr_px = float(panel[p.ticker].iloc[at_pos])
            if not np.isfinite(curr_px):
                # Forced exit at last available finite price
                slc = panel[p.ticker].iloc[panel_idx.get_loc(p.entry_date): at_pos + 1]
                last_finite = slc.dropna()
                if last_finite.empty:
                    # Catastrophic loss
                    p.exit_date = date_t
                    p.exit_px = 0.0
                    p.forced_exit = True
                    trades.append(_pos_to_dict(p, "delist", cost_factor))
                    positions.remove(p)
                    continue
                curr_px = float(last_finite.iloc[-1])
            cash += p.shares * curr_px * (1.0 - cost_factor)
            p.exit_date = date_t
            p.exit_px = curr_px
            trades.append(_pos_to_dict(p, reason, cost_factor))
            positions.remove(p)

    # Iterate through each panel day, but we only need to "do" things at
    # month-ends and at days where exits may trigger. To keep it simple, we
    # walk daily but skip cheap days.
    panel_arr = panel.to_numpy()  # for speed
    # Map ticker -> column index
    col_idx = {t: i for i, t in enumerate(panel.columns)}

    # We start at first month-end's panel position
    cur_panel_pos = month_positions[0]
    next_month_idx = 0

    while cur_panel_pos <= last_panel_pos:
        date_t = panel_idx[cur_panel_pos]

        # 1) Update peaks + check exits for each position
        for p in positions[:]:
            ci = col_idx.get(p.ticker)
            if ci is None:
                continue
            curr_px = panel_arr[cur_panel_pos, ci]
            if not np.isfinite(curr_px):
                # Maybe delisted or temp NaN — count days, check if it's gone
                # If many consecutive NaNs, treat as delisted
                # Simple heuristic: if no finite for last 21 days, treat as delisted
                start = max(0, cur_panel_pos - 21)
                if not np.any(np.isfinite(panel_arr[start: cur_panel_pos + 1, ci])):
                    # Forced exit at 0
                    cash += p.shares * 0.0
                    p.exit_date = date_t
                    p.exit_px = 0.0
                    p.forced_exit = True
                    trades.append(_pos_to_dict(p, "delist", cost_factor))
                    positions.remove(p)
                continue
            curr_px = float(curr_px)
            if curr_px > p.peak:
                p.peak = curr_px
            should_exit, reason = evaluate_exit_check(p, cur_panel_pos, curr_px)
            if should_exit:
                cash += p.shares * curr_px * (1.0 - cost_factor)
                p.exit_date = date_t
                p.exit_px = curr_px
                trades.append(_pos_to_dict(p, reason, cost_factor))
                positions.remove(p)

        # Are we at a month-end? If so, do month-end logic.
        is_month_end = (next_month_idx < len(month_positions) and
                        cur_panel_pos == month_positions[next_month_idx])
        if is_month_end:
            # 2) If monthly rebalance, sell everything
            if exit_rule.monthly_rebalance and positions:
                force_close_all(cur_panel_pos, "monthly_rebalance")

            # 3) Add deposit
            cash += monthly_deposit
            deposits.append((date_t, -monthly_deposit))

            # 4) Score features
            try:
                feats = load_features(date_t)
            except Exception:
                feats = None

            picks = []
            if feats is not None:
                scores = strat.score_fn(feats).dropna()
                # Exclude bench, and existing positions (don't double up)
                excl = list(exclude) + [p.ticker for p in positions if not exit_rule.monthly_rebalance]
                scores = scores.drop(labels=[t for t in excl if t in scores.index], errors="ignore")
                if not scores.empty:
                    top = scores.sort_values(ascending=False).head(strat.top_k)
                    picks = list(top.index)

            # 5) If we have picks AND cash, deploy
            if picks and cash > 0:
                per_pick = cash / len(picks)
                for tkr in picks:
                    ci = col_idx.get(tkr)
                    if ci is None:
                        continue
                    px = panel_arr[cur_panel_pos, ci]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    px = float(px)
                    # Synthetic delisting check — probability scaled to expected
                    # holding period. For monthly_rebalance, ~1mo. For hold_forever
                    # or fixed_3y, longer. We approximate with a per-entry annualized
                    # rate and scale to expected holding years.
                    if delist_alpha > 0:
                        if exit_rule.monthly_rebalance:
                            expected_years = 1.0 / 12.0
                        elif exit_rule.days > 0:
                            expected_years = exit_rule.days / 252.0
                        elif exit_rule.trail > 0:
                            # Empirically about 12-18 months for trail_25/35
                            expected_years = 1.5 if exit_rule.trail < 0.4 else 2.0
                        else:
                            # hold_forever: until eval
                            expected_years = max(1.0, (eval_at_panel - date_t).days / 365.25)
                        # Cap expected_years to avoid runaway
                        expected_years = min(expected_years,
                                             max(1.0, (eval_at_panel - date_t).days / 365.25))
                        p_del = 1.0 - (1.0 - delist_alpha) ** expected_years
                        if rng.random() < p_del:
                            shares = (per_pick * (1.0 - cost_factor)) / px
                            trades.append({
                                "ticker": tkr, "entry_date": date_t, "exit_date": date_t,
                                "entry_px": px, "exit_px": 0.0, "shares": shares,
                                "ret": -1.0, "days_held": 0, "reason": "synthetic_delist",
                                "pnl": -per_pick,
                            })
                            cash -= per_pick  # locked in loss
                            continue
                    shares = (per_pick * (1.0 - cost_factor)) / px
                    cash -= per_pick
                    positions.append(Position(
                        ticker=tkr, entry_date=date_t, entry_px=px,
                        shares=shares, peak=px,
                    ))
            next_month_idx += 1

        # Snapshot equity each month-end (sparse curve to limit memory)
        if is_month_end:
            mtm = cash
            n_pos = len(positions)
            for p in positions:
                ci = col_idx.get(p.ticker)
                if ci is None:
                    continue
                px = panel_arr[cur_panel_pos, ci]
                if np.isfinite(px):
                    mtm += p.shares * float(px)
            equity_history.append({"date": date_t, "equity": mtm, "cash": cash, "n_positions": n_pos})

        cur_panel_pos += 1

    # Final mark-to-market at eval_at_panel
    final_eq = cash
    for p in positions[:]:
        ci = col_idx.get(p.ticker)
        if ci is None:
            continue
        px = panel_arr[eval_pos, ci]
        if not np.isfinite(px):
            # use last finite
            slc = panel_arr[: eval_pos + 1, ci]
            mask = np.isfinite(slc)
            if mask.any():
                px = float(slc[mask][-1])
            else:
                px = 0.0
        else:
            px = float(px)
        final_eq += p.shares * px

    total_deposited = -sum(d for _, d in deposits)
    years = (eval_at_panel - deposits[0][0]).days / 365.25 if deposits else 1.0
    cagr_total = (final_eq / max(total_deposited, 1e-9)) ** (1.0 / max(years, 0.1)) - 1.0

    cashflows = list(deposits) + [(eval_at_panel, final_eq)]
    cagr_xirr = xirr(cashflows)

    eq_df = pd.DataFrame(equity_history)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return CompoundResult(
        strategy=strat.name,
        exit_rule=exit_rule.name,
        top_k=strat.top_k,
        n_months=len(deposits),
        n_trades=len(trades),
        total_deposited=total_deposited,
        final_equity=final_eq,
        cagr_money_weighted=cagr_xirr,
        cagr_total_money=cagr_total,
        equity_curve=eq_df,
        trades=trades_df,
    )


def _pos_to_dict(p: Position, reason: str, cost_factor: float) -> dict:
    if p.exit_px is None:
        return {}
    days_held = (p.exit_date - p.entry_date).days
    raw_ret = p.exit_px / p.entry_px - 1.0 if p.entry_px > 0 else -1.0
    # net of round-trip cost
    net_ret = (1 + raw_ret) * (1 - cost_factor) ** 2 - 1.0
    return {
        "ticker": p.ticker,
        "entry_date": p.entry_date,
        "exit_date": p.exit_date,
        "entry_px": p.entry_px,
        "exit_px": p.exit_px,
        "shares": p.shares,
        "ret": net_ret,
        "days_held": days_held,
        "reason": reason,
        "pnl": p.shares * (p.exit_px - p.entry_px),
    }


def benchmark_spy_dca(
    panel: pd.DataFrame,
    start: str,
    end: str,
    monthly_deposit: float = 1.0,
    eval_at: Optional[pd.Timestamp] = None,
) -> dict:
    """SPY DCA: deposit $1 each month, hold to eval, with no rebalance/exit."""
    months = load_feature_months()
    months = [m for m in months if pd.Timestamp(start) <= m <= pd.Timestamp(end)]
    panel_idx = panel.index
    if eval_at is None:
        eval_at = panel_idx[-1]
    eval_pos = panel_idx.searchsorted(eval_at, side="right") - 1
    eval_at_panel = panel_idx[eval_pos]
    spy = panel["SPY"]
    cashflows = []
    final = 0.0
    for m in months:
        pos = panel_idx.searchsorted(m)
        if pos >= len(panel_idx):
            break
        if panel_idx[pos] != m:
            pos = max(0, pos - 1)
        px = float(spy.iloc[pos])
        if not np.isfinite(px):
            continue
        cashflows.append((panel_idx[pos], -monthly_deposit))
        eval_px = float(spy.iloc[eval_pos])
        final += (monthly_deposit / px) * eval_px
    cashflows.append((eval_at_panel, final))
    cagr_xirr = xirr(cashflows)
    total_dep = -sum(c[1] for c in cashflows[:-1])
    years = (eval_at_panel - cashflows[0][0]).days / 365.25
    cagr_total = (final / total_dep) ** (1.0 / max(years, 0.1)) - 1.0
    return {"final": final, "deposited": total_dep, "cagr_xirr": cagr_xirr, "cagr_total": cagr_total}


__all__ = [
    "ExitSpec", "REINVEST_RULES", "Position", "CompoundResult",
    "Strategy", "run_compound", "benchmark_spy_dca", "BENCH_EXCLUDED",
]
