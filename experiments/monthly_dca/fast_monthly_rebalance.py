"""Fast specialized monthly-rebalance backtester.

When the exit rule is `monthly_rebalance`, we don't need a daily loop —
each month we sell everything and buy fresh. So we only need to compute
month-to-month returns per pick.

This is 10-100x faster than the generic compound_engine for monthly_rebalance.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import benchmark_spy_dca, BENCH_EXCLUDED
from experiments.monthly_dca.fast_engine import (
    load_features, load_feature_months, load_panel, xirr,
)


def run_monthly_rebalance(
    panel: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], pd.Series],
    top_k: int = 5,
    start: str = "2002-01-31",
    end: str = "2024-12-31",
    monthly_deposit: float = 1.0,
    eval_at: pd.Timestamp | None = None,
    cost_bps: float = 5.0,
    delist_alpha: float = 0.0,
    delist_seed: int = 0,
    exclude=BENCH_EXCLUDED,
):
    months = load_feature_months()
    months = [m for m in months if pd.Timestamp(start) <= m <= pd.Timestamp(end)]
    if not months:
        raise ValueError("no months")
    if eval_at is None:
        eval_at = panel.index[-1]
    eval_at = pd.Timestamp(eval_at)

    panel_idx = panel.index
    eval_pos = panel_idx.searchsorted(eval_at, side="right") - 1
    eval_at_panel = panel_idx[eval_pos]

    # Find the panel position for each month-end
    month_pos = []
    for m in months:
        pos = panel_idx.searchsorted(m)
        if pos >= len(panel_idx):
            break
        if panel_idx[pos] != m:
            pos = max(0, pos - 1)
        month_pos.append(pos)
    if len(month_pos) < 2:
        raise ValueError("not enough months")

    # Append eval_at as end position
    rng = np.random.default_rng(delist_seed)
    cost = cost_bps / 10000.0

    cash = 0.0
    deposits = []
    equity_curve = []
    trades_log = []

    panel_arr = panel.to_numpy()
    col_idx = {t: i for i, t in enumerate(panel.columns)}

    for i, pos in enumerate(month_pos):
        date_t = panel_idx[pos]
        # Add monthly deposit
        cash += monthly_deposit
        deposits.append((date_t, -monthly_deposit))

        try:
            feats = load_features(date_t)
        except Exception:
            equity_curve.append({"date": date_t, "equity": cash})
            continue

        scores = score_fn(feats).dropna()
        scores = scores.drop(labels=[t for t in exclude if t in scores.index], errors="ignore")
        if scores.empty:
            equity_curve.append({"date": date_t, "equity": cash})
            continue
        top = scores.sort_values(ascending=False).head(top_k)

        # Determine sell date (next month's pos, or eval_at)
        if i + 1 < len(month_pos):
            sell_pos = month_pos[i + 1]
            sell_date = panel_idx[sell_pos]
        else:
            sell_pos = eval_pos
            sell_date = eval_at_panel

        # Allocate cash equally
        per_pick = cash / len(top)
        new_value = 0.0
        for tkr, score in top.items():
            ci = col_idx.get(tkr)
            if ci is None:
                continue
            entry_px = panel_arr[pos, ci]
            if not np.isfinite(entry_px) or entry_px <= 0:
                # Couldn't enter, return cash
                new_value += per_pick
                continue
            exit_px = panel_arr[sell_pos, ci]
            if not np.isfinite(exit_px):
                # Walk back to last finite
                slc = panel_arr[pos: sell_pos + 1, ci]
                mask = np.isfinite(slc)
                if mask.any():
                    exit_px = float(slc[mask][-1])
                else:
                    exit_px = 0.0
            # Synthetic delisting check (1-month horizon)
            if delist_alpha > 0:
                p_del = 1.0 - (1.0 - delist_alpha) ** (1.0 / 12.0)
                if rng.random() < p_del:
                    new_value += 0.0
                    trades_log.append({
                        "ticker": tkr, "entry_date": date_t, "exit_date": sell_date,
                        "entry_px": float(entry_px), "exit_px": 0.0, "ret": -1.0,
                        "reason": "synthetic_delist", "score": float(score),
                    })
                    continue
            # Compute net return after costs
            shares = (per_pick * (1.0 - cost)) / float(entry_px)
            ret = (float(exit_px) - float(entry_px)) / float(entry_px)
            net_value = shares * float(exit_px) * (1.0 - cost)
            new_value += net_value
            trades_log.append({
                "ticker": tkr, "entry_date": date_t, "exit_date": sell_date,
                "entry_px": float(entry_px), "exit_px": float(exit_px),
                "ret": ret, "reason": "monthly_rebalance", "score": float(score),
            })
        cash = new_value
        equity_curve.append({"date": date_t, "equity": cash})

    # Final equity at eval_at
    final_eq = cash
    deposited = -sum(d for _, d in deposits)
    cashflows = list(deposits) + [(eval_at_panel, final_eq)]
    cagr = xirr(cashflows)
    years = (eval_at_panel - deposits[0][0]).days / 365.25
    cagr_total = (final_eq / deposited) ** (1.0 / max(years, 0.1)) - 1.0

    return {
        "cagr_xirr": cagr,
        "cagr_total": cagr_total,
        "final_equity": final_eq,
        "deposited": deposited,
        "n_trades": len(trades_log),
        "trades": pd.DataFrame(trades_log),
        "equity_curve": pd.DataFrame(equity_curve),
        "n_months": len(deposits),
    }


def smoke_test():
    from experiments.monthly_dca.strategies_ensemble import strategy_rotation
    from experiments.monthly_dca.strategies_apex import apex_balanced
    panel = load_panel()
    print("Smoke test: strategy_rotation k=5 monthly_rebalance, full 2002-2024", flush=True)
    res = run_monthly_rebalance(panel, strategy_rotation, top_k=5,
                                 start="2002-01-31", end="2024-12-31",
                                 eval_at=pd.Timestamp("2026-05-07"))
    print(f"  CAGR={res['cagr_xirr']:.4f} final={res['final_equity']:.2f}/dep={res['deposited']:.0f} "
          f"n_trades={res['n_trades']}", flush=True)


if __name__ == "__main__":
    smoke_test()
