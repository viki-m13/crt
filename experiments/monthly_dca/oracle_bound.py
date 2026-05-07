"""Compute the *theoretical upper bound* on CAGR if we had perfect foresight.

For each month-end, pick the K stocks that delivered the best forward N-year
return (oracle). Compute the DCA-portfolio CAGR. This tells us the achievable
ceiling for our universe -- and how far short any honest strategy must fall.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import (
    DEFAULT_RULES,
    ExitRule,
    load_features,
    load_panel,
    simulate_exit,
    xirr,
)
from experiments.monthly_dca.backtester import month_end_dates


def oracle_bound(
    panel: pd.DataFrame,
    start: str = "2017-12-31",
    end: str = "2024-12-31",
    top_k: int = 5,
    rule: ExitRule = ExitRule("hold_forever"),
) -> dict:
    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp(end))]
    eval_at = panel.index.max()
    eval_pos = panel.index.searchsorted(eval_at, side="right") - 1

    spy = panel["SPY"]
    cashflows: list[tuple[pd.Timestamp, float]] = []
    cashflows_spy: list[tuple[pd.Timestamp, float]] = []
    sum_terminal = 0.0
    sum_terminal_spy = 0.0

    bench_excluded = {"SPY", "QQQ", "IWM", "VTI", "RSP"}

    rets_per_pick = []
    for asof in months:
        try:
            feats = load_features(asof)
        except FileNotFoundError:
            continue
        # Compute forward return under the rule for every ticker, then take top K
        candidates = [t for t in feats.index if t not in bench_excluded]
        rets = np.full(len(candidates), np.nan)
        pos = panel.index.searchsorted(asof)
        if pos >= len(panel.index):
            continue
        for i, t in enumerate(candidates):
            if t not in panel.columns:
                continue
            arr = panel[t].iloc[pos: eval_pos + 1].to_numpy(dtype=float)
            if len(arr) == 0 or not np.isfinite(arr[0]):
                continue
            r, _, _ = simulate_exit(arr, rule)
            rets[i] = r
        valid = np.isfinite(rets)
        if not valid.any():
            continue
        # top K
        order = np.argsort(rets)[::-1]
        chosen = [candidates[i] for i in order[:top_k] if np.isfinite(rets[i])]
        chosen_rets = [rets[i] for i in order[:top_k] if np.isfinite(rets[i])]
        for cr in chosen_rets:
            cashflows.append((pd.Timestamp(asof), -1.0))
            sum_terminal += 1 + cr
            rets_per_pick.append(cr)
        # SPY same dates
        spy_arr = spy.iloc[pos: eval_pos + 1].to_numpy(dtype=float)
        spy_r, _, _ = simulate_exit(spy_arr, rule)
        for _ in chosen_rets:
            cashflows_spy.append((pd.Timestamp(asof), -1.0))
            sum_terminal_spy += 1 + (spy_r if np.isfinite(spy_r) else 0)

    if not cashflows:
        return {"cagr": float("nan")}
    cashflows.append((panel.index[eval_pos], sum_terminal))
    cashflows_spy.append((panel.index[eval_pos], sum_terminal_spy))
    return {
        "n_picks": len(rets_per_pick),
        "median_pick_ret": float(np.nanmedian(rets_per_pick)),
        "win_rate": float(np.mean([r > 0 for r in rets_per_pick])),
        "cagr_dca": xirr(cashflows),
        "cagr_spy_dca": xirr(cashflows_spy),
    }


def main() -> None:
    panel = load_panel()
    print("Universe ceiling (oracle picks at each month-end):")
    for k in (1, 3, 5, 10):
        for rule in (ExitRule("hold_forever"), ExitRule("fixed_3y", days=252 * 3), ExitRule("fixed_1y", days=252)):
            r = oracle_bound(panel, top_k=k, rule=rule)
            print(f"  top_k={k:2d}  rule={rule.name:12s}  n={r['n_picks']:5d}  "
                  f"median_ret={r['median_pick_ret']:+.3f}  win_rate={r['win_rate']:.3f}  "
                  f"CAGR={r['cagr_dca']:.3f}  SPY_DCA={r['cagr_spy_dca']:.3f}")


if __name__ == "__main__":
    main()
