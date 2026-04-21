"""Step 14: walk-forward sanity check.

Split the spine into halves and quartiles. Rerun the leading strategies
on each subperiod to ensure the edge isn't a single-period artifact.
"""
import math
import numpy as np
from bt_core import (load_market, load_and_prep, simulate, simulate_benchmark,
                     compute_metrics, fmt_metrics, StrategyConfig)
import bt_core

md, _ = load_and_prep()


def run_window(start_month, end_month, label):
    # Truncate the simulation to a window. Easiest: artificially set start_month_idx;
    # we run normally but only compute metrics over [start_month, end_month] of equity.
    print(f"\n## Window {label}: {md.all_dates[md.month_first_idx[start_month]]} → "
          f"{md.all_dates[md.month_first_idx[min(end_month, len(md.month_first_idx)-1)]]}")
    # SPY benchmark for this window
    b = simulate_benchmark(md, ["SPY"], 5000, start_month, entry_delay=1)
    bm_total = compute_metrics(md, b.equity, b.total_invested)
    # Truncate equity to window for fair comparison
    end_idx = md.month_first_idx[min(end_month, len(md.month_first_idx) - 1)]
    # For a within-window CAGR, compute from the equity curve from rebalance start
    # to end_idx, using contributions invested up to end_idx.
    def window_metrics(eq, contributions_per_month):
        # Trim eq to within window and compute metrics
        start_idx = md.month_first_idx[start_month]
        # Estimate invested up to end_idx
        n_months = min(end_month, len(md.month_first_idx) - 1) - start_month
        invested = contributions_per_month * n_months
        if invested <= 0:
            return None
        # Build a "fake" md slice
        sub = bt_core.MarketData(
            all_dates=md.all_dates[start_idx:end_idx + 1],
            date_idx={dd: i for i, dd in enumerate(md.all_dates[start_idx:end_idx + 1])},
            prices=md.prices, finals=md.finals,
            month_first_idx=[i - start_idx for i in md.month_first_idx
                             if start_idx <= i <= end_idx],
            bench_filled=md.bench_filled, stocks=md.stocks,
            items_by_ticker=md.items_by_ticker,
        )
        eq_sub = eq[start_idx:end_idx + 1]
        # We need to subtract initial equity to get just window contributions
        # Easiest: re-treat starting equity as "invested" then add monthly inflows
        # Approximate window CAGR by ratio (final / start_market_value over yrs).
        # For an apples-comparison let me just report the in-window total return on invested + retained_value.
        starting_value = eq[start_idx] if eq[start_idx] > 0 else 0
        final_value = eq[end_idx]
        # treat starting_value as already invested, then add monthly inflows
        total_in = starting_value + invested
        if total_in <= 0:
            return None
        tr = (final_value - total_in) / total_in
        yrs = (end_idx - start_idx) / 252
        cagr = (final_value / total_in) ** (1 / yrs) - 1 if yrs > 0 else 0.0
        return {"cagr": cagr, "tr": tr, "start_val": starting_value,
                "invested_in_window": invested, "final": final_value, "yrs": yrs}

    bw = window_metrics(b.equity, 1000)
    if bw:
        print(f"  SPY DCA               CAGR {bw['cagr']*100:+6.2f}%  TR {bw['tr']*100:+6.2f}%  yrs={bw['yrs']:.1f}")

    for n in [1, 3, 5]:
        cfg = StrategyConfig(top_n=n, hold_days=5000, weighting="rank",
                             start_month_idx=start_month, entry_delay=1)
        r = simulate(md, md.stocks, cfg)
        rw = window_metrics(r.equity, 1000)
        if rw:
            excess = (rw["cagr"] - bw["cagr"]) * 100
            print(f"  Top-{n} rank hold-fwd  CAGR {rw['cagr']*100:+6.2f}%  TR {rw['tr']*100:+6.2f}%  excess {excess:+.2f}pp")


total_months = len(md.month_first_idx)
m_start = next(m for m, di in enumerate(md.month_first_idx)
               if md.all_dates[di] >= md.all_dates[md.month_first_idx[10]])
print(f"Total months in spine: {total_months}, first usable: {m_start}")

# Halves
half = (total_months - m_start) // 2
run_window(m_start, m_start + half, "first half")
run_window(m_start + half, total_months - 1, "second half")

# Quartiles
q = (total_months - m_start) // 4
run_window(m_start,           m_start + q,         "Q1")
run_window(m_start + q,       m_start + 2*q,       "Q2")
run_window(m_start + 2*q,     m_start + 3*q,       "Q3")
run_window(m_start + 3*q,     total_months - 1,    "Q4")
