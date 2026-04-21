"""Step 22: pick finalists, walk-forward validate to check for 1Y overfitting.

Finalists:
  A. Baseline Top-5 rank (no gates)                    — 20y +9.26, 1y +2.58
  B. CAP5 only                                         — 20y +9.55, 1y +8.00 (strictly better than A)
  C. CAP5 + VAL20                                      — 20y +7.11, 1y +11.24
  D. CAP5 + VAL20 + REB126 (Top-1)                     — 20y +6.94, 1y +21.28, DD 39.7
  E. CAP5 + VAL10 + REB126 (Top-1)                     — 20y +6.15, 1y +24.69, DD 35.3 (< SPY!)

Validation: split 20Y into 4 quartiles of ~5Y each, report CAGR/excess per window
per strategy. If a strategy is great in the last window only, that's overfitting.
Also report full-20Y Sharpe, MaxDD, and Calmar.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
TO = len(md.all_dates)


def win_cagr(eq, invested, from_i, to_i):
    if invested <= 0 or to_i <= from_i:
        return None
    yrs = (to_i - from_i) / 252
    final = eq[to_i - 1]
    if final <= 0:
        return -1.0
    return (final / invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0


def win_mdd(eq, from_i, to_i):
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak: peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd: mdd = dd
    return mdd


# Quartile boundaries in months
Q_MONTHS = [
    ("Q1 2006-2011", 0,   TOTAL_M // 4),
    ("Q2 2011-2016", TOTAL_M // 4, TOTAL_M // 2),
    ("Q3 2016-2021", TOTAL_M // 2, 3 * TOTAL_M // 4),
    ("Q4 2021-2026", 3 * TOTAL_M // 4, TOTAL_M),
]


def spy_for(start_m_local, n_months):
    b = simulate_benchmark(md, ["SPY"], 5000, start_m_local, entry_delay=1)
    from_i = md.month_first_idx[start_m_local]
    to_i = md.month_first_idx[start_m_local + n_months] if (start_m_local + n_months) < TOTAL_M else TO
    return win_cagr(b.equity, 1000 * n_months, from_i, to_i), win_mdd(b.equity, from_i, to_i)


def strat_for(start_m_local, n_months, **kwargs):
    cfg = StrategyConfig(hold_days=5000, weighting="rank",
                         start_month_idx=start_m_local, entry_delay=1, **kwargs)
    r = simulate(md, md.stocks, cfg)
    from_i = md.month_first_idx[start_m_local]
    to_i = md.month_first_idx[start_m_local + n_months] if (start_m_local + n_months) < TOTAL_M else TO
    return (win_cagr(r.equity, r.total_invested, from_i, to_i),
            win_mdd(r.equity, from_i, to_i))


STRATS = [
    ("A baseline Top-5",       dict(top_n=5)),
    ("B CAP5 Top-5",           dict(top_n=5, max_ticker_frac=0.05)),
    ("C CAP5+VAL20 Top-5",     dict(top_n=5, max_ticker_frac=0.05,
                                    value_lookback_days=756, value_min_underperf=0.20)),
    ("D CAP5+VAL20+REB126 T1", dict(top_n=1, max_ticker_frac=0.05,
                                    value_lookback_days=756, value_min_underperf=0.20,
                                    rebound_lookback_days=126, rebound_min_return=0.05)),
    ("E CAP5+VAL10+REB126 T1", dict(top_n=1, max_ticker_frac=0.05,
                                    value_lookback_days=756, value_min_underperf=0.10,
                                    rebound_lookback_days=126, rebound_min_return=0.05)),
    ("F CAP5+VAL10+REB126 T3", dict(top_n=3, max_ticker_frac=0.05,
                                    value_lookback_days=756, value_min_underperf=0.10,
                                    rebound_lookback_days=126, rebound_min_return=0.05)),
    ("G CAP5+REB63 Top-5",     dict(top_n=5, max_ticker_frac=0.05,
                                    rebound_lookback_days=63, rebound_min_return=0.00)),
]


print("## Per-quartile CAGR (CAGR / SPY_CAGR / excess-pp)")
print(f"{'strategy':28s} | " + " | ".join(f"{lbl:16s}" for lbl, _, _ in Q_MONTHS))

# Precompute per-quartile SPY baseline
spy_q = []
for _, s_m, e_m in Q_MONTHS:
    cagr, mdd = spy_for(s_m, e_m - s_m)
    spy_q.append((cagr, mdd))

print(f"{'SPY DCA':28s} | " + " | ".join(
    f"{c*100:+6.2f}% DD {m*100:4.1f}%" for c, m in spy_q))

all_results = {}
for name, kwargs in STRATS:
    cells = []
    for (lbl, s_m, e_m), (spy_c, spy_m) in zip(Q_MONTHS, spy_q):
        c, m = strat_for(s_m, e_m - s_m, **kwargs)
        ex = (c - spy_c) * 100
        cells.append((c, m, ex))
    all_results[name] = cells
    print(f"{name:28s} | " + " | ".join(
        f"{c*100:+6.2f}% ({ex:+4.1f})" for c, m, ex in cells))


print("\n## Full 20Y headline")
print(f"{'strategy':28s}  {'CAGR':>8s} {'ex':>6s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'Calmar':>7s}")

bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
calmar = bm['cagr'] / bm['maxdd'] if bm['maxdd'] > 0 else 0
print(f"{'SPY DCA':28s}  {bm['cagr']*100:+7.2f}% {0.0:+5.2f}  {bm['maxdd']*100:+6.2f}%  {bm['sharpe']:>7.2f}  {calmar:>7.2f}")

for name, kwargs in STRATS:
    cfg = StrategyConfig(hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1, **kwargs)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    print(f"{name:28s}  {m['cagr']*100:+7.2f}% {ex:+5.2f}  {m['maxdd']*100:+6.2f}%  {m['sharpe']:>7.2f}  {calmar:>7.2f}")


print("\n## Rolling 5Y windows (CAGR, excess over SPY in pp)")
# 5Y = 60 months; step 12 months
windows = []
s_m = 0
while s_m + 60 <= TOTAL_M:
    windows.append((s_m, s_m + 60))
    s_m += 12
print(f"  {len(windows)} overlapping 5Y windows")
header = f"  {'window':18s}  " + "  ".join(f"{n[:18]:>18s}" for n, _ in STRATS)
print(header)
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc = win_cagr(b.equity, 1000 * 60, from_i, to_i)
    label = f"{md.all_dates[from_i][:7]}->{md.all_dates[to_i-1][:7]}"
    cells = []
    for name, kwargs in STRATS:
        cfg = StrategyConfig(hold_days=5000, weighting="rank",
                             start_month_idx=s_m, entry_delay=1, **kwargs)
        r = simulate(md, md.stocks, cfg)
        c = win_cagr(r.equity, r.total_invested, from_i, to_i)
        ex = (c - bc) * 100 if c is not None else float("nan")
        cells.append(f"{ex:+5.1f}pp")
    print(f"  {label:18s}  " + "  ".join(f"{s:>18s}" for s in cells))
