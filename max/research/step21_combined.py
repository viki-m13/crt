"""Step 21: combine winning gates from step20.

Winners from step20:
  - E1 max_ticker_frac=0.05          (concentration cap: 20y +9.55, 1Y +8.00)
  - E4 value lb=756d underperf>=+20% (value factor: 20y +9.70, 1Y +7.67)
  - E5 rebound lb=126d thr>=+5%      (rebound: 20y +8.18, 1Y +8.11, MaxDD 45%)
  - E5 rebound lb=63d thr>=+0%       (rebound: 20y +7.92, 1Y +8.95, MaxDD 39%)

Test pairwise and triple combos, then sweep Top-N.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
START_1Y = TOTAL_M - 12
START_5Y = TOTAL_M - 60
FROM_1Y = md.month_first_idx[START_1Y]
FROM_5Y = md.month_first_idx[START_5Y]
TO = len(md.all_dates)


def win_metrics(eq, invested, from_i, to_i):
    if invested <= 0:
        return None
    yrs = (to_i - from_i) / 252
    final = eq[to_i - 1]
    cagr = (final / invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    return dict(cagr=cagr, mdd=mdd, final=final, invested=invested)


bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm_full = compute_metrics(md, bench_full.equity, bench_full.total_invested)
bench_1y = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bm_1y = win_metrics(bench_1y.equity, 1000 * 12, FROM_1Y, TO)
bench_5y = simulate_benchmark(md, ["SPY"], 5000, START_5Y, entry_delay=1)
bm_5y = win_metrics(bench_5y.equity, 1000 * 60, FROM_5Y, TO)

print(f"SPY  20y: CAGR {bm_full['cagr']*100:+6.2f}%  MaxDD {bm_full['maxdd']*100:+6.2f}%")
print(f"SPY   5y: CAGR {bm_5y['cagr']*100:+6.2f}%  MaxDD {bm_5y['mdd']*100:+6.2f}%")
print(f"SPY   1y: CAGR {bm_1y['cagr']*100:+6.2f}%  MaxDD {bm_1y['mdd']*100:+6.2f}%\n")


def run(label, top_n=5, **kwargs):
    cfg = StrategyConfig(top_n=top_n, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1, **kwargs)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)

    cfg5 = StrategyConfig(top_n=top_n, hold_days=5000, weighting="rank",
                          start_month_idx=START_5Y, entry_delay=1, **kwargs)
    r5 = simulate(md, md.stocks, cfg5)
    m5 = win_metrics(r5.equity, r5.total_invested, FROM_5Y, TO)

    cfg1 = StrategyConfig(top_n=top_n, hold_days=5000, weighting="rank",
                          start_month_idx=START_1Y, entry_delay=1, **kwargs)
    r1 = simulate(md, md.stocks, cfg1)
    m1 = win_metrics(r1.equity, r1.total_invested, FROM_1Y, TO)

    if m is None or m1 is None or m5 is None:
        print(f"  {label:50s}  FAILED"); return None

    ex20 = (m["cagr"] - bm_full["cagr"]) * 100
    ex5 = (m5["cagr"] - bm_5y["cagr"]) * 100
    ex1 = (m1["cagr"] - bm_1y["cagr"]) * 100
    print(f"  {label:50s}  20y {m['cagr']*100:+6.2f}% ({ex20:+5.2f}pp) DD {m['maxdd']*100:+5.1f}% Sh {m['sharpe']:.2f}  "
          f"5y {m5['cagr']*100:+6.2f}% ({ex5:+5.2f}pp)  "
          f"1y {m1['cagr']*100:+6.2f}% ({ex1:+5.2f}pp)")
    return dict(m=m, m5=m5, m1=m1, ex20=ex20, ex5=ex5, ex1=ex1)


print("## Baseline reference")
run("baseline Top-5 rank")

# Gate building blocks
CAP5   = dict(max_ticker_frac=0.05)
VAL20  = dict(value_lookback_days=756, value_min_underperf=0.20)
VAL10  = dict(value_lookback_days=756, value_min_underperf=0.10)
REB63  = dict(rebound_lookback_days=63,  rebound_min_return=0.00)
REB126 = dict(rebound_lookback_days=126, rebound_min_return=0.05)

print("\n## Pairs (Top-5)")
run("CAP5 + VAL20",  **CAP5, **VAL20)
run("CAP5 + REB126", **CAP5, **REB126)
run("CAP5 + REB63",  **CAP5, **REB63)
run("VAL20 + REB126",**VAL20, **REB126)
run("VAL20 + REB63", **VAL20, **REB63)
run("VAL10 + REB126",**VAL10, **REB126)

print("\n## Triples (Top-5)")
run("CAP5 + VAL20 + REB126", **CAP5, **VAL20, **REB126)
run("CAP5 + VAL20 + REB63",  **CAP5, **VAL20, **REB63)
run("CAP5 + VAL10 + REB126", **CAP5, **VAL10, **REB126)

print("\n## Best combo — sweep Top-N")
best = dict(**CAP5, **VAL20, **REB126)
for n in [1, 3, 5, 8, 10]:
    run(f"Top-{n} CAP5+VAL20+REB126", top_n=n, **best)

print("\n## Alt triple — sweep Top-N")
alt = dict(**CAP5, **VAL20, **REB63)
for n in [1, 3, 5, 8, 10]:
    run(f"Top-{n} CAP5+VAL20+REB63", top_n=n, **alt)

print("\n## Softer value (10%) — sweep Top-N")
soft = dict(**CAP5, **VAL10, **REB126)
for n in [1, 3, 5, 8, 10]:
    run(f"Top-{n} CAP5+VAL10+REB126", top_n=n, **soft)
