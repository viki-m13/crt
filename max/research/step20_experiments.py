"""Step 20: systematic gate/filter experiments on the 20y extended spine.

Each experiment runs Top-5 rank-weighted hold-forever on the full 2006-2026
extended history and reports headline + trailing-1Y to make sure we're not
buying 20y CAGR at the cost of recent performance.

E1: per-ticker concentration cap (0, 5%, 8%, 10%, 15%)
E2: market-regime gate (SPY 200DMA — off, scale-down to 0, redirect to SPY)
E3: zombie filter (reject picks whose trailing 3y/5y return < threshold)
E4: price-based value factor (require underperformed SPY by X% over 3y/5y)
E5: rebound confirmation (require positive trailing 63d/126d return)
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
START_1Y = TOTAL_M - 12
FROM_1Y = md.month_first_idx[START_1Y]
TO = len(md.all_dates)


def win_metrics(eq, invested, from_i, to_i):
    if invested <= 0:
        return None
    yrs = (to_i - from_i) / 252
    final = eq[to_i - 1]
    cagr = (final / invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    tr = (final - invested) / invested
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    return dict(cagr=cagr, tr=tr, mdd=mdd, final=final, invested=invested)


# Benchmarks
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm_full = compute_metrics(md, bench_full.equity, bench_full.total_invested)
bench_1y = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bm_1y = win_metrics(bench_1y.equity, 1000 * 12, FROM_1Y, TO)

print(f"SPY full 20y: CAGR {bm_full['cagr']*100:+6.2f}%  MaxDD {bm_full['maxdd']*100:+6.2f}%")
print(f"SPY 1Y     : CAGR {bm_1y['cagr']*100:+6.2f}%  MaxDD {bm_1y['mdd']*100:+6.2f}%\n")


def run(label, **kwargs):
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1, **kwargs)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)

    # 1Y window using the same strategy but starting from START_1Y
    cfg1 = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                          start_month_idx=START_1Y, entry_delay=1, **kwargs)
    r1 = simulate(md, md.stocks, cfg1)
    m1 = win_metrics(r1.equity, r1.total_invested, FROM_1Y, TO)

    if m is None or m1 is None:
        print(f"  {label:42s}  FAILED (no positions?)"); return

    ex20 = (m["cagr"] - bm_full["cagr"]) * 100
    ex1 = (m1["cagr"] - bm_1y["cagr"]) * 100
    n_pos = len([p for p in r.positions])
    print(f"  {label:42s}  20y CAGR {m['cagr']*100:+6.2f}% ({ex20:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"1Y {m1['cagr']*100:+6.2f}% ({ex1:+5.2f}pp)  "
          f"n_pos {n_pos}")


print("## E0: baseline")
run("baseline Top-5 rank hold-forever")

print("\n## E1: per-ticker concentration cap")
for frac in [0.05, 0.08, 0.10, 0.15, 0.20]:
    run(f"max_ticker_frac={frac:.2f}", max_ticker_frac=frac)

print("\n## E2: market-regime gate (SPY 200DMA)")
for sd in [0.0, 0.25, 0.5, 1.0]:
    run(f"regime_gate scale_down={sd:.2f}", regime_gate=True, regime_scale_down=sd)
# Redirect skipped DCA into SPY when bear regime
for sd in [0.0]:
    run(f"regime_gate SD={sd:.2f} + fallback=SPY",
        regime_gate=True, regime_scale_down=sd, fallback_ticker="SPY")

print("\n## E3: zombie filter (trailing 3y/5y return floor)")
for lb, thr in [(756, -0.50), (756, -0.30), (756, -0.10), (1260, -0.50), (1260, -0.20)]:
    run(f"zombie lb={lb}d thr={thr:+.0%}",
        zombie_lookback_days=lb, zombie_min_return=thr)

print("\n## E4: price-based value factor (underperformed SPY by X%)")
for lb, mu in [(756, 0.0), (756, 0.10), (756, 0.20), (1260, 0.0), (1260, 0.20)]:
    run(f"value lb={lb}d underperf>={mu:+.0%}",
        value_lookback_days=lb, value_min_underperf=mu)

print("\n## E5: rebound confirmation (positive trailing 63d/126d return)")
for lb, thr in [(63, 0.0), (63, 0.05), (126, 0.0), (126, 0.05), (252, 0.0)]:
    run(f"rebound lb={lb}d thr>={thr:+.0%}",
        rebound_lookback_days=lb, rebound_min_return=thr)
