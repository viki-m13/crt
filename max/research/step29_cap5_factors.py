"""Step 29: combine CAP5's 5% per-ticker cap with filters that step20
showed had individual promise BUT were never tested in combination.

Key insight from step20 (no 5% cap):
  - value lb=756d underperf>=+20%  →  CAP5 +9.70pp (vs +9.55 baseline CAP5)
    This was the ONLY filter that beat baseline CAGR. Sharpe 1.38 vs 1.34.
  - rebound lb=63d (bull-only) →  reduces MaxDD but loses CAGR

Step26/27/28 already ruled out simple CAP5R / CAP5RB (rebound alone).
Step 29 asks: does CAP5 + value, CAP5 + value + bull-rebound, or CAP5 +
value + sector_cap BEAT CAP5 on 20Y CAGR AND hold the 1Y recent edge?

Ranking rule: a variant is a candidate for production only if
  1. 20Y CAGR >= CAP5,   AND
  2. Rolling 10Y CAGR: CAP5-wins >= 6/11 (majority),   AND
  3. Trailing 1Y >= SPY (must survive current regime),  AND
  4. MaxDD <= CAP5 or Sharpe > CAP5.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
TO = len(md.all_dates)


def win_cagr_mdd(eq, invested, from_i, to_i):
    yrs = (to_i - from_i) / 252
    if invested <= 0 or yrs <= 0:
        return None, None
    final = eq[to_i - 1]
    if final <= 0:
        return -1.0, 1.0
    cagr = (final / invested) ** (1 / yrs) - 1
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    return cagr, mdd


BASE = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
            entry_delay=1)

# Variants to test
variants = [
    ("CAP5 baseline  ", dict(**BASE)),

    # Pure value factor added to CAP5 (the 5% cap was missing in step20)
    ("CAP5V 756/20   ", dict(**BASE, value_lookback_days=756, value_min_underperf=0.20)),
    ("CAP5V 756/10   ", dict(**BASE, value_lookback_days=756, value_min_underperf=0.10)),
    ("CAP5V 756/0    ", dict(**BASE, value_lookback_days=756, value_min_underperf=0.00)),
    ("CAP5V 504/20   ", dict(**BASE, value_lookback_days=504, value_min_underperf=0.20)),
    ("CAP5V 1008/20  ", dict(**BASE, value_lookback_days=1008, value_min_underperf=0.20)),
    ("CAP5V 1260/20  ", dict(**BASE, value_lookback_days=1260, value_min_underperf=0.20)),

    # Value + bull-only rebound (tries to keep value's CAGR lift AND
    # CAP5RB's MaxDD improvement)
    ("CAP5V+RB 756/20/63",
     dict(**BASE, value_lookback_days=756, value_min_underperf=0.20,
          rebound_lookback_days=63, rebound_only_in_bull=True)),
    ("CAP5V+RB 756/10/63",
     dict(**BASE, value_lookback_days=756, value_min_underperf=0.10,
          rebound_lookback_days=63, rebound_only_in_bull=True)),
    ("CAP5V+RB 756/20/126",
     dict(**BASE, value_lookback_days=756, value_min_underperf=0.20,
          rebound_lookback_days=126, rebound_only_in_bull=True)),

    # Value + sector_cap (diversification overlay)
    ("CAP5V+Sec2 756/20",
     dict(**BASE, value_lookback_days=756, value_min_underperf=0.20, sector_cap=2)),
    ("CAP5V+Sec3 756/20",
     dict(**BASE, value_lookback_days=756, value_min_underperf=0.20, sector_cap=3)),

    # Value + top_n variants
    ("CAP5V top_n=4 756/20",
     {**BASE, **dict(top_n=4, value_lookback_days=756, value_min_underperf=0.20)}),
    ("CAP5V top_n=6 756/20",
     {**BASE, **dict(top_n=6, value_lookback_days=756, value_min_underperf=0.20)}),

    # Tighter rebound thresholds, value required
    ("CAP5V 756/20 + RB126 bull thr=5%",
     dict(**BASE, value_lookback_days=756, value_min_underperf=0.20,
          rebound_lookback_days=126, rebound_min_return=0.05,
          rebound_only_in_bull=True)),
]


print("## 1. Headline 20Y")
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
print(f"  SPY DCA                 CAGR {bm['cagr']*100:+6.2f}%  MaxDD {bm['maxdd']*100:+6.2f}%  Sharpe {bm['sharpe']:.2f}")

headline = {}
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    print(f"  {label:24s} CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"Calmar {calmar:.2f}")
    headline[label.strip()] = dict(cagr=m['cagr'], mdd=m['maxdd'],
                                   sharpe=m['sharpe'], calmar=calmar)


print("\n## 2. Trailing 1Y (last 12 months)")
START_1Y = TOTAL_M - 12
FROM_1Y = md.month_first_idx[START_1Y]
b1 = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bc1, bd1 = win_cagr_mdd(b1.equity, 1000 * 12, FROM_1Y, TO)
print(f"  SPY DCA                 CAGR {bc1*100:+6.2f}%  MaxDD {bd1*100:+6.2f}%")
trail_1y = {}
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=START_1Y, **k)
    r = simulate(md, md.stocks, cfg)
    c, d = win_cagr_mdd(r.equity, r.total_invested, FROM_1Y, TO)
    ex = (c - bc1) * 100
    print(f"  {label:24s} CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")
    trail_1y[label.strip()] = c


print("\n## 3. Rolling 10Y windows — 11 windows, CAGR excess vs CAP5 (pp)")
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12
print(f"  Reports count of rolling-10Y wins vs CAP5 baseline.")

# Compute per-variant rolling 10Y CAGRs
window_cagrs = {label.strip(): [] for label, _ in variants}
window_spy = []
window_labels = []
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    nm = e_m - s_m
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, _ = win_cagr_mdd(b.equity, 1000 * nm, from_i, to_i)
    window_spy.append(bc)
    window_labels.append(f"{md.all_dates[from_i][:7]}->{md.all_dates[to_i-1][:7]}")
    for label, k in variants:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, _ = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        window_cagrs[label.strip()].append(c)

cap5 = window_cagrs["CAP5 baseline"]
print(f"\n  {'variant':26s}  {'wins vs CAP5':>12s}  {'med Δpp':>8s}  {'worst Δpp':>10s}  {'wins vs SPY':>11s}")
for label, _ in variants:
    nm2 = label.strip()
    cagrs = window_cagrs[nm2]
    diffs = [(c - c5) * 100 for c, c5 in zip(cagrs, cap5)]
    wins_cap5 = sum(1 for d in diffs if d > 0)
    wins_spy = sum(1 for c, s in zip(cagrs, window_spy) if c > s)
    med = sorted(diffs)[len(diffs) // 2]
    worst = min(diffs)
    print(f"  {label:26s}  {wins_cap5:>8d}/11    {med:+7.2f}  {worst:+9.2f}   {wins_spy:>7d}/11")


print("\n## 4. Candidates that beat CAP5 on 20Y CAGR")
cap5_cagr = headline["CAP5 baseline"]["cagr"]
cap5_sharpe = headline["CAP5 baseline"]["sharpe"]
cap5_mdd = headline["CAP5 baseline"]["mdd"]
for label, _ in variants:
    nm2 = label.strip()
    if nm2 == "CAP5 baseline":
        continue
    h = headline[nm2]
    if h["cagr"] > cap5_cagr:
        sharpe_improved = h["sharpe"] > cap5_sharpe
        mdd_improved = h["mdd"] < cap5_mdd
        one_y = trail_1y[nm2]
        cagrs = window_cagrs[nm2]
        wins = sum(1 for c, c5 in zip(cagrs, cap5) if c > c5)
        ok_1y = one_y > bc1
        tag = []
        if sharpe_improved: tag.append("Sharpe+")
        if mdd_improved: tag.append("MaxDD+")
        if wins >= 6: tag.append(f"rolling{wins}/11")
        if ok_1y: tag.append("1Y>SPY")
        print(f"  {label:26s}  20Y +{(h['cagr']-cap5_cagr)*100:.2f}pp  1Y {one_y*100:+.2f}%  rolling {wins}/11  [{', '.join(tag)}]")


print("\n## 5. GFC 10Y window detail (2006-2016) — stress test")
s_m = 0
e_m = 120
from_i = md.month_first_idx[s_m]
to_i = md.month_first_idx[e_m]
b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
bc, bd = win_cagr_mdd(b.equity, 1000 * 120, from_i, to_i)
print(f"  SPY DCA                 CAGR {bc*100:+6.2f}%  MaxDD {bd*100:+6.2f}%")
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=s_m, **k)
    r = simulate(md, md.stocks, cfg)
    c, d = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
    ex = (c - bc) * 100
    print(f"  {label:24s} CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")

print("\n## DONE")
