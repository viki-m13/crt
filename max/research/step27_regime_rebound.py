"""Step 27: regime-conditional rebound gate.

Step26 found CAP5R (CAP5 + 63d rebound) lowers MaxDD (46%->39%) and
massively boosts recent 1Y (+14.77pp) BUT loses to CAP5 on CAGR in
every single rolling 10Y window — the gate kneecaps V-shaped
recoveries during GFC-style crashes.

Hypothesis: apply the rebound gate ONLY when SPY is >= 200DMA
("in-bull"). Bear markets revert to plain CAP5 so we don't miss
the mean-reversion rally off the lows.

We test this ("CAP5RB" = CAP5 rebound bull-only) across the same
battery as step25:
  - Headline 20Y
  - Rolling 10Y windows (all 11)
  - Trailing 1Y
  - Head-to-head vs both CAP5 and CAP5R
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


# Variants
CAP5 = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
            entry_delay=1)
CAP5R = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
             entry_delay=1, rebound_lookback_days=63, rebound_min_return=0.0)
CAP5RB = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
              entry_delay=1, rebound_lookback_days=63, rebound_min_return=0.0,
              rebound_only_in_bull=True)

# Also test a few tuning variants of CAP5RB
CAP5RB_126 = {**CAP5RB, "rebound_lookback_days": 126}
CAP5RB_6 = {**CAP5RB, "top_n": 6}


print("## 1. Headline 20Y")
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
print(f"  SPY DCA       CAGR {bm['cagr']*100:+6.2f}%  MaxDD {bm['maxdd']*100:+6.2f}%  Sharpe {bm['sharpe']:.2f}")

variants = [
    ("CAP5 (baseline) ", CAP5),
    ("CAP5R (rebound) ", CAP5R),
    ("CAP5RB (bull-on)", CAP5RB),
    ("CAP5RB_126      ", CAP5RB_126),
    ("CAP5RB + top_n=6", CAP5RB_6),
]

headline = {}
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    print(f"  {label} CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"Calmar {calmar:.2f}")
    headline[label.strip()] = r


print("\n## 2. Trailing 1Y (last 12 months)")
START_1Y = TOTAL_M - 12
FROM_1Y = md.month_first_idx[START_1Y]
b1 = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bc1, bd1 = win_cagr_mdd(b1.equity, 1000 * 12, FROM_1Y, TO)
print(f"  SPY DCA       CAGR {bc1*100:+6.2f}%  MaxDD {bd1*100:+6.2f}%")
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=START_1Y, **k)
    r = simulate(md, md.stocks, cfg)
    c, d = win_cagr_mdd(r.equity, r.total_invested, FROM_1Y, TO)
    ex = (c - bc1) * 100
    print(f"  {label} CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")


print("\n## 3. Rolling 10Y windows — CAGR excess vs SPY (pp)")
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12
print(f"  {len(windows)} overlapping 10Y windows")
print(f"  {'window':20s}  {'SPY':>7s}  {'CAP5':>8s}  {'CAP5R':>8s}  {'CAP5RB':>8s}  {'CAP5RB126':>8s}")
win_stats = {label.strip(): {"beats_spy": 0, "beats_cap5": 0, "beats_cap5r": 0,
                             "excesses": [], "cagrs": []} for label, _ in variants}
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    nm = e_m - s_m
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, _ = win_cagr_mdd(b.equity, 1000 * nm, from_i, to_i)
    row = {}
    for label, k in variants:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, _ = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        row[label.strip()] = c
        ex = (c - bc) * 100
        win_stats[label.strip()]["excesses"].append(ex)
        win_stats[label.strip()]["cagrs"].append(c)
        if c > bc:
            win_stats[label.strip()]["beats_spy"] += 1
    cap5_c = row["CAP5 (baseline)"]
    cap5r_c = row["CAP5R (rebound)"]
    for label, _ in variants:
        nm2 = label.strip()
        if row[nm2] > cap5_c:
            win_stats[nm2]["beats_cap5"] += 1
        if row[nm2] > cap5r_c:
            win_stats[nm2]["beats_cap5r"] += 1
    lbl = f"{md.all_dates[from_i][:7]}->{md.all_dates[to_i-1][:7]}"
    print(f"  {lbl:20s}  {bc*100:+6.2f}%  {(row['CAP5 (baseline)']-bc)*100:+6.1f}  "
          f"{(row['CAP5R (rebound)']-bc)*100:+6.1f}  "
          f"{(row['CAP5RB (bull-on)']-bc)*100:+6.1f}  "
          f"{(row['CAP5RB_126']-bc)*100:+6.1f}")

print("\n  Summary (11 windows):")
for label, _ in variants:
    nm2 = label.strip()
    s = win_stats[nm2]
    med_ex = sorted(s["excesses"])[len(s["excesses"]) // 2]
    print(f"    {label} beats SPY: {s['beats_spy']:2d}/11  beats CAP5: {s['beats_cap5']:2d}/11  "
          f"beats CAP5R: {s['beats_cap5r']:2d}/11  median excess {med_ex:+5.2f}pp")


print("\n## 4. GFC decade specifically (2006-03 -> 2016-03)")
# Force the exact first window
s_m = 0
e_m = 120 if 120 <= TOTAL_M else TOTAL_M
from_i = md.month_first_idx[s_m]
to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
bc, bd = win_cagr_mdd(b.equity, 1000 * (e_m - s_m), from_i, to_i)
print(f"  SPY DCA       CAGR {bc*100:+6.2f}%  MaxDD {bd*100:+6.2f}%")
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=s_m, **k)
    r = simulate(md, md.stocks, cfg)
    c, d = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
    ex = (c - bc) * 100
    print(f"  {label} CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")
