"""Step 38d: rebound-confirmed DCA scaling.

Step 38b found that naive DCA-scaling at SPY -15% drawdown improves
20Y CAGR, but step 38c showed the entire gain comes from GFC 2008-10
and dd-scaling actually LOSES in normal 10Y windows (2010-2022).

Problem: -15% drawdowns that DON'T lead to bigger crashes are false
positives. Scaling up DCA into them catches falling knives.

Test: only scale DCA if SPY is BOTH below -20% AND above its 50-day
SMA (i.e., has started to recover). This is the "fish in the bottom
of the V" signal vs catching a falling knife.

Variants:
  - flat (baseline)
  - 3x at SPY -20% (naive)
  - 3x at SPY -20% AND SPY above 50DMA (confirmed rebound)
  - 3x at SPY -20% AND SPY > 20D high (stronger rebound)
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (compute_metrics, StrategyConfig, DCA_MONTHLY,
                     simulate_benchmark)
from bt_core_ext import load_and_prep_ext
from step38_dca_scaling import simulate_dd_scaled


md, start_m = load_and_prep_ext()
spy = md.bench_filled["SPY"]

# Peak/dd
pk = 0.0
spy_peak = np.zeros_like(spy)
for i, v in enumerate(spy):
    if v > pk: pk = v
    spy_peak[i] = pk

def dd(di):
    return 1.0 - spy[di] / spy_peak[di] if spy_peak[di] > 0 else 0.0

# SMAs
def sma(arr, w):
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        if i + 1 >= w:
            out[i] = float(np.mean(arr[i + 1 - w : i + 1]))
    return out

spy_50 = sma(spy, 50)
spy_200 = sma(spy, 200)
spy_20h = np.full_like(spy, np.nan)
for i in range(len(spy)):
    if i >= 20:
        spy_20h[i] = float(np.max(spy[i - 20 : i + 1]))

# Scalers
def flat(di, md): return DCA_MONTHLY

def scale_3x_minus_20(di, md):
    return DCA_MONTHLY * (3.0 if dd(di) >= 0.20 else 1.0)

def scale_3x_minus_20_above_50dma(di, md):
    return DCA_MONTHLY * (3.0 if (dd(di) >= 0.20 and
                                   math.isfinite(spy_50[di]) and
                                   spy[di] >= spy_50[di]) else 1.0)

def scale_3x_minus_20_above_20d_high(di, md):
    return DCA_MONTHLY * (3.0 if (dd(di) >= 0.20 and
                                   math.isfinite(spy_20h[di]) and
                                   spy[di] >= spy_20h[di] * 0.99) else 1.0)

# Also: scale during SPY BELOW 200DMA (persistent bear)
def scale_3x_below_200dma(di, md):
    return DCA_MONTHLY * (3.0 if (math.isfinite(spy_200[di]) and
                                   spy[di] < spy_200[di] * 0.90) else 1.0)


CONTENDERS = [
    ("flat (baseline)",                     flat),
    ("3x at -20%",                          scale_3x_minus_20),
    ("3x at -20% + above 50DMA",            scale_3x_minus_20_above_50dma),
    ("3x at -20% + above 20D high",         scale_3x_minus_20_above_20d_high),
    ("3x when SPY<0.9*200DMA (bear trend)", scale_3x_below_200dma),
]

cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)


# Full 20Y
print("\n" + "=" * 100)
print(f"{'variant':40s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Invested':>11s}  {'Final':>13s}")
print("-" * 100)

runs = {}
baseline_cagr = None
baseline_invest = None
baseline_final = None

for label, sc in CONTENDERS:
    r = simulate_dd_scaled(md, md.stocks, cfg, sc)
    m = compute_metrics(md, r.equity, r.total_invested)
    runs[label] = r
    if label == "flat (baseline)":
        baseline_cagr = m['cagr']; baseline_invest = m['invested']; baseline_final = m['final']
    print(f"{label:40s}  {m['cagr']*100:+6.2f}%  {m['sharpe']:6.2f}  "
          f"{-m['maxdd']*100:+6.2f}%  ${m['invested']:10,.0f}  ${m['final']:12,.0f}")


def win_metric(eq, from_i, to_i):
    if from_i >= to_i or to_i > len(eq): return None
    start = max(eq[from_i], 0.01)
    end = eq[to_i - 1]
    if end <= 0: return None
    yrs = (to_i - from_i) / 252
    cagr = (end / start) ** (1 / yrs) - 1
    peak, mdd = start, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak: peak = eq[i]
        if peak > 0:
            d = (peak - eq[i]) / peak
            if d > mdd: mdd = d
    rets = []
    for i in range(from_i + 1, to_i):
        if eq[i - 1] > 0:
            rets.append(eq[i] / eq[i - 1] - 1)
    sharpe = 0.0
    if len(rets) >= 10 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * math.sqrt(252)
    return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe}


# Rolling 10Y windows, step 2Y
print("\n## Rolling 10Y win rate vs flat baseline")
WIN_D = 10 * 252; STEP_D = 2 * 252
windows = [(i, i + WIN_D) for i in range(0, len(md.all_dates) - WIN_D, STEP_D)]
# Also add post-GFC windows
for y_start in ("2016", "2018"):
    for i, dd_ in enumerate(md.all_dates):
        if dd_ >= f"{y_start}-01-01":
            if i + 2000 < len(md.all_dates):  # need ~8Y left
                windows.append((i, min(i + WIN_D, len(md.all_dates) - 1)))
            break

for (from_i, to_i) in windows:
    d_from = md.all_dates[from_i][:7]
    d_to = md.all_dates[to_i - 1][:7]
    print(f"\n  {d_from} → {d_to}:")
    for label, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            print(f"    {label:40s}  CAGR {m['cagr']*100:+7.2f}%  Sharpe {m['sharpe']:.2f}")

# Calendar year since 2008
print("\n## Calendar years (2008+)")
yr_starts = {}
for i, dd_ in enumerate(md.all_dates):
    yr = dd_[:4]
    if yr not in yr_starts and int(yr) >= 2008:
        yr_starts[yr] = i

yr_keys = sorted(yr_starts.keys())
print(f"{'year':>5s}", end="")
for label, _ in CONTENDERS:
    print(f"  {label[:20]:>20s}", end="")
print()

wins = {label: 0 for label, _ in CONTENDERS}
n_yrs = 0
for i, yr in enumerate(yr_keys[:-1]):
    from_i = yr_starts[yr]
    to_i = yr_starts[yr_keys[i+1]]
    results = {}
    for label, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            results[label] = m['cagr']
    if not results: continue
    n_yrs += 1
    base = results.get("flat (baseline)", 0)
    print(f"  {yr}", end="")
    for label, _ in CONTENDERS:
        c = results.get(label, 0)
        print(f"  {c*100:+18.2f}%", end="")
        if label != "flat (baseline)" and c > base:
            wins[label] += 1
    print()

print(f"\nWin rate vs flat ({n_yrs} years):")
for label in wins:
    if label == "flat (baseline)": continue
    print(f"  {label:45s}  {wins[label]:3d}/{n_yrs}")


print("\n## DONE")
