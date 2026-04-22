"""Step 38c: rolling-window validation of DCA-scaling.

Step 38b found 5x-at-SPY-15% improves CAP5 CAGR by +2.04pp (20Y).
Validate that this isn't a period-specific fluke by running the
top 3 variants on:
  - Rolling 10Y windows (11 overlapping)
  - Calendar year-by-year
  - GFC decade (2008-2018)
  - Pre-COVID vs post-COVID
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
spy_peak = np.zeros_like(spy)
pk = 0.0
for i, v in enumerate(spy):
    if v > pk: pk = v
    spy_peak[i] = pk
def spy_dd(di):
    if spy_peak[di] <= 0: return 0.0
    return 1.0 - spy[di] / spy_peak[di]


def mk_scaler(threshold, multiplier):
    def sc(di, md):
        return DCA_MONTHLY * (multiplier if spy_dd(di) >= threshold else 1.0)
    return sc


def flat(di, md): return DCA_MONTHLY


cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)


CONTENDERS = [
    ("flat (baseline)", flat),
    ("2x @ -15%", mk_scaler(0.15, 2.0)),
    ("3x @ -15%", mk_scaler(0.15, 3.0)),
    ("5x @ -15%", mk_scaler(0.15, 5.0)),
    ("5x @ -10%", mk_scaler(0.10, 5.0)),
    ("3x @ -10%", mk_scaler(0.10, 3.0)),
]


# Run full-period once per variant, then slice equity curves.
runs = {}
for label, sc in CONTENDERS:
    r = simulate_dd_scaled(md, md.stocks, cfg, sc)
    runs[label] = r

spy_bench = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)


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
            dd = (peak - eq[i]) / peak
            if dd > mdd: mdd = dd
    rets = []
    for i in range(from_i + 1, to_i):
        if eq[i - 1] > 0:
            rets.append(eq[i] / eq[i - 1] - 1)
    sharpe = 0.0
    if len(rets) >= 10 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * math.sqrt(252)
    return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe}


# Rolling 10Y windows, step 2Y
print("## Rolling 10Y windows")
WIN_D = 10 * 252
STEP_D = 2 * 252
baseline_label = "flat (baseline)"

win_from_list = list(range(0, len(md.all_dates) - WIN_D, STEP_D))
print(f"{'window':21s}", end="")
for label, _ in CONTENDERS:
    print(f"  {label[:11]:>11s}", end="")
print()

cagr_sums = {label: [] for label, _ in CONTENDERS}
for from_i in win_from_list:
    to_i = from_i + WIN_D
    d_from = md.all_dates[from_i][:7]
    d_to = md.all_dates[to_i - 1][:7]
    print(f"{d_from} → {d_to}", end="")
    base_cagr = None
    for label, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m is None:
            print("          —", end="")
            continue
        cagr_sums[label].append(m['cagr'])
        if label == baseline_label:
            base_cagr = m['cagr']
            print(f"  {m['cagr']*100:+10.2f}%", end="")
        else:
            delta = (m['cagr'] - base_cagr) * 100 if base_cagr is not None else 0
            print(f"  {m['cagr']*100:+6.2f}% ({delta:+5.2f})", end="")
    print()

print("\nRolling 10Y summary:")
print(f"{'variant':20s}  {'median CAGR':>11s}  {'mean CAGR':>10s}  {'wins vs flat':>13s}")
base_cagrs = cagr_sums[baseline_label]
for label, _ in CONTENDERS:
    cs = cagr_sums[label]
    if not cs: continue
    wins = sum(1 for a, b in zip(cs, base_cagrs) if a > b) if label != baseline_label else len(cs)
    print(f"{label:20s}  {np.median(cs)*100:+10.2f}%  {np.mean(cs)*100:+9.2f}%  {wins:13d}/{len(cs)}")


# Calendar-year 1Y
print("\n## Calendar year 1Y")
yr_starts = {}
for i, dd in enumerate(md.all_dates):
    yr = dd[:4]
    if yr not in yr_starts:
        yr_starts[yr] = i
yr_keys = sorted(yr_starts.keys())

yr_cagrs = {label: {} for label, _ in CONTENDERS}
for i, yr in enumerate(yr_keys[:-1]):
    from_i = yr_starts[yr]
    to_i = yr_starts[yr_keys[i+1]]
    for label, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            yr_cagrs[label][yr] = m['cagr']

print(f"{'year':>6s}", end="")
for label, _ in CONTENDERS:
    print(f"  {label[:11]:>11s}", end="")
print()
for yr in yr_keys[:-1]:
    print(f"  {yr}", end="")
    base = yr_cagrs[baseline_label].get(yr, 0)
    for label, _ in CONTENDERS:
        c = yr_cagrs[label].get(yr)
        if c is None:
            print("           —", end="")
        else:
            mark = "+" if (label != baseline_label and c > base) else " "
            print(f"  {c*100:+10.2f}%{mark}", end="")
    print()

print(f"\n{'variant':20s}  {'median':>8s}  {'mean':>8s}  {'wins':>10s}")
for label, _ in CONTENDERS:
    cs = list(yr_cagrs[label].values())
    if not cs: continue
    if label == baseline_label:
        wins = len(cs)
    else:
        wins = sum(1 for yr, c in yr_cagrs[label].items() if c > yr_cagrs[baseline_label].get(yr, 0))
    print(f"{label:20s}  {np.median(cs)*100:+7.2f}%  {np.mean(cs)*100:+7.2f}%  {wins:5d}/{len(cs)}")


# GFC decade and post-COVID
print("\n## GFC decade (2008-2018) and post-COVID (2020-2026)")
for name, yr_from, yr_to in [("GFC 2008-2018", "2008", "2018"), ("Post-COVID 2020-2026", "2020", "2026")]:
    print(f"\n{name}:")
    from_i = yr_starts.get(yr_from)
    to_i = yr_starts.get(yr_to, len(md.all_dates))
    if from_i is None:
        continue
    for label, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            print(f"  {label:20s}  CAGR {m['cagr']*100:+.2f}%  Sharpe {m['sharpe']:.2f}  MaxDD {-m['mdd']*100:+.2f}%")

print("\n## DONE")
