"""Step 39b: rolling-window validation of signal smoothing.

Step 39 found CAP5 with 6M trailing average score improves 20Y CAGR by
+0.70pp at zero extra capital. Validate whether this is robust across
rolling 10Y windows or another GFC artifact.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (compute_metrics, StrategyConfig, simulate_benchmark)
from bt_core_ext import load_and_prep_ext
from step39_signal_smoothing import simulate_smoothed, current, trailing


md, start_m = load_and_prep_ext()
cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

CONTENDERS = [
    ("current (incumbent)", current, None),
    ("trailing 2M",         trailing(2), None),
    ("trailing 3M",         trailing(3), None),
    ("trailing 6M",         trailing(6), None),
    ("trailing 12M",        trailing(12), None),
]

# Run simulations once, slice equity curves for windows
runs = {}
for label, sf, pm in CONTENDERS:
    r = simulate_smoothed(md, md.stocks, cfg, sf, persist_months=pm)
    runs[label] = r


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


# Rolling 10Y, step 1Y
print("## Rolling 10Y windows (step 1Y, shows all available windows)")
WIN_D = 10 * 252; STEP_D = 1 * 252
print(f"{'window':21s}", end="")
for label, _, _ in CONTENDERS:
    print(f"  {label[:15]:>15s}", end="")
print()

stats = {label: [] for label, _, _ in CONTENDERS}
for from_i in range(0, len(md.all_dates) - WIN_D, STEP_D):
    to_i = from_i + WIN_D
    d_from = md.all_dates[from_i][:7]
    d_to = md.all_dates[to_i - 1][:7]
    print(f"{d_from} → {d_to}", end="")
    for label, _, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            stats[label].append(m['cagr'])
            print(f"  {m['cagr']*100:+14.2f}%", end="")
        else:
            print(f"  {'—':>15s}", end="")
    print()

print(f"\n{'variant':25s}  {'median CAGR':>11s}  {'mean CAGR':>10s}  {'wins vs incumbent':>17s}")
base_cagrs = stats.get("current (incumbent)", [])
for label, _, _ in CONTENDERS:
    cs = stats[label]
    if not cs: continue
    wins = sum(1 for a, b in zip(cs, base_cagrs) if a > b) if label != "current (incumbent)" else len(cs)
    print(f"{label:25s}  {np.median(cs)*100:+10.2f}%  {np.mean(cs)*100:+9.2f}%  {wins:4d}/{len(cs)}")


# Calendar year
print("\n## Calendar year (2008+)")
yr_starts = {}
for i, dd_ in enumerate(md.all_dates):
    yr = dd_[:4]
    if yr not in yr_starts and int(yr) >= 2008:
        yr_starts[yr] = i

yr_keys = sorted(yr_starts.keys())
wins = {label: 0 for label, _, _ in CONTENDERS}
n_yrs = 0
yr_data = {}
for i, yr in enumerate(yr_keys[:-1]):
    from_i = yr_starts[yr]
    to_i = yr_starts[yr_keys[i+1]]
    results = {}
    for label, _, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            results[label] = m['cagr']
    if not results: continue
    n_yrs += 1
    yr_data[yr] = results
    base = results.get("current (incumbent)", 0)
    for label in results:
        if label != "current (incumbent)" and results[label] > base:
            wins[label] += 1

print(f"{'year':>5s}", end="")
for label, _, _ in CONTENDERS:
    print(f"  {label[:15]:>15s}", end="")
print()
for yr in yr_keys[:-1]:
    if yr not in yr_data: continue
    print(f"  {yr}", end="")
    for label, _, _ in CONTENDERS:
        c = yr_data[yr].get(label, 0)
        print(f"  {c*100:+14.2f}%", end="")
    print()

print(f"\nWin rate vs incumbent:")
for label, _, _ in CONTENDERS:
    if label == "current (incumbent)": continue
    print(f"  {label:25s}  {wins[label]:3d}/{n_yrs}")


# GFC-only vs post-GFC-only
print("\n## Period split")
for name, yr_from, yr_to in [("GFC 2008-2012", "2008", "2012"),
                              ("Post-GFC bull 2012-2020", "2012", "2020"),
                              ("Post-COVID 2020-2026", "2020", "2026")]:
    print(f"\n{name}:")
    from_i = yr_starts.get(yr_from)
    to_i = yr_starts.get(yr_to, len(md.all_dates))
    if from_i is None: continue
    for label, _, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            print(f"  {label:25s}  CAGR {m['cagr']*100:+.2f}%  Sharpe {m['sharpe']:.2f}  MaxDD {-m['mdd']*100:+.1f}%")

print("\n## DONE")
