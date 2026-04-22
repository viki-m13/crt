"""Step 39g: smoothing + defensive gates.

Test whether SMA 12M smoothing combines additively with other
step36 building blocks:
- sector_cap (prevent over-concentration in any single sector)
- rebound_only_in_bull (require SPY>200DMA for strong rebound)
- rebound gates
- score_threshold

Since smoothing changes pick composition (more financials, less NEM/SMCI),
these gates may interact differently with smoothed vs raw ranking.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import simulate, compute_metrics, StrategyConfig
from bt_core_ext import load_and_prep_ext


md, start_m = load_and_prep_ext()


def go(label, **overrides):
    cfg = StrategyConfig(
        start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
        hold_days=5000, weighting="rank", entry_delay=1,
        **overrides,
    )
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    return (label, m, r)


CONTENDERS = [
    go("baseline (no smooth)",                smoothing_months=0),
    go("SMA 12M",                             smoothing_months=12),
    go("SMA 12M + sector_cap=2",              smoothing_months=12, sector_cap=2),
    go("SMA 12M + sector_cap=3",              smoothing_months=12, sector_cap=3),
    go("SMA 12M + rebound 63d",               smoothing_months=12, rebound_lookback_days=63,
       rebound_min_return=0.0),
    go("SMA 12M + rebound 63d bull-only",     smoothing_months=12, rebound_lookback_days=63,
       rebound_min_return=0.0, rebound_only_in_bull=True),
    go("SMA 12M + score_thresh=50",           smoothing_months=12, score_threshold=50.0),
    go("SMA 12M + score_thresh=70",           smoothing_months=12, score_threshold=70.0),
    go("SMA 12M + min_score=20",              smoothing_months=12, min_score=20.0),
    go("SMA 12M + zombie -50% over 3Y",       smoothing_months=12,
       zombie_lookback_days=756, zombie_min_return=-0.5),
]

print(f"\n{'variant':40s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Final':>13s}  {'Δ vs SMA12M':>12s}")
print("-" * 110)
sma_cagr = None
for label, m, _ in CONTENDERS:
    if label == "SMA 12M" and 'cagr' in m: sma_cagr = m['cagr']
for label, m, _ in CONTENDERS:
    if 'cagr' not in m:
        print(f"{label:40s}  (no picks — all filtered out)")
        continue
    delta = (m['cagr'] - sma_cagr) * 100 if sma_cagr is not None else 0
    print(f"{label:40s}  {m['cagr']*100:+6.2f}%  {m['sharpe']:6.2f}  "
          f"{-m['maxdd']*100:+6.2f}%  ${m['final']:12,.0f}  {delta:+8.2f}pp")


def win_metric(eq, from_i, to_i):
    if from_i >= to_i or to_i > len(eq): return None
    s = max(eq[from_i], 0.01); e = eq[to_i - 1]
    if e <= 0: return None
    yrs = (to_i - from_i) / 252
    return {"cagr": (e / s) ** (1 / yrs) - 1}


# Rolling 10Y
print("\n## Rolling 10Y (step 2Y) — does adding a gate lose alpha?")
WIN_D = 10 * 252
header = f"{'window':21s}"
for label, _, _ in CONTENDERS:
    header += f"  {label[:15]:>15s}"
print(header)
wins = {label: 0 for label, _, _ in CONTENDERS}
n_win = 0
base_run = next(r for lbl, _, r in CONTENDERS if lbl == "baseline (no smooth)")
for from_i in range(0, len(md.all_dates) - WIN_D, 2 * 252):
    to_i = from_i + WIN_D
    bm = win_metric(base_run.equity, from_i, to_i)
    if not bm: continue
    n_win += 1
    row = f"{md.all_dates[from_i][:7]} → {md.all_dates[to_i-1][:7]}"
    for label, _, r in CONTENDERS:
        m = win_metric(r.equity, from_i, to_i)
        if m:
            row += f"  {m['cagr']*100:+14.2f}%"
            if m['cagr'] > bm['cagr']: wins[label] += 1
        else:
            row += f"  {'—':>15s}"
    print(row)
print(f"\nwins vs baseline ({n_win} windows):")
for label, _, _ in CONTENDERS:
    if "baseline" in label: continue
    print(f"  {label:40s}  {wins[label]:3d}/{n_win}")


# Calendar year
print("\n## Calendar year (SMA 12M variant wins)")
yr_starts = {}
for i, dd_ in enumerate(md.all_dates):
    yr = dd_[:4]
    if yr not in yr_starts and int(yr) >= 2008:
        yr_starts[yr] = i
yr_keys = sorted(yr_starts.keys())
yr_wins = {label: 0 for label, _, _ in CONTENDERS}
n_yr = 0
for i, yr in enumerate(yr_keys[:-1]):
    fi = yr_starts[yr]
    ti = yr_starts[yr_keys[i+1]]
    bm = win_metric(base_run.equity, fi, ti)
    if not bm: continue
    n_yr += 1
    for label, _, r in CONTENDERS:
        m = win_metric(r.equity, fi, ti)
        if m and m['cagr'] > bm['cagr']: yr_wins[label] += 1
print(f"\ncalendar year wins vs baseline ({n_yr} years):")
for label, _, _ in CONTENDERS:
    if "baseline" in label: continue
    print(f"  {label:40s}  {yr_wins[label]:3d}/{n_yr}")

print("\n## DONE")
