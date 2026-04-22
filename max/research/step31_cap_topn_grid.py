"""Step 31: cap × top_n grid to find the true joint optimum.

Step30 found:
  - cap=10% has 10/11 rolling-10Y wins vs CAP5 (median +1.79pp) but
    -0.15pp on 20Y CAGR.
  - top_n=3 has 8/11 rolling wins BUT loses 1Y (-4pp).
  - CAP5 (cap=5%, top_n=5) wins 20Y.

Hypothesis: there's a sweet spot at (cap, top_n) that delivers both
higher 20Y CAGR AND the rolling-window stability of the higher-cap
variants. This step grids over (cap ∈ {5, 7, 10, 15, 20, None}) ×
(top_n ∈ {3, 4, 5, 6, 7}) and reports full metrics.
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


BASE = dict(hold_days=5000, weighting="rank", entry_delay=1)
CAPS = [None, 0.05, 0.07, 0.10, 0.15, 0.20]
TOPNS = [3, 4, 5, 6, 7]

# Build variant list
variants = []
for cap in CAPS:
    for tn in TOPNS:
        label = f"cap={'none' if cap is None else f'{int(cap*100)}%':>4s} top_n={tn}"
        cfg_dict = dict(**BASE, top_n=tn, max_ticker_frac=cap)
        variants.append((label, cfg_dict))


# Precompute SPY benchmarks for each rolling window to save time
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12

bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm_full = compute_metrics(md, bench_full.equity, bench_full.total_invested)
START_1Y = TOTAL_M - 12
FROM_1Y = md.month_first_idx[START_1Y]
b1 = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bc1, bd1 = win_cagr_mdd(b1.equity, 1000 * 12, FROM_1Y, TO)

spy_rolling = []
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, _ = win_cagr_mdd(b.equity, 1000 * (e_m - s_m), from_i, to_i)
    spy_rolling.append(bc)


print(f"SPY 20Y CAGR {bm_full['cagr']*100:+.2f}%  MaxDD {bm_full['maxdd']*100:+.2f}%")
print(f"SPY 1Y  CAGR {bc1*100:+.2f}%\n")


results = {}
print(f"  {'variant':26s}  {'20Y':>7s}  {'MaxDD':>7s}  {'Sharpe':>6s}  {'1Y':>7s}  "
      f"{'rol wins vs SPY':>16s}  {'med ex vs SPY':>14s}")

for label, k in variants:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)

    cfg1 = StrategyConfig(start_month_idx=START_1Y, **k)
    r1 = simulate(md, md.stocks, cfg1)
    c1, _ = win_cagr_mdd(r1.equity, r1.total_invested, FROM_1Y, TO)

    rolling = []
    for (s_m, e_m), spy_c in zip(windows, spy_rolling):
        from_i = md.month_first_idx[s_m]
        to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
        cfg_w = StrategyConfig(start_month_idx=s_m, **k)
        rr = simulate(md, md.stocks, cfg_w)
        c, _ = win_cagr_mdd(rr.equity, rr.total_invested, from_i, to_i)
        rolling.append(c)

    wins_spy = sum(1 for c, s in zip(rolling, spy_rolling) if c > s)
    diffs_spy = [(c - s) * 100 for c, s in zip(rolling, spy_rolling)]
    med_ex_spy = sorted(diffs_spy)[len(diffs_spy) // 2]

    results[label] = dict(
        cagr=m['cagr'], mdd=m['maxdd'], sharpe=m['sharpe'],
        cagr_1y=c1, rolling=rolling, wins_spy=wins_spy, med_ex_spy=med_ex_spy)

    print(f"  {label:26s}  {m['cagr']*100:+.2f}%  {m['maxdd']*100:+.2f}%  {m['sharpe']:+.2f}  "
          f"{c1*100:+.2f}%  {wins_spy:>10d}/11       {med_ex_spy:+.2f}pp")


# Reference: CAP5 (cap=5%, top_n=5)
ref = results["cap=  5% top_n=5"]
print("\n## Candidates that beat CAP5 on ALL of {20Y CAGR, 1Y CAGR, rolling 10Y wins}")
print(f"  CAP5 ref: 20Y {ref['cagr']*100:+.2f}%  1Y {ref['cagr_1y']*100:+.2f}%  "
      f"Sharpe {ref['sharpe']:.2f}  MaxDD {ref['mdd']*100:+.2f}%  "
      f"rolling median-ex-SPY {ref['med_ex_spy']:+.2f}pp  rolling-wins-SPY {ref['wins_spy']}/11")
print()
n_cands = 0
for label, _ in variants:
    if label == "cap=  5% top_n=5":
        continue
    r = results[label]
    if r['cagr'] > ref['cagr'] and r['cagr_1y'] > ref['cagr_1y']:
        # How does rolling-window behavior compare to CAP5?
        rolling_vs_cap5 = [(c - cap5c) * 100 for c, cap5c in zip(r['rolling'], ref['rolling'])]
        wins_cap5 = sum(1 for d in rolling_vs_cap5 if d > 0)
        med_vs_cap5 = sorted(rolling_vs_cap5)[len(rolling_vs_cap5) // 2]
        print(f"  CANDIDATE: {label}")
        print(f"     20Y {(r['cagr']-ref['cagr'])*100:+.2f}pp  1Y {(r['cagr_1y']-ref['cagr_1y'])*100:+.2f}pp  "
              f"Sharpe {r['sharpe']-ref['sharpe']:+.2f}  MaxDD {(r['mdd']-ref['mdd'])*100:+.2f}pp  "
              f"vs-CAP5 rolling {wins_cap5}/11  med {med_vs_cap5:+.2f}pp  worst {min(rolling_vs_cap5):+.2f}pp")
        n_cands += 1
if n_cands == 0:
    print("  None. CAP5 (cap=5%, top_n=5) wins on the joint frontier.")


# Heatmap: 20Y CAGR
print("\n## 20Y CAGR heatmap (rank-weighted hold-forever)")
print(f"  {'cap / top_n':>12s}  " + "  ".join(f"{tn:>6d}" for tn in TOPNS))
for cap in CAPS:
    row = f"  {'none' if cap is None else f'{int(cap*100)}%':>12s}"
    for tn in TOPNS:
        lbl = f"cap={'none' if cap is None else f'{int(cap*100)}%':>4s} top_n={tn}"
        row += f"  {results[lbl]['cagr']*100:+6.2f}"
    print(row)

print("\n## 1Y CAGR heatmap")
print(f"  {'cap / top_n':>12s}  " + "  ".join(f"{tn:>6d}" for tn in TOPNS))
for cap in CAPS:
    row = f"  {'none' if cap is None else f'{int(cap*100)}%':>12s}"
    for tn in TOPNS:
        lbl = f"cap={'none' if cap is None else f'{int(cap*100)}%':>4s} top_n={tn}"
        row += f"  {results[lbl]['cagr_1y']*100:+6.2f}"
    print(row)

print("\n## Rolling median-excess vs SPY heatmap (pp)")
print(f"  {'cap / top_n':>12s}  " + "  ".join(f"{tn:>6d}" for tn in TOPNS))
for cap in CAPS:
    row = f"  {'none' if cap is None else f'{int(cap*100)}%':>12s}"
    for tn in TOPNS:
        lbl = f"cap={'none' if cap is None else f'{int(cap*100)}%':>4s} top_n={tn}"
        row += f"  {results[lbl]['med_ex_spy']:+6.2f}"
    print(row)

print("\n## MaxDD heatmap")
print(f"  {'cap / top_n':>12s}  " + "  ".join(f"{tn:>6d}" for tn in TOPNS))
for cap in CAPS:
    row = f"  {'none' if cap is None else f'{int(cap*100)}%':>12s}"
    for tn in TOPNS:
        lbl = f"cap={'none' if cap is None else f'{int(cap*100)}%':>4s} top_n={tn}"
        row += f"  {results[lbl]['mdd']*100:+6.2f}"
    print(row)

print("\n## DONE")
