"""Step 23: final validation of the CAP5 strategy (Top-5 rank + 5% concentration cap).

This is the only gate from step20/21 that strictly improved on baseline across
ALL windows (full 20Y, trailing 1Y, 5Y). It also lowered MaxDD.

Validation battery:
  1. Headline 20Y
  2. First-half vs second-half (10Y walk-forward, step18-style)
  3. Rolling 10Y windows
  4. Jackknife: drop each of the 96 tickers, recompute
  5. Bootstrap: 200 random 50-ticker subsets
  6. Transaction cost stress: 10/25/50/100 bps
  7. Compare head-to-head vs baseline across all windows
"""
import math, sys, os, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     StrategyConfig)
from bt_core_ext import load_and_prep_ext

random.seed(42)

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
TO = len(md.all_dates)


def win_cagr_mdd(eq, invested, from_i, to_i):
    yrs = (to_i - from_i) / 252
    if invested <= 0 or yrs <= 0:
        return None, None
    final = eq[to_i - 1]
    if final <= 0: return -1.0, 1.0
    cagr = (final / invested) ** (1 / yrs) - 1
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak: peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd: mdd = dd
    return cagr, mdd


CAP5 = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
            entry_delay=1)
BASE = dict(top_n=5, hold_days=5000, weighting="rank", entry_delay=1)


print("## 1. Headline 20Y")
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
print(f"  SPY DCA   CAGR {bm['cagr']*100:+6.2f}%  MaxDD {bm['maxdd']*100:+6.2f}%  Sharpe {bm['sharpe']:.2f}")

for label, k in [("Baseline T5 rank", BASE), ("CAP5 T5 rank   ", CAP5)]:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    print(f"  {label}  CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  Calmar {calmar:.2f}")


print("\n## 2. Walk-forward: 10Y first-half vs 10Y second-half")
H1 = (0, TOTAL_M // 2)
H2 = (TOTAL_M // 2, TOTAL_M)

for lbl, (s_m, e_m) in [("First 10Y ", H1), ("Second 10Y", H2)]:
    nm = e_m - s_m
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, bd = win_cagr_mdd(b.equity, 1000 * nm, from_i, to_i)
    print(f"  {lbl} ({md.all_dates[from_i][:7]} -> {md.all_dates[to_i-1][:7]})")
    print(f"    SPY DCA    CAGR {bc*100:+6.2f}%  MaxDD {bd*100:+6.2f}%")
    for label, k in [("Baseline T5", BASE), ("CAP5 T5    ", CAP5)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, d = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        ex = (c - bc) * 100
        print(f"    {label} CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")


print("\n## 3. Rolling 10Y windows (CAGR excess in pp)")
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12
print(f"  {len(windows)} overlapping 10Y windows")
print(f"  {'window':20s}  {'SPY':>7s}  {'Base':>9s}  {'CAP5':>9s}  {'Δ vs base':>9s}")
cap5_beats_base = 0
cap5_beats_spy = 0
base_beats_spy = 0
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    nm = e_m - s_m
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, _ = win_cagr_mdd(b.equity, 1000 * nm, from_i, to_i)
    rs = {}
    for name, k in [("base", BASE), ("cap5", CAP5)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, _ = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        rs[name] = c
    lbl = f"{md.all_dates[from_i][:7]}->{md.all_dates[to_i-1][:7]}"
    exb = (rs['base'] - bc) * 100
    exc = (rs['cap5'] - bc) * 100
    d = exc - exb
    if rs['cap5'] > rs['base']: cap5_beats_base += 1
    if rs['cap5'] > bc: cap5_beats_spy += 1
    if rs['base'] > bc: base_beats_spy += 1
    print(f"  {lbl:20s}  {bc*100:+6.2f}%  {rs['base']*100:+6.2f}% {exb:+5.1f}  "
          f"{rs['cap5']*100:+6.2f}% {exc:+5.1f}  {d:+5.2f}")
print(f"  CAP5 beats baseline: {cap5_beats_base}/{len(windows)}")
print(f"  CAP5 beats SPY     : {cap5_beats_spy}/{len(windows)}")
print(f"  Base beats SPY     : {base_beats_spy}/{len(windows)}")


print("\n## 4. Jackknife (drop each of 96 tickers)")
drops = []
for drop_tk in sorted(md.stocks):
    if drop_tk == "SPY": continue
    keep = [t for t in md.stocks if t != drop_tk]
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5)
    r = simulate(md, keep, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    drops.append((drop_tk, m['cagr'], ex))

drops.sort(key=lambda x: x[2])
neg = [d for d in drops if d[2] < 0]
print(f"  Drops that made CAP5 lose to SPY: {len(neg)}/{len(drops)}")
print(f"  Worst 5 drops:")
for tk, c, ex in drops[:5]:
    print(f"    drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")
print(f"  Best 5 drops:")
for tk, c, ex in drops[-5:][::-1]:
    print(f"    drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")


print("\n## 5. Bootstrap: 200 random 50-ticker rosters")
all_stocks = [s for s in md.stocks if s != "SPY"]
boot_ex = []
boot_neg = 0
for i in range(200):
    sub = random.sample(all_stocks, 50)
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5)
    r = simulate(md, sub, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    boot_ex.append(ex)
    if ex < 0: boot_neg += 1

boot_ex.sort()
print(f"  Negative outcomes: {boot_neg}/200")
print(f"  Median excess    : {boot_ex[100]:+5.2f}pp")
print(f"  5th percentile   : {boot_ex[10]:+5.2f}pp")
print(f"  95th percentile  : {boot_ex[189]:+5.2f}pp")
print(f"  Min              : {boot_ex[0]:+5.2f}pp")
print(f"  Max              : {boot_ex[-1]:+5.2f}pp")


print("\n## 6. Transaction cost stress")
def run_with_tcost(bps):
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5)
    r = simulate(md, md.stocks, cfg)
    rate = bps / 10000.0
    for pos in r.positions:
        pos["shares"] *= (1 - rate)
    n = len(md.all_dates)
    equity = np.zeros(n); cash = 0.0
    for d in range(n):
        for pos in r.positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            if d >= pos["sell_idx"]:
                px = md.prices[pos["tk"]][d]
                if math.isfinite(px) and px > 0:
                    cash += pos["shares"] * px; pos["sold"] = True
        open_val = 0.0
        for pos in r.positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    return compute_metrics(md, equity, r.total_invested)

for bps in [0, 10, 25, 50, 100]:
    m = run_with_tcost(bps)
    ex = (m['cagr'] - bm['cagr']) * 100
    print(f"  {bps:>3d} bps  CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {m['maxdd']*100:+6.2f}%")
