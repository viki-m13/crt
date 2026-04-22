"""Step 26: full validation battery for CAP5R = CAP5 + rebound-63d gate.

CAP5R adds a "positive trailing 63-day return" filter to CAP5 picks. The
rationale is to let recent price action confirm the model's long-horizon
conviction before committing capital. Question: does that marginal rule
preserve CAP5's structural edge across every stress test, or does it
introduce a regime in which we'd clearly prefer vanilla CAP5?

Battery (mirrors step23 structure, side-by-side with CAP5):
  1. Headline 20Y + SPY
  2. Walk-forward: first 10Y vs second 10Y, both configs
  3. Rolling 10Y windows: 11 windows, both configs (win counts, medians)
  4. Jackknife: drop each of 96 tickers, recompute CAP5R vs SPY
  5. Bootstrap: 200 random 50-ticker rosters, CAP5R only (CAP5 is the
     reference baseline from step23 -- recorded for quick comparison)
  6. Transaction cost stress: 0/10/25/50/100 bps for CAP5R
  7. Head-to-head CAP5R vs CAP5 in the 11 rolling 10Y windows on
     CAGR, Sharpe, MaxDD
  8. Trailing 1Y for both configs (promotion-relevant diagnostic)
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


def win_sharpe(eq, md, from_i, to_i):
    """Sharpe computed over monthly returns falling within [from_i, to_i)."""
    rets = []
    mfi = md.month_first_idx
    for i in range(1, len(mfi)):
        prev_idx = mfi[i - 1]
        cur_idx = mfi[i]
        if prev_idx < from_i or cur_idx >= to_i:
            continue
        prev, cur = eq[prev_idx], eq[cur_idx]
        if prev > 0:
            rets.append(cur / prev - 1)
    if len(rets) < 2:
        return 0.0
    a = np.array(rets)
    s = a.std()
    return (a.mean() / s) * math.sqrt(12) if s > 0 else 0.0


CAP5 = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
            entry_delay=1)
CAP5R = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
             entry_delay=1, rebound_lookback_days=63, rebound_min_return=0.0)


# ==========================================================================
# 1. Headline 20Y
# ==========================================================================
print("## 1. Headline 20Y")
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
print(f"  SPY DCA    CAGR {bm['cagr']*100:+6.2f}%  MaxDD {bm['maxdd']*100:+6.2f}%  "
      f"Sharpe {bm['sharpe']:.2f}")

headline = {}
for label, k in [("CAP5 ", CAP5), ("CAP5R", CAP5R)]:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    print(f"  {label}      CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  Calmar {calmar:.2f}")
    headline[label.strip()] = m


# ==========================================================================
# 2. Walk-forward: 10Y first-half vs 10Y second-half
# ==========================================================================
print("\n## 2. Walk-forward: 10Y first-half vs 10Y second-half")
H1 = (0, TOTAL_M // 2)
H2 = (TOTAL_M // 2, TOTAL_M)

for lbl, (s_m, e_m) in [("First 10Y ", H1), ("Second 10Y", H2)]:
    nm = e_m - s_m
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, bd = win_cagr_mdd(b.equity, 1000 * nm, from_i, to_i)
    bs = win_sharpe(b.equity, md, from_i, to_i)
    print(f"  {lbl} ({md.all_dates[from_i][:7]} -> {md.all_dates[to_i-1][:7]})")
    print(f"    SPY DCA    CAGR {bc*100:+6.2f}%  MaxDD {bd*100:+6.2f}%  Sharpe {bs:.2f}")
    for label, k in [("CAP5 ", CAP5), ("CAP5R", CAP5R)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, dd = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        sh = win_sharpe(r.equity, md, from_i, to_i)
        ex = (c - bc) * 100
        print(f"    {label}     CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  "
              f"MaxDD {dd*100:+6.2f}%  Sharpe {sh:.2f}")


# ==========================================================================
# 3. Rolling 10Y windows -- both configs side-by-side
# ==========================================================================
print("\n## 3. Rolling 10Y windows (CAGR excess in pp)")
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12
print(f"  {len(windows)} overlapping 10Y windows")
print(f"  {'window':20s}  {'SPY':>7s}  {'CAP5':>9s}  {'ex':>5s}   "
      f"{'CAP5R':>9s}  {'ex':>5s}   {'Δ(R-5)':>7s}")

rolling = []  # (cagr_cap5, cagr_cap5r, mdd5, mddR, sharpe5, sharpeR, spy_cagr, label)
cap5r_beats_cap5_cagr = 0
cap5r_beats_cap5_sharpe = 0
cap5r_beats_cap5_mdd = 0  # lower mdd == better
cap5r_beats_spy = 0
cap5_beats_spy = 0
cap5r_excess = []
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    nm = e_m - s_m
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, _ = win_cagr_mdd(b.equity, 1000 * nm, from_i, to_i)
    rs = {}
    for name, k in [("cap5", CAP5), ("cap5r", CAP5R)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, dd = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        sh = win_sharpe(r.equity, md, from_i, to_i)
        rs[name] = (c, dd, sh)
    lbl = f"{md.all_dates[from_i][:7]}->{md.all_dates[to_i-1][:7]}"
    ex5 = (rs['cap5'][0] - bc) * 100
    exR = (rs['cap5r'][0] - bc) * 100
    d = exR - ex5
    rolling.append((rs['cap5'], rs['cap5r'], bc, lbl))
    cap5r_excess.append(exR)
    if rs['cap5r'][0] > rs['cap5'][0]: cap5r_beats_cap5_cagr += 1
    if rs['cap5r'][2] > rs['cap5'][2]: cap5r_beats_cap5_sharpe += 1
    if rs['cap5r'][1] < rs['cap5'][1]: cap5r_beats_cap5_mdd += 1
    if rs['cap5r'][0] > bc: cap5r_beats_spy += 1
    if rs['cap5'][0] > bc: cap5_beats_spy += 1
    print(f"  {lbl:20s}  {bc*100:+6.2f}%  "
          f"{rs['cap5'][0]*100:+6.2f}% {ex5:+5.1f}   "
          f"{rs['cap5r'][0]*100:+6.2f}% {exR:+5.1f}   {d:+6.2f}")
print(f"  CAP5R beats CAP5 on CAGR  : {cap5r_beats_cap5_cagr}/{len(windows)}")
print(f"  CAP5R beats CAP5 on Sharpe: {cap5r_beats_cap5_sharpe}/{len(windows)}")
print(f"  CAP5R beats CAP5 on MaxDD : {cap5r_beats_cap5_mdd}/{len(windows)} (lower is better)")
print(f"  CAP5R beats SPY           : {cap5r_beats_spy}/{len(windows)}")
print(f"  CAP5  beats SPY           : {cap5_beats_spy}/{len(windows)}")
cap5r_excess_sorted = sorted(cap5r_excess)
med_excess = cap5r_excess_sorted[len(cap5r_excess_sorted)//2]
print(f"  CAP5R median rolling excess vs SPY: {med_excess:+5.2f}pp  "
      f"min {cap5r_excess_sorted[0]:+5.2f}pp  max {cap5r_excess_sorted[-1]:+5.2f}pp")
neg_r = sum(1 for e in cap5r_excess if e < 0)
if neg_r > 0:
    print(f"  *** FLAG: {neg_r}/{len(windows)} rolling windows have NEGATIVE CAP5R excess vs SPY")
else:
    print(f"  CAP5R negative-excess windows: 0/{len(windows)}")


# ==========================================================================
# 4. Jackknife on CAP5R
# ==========================================================================
print("\n## 4. Jackknife on CAP5R (drop each of 96 tickers)")
drops = []
for drop_tk in sorted(md.stocks):
    if drop_tk == "SPY":
        continue
    keep = [t for t in md.stocks if t != drop_tk]
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5R)
    r = simulate(md, keep, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    drops.append((drop_tk, m['cagr'], ex))

drops.sort(key=lambda x: x[2])
neg = [d for d in drops if d[2] < 0]
print(f"  Drops that made CAP5R lose to SPY: {len(neg)}/{len(drops)}")
if neg:
    print(f"  *** FLAG: jackknife shows CAP5R loses to SPY under {len(neg)} drops ***")
    for tk, c, ex in neg:
        print(f"      drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")
print(f"  Worst 5 drops:")
for tk, c, ex in drops[:5]:
    print(f"    drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")
print(f"  Best 5 drops:")
for tk, c, ex in drops[-5:][::-1]:
    print(f"    drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")


# ==========================================================================
# 5. Bootstrap: 200 random 50-ticker rosters (CAP5R)
# ==========================================================================
print("\n## 5. Bootstrap CAP5R: 200 random 50-ticker rosters")
all_stocks = [s for s in md.stocks if s != "SPY"]
boot_ex = []
boot_neg = 0
for i in range(200):
    sub = random.sample(all_stocks, 50)
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5R)
    r = simulate(md, sub, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    boot_ex.append(ex)
    if ex < 0:
        boot_neg += 1

boot_ex.sort()
print(f"  Negative outcomes: {boot_neg}/200")
print(f"  Median excess    : {boot_ex[100]:+5.2f}pp")
print(f"  5th percentile   : {boot_ex[10]:+5.2f}pp")
print(f"  95th percentile  : {boot_ex[189]:+5.2f}pp")
print(f"  Min              : {boot_ex[0]:+5.2f}pp")
print(f"  Max              : {boot_ex[-1]:+5.2f}pp")
if boot_neg > 0:
    print(f"  *** FLAG: {boot_neg}/200 bootstrap rosters have NEGATIVE CAP5R excess vs SPY")


# ==========================================================================
# 6. Transaction cost stress (CAP5R)
# ==========================================================================
print("\n## 6. Transaction cost stress on CAP5R")
def run_with_tcost(bps, kwargs):
    cfg = StrategyConfig(start_month_idx=start_m, **kwargs)
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
                    cash += pos["shares"] * px
                    pos["sold"] = True
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
    m = run_with_tcost(bps, CAP5R)
    ex = (m['cagr'] - bm['cagr']) * 100
    print(f"  {bps:>3d} bps  CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}")


# ==========================================================================
# 7. Head-to-head summary over 11 rolling 10Y windows
# ==========================================================================
print("\n## 7. Head-to-head CAP5R vs CAP5 (rolling 10Y)")
print(f"  Window                CAGR5   CAGR R  Δcagr   Shp5   ShpR  Δshp   DD5    DDR   Δdd")
diffs_cagr, diffs_sh, diffs_dd = [], [], []
for (c5, c5r, bc, lbl) in rolling:
    dcagr = (c5r[0] - c5[0]) * 100
    dsh = c5r[2] - c5[2]
    ddd = (c5r[1] - c5[1]) * 100
    diffs_cagr.append(dcagr)
    diffs_sh.append(dsh)
    diffs_dd.append(ddd)
    print(f"  {lbl:20s}  {c5[0]*100:+6.2f} {c5r[0]*100:+6.2f}  {dcagr:+5.2f}  "
          f"{c5[2]:+5.2f} {c5r[2]:+5.2f}  {dsh:+5.2f}  "
          f"{c5[1]*100:+6.2f} {c5r[1]*100:+6.2f}  {ddd:+5.2f}")
diffs_cagr.sort(); diffs_sh.sort(); diffs_dd.sort()
print(f"  Median Δcagr : {diffs_cagr[len(diffs_cagr)//2]:+5.2f}pp  "
      f"(R>5 in {sum(1 for d in diffs_cagr if d>0)}/{len(diffs_cagr)})")
print(f"  Median Δsharpe: {diffs_sh[len(diffs_sh)//2]:+5.2f}    "
      f"(R>5 in {sum(1 for d in diffs_sh if d>0)}/{len(diffs_sh)})")
print(f"  Median Δmaxdd : {diffs_dd[len(diffs_dd)//2]:+5.2f}pp  "
      f"(R<5 in {sum(1 for d in diffs_dd if d<0)}/{len(diffs_dd)})  [lower MaxDD = better]")


# ==========================================================================
# 8. Trailing 1Y diagnostic
# ==========================================================================
print("\n## 8. Trailing 1Y")
s_m_1y = TOTAL_M - 12
from_i = md.month_first_idx[s_m_1y]
to_i = TO
b = simulate_benchmark(md, ["SPY"], 5000, s_m_1y, entry_delay=1)
bc1, bd1 = win_cagr_mdd(b.equity, 1000 * 12, from_i, to_i)
bs1 = win_sharpe(b.equity, md, from_i, to_i)
print(f"  Window: {md.all_dates[from_i]} -> {md.all_dates[-1]}")
print(f"  SPY DCA    CAGR {bc1*100:+6.2f}%  MaxDD {bd1*100:+6.2f}%  Sharpe {bs1:.2f}")
for label, k in [("CAP5 ", CAP5), ("CAP5R", CAP5R)]:
    cfg = StrategyConfig(start_month_idx=s_m_1y, **k)
    r = simulate(md, md.stocks, cfg)
    c, dd = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
    sh = win_sharpe(r.equity, md, from_i, to_i)
    ex = (c - bc1) * 100
    print(f"  {label}      CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {dd*100:+6.2f}%  Sharpe {sh:.2f}")
