"""Step 28: full validation of CAP5RB (= CAP5 + regime-conditional rebound gate).

CAP5RB adds a 63d rebound gate active ONLY when SPY >= 200DMA.  In bear
regimes the gate is disabled so V-shape recoveries (e.g. GFC, COVID) are not
missed.  Baseline comparison is CAP5.

Validation battery (mirror of step23):
  1. Headline 20Y  (CAP5RB + CAP5 + SPY)
  2. Walk-forward 10Y H1 vs 10Y H2  (both)
  3. All 11 rolling 10Y windows, excess pp vs SPY (both)
  4. Jackknife: drop each of 96 tickers, rerun CAP5RB, count neg-vs-SPY
  5. Bootstrap: 200 random 50-ticker rosters (CAP5RB only)
  6. Transaction cost stress: 0/10/25/50/100 bps (CAP5RB)
  7. Head-to-head CAP5RB vs CAP5 across all 11 rolling 10Y windows:
     CAGR / Sharpe / MaxDD wins
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


def win_metrics(eq, invested, from_i, to_i):
    """CAGR, MaxDD, Sharpe within a window."""
    yrs = (to_i - from_i) / 252
    if invested <= 0 or yrs <= 0:
        return None, None, None
    final = eq[to_i - 1]
    if final <= 0:
        return -1.0, 1.0, 0.0
    cagr = (final / invested) ** (1 / yrs) - 1
    peak, mdd = 0.0, 0.0
    rets = []
    prev = None
    for i in range(from_i, to_i):
        v = eq[i]
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
        if prev is not None and prev > 0 and math.isfinite(v) and math.isfinite(prev):
            rets.append(v / prev - 1)
        prev = v
    if len(rets) > 1:
        a = np.asarray(rets)
        mu = float(np.mean(a))
        sd = float(np.std(a, ddof=1))
        sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    return cagr, mdd, sharpe


CAP5 = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
            entry_delay=1)
CAP5RB = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
              entry_delay=1, rebound_lookback_days=63, rebound_min_return=0.0,
              rebound_only_in_bull=True)


# ============================================================================
print("## 1. Headline 20Y")
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
print(f"  SPY DCA      CAGR {bm['cagr']*100:+6.2f}%                MaxDD {bm['maxdd']*100:+6.2f}%  Sharpe {bm['sharpe']:.2f}")

full_results = {}
for label, k in [("CAP5        ", CAP5), ("CAP5RB      ", CAP5RB)]:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    full_results[label.strip()] = m
    print(f"  {label} CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  Calmar {calmar:.2f}")


# ============================================================================
print("\n## 2. Walk-forward: 10Y first-half vs 10Y second-half")
H1 = (0, TOTAL_M // 2)
H2 = (TOTAL_M // 2, TOTAL_M)

for lbl, (s_m, e_m) in [("First 10Y ", H1), ("Second 10Y", H2)]:
    nm = e_m - s_m
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, bd, bs = win_metrics(b.equity, 1000 * nm, from_i, to_i)
    print(f"  {lbl} ({md.all_dates[from_i][:7]} -> {md.all_dates[to_i-1][:7]})")
    print(f"    SPY DCA      CAGR {bc*100:+6.2f}%                MaxDD {bd*100:+6.2f}%  Sharpe {bs:.2f}")
    for label, k in [("CAP5    ", CAP5), ("CAP5RB  ", CAP5RB)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, d, s = win_metrics(r.equity, r.total_invested, from_i, to_i)
        ex = (c - bc) * 100
        print(f"    {label}     CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%  Sharpe {s:.2f}")


# ============================================================================
print("\n## 3. Rolling 10Y windows (CAGR excess in pp)")
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12
print(f"  {len(windows)} overlapping 10Y windows")
print(f"  {'window':20s}  {'SPY':>7s}  {'CAP5':>17s}  {'CAP5RB':>17s}   {'RB-CAP5':>7s}")
rb_beats_cap5_cagr = 0
rb_beats_spy = 0
cap5_beats_spy = 0
rb_neg_vs_spy_windows = []
rolling_rows = []
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    nm = e_m - s_m
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, bd, bs = win_metrics(b.equity, 1000 * nm, from_i, to_i)
    row = {"lbl": f"{md.all_dates[from_i][:7]}->{md.all_dates[to_i-1][:7]}",
           "spy": (bc, bd, bs)}
    for name, k in [("cap5", CAP5), ("rb", CAP5RB)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, d, s = win_metrics(r.equity, r.total_invested, from_i, to_i)
        row[name] = (c, d, s)
    rolling_rows.append(row)
    ex_c = (row['cap5'][0] - bc) * 100
    ex_r = (row['rb'][0] - bc) * 100
    delta = ex_r - ex_c
    if row['rb'][0] > row['cap5'][0]: rb_beats_cap5_cagr += 1
    if row['rb'][0] > bc: rb_beats_spy += 1
    else: rb_neg_vs_spy_windows.append((row['lbl'], ex_r))
    if row['cap5'][0] > bc: cap5_beats_spy += 1
    print(f"  {row['lbl']:20s}  {bc*100:+6.2f}%  "
          f"{row['cap5'][0]*100:+6.2f}% ({ex_c:+5.2f}pp)  "
          f"{row['rb'][0]*100:+6.2f}% ({ex_r:+5.2f}pp)   {delta:+6.2f}")
print(f"  CAP5RB beats CAP5 (CAGR): {rb_beats_cap5_cagr}/{len(windows)}")
print(f"  CAP5RB beats SPY        : {rb_beats_spy}/{len(windows)}")
print(f"  CAP5   beats SPY        : {cap5_beats_spy}/{len(windows)}")
if rb_neg_vs_spy_windows:
    print(f"  !! CAP5RB negative-excess windows: {rb_neg_vs_spy_windows}")


# ============================================================================
print("\n## 4. Jackknife (drop each of 96 tickers, run CAP5RB)")
drops = []
for drop_tk in sorted(md.stocks):
    if drop_tk == "SPY": continue
    keep = [t for t in md.stocks if t != drop_tk]
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5RB)
    r = simulate(md, keep, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    drops.append((drop_tk, m['cagr'], ex))

drops.sort(key=lambda x: x[2])
neg = [d for d in drops if d[2] < 0]
print(f"  Drops that made CAP5RB lose to SPY: {len(neg)}/{len(drops)}")
print(f"  Worst 5 drops:")
for tk, c, ex in drops[:5]:
    print(f"    drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")
print(f"  Best 5 drops:")
for tk, c, ex in drops[-5:][::-1]:
    print(f"    drop {tk:6s} -> CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)")
if neg:
    print(f"  !! FLAGGED sign-flipping drops: {[d[0] for d in neg]}")


# ============================================================================
print("\n## 5. Bootstrap: 200 random 50-ticker rosters (CAP5RB)")
all_stocks = [s for s in md.stocks if s != "SPY"]
boot_ex = []
boot_neg = 0
for i in range(200):
    sub = random.sample(all_stocks, 50)
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5RB)
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


# ============================================================================
print("\n## 6. Transaction cost stress (CAP5RB)")
def run_with_tcost(bps):
    cfg = StrategyConfig(start_month_idx=start_m, **CAP5RB)
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


# ============================================================================
print("\n## 7. Head-to-head CAP5RB vs CAP5 across 11 rolling 10Y windows")
cagr_wins = 0
sharpe_wins = 0
mdd_wins = 0  # lower is better
ties = {"cagr": 0, "sharpe": 0, "mdd": 0}
print(f"  {'window':20s}  {'CAGR c5':>8s} {'CAGR rb':>8s}  "
      f"{'Shp c5':>6s} {'Shp rb':>6s}  {'DD c5':>7s} {'DD rb':>7s}  winner")
for row in rolling_rows:
    c5c, c5d, c5s = row['cap5']
    rbc, rbd, rbs = row['rb']
    # CAGR win
    if rbc > c5c: cagr_wins += 1; w_c = "RB"
    elif rbc < c5c: w_c = "C5"
    else: ties["cagr"] += 1; w_c = "="
    # Sharpe win
    if rbs > c5s: sharpe_wins += 1; w_s = "RB"
    elif rbs < c5s: w_s = "C5"
    else: ties["sharpe"] += 1; w_s = "="
    # MaxDD win (lower is better)
    if rbd < c5d: mdd_wins += 1; w_d = "RB"
    elif rbd > c5d: w_d = "C5"
    else: ties["mdd"] += 1; w_d = "="
    print(f"  {row['lbl']:20s}  {c5c*100:+7.2f}% {rbc*100:+7.2f}%  "
          f"{c5s:+6.2f} {rbs:+6.2f}  {c5d*100:+6.2f}% {rbd*100:+6.2f}%  "
          f"{w_c}/{w_s}/{w_d}")
N = len(rolling_rows)
print(f"  CAP5RB CAGR wins  : {cagr_wins}/{N}  (ties {ties['cagr']})")
print(f"  CAP5RB Sharpe wins: {sharpe_wins}/{N}  (ties {ties['sharpe']})")
print(f"  CAP5RB MaxDD wins : {mdd_wins}/{N}  (ties {ties['mdd']})  (lower is better)")


# ============================================================================
print("\n## 8. Trailing-period snapshot (headline frames)")
# Trailing 1Y
def trailing_window(months):
    s_m = max(0, TOTAL_M - months)
    e_m = TOTAL_M
    from_i = md.month_first_idx[s_m]; to_i = TO
    b = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bc, bd, bs = win_metrics(b.equity, 1000 * (e_m - s_m), from_i, to_i)
    print(f"  Trailing {months}M  ({md.all_dates[from_i][:7]} -> {md.all_dates[to_i-1][:7]})")
    print(f"    SPY DCA      CAGR {bc*100:+6.2f}%                MaxDD {bd*100:+6.2f}%")
    for label, k in [("CAP5    ", CAP5), ("CAP5RB  ", CAP5RB)]:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, d, s = win_metrics(r.equity, r.total_invested, from_i, to_i)
        ex = (c - bc) * 100
        print(f"    {label}     CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")

for mo in [12, 60, 120]:
    trailing_window(mo)

print("\n## DONE")
