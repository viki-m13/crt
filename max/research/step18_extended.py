"""Step 18: full validation battery on the extended history.

Loads the 20-year bt_series from bt_ext.parquet and reruns all the key
tests from steps 1, 13, 14, 17 against that much-longer record:

  A. Baseline: Top-1/3/5/10 rank hold-forever vs SPY DCA.
  B. Walk-forward: halves and quartiles over the entire extended spine.
  C. Sub-era slices: pre-GFC, GFC+recovery, 2010s, 2020-present.
  D. Jackknife by ticker (ticker-level robustness).
  E. Random-subset bootstrap (universe robustness).
  F. NVDA ablation.
  G. Transaction-cost sensitivity.
"""
import random, sys, os, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig, DCA_MONTHLY)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
print(f"Extended spine: {md.all_dates[0]} → {md.all_dates[-1]}  "
      f"N days={len(md.all_dates)}  N stocks={len(md.stocks)}  "
      f"Start month idx (>=3 tickers): {start_m}")
print(f"Start month: {md.all_dates[md.month_first_idx[start_m]]}\n")


def bench(start):
    b = simulate_benchmark(md, ["SPY"], 5000, start, entry_delay=1)
    return compute_metrics(md, b.equity, b.total_invested)


bm = bench(start_m)
print(f"SPY DCA B&H (benchmark): {fmt_metrics(bm)}\n")


def run_strat(universe, top_n, start=None):
    s = start if start is not None else start_m
    cfg = StrategyConfig(top_n=top_n, hold_days=5000, weighting="rank",
                         start_month_idx=s, entry_delay=1)
    r = simulate(md, universe, cfg)
    return compute_metrics(md, r.equity, r.total_invested), r


# ---- A. Baseline ----
print("## A. Baseline — Top-N rank hold-forever on extended history")
for n in [1, 3, 5, 10]:
    m, _ = run_strat(md.stocks, n)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  Top-{n:2d} rank, hold-forever | {fmt_metrics(m)}  excess {ex:+.2f}pp")


# ---- B. Walk-forward on the extended spine ----
print("\n## B. Walk-forward on extended spine")

def run_window(s_m, e_m, label):
    bw = simulate_benchmark(md, ["SPY"], 5000, s_m, entry_delay=1)
    bm_full = compute_metrics(md, bw.equity, bw.total_invested)
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                         start_month_idx=s_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    # Window metrics: cap equity comparison to [s_m, e_m]
    start_idx = md.month_first_idx[s_m]
    end_idx = md.month_first_idx[min(e_m, len(md.month_first_idx) - 1)]
    def win(eq, ctrib_per_month):
        n_months = min(e_m, len(md.month_first_idx) - 1) - s_m
        invested = ctrib_per_month * n_months
        if invested <= 0:
            return None
        start_val = eq[start_idx] if eq[start_idx] > 0 else 0
        final_val = eq[end_idx]
        total_in = start_val + invested
        if total_in <= 0:
            return None
        yrs = (end_idx - start_idx) / 252
        cagr = (final_val / total_in) ** (1 / yrs) - 1 if yrs > 0 else 0.0
        return {"cagr": cagr, "yrs": yrs, "final": final_val, "invested": total_in}
    bw_m = win(bw.equity, 1000)
    st_m = win(r.equity, 1000)
    if bw_m and st_m:
        ex = (st_m["cagr"] - bw_m["cagr"]) * 100
        print(f"  {label:14s} ({md.all_dates[start_idx][:10]}→{md.all_dates[end_idx][:10]}, {st_m['yrs']:.1f}y)  "
              f"SPY {bw_m['cagr']*100:+6.2f}%  Top-5 {st_m['cagr']*100:+6.2f}%  excess {ex:+.2f}pp")

total_months = len(md.month_first_idx)
usable = total_months - start_m
# Halves
h = usable // 2
run_window(start_m, start_m + h, "first half")
run_window(start_m + h, total_months - 1, "second half")
# Quartiles
q = usable // 4
run_window(start_m,       start_m + q,   "Q1")
run_window(start_m + q,   start_m + 2*q, "Q2")
run_window(start_m + 2*q, start_m + 3*q, "Q3")
run_window(start_m + 3*q, total_months-1,"Q4")


# ---- C. Named sub-eras (approximate) ----
print("\n## C. Named sub-eras (fixed boundaries)")

def month_idx_for(date_prefix):
    for m, di in enumerate(md.month_first_idx):
        if md.all_dates[di] >= date_prefix:
            return m
    return len(md.month_first_idx) - 1

eras = [
    ("pre-GFC       ", "2005-01-01", "2008-09-01"),
    ("GFC+recovery  ", "2008-09-01", "2013-01-01"),
    ("2010s bull    ", "2013-01-01", "2020-01-01"),
    ("COVID+present ", "2020-01-01", md.all_dates[-1]),
]
for label, start_date, end_date in eras:
    sm = month_idx_for(start_date)
    em = month_idx_for(end_date)
    if sm >= em or sm < start_m:
        sm = max(sm, start_m)
    if em <= sm + 6:
        print(f"  {label:14s} skipped (insufficient data before spine start)")
        continue
    run_window(sm, em, label)


# ---- D. Jackknife by ticker ----
print("\n## D. Jackknife — drop one ticker at a time")
base_m, _ = run_strat(md.stocks, 5)
base_ex = (base_m["cagr"] - bm["cagr"]) * 100
print(f"  Full universe baseline: CAGR {base_m['cagr']*100:+.2f}%  excess {base_ex:+.2f}pp")
excesses = []
for tk in md.stocks:
    m, _ = run_strat([t for t in md.stocks if t != tk], 5)
    if not m:
        continue
    ex = (m["cagr"] - bm["cagr"]) * 100
    excesses.append((tk, ex, m["cagr"] * 100))
excesses.sort(key=lambda x: x[1])
print("  Biggest CAGR loss when dropped:")
for tk, ex, cagr in excesses[:8]:
    print(f"    drop {tk:6s}  excess {ex:+6.2f}pp  CAGR {cagr:+6.2f}%")
print("  Drops that hurt LEAST (or helped):")
for tk, ex, cagr in excesses[-5:]:
    print(f"    drop {tk:6s}  excess {ex:+6.2f}pp  CAGR {cagr:+6.2f}%")
arr = np.array([e[1] for e in excesses])
print(f"  Distribution of excess CAGR (pp): min {arr.min():+.2f}  p10 {np.percentile(arr,10):+.2f}  "
      f"med {np.median(arr):+.2f}  p90 {np.percentile(arr,90):+.2f}  max {arr.max():+.2f}")
print(f"  n runs excess<0: {int((arr<0).sum())}/{len(arr)}  "
      f"excess<3pp: {int((arr<3).sum())}/{len(arr)}  "
      f"excess>5pp: {int((arr>5).sum())}/{len(arr)}")


# ---- E. Bootstrap ----
print("\n## E. Bootstrap — 200 runs on random 50-ticker subsets")
random.seed(42)
boot = []
for i in range(200):
    sub = random.sample(md.stocks, min(50, len(md.stocks)))
    m, _ = run_strat(sub, 5)
    if m:
        boot.append((m["cagr"] - bm["cagr"]) * 100)
arr = np.array(boot)
print(f"  Distribution of excess CAGR (pp): min {arr.min():+.2f}  p10 {np.percentile(arr,10):+.2f}  "
      f"med {np.median(arr):+.2f}  mean {arr.mean():+.2f}  p90 {np.percentile(arr,90):+.2f}  max {arr.max():+.2f}")
print(f"  n runs losing vs SPY: {int((arr<0).sum())}/{len(arr)}  "
      f"excess>3pp: {int((arr>3).sum())}/{len(arr)}  "
      f"excess>5pp: {int((arr>5).sum())}/{len(arr)}")


# ---- F. NVDA ablation ----
print("\n## F. NVDA ablation")
if "NVDA" in md.stocks:
    m, _ = run_strat([t for t in md.stocks if t != "NVDA"], 5)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  Drop NVDA: {fmt_metrics(m)}  excess {ex:+.2f}pp")
    top_winners = sorted(excesses[:5], key=lambda x: x[0])
    tks = [e[0] for e in excesses[:5]]
    m, _ = run_strat([t for t in md.stocks if t not in tks], 5)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  Drop top-5 jackknife contributors {tks}: {fmt_metrics(m)}  excess {ex:+.2f}pp")


# ---- G. Transaction cost sensitivity ----
print("\n## G. Transaction cost sensitivity (bps on entry notional)")

def run_with_tcost(bps):
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1)
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

for bps in [0, 5, 10, 25, 50]:
    m = run_with_tcost(bps)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  entry {bps:3d} bps | {fmt_metrics(m)}  excess {ex:+.2f}pp")
