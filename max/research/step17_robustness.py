"""Step 17: robustness battery — what the edge looks like under stress.

Four tests, all on the winning Top-5 rank hold-forever strategy:

  1. NVDA ablation: drop NVDA from the universe. NVDA drives 27% of PnL
     in the baseline, so this is the single biggest concentration test.
  2. Jackknife by ticker: drop each of the 97 stocks one at a time;
     record the excess-CAGR each run. If any drop causes the edge to
     collapse, we depend too much on that name.
  3. Random-subset bootstrap: 200 runs on a random 50-stock subset.
     Distribution of excess-CAGR tells us whether the edge is a pattern
     or a single lucky roster.
  4. Transaction-cost sensitivity: add per-trade cost (5 / 10 / 25 bps
     on entry) and see how the edge degrades.
"""
import math, random, numpy as np
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY, StrategyConfig)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark SPY DCA B&H: {fmt_metrics(bm)}\n")


def baseline(universe):
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, universe, cfg)
    return compute_metrics(md, r.equity, r.total_invested)


# Full baseline reference
base_full = baseline(md.stocks)
ex_full = (base_full["cagr"] - bm["cagr"]) * 100
print(f"Full universe (97 stocks) baseline: {fmt_metrics(base_full)}  excess {ex_full:+.2f}pp\n")


# ---- 1. NVDA ablation ----
print("## 1. NVDA ablation")
no_nvda = [t for t in md.stocks if t != "NVDA"]
m = baseline(no_nvda)
ex = (m["cagr"] - bm["cagr"]) * 100
print(f"  Drop NVDA: {fmt_metrics(m)}  excess {ex:+.2f}pp  (baseline {ex_full:+.2f}pp)")
# Also drop top-5 winners together (NVDA, NEM, AVGO, INTC, NFLX)
top_winners = ["NVDA", "NEM", "AVGO", "INTC", "NFLX"]
m = baseline([t for t in md.stocks if t not in top_winners])
ex = (m["cagr"] - bm["cagr"]) * 100
print(f"  Drop top-5 winners {top_winners}: {fmt_metrics(m)}  excess {ex:+.2f}pp")


# ---- 2. Jackknife ----
print("\n## 2. Jackknife: drop one ticker at a time")
excesses = []
for tk in md.stocks:
    u = [t for t in md.stocks if t != tk]
    m = baseline(u)
    if not m:
        continue
    ex = (m["cagr"] - bm["cagr"]) * 100
    excesses.append((tk, ex, m["cagr"] * 100))

excesses.sort(key=lambda x: x[1])
print(f"  Drops that hurt MOST (biggest CAGR loss when removed):")
for tk, ex, cagr in excesses[:8]:
    print(f"    drop {tk:6s}  excess {ex:+6.2f}pp  CAGR {cagr:+6.2f}%")
print(f"  Drops that hurt LEAST (or helped):")
for tk, ex, cagr in excesses[-5:]:
    print(f"    drop {tk:6s}  excess {ex:+6.2f}pp  CAGR {cagr:+6.2f}%")

arr = np.array([e[1] for e in excesses])
print(f"\n  Jackknife distribution of excess CAGR (pp):")
print(f"    min    {arr.min():+6.2f}")
print(f"    p10    {np.percentile(arr, 10):+6.2f}")
print(f"    p25    {np.percentile(arr, 25):+6.2f}")
print(f"    median {np.median(arr):+6.2f}")
print(f"    p75    {np.percentile(arr, 75):+6.2f}")
print(f"    p90    {np.percentile(arr, 90):+6.2f}")
print(f"    max    {arr.max():+6.2f}")
print(f"    n runs where excess < 0:   {int((arr < 0).sum())} / {len(arr)}")
print(f"    n runs where excess < 3pp: {int((arr < 3).sum())} / {len(arr)}")
print(f"    n runs where excess > 5pp: {int((arr > 5).sum())} / {len(arr)}")


# ---- 3. Random subset bootstrap ----
print("\n## 3. Bootstrap: 200 runs on random 50-stock subsets")
random.seed(42)
boot_excesses = []
for i in range(200):
    sub = random.sample(md.stocks, 50)
    m = baseline(sub)
    if not m:
        continue
    ex = (m["cagr"] - bm["cagr"]) * 100
    boot_excesses.append(ex)

arr = np.array(boot_excesses)
print(f"  Distribution of excess CAGR (pp) across {len(arr)} random rosters:")
print(f"    min    {arr.min():+6.2f}")
print(f"    p10    {np.percentile(arr, 10):+6.2f}")
print(f"    p25    {np.percentile(arr, 25):+6.2f}")
print(f"    median {np.median(arr):+6.2f}")
print(f"    mean   {arr.mean():+6.2f}")
print(f"    p75    {np.percentile(arr, 75):+6.2f}")
print(f"    p90    {np.percentile(arr, 90):+6.2f}")
print(f"    max    {arr.max():+6.2f}")
print(f"    n runs losing vs SPY:      {int((arr < 0).sum())} / {len(arr)}")
print(f"    n runs with excess > 3pp:  {int((arr > 3).sum())} / {len(arr)}")
print(f"    n runs with excess > 5pp:  {int((arr > 5).sum())} / {len(arr)}")


# ---- 4. Transaction-cost sensitivity ----
print("\n## 4. Transaction cost sensitivity (bps charged on entry notional)")

def baseline_with_tcost(universe, bps):
    """Run the strategy and deduct `bps` of the entry notional as a cost
    at the time of each buy. For hold-forever there are no exit costs."""
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, universe, cfg)
    cost_rate = bps / 10000.0
    # Scale down each position's shares by (1 - cost_rate) to emulate that
    # fraction of the entry being eaten by cost/slippage.
    # Equivalently: same shares but we deduct cost_rate * cost from equity
    # for each position, ongoing. Simpler: rebuild equity.
    n = len(md.all_dates)
    equity = np.zeros(n)
    cash = 0.0
    # Apply entry slippage: reduce shares bought by (1 - cost_rate).
    for pos in r.positions:
        pos["shares"] *= (1 - cost_rate)
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
    m = baseline_with_tcost(md.stocks, bps)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  entry {bps:3d} bps | {fmt_metrics(m)}  excess {ex:+.2f}pp")
