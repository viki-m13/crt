"""Step 24: show the CAP5 strategy's current-month picks and recent picks.

Goal: the user asked "what stocks to buy each month that will rebound and are
currently undervalued". This script shows:
  1. This month's picks under CAP5 strategy
  2. Single-stock pick (Top-1)
  3. Last 6 months of picks and what they did
  4. Current ticker concentration (are we near the 5% cap on anything?)
"""
import math, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from bt_core import StrategyConfig, simulate
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)

# Run the full strategy through today, so we can inspect cumulative state.
CAP5 = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000, weighting="rank",
            entry_delay=1)
cfg = StrategyConfig(start_month_idx=start_m, **CAP5)
r = simulate(md, md.stocks, cfg)

# Group positions by month (buy month)
by_month = defaultdict(list)
for pos in r.positions:
    buy_date = md.all_dates[pos["buy_idx"]]
    month_key = buy_date[:7]
    by_month[month_key].append(pos)

months = sorted(by_month.keys())
print(f"Total positions opened: {len(r.positions)} across {len(months)} months\n")

# Last 6 months
print("## Last 6 months of CAP5 picks")
print(f"  {'month':10s}  {'picks':60s}")
for m in months[-6:]:
    picks = by_month[m]
    picks_str = ", ".join(f"{p['tk']}({p.get('weight', 0)*100:.0f}%)" for p in picks)
    print(f"  {m:10s}  {picks_str}")

# The very last month = most recent picks
last_m = months[-1]
print(f"\n## Most recent picks ({last_m})")
last_picks = by_month[last_m]
last_picks_sorted = sorted(last_picks, key=lambda p: -p.get("weight", 0))
print(f"  {'tk':6s}  {'weight':>7s}  {'buy_px':>8s}  {'cur_px':>8s}  {'ret':>7s}")
end_day = len(md.all_dates) - 1
for p in last_picks_sorted:
    buy_px = md.prices[p['tk']][p['buy_idx']]
    cur_px = md.prices[p['tk']][end_day]
    ret = (cur_px / buy_px - 1) if (math.isfinite(buy_px) and math.isfinite(cur_px) and buy_px > 0) else float("nan")
    w = p.get("weight", 0)
    print(f"  {p['tk']:6s}  {w*100:>6.1f}%  ${buy_px:>7.2f}  ${cur_px:>7.2f}  {ret*100:>+6.1f}%")

# Current concentration: who's at highest % of total cost basis?
print(f"\n## Current concentration (cumulative cost basis per ticker)")
total_cost = sum(p["cost"] for p in r.positions)
by_tk_cost = defaultdict(float)
by_tk_n = defaultdict(int)
for p in r.positions:
    by_tk_cost[p["tk"]] += p["cost"]
    by_tk_n[p["tk"]] += 1
ranked = sorted(by_tk_cost.items(), key=lambda x: -x[1])
print(f"  {'tk':6s}  {'n_buys':>6s}  {'cost':>9s}  {'% total':>7s}  {'cur_val':>9s}  {'pnl%':>6s}")
for tk, c in ranked[:15]:
    cur_val = 0.0
    for p in r.positions:
        if p["tk"] == tk and not p.get("sold", False):
            px = md.prices[tk][end_day]
            if math.isfinite(px): cur_val += p["shares"] * px
    rr = (cur_val - c) / c * 100 if c > 0 else 0
    pct = c / total_cost * 100
    print(f"  {tk:6s}  {by_tk_n[tk]:>6d}  ${c:>7,.0f}  {pct:>6.2f}%  ${cur_val:>7,.0f}  {rr:>+5.1f}%")

# Top-1 "single stock" picks — what would Top-1 version have picked?
print("\n## Top-1 CAP5 single-stock picks: last 6 months")
cfg1 = StrategyConfig(start_month_idx=start_m, top_n=1, max_ticker_frac=0.05,
                      hold_days=5000, weighting="rank", entry_delay=1)
r1 = simulate(md, md.stocks, cfg1)
by_m1 = defaultdict(list)
for p in r1.positions:
    by_m1[md.all_dates[p["buy_idx"]][:7]].append(p)
for m in sorted(by_m1.keys())[-6:]:
    p = by_m1[m][0] if by_m1[m] else None
    if not p: continue
    buy_px = md.prices[p['tk']][p['buy_idx']]
    cur_px = md.prices[p['tk']][end_day]
    ret = (cur_px / buy_px - 1) if (math.isfinite(buy_px) and math.isfinite(cur_px) and buy_px > 0) else float("nan")
    print(f"  {m:10s}  {p['tk']:6s}  buy ${buy_px:>7.2f}  now ${cur_px:>7.2f}  ret {ret*100:+6.1f}%")
