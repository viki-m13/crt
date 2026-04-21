"""Step 19: trailing 12-month diagnostic for the Max Top-5 strategy.

Questions:
  1. Is the 1Y window actually bad? How bad vs SPY?
  2. Which picks drove the underperformance? Per-pick contribution.
  3. Which tickers were selected most often? Are they momentum names or
     beaten-down names?
  4. What did the scanner SAY about the losers at buy time (score / prob /
     edge / conviction) vs what happened?
"""
import math, sys, os
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, _ = load_and_prep_ext()
total_m = len(md.month_first_idx)
start_m = total_m - 12  # trailing 12 months
end_m = total_m

from_idx = md.month_first_idx[start_m]
to_idx = md.all_dates.__len__()
print(f"Window: {md.all_dates[from_idx]} -> {md.all_dates[-1]}  "
      f"({(to_idx - from_idx)/252:.2f}y, {end_m - start_m} months)\n")


def window_metrics(eq, total_invested, from_idx, to_idx):
    """CAGR/return over [from_idx, to_idx)."""
    if total_invested <= 0:
        return None
    final = eq[to_idx - 1]
    yrs = (to_idx - from_idx) / 252
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    tr = (final - total_invested) / total_invested
    # MaxDD within window
    peak, mdd = 0.0, 0.0
    for i in range(from_idx, to_idx):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    return dict(cagr=cagr, tr=tr, final=final, invested=total_invested, mdd=mdd)


# ---- 1. Overall window numbers ----
print("## 1. Trailing 1Y overall")
bench = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = window_metrics(bench.equity, 1000 * (end_m - start_m), from_idx, to_idx)
print(f"  SPY DCA    CAGR {bm['cagr']*100:+6.2f}%  TR {bm['tr']*100:+6.2f}%  "
      f"MaxDD {bm['mdd']*100:+6.2f}%  Invested ${bm['invested']:,.0f}  Final ${bm['final']:,.0f}")

for n in [1, 3, 5, 10]:
    cfg = StrategyConfig(top_n=n, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = window_metrics(r.equity, 1000 * (end_m - start_m), from_idx, to_idx)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  Top-{n:2d}      CAGR {m['cagr']*100:+6.2f}%  TR {m['tr']*100:+6.2f}%  "
          f"MaxDD {m['mdd']*100:+6.2f}%  Final ${m['final']:,.0f}  excess {ex:+.2f}pp")


# ---- 2. Per-pick breakdown for Top-5 ----
print("\n## 2. Top-5 picks in the trailing 1Y window — per-position P&L")
cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                     start_month_idx=start_m, entry_delay=1)
r = simulate(md, md.stocks, cfg)

# `r.positions` is the ordered list of positions opened during the sim.
# Compute per-position current value at the final day and return.
end_day = to_idx - 1
rows = []
for pos in r.positions:
    if pos["buy_idx"] < from_idx:
        continue  # defensive; shouldn't happen given start_m
    tk = pos["tk"]
    buy_px = md.prices[tk][pos["buy_idx"]]
    cur_px = md.prices[tk][end_day]
    if not (math.isfinite(buy_px) and math.isfinite(cur_px) and buy_px > 0):
        continue
    ret = cur_px / buy_px - 1
    value_now = pos["shares"] * cur_px
    rows.append({
        "tk": tk,
        "buy_date": md.all_dates[pos["buy_idx"]],
        "buy_px": buy_px,
        "cur_px": cur_px,
        "cost": pos["shares"] * buy_px,  # actual cost
        "value": value_now,
        "ret": ret,
        "alloc_weight": pos.get("weight", None),
    })

print(f"  Total positions opened: {len(rows)}")
total_cost = sum(r["cost"] for r in rows)
total_value = sum(r["value"] for r in rows)
total_pnl = total_value - total_cost
print(f"  Total invested ${total_cost:,.0f}  current value ${total_value:,.0f}  "
      f"PnL ${total_pnl:,.0f} ({total_pnl/total_cost*100:+.2f}%)")

# Biggest losers (by $ PnL, not % — concentrates on what actually cost us)
rows.sort(key=lambda r: r["value"] - r["cost"])
print("\n  10 worst positions by $ PnL:")
print(f"    {'tk':6s} {'buy_date':12s} {'buy_px':>9s} {'cur_px':>9s} "
      f"{'ret':>8s} {'cost':>9s} {'value':>9s} {'$PnL':>9s}")
for r_ in rows[:10]:
    pnl = r_["value"] - r_["cost"]
    print(f"    {r_['tk']:6s} {r_['buy_date']:12s} {r_['buy_px']:>9.2f} {r_['cur_px']:>9.2f} "
          f"{r_['ret']*100:>7.1f}% ${r_['cost']:>7.0f} ${r_['value']:>7.0f} ${pnl:>+7.0f}")

print("\n  10 best positions by $ PnL:")
for r_ in rows[-10:][::-1]:
    pnl = r_["value"] - r_["cost"]
    print(f"    {r_['tk']:6s} {r_['buy_date']:12s} {r_['buy_px']:>9.2f} {r_['cur_px']:>9.2f} "
          f"{r_['ret']*100:>7.1f}% ${r_['cost']:>7.0f} ${r_['value']:>7.0f} ${pnl:>+7.0f}")


# ---- 3. Pick frequency and by-ticker contribution ----
print("\n## 3. By-ticker PnL aggregated across the 12-month window")
by_tk = defaultdict(lambda: {"n": 0, "cost": 0.0, "value": 0.0})
for r_ in rows:
    by_tk[r_["tk"]]["n"] += 1
    by_tk[r_["tk"]]["cost"] += r_["cost"]
    by_tk[r_["tk"]]["value"] += r_["value"]

ranked = []
for tk, d in by_tk.items():
    pnl = d["value"] - d["cost"]
    rr = pnl / d["cost"] if d["cost"] > 0 else 0.0
    ranked.append((tk, d["n"], d["cost"], d["value"], pnl, rr))
ranked.sort(key=lambda x: x[4])

print(f"  {'tk':6s} {'n':>3s} {'cost':>9s} {'value':>9s} {'$PnL':>9s} {'ret':>7s}")
print("  --- worst 10 contributors ---")
for tk, n, c, v, pnl, rr in ranked[:10]:
    print(f"  {tk:6s} {n:>3d} ${c:>7.0f} ${v:>7.0f} ${pnl:>+7.0f} {rr*100:>+6.1f}%")
print("  --- best 10 contributors ---")
for tk, n, c, v, pnl, rr in ranked[-10:][::-1]:
    print(f"  {tk:6s} {n:>3d} ${c:>7.0f} ${v:>7.0f} ${pnl:>+7.0f} {rr*100:>+6.1f}%")


# ---- 4. What was the scanner SAYING about the worst picks at buy time? ----
print("\n## 4. Scanner score at buy time for worst picks")
print("     (the scoring input that triggered the buy)")
worst = rows[:10]
print(f"  {'tk':6s} {'buy_date':12s} {'score':>7s} {'ret':>7s}")
for r_ in worst:
    tk = r_["tk"]
    buy_idx = md.date_idx.get(r_["buy_date"])
    if buy_idx is None:
        continue
    score = md.finals[tk][buy_idx]
    print(f"  {tk:6s} {r_['buy_date']:12s} {score:>7.2f} {r_['ret']*100:>+6.1f}%")


# ---- 5. Monthly winner counts ----
print("\n## 5. Month-by-month pick mix and hit rate")
print(f"  {'date':10s}  {'picks':<50s}  {'avg_ret':>8s}")
monthly = defaultdict(list)
for r_ in rows:
    monthly[r_["buy_date"]].append(r_)
for d in sorted(monthly.keys()):
    mrows = monthly[d]
    picks_str = ",".join(f"{m['tk']}({m['ret']*100:+.0f}%)" for m in mrows)
    avg = sum(m["ret"] for m in mrows) / len(mrows)
    print(f"  {d:10s}  {picks_str:<50s}  {avg*100:+7.1f}%")
