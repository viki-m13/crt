"""Step 16: survivorship / within-universe loser diagnostic.

Does hold-forever's CAGR come from avoiding sells during drawdowns that
recovered, or is it riding companies that would have been screened out
by a sell discipline? Two diagnostics:

  A. Per-pick PnL — for every buy in the shipped Top-5 strategy, what
     is its return from buy to end-of-spine, and what is the intra-hold
     max drawdown? Flag picks that drew down >40% and still ended ugly.
  B. Simulate a realistic hard stop: if any position drops >50% from
     buy, sell it. If hold-forever outperforms even with this stop,
     then "riding irrecoverable losers" isn't doing the work.
"""
import math, numpy as np
from collections import Counter
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY, StrategyConfig)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark SPY DCA B&H: {fmt_metrics(bm)}\n")

# ---- A. Per-pick PnL for the shipped Top-5 strategy ----
cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                     start_month_idx=start_m, entry_delay=1)
r = simulate(md, md.stocks, cfg)
end_idx = len(md.all_dates) - 1
records = []
sim = r  # keep a reference; `r` is reused as a loop variable below
for pos in sim.positions:
    tk = pos["tk"]
    buy_idx = pos["buy_idx"]
    buy_px = pos["buy_price"]
    shares = pos["shares"]
    # final mark-to-market
    end_px = md.prices[tk][end_idx]
    if not (math.isfinite(end_px) and end_px > 0):
        # walk back to last finite
        j = end_idx
        while j >= buy_idx and not (math.isfinite(md.prices[tk][j]) and md.prices[tk][j] > 0):
            j -= 1
        end_px = md.prices[tk][j] if j >= buy_idx else buy_px
    ret = (end_px / buy_px - 1.0) if buy_px > 0 else 0.0
    # intra-hold max drawdown on this position (price-path)
    worst = 0.0
    peak = buy_px
    for d in range(buy_idx, end_idx + 1):
        px = md.prices[tk][d]
        if not (math.isfinite(px) and px > 0):
            continue
        if px > peak: peak = px
        dd = (px - peak) / peak  # negative
        if dd < worst: worst = dd
    records.append({"tk": tk, "buy_idx": buy_idx, "buy_px": buy_px,
                    "end_px": end_px, "ret": ret, "worst_dd": worst,
                    "pnl": shares * (end_px - buy_px)})

records.sort(key=lambda x: x["pnl"])
tot_pnl = sum(r["pnl"] for r in records)

print("## Per-pick outcomes (Top-5 rank hold-forever, all 300 buys)")
n_loss = sum(1 for r in records if r["ret"] < 0)
n_ugly = sum(1 for r in records if r["ret"] < -0.20)
n_ugly50 = sum(1 for r in records if r["ret"] < -0.50)
print(f"  buys total:         {len(records)}")
print(f"  ending in loss:     {n_loss} ({n_loss/len(records)*100:.1f}%)")
print(f"  ending below -20%:  {n_ugly} ({n_ugly/len(records)*100:.1f}%)")
print(f"  ending below -50%:  {n_ugly50} ({n_ugly50/len(records)*100:.1f}%)")
print(f"  total PnL:          ${tot_pnl:,.0f}")

# Concentration: top 10 picks as share of total profit
records_by_pnl = sorted(records, key=lambda x: -x["pnl"])
top10_pnl = sum(r["pnl"] for r in records_by_pnl[:10])
print(f"  top 10 picks = ${top10_pnl:,.0f} ({top10_pnl/tot_pnl*100:.1f}% of PnL)")

print("\n## Biggest losers that never recovered (final return, buy date)")
for r in records[:10]:
    d = md.all_dates[r["buy_idx"]]
    print(f"  {r['tk']:6s} bought {d} ret {r['ret']*100:+7.1f}%  worstDD {r['worst_dd']*100:+6.1f}%  pnl ${r['pnl']:+8,.0f}")

print("\n## Biggest winners")
for r in records_by_pnl[:10]:
    d = md.all_dates[r["buy_idx"]]
    print(f"  {r['tk']:6s} bought {d} ret {r['ret']*100:+7.1f}%  worstDD {r['worst_dd']*100:+6.1f}%  pnl ${r['pnl']:+8,.0f}")

# Ticker-level concentration
print("\n## PnL by ticker (aggregated across all DCA buys of that ticker)")
by_tk = {}
for r in records:
    by_tk.setdefault(r["tk"], 0.0)
    by_tk[r["tk"]] += r["pnl"]
for tk, p in sorted(by_tk.items(), key=lambda x: -x[1])[:15]:
    cnt = sum(1 for r in records if r["tk"] == tk)
    print(f"  {tk:6s} n_buys={cnt:3d}  total PnL ${p:+10,.0f}")

# ---- B. Hard -50% stop ----
print("\n## Strategy with a -50% hard stop (sell any position down 50% from buy)")
for stop in [0.30, 0.40, 0.50, 0.60]:
    cfg2 = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                          start_month_idx=start_m, entry_delay=1,
                          dd_stop=stop)
    try:
        r2 = simulate(md, md.stocks, cfg2)
        m2 = compute_metrics(md, r2.equity, r2.total_invested)
        ex = (m2["cagr"] - bm["cagr"]) * 100
        print(f"  stop -{int(stop*100)}% | {fmt_metrics(m2)}  excess {ex:+.2f}pp")
    except TypeError as e:
        print(f"  stop -{int(stop*100)}% | config not supported: {e}")

# Reference
m_ref = compute_metrics(md, sim.equity, sim.total_invested)
ex = (m_ref["cagr"] - bm["cagr"]) * 100
print(f"  no stop  | {fmt_metrics(m_ref)}  excess {ex:+.2f}pp")
