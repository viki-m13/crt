"""Step 2: honest Top-N × hold grid — no per-pick horizon look-ahead."""
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark (SPY DCA B&H): {fmt_metrics(bm)}")
print()
print("## Top-N × hold grid (equal weight, next-day-open entry)")
best = None
for n in [1, 2, 3, 5, 10]:
    for hd in [60, 126, 252, 504, 756, 1260, 5000]:
        cfg = StrategyConfig(top_n=n, hold_days=hd, weighting="equal", start_month_idx=start_m,
                             entry_delay=1)
        r = simulate(md, md.stocks, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        mark = " ★" if m["cagr"] > bm["cagr"] else ""
        print(f"  Top-{n:2d} EW {hd:5d}d | {fmt_metrics(m)}{mark}")
        if best is None or m["cagr"] > best[0]["cagr"]:
            best = (m, cfg)
print()
print(f"=> CAGR-best: Top-{best[1].top_n} EW {best[1].hold_days}d | {fmt_metrics(best[0])}")
print(f"   Excess CAGR vs SPY DCA B&H: {(best[0]['cagr']-bm['cagr'])*100:+.2f}pp")
