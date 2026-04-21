"""Step 3: weighting schemes for Top-N, hold-forever."""
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()
print("## Weighting schemes (hold=forever)")
for n in [3, 5, 10]:
    for w in ["equal", "rank", "score"]:
        cfg = StrategyConfig(top_n=n, hold_days=5000, weighting=w, start_month_idx=start_m,
                             entry_delay=1)
        r = simulate(md, md.stocks, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        excess = (m["cagr"] - bm["cagr"]) * 100
        print(f"  Top-{n:2d} {w:5s} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
