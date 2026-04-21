"""Step 6: trailing stop / drawdown stop on Top-5 rank, hold-forever."""
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()
print("## Trailing stop (Top-5 rank, hold-forever)")
for ts in [None, 0.15, 0.25, 0.35, 0.50]:
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank", trail_stop=ts,
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  trail={ts} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
print()
print("## DD-from-cost stop")
for ds in [None, 0.20, 0.35, 0.50]:
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank", dd_stop=ds,
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  dd_stop={ds} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
