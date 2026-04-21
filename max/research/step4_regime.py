"""Step 4: SPY 200SMA regime gate on Top-5 rank, hold-forever."""
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()
print("## Regime gate (SPY 200SMA, Top-5 rank, hold-forever)")
for sd in [1.0, 0.75, 0.5, 0.25, 0.0]:
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank",
                         regime_gate=(sd < 1.0), regime_scale_down=sd,
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    excess = (m["cagr"] - bm["cagr"]) * 100
    label = "none" if sd == 1.0 else f"scale={sd}"
    print(f"  gate {label:10s} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
