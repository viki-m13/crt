"""Step 1: establish the single benchmark we must beat — SPY DCA buy-and-hold."""
from bt_core import load_and_prep, simulate_benchmark, compute_metrics, fmt_metrics

md, start_m = load_and_prep()
print(f"Spine: {md.all_dates[0]} → {md.all_dates[-1]}  ({len(md.all_dates)} trading days)")
print(f"Start month: {md.all_dates[md.month_first_idx[start_m]]}")
print()
print("## SPY DCA benchmarks (next-day-open entry)")
for hd, name in [(252, "hold 1Y then cash"), (756, "hold 3Y then cash"),
                 (1260, "hold 5Y (buy-and-hold)"), (5000, "infinite hold (B&H)")]:
    b = simulate_benchmark(md, ["SPY"], hd, start_m, entry_delay=1)
    m = compute_metrics(md, b.equity, b.total_invested)
    print(f"  SPY DCA {name:26s} | {fmt_metrics(m)}")
print()
print("=> THE benchmark to beat: SPY DCA buy-and-hold (infinite hold).")
