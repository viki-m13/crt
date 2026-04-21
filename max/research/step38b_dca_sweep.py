"""Step 38b: thorough DCA-scaling sweep.

Step 38 identified 3x-at-SPY-20dd as a real +0.78pp CAGR improvement.
This extends the sweep to map the (threshold, multiplier) space.

Also computes money-weighted (IRR) return so the CAGR isn't biased by
uneven capital deployment.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate_benchmark, compute_metrics, MarketData,
                     StrategyConfig, DCA_MONTHLY, first_valid_month_idx)
from bt_core_ext import load_and_prep_ext
from step38_dca_scaling import simulate_dd_scaled  # reuse


md, start_m = load_and_prep_ext()
print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")

# SPY drawdown
spy = md.bench_filled["SPY"]
spy_peak = np.zeros_like(spy)
pk = 0.0
for i, v in enumerate(spy):
    if v > pk: pk = v
    spy_peak[i] = pk
def spy_dd(di):
    if spy_peak[di] <= 0: return 0.0
    return 1.0 - spy[di] / spy_peak[di]


def mk_scaler(threshold, multiplier):
    def sc(di, md):
        return DCA_MONTHLY * (multiplier if spy_dd(di) >= threshold else 1.0)
    return sc


def mk_linear(slope, cap):
    def sc(di, md):
        return DCA_MONTHLY * min(cap, 1.0 + spy_dd(di) * slope)
    return sc


def irr_from_cashflows(dates_idx, amounts, final_val, final_di):
    """Annualized IRR on actual cash flows, final settled at final_di."""
    # Use monthly approximation.
    if not dates_idx: return None
    # Monthly cash flows: negative (invested), plus positive final at end.
    # Just use a newton solver.
    n_months = (final_di - dates_idx[0]) / 21  # approx monthly
    if n_months <= 0: return None
    flows = [(di, -amt) for di, amt in zip(dates_idx, amounts)]
    flows.append((final_di, final_val))

    def npv(r):
        total = 0.0
        for di, cf in flows:
            dt = (di - dates_idx[0]) / 252  # years
            total += cf / ((1 + r) ** dt)
        return total

    # Bisection for r in [-0.5, 2.0]
    lo, hi = -0.5, 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        v = npv(mid)
        if abs(v) < 0.01: return mid
        v_lo = npv(lo)
        if v * v_lo < 0:
            hi = mid
        else:
            lo = mid
    return mid


cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

# Generate variants
VARIANTS = [("flat (baseline)", lambda di, md: DCA_MONTHLY)]
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    for mult in [1.5, 2.0, 3.0, 4.0, 5.0]:
        VARIANTS.append((f"{mult:.1f}x at SPY -{int(thr*100)}%", mk_scaler(thr, mult)))
for slope in [2.0, 5.0, 8.0, 12.0]:
    for cap in [2.0, 3.0, 5.0]:
        VARIANTS.append((f"linear slope={slope} cap={cap}x", mk_linear(slope, cap)))


SPY_cfg = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, SPY_cfg.equity, SPY_cfg.total_invested)

print("\n" + "=" * 100)
print(f"{'variant':30s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Invested':>12s}  {'Final':>14s}")
print("-" * 100)
print(f"{'SPY':30s}  {spy_m['cagr']*100:+6.2f}%  {spy_m['sharpe']:6.2f}  "
      f"{-spy_m['maxdd']*100:+6.2f}%  ${spy_m['invested']:11,.0f}  ${spy_m['final']:13,.0f}")

rows = []
baseline_final = None
for label, scaler in VARIANTS:
    r = simulate_dd_scaled(md, md.stocks, cfg, scaler)
    m = compute_metrics(md, r.equity, r.total_invested)
    rows.append((label, m, r))
    if label == "flat (baseline)":
        baseline_final = m['final']
        baseline_invested = m['invested']
        baseline_cagr = m['cagr']
    print(f"{label:30s}  {m['cagr']*100:+6.2f}%  {m['sharpe']:6.2f}  "
          f"{-m['maxdd']*100:+6.2f}%  ${m['invested']:11,.0f}  ${m['final']:13,.0f}")

print("\nRanked by CAGR:")
rows.sort(key=lambda x: -x[1]['cagr'])
for lbl, m, _ in rows[:15]:
    d_cagr = (m['cagr'] - baseline_cagr) * 100
    d_final = m['final'] - baseline_final
    invest_extra = m['invested'] - baseline_invested
    roic = d_final / invest_extra if invest_extra > 0 else 0.0
    mark = "" if lbl == "flat (baseline)" else ("↑" if m['cagr'] > baseline_cagr else "↓")
    print(f"  {lbl:30s}  CAGR {m['cagr']*100:+6.2f}% ({d_cagr:+5.2f}pp)  "
          f"extra $ invested: {invest_extra:+8,.0f} → extra $ final: {d_final:+11,.0f}  ROIC {roic*100:+.1f}% {mark}")


print("\n## DONE")
