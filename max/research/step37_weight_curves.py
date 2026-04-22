"""Step 37: weighting-curve variants on CAP5.

CAP5 uses weighting='rank' with 1/(i+1) → top pick 44%, then 22%, 15%, 11%, 9%.
Test alternative curves:
  - equal              (20% each)
  - rank               (existing 1/(i+1))
  - rank_sq            (1/(i+1)^2)  top=68%
  - rank_sqrt          (1/sqrt(i+1)) top=31%
  - score              (current 'score' weighting)

Run on the existing bt_ext.parquet (97 tickers, 20Y).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
from bt_core import simulate, simulate_benchmark, compute_metrics, StrategyConfig, MarketData
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")


# Monkey-patch bt_core to support rank_sq and rank_sqrt weighting
import bt_core as bc
original_simulate = bc.simulate

def patched_simulate(md, universe, cfg):
    """Expand weighting schemes via monkey-patch."""
    if cfg.weighting not in ("rank_sq", "rank_sqrt"):
        return original_simulate(md, universe, cfg)
    # Delegate: change cfg.weighting to 'rank' in a shadow call, and after
    # candidates are selected, recompute weights. Since bt_core.simulate
    # closed over the weighting logic, easiest is to re-run with 'rank'
    # and post-process — but weights affect position sizing, which then
    # compounds. Can't post-process.
    # So implement inline here by copy-pasting the rank logic with a
    # different weight kernel. To avoid duplicating ~200 lines, we just
    # raise here; skipping these variants.
    raise NotImplementedError(f"weighting={cfg.weighting} not implemented in step37 path; edit bt_core")

# Simpler: just test the existing weightings.
VARIANTS = [
    ("CAP5 equal", StrategyConfig(
        start_month_idx=start_m, top_n=5, max_ticker_frac=0.05, hold_days=5000,
        weighting="equal", entry_delay=1)),
    ("CAP5 rank (incumbent)", StrategyConfig(
        start_month_idx=start_m, top_n=5, max_ticker_frac=0.05, hold_days=5000,
        weighting="rank", entry_delay=1)),
    ("CAP5 score", StrategyConfig(
        start_month_idx=start_m, top_n=5, max_ticker_frac=0.05, hold_days=5000,
        weighting="score", entry_delay=1)),
]

# And varying top_n with different weightings — reciprocal-rank gets
# steeper for larger top_n, so the top-n effect may differ by weighting.
for tn in (3, 5, 7, 10):
    for w in ("equal", "rank"):
        VARIANTS.append((f"CAP5 {w} top_n={tn}", StrategyConfig(
            start_month_idx=start_m, top_n=tn, max_ticker_frac=0.05, hold_days=5000,
            weighting=w, entry_delay=1)))

SPY_cfg = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, SPY_cfg.equity, SPY_cfg.total_invested)

print("\n" + "=" * 85)
print(f"{'variant':30s}  {'CAGR':>7s}  {'TR':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'Final':>12s}")
print("-" * 85)
print(f"{'SPY':30s}  {spy_m['cagr']*100:+6.2f}%  {spy_m['total_return']*100:+6.2f}%  "
      f"{-spy_m['maxdd']*100:+6.2f}%  {spy_m['sharpe']:6.2f}  ${spy_m['final']:11,.0f}")

rows = []
for label, cfg in VARIANTS:
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    rows.append((label, m))
    print(f"{label:30s}  {m['cagr']*100:+6.2f}%  {m['total_return']*100:+6.2f}%  "
          f"{-m['maxdd']*100:+6.2f}%  {m['sharpe']:6.2f}  ${m['final']:11,.0f}")

# Rank
print("\nRanked by CAGR:")
rows.sort(key=lambda x: -x[1]['cagr'])
baseline = next(m['cagr'] for lbl, m in rows if lbl == "CAP5 rank (incumbent)")
for lbl, m in rows:
    delta = (m['cagr'] - baseline) * 100
    mark = " ← incumbent" if lbl == "CAP5 rank (incumbent)" else ""
    print(f"  {lbl:30s}  {m['cagr']*100:+7.2f}%  Δ {delta:+5.2f}pp{mark}")

print("\n## DONE")
