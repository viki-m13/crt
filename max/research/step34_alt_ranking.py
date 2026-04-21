"""Step 34: alternative ranking formulas.

CAP5 currently ranks candidates by `final` = conviction score (quality ×
10D_prob × pullback_gate). This step tests alternative composite formulas
that re-weight the signal components at ranking time.

Requires the regenerated bt_ext.parquet with `wash` and `final_raw` columns.

Formulas tested:
  - final              (incumbent)
  - final_raw          (edge × wash_adjust) — the raw scanner score
  - wash               (rank by washout alone — contrarian baseline)
  - final_x_wash       (final × wash/100) — bigger pullback = higher rank
  - raw_x_wash         (raw × wash/100)
  - final+alpha_wash   (final × (1 + alpha × wash/100))  with alpha ∈ {0.25, 0.5, 1.0, 2.0}

All variants use the CAP5 config: top_n=5, cap=5%, rank_weighting, hold=forever.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from bt_core import simulate, simulate_benchmark, compute_metrics, StrategyConfig
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
print(f"Loaded {len(md.stocks)} stocks, "
      f"{len(md.all_dates)} dates ({md.all_dates[0]} → {md.all_dates[-1]})")
has_wash = md.washes is not None
has_raw = md.finals_raw is not None
print(f"Has wash column: {has_wash}    Has final_raw column: {has_raw}")

if not has_wash or not has_raw:
    print("ERROR: bt_ext.parquet missing wash/final_raw columns.")
    print("Re-run regen_scores_ext.py after updating daily_scan_max.py.")
    sys.exit(1)


# Baseline CAP5 config — everything else matches.
def mk(rank_formula="final", alpha=0.5, label=None):
    return StrategyConfig(
        start_month_idx=start_m,
        top_n=5, max_ticker_frac=0.05, hold_days=5000,
        weighting="rank", entry_delay=1,
        rank_formula=rank_formula, rank_alpha=alpha,
        label=label or rank_formula,
    )


VARIANTS = [
    ("CAP5 (final)          ", mk("final", label="final")),
    ("CAP5 (final_raw)      ", mk("final_raw", label="final_raw")),
    ("CAP5 (wash)           ", mk("wash", label="wash")),
    ("CAP5 (final×wash)     ", mk("final_x_wash", label="final_x_wash")),
    ("CAP5 (raw×wash)       ", mk("raw_x_wash", label="raw_x_wash")),
    ("CAP5 (final+0.25*wash)", mk("final+alpha_wash", 0.25, "alpha=0.25")),
    ("CAP5 (final+0.50*wash)", mk("final+alpha_wash", 0.50, "alpha=0.50")),
    ("CAP5 (final+1.00*wash)", mk("final+alpha_wash", 1.00, "alpha=1.00")),
    ("CAP5 (final+2.00*wash)", mk("final+alpha_wash", 2.00, "alpha=2.00")),
]


# Also run benchmarks.
SPY_cfg = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, SPY_cfg.equity, SPY_cfg.total_invested)

print("\n" + "=" * 90)
print(f"{'variant':25s}  {'CAGR':>8s}  {'TR':>9s}  {'MaxDD':>8s}  {'Sharpe':>7s}  {'Final':>12s}")
print("-" * 90)
print(f"{'SPY benchmark':25s}  {spy_m['cagr']*100:+7.2f}%  {spy_m['total_return']*100:+7.2f}%  "
      f"{-spy_m['maxdd']*100:+7.2f}%  {spy_m['sharpe']:6.2f}  ${spy_m['final']:11,.0f}")

rows = []
for label, cfg in VARIANTS:
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    rows.append((label, m, r))
    print(f"{label:25s}  {m['cagr']*100:+7.2f}%  {m['total_return']*100:+7.2f}%  "
          f"{-m['maxdd']*100:+7.2f}%  {m['sharpe']:6.2f}  ${m['final']:11,.0f}")

# Rank by CAGR
print("\n" + "=" * 90)
print("Ranked by CAGR:")
rows.sort(key=lambda x: -x[1]['cagr'])
baseline_cagr = None
for lbl, m, _ in rows:
    if "final)" in lbl.strip() and "raw" not in lbl and "x" not in lbl.lower() and "+" not in lbl:
        baseline_cagr = m['cagr']; break
for lbl, m, _ in rows:
    delta = f"{(m['cagr']-baseline_cagr)*100:+5.2f}pp" if baseline_cagr else ""
    print(f"  {lbl:25s}  {m['cagr']*100:+7.2f}%  vs incumbent {delta}")


# Top-5 last-month picks summary per variant
print("\n" + "=" * 90)
print("Last-month top-5 picks (most recent rebalance) per variant:")
for lbl, m, r in rows:
    if not r.picks_by_month:
        continue
    last_date, picks = r.picks_by_month[-1]
    tks = [t for t, _ in picks]
    print(f"  {lbl:25s}  {last_date}: {tks}")

print("\n## DONE")
