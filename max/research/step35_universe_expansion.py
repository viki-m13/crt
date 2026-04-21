"""Step 35: universe expansion test.

fetch_new_tickers.py added 31 new tickers to the raw parquets. This step
validates whether CAP5 on the expanded 128-ticker universe beats CAP5 on
the original 97-ticker universe.

Candidates for expansion (31 tickers, all with ≥15Y history):
  ORCL, QCOM, KLAC, LRCX, ASML, TER, NTAP, GRMN, SWKS, INFY,   # Tech
  MA, CME, BX, ARCC,                                           # Fin
  VRTX,                                                        # Health
  AZO, EXPE, STZ, PSA, JCI, EMR, SCCO, CPRT, LUV,              # Disc/Indust
  SHW, ADP, ACN, ISRG, REGN, CHTR, GIS                         # Defensive/other

Requires second regen on 128-ticker parquets (run AFTER first regen completes).

Compares:
  - CAP5 on 97 tickers (baseline)
  - CAP5 on 128 tickers (full expansion)
  - CAP5 on 97 + top-10 additions (curated subset)
  - CAP5 on 97 + bottom-10 additions (sanity check, see if low-quality hurts)

Also reports which of the new tickers actually get picked during the 20Y.
"""
import os, sys, json
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import simulate, simulate_benchmark, compute_metrics, StrategyConfig
from bt_core_ext import load_and_prep_ext

NEW_TICKERS = {
    "ORCL", "QCOM", "KLAC", "LRCX", "ASML", "TER", "NTAP", "GRMN", "SWKS", "INFY",
    "MA", "CME", "BX", "ARCC", "VRTX",
    "AZO", "EXPE", "STZ", "PSA", "JCI", "EMR", "SCCO", "CPRT", "LUV",
    "SHW", "ADP", "ACN", "ISRG", "REGN", "CHTR", "GIS",
}


md, start_m = load_and_prep_ext()
print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")
available_new = sorted([t for t in NEW_TICKERS if t in md.stocks])
print(f"New tickers in parquet: {len(available_new)}/{len(NEW_TICKERS)}")
print(f"  {available_new}")

original_universe = [t for t in md.stocks if t not in NEW_TICKERS]
full_universe = list(md.stocks)
print(f"Original (without new): {len(original_universe)} tickers")
print(f"Full (with new):        {len(full_universe)} tickers")


def mk(universe_filter=None, label=""):
    return StrategyConfig(
        start_month_idx=start_m,
        top_n=5, max_ticker_frac=0.05, hold_days=5000,
        weighting="rank", entry_delay=1,
        universe_filter=set(universe_filter) if universe_filter is not None else None,
        label=label,
    )


# 1. Original 97-ticker baseline
# 2. Full 128-ticker universe
# 3. Original + each individual new ticker (marginal effect)

SPY_cfg = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, SPY_cfg.equity, SPY_cfg.total_invested)

print("\n" + "=" * 90)
print(f"{'variant':35s}  {'CAGR':>8s}  {'TR':>9s}  {'MaxDD':>8s}  {'Sharpe':>7s}  {'Final':>12s}")
print("-" * 90)
print(f"{'SPY benchmark':35s}  {spy_m['cagr']*100:+7.2f}%  {spy_m['total_return']*100:+7.2f}%  "
      f"{-spy_m['maxdd']*100:+7.2f}%  {spy_m['sharpe']:6.2f}  ${spy_m['final']:11,.0f}")

# Baseline: original 97 tickers
r_orig = simulate(md, md.stocks, mk(universe_filter=original_universe, label="CAP5-97"))
m_orig = compute_metrics(md, r_orig.equity, r_orig.total_invested)
print(f"{'CAP5 (original 97)':35s}  {m_orig['cagr']*100:+7.2f}%  {m_orig['total_return']*100:+7.2f}%  "
      f"{-m_orig['maxdd']*100:+7.2f}%  {m_orig['sharpe']:6.2f}  ${m_orig['final']:11,.0f}")

# Full: 128 tickers
r_full = simulate(md, md.stocks, mk(universe_filter=full_universe, label="CAP5-full"))
m_full = compute_metrics(md, r_full.equity, r_full.total_invested)
print(f"{'CAP5 (full 128)':35s}  {m_full['cagr']*100:+7.2f}%  {m_full['total_return']*100:+7.2f}%  "
      f"{-m_full['maxdd']*100:+7.2f}%  {m_full['sharpe']:6.2f}  ${m_full['final']:11,.0f}")

delta_full_vs_orig = (m_full['cagr'] - m_orig['cagr']) * 100
print(f"\nΔ CAGR (full vs orig): {delta_full_vs_orig:+.2f}pp")


# How often are new tickers actually picked?
print("\n" + "=" * 90)
print("New-ticker pick frequency (CAP5 full 128):")
pick_counter = Counter()
total_picks = 0
for date, picks in r_full.picks_by_month:
    for tk, score in picks:
        pick_counter[tk] += 1
        total_picks += 1

new_picks = {t: pick_counter.get(t, 0) for t in available_new}
new_picks_sorted = sorted(new_picks.items(), key=lambda x: -x[1])
total_new = sum(new_picks.values())
print(f"Total picks across all months: {total_picks}")
print(f"Picks on NEW tickers: {total_new} ({total_new/total_picks*100:.1f}%)")
print(f"\nPer-new-ticker pick count (sorted descending):")
for tk, cnt in new_picks_sorted:
    print(f"  {tk:6s}  {cnt:4d}  ({cnt/total_picks*100:.2f}% of all picks)")


# Marginal test: original + each single new ticker (does any one add value?)
print("\n" + "=" * 90)
print("Marginal effect — add ONE new ticker at a time:")
print(f"{'+ ticker':10s}  {'CAGR':>8s}  {'ΔvsOrig':>7s}  {'MaxDD':>8s}  {'Picks':>6s}")
print("-" * 60)
marg_results = []
for new_tk in available_new:
    uf = set(original_universe) | {new_tk}
    cfg = mk(universe_filter=uf, label=f"orig+{new_tk}")
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    # Count picks on new_tk
    tk_picks = sum(1 for d, picks in r.picks_by_month for tk, _ in picks if tk == new_tk)
    delta = (m['cagr'] - m_orig['cagr']) * 100
    marg_results.append((new_tk, m, delta, tk_picks))
    print(f"  {new_tk:8s}  {m['cagr']*100:+7.2f}%  {delta:+6.2f}  "
          f"{-m['maxdd']*100:+7.2f}%  {tk_picks:6d}")

# Top-5 positive marginal contributors
marg_results.sort(key=lambda x: -x[2])
print(f"\nTop-10 positive marginal contributors (vs original CAP5 {m_orig['cagr']*100:+.2f}%):")
for tk, m, delta, picks in marg_results[:10]:
    print(f"  +{tk:6s} → {m['cagr']*100:+.2f}% ({delta:+.2f}pp, {picks} picks)")
print(f"\nTop-10 negative marginal contributors:")
for tk, m, delta, picks in marg_results[-10:]:
    print(f"  +{tk:6s} → {m['cagr']*100:+.2f}% ({delta:+.2f}pp, {picks} picks)")


# Curated: original + all positive marginal contributors
pos_contribs = [tk for tk, _, delta, _ in marg_results if delta > 0]
print(f"\nPositive marginal contributors: {len(pos_contribs)}")
print(f"  {pos_contribs}")
if pos_contribs:
    uf = set(original_universe) | set(pos_contribs)
    cfg = mk(universe_filter=uf, label="orig+positives")
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    delta = (m['cagr'] - m_orig['cagr']) * 100
    print(f"\n  CAP5 (orig + {len(pos_contribs)} positives):")
    print(f"    CAGR {m['cagr']*100:+.2f}% ({delta:+.2f}pp)  MaxDD {-m['maxdd']*100:+.2f}%  "
          f"Sharpe {m['sharpe']:.2f}  Final ${m['final']:,.0f}")
    # ⚠️ This is IN-SAMPLE selection and should not be treated as a predictive
    # result; reported only to measure the theoretical max gain from perfect
    # hindsight ticker selection within the added pool.

# Save summary
out = {
    "baseline_orig_97": {"cagr": m_orig['cagr'], "maxdd": m_orig['maxdd'], "sharpe": m_orig['sharpe']},
    "full_128": {"cagr": m_full['cagr'], "maxdd": m_full['maxdd'], "sharpe": m_full['sharpe']},
    "marginal": {tk: {"cagr": m['cagr'], "delta_pp": d, "picks": p} for tk, m, d, p in marg_results},
}
with open("/home/user/crt/max/research/step35_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\n## DONE")
