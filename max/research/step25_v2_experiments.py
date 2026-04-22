"""Step 25 v2: CAP5 improvement experiments.

Baseline: Top-5 rank-weighted, 5% concentration cap, hold-forever, entry_delay=1.
Current CAP5: 20Y CAGR +17.41%, Sharpe 1.34, MaxDD -46.15%, 1Y +2.58pp excess.

E1: cap fraction sweep on Top-5 rank (no prior 5% cap)
E2: top_n sweep with 5% cap
E3: sector_cap on CAP5
E4: rebound gate on CAP5 (isolated)
E5: regime scale-down (not full off) on CAP5
E6: combined top winners
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
START_1Y = TOTAL_M - 12
FROM_1Y = md.month_first_idx[START_1Y]
TO = len(md.all_dates)


def win_metrics(eq, invested, from_i, to_i):
    if invested <= 0:
        return None
    yrs = (to_i - from_i) / 252
    final = eq[to_i - 1]
    cagr = (final / invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    tr = (final - invested) / invested
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    return dict(cagr=cagr, tr=tr, mdd=mdd, final=final, invested=invested)


# Benchmarks
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm_full = compute_metrics(md, bench_full.equity, bench_full.total_invested)
bench_1y = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bm_1y = win_metrics(bench_1y.equity, 1000 * 12, FROM_1Y, TO)

print(f"SPY full 20y: CAGR {bm_full['cagr']*100:+6.2f}%  MaxDD {bm_full['maxdd']*100:+6.2f}%")
print(f"SPY 1Y     : CAGR {bm_1y['cagr']*100:+6.2f}%  MaxDD {bm_1y['mdd']*100:+6.2f}%\n")


# Store results for E6 composition
RESULTS = {}


def run(label, top_n=5, **kwargs):
    cfg = StrategyConfig(top_n=top_n, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1, **kwargs)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)

    cfg1 = StrategyConfig(top_n=top_n, hold_days=5000, weighting="rank",
                          start_month_idx=START_1Y, entry_delay=1, **kwargs)
    r1 = simulate(md, md.stocks, cfg1)
    m1 = win_metrics(r1.equity, r1.total_invested, FROM_1Y, TO)

    if m is None or m1 is None or not m:
        print(f"  {label:48s}  FAILED (no positions?)"); return None

    ex20 = (m["cagr"] - bm_full["cagr"]) * 100
    ex1 = (m1["cagr"] - bm_1y["cagr"]) * 100
    n_pos = len(r.positions)
    print(f"  {label:48s}  20y CAGR {m['cagr']*100:+6.2f}% ({ex20:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"1Y {m1['cagr']*100:+6.2f}% ({ex1:+5.2f}pp)  "
          f"n_pos {n_pos}")
    RESULTS[label] = dict(cagr20=m['cagr'], ex20=ex20, mdd=m['maxdd'],
                          sharpe=m['sharpe'], cagr1=m1['cagr'], ex1=ex1,
                          n_pos=n_pos)
    return RESULTS[label]


print("## E0: CAP5 baseline (Top-5 rank, 5% cap)")
run("CAP5 baseline (max_ticker_frac=0.05)", max_ticker_frac=0.05)

print("\n## E1: cap fraction sweep on Top-5 rank")
for frac in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    run(f"E1 cap_frac={frac:.2f}", max_ticker_frac=frac)

print("\n## E2: top_n sweep with 5% cap")
for n in [2, 3, 4, 5, 6, 7, 8, 10]:
    run(f"E2 top_n={n} cap=0.05", top_n=n, max_ticker_frac=0.05)

print("\n## E3: sector_cap on CAP5")
for sc in [1, 2, 3, 4]:
    run(f"E3 sector_cap={sc} cap=0.05", max_ticker_frac=0.05, sector_cap=sc)

print("\n## E4: rebound gate on CAP5 (isolated)")
for lb, thr in [(63, 0.0), (63, 0.05), (126, 0.0), (126, 0.05),
                (252, 0.0), (252, 0.05), (252, 0.10)]:
    run(f"E4 rebound lb={lb}d thr>={thr:+.0%} cap=0.05",
        max_ticker_frac=0.05,
        rebound_lookback_days=lb, rebound_min_return=thr)

print("\n## E5: regime scale-down on CAP5")
for sd in [0.3, 0.5, 0.7]:
    run(f"E5 regime_gate SD={sd:.2f} cap=0.05",
        max_ticker_frac=0.05, regime_gate=True, regime_scale_down=sd)


# ----- E6: pick top performers from E1-E5 and combine -----
print("\n## E6: combined winners")

# Select top-N by a composite score: 20Y CAGR excess + 0.5 * 1Y CAGR excess - 0.2*(maxdd-0.46)*100
def rank_score(r):
    return r['ex20'] + 0.5 * r['ex1'] - max(0, (r['mdd'] - 0.46) * 20)

# Gather the E1-E5 candidates (exclude baseline duplicates)
candidates = {}
for lbl, r in RESULTS.items():
    if lbl.startswith("E1 ") or lbl.startswith("E2 ") or lbl.startswith("E3 ") \
       or lbl.startswith("E4 ") or lbl.startswith("E5 "):
        candidates[lbl] = r

ranked = sorted(candidates.items(), key=lambda kv: rank_score(kv[1]), reverse=True)
print("\n  Top 8 standalone candidates by composite score:")
for lbl, r in ranked[:8]:
    print(f"    {lbl:48s}  score={rank_score(r):+5.2f}  "
          f"20y {r['cagr20']*100:+6.2f}% ({r['ex20']:+5.2f}pp)  "
          f"MDD {r['mdd']*100:+6.2f}%  1Y ({r['ex1']:+5.2f}pp)")

# Derive characteristic knobs from top individual winners
# Now construct combined configs. Hand-pick combinations that use distinct knobs:
# (a) best cap from E1 (often stays 0.05 or near)
# (b) best top_n from E2
# (c) best sector_cap from E3
# (d) best rebound from E4
# (e) best regime SD from E5

def best_of(prefix):
    subset = [(l, r) for l, r in candidates.items() if l.startswith(prefix)]
    if not subset:
        return None
    return max(subset, key=lambda kv: rank_score(kv[1]))

best_e1 = best_of("E1 ")
best_e2 = best_of("E2 ")
best_e3 = best_of("E3 ")
best_e4 = best_of("E4 ")
best_e5 = best_of("E5 ")

print("\n  Best-per-experiment:")
for tag, b in [("E1", best_e1), ("E2", best_e2), ("E3", best_e3),
               ("E4", best_e4), ("E5", best_e5)]:
    if b: print(f"    {tag}: {b[0]}  score={rank_score(b[1]):+5.2f}")


def parse_e1(label):
    # "E1 cap_frac=0.05"
    return float(label.split("=")[1])

def parse_e2(label):
    # "E2 top_n=5 cap=0.05"
    return int(label.split("top_n=")[1].split(" ")[0])

def parse_e3(label):
    # "E3 sector_cap=2 cap=0.05"
    return int(label.split("sector_cap=")[1].split(" ")[0])

def parse_e4(label):
    # "E4 rebound lb=63d thr>=+0% cap=0.05"
    lb = int(label.split("lb=")[1].split("d")[0])
    thr_str = label.split("thr>=")[1].split("%")[0]
    thr = float(thr_str) / 100.0
    return lb, thr

def parse_e5(label):
    # "E5 regime_gate SD=0.50 cap=0.05"
    return float(label.split("SD=")[1].split(" ")[0])


# Baseline knob values
cap = parse_e1(best_e1[0]) if best_e1 else 0.05
top_n_best = parse_e2(best_e2[0]) if best_e2 else 5
sector_cap_best = parse_e3(best_e3[0]) if best_e3 else None
reb_lb, reb_thr = parse_e4(best_e4[0]) if best_e4 else (None, 0.0)
regime_sd = parse_e5(best_e5[0]) if best_e5 else None

print(f"\n  Extracted knobs: cap={cap} top_n={top_n_best} sector_cap={sector_cap_best} "
      f"rebound=({reb_lb},{reb_thr}) regime_sd={regime_sd}")

# Combined configs (all keep max_ticker_frac=cap)
print("\n  Combined variants:")

# Pairwise combos first
run(f"E6a cap={cap} + top_n={top_n_best}",
    top_n=top_n_best, max_ticker_frac=cap)

if sector_cap_best is not None:
    run(f"E6b cap={cap} + sector_cap={sector_cap_best}",
        max_ticker_frac=cap, sector_cap=sector_cap_best)

if reb_lb is not None:
    run(f"E6c cap={cap} + rebound lb={reb_lb}d thr>={reb_thr:+.0%}",
        max_ticker_frac=cap,
        rebound_lookback_days=reb_lb, rebound_min_return=reb_thr)

if regime_sd is not None:
    run(f"E6d cap={cap} + regime_sd={regime_sd:.2f}",
        max_ticker_frac=cap, regime_gate=True, regime_scale_down=regime_sd)

# Triple combos
if sector_cap_best is not None and reb_lb is not None:
    run(f"E6e cap + top_n + sector_cap + rebound",
        top_n=top_n_best, max_ticker_frac=cap, sector_cap=sector_cap_best,
        rebound_lookback_days=reb_lb, rebound_min_return=reb_thr)

if sector_cap_best is not None and regime_sd is not None:
    run(f"E6f cap + top_n + sector_cap + regime_sd",
        top_n=top_n_best, max_ticker_frac=cap, sector_cap=sector_cap_best,
        regime_gate=True, regime_scale_down=regime_sd)

if reb_lb is not None and regime_sd is not None:
    run(f"E6g cap + top_n + rebound + regime_sd",
        top_n=top_n_best, max_ticker_frac=cap,
        rebound_lookback_days=reb_lb, rebound_min_return=reb_thr,
        regime_gate=True, regime_scale_down=regime_sd)

# All five combined
if sector_cap_best is not None and reb_lb is not None and regime_sd is not None:
    run(f"E6h cap + top_n + sector_cap + rebound + regime_sd",
        top_n=top_n_best, max_ticker_frac=cap, sector_cap=sector_cap_best,
        rebound_lookback_days=reb_lb, rebound_min_return=reb_thr,
        regime_gate=True, regime_scale_down=regime_sd)


# Final ranking
print("\n## FINAL: Top-10 across ALL experiments by composite score")
all_ranked = sorted(RESULTS.items(), key=lambda kv: rank_score(kv[1]), reverse=True)
for lbl, r in all_ranked[:10]:
    print(f"  {lbl:52s}  score={rank_score(r):+5.2f}  "
          f"20y {r['cagr20']*100:+6.2f}% ({r['ex20']:+5.2f}pp)  "
          f"MDD {r['mdd']*100:+6.2f}%  Sh {r['sharpe']:.2f}  "
          f"1Y ({r['ex1']:+5.2f}pp)")
