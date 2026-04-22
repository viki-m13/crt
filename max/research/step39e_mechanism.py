"""Step 39e: understand the smoothing mechanism.

Step 39d showed SMA 60M gives +1.34pp CAGR. Understand *why*:
  - Pick turnover rate: do long smooths keep the same names?
  - Sector concentration: are we always buying the same sector?
  - Find the ceiling: SMA 96M, 120M. Does it break?
  - Compare vs a pure "average historical score" static ranking
  - Is this actually momentum in disguise?
"""
import os, sys, math
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (compute_metrics, StrategyConfig, SECTOR_MAP, simulate_benchmark)
from bt_core_ext import load_and_prep_ext
from step39_signal_smoothing import simulate_smoothed, current, trailing


md, start_m = load_and_prep_ext()
cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

# Test smoothing ceiling
print("## Smoothing ceiling: does it break at very long windows?")
for months in [24, 48, 60, 96, 120, 180]:
    r = simulate_smoothed(md, md.stocks, cfg, trailing(months), None)
    m = compute_metrics(md, r.equity, r.total_invested)
    print(f"  SMA {months:3d}M  CAGR {m['cagr']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"MaxDD {-m['maxdd']*100:+.2f}%  Final ${m['final']:11,.0f}")

# Pick turnover analysis
print("\n## Pick turnover analysis")
for months in [0, 6, 24, 60]:
    sf = current if months == 0 else trailing(months)
    r = simulate_smoothed(md, md.stocks, cfg, sf, None)
    # Count unique tickers per rolling 12 months of picks
    prev_picks = None
    total_changes = 0
    total_compares = 0
    all_tickers = set()
    all_picks = []
    for dt, picks in r.picks_by_month:
        tkrs = set(tk for tk, _ in picks)
        all_tickers.update(tkrs)
        all_picks.append(tkrs)
        if prev_picks is not None:
            # Fraction of picks that are new vs last month
            new = tkrs - prev_picks
            total_changes += len(new)
            total_compares += cfg.top_n
        prev_picks = tkrs
    avg_turnover = total_changes / total_compares if total_compares else 0
    # Count of total ticker-months picked per ticker (concentration)
    tk_counts = Counter()
    for ps in all_picks:
        for tk in ps: tk_counts[tk] += 1
    top5 = tk_counts.most_common(5)
    label = f"incumbent" if months == 0 else f"SMA {months}M"
    print(f"  {label:15s}  unique tickers used: {len(all_tickers):3d}/{len(md.stocks)}  "
          f"avg monthly turnover: {avg_turnover*100:.1f}%")
    print(f"                   most-picked: {', '.join(f'{tk}({c})' for tk, c in top5)}")


# Sector concentration
print("\n## Sector concentration of picks")
for months in [0, 6, 24, 60]:
    sf = current if months == 0 else trailing(months)
    r = simulate_smoothed(md, md.stocks, cfg, sf, None)
    sector_months = Counter()
    for dt, picks in r.picks_by_month:
        for tk, _ in picks:
            sec = SECTOR_MAP.get(tk, "?")
            sector_months[sec] += 1
    total = sum(sector_months.values())
    label = f"incumbent" if months == 0 else f"SMA {months}M"
    print(f"  {label:15s}  top sectors:")
    for sec, cnt in sector_months.most_common(5):
        print(f"    {sec:20s} {cnt/total*100:5.1f}%")


# Momentum check: does SMA_X-month-old SPY-relative price correlate with picks?
print("\n## Momentum check: do SMA-smooth picks follow recent winners?")
# For each month, for each pick, compute its 1Y prior return. Compare distributions.
results = {}
for months in [0, 6, 24, 60]:
    sf = current if months == 0 else trailing(months)
    r = simulate_smoothed(md, md.stocks, cfg, sf, None)
    priors = []
    for dt, picks in r.picks_by_month:
        di = md.all_dates.index(dt) if dt in md.all_dates else None
        if di is None or di < 252: continue
        for tk, _ in picks:
            px = md.prices.get(tk)
            if px is None: continue
            p_now = px[di]; p_prior = px[di - 252]
            if math.isfinite(p_now) and math.isfinite(p_prior) and p_prior > 0:
                priors.append(p_now / p_prior - 1)
    if priors:
        results[months] = priors
        label = f"incumbent" if months == 0 else f"SMA {months}M"
        print(f"  {label:15s}  1Y prior return at pick:  "
              f"median {np.median(priors)*100:+6.2f}%  mean {np.mean(priors)*100:+6.2f}%  "
              f"Q25 {np.percentile(priors, 25)*100:+6.2f}%  Q75 {np.percentile(priors, 75)*100:+6.2f}%")


# "Static ranking" ablation — just rank by all-history mean score and buy top-5 forever
print("\n## Static all-history ranking (no time-varying signal)")

def static_score_fn(f, di):
    valid = f[np.isfinite(f)]
    return float(np.mean(valid)) if len(valid) > 0 else None


r = simulate_smoothed(md, md.stocks, cfg, static_score_fn, None)
m = compute_metrics(md, r.equity, r.total_invested)
print(f"  static all-history mean:  CAGR {m['cagr']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
      f"MaxDD {-m['maxdd']*100:+.2f}%  Final ${m['final']:11,.0f}")

# Peek-forward static: ceiling of the static idea (uses forward info, just for reference)
def peek_score_fn(f, di):
    # Uses all future info — this is the CHEATING upper bound
    valid = f[np.isfinite(f)]
    return float(np.mean(valid)) if len(valid) > 0 else None

# The peek version would literally be the same since we use ALL history. Skip.


# Direct per-window comparison: does SMA 60M's edge hold in every rolling 10Y window?
print("\n## SMA 60M vs incumbent — rolling 10Y detail")
r_inc = simulate_smoothed(md, md.stocks, cfg, current, None)
r_long = simulate_smoothed(md, md.stocks, cfg, trailing(60), None)

def win_metric(eq, from_i, to_i):
    if from_i >= to_i or to_i > len(eq): return None
    start = max(eq[from_i], 0.01)
    end = eq[to_i - 1]
    if end <= 0: return None
    yrs = (to_i - from_i) / 252
    return {"cagr": (end / start) ** (1 / yrs) - 1}


WIN_D = 10 * 252
wins = 0; n = 0
for from_i in range(0, len(md.all_dates) - WIN_D, 252):
    to_i = from_i + WIN_D
    m_i = win_metric(r_inc.equity, from_i, to_i)
    m_l = win_metric(r_long.equity, from_i, to_i)
    if not (m_i and m_l): continue
    n += 1
    d_from = md.all_dates[from_i][:7]
    d_to = md.all_dates[to_i - 1][:7]
    delta = (m_l['cagr'] - m_i['cagr']) * 100
    mark = "+win" if delta > 0 else "-loss"
    if delta > 0: wins += 1
    print(f"  {d_from} → {d_to}  incumbent {m_i['cagr']*100:+6.2f}%  SMA60M {m_l['cagr']*100:+6.2f}%  Δ {delta:+5.2f}pp  {mark}")
print(f"\n  SMA 60M win rate: {wins}/{n}")

print("\n## DONE")
