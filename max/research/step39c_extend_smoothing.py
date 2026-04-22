"""Step 39c: extend smoothing investigation.

Step 39 found trailing-6M SMA of `final` score wins 10/10 rolling 10Y
windows at +0.70pp CAGR vs incumbent. Push further:

1. EMA vs SMA — does exponential weighting (fresher bias) match SMA?
2. Longer windows (12M, 18M, 24M) — keep improving or decay?
3. Smooth + persist combo — already tried in step39, report full detail
4. Regime check: does smoothing help more in bull/bear/crisis?
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate_benchmark, compute_metrics, StrategyConfig,
                     first_valid_month_idx)
from bt_core_ext import load_and_prep_ext
from step39_signal_smoothing import simulate_smoothed, current, trailing


md, start_m = load_and_prep_ext()
cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

spy_r = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, spy_r.equity, spy_r.total_invested)


def ema(months, alpha_scale=1.0):
    """Exponentially weighted average over `months` with half-life ~months/2.

    Simpler: use α = 2/(N+1) where N = months*21 (daily samples).
    """
    n_days = months * 21
    alpha = 2.0 / (n_days + 1) * alpha_scale

    def sm(f, di):
        lo = max(0, di - n_days * 3)  # cover a few half-lives
        acc = None
        for i in range(lo, di + 1):
            v = f[i]
            if not np.isfinite(v):
                continue
            acc = v if acc is None else acc + alpha * (v - acc)
        return float(acc) if acc is not None else None
    return sm


def hybrid(sma_months, weight_current):
    """Blend trailing SMA with current value."""
    sma_fn = trailing(sma_months)

    def sm(f, di):
        s = sma_fn(f, di)
        c = f[di]
        if s is None or not np.isfinite(c):
            return s if s is not None else (float(c) if np.isfinite(c) else None)
        return weight_current * float(c) + (1 - weight_current) * s
    return sm


CONTENDERS = [
    ("current (incumbent)",  current, None),
    ("SMA 6M",               trailing(6), None),
    ("SMA 12M",              trailing(12), None),
    ("SMA 18M",              trailing(18), None),
    ("SMA 24M",              trailing(24), None),
    ("EMA ~6M",              ema(6), None),
    ("EMA ~12M",             ema(12), None),
    ("hybrid 6M+0.25curr",   hybrid(6, 0.25), None),
    ("hybrid 6M+0.5curr",    hybrid(6, 0.5), None),
]

print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")
print(f"\nSPY: CAGR {spy_m['cagr']*100:+.2f}%  Sharpe {spy_m['sharpe']:.2f}")

print("\n" + "=" * 100)
print(f"{'variant':25s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Final':>13s}  {'Δ vs incumb':>11s}")
print("-" * 100)

runs = {}
incumbent_cagr = None
rows = []
for label, sf, pm in CONTENDERS:
    r = simulate_smoothed(md, md.stocks, cfg, sf, persist_months=pm)
    m = compute_metrics(md, r.equity, r.total_invested)
    runs[label] = r
    if "incumbent" in label:
        incumbent_cagr = m['cagr']
    rows.append((label, m))

for label, m in rows:
    delta = (m['cagr'] - incumbent_cagr) * 100
    print(f"{label:25s}  {m['cagr']*100:+6.2f}%  {m['sharpe']:6.2f}  "
          f"{-m['maxdd']*100:+6.2f}%  ${m['final']:12,.0f}  {delta:+8.2f}pp")


def win_metric(eq, from_i, to_i):
    if from_i >= to_i or to_i > len(eq): return None
    start = max(eq[from_i], 0.01)
    end = eq[to_i - 1]
    if end <= 0: return None
    yrs = (to_i - from_i) / 252
    cagr = (end / start) ** (1 / yrs) - 1
    peak, mdd = start, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak: peak = eq[i]
        if peak > 0:
            d = (peak - eq[i]) / peak
            if d > mdd: mdd = d
    return {"cagr": cagr, "mdd": mdd}


# Rolling 10Y
print("\n## Rolling 10Y CAGR (step 1Y)")
WIN_D = 10 * 252; STEP_D = 1 * 252
print(f"{'window':21s}", end="")
for label, _, _ in CONTENDERS:
    print(f"  {label[:13]:>13s}", end="")
print()

stats = {label: [] for label, _, _ in CONTENDERS}
for from_i in range(0, len(md.all_dates) - WIN_D, STEP_D):
    to_i = from_i + WIN_D
    d_from = md.all_dates[from_i][:7]
    d_to = md.all_dates[to_i - 1][:7]
    print(f"{d_from} → {d_to}", end="")
    for label, _, _ in CONTENDERS:
        m = win_metric(runs[label].equity, from_i, to_i)
        if m:
            stats[label].append(m['cagr'])
            print(f"  {m['cagr']*100:+12.2f}%", end="")
        else:
            print(f"  {'—':>13s}", end="")
    print()

print(f"\n{'variant':25s}  {'median':>8s}  {'mean':>7s}  {'wins vs incumb':>14s}")
base_cagrs = stats.get("current (incumbent)", [])
for label, _, _ in CONTENDERS:
    cs = stats[label]
    if not cs: continue
    wins = sum(1 for a, b in zip(cs, base_cagrs) if a > b) if "incumbent" not in label else len(cs)
    print(f"{label:25s}  {np.median(cs)*100:+7.2f}%  {np.mean(cs)*100:+6.2f}%  {wins:4d}/{len(cs)}")


# Sensitivity: top_n × cap on SMA-6M
print("\n## Sensitivity (SMA 6M): top_n × cap")
print(f"{'top_n':>5s}  {'cap=3%':>10s}  {'cap=5%':>10s}  {'cap=7%':>10s}  {'cap=10%':>10s}")
for tn in [3, 5, 7]:
    row = f"{tn:5d}"
    for cap_pct in [3, 5, 7, 10]:
        cfg2 = StrategyConfig(start_month_idx=start_m, top_n=tn,
                              max_ticker_frac=cap_pct / 100.0,
                              hold_days=5000, weighting="rank", entry_delay=1)
        r_inc = simulate_smoothed(md, md.stocks, cfg2, current, None)
        r_sm = simulate_smoothed(md, md.stocks, cfg2, trailing(6), None)
        m_inc = compute_metrics(md, r_inc.equity, r_inc.total_invested)
        m_sm = compute_metrics(md, r_sm.equity, r_sm.total_invested)
        delta = (m_sm['cagr'] - m_inc['cagr']) * 100
        row += f"  {delta:+7.2f}pp"
    print(row)


# Calendar year
print("\n## Calendar year win rate (SMA 6M vs incumbent)")
yr_starts = {}
for i, dd_ in enumerate(md.all_dates):
    yr = dd_[:4]
    if yr not in yr_starts and int(yr) >= 2008:
        yr_starts[yr] = i

yr_keys = sorted(yr_starts.keys())
wins = 0; losses = 0; n_yrs = 0
for i, yr in enumerate(yr_keys[:-1]):
    from_i = yr_starts[yr]
    to_i = yr_starts[yr_keys[i+1]]
    m_inc = win_metric(runs["current (incumbent)"].equity, from_i, to_i)
    m_sm = win_metric(runs["SMA 6M"].equity, from_i, to_i)
    if not (m_inc and m_sm): continue
    n_yrs += 1
    delta = (m_sm['cagr'] - m_inc['cagr']) * 100
    mark = "  +win" if m_sm['cagr'] > m_inc['cagr'] else "  -loss"
    if m_sm['cagr'] > m_inc['cagr']: wins += 1
    else: losses += 1
    print(f"  {yr}  incumbent {m_inc['cagr']*100:+6.2f}%  SMA6M {m_sm['cagr']*100:+6.2f}%  Δ {delta:+5.2f}pp{mark}")

print(f"\nCalendar-year SMA6M wins: {wins}/{n_yrs} ({wins/n_yrs*100:.0f}%)")

print("\n## DONE")
