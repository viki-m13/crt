"""Step 30: final dimensions left untested for CAP5 — weighting,
score_threshold, top_n, and hold_days.

Step20/25/29 thoroughly tested filter overlays (rebound, value, sector,
regime). This step tests the remaining CAP5 internal knobs:

  - weighting: rank vs score vs equal
  - score_threshold (raise the floor on `final` to require min conviction)
  - top_n (3, 4, 5, 6, 7, 8)
  - hold_days (forced rotation: 1y, 2y, 3y, 5y vs hold-forever 5000)
  - max_ticker_frac (1%, 3%, 5% with a no-cap baseline for comparison)

Goal: find ANY variant that beats CAP5 (rank/5%/top5/forever) on 20Y CAGR
without losing recent 1Y or paying a structural rolling-window penalty.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics,
                     StrategyConfig)
from bt_core_ext import load_and_prep_ext

md, start_m = load_and_prep_ext()
TOTAL_M = len(md.month_first_idx)
TO = len(md.all_dates)


def win_cagr_mdd(eq, invested, from_i, to_i):
    yrs = (to_i - from_i) / 252
    if invested <= 0 or yrs <= 0:
        return None, None
    final = eq[to_i - 1]
    if final <= 0:
        return -1.0, 1.0
    cagr = (final / invested) ** (1 / yrs) - 1
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    return cagr, mdd


CAP5_DEFAULT = dict(top_n=5, max_ticker_frac=0.05, hold_days=5000,
                    weighting="rank", entry_delay=1)

variants = [
    ("CAP5 (baseline)        ", dict(**CAP5_DEFAULT)),

    # Weighting axis
    ("weighting=score        ", {**CAP5_DEFAULT, "weighting": "score"}),
    ("weighting=equal        ", {**CAP5_DEFAULT, "weighting": "equal"}),

    # Score threshold (require min final score)
    ("score_threshold=5      ", {**CAP5_DEFAULT, "score_threshold": 5.0}),
    ("score_threshold=10     ", {**CAP5_DEFAULT, "score_threshold": 10.0}),
    ("score_threshold=15     ", {**CAP5_DEFAULT, "score_threshold": 15.0}),
    ("score_threshold=20     ", {**CAP5_DEFAULT, "score_threshold": 20.0}),

    # min_score (different from threshold — set a floor that excludes very low scores)
    ("min_score=1            ", {**CAP5_DEFAULT, "min_score": 1.0}),
    ("min_score=5            ", {**CAP5_DEFAULT, "min_score": 5.0}),
    ("min_score=10           ", {**CAP5_DEFAULT, "min_score": 10.0}),

    # top_n axis
    ("top_n=3                ", {**CAP5_DEFAULT, "top_n": 3}),
    ("top_n=4                ", {**CAP5_DEFAULT, "top_n": 4}),
    ("top_n=6                ", {**CAP5_DEFAULT, "top_n": 6}),
    ("top_n=7                ", {**CAP5_DEFAULT, "top_n": 7}),
    ("top_n=8                ", {**CAP5_DEFAULT, "top_n": 8}),
    ("top_n=10               ", {**CAP5_DEFAULT, "top_n": 10}),

    # hold_days axis (forced rotation — tax events ignored)
    ("hold=1y                ", {**CAP5_DEFAULT, "hold_days": 252}),
    ("hold=2y                ", {**CAP5_DEFAULT, "hold_days": 504}),
    ("hold=3y                ", {**CAP5_DEFAULT, "hold_days": 756}),
    ("hold=5y                ", {**CAP5_DEFAULT, "hold_days": 1260}),

    # Cap fraction axis
    ("cap=1%                 ", {**CAP5_DEFAULT, "max_ticker_frac": 0.01}),
    ("cap=2%                 ", {**CAP5_DEFAULT, "max_ticker_frac": 0.02}),
    ("cap=3%                 ", {**CAP5_DEFAULT, "max_ticker_frac": 0.03}),
    ("cap=4%                 ", {**CAP5_DEFAULT, "max_ticker_frac": 0.04}),
    ("cap=7%                 ", {**CAP5_DEFAULT, "max_ticker_frac": 0.07}),
    ("cap=10%                ", {**CAP5_DEFAULT, "max_ticker_frac": 0.10}),
    ("no cap                 ", {**CAP5_DEFAULT, "max_ticker_frac": None}),
]


print("## 1. Headline 20Y")
bench_full = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, bench_full.equity, bench_full.total_invested)
print(f"  SPY DCA              CAGR {bm['cagr']*100:+6.2f}%  MaxDD {bm['maxdd']*100:+6.2f}%  Sharpe {bm['sharpe']:.2f}")

headline = {}
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m['cagr'] - bm['cagr']) * 100
    calmar = m['cagr'] / m['maxdd'] if m['maxdd'] > 0 else 0
    print(f"  {label} CAGR {m['cagr']*100:+6.2f}% ({ex:+5.2f}pp)  "
          f"MaxDD {m['maxdd']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"Calmar {calmar:.2f}")
    headline[label.strip()] = dict(cagr=m['cagr'], mdd=m['maxdd'],
                                   sharpe=m['sharpe'])


print("\n## 2. Trailing 1Y (last 12 months)")
START_1Y = TOTAL_M - 12
FROM_1Y = md.month_first_idx[START_1Y]
b1 = simulate_benchmark(md, ["SPY"], 5000, START_1Y, entry_delay=1)
bc1, bd1 = win_cagr_mdd(b1.equity, 1000 * 12, FROM_1Y, TO)
print(f"  SPY DCA              CAGR {bc1*100:+6.2f}%  MaxDD {bd1*100:+6.2f}%")
trail_1y = {}
for label, k in variants:
    cfg = StrategyConfig(start_month_idx=START_1Y, **k)
    r = simulate(md, md.stocks, cfg)
    c, d = win_cagr_mdd(r.equity, r.total_invested, FROM_1Y, TO)
    ex = (c - bc1) * 100
    print(f"  {label} CAGR {c*100:+6.2f}% ({ex:+5.2f}pp)  MaxDD {d*100:+6.2f}%")
    trail_1y[label.strip()] = c


print("\n## 3. Candidates that beat CAP5 on BOTH 20Y CAGR AND 1Y")
cap5_h = headline["CAP5 (baseline)"]
cap5_1y = trail_1y["CAP5 (baseline)"]
print(f"  Reference: CAP5 20Y {cap5_h['cagr']*100:+.2f}%  Sharpe {cap5_h['sharpe']:.2f}  MaxDD {cap5_h['mdd']*100:+.2f}%  1Y {cap5_1y*100:+.2f}%")
print()
n_winners = 0
for label, _ in variants:
    nm = label.strip()
    if nm == "CAP5 (baseline)":
        continue
    h = headline[nm]
    one_y = trail_1y[nm]
    if h["cagr"] > cap5_h["cagr"] and one_y > cap5_1y:
        d_cagr = (h["cagr"] - cap5_h["cagr"]) * 100
        d_1y = (one_y - cap5_1y) * 100
        d_sharpe = h["sharpe"] - cap5_h["sharpe"]
        d_mdd = (h["mdd"] - cap5_h["mdd"]) * 100
        print(f"  CANDIDATE: {label} 20Y {d_cagr:+.2f}pp  1Y {d_1y:+.2f}pp  "
              f"Sharpe {d_sharpe:+.2f}  MaxDD {d_mdd:+.2f}pp")
        n_winners += 1

if n_winners == 0:
    print("  None. CAP5 dominates the simple-axis space.")


print("\n## 4. Rolling 10Y windows — vs CAP5 on CAGR")
windows = []
s_m = 0
while s_m + 120 <= TOTAL_M:
    windows.append((s_m, s_m + 120))
    s_m += 12

window_cagrs = {label.strip(): [] for label, _ in variants}
for s_m, e_m in windows:
    from_i = md.month_first_idx[s_m]
    to_i = md.month_first_idx[e_m] if e_m < TOTAL_M else TO
    nm = e_m - s_m
    for label, k in variants:
        cfg = StrategyConfig(start_month_idx=s_m, **k)
        r = simulate(md, md.stocks, cfg)
        c, _ = win_cagr_mdd(r.equity, r.total_invested, from_i, to_i)
        window_cagrs[label.strip()].append(c)

cap5 = window_cagrs["CAP5 (baseline)"]
print(f"  {'variant':24s}  wins/11   med Δpp   worst Δpp")
for label, _ in variants:
    nm = label.strip()
    if nm == "CAP5 (baseline)":
        print(f"  {label}  --       {0.0:+.2f}    {0.0:+.2f}")
        continue
    diffs = [(c - c5) * 100 for c, c5 in zip(window_cagrs[nm], cap5)]
    wins = sum(1 for d in diffs if d > 0)
    med = sorted(diffs)[len(diffs) // 2]
    worst = min(diffs)
    print(f"  {label}  {wins:>2d}/11    {med:+.2f}    {worst:+.2f}")

print("\n## DONE")
