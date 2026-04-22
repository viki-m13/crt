"""Step 33: walk-forward adaptive CAP5.

Parameter tuning on the FULL 20Y period is vulnerable to regime-specific
overfitting. This step lets CAP5 adapt its parameters each year based
on trailing 5Y performance, without using future information.

Protocol:
  - Each January, look at all CAP5 variants (cap × top_n) and pick the
    one with best trailing 5Y CAGR starting from that rebalance date.
  - Use the winning parameters for the NEXT 12 months.
  - Burn-in: first 5Y uses plain CAP5 (no trailing data yet).

We compare:
  - CAP5 static (cap=5%, top_n=5) — incumbent
  - WF-CAGR: walk-forward chosen on trailing 5Y CAGR
  - WF-Sharpe: walk-forward chosen on trailing 5Y Sharpe
  - WF-Calmar: walk-forward chosen on trailing 5Y Calmar

The walk-forward universe of candidates is the (cap, top_n) grid that
step31 tested, since those are the two dimensions of meaningful variation.

This is STRICTLY point-in-time — at each rebalance, only uses data
BEFORE the rebalance date.
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


def win_cagr_mdd_sharpe(eq, invested, from_i, to_i, md_):
    yrs = (to_i - from_i) / 252
    if invested <= 0 or yrs <= 0:
        return None, None, None, None
    final = eq[to_i - 1]
    if final <= 0:
        return -1.0, 1.0, -999, -999
    cagr = (final / invested) ** (1 / yrs) - 1
    peak, mdd = 0.0, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    # Sharpe: annualized daily returns
    rets = []
    for i in range(from_i + 1, to_i):
        if eq[i - 1] > 0:
            r = eq[i] / eq[i - 1] - 1
            rets.append(r)
    if len(rets) < 10 or np.std(rets) == 0:
        sharpe = 0.0
    else:
        sharpe = np.mean(rets) / np.std(rets) * math.sqrt(252)
    calmar = cagr / mdd if mdd > 0 else 0.0
    return cagr, mdd, sharpe, calmar


# Candidate grid: same as step31
CAPS = [None, 0.05, 0.07, 0.10, 0.15, 0.20]
TOPNS = [3, 4, 5, 6, 7]
CANDIDATES = []
for cap in CAPS:
    for tn in TOPNS:
        CANDIDATES.append(dict(top_n=tn, max_ticker_frac=cap, hold_days=5000,
                               weighting="rank", entry_delay=1))


def label(k):
    c = k.get("max_ticker_frac")
    return f"cap={'none' if c is None else f'{int(c*100)}%'}, top_n={k['top_n']}"


# Precompute the full-period simulation for each candidate — we'll
# slice equity curves to any (from, to) window for trailing-window analysis.
print(f"Precomputing {len(CANDIDATES)} candidate full-period simulations...")
cand_runs = {}
for i, k in enumerate(CANDIDATES):
    cfg = StrategyConfig(start_month_idx=start_m, **k)
    r = simulate(md, md.stocks, cfg)
    cand_runs[label(k)] = r
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(CANDIDATES)}")
print("Done precomputing.")


def trailing_metric(eq, invested_before_window, from_i, to_i, metric="cagr"):
    """Compute trailing metric from equity curve.

    `invested_before_window`: estimated invested at from_i (DCA-adjusted).
    We use a crude proxy: equity[from_i] as the 'starting capital' for
    the trailing window — this treats prior DCA dollars as initial wealth.
    """
    if from_i >= to_i or to_i > len(eq):
        return -999
    start_val = eq[from_i] if eq[from_i] > 0 else 0.01
    end_val = eq[to_i - 1]
    if end_val <= 0:
        return -999
    yrs = (to_i - from_i) / 252
    cagr = (end_val / start_val) ** (1 / yrs) - 1
    if metric == "cagr":
        return cagr
    # Max drawdown over the window
    peak, mdd = start_val, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    if metric == "mdd":
        return -mdd  # minimize MDD -> return negated
    if metric == "calmar":
        return cagr / mdd if mdd > 0 else cagr
    # Sharpe
    rets = []
    for i in range(from_i + 1, to_i):
        if eq[i - 1] > 0:
            r = eq[i] / eq[i - 1] - 1
            rets.append(r)
    if len(rets) < 10 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * math.sqrt(252)


# Walk-forward: at each Jan rebalance, pick the best candidate by metric.
# Burn-in: first 5Y, use CAP5 (cap=5%, top_n=5).
CAP5_LABEL = "cap=5%, top_n=5"

BURNIN_MONTHS = 60  # 5Y


def run_walkforward(metric):
    """Construct composite equity curve using walk-forward chosen params.

    Because DCA + rebalancing with different param sets produces different
    position sets, we can't just stitch equity curves. Instead, we do
    this: at each rebalance month, use the chosen candidate's
    point-in-time decision (which only uses data up to that month) for
    that month's DCA. We simulate the full strategy with param changes.

    Implementation: monthly choice vector → fresh simulation that reads
    the choice at each month. For simplicity here, we use yearly rebalance
    and precompute each year's equity delta using the chosen param.

    That's too complex. Instead, use this pragmatic approximation:
      - At each rebalance, pick the candidate with best trailing metric.
      - The composite equity is the chosen candidate's equity from now on
        (not stitched — we run a full sim per year segment).

    For simpler analysis, we just REPORT which candidate would be chosen
    each year (and its forward 12M performance), and report the ensemble
    EQUAL-WEIGHT of all yearly-chosen-candidates' forward 12M CAGRs.
    """
    print(f"\n## Walk-forward adaptive (metric={metric})")
    # Each rebalance at month m (m = 60, 72, 84, ...)
    yrly_choices = []
    yrly_forward_cagrs = []
    yrly_cap5_forward_cagrs = []
    yrly_windows = []

    for m in range(BURNIN_MONTHS, TOTAL_M - 12, 12):
        trail_from_m = max(0, m - 60)
        trail_from_i = md.month_first_idx[trail_from_m]
        trail_to_i = md.month_first_idx[m]

        # Score each candidate by trailing metric using its precomputed equity curve
        best_score = -999
        best_lbl = CAP5_LABEL
        for k in CANDIDATES:
            lbl = label(k)
            eq = cand_runs[lbl].equity
            s = trailing_metric(eq, None, trail_from_i, trail_to_i, metric=metric)
            if s > best_score:
                best_score = s
                best_lbl = lbl

        # Evaluate forward 12 months of chosen candidate
        fwd_from_i = md.month_first_idx[m]
        fwd_to_i = md.month_first_idx[m + 12] if m + 12 < TOTAL_M else TO
        fwd_eq = cand_runs[best_lbl].equity
        fwd_cagr = trailing_metric(fwd_eq, None, fwd_from_i, fwd_to_i, metric="cagr")
        yrly_forward_cagrs.append(fwd_cagr)
        yrly_choices.append(best_lbl)

        # Compare to CAP5 forward 12M
        cap5_eq = cand_runs[CAP5_LABEL].equity
        cap5_fwd = trailing_metric(cap5_eq, None, fwd_from_i, fwd_to_i, metric="cagr")
        yrly_cap5_forward_cagrs.append(cap5_fwd)
        yrly_windows.append((md.all_dates[fwd_from_i][:7], md.all_dates[fwd_to_i - 1][:7]))

    # Summary
    print(f"  {'year':>20s}  {'chosen param':>22s}  {'fwd 12M':>8s}  {'CAP5 fwd':>8s}  {'Δ':>6s}")
    wins_vs_cap5 = 0
    for (s, e), chosen, fw, cap5_fw in zip(yrly_windows, yrly_choices, yrly_forward_cagrs, yrly_cap5_forward_cagrs):
        delta = (fw - cap5_fw) * 100
        mark = "+" if fw > cap5_fw else " "
        if fw > cap5_fw:
            wins_vs_cap5 += 1
        print(f"  {s}->{e:>7s}  {chosen:>22s}  {fw*100:+7.2f}%  {cap5_fw*100:+7.2f}%  {delta:+5.2f}pp {mark}")

    n = len(yrly_forward_cagrs)
    print(f"\n  Walk-forward won {wins_vs_cap5}/{n} forward 12M windows vs static CAP5")
    # Average log returns (proxy for compounding)
    mean_wf = np.mean([math.log(1 + c) for c in yrly_forward_cagrs])
    mean_cap5 = np.mean([math.log(1 + c) for c in yrly_cap5_forward_cagrs])
    print(f"  Mean log-return: walk-forward {math.exp(mean_wf)-1:+.4f}  vs  CAP5 {math.exp(mean_cap5)-1:+.4f}")
    return yrly_choices, yrly_forward_cagrs, yrly_cap5_forward_cagrs


print("## CAP5 static reference (full 20Y)")
cap5_r = cand_runs[CAP5_LABEL]
m = compute_metrics(md, cap5_r.equity, cap5_r.total_invested)
print(f"  CAGR {m['cagr']*100:+.2f}%  MaxDD {m['maxdd']*100:+.2f}%  Sharpe {m['sharpe']:.2f}")


# Run walk-forward with different selection metrics
for metric in ["cagr", "sharpe", "calmar"]:
    run_walkforward(metric)


print("\n## DONE")
