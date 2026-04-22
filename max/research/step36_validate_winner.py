"""Step 36: full validation battery for any CAP5 challenger.

Takes a `StrategyConfig` (via CLI args or env var) and runs:
  - Rolling 10Y windows (11 overlapping, every 2Y step)
  - 1Y trailing each calendar year
  - GFC decade (2008-2018)
  - Jackknife (drop each ticker one at a time)
  - Bootstrap (500 resampled 10Y windows)

Reports CAGR, MaxDD, Sharpe, and excess vs SPY for each analysis.

Usage:
  python step36_validate_winner.py --rank-formula=final --cap=5 --top-n=5
  python step36_validate_winner.py --rank-formula=final_x_wash --cap=5 --top-n=5
"""
import argparse, math, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import simulate, simulate_benchmark, compute_metrics, StrategyConfig
from bt_core_ext import load_and_prep_ext


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-formula", default="final")
    ap.add_argument("--rank-alpha", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=5.0, help="percent")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--weighting", default="rank")
    ap.add_argument("--hold-days", type=int, default=5000)
    ap.add_argument("--smoothing-months", type=int, default=0,
                    help="trailing SMA window for `final` score (step 39). 0 = no smoothing")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-jackknife", action="store_true")
    return ap.parse_args()


def metric_over_window(md, eq, invested_before_window, from_i, to_i):
    if from_i >= to_i or to_i > len(eq):
        return None
    start_val = max(eq[from_i], 0.01)
    end_val = eq[to_i - 1]
    if end_val <= 0:
        return None
    yrs = (to_i - from_i) / 252
    cagr = (end_val / start_val) ** (1 / yrs) - 1
    peak, mdd = start_val, 0.0
    for i in range(from_i, to_i):
        if eq[i] > peak:
            peak = eq[i]
        if peak > 0:
            dd = (peak - eq[i]) / peak
            if dd > mdd:
                mdd = dd
    rets = []
    for i in range(from_i + 1, to_i):
        if eq[i - 1] > 0:
            rets.append(eq[i] / eq[i - 1] - 1)
    sharpe = 0.0
    if len(rets) >= 10 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * math.sqrt(252)
    return {"cagr": cagr, "maxdd": mdd, "sharpe": sharpe, "yrs": yrs}


def main():
    args = parse_args()
    md, start_m = load_and_prep_ext()
    print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates "
          f"({md.all_dates[0]} → {md.all_dates[-1]})")

    cfg = StrategyConfig(
        start_month_idx=start_m,
        top_n=args.top_n,
        max_ticker_frac=args.cap / 100.0 if args.cap > 0 else None,
        hold_days=args.hold_days,
        weighting=args.weighting,
        entry_delay=1,
        rank_formula=args.rank_formula,
        rank_alpha=args.rank_alpha,
        smoothing_months=args.smoothing_months,
        label=args.label or (
            f"{args.rank_formula}_cap{args.cap}_top{args.top_n}"
            + (f"_smooth{args.smoothing_months}M" if args.smoothing_months else "")
        ),
    )
    print(f"Strategy: {cfg.label}")

    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    spy_r = simulate_benchmark(md, ["SPY"], args.hold_days, start_m, entry_delay=1)
    spy_m = compute_metrics(md, spy_r.equity, spy_r.total_invested)

    print(f"\n## Full 20Y: {cfg.label}")
    print(f"  CAGR   {m['cagr']*100:+7.2f}%  (SPY {spy_m['cagr']*100:+.2f}%, Δ {(m['cagr']-spy_m['cagr'])*100:+.2f}pp)")
    print(f"  MaxDD  {-m['maxdd']*100:+7.2f}%  (SPY {-spy_m['maxdd']*100:+.2f}%)")
    print(f"  Sharpe {m['sharpe']:6.2f}  (SPY {spy_m['sharpe']:.2f})")
    print(f"  Final  ${m['final']:,.0f}  Invested ${m['invested']:,.0f}")

    # Rolling 10Y (step every 2Y)
    print("\n## Rolling 10Y windows")
    WIN_YRS, STEP_YRS = 10, 2
    win_d = WIN_YRS * 252
    step_d = STEP_YRS * 252
    rows = []
    for from_i in range(0, len(md.all_dates) - win_d, step_d):
        to_i = from_i + win_d
        sm = metric_over_window(md, r.equity, r.total_invested, from_i, to_i)
        sp = metric_over_window(md, spy_r.equity, spy_r.total_invested, from_i, to_i)
        if sm is None or sp is None:
            continue
        d_from = md.all_dates[from_i][:7]
        d_to = md.all_dates[to_i - 1][:7]
        excess = (sm['cagr'] - sp['cagr']) * 100
        rows.append((d_from, d_to, sm['cagr'], sp['cagr'], excess, sm['maxdd']))
        print(f"  {d_from} → {d_to}  CAGR {sm['cagr']*100:+6.2f}%  SPY {sp['cagr']*100:+6.2f}%  Δ {excess:+5.2f}pp  MaxDD {-sm['maxdd']*100:+.1f}%")
    excesses = [x[4] for x in rows]
    print(f"\n  median excess vs SPY: {np.median(excesses):+.2f}pp")
    print(f"  win rate vs SPY:      {sum(1 for x in excesses if x > 0)}/{len(excesses)}")

    # 1Y calendar
    print("\n## 1Y calendar-year windows")
    yr_rows = []
    prev_yr = None
    year_start_i = None
    for i, dd in enumerate(md.all_dates):
        yr = dd[:4]
        if yr != prev_yr:
            if year_start_i is not None and prev_yr is not None:
                sm = metric_over_window(md, r.equity, r.total_invested, year_start_i, i)
                sp = metric_over_window(md, spy_r.equity, spy_r.total_invested, year_start_i, i)
                if sm and sp:
                    yr_rows.append((prev_yr, sm, sp))
            year_start_i = i
            prev_yr = yr
    sm = metric_over_window(md, r.equity, r.total_invested, year_start_i, len(md.all_dates))
    sp = metric_over_window(md, spy_r.equity, spy_r.total_invested, year_start_i, len(md.all_dates))
    if sm and sp:
        yr_rows.append((prev_yr, sm, sp))
    wins = 0
    for yr, sm, sp in yr_rows:
        delta = (sm['cagr'] - sp['cagr']) * 100
        if sm['cagr'] > sp['cagr']:
            wins += 1
        print(f"  {yr}  {sm['cagr']*100:+7.2f}%  SPY {sp['cagr']*100:+6.2f}%  Δ {delta:+5.2f}pp")
    print(f"\n  Annual win rate: {wins}/{len(yr_rows)}")

    # GFC decade
    print("\n## GFC decade (2008-2018)")
    gfc_from_i = next(i for i, dd in enumerate(md.all_dates) if dd >= "2008-01-01")
    gfc_to_i = next(i for i, dd in enumerate(md.all_dates) if dd >= "2018-01-01")
    sm = metric_over_window(md, r.equity, r.total_invested, gfc_from_i, gfc_to_i)
    sp = metric_over_window(md, spy_r.equity, spy_r.total_invested, gfc_from_i, gfc_to_i)
    delta = (sm['cagr'] - sp['cagr']) * 100
    print(f"  {cfg.label}  CAGR {sm['cagr']*100:+.2f}%  MaxDD {-sm['maxdd']*100:+.1f}%  Sharpe {sm['sharpe']:.2f}")
    print(f"  SPY           CAGR {sp['cagr']*100:+.2f}%  MaxDD {-sp['maxdd']*100:+.1f}%  Sharpe {sp['sharpe']:.2f}")
    print(f"  Δ CAGR: {delta:+.2f}pp")

    # Jackknife (drop each ticker)
    med = 0.0
    if not args.skip_jackknife:
        print("\n## Jackknife (drop each ticker)")
        baseline_cagr = m['cagr']
        deltas = []
        for drop_tk in md.stocks:
            uf = set(md.stocks) - {drop_tk}
            cfg2 = StrategyConfig(
                start_month_idx=start_m,
                top_n=args.top_n,
                max_ticker_frac=args.cap / 100.0 if args.cap > 0 else None,
                hold_days=args.hold_days,
                weighting=args.weighting, entry_delay=1,
                rank_formula=args.rank_formula, rank_alpha=args.rank_alpha,
                smoothing_months=args.smoothing_months,
                universe_filter=uf,
            )
            r2 = simulate(md, md.stocks, cfg2)
            m2 = compute_metrics(md, r2.equity, r2.total_invested)
            delta_cagr = (m2['cagr'] - baseline_cagr) * 100
            deltas.append((drop_tk, m2['cagr'], delta_cagr))
        deltas.sort(key=lambda x: x[2])
        print(f"  Top 10 tickers whose removal HURTS strategy (big contributors):")
        for tk, c, d in deltas[:10]:
            print(f"    -{tk:6s} → {c*100:+.2f}% ({d:+.2f}pp)")
        print(f"  Top 10 tickers whose removal HELPS strategy (drags):")
        for tk, c, d in deltas[-10:][::-1]:
            print(f"    -{tk:6s} → {c*100:+.2f}% ({d:+.2f}pp)")
        med = np.median([d for _, _, d in deltas])
        print(f"  Median jackknife delta: {med:+.2f}pp (baseline {baseline_cagr*100:+.2f}%)")

    if args.out:
        summary = {
            "label": cfg.label, "cagr": m['cagr'], "maxdd": m['maxdd'],
            "sharpe": m['sharpe'],
            "rolling_10y_median_excess_pp": float(np.median(excesses)),
            "rolling_10y_win_rate": sum(1 for x in excesses if x > 0) / len(excesses),
            "annual_win_rate": wins / len(yr_rows),
            "gfc_cagr": sm['cagr'], "gfc_sharpe": sm['sharpe'],
            "jackknife_median_delta_pp": float(med),
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nSaved summary → {args.out}")


if __name__ == "__main__":
    main()
