"""Step 38: dynamic DCA sizing based on SPY drawdown.

Classic DCA deploys fixed $1000/month. A "value averaging" or "drawdown-
scaled DCA" deploys MORE when the market is down (buy the dip
systematically). Test whether scaling monthly DCA by SPY drawdown
improves returns.

Variants:
  - flat              (incumbent, $1000/month always)
  - 2x_at_20dd        ($1000 normal, $2000 when SPY ≤20% off high)
  - 3x_at_20dd        ($3000 when SPY ≤20% off high)
  - dd_linear         ($1000 + dd*$5000 proportional, capped at 3x)
  - scale_down_20dd   ($500 when SPY ≤20% off high — the opposite test)

If the market timing helps, dd-scaled variants should beat flat on CAGR
AND Sharpe. If it's a wash, CAGR might tie but Sharpe should improve
from averaging into cheaper prices.

BUT: if you're adding more dollars per month in bear markets, dollar-
weighted returns will look better than time-weighted. We use
total_return and CAGR on invested capital, which is the relevant metric.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, simulate_benchmark, compute_metrics, MarketData,
                     StrategyConfig, DCA_MONTHLY, SECTOR_MAP, first_valid_month_idx, spy_200sma)
from bt_core_ext import load_and_prep_ext


def simulate_dd_scaled(md, universe, cfg, dca_scaler):
    """Same as bt_core.simulate but DCA amount varies by dca_scaler(di)."""
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    picks_by_month = []
    cash = 0.0
    total_invested = 0.0

    eff_universe = [t for t in universe if not cfg.universe_filter or t in cfg.universe_filter]
    if cfg.start_month_idx is None:
        cfg.start_month_idx = first_valid_month_idx(md, eff_universe, min_tickers=3)

    for m in range(cfg.start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        monthly = dca_scaler(di, md)  # scaled DCA
        if monthly <= 0:
            continue

        cand = []
        for tk in eff_universe:
            f = md.finals.get(tk); p = md.prices.get(tk)
            if f is None or p is None:
                continue
            fv, pv = f[di], p[di]
            if not (math.isfinite(fv) and fv > cfg.min_score and math.isfinite(pv) and pv > 0):
                continue
            cand.append((tk, fv, pv))
        if not cand:
            continue
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[: cfg.top_n]

        if cfg.max_ticker_frac is not None:
            invested_after = total_invested + monthly
            cap_dollars = cfg.max_ticker_frac * invested_after
            by_tk = {}
            for pos in positions:
                by_tk[pos["tk"]] = by_tk.get(pos["tk"], 0.0) + pos["cost"]
            def ok(tk):
                return by_tk.get(tk, 0.0) < cap_dollars
            filtered = [c for c in picks if ok(c[0])]
            i = len(picks); seen = set(c[0] for c in filtered)
            while len(filtered) < cfg.top_n and i < len(cand):
                c = cand[i]; i += 1
                if c[0] in seen: continue
                if ok(c[0]):
                    filtered.append(c); seen.add(c[0])
            picks = filtered
            if not picks:
                continue

        if cfg.weighting == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
            s = sum(raw); weights = [r / s for r in raw]
        else:
            weights = [1.0 / len(picks)] * len(picks)

        entry_idx = di + cfg.entry_delay
        if entry_idx >= n:
            continue
        adj_picks, adj_weights = [], []
        for (tk, fv, _), w in zip(picks, weights):
            ep = md.prices.get(tk)
            if ep is None: continue
            px = ep[entry_idx]
            if math.isfinite(px) and px > 0:
                adj_picks.append((tk, fv, px))
                adj_weights.append(w)
        if not adj_picks: continue
        sw = sum(adj_weights)
        if sw <= 0: continue
        adj_weights = [w / sw for w in adj_weights]

        total_invested += monthly
        picks_by_month.append((md.all_dates[di], [(tk, fv) for tk, fv, _ in adj_picks]))
        for (tk, fv, pv), w in zip(adj_picks, adj_weights):
            alloc = monthly * w
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + cfg.hold_days,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv,
                "sold": False, "sell_price": 0.0, "peak": pv,
            })

    # Settle
    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px) and px > 0:
                if px > pos["peak"]:
                    pos["peak"] = px
            if d >= pos["sell_idx"] and math.isfinite(px) and px > 0:
                cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px

        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    from bt_core import SimResult
    return SimResult(equity=equity, total_invested=total_invested,
                     positions=positions, picks_by_month=picks_by_month)


md, start_m = load_and_prep_ext()
print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")

# Precompute SPY peak & drawdown series
spy = md.bench_filled["SPY"]
spy_peak = np.zeros_like(spy)
pk = 0.0
for i, v in enumerate(spy):
    if v > pk: pk = v
    spy_peak[i] = pk

def spy_dd(di):
    if spy_peak[di] <= 0: return 0.0
    return 1.0 - spy[di] / spy_peak[di]

# Scalers
def flat(di, md): return DCA_MONTHLY
def dca_2x_at_20dd(di, md): return DCA_MONTHLY * (2.0 if spy_dd(di) >= 0.20 else 1.0)
def dca_3x_at_20dd(di, md): return DCA_MONTHLY * (3.0 if spy_dd(di) >= 0.20 else 1.0)
def dca_dd_linear(di, md):
    dd = spy_dd(di)
    return DCA_MONTHLY * min(3.0, 1.0 + dd * 5.0)
def dca_scale_down(di, md): return DCA_MONTHLY * (0.5 if spy_dd(di) >= 0.20 else 1.0)

VARIANTS = [
    ("flat (incumbent)    ", flat),
    ("2x at SPY -20%      ", dca_2x_at_20dd),
    ("3x at SPY -20%      ", dca_3x_at_20dd),
    ("linear dd-scaled    ", dca_dd_linear),
    ("0.5x at SPY -20% (downside test)", dca_scale_down),
]

cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

SPY_cfg = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, SPY_cfg.equity, SPY_cfg.total_invested)

print("\n" + "=" * 100)
print(f"{'variant':35s}  {'CAGR':>7s}  {'TR':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'Invested':>12s}  {'Final':>14s}")
print("-" * 100)
print(f"{'SPY':35s}  {spy_m['cagr']*100:+6.2f}%  {spy_m['total_return']*100:+6.2f}%  "
      f"{-spy_m['maxdd']*100:+6.2f}%  {spy_m['sharpe']:6.2f}  ${spy_m['invested']:11,.0f}  ${spy_m['final']:13,.0f}")

for label, scaler in VARIANTS:
    r = simulate_dd_scaled(md, md.stocks, cfg, scaler)
    m = compute_metrics(md, r.equity, r.total_invested)
    print(f"{label:35s}  {m['cagr']*100:+6.2f}%  {m['total_return']*100:+6.2f}%  "
          f"{-m['maxdd']*100:+6.2f}%  {m['sharpe']:6.2f}  ${m['invested']:11,.0f}  ${m['final']:13,.0f}")

print("\nNOTE: variants that deploy extra capital on dips mechanically have MORE invested.")
print("Compare CAGR (IRR on actual cash flows) and final $ to judge real improvement.\n")
print("## DONE")
