"""Step 39f: does smoothing + crisis dd-scaling stack?

Two robust findings to date:
- Signal smoothing (SMA 12-24M) → +0.9-1.2pp CAGR, robust across windows
- DCA scaling @ SPY -20% → +1-2pp CAGR, GFC-dependent but helps in crises

Test whether SMA 12M + 3x@-20dd adds alphas, or whether the smoothed
signal already captures the GFC-recovery gain from dd-scaling.

Extends step38d to add a smoothing_months variant of dd-scaling.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate, compute_metrics, DCA_MONTHLY, StrategyConfig,
                     first_valid_month_idx, SimResult)
from bt_core_ext import load_and_prep_ext


md, start_m = load_and_prep_ext()

# Precompute SPY drawdown
spy = md.bench_filled["SPY"]
pk = 0.0
spy_peak = np.zeros_like(spy)
for i, v in enumerate(spy):
    if v > pk: pk = v
    spy_peak[i] = pk

def dd(di):
    return 1.0 - spy[di] / spy_peak[di] if spy_peak[di] > 0 else 0.0


def simulate_v2(md, universe, cfg, dca_scaler=None):
    """Bt_core simulate with optional per-month DCA scaler.

    Uses smoothing_months from cfg natively (step 39 integration).
    """
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    picks_by_month = []
    total_invested = 0.0
    cash = 0.0
    smooth_d = cfg.smoothing_months * 21 if cfg.smoothing_months else 0

    eff_universe = [t for t in universe if not cfg.universe_filter or t in cfg.universe_filter]
    if cfg.start_month_idx is None:
        cfg.start_month_idx = first_valid_month_idx(md, eff_universe, min_tickers=3)

    for m in range(cfg.start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        monthly = dca_scaler(di) if dca_scaler else DCA_MONTHLY
        if monthly <= 0: continue

        cand = []
        for tk in eff_universe:
            f = md.finals.get(tk); p = md.prices.get(tk)
            if f is None or p is None: continue
            pv = p[di]
            if not (math.isfinite(pv) and pv > 0): continue
            if smooth_d > 0:
                lo = max(0, di - smooth_d)
                vals = f[lo:di + 1]
                valid = vals[np.isfinite(vals)]
                if len(valid) == 0: continue
                fv = float(np.mean(valid))
            else:
                fv = f[di]
            if not (math.isfinite(fv) and fv > 0): continue
            cand.append((tk, fv, pv))
        if not cand: continue
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[: cfg.top_n]

        if cfg.max_ticker_frac is not None:
            invested_after = total_invested + monthly
            cap_dollars = cfg.max_ticker_frac * invested_after
            by_tk = {}
            for pos in positions:
                by_tk[pos["tk"]] = by_tk.get(pos["tk"], 0.0) + pos["cost"]
            def ok(tk): return by_tk.get(tk, 0.0) < cap_dollars
            filtered = [c for c in picks if ok(c[0])]
            i = len(picks); seen = set(c[0] for c in filtered)
            while len(filtered) < cfg.top_n and i < len(cand):
                c = cand[i]; i += 1
                if c[0] in seen: continue
                if ok(c[0]):
                    filtered.append(c); seen.add(c[0])
            picks = filtered
            if not picks: continue

        if cfg.weighting == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
            s = sum(raw); weights = [r / s for r in raw]
        else:
            weights = [1.0 / len(picks)] * len(picks)

        entry_idx = di + cfg.entry_delay
        if entry_idx >= n: continue
        adj, adj_w = [], []
        for (tk, fv, _), w in zip(picks, weights):
            ep = md.prices.get(tk)
            if ep is None: continue
            px = ep[entry_idx]
            if math.isfinite(px) and px > 0:
                adj.append((tk, fv, px)); adj_w.append(w)
        if not adj: continue
        sw = sum(adj_w)
        if sw <= 0: continue
        adj_w = [w / sw for w in adj_w]

        total_invested += monthly
        picks_by_month.append((md.all_dates[di], [(tk, fv) for tk, fv, _ in adj]))
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = monthly * w
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + cfg.hold_days,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv,
                "sold": False, "sell_price": 0.0, "peak": pv,
            })

    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]: continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px) and px > 0 and px > pos["peak"]:
                pos["peak"] = px
            if d >= pos["sell_idx"] and math.isfinite(px) and px > 0:
                cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px
        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]: continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    return SimResult(equity=equity, total_invested=total_invested,
                     positions=positions, picks_by_month=picks_by_month)


def flat(di): return DCA_MONTHLY
def scale_3x_m20(di): return DCA_MONTHLY * (3.0 if dd(di) >= 0.20 else 1.0)
def scale_2x_m15(di): return DCA_MONTHLY * (2.0 if dd(di) >= 0.15 else 1.0)


cfg_base = StrategyConfig(
    start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
    hold_days=5000, weighting="rank", entry_delay=1,
)


def cfg_with(smooth_m):
    return StrategyConfig(
        start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
        hold_days=5000, weighting="rank", entry_delay=1,
        smoothing_months=smooth_m,
    )


MATRIX = [
    ("baseline (flat, no smooth)",       cfg_with(0),  flat),
    ("flat + SMA 12M",                    cfg_with(12), flat),
    ("flat + SMA 24M",                    cfg_with(24), flat),
    ("3x@-20dd + no smooth",              cfg_with(0),  scale_3x_m20),
    ("3x@-20dd + SMA 12M",                cfg_with(12), scale_3x_m20),
    ("3x@-20dd + SMA 24M",                cfg_with(24), scale_3x_m20),
    ("2x@-15dd + no smooth",              cfg_with(0),  scale_2x_m15),
    ("2x@-15dd + SMA 12M",                cfg_with(12), scale_2x_m15),
]

print(f"\n{'variant':40s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Inv':>10s}  {'Final':>13s}")
print("-" * 100)
runs = {}
for label, cfg, sc in MATRIX:
    r = simulate_v2(md, md.stocks, cfg, sc)
    m = compute_metrics(md, r.equity, r.total_invested)
    runs[label] = (r, m)
    print(f"{label:40s}  {m['cagr']*100:+6.2f}%  {m['sharpe']:6.2f}  "
          f"{-m['maxdd']*100:+6.2f}%  ${m['invested']:9,.0f}  ${m['final']:12,.0f}")


# Rolling 10Y
print("\n## Rolling 10Y CAGR (step 2Y)")
def win_metric(eq, from_i, to_i):
    if from_i >= to_i or to_i > len(eq): return None
    s = max(eq[from_i], 0.01); e = eq[to_i - 1]
    if e <= 0: return None
    yrs = (to_i - from_i) / 252
    return {"cagr": (e / s) ** (1 / yrs) - 1}


WIN_D = 10 * 252
print(f"{'window':21s}", end="")
for label, _, _ in MATRIX:
    print(f"  {label[:15]:>15s}", end="")
print()
win_counts = {label: 0 for label, _, _ in MATRIX}
n_win = 0
base_eq = runs["baseline (flat, no smooth)"][0].equity
for from_i in range(0, len(md.all_dates) - WIN_D, 2 * 252):
    to_i = from_i + WIN_D
    d_from = md.all_dates[from_i][:7]
    d_to = md.all_dates[to_i - 1][:7]
    row = f"{d_from} → {d_to}"
    base_m = win_metric(base_eq, from_i, to_i)
    if not base_m: continue
    n_win += 1
    for label, _, _ in MATRIX:
        m = win_metric(runs[label][0].equity, from_i, to_i)
        if m:
            row += f"  {m['cagr']*100:+14.2f}%"
            if m['cagr'] > base_m['cagr']: win_counts[label] += 1
        else:
            row += f"  {'—':>15s}"
    print(row)
print(f"\nwins vs baseline ({n_win} windows):")
for label, _, _ in MATRIX:
    if "baseline" in label: continue
    print(f"  {label:40s}  {win_counts[label]:3d}/{n_win}")

print("\n## DONE")
