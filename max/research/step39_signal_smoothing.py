"""Step 39: signal smoothing / persistence filters.

CAP5 uses the current month's `final` score to rank. This creates some
noise: a ticker might be top-ranked this month due to a single strong
reading, then drop out next month. Test whether requiring persistence
or smoothing the signal improves quality.

Variants:
  - current       (incumbent — use today's final)
  - trailing_3M   (use 3-month moving average of final score)
  - trailing_6M   (use 6-month moving average)
  - persist_2M    (require ticker to rank top-10 for 2 consecutive months)

All use CAP5 config otherwise.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate_benchmark, compute_metrics, MarketData,
                     StrategyConfig, DCA_MONTHLY, first_valid_month_idx, SECTOR_MAP)
from bt_core_ext import load_and_prep_ext


def simulate_smoothed(md, universe, cfg, smooth_fn, persist_months=None):
    """Simulate with smoothed score.

    smooth_fn(finals_arr, di) -> float smoothed score at di
    persist_months: if set, require the ticker to have been in top-10
      for the last `persist_months` consecutive months, else skip.
    """
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    picks_by_month = []
    total_invested = 0.0
    cash = 0.0

    eff_universe = [t for t in universe if not cfg.universe_filter or t in cfg.universe_filter]
    if cfg.start_month_idx is None:
        cfg.start_month_idx = first_valid_month_idx(md, eff_universe, min_tickers=3)

    # Precompute per-month candidate rankings (for persistence check)
    monthly_ranks = {}  # m -> {tk: rank}
    if persist_months:
        for m in range(cfg.start_month_idx, len(md.month_first_idx)):
            di = md.month_first_idx[m]
            rs = []
            for tk in eff_universe:
                f = md.finals.get(tk); p = md.prices.get(tk)
                if f is None or p is None: continue
                fv, pv = smooth_fn(f, di), p[di]
                if fv is None or not (math.isfinite(fv) and fv > 0 and
                                        math.isfinite(pv) and pv > 0):
                    continue
                rs.append((tk, fv))
            rs.sort(key=lambda x: x[1], reverse=True)
            monthly_ranks[m] = {tk: i for i, (tk, _) in enumerate(rs[:10])}

    for m in range(cfg.start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        monthly = DCA_MONTHLY

        cand = []
        for tk in eff_universe:
            f = md.finals.get(tk); p = md.prices.get(tk)
            if f is None or p is None: continue
            fv_smooth = smooth_fn(f, di)
            pv = p[di]
            if fv_smooth is None or not (math.isfinite(fv_smooth) and fv_smooth > 0 and
                                            math.isfinite(pv) and pv > 0):
                continue
            if persist_months:
                ok = all(tk in monthly_ranks.get(m - k, {}) for k in range(persist_months))
                if not ok:
                    continue
            cand.append((tk, fv_smooth, pv))
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
                adj.append((tk, fv, px))
                adj_w.append(w)
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
    from bt_core import SimResult
    return SimResult(equity=equity, total_invested=total_invested,
                     positions=positions, picks_by_month=picks_by_month)


md, start_m = load_and_prep_ext()
print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")

# Smoothing functions: f is per-ticker array of finals, di is index
def current(f, di): return f[di]

def trailing(months):
    window_days = months * 21
    def sm(f, di):
        lo = max(0, di - window_days)
        vals = f[lo:di + 1]
        valid = vals[np.isfinite(vals)]
        return float(np.mean(valid)) if len(valid) > 0 else None
    return sm


cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

SPY_cfg = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
spy_m = compute_metrics(md, SPY_cfg.equity, SPY_cfg.total_invested)

CONTENDERS = [
    ("CAP5 current (incumbent)", current, None),
    ("CAP5 trailing 2M avg",     trailing(2), None),
    ("CAP5 trailing 3M avg",     trailing(3), None),
    ("CAP5 trailing 6M avg",     trailing(6), None),
    ("CAP5 + persist_2M",        current, 2),
    ("CAP5 + persist_3M",        current, 3),
    ("CAP5 tr3M + persist_2M",   trailing(3), 2),
]

print("\n" + "=" * 95)
print(f"{'variant':30s}  {'CAGR':>7s}  {'TR':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'Final':>12s}")
print("-" * 95)
print(f"{'SPY':30s}  {spy_m['cagr']*100:+6.2f}%  {spy_m['total_return']*100:+6.2f}%  "
      f"{-spy_m['maxdd']*100:+6.2f}%  {spy_m['sharpe']:6.2f}  ${spy_m['final']:11,.0f}")

rows = []
for label, sf, pm in CONTENDERS:
    r = simulate_smoothed(md, md.stocks, cfg, sf, persist_months=pm)
    m = compute_metrics(md, r.equity, r.total_invested)
    rows.append((label, m, r))
    print(f"{label:30s}  {m['cagr']*100:+6.2f}%  {m['total_return']*100:+6.2f}%  "
          f"{-m['maxdd']*100:+6.2f}%  {m['sharpe']:6.2f}  ${m['final']:11,.0f}")

print("\nRanked by CAGR:")
rows.sort(key=lambda x: -x[1]['cagr'])
base_cagr = next(m['cagr'] for lbl, m, _ in rows if "incumbent" in lbl)
for lbl, m, _ in rows:
    delta = (m['cagr'] - base_cagr) * 100
    mark = " ← incumbent" if "incumbent" in lbl else ""
    print(f"  {lbl:30s}  {m['cagr']*100:+7.2f}%  Δ {delta:+5.2f}pp{mark}")

print("\n## DONE")
