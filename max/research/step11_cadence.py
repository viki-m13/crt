"""Step 11: rebalance cadence (weekly vs biweekly vs monthly)."""
import numpy as np
import math
from bt_core import (load_and_prep, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()


def first_indices_every(period_bars: int, start_di: int):
    """Indices spaced ~period_bars apart, starting at start_di."""
    n = len(md.all_dates)
    out = []
    di = start_di
    while di < n:
        out.append(di)
        di += period_bars
    return out


def run_cadence(cadence_label: str, indices, dca_per_period: float, top_n=5, hold=5000, weighting="rank"):
    n = len(md.all_dates)
    equity = np.zeros(n); positions = []; cash = 0.0; total_invested = 0.0
    for di in indices:
        ei = di + 1
        if ei >= n:
            continue
        cand = []
        for tk in md.stocks:
            f = md.finals[tk][di]; p = md.prices[tk][di]
            if not (math.isfinite(f) and f > 0 and math.isfinite(p) and p > 0):
                continue
            cand.append((tk, f, p))
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[:top_n]
        if not picks:
            continue
        if weighting == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
        else:
            raw = [1.0] * len(picks)
        sw = sum(raw); weights = [r / sw for r in raw]
        adj = []; adj_w = []
        for (tk, fv, _), w in zip(picks, weights):
            px = md.prices[tk][ei]
            if math.isfinite(px) and px > 0:
                adj.append((tk, fv, px)); adj_w.append(w)
        if not adj:
            continue
        sw = sum(adj_w); adj_w = [w / sw for w in adj_w]
        total_invested += dca_per_period
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = dca_per_period * w
            positions.append({"tk": tk, "buy_idx": ei, "sell_idx": ei + hold,
                              "shares": alloc / pv, "cost": alloc, "buy_price": pv, "sold": False})
    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            if d >= pos["sell_idx"]:
                px = md.prices[pos["tk"]][d]
                if math.isfinite(px) and px > 0:
                    cash += pos["shares"] * px; pos["sold"] = True
        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    return compute_metrics(md, equity, total_invested)


# Total invested should be ~$60k either way: monthly $1000 * 60 months ≈ same as
# weekly $230 * 260 weeks. Use rate $1000/month -> per-period = 1000 * (period_bars/21)
print("## Rebalance cadence (Top-5 rank, hold-forever, total ~same DCA $/year)")
start_di = md.month_first_idx[start_m]
for label, period_bars in [("monthly", 21), ("biweekly", 10), ("weekly", 5)]:
    idx = first_indices_every(period_bars, start_di)
    per = DCA_MONTHLY * (period_bars / 21.0)
    m = run_cadence(label, idx, per)
    if not m:
        print(f"  {label} | (no data)"); continue
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  {label:9s} ({len(idx)} buys @ ${per:.0f}) | {fmt_metrics(m)}  excess {excess:+.2f}pp")
