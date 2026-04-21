"""Step 7: point-in-time trend-quality filter.

Quality proxy = fraction of last N days where price > 200-day SMA (computed
strictly from the ticker's own prices, no look-ahead). We filter candidates
at each rebalance date to those with quality_proxy >= threshold.
"""
import numpy as np
import math
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig, SECTOR_MAP)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()

# Build rolling quality arrays per ticker
N_QUALITY_WINDOW = 252   # look back 1Y to measure time-above-200SMA
SMA_WINDOW = 200

def rolling_quality(price: np.ndarray) -> np.ndarray:
    n = len(price)
    sma = np.full(n, np.nan)
    for i in range(n):
        if i + 1 >= SMA_WINDOW:
            w = price[i + 1 - SMA_WINDOW : i + 1]
            if np.isfinite(w).all():
                sma[i] = float(w.mean())
    above = (price > sma).astype(float)
    valid = np.isfinite(sma) & np.isfinite(price)
    quality = np.full(n, np.nan)
    # Require at least 100 bars of valid data in the window
    min_bars = 100
    for i in range(n):
        start = max(0, i + 1 - N_QUALITY_WINDOW)
        mask = valid[start : i + 1]
        a = above[start : i + 1][mask]
        if a.size >= min_bars:
            quality[i] = float(a.mean())
    return quality

qualities = {tk: rolling_quality(md.prices[tk]) for tk in md.stocks}


def run_with_quality_filter(threshold: float) -> dict:
    # Build a simulate-like run using the quality filter at each rebalance
    from bt_core import DCA_MONTHLY, TRADING_DAYS_YR
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    cash = 0.0
    total_invested = 0.0
    for m in range(start_m, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        entry_idx = di + 1
        if entry_idx >= n:
            continue
        cand = []
        for tk in md.stocks:
            q = qualities[tk][di]
            # Neutral default (0.5) when quality can't be computed — don't discard.
            if not math.isfinite(q):
                q = 0.5
            if q < threshold:
                continue
            f = md.finals.get(tk)[di]; p = md.prices.get(tk)[di]
            if not (math.isfinite(f) and f > 0 and math.isfinite(p) and p > 0):
                continue
            cand.append((tk, f, p))
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[:5]
        if not picks:
            continue
        raw = [1.0 / (i + 1) for i in range(len(picks))]
        sw = sum(raw); weights = [r / sw for r in raw]
        adj = []
        adj_w = []
        for (tk, fv, _), w in zip(picks, weights):
            px = md.prices[tk][entry_idx]
            if math.isfinite(px) and px > 0:
                adj.append((tk, fv, px)); adj_w.append(w)
        if not adj:
            continue
        sw = sum(adj_w); adj_w = [w / sw for w in adj_w]
        total_invested += DCA_MONTHLY
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = DCA_MONTHLY * w
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + 5000,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv,
                "sold": False, "sell_price": 0.0, "peak": pv,
            })
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


print("## Quality threshold (Top-5 rank, hold-forever)")
print(f"  Quality proxy = fraction of last 504 bars where close > 200SMA (point-in-time)")
for thr in [0.0, 0.30, 0.45, 0.55, 0.65, 0.75]:
    m = run_with_quality_filter(thr)
    if not m:
        print(f"  thr={thr:.2f} | (no data)")
        continue
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  thr={thr:.2f} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
