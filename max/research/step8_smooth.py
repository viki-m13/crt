"""Step 8: rolling-average final_score for ranking (noise reduction).

Instead of ranking by final_score[di], rank by mean(final_score[di-W:di]).
"""
import numpy as np
import math
from bt_core import (load_and_prep, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()

def smoothed_finals(w: int) -> dict:
    """Rolling mean of final_score, NaN-safe, over last w bars."""
    out = {}
    for tk, arr in md.finals.items():
        n = len(arr)
        sm = np.full(n, np.nan)
        for i in range(n):
            if i + 1 < w:
                continue
            window = arr[i + 1 - w : i + 1]
            vs = window[np.isfinite(window)]
            if vs.size >= max(3, w // 2):
                sm[i] = float(vs.mean())
        out[tk] = sm
    return out


def run(smooth_finals, top_n=5, weighting="rank", hold=5000) -> dict:
    n = len(md.all_dates)
    equity = np.zeros(n); positions = []
    cash = 0.0; total_invested = 0.0
    for m in range(start_m, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        ei = di + 1
        if ei >= n:
            continue
        cand = []
        for tk in md.stocks:
            f = smooth_finals[tk][di]
            p = md.prices[tk][di]
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
        adj = []
        adj_w = []
        for (tk, fv, _), w in zip(picks, weights):
            px = md.prices[tk][ei]
            if math.isfinite(px) and px > 0:
                adj.append((tk, fv, px)); adj_w.append(w)
        if not adj:
            continue
        sw = sum(adj_w); adj_w = [w / sw for w in adj_w]
        total_invested += DCA_MONTHLY
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = DCA_MONTHLY * w
            positions.append({
                "tk": tk, "buy_idx": ei, "sell_idx": ei + hold,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv, "sold": False,
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


print("## Score smoothing (Top-5 rank, hold-forever)")
# First, baseline with no smoothing (window=1)
for w in [1, 3, 5, 10, 20, 40, 60]:
    sf = smoothed_finals(w)
    m = run(sf)
    if not m:
        print(f"  smooth_w={w:3d} | (no data)")
        continue
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  smooth_w={w:3d} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
