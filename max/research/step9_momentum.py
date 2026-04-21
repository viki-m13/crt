"""Step 9: rank by score MOMENTUM (recent score - older score) instead of level."""
import numpy as np
import math
from bt_core import (load_and_prep, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()


def run_with_score_fn(score_fn, top_n=5, hold=5000):
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
            sv = score_fn(tk, di)
            p = md.prices[tk][di]
            if not (math.isfinite(sv) and math.isfinite(p) and p > 0):
                continue
            cand.append((tk, sv, p))
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[:top_n]
        if not picks:
            continue
        raw = [1.0 / (i + 1) for i in range(len(picks))]
        sw = sum(raw); weights = [r / sw for r in raw]
        adj = []; adj_w = []
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


print("## Score momentum: rank by f[di] - f[di-W]")
for W in [10, 30, 60, 120]:
    def fn(tk, di, W=W):
        f = md.finals[tk]
        if di - W < 0:
            return float("nan")
        a, b = f[di], f[di - W]
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan")
        return a - b
    m = run_with_score_fn(fn)
    if not m:
        print(f"  W={W} | (no data)"); continue
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  delta_W={W:3d} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
print()
print("## Score momentum: rank by ratio f[di] / f[di-W]")
for W in [10, 30, 60, 120]:
    def fn(tk, di, W=W):
        f = md.finals[tk]
        if di - W < 0:
            return float("nan")
        a, b = f[di], f[di - W]
        if not (math.isfinite(a) and math.isfinite(b) and b > 1e-9):
            return float("nan")
        return a / b
    m = run_with_score_fn(fn)
    if not m:
        print(f"  W={W} | (no data)"); continue
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  ratio_W={W:3d} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
print()
print("## Combined: rank by f[di] * (f[di]/f[di-W]) — level boosted by momentum")
for W in [10, 30, 60, 120]:
    def fn(tk, di, W=W):
        f = md.finals[tk]
        if di - W < 0:
            return f[di] if math.isfinite(f[di]) else float("nan")
        a, b = f[di], f[di - W]
        if not (math.isfinite(a) and math.isfinite(b) and b > 1e-9):
            return float("nan")
        return a * (a / b)
    m = run_with_score_fn(fn)
    if not m:
        print(f"  W={W} | (no data)"); continue
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"  level*ratio_W={W:3d} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
