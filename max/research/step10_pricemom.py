"""Step 10: classic 12-1M price momentum factor and combined with final_score."""
import numpy as np
import math
from bt_core import (load_and_prep, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark: {fmt_metrics(bm)}")
print()


def price_momentum(tk: str, di: int, lookback=252, skip=21) -> float:
    p = md.prices[tk]
    if di - lookback < 0:
        return float("nan")
    end = di - skip
    start = di - lookback
    if end < 0 or start < 0:
        return float("nan")
    p1, p2 = p[start], p[end]
    if not (math.isfinite(p1) and math.isfinite(p2) and p1 > 0):
        return float("nan")
    return p2 / p1 - 1


def run_with_score_fn(score_fn, top_n=5, hold=5000):
    n = len(md.all_dates)
    equity = np.zeros(n); positions = []; cash = 0.0; total_invested = 0.0
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


print("## Pure price momentum (no analog signal)")
for lb in [126, 252]:
    for sk in [0, 21]:
        m = run_with_score_fn(lambda tk, di, lb=lb, sk=sk: price_momentum(tk, di, lb, sk))
        excess = (m["cagr"] - bm["cagr"]) * 100
        print(f"  pmom lb={lb} skip={sk} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
print()
print("## Final_score × price-momentum (sign-aware)")
for lb in [126, 252]:
    for sk in [0, 21]:
        def fn(tk, di, lb=lb, sk=sk):
            f = md.finals[tk][di]
            pm = price_momentum(tk, di, lb, sk)
            if not (math.isfinite(f) and math.isfinite(pm)):
                return float("nan")
            # Boost positive momentum stocks, penalize negative
            return f * (1 + pm)
        m = run_with_score_fn(fn)
        excess = (m["cagr"] - bm["cagr"]) * 100
        print(f"  f*(1+pmom_{lb}_{sk}) | {fmt_metrics(m)}  excess {excess:+.2f}pp")
print()
print("## Final_score, but with price momentum > 0 filter")
for lb in [126, 252]:
    for sk in [0, 21]:
        def fn(tk, di, lb=lb, sk=sk):
            f = md.finals[tk][di]
            pm = price_momentum(tk, di, lb, sk)
            if not math.isfinite(f):
                return float("nan")
            if math.isfinite(pm) and pm < 0:
                return float("nan")
            return f
        m = run_with_score_fn(fn)
        excess = (m["cagr"] - bm["cagr"]) * 100
        print(f"  f gated by pmom>0 lb={lb} skip={sk} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
