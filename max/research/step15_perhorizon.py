"""Step 15: per-pick horizon diagnostic (LOOK-AHEAD — not a real strategy).

Use today's bestHorizonFor per ticker and hold each historical buy for
that many trading days. This is deliberately dishonest — the ranking
info is point-in-time but the per-ticker horizon is not. We run it
purely as an upper bound: if look-ahead-optimized per-pick horizons
can't beat hold-forever, the honest version can't either.
"""
import json, math
import numpy as np
from bt_core import (load_and_prep, simulate_benchmark, compute_metrics,
                     fmt_metrics, DCA_MONTHLY, StrategyConfig, simulate)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)
print(f"Benchmark SPY DCA B&H: {fmt_metrics(bm)}\n")

HORIZONS = {"10d": 10, "30d": 30, "60d": 60, "3m": 63, "6m": 126,
            "1y": 252, "3y": 756, "5y": 1260}
full = json.load(open("/home/user/crt/max/docs/data/full.json"))

best = {}  # ticker -> trading days
for it in full["items"]:
    probs = {h: it.get("prob_" + h) for h in HORIZONS}
    best_h, best_p = None, -1.0
    for h, p in probs.items():
        if p is not None and p > best_p:
            best_p, best_h = p, h
    if best_h is not None:
        best[it["ticker"]] = HORIZONS[best_h]


def run_perpick(top_n=5, weighting="rank", fallback=5000):
    """Like bt_core.simulate but each position's hold length comes from
    the per-ticker best horizon (look-ahead)."""
    n = len(md.all_dates)
    positions = []
    total_invested = 0.0
    for m in range(start_m, len(md.month_first_idx)):
        di = md.month_first_idx[m]
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
        total_invested += DCA_MONTHLY
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = DCA_MONTHLY * w
            hold = best.get(tk, fallback)
            positions.append({"tk": tk, "buy_idx": ei, "sell_idx": ei + hold,
                              "shares": alloc / pv, "buy_price": pv, "sold": False})
    # Replay
    equity = np.zeros(n); cash = 0.0
    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            if d >= pos["sell_idx"]:
                px = md.prices[pos["tk"]][d]
                if not (math.isfinite(px) and px > 0):
                    px = pos["buy_price"]
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


print("## Per-pick horizon (look-ahead — upper bound only)")
for n in [1, 3, 5, 10]:
    m = run_perpick(top_n=n, weighting="rank")
    if not m:
        print(f"  Top-{n} | (no data)"); continue
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  Top-{n} rank, per-pick best horizon | {fmt_metrics(m)}  excess {ex:+.2f}pp")

print("\n## Reference: Top-N rank, hold-forever (honest)")
for n in [1, 3, 5, 10]:
    cfg = StrategyConfig(top_n=n, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    ex = (m["cagr"] - bm["cagr"]) * 100
    print(f"  Top-{n} rank, hold-forever         | {fmt_metrics(m)}  excess {ex:+.2f}pp")

# Distribution of hold lengths actually used by the strategy's picks
print("\n## Hold-length distribution across Top-5 picks (per-pick mode)")
from collections import Counter
used = []
for m in range(start_m, len(md.month_first_idx)):
    di = md.month_first_idx[m]
    cand = []
    for tk in md.stocks:
        f = md.finals[tk][di]; p = md.prices[tk][di]
        if math.isfinite(f) and f > 0 and math.isfinite(p) and p > 0:
            cand.append((tk, f))
    cand.sort(key=lambda x: x[1], reverse=True)
    for tk, _ in cand[:5]:
        used.append(best.get(tk, 5000))
c = Counter(used)
tot = sum(c.values())
for h, n in sorted(c.items()):
    pct = n / tot * 100 if tot else 0
    print(f"  hold={h:4d}d: {n:4d} picks ({pct:5.1f}%)")
