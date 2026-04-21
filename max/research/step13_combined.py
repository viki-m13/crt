"""Step 13: combine best ideas — finalize the strategy.

Champions to compare:
  - SPY DCA B&H               (benchmark)
  - Top-1 rank hold-forever   (single-pick concentration)
  - Top-3 rank hold-forever   (Sharpe-best B&H)
  - Top-5 rank hold-forever   (more diversified)
  - Top-3 rotation grace=0    (highest CAGR, high vol)
  - Top-5 rotation grace=3    (rotation moderated)
  - Top-3 rank hold + sector-cap-2 (diversification)
"""
import numpy as np
import math
from bt_core import (load_and_prep, simulate, simulate_benchmark, compute_metrics,
                     fmt_metrics, StrategyConfig, DCA_MONTHLY, SECTOR_MAP)

md, start_m = load_and_prep()
b = simulate_benchmark(md, ["SPY"], 5000, start_m, entry_delay=1)
bm = compute_metrics(md, b.equity, b.total_invested)


def run_dca_rotate(top_n=5, weighting="rank", grace=0, sector_cap=None):
    n = len(md.all_dates)
    positions = []
    rebalance_events = []
    for m in range(start_m, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        cand = []
        for tk in md.stocks:
            f = md.finals[tk][di]; p = md.prices[tk][di]
            if not (math.isfinite(f) and f > 0 and math.isfinite(p) and p > 0):
                continue
            cand.append((tk, f, p))
        cand.sort(key=lambda x: x[1], reverse=True)
        # Apply sector cap on the picks
        if sector_cap is not None:
            per_sec = {}
            picks = []
            i = 0
            while i < len(cand) and len(picks) < top_n:
                tk, fv, pv = cand[i]; i += 1
                sec = SECTOR_MAP.get(tk, "Other")
                if per_sec.get(sec, 0) < sector_cap:
                    picks.append((tk, fv, pv))
                    per_sec[sec] = per_sec.get(sec, 0) + 1
        else:
            picks = cand[:top_n]
        rebalance_events.append((di, picks))
    pick_sets = [set(p[0] for p in picks) for di, picks in rebalance_events]

    open_positions = []
    cash = 0.0
    total_invested = 0.0
    for ev_idx, (di, picks) in enumerate(rebalance_events):
        ei = di + 1
        if ei >= n:
            continue
        cur_set = pick_sets[ev_idx]
        new_open = []
        for pos in open_positions:
            if pos["tk"] in cur_set:
                pos["miss"] = 0; new_open.append(pos)
                continue
            pos["miss"] = pos.get("miss", 0) + 1
            if pos["miss"] > grace:
                px = md.prices[pos["tk"]][ei]
                if not (math.isfinite(px) and px > 0):
                    px = pos["buy_price"]
                cash += pos["shares"] * px
                pos["sold_at"] = ei
            else:
                new_open.append(pos)
        open_positions = new_open
        budget = DCA_MONTHLY + cash
        cash = 0.0
        if not picks:
            cash = budget - DCA_MONTHLY; continue
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
            cash = budget - DCA_MONTHLY; continue
        sw = sum(adj_w); adj_w = [w / sw for w in adj_w]
        total_invested += DCA_MONTHLY
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = budget * w
            pos = {"tk": tk, "buy_idx": ei, "shares": alloc / pv, "buy_price": pv,
                   "miss": 0, "sold_at": None}
            open_positions.append(pos); positions.append(pos)

    # Day-by-day equity replay
    inflow_set = set(md.month_first_idx[m] + 1 for m in range(start_m, len(md.month_first_idx))
                     if md.month_first_idx[m] + 1 < n)
    equity = np.zeros(n)
    cash_running = 0.0
    pos_iter = iter(sorted(positions, key=lambda p: p["buy_idx"]))
    next_pos = next(pos_iter, None)
    pos_alive = []
    for d in range(n):
        if d in inflow_set:
            cash_running += DCA_MONTHLY
        while next_pos is not None and next_pos["buy_idx"] == d:
            cash_running -= next_pos["shares"] * next_pos["buy_price"]
            pos_alive.append(next_pos)
            next_pos = next(pos_iter, None)
        for pos in list(pos_alive):
            if pos["sold_at"] == d:
                px = md.prices[pos["tk"]][d]
                if not (math.isfinite(px) and px > 0):
                    px = pos["buy_price"]
                cash_running += pos["shares"] * px
                pos_alive.remove(pos)
        open_val = 0.0
        for pos in pos_alive:
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash_running + open_val
    return compute_metrics(md, equity, total_invested), equity


print("## Final comparison")
print(f"{'Strategy':40s} {'CAGR':>8s} {'TR':>8s} {'Shrp':>5s} {'MaxDD':>7s} {'Final':>10s} {'Excess':>7s}")
def show(name, m):
    excess = (m["cagr"] - bm["cagr"]) * 100
    print(f"{name:40s} {m['cagr']*100:+7.2f}% {m['total_return']*100:+7.2f}% {m['sharpe']:5.2f} {-m['maxdd']*100:+7.2f}% ${m['final']:9,.0f} {excess:+6.2f}pp")

show("SPY DCA buy-and-hold (BENCHMARK)", bm)

# Buy-and-hold variants
for n in [1, 3, 5, 10]:
    cfg = StrategyConfig(top_n=n, hold_days=5000, weighting="rank",
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    show(f"Top-{n} rank, hold-forever (DCA)", m)

# Sector capped
for sc in [1, 2]:
    cfg = StrategyConfig(top_n=5, hold_days=5000, weighting="rank", sector_cap=sc,
                         start_month_idx=start_m, entry_delay=1)
    r = simulate(md, md.stocks, cfg)
    m = compute_metrics(md, r.equity, r.total_invested)
    show(f"Top-5 rank hold-forever, sector_cap={sc}", m)

# Rotation
for n in [3, 5]:
    for g in [0, 1, 3]:
        rm, _ = run_dca_rotate(top_n=n, weighting="rank", grace=g)
        show(f"Rotation Top-{n} rank grace={g}", rm)

# Sector-capped rotation
for n in [5]:
    for g in [3]:
        for sc in [1, 2]:
            rm, _ = run_dca_rotate(top_n=n, weighting="rank", grace=g, sector_cap=sc)
            show(f"Rotation Top-{n} rank grace={g} sec_cap={sc}", rm)
