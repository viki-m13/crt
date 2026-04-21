"""Step 12: rotation — sell positions when ticker drops out of top-N at next rebalance.

Two variants:
  - "rotate-only" (no new DCA): rebalance fixed equity. Not directly comparable to DCA.
  - "DCA-then-rotate": still DCA $1k/month into current top-N, but ALSO sell any
    holdings whose ticker isn't in this month's top-N. Cash from sales is added to
    this month's DCA pot and reinvested into current top-N.
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


def run_dca_rotate(top_n=5, weighting="rank", grace=0):
    """grace: keep holding a stock for `grace` extra rebalances after it falls out."""
    n = len(md.all_dates)
    equity = np.zeros(n)
    # We track positions per ticker (aggregate shares per ticker), not per buy.
    # Simpler: list of buys (immutable), but we may need to sell partial shares.
    holdings = {}  # tk -> {"shares": x, "miss_count": k}
    last_seen_top = {}  # tk -> last rebalance month index in top-N
    cash = 0.0; total_invested = 0.0
    cur_month = -1

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
        pick_set = {p[0] for p in picks}

        # Sell any holdings not in picks (after grace period)
        sell_tks = []
        for tk, h in holdings.items():
            if tk in pick_set:
                h["miss_count"] = 0
                continue
            h["miss_count"] = h.get("miss_count", 0) + 1
            if h["miss_count"] > grace:
                sell_tks.append(tk)
        for tk in sell_tks:
            px = md.prices[tk][ei]
            if not (math.isfinite(px) and px > 0):
                # Can't sell at next-day open — fall back to today
                px = md.prices[tk][di]
            if math.isfinite(px) and px > 0:
                cash += holdings[tk]["shares"] * px
            del holdings[tk]

        # Allocate this month's DCA + freed cash into current top-N
        # (Don't redeploy ALL cash — only this month's $1000 + sale proceeds of last cycle)
        # Actually let's redeploy ALL cash to keep capital fully invested.
        budget = DCA_MONTHLY + cash
        cash = 0.0

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
            cash = budget - DCA_MONTHLY  # restore freed cash
            continue
        sw = sum(adj_w); adj_w = [w / sw for w in adj_w]
        total_invested += DCA_MONTHLY
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = budget * w
            shares_to_add = alloc / pv
            if tk in holdings:
                holdings[tk]["shares"] += shares_to_add
            else:
                holdings[tk] = {"shares": shares_to_add, "miss_count": 0}

    # Compute equity over time. We need to roll forward: simulate from beginning.
    # Easier: re-track holdings day-by-day. But the loop above already mutated state
    # without recording per-day. Let's redo with proper buy/sell records.
    # SIMPLIFY: let me re-implement using per-buy record.
    return None  # placeholder


def run_dca_rotate_v2(top_n=5, weighting="rank", grace=0):
    """Re-implementation with proper day-by-day equity tracking using buy/sell records."""
    n = len(md.all_dates)
    # Each position is a separate buy with shares; we mark sold on rebalance.
    positions = []  # list of {tk, buy_idx, shares, sold, sell_idx}
    cash = 0.0; total_invested = 0.0
    miss_count = {}  # ticker -> consecutive misses

    rebalance_events = []  # (di, picks_set, picks_with_weights)
    for m in range(start_m, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        cand = []
        for tk in md.stocks:
            f = md.finals[tk][di]; p = md.prices[tk][di]
            if not (math.isfinite(f) and f > 0 and math.isfinite(p) and p > 0):
                continue
            cand.append((tk, f, p))
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[:top_n]
        rebalance_events.append((di, picks))

    n_events = len(rebalance_events)
    # Pre-compute pick sets per event
    pick_sets = [set(p[0] for p in picks) for di, picks in rebalance_events]

    # Helper: find last buy of a ticker (alive)
    open_positions = []  # list ordered by buy time

    for ev_idx, (di, picks) in enumerate(rebalance_events):
        ei = di + 1
        if ei >= n:
            continue

        # Sell positions whose ticker no longer in current pick set (after grace)
        cur_set = pick_sets[ev_idx]
        new_open = []
        for pos in open_positions:
            tk = pos["tk"]
            if tk in cur_set:
                pos["miss"] = 0
                new_open.append(pos)
                continue
            pos["miss"] = pos.get("miss", 0) + 1
            if pos["miss"] > grace:
                # Sell at ei
                px = md.prices[tk][ei]
                if not (math.isfinite(px) and px > 0):
                    px = pos["buy_price"]
                cash += pos["shares"] * px
                pos["sold_at"] = ei
            else:
                new_open.append(pos)
        open_positions = new_open

        # Buy: budget = $1000 + cash from sells
        budget = DCA_MONTHLY + cash
        cash = 0.0
        if not picks:
            cash = budget - DCA_MONTHLY
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
            cash = budget - DCA_MONTHLY
            continue
        sw = sum(adj_w); adj_w = [w / sw for w in adj_w]
        total_invested += DCA_MONTHLY
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = budget * w
            pos = {"tk": tk, "buy_idx": ei, "shares": alloc / pv, "buy_price": pv,
                   "miss": 0, "sold_at": None}
            open_positions.append(pos)
            positions.append(pos)

    # Replay day-by-day
    equity = np.zeros(n); cash_running = 0.0
    # We need to recompute cash; do a full pass.
    # To avoid double-counting, we'll rebuild from positions list with buy/sell timestamps.
    # Note: above we mutated cash incrementally. Let's recompute:
    cash_running = 0.0
    invested_so_far = 0.0
    # Order events by chronological buy/sell
    events = []
    for pos in positions:
        events.append((pos["buy_idx"], "buy", pos))
        if pos["sold_at"] is not None:
            events.append((pos["sold_at"], "sell", pos))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "sell" else 1))  # sells first on same day

    # We don't separately track DCA inflows — they're embedded in buys as additional shares.
    # The total_invested is what was deposited externally. Cash from sells is recycled.
    # For day-by-day equity: at each day, sum (open positions value at d's close) + cash_running.
    # Cash_running starts at 0; at sells, cash_running += proceeds; buys consume cash + DCA inflow.
    # External inflows happen at each event's `di` month. Let me just track cash from sells and
    # treat external inflows as separate entries.

    # Identify per-month external inflow dates
    inflow_idx = []
    for ev_idx, (di, picks) in enumerate(rebalance_events):
        ei = di + 1
        if ei < n:
            inflow_idx.append(ei)
    inflow_set = set(inflow_idx)

    # Walk day by day
    cash_running = 0.0
    inflow_amount = DCA_MONTHLY
    pos_alive = []  # positions currently held
    pos_iter = iter(sorted(positions, key=lambda p: p["buy_idx"]))
    next_pos = next(pos_iter, None)
    sold_today = set()
    for d in range(n):
        # Apply inflow
        if d in inflow_set:
            cash_running += inflow_amount
        # Apply buys at d
        while next_pos is not None and next_pos["buy_idx"] == d:
            cost = next_pos["shares"] * next_pos["buy_price"]
            cash_running -= cost
            pos_alive.append(next_pos)
            next_pos = next(pos_iter, None)
        # Apply sells at d
        for pos in list(pos_alive):
            if pos["sold_at"] == d:
                px = md.prices[pos["tk"]][d]
                if not (math.isfinite(px) and px > 0):
                    px = pos["buy_price"]
                cash_running += pos["shares"] * px
                pos_alive.remove(pos)
        # Mark to market
        open_val = 0.0
        for pos in pos_alive:
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash_running + open_val
    return compute_metrics(md, equity, total_invested)


print("## DCA + Rotation: sell when no longer in top-N")
for n_pick in [3, 5, 10]:
    for grace in [0, 1, 3]:
        m = run_dca_rotate_v2(top_n=n_pick, weighting="rank", grace=grace)
        if not m:
            print(f"  top-{n_pick} grace={grace} | (no data)"); continue
        excess = (m["cagr"] - bm["cagr"]) * 100
        print(f"  top-{n_pick} grace={grace} | {fmt_metrics(m)}  excess {excess:+.2f}pp")
