"""Reproduce the CreditFloor v3 ("Sigma-Clear") validation tables from
the deep replay row table. See VALIDATION.md for the protocol and
replay.py for how the rows are generated.

Asserts the design-window invariant (2008-2018: zero losses) and
prints the untouched-validation (2019-2026) and P&L tables. Exits
non-zero if the design invariant is broken (e.g. after a replay regen
on new data the frozen rule no longer holds in-design — that must be
investigated, not shipped).

Inputs (see VALIDATION.md §7 for how to build them):
    results/replay_rows_full.csv.gz   deep replay, full-history panel
    results/optionable.json           listed-options map
    cache_full/*.json                 full-history panel (for listing dates)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

from replay_analysis import HERE, attach_pricing, load_rows
from research import CAP_LONG, CAP_SHORT, HIST_CLEAR, K_SIGMA
from pricing import MIN_TRADEABLE_FILL

DESIGN_END = "2018-12-31"
ROWS = os.path.join(HERE, "results", "replay_rows_full.csv.gz")


def ticker_start_dates() -> dict[str, str]:
    starts = {}
    for p in glob.glob(os.path.join(HERE, "cache_full", "*.json")):
        t = os.path.basename(p)[:-5]
        with open(p) as fh:
            head = fh.read(200)
        m = re.search(r'"dates": \["(\d{4}-\d{2}-\d{2})"', head)
        if m:
            starts[t] = m.group(1)
    return starts


def main() -> int:
    R = load_rows(ROWS)
    res = R["win"] >= 0
    is_put = R["side"] == "put"
    h = R["horizon"]
    sig_d = R["sigma60"] / np.sqrt(252)
    move = R["move"]
    hist = R["hist_max"]
    year = np.array([d[:4] for d in R["date"]])

    with open(os.path.join(HERE, "results", "optionable.json")) as fh:
        opt = json.load(fh)["optionable"]
    optionable = np.array([opt.get(t, False) for t in R["ticker"]])
    starts = ticker_start_dates()
    start_arr = np.array([starts.get(t, "2026-01-01") for t in R["ticker"]])
    years10 = (R["date"].astype("datetime64[D]")
               - start_arr.astype("datetime64[D]")).astype(int) >= 3652

    # Frozen v3 publication layer (must match research.py constants).
    caps = np.where(h >= 42, CAP_LONG, CAP_SHORT)
    b = K_SIGMA * sig_d * np.sqrt(h) + 0.01
    okb = np.isfinite(b) & (b <= caps) & (b >= HIST_CLEAR * hist)
    Rt = dict(R)
    Rt["buffer"] = b
    attach_pricing(Rt, iv_mult=1.30)
    pub = optionable & years10 & okb & (Rt["net"] >= MIN_TRADEABLE_FILL)
    win = np.where(is_put, move >= -b, move <= b)

    # Trade economics, 1 contract per published rung.
    S, width, net = R["spot"], Rt["width"], Rt["net"]
    K_s = np.where(is_put, S * (1 - b), S * (1 + b))
    S_T = R["close_at_expiry"]
    intr = np.where(is_put,
                    np.minimum(np.maximum(K_s - S_T, 0), width),
                    np.minimum(np.maximum(S_T - K_s, 0), width))
    pnl = (net - intr) * 100.0

    # Dedup: one trade per (ticker, side, horizon, expiry), first publish.
    order = np.lexsort((R["date"],))
    seen: set = set()
    keep = np.zeros(len(S), bool)
    for i in order:
        if not (pub[i] and res[i]):
            continue
        key = (R["ticker"][i], R["side"][i], int(h[i]), R["expiry_date"][i])
        if key in seen:
            continue
        seen.add(key)
        keep[i] = True

    design = R["date"] <= DESIGN_END
    fail = 0
    for lbl, mask in (("ALL RUNGS", pub & res), ("DEDUPED TRADES", keep)):
        print(f"=== {lbl} ===")
        print(f"{'year':>5} {'n':>6} {'losses':>6} {'credit$':>9} {'pnl$':>9}")
        for y in sorted(set(year[mask])):
            mm = mask & (year == y)
            nl = int((intr[mm] > 0).sum())
            print(f"{y:>5} {int(mm.sum()):>6} {nl:>6} "
                  f"{net[mm].sum()*100:>9.0f} {pnl[mm].sum():>9.0f}")
        md, mv = mask & design, mask & ~design
        ld, lv = int((intr[md] > 0).sum()), int((intr[mv] > 0).sum())
        print(f"  design 2008-2018:    n={int(md.sum()):>5} losses={ld} "
              f"pnl=${pnl[md].sum():.0f}")
        print(f"  validation 2019-26:  n={int(mv.sum()):>5} losses={lv} "
              f"pnl=${pnl[mv].sum():.0f}  "
              f"win_rate={float((intr[mv] == 0).mean())*100:.2f}%")
        if ld != 0:
            print("  !! DESIGN-WINDOW INVARIANT BROKEN (expected 0 losses)")
            fail = 1
        print()
    return fail


if __name__ == "__main__":
    sys.exit(main())
