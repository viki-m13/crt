#!/usr/bin/env python3
"""Validation of the muni execution-audit engine (api/_muniaudit.py).

A report that tells an advisor their client lost money has to be right, so
these check the things that would embarrass it: the side convention, the
direction of "cost", the refusal to invent a benchmark, and the dollar
conversion that was already wrong by 100x once in this project.

    python tests/test_muniaudit.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import _muniaudit as M  # noqa: E402

FAIL: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    if cond:
        print(f"  ok    {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label} {detail}")


def prints(*rows):
    return [{"price": p, "par": q, "side": s} for p, q, s in rows]


def main():
    print("=" * 78)
    print("MUNI EXECUTION AUDIT ENGINE")
    print("=" * 78)

    # A day where institutions bought at 101.00 and our client paid 102.00.
    day = prints((101.00, 5_000_000, "S"), (101.05, 2_000_000, "S"),
                 (102.00, 25_000, "S"), (100.20, 3_000_000, "P"))

    print("\n--- 1. a client who overpaid ---")
    t = M.Trade("123456AB1", "2024-05-01", "buy", 100_000, 102.00)
    v = M.audit_trade(t, day)
    check(v.assessable, "assessable when the tape has a comparable print")
    check(v.benchmark_kind == "institutional",
          "prefers the institutional benchmark", v.benchmark_kind)
    check(abs(v.benchmark - 101.0143) < 0.01,
          "benchmark is the par-weighted institutional price",
          f"{v.benchmark:.4f}")
    check(v.cost_points > 0.98, "overpayment is positive cost",
          f"{v.cost_points:+.3f} pts")
    # 0.986 points on $100,000 par = $986. The 100x error would give $9.86.
    check(950 < v.cost_dollars < 1000,
          "points converted to dollars correctly (1 pt = $1,000 per $100k)",
          f"${v.cost_dollars:,.2f}")

    print("\n--- 2. direction: a SELLER is hurt by receiving LESS ---")
    sell_day = prints((100.00, 4_000_000, "P"), (98.50, 30_000, "P"),
                      (101.00, 3_000_000, "S"))
    bad_sale = M.audit_trade(
        M.Trade("123456AB1", "2024-05-01", "sell", 100_000, 98.50), sell_day)
    check(bad_sale.cost_points > 0,
          "selling below the benchmark is a positive cost, not a gain",
          f"{bad_sale.cost_points:+.3f} pts")
    good_sale = M.audit_trade(
        M.Trade("123456AB1", "2024-05-01", "sell", 100_000, 100.50), sell_day)
    check(good_sale.cost_points < 0, "selling above the benchmark is a credit",
          f"{good_sale.cost_points:+.3f} pts")

    print("\n--- 3. the side convention (getting this wrong invents penalties) ---")
    # The client's BUY must be compared to dealer-sold-to-customer ('S')
    # prints, never to 'P' prints on the other side of the spread.
    check(M._emma_side_for("buy") == "S", "a client buy maps to EMMA side S")
    check(M._emma_side_for("sell") == "P", "a client sell maps to EMMA side P")
    only_p = prints((99.00, 2_000_000, "P"))
    v2 = M.audit_trade(M.Trade("X", "2024-05-01", "buy", 50_000, 102.0), only_p)
    check(not v2.assessable,
          "a buy is NOT benchmarked against sell-side prints alone")

    print("\n--- 4. it refuses to invent a benchmark ---")
    v3 = M.audit_trade(M.Trade("X", "2024-05-01", "buy", 50_000, 102.0), [])
    check(not v3.assessable and "not assessable" in v3.reason,
          "no prints that day means no verdict")
    check(v3.cost_dollars is None, "no cost is reported when not assessable")

    print("\n--- 5. falls back honestly when no institutional print exists ---")
    retail_only = prints((101.50, 20_000, "S"), (101.80, 15_000, "S"))
    v4 = M.audit_trade(M.Trade("X", "2024-05-01", "buy", 25_000, 102.5),
                       retail_only)
    check(v4.benchmark_kind == "customer", "falls back to customer prints")
    check(any("weaker yardstick" in n for n in v4.notes),
          "and says the yardstick is weaker")

    print("\n--- 6. percentile among same-side prints ---")
    many = prints(*[(100.0 + i * 0.1, 25_000, "S") for i in range(10)])
    mid = M.audit_trade(M.Trade("X", "2024-05-01", "buy", 25_000, 100.5), many)
    check(40 <= mid.percentile <= 60, "a mid-range fill sits mid-percentile",
          f"{mid.percentile:.0f}%")
    worst = M.audit_trade(M.Trade("X", "2024-05-01", "buy", 25_000, 101.0), many)
    check(worst.percentile >= 85, "the worst fill sits at a high percentile",
          f"{worst.percentile:.0f}%")

    print("\n--- 7. malformed input is rejected, not guessed at ---")
    for bad, why in ((M.Trade("X", "d", "hold", 1000, 100), "unknown side"),
                     (M.Trade("X", "d", "buy", 0, 100), "zero par"),
                     (M.Trade("X", "d", "buy", 1000, 0), "zero price")):
        r = M.audit_trade(bad, day)
        check(not r.assessable, f"{why} is refused")

    print("\n--- 8. portfolio totals ---")
    vs = [
        M.audit_trade(M.Trade("A", "d", "buy", 100_000, 102.00), day),
        M.audit_trade(M.Trade("B", "d", "buy", 50_000, 101.00), day),
        M.audit_trade(M.Trade("C", "d", "buy", 25_000, 102.50), day),
        M.audit_trade(M.Trade("D", "d", "buy", 25_000, 100.0), []),   # skipped
    ]
    tot = M.audit_portfolio(vs)
    check(tot["trades_submitted"] == 4, "counts everything submitted")
    check(tot["trades_assessed"] == 3, "assesses only what the tape supports")
    check(tot["trades_not_assessable"] == 1,
          "reports the unassessable rather than dropping it")
    check(tot["total_cost_dollars"] > 0, "totals the cost",
          f"${tot['total_cost_dollars']:,.0f}")
    check(tot["worst"].trade.cusip == "A",
          "identifies the worst trade by dollars", tot["worst"].trade.cusip)
    check(abs(tot["par_assessed"] - 175_000) < 1, "sums assessed par only")

    print("\n--- 9. serialisation keeps the numbers intact ---")
    d = M.to_dict(vs[0])
    check(d["cost_dollars"] == round(vs[0].cost_dollars, 2),
          "cost survives serialisation")
    check(set(["cusip", "assessable", "benchmark", "cost_dollars", "notes"])
          <= set(d), "payload has the fields the UI needs")

    print("\n" + "=" * 78)
    print(f"{N[0] - len(FAIL)}/{N[0]} checks passed")
    if FAIL:
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
