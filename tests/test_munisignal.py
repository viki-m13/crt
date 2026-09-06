#!/usr/bin/env python3
"""Validation of the muni dislocation scanner (api/_munisignal.py).

The strategy this implements was validated elsewhere; what these check is
that the IMPLEMENTATION matches the validated rules, because an
approximation inherits none of that evidence. The specific things that would
silently break it: look-ahead in the trailing median, a mid built from the
wrong precedence, a limit computed from today instead of the prior mark, and
firing on bonds too illiquid to trade.

    python tests/test_munisignal.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import _munisignal as S  # noqa: E402

FAIL: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    if cond:
        print(f"  ok    {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label} {detail}")


def days(start_day: int, n: int, buy: float, sell=None, dealer=None):
    """n consecutive daily prints starting at 2024-01-{start_day}."""
    out = []
    for i in range(n):
        d = f"2024-01-{start_day + i:02d}"
        out.append(S.DayPrints(date=d, buy=buy, sell=sell, dealer=dealer))
    return out


def main():
    print("=" * 78)
    print("MUNI DISLOCATION SCANNER")
    print("=" * 78)

    print("\n--- 1. mid follows the validated precedence ---")
    check(S.DayPrints("d", buy=101, sell=99, dealer=100.5).mid == 100.5,
          "inter-dealer print wins when present")
    check(S.DayPrints("d", buy=101, sell=99).mid == 100.0,
          "else the customer buy/sell midpoint")
    check(S.DayPrints("d", buy=101).mid == 101.0,
          "else whichever single side exists")
    check(S.DayPrints("d").mid is None, "no prints means no mid")

    print("\n--- 2. a genuine dislocation fires ---")
    hist = days(1, 20, buy=100.0, sell=99.0)          # trend mid = 99.5
    hist.append(S.DayPrints("2024-01-25", buy=95.0, sell=94.0))
    sig = S.evaluate("BOND1", hist, "2024-01-25", "TEST MUNI 5% 2035")
    check(sig.dislocated, "fires when >=3 points below trend")
    check(abs(sig.trailing_median - 99.5) < 1e-6, "trailing median is right",
          f"{sig.trailing_median}")
    check(abs(sig.discount_pts - 4.5) < 1e-6, "discount measured correctly",
          f"{sig.discount_pts:.2f} pts")

    print("\n--- 3. the limit price is the deliverable ---")
    # latest prior mid is 99.5, so the limit must be 99.75
    check(abs(sig.prior_mid - 99.5) < 1e-6, "limit is set off the PRIOR mid",
          f"{sig.prior_mid}")
    check(abs(sig.limit_price - 99.75) < 1e-6, "limit = prior mid + 0.25",
          f"{sig.limit_price}")
    check(any("inside" in n for n in sig.notes),
          "a buy below the limit is flagged tradeable")

    print("\n--- 4. it declines to chase a falling bond ---")
    # The case the limit rule exists for: the bond has been sliding, so the
    # LAST mark is far below the 60-day median. A print that is still 3+
    # points under the median can nonetheless be well above the recent mark
    # — that is chasing, and the tested rule refuses it.
    h2 = days(1, 15, buy=100.0, sell=99.0)          # mid 99.5, sets the median
    h2 += [S.DayPrints(f"2024-01-{16+i:02d}", buy=92.0, sell=91.0)
           for i in range(5)]                        # slid to mid 91.5
    h2.append(S.DayPrints("2024-01-25", buy=96.4, sell=95.4))
    s2 = S.evaluate("BOND2", h2, "2024-01-25")
    check(s2.dislocated, "still dislocated vs the 60-day median",
          f"{s2.discount_pts:.2f} pts below {s2.trailing_median}")
    check(abs(s2.limit_price - 91.75) < 1e-6,
          "limit tracks the LATEST mark, not the stale median",
          f"limit {s2.limit_price}")
    check(s2.buy_price > s2.limit_price, "the print is above that limit",
          f"{s2.buy_price} > {s2.limit_price}")
    check(any("ABOVE" in n for n in s2.notes),
          "and the output says so rather than recommending it")

    print("\n--- 5. thresholds are the locked ones ---")
    h3 = days(1, 20, buy=100.0, sell=99.0)
    h3.append(S.DayPrints("2024-01-25", buy=97.0, sell=96.0))   # 2.5 below
    s3 = S.evaluate("BOND3", h3, "2024-01-25")
    check(not s3.dislocated, "2.5 points does NOT fire (below the 3.0 lock)")
    check("does not survive the control" in s3.reason,
          "and explains why rather than just saying no")
    check(S.DISCOUNT_PTS == 3.0 and S.LIMIT_CAP == 0.25,
          "locked constants unchanged", f"{S.DISCOUNT_PTS}, {S.LIMIT_CAP}")

    print("\n--- 6. no look-ahead ---")
    # today's own print must not enter its own trailing median
    hist_la = days(1, 20, buy=100.0, sell=99.0)
    hist_la.append(S.DayPrints("2024-01-25", buy=50.0, sell=50.0))
    s_la = S.evaluate("BOND4", hist_la, "2024-01-25")
    check(abs(s_la.trailing_median - 99.5) < 1e-6,
          "a huge move today does not move today's own trend",
          f"{s_la.trailing_median}")
    check(s_la.prior_mid == 99.5, "prior mid excludes today")

    print("\n--- 7. the liquidity gate ---")
    thin = [S.DayPrints("2024-01-01", buy=100.0, sell=99.0),
            S.DayPrints("2024-01-05", buy=100.0, sell=99.0),
            S.DayPrints("2024-01-09", buy=100.0, sell=99.0),
            S.DayPrints("2024-01-25", buy=95.0, sell=94.0)]
    st = S.evaluate("THIN", thin, "2024-01-25")
    check(not st.dislocated, "an illiquid bond does not fire")
    check("illiquid" in st.reason, "and says it is a liquidity rejection")

    print("\n--- 8. degenerate cases ---")
    check(not S.evaluate("X", [], "2024-01-25").dislocated, "empty history")
    no_buy = days(1, 20, buy=100.0, sell=99.0)
    no_buy.append(S.DayPrints("2024-01-25", sell=94.0))
    check(not S.evaluate("X", no_buy, "2024-01-25").dislocated,
          "a sell-only day cannot be bought")

    print("\n--- 9. the price verdict an advisor actually asks for ---")
    v = S.price_verdict(99.50, sig)
    check(v["verdict"] == "within limit", "a fair quote passes", v["verdict"])
    v2 = S.price_verdict(101.25, sig)
    check(v2["verdict"] == "too rich", "an expensive quote is refused")
    check(abs(v2["over_limit_pts"] - 1.50) < 1e-6, "overpayment in points",
          f"{v2['over_limit_pts']:.2f}")
    # 1.5 points on $100k = $1,500 — the conversion that was wrong by 100x once
    check(abs(v2["over_limit_per_100k"] - 1500) < 1,
          "and in dollars per $100k", f"${v2['over_limit_per_100k']:,.0f}")
    vnone = S.price_verdict(100.0, S.Signal("X", "d"))
    check(vnone["verdict"] == "unknown",
          "no reference price means no verdict, not a guess")

    print("\n--- 10. scanning many bonds ---")
    bonds = {
        "A": ("CHEAP ONE", hist),
        "B": ("NOT CHEAP", days(1, 20, buy=100.0, sell=99.0)
              + [S.DayPrints("2024-01-25", buy=99.0, sell=98.0)]),
        "C": ("CHEAPEST", days(1, 20, buy=100.0, sell=99.0)
              + [S.DayPrints("2024-01-25", buy=90.0, sell=89.0)]),
    }
    hits = S.scan(bonds, "2024-01-25")
    check(len(hits) == 2, "only the dislocated bonds are returned", f"{len(hits)}")
    check(hits[0].security_id == "C", "sorted cheapest first")
    quiet = S.scan({"B": bonds["B"]}, "2024-01-25")
    check(quiet == [], "an empty result is a real answer, not an error")

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
