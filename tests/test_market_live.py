#!/usr/bin/env python3
"""Live network checks for the market layer. Requires internet.

Separate from the offline suites on purpose: this one can legitimately fail
because an upstream is throttling us, and that must never be confused with a
logic regression. Throttled runs report SKIP, not FAIL.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import _market as M  # noqa: E402

FAIL: list[str] = []
SKIP: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    if cond:
        print(f"  ok    {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label} {detail}")


def skip(label, why):
    SKIP.append(label)
    print(f"  skip  {label} — {why}")


GLOBAL = [
    ("AAPL", "USD", "US mega cap"),
    ("7203.T", "JPY", "Tokyo"),
    ("ASML.AS", "EUR", "Amsterdam"),
    ("MC.PA", "EUR", "Paris"),
    ("NESN.SW", "CHF", "Zurich"),
    ("0700.HK", "HKD", "Hong Kong"),
    ("RELIANCE.NS", "INR", "India"),
    ("BHP.AX", "AUD", "Australia"),
    ("SHOP.TO", "CAD", "Toronto"),
    ("005930.KS", "KRW", "Korea"),
]


def main():
    print("=" * 78)
    print("LIVE MARKET LAYER")
    print("=" * 78)

    print("\n--- global coverage ---")
    throttled = 0
    verified = 0
    for sym, cur, where in GLOBAL:
        q = M.quote(sym, rng="1y")
        if not q.get("ok") and q.get("reason") in ("rate_limited", "unreachable"):
            throttled += 1
            skip(f"{sym} ({where})", q.get("reason"))
            continue
        verified += 1
        ok = q.get("ok") and q.get("currency") == cur and q.get("ann_vol") is not None
        check(ok, f"{sym:12} {where}",
              f"{str(q.get('name'))[:22]:24} {q.get('currency')} "
              f"vol={q.get('ann_vol', 0):.0%} dd={q.get('max_dd', 0):+.0%}"
              if q.get("ok") else str(q.get("reason")))
        time.sleep(0.25)

    if throttled and verified == 0:
        print("\n  upstream is throttling this IP; live coverage unverified this run")
        print(f"\n{N[0] - len(FAIL)}/{N[0]} checks passed, {len(SKIP)} skipped")
        return 0

    print("\n--- rejection of things that must never be recommended ---")
    for bad, why in (("FAKETICKERXYZ", "invented ticker"),
                     ("ZZZZZZZZ", "nonsense symbol")):
        q = M.quote(bad)
        if q.get("reason") in ("rate_limited", "unreachable"):
            skip(bad, "throttled")
            continue
        check(not q.get("ok"), f"{why} is rejected", q.get("reason", ""))

    print("\n--- concurrency and caching ---")
    t0 = time.time()
    batch = M.quotes(["MSFT", "KO", "JNJ", "XOM", "PG", "V"], rng="1y")
    dt = time.time() - t0
    live = [k for k, v in batch.items() if v.get("ok")]
    if not live:
        skip("concurrent batch", "throttled")
    else:
        check(len(live) >= 4, f"batch verified {len(live)}/6 concurrently", f"{dt:.1f}s")
        check(dt < 20, "batch completes within a serverless budget", f"{dt:.1f}s")
        t1 = time.time()
        M.quotes(live, rng="1y")
        check(time.time() - t1 < 1.0, "second call served from cache",
              f"{time.time()-t1:.2f}s")

    print("\n--- computed metrics are sane ---")
    q = M.quote("AAPL", rng="2y")
    if not q.get("ok"):
        skip("metric sanity", q.get("reason", "unavailable"))
    else:
        check(0.05 < q["ann_vol"] < 1.2, "volatility in a believable range",
              f"{q['ann_vol']:.1%}")
        check(-0.95 < q["max_dd"] <= 0.0, "drawdown is negative and bounded",
              f"{q['max_dd']:.1%}")
        check(0.0 <= q["pos_52w"] <= 1.0, "52-week position is a fraction",
              f"{q['pos_52w']:.2f}")
        check(q["bars"] > 400, "two years of history returned", f"{q['bars']} bars")
        check(q.get("price", 0) > 0, "a live price came back", str(q.get("price")))

    print("\n" + "=" * 78)
    print(f"{N[0] - len(FAIL)}/{N[0]} checks passed, {len(SKIP)} skipped")
    if FAIL:
        for f in FAIL:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
