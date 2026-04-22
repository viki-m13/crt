#!/usr/bin/env python3
"""Back-fill take-profit fields (tp_price, sl_price, tp_pct, sl_pct,
tp_time_stop_bars) on existing max/docs/data payloads so the live webapp
gets concrete buy/sell signals without waiting for the next scheduled
scan.

Strategy (see max/research/RESEARCH_TP_STRATEGY.md):
  - tp_price = last_price × 1.10
  - sl_price = last_price × 0.85
  - time_stop = 252 trading days (~12 months)

Idempotent. Running twice overwrites with the same values.
"""
from __future__ import annotations

import json
import os
import sys

OUT_DIR = os.path.join("max", "docs", "data")
TICKER_DIR = os.path.join(OUT_DIR, "tickers")
FULL_PATH = os.path.join(OUT_DIR, "full.json")

TP_PCT = 10.0
SL_PCT = 15.0
TP_TIME_STOP_BARS = 252


def tp_fields(last_price):
    if last_price is None:
        return None
    try:
        lp = float(last_price)
    except (TypeError, ValueError):
        return None
    if lp != lp or lp <= 0:  # NaN or non-positive
        return None
    return {
        "tp_price": lp * (1.0 + TP_PCT / 100.0),
        "sl_price": lp * (1.0 - SL_PCT / 100.0),
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tp_time_stop_bars": TP_TIME_STOP_BARS,
    }


def migrate_full() -> tuple[int, int]:
    with open(FULL_PATH, "r") as f:
        d = json.load(f)
    items_patched = 0
    for it in d.get("items", []):
        if it.get("is_crypto"):
            continue  # TP strategy is stock-only for now
        fields = tp_fields(it.get("last_price"))
        if fields:
            it.update(fields)
            items_patched += 1

    details_patched = 0
    for tk, detail in d.get("details", {}).items():
        if not isinstance(detail, dict):
            continue
        # Find last_price via the series object
        series = detail.get("series") or {}
        prices = series.get("prices") or []
        lp = None
        for p in reversed(prices):
            try:
                if p is not None and float(p) > 0:
                    lp = float(p)
                    break
            except (TypeError, ValueError):
                continue
        fields = tp_fields(lp)
        if fields:
            detail.update(fields)
            details_patched += 1

    model = d.get("model") or {}
    model["take_profit"] = {
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "time_stop_bars": TP_TIME_STOP_BARS,
    }
    if model.get("version", "").startswith("v10"):
        model["version"] = "v11-max-tp-migrated"
    d["model"] = model

    with open(FULL_PATH, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    return items_patched, details_patched


def migrate_tickers() -> tuple[int, int]:
    n_ok = 0
    n_skip = 0
    if not os.path.isdir(TICKER_DIR):
        return 0, 0
    for name in sorted(os.listdir(TICKER_DIR)):
        if not name.endswith(".json"):
            continue
        # Skip crypto ticker files (ends with -USD or _USD)
        base = name.replace(".json", "")
        if base.endswith("-USD") or base.endswith("_USD"):
            n_skip += 1
            continue
        path = os.path.join(TICKER_DIR, name)
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except Exception:
            n_skip += 1
            continue
        series = d.get("series") or {}
        prices = series.get("prices") or []
        lp = None
        for p in reversed(prices):
            try:
                if p is not None and float(p) > 0:
                    lp = float(p)
                    break
            except (TypeError, ValueError):
                continue
        fields = tp_fields(lp)
        if fields:
            d.update(fields)
            with open(path, "w") as f:
                json.dump(d, f, separators=(",", ":"))
            n_ok += 1
        else:
            n_skip += 1
    return n_ok, n_skip


def main():
    if not os.path.exists(FULL_PATH):
        print(f"[err] {FULL_PATH} not found", file=sys.stderr)
        return 1
    i, dt = migrate_full()
    print(f"[full.json] patched {i} items, {dt} details")
    ok, skip = migrate_tickers()
    print(f"[tickers/] patched {ok} (stocks), skipped {skip} (crypto/no-price)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
