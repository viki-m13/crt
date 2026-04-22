#!/usr/bin/env python3
"""Post-process existing max/docs/data payloads so they carry the SMA 12M
smoothed conviction series (`final_smooth12m`) and today's smoothed reading
(`conviction_smooth12m`) without needing to re-fetch market data.

Idempotent. Running it twice is a no-op (existing `final_smooth12m` arrays
are overwritten with the same values).

Background: step39 research identified CAP5+SMA12M (trailing 12-month mean
of the conviction series) as the production winner over raw conviction
ranking — +0.90pp CAGR on the 96T universe, +1.68pp on the 128T expansion,
wins 5/5 rolling 10Y windows vs baseline AND vs SPY. Production scanner
(`daily_scan_max.py` v10-max) emits these fields natively. This migration
back-fills them on payloads produced by older scanner versions.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

OUT_DIR = os.path.join("max", "docs", "data")
TICKER_DIR = os.path.join(OUT_DIR, "tickers")
FULL_PATH = os.path.join(OUT_DIR, "full.json")

SMOOTH_WINDOW_BARS = 12 * 21  # 252 trading days ≈ 12 calendar months


def trailing_mean(series: list[Any], window: int) -> list[float | None]:
    """Trailing N-bar mean of a series (None/NaN-safe).

    Early rows (before `window` history) use the expanding mean — this
    matches `pandas.rolling(min_periods=1).mean()` semantics and what the
    scanner emits natively, so the two code paths agree exactly.
    """
    out: list[float | None] = [None] * len(series)
    s = 0.0
    n = 0
    # We maintain a simple running sum over the last `window` valid entries.
    # Since most series have ~5000 entries, an O(len * window) approach would
    # be slow — use a deque-style rolling instead.
    from collections import deque

    buf: deque[float] = deque()
    for i, v in enumerate(series):
        try:
            x = float(v) if v is not None else None
            if x is not None and x != x:  # NaN check
                x = None
        except (TypeError, ValueError):
            x = None
        if x is not None:
            buf.append(x)
            s += x
            if len(buf) > window:
                s -= buf.popleft()
            out[i] = s / len(buf) if len(buf) > 0 else None
        else:
            # Preserve NaN positions as None so downstream JSON keeps null.
            # Don't advance the buffer — mirror the scanner's fillna(0) by
            # treating None as a real 0? Actually, scanner does ffill then
            # fillna(0) before rolling, so zeros contribute. Match that.
            buf.append(0.0)
            s += 0.0
            if len(buf) > window:
                s -= buf.popleft()
            out[i] = s / len(buf) if len(buf) > 0 else None
    return out


def migrate_ticker_json(path: str) -> bool:
    try:
        with open(path, "r") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[skip] {path}: {e}", file=sys.stderr)
        return False
    series = doc.get("series") or {}
    final = series.get("final")
    if not final or not isinstance(final, list) or len(final) == 0:
        return False
    smoothed = trailing_mean(final, SMOOTH_WINDOW_BARS)
    series["final_smooth12m"] = smoothed
    doc["series"] = series
    # Backfill the top-level smoothed reading too so the UI's per-ticker
    # detail object has it.
    last = smoothed[-1] if smoothed else None
    if last is not None and last > 0:
        doc["conviction_smooth12m"] = float(last)
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    return True


def migrate_full_json() -> tuple[int, int]:
    with open(FULL_PATH, "r") as f:
        full = json.load(f)
    bt_series = full.get("bt_series") or {}
    items = full.get("items") or []
    details = full.get("details") or {}

    # Build a lookup: ticker -> smoothed_last from bt_series.
    smoothed_last: dict[str, float] = {}
    for tk, s in bt_series.items():
        final = s.get("final") or []
        if not final:
            continue
        smoothed = trailing_mean(final, SMOOTH_WINDOW_BARS)
        s["final_smooth12m"] = smoothed
        if smoothed:
            last = smoothed[-1]
            if last is not None and last > 0:
                smoothed_last[tk] = float(last)

    # Patch items array with conviction_smooth12m.
    patched_items = 0
    for it in items:
        tk = it.get("ticker")
        if tk and tk in smoothed_last:
            it["conviction_smooth12m"] = smoothed_last[tk]
            patched_items += 1

    # Patch details dict with conviction_smooth12m + series.final_smooth12m.
    patched_details = 0
    for tk, detail in details.items():
        if not isinstance(detail, dict):
            continue
        s = detail.get("series") or {}
        if "final" in s and isinstance(s["final"], list) and len(s["final"]):
            smoothed = trailing_mean(s["final"], SMOOTH_WINDOW_BARS)
            s["final_smooth12m"] = smoothed
            detail["series"] = s
            if smoothed and smoothed[-1] and smoothed[-1] > 0:
                detail["conviction_smooth12m"] = float(smoothed[-1])
            patched_details += 1
        elif tk in smoothed_last:
            detail["conviction_smooth12m"] = smoothed_last[tk]

    # Bump model version to reflect the smoothing field presence.
    model = full.get("model") or {}
    if model.get("version", "").startswith("v9"):
        model["version"] = "v10-max-migrated"
    model["cap5_smoothing_months"] = 12
    full["model"] = model
    full["bt_series"] = bt_series
    full["items"] = items
    full["details"] = details
    with open(FULL_PATH, "w") as f:
        json.dump(full, f, separators=(",", ":"))
    return patched_items, patched_details


def main() -> int:
    if not os.path.exists(FULL_PATH):
        print(f"[err] {FULL_PATH} not found", file=sys.stderr)
        return 1
    items_n, details_n = migrate_full_json()
    print(f"[full.json] patched {items_n} items, {details_n} details")

    # Per-ticker files
    if os.path.isdir(TICKER_DIR):
        n_ok = 0
        n_skip = 0
        for name in sorted(os.listdir(TICKER_DIR)):
            if not name.endswith(".json"):
                continue
            p = os.path.join(TICKER_DIR, name)
            if migrate_ticker_json(p):
                n_ok += 1
            else:
                n_skip += 1
        print(f"[tickers/] patched {n_ok}, skipped {n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
