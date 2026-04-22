#!/usr/bin/env python3
"""Back-fill dynamic ATR-scaled TP/SL fields on existing max/docs/data payloads.

Recomputes ATR14 from each ticker's series.prices (using daily close-to-close
returns as a proxy; the original scanner uses true-range which requires OHLC
but for migration-only approximation we use close-based volatility). Replaces
old fixed TP/SL fields with dynamic ones matching the v12-max-tp-atr scanner
output.

Research (see max/research/step47_summary.md): TP distance = max(0.05, min(7 ×
ATR14%, 0.25)); SL distance = max(0.05, min(7 × ATR14%, 0.12)).

Idempotent. Running twice produces identical output.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Iterable

OUT_DIR = os.path.join("max", "docs", "data")
TICKER_DIR = os.path.join(OUT_DIR, "tickers")
FULL_PATH = os.path.join(OUT_DIR, "full.json")

TP_ATR_K = 7.0
SL_ATR_K = 7.0
TP_CAP = 0.25
TP_FLOOR = 0.05
SL_CAP = 0.12
SL_FLOOR = 0.05
TP_TIME_STOP_BARS = 252
ATR_WINDOW = 14


def estimate_atr_pct_from_prices(prices: Iterable) -> float | None:
    """Close-to-close proxy for ATR14 as a fraction of the last price.

    True ATR uses High/Low/PrevClose per bar. We don't have those in the
    webapp's JSON payload (only close via series.prices), so we approximate
    ATR14 with the mean absolute log-return over the last 14 bars scaled to
    price — a proxy that correlates tightly with true ATR for daily data.
    """
    vals: list[float] = []
    for p in prices:
        try:
            if p is None:
                continue
            x = float(p)
            if x != x or x <= 0:
                continue
            vals.append(x)
        except (TypeError, ValueError):
            continue
    if len(vals) < ATR_WINDOW + 1:
        return None
    last = vals[-1]
    returns = [abs(math.log(vals[i] / vals[i - 1])) for i in range(1, len(vals))]
    recent = returns[-ATR_WINDOW:]
    if not recent:
        return None
    # Mean absolute log-return ≈ ATR14 / price for daily bars.
    atr_pct = sum(recent) / len(recent)
    if not math.isfinite(atr_pct) or atr_pct <= 0:
        return None
    return atr_pct


def atr_scaled_frac(atr_pct: float | None, k: float, floor: float, cap: float) -> float:
    if atr_pct is None or not math.isfinite(atr_pct) or atr_pct <= 0:
        return floor
    d = k * atr_pct
    return max(floor, min(d, cap))


def tp_sl_fields(last_price: float | None, prices: list | None) -> dict | None:
    if last_price is None:
        return None
    try:
        lp = float(last_price)
    except (TypeError, ValueError):
        return None
    if lp <= 0 or lp != lp:
        return None
    atr_pct = estimate_atr_pct_from_prices(prices or []) if prices else None
    tp_frac = atr_scaled_frac(atr_pct, TP_ATR_K, TP_FLOOR, TP_CAP)
    sl_frac = atr_scaled_frac(atr_pct, SL_ATR_K, SL_FLOOR, SL_CAP)
    return {
        "tp_price": lp * (1.0 + tp_frac),
        "sl_price": lp * (1.0 - sl_frac),
        "tp_pct": tp_frac * 100.0,
        "sl_pct": sl_frac * 100.0,
        "atr14_pct": atr_pct,
        "tp_atr_k": TP_ATR_K,
        "sl_atr_k": SL_ATR_K,
        "tp_time_stop_bars": TP_TIME_STOP_BARS,
    }


def migrate_full() -> tuple[int, int]:
    with open(FULL_PATH, "r") as f:
        d = json.load(f)
    bt = d.get("bt_series") or {}

    items_patched = 0
    for it in d.get("items", []):
        if it.get("is_crypto"):
            continue
        tk = it.get("ticker")
        prices = (bt.get(tk) or {}).get("prices")
        f = tp_sl_fields(it.get("last_price"), prices)
        if f:
            it.update(f)
            items_patched += 1

    details_patched = 0
    for tk, detail in d.get("details", {}).items():
        if not isinstance(detail, dict):
            continue
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
        f = tp_sl_fields(lp, prices)
        if f:
            detail.update(f)
            details_patched += 1

    model = d.get("model") or {}
    model["take_profit"] = {
        "mode": "atr_dynamic",
        "tp_atr_k": TP_ATR_K,
        "sl_atr_k": SL_ATR_K,
        "tp_cap": TP_CAP,
        "tp_floor": TP_FLOOR,
        "sl_cap": SL_CAP,
        "sl_floor": SL_FLOOR,
        "time_stop_bars": TP_TIME_STOP_BARS,
    }
    if model.get("version", "").startswith("v11"):
        model["version"] = "v12-max-tp-atr-migrated"
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
        f = tp_sl_fields(lp, prices)
        if f:
            d.update(f)
            with open(path, "w") as fh:
                json.dump(d, fh, separators=(",", ":"))
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
    print(f"[tickers/] patched {ok} (stocks), skipped {skip} (crypto/no-prices)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
