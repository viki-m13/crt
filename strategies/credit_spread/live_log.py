"""Live signal log — survivorship-bias-free tracking for CreditFloor.

Every cron run publishes a set of (ticker, side, horizon, strike, expiry)
predictions. This module persists every one of those predictions to an
append-only log and resolves them deterministically when their expiry
is reached or a breach is observed.

File layout (``spreads/docs/data/live_log.json``)
-------------------------------------------------

    {
      "generated_at": "...",
      "summary": {
        "total":    N,
        "pending":  N,
        "resolved": N,
        "wins":     N,
        "losses":   N,
        "win_rate": float|null,
        "put":  {total, pending, wins, losses, win_rate},
        "call": {total, pending, wins, losses, win_rate},
        "first_publish_date": "YYYY-MM-DD",
        "last_resolved_at":   "YYYY-MM-DDTHH:MM:SSZ"
      },
      "signals": [ {signal}, ... ]     # append-only, sorted by id
    }

Each signal
-----------

    {
      "id":              "<pub_date>:<ticker>:<side>:<h>",
      "publish_date":    "YYYY-MM-DD",     # end_date of the scan run
      "ticker":          "AAPL",
      "side":            "put" | "call",
      "horizon":         21,
      "expiry_date":     "YYYY-MM-DD",     # NYSE-calendar projected
      "spot_at_publish": 307.33,
      "strike":          263.40,
      "buffer_pct":      14.29,
      "variant":         "plain" | "regime",

      # Resolution fields — start null, set when status flips
      "status":          "pending" | "win" | "loss",
      "resolved_at":     null | "ISO-UTC",
      "close_at_expiry": null | float,     # set on resolution
      "forward_close_min": null | float,   # informational; min close over (pub, expiry]
      "forward_close_max": null | float    # informational; max close over (pub, expiry]
    }

Semantics
---------

Resolution uses **close-at-expiry** — the short strike is safe iff the
stock's close on ``expiry_date`` is on the right side of it. Intraday
moves and intermediate-close excursions do not resolve the signal; only
the close on the expiry date matters.

- WIN  iff close on ``expiry_date`` >= strike (put) or <= strike (call).
- LOSS iff close on ``expiry_date`` <  strike (put) or >  strike (call).
- PENDING until the series contains data for ``expiry_date``.

The backtest uses a strictly tighter criterion (path-minimum for puts,
path-maximum for calls) — so a backtest that shows 100% path-coverage
is a lower bound on what live close-at-expiry resolution will show.

- PENDING stays pending indefinitely — it survives the ticker dropping
  off today's eligibility list. This is the anti-survivorship-bias
  machinery: a CPB-style crash that removes the ticker from today's
  scan still resolves its pending log entries correctly at expiry.
- Duplicates are silently deduped on (publish_date, ticker, side, horizon).

Usage
-----

    from live_log import update_live_log
    log = update_live_log(signals_json_path, live_log_path)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from common import TICKERS_DIR, load_series


LIVE_LOG_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_existing_log(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {
            "version": LIVE_LOG_VERSION,
            "generated_at": None,
            "summary": {},
            "signals": [],
        }
    with open(path, "r") as fh:
        blob = json.load(fh)
    blob.setdefault("version", LIVE_LOG_VERSION)
    blob.setdefault("signals", [])
    return blob


def _atomic_write(path: str, blob: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, path)


# ------------------------- enumeration -----------------------------------
#
# Every rung of every ladder of every ticker becomes an independent
# signal. Users can trade any one rung, so honesty requires all are
# tracked.


def _iter_signal_rungs(signals_blob: dict[str, Any]):
    """Yield (publish_date, ticker, side, rung_dict, spot) tuples."""
    for side_key, side_name in (("put_signals", "put"), ("call_signals", "call")):
        for s in signals_blob.get(side_key, []):
            publish_date = s["end_date"]  # last close date of the scan
            ticker = s["ticker"]
            spot = s["today_close"]
            for rung in s.get("ladder", []):
                yield publish_date, ticker, side_name, rung, spot


def _make_signal(publish_date: str, ticker: str, side: str, rung: dict, spot: float) -> dict:
    return {
        "id": f"{publish_date}:{ticker}:{side}:{rung['horizon']}",
        "publish_date": publish_date,
        "ticker": ticker,
        "side": side,
        "horizon": rung["horizon"],
        "expiry_date": rung["expiry_date"],
        "spot_at_publish": spot,
        "strike": rung["strike"],
        "buffer_pct": rung["buffer_pct"],
        "variant": rung["variant"],
        "status": "pending",
        "resolved_at": None,
        "close_at_expiry": None,
        "forward_close_min": None,
        "forward_close_max": None,
    }


def _append_new_signals(log: dict, signals_blob: dict) -> int:
    existing_ids = {s["id"] for s in log["signals"]}
    added = 0
    for publish_date, ticker, side, rung, spot in _iter_signal_rungs(signals_blob):
        sig = _make_signal(publish_date, ticker, side, rung, spot)
        if sig["id"] in existing_ids:
            continue
        log["signals"].append(sig)
        existing_ids.add(sig["id"])
        added += 1
    # Stable sort: by publish_date then ticker then side then horizon
    log["signals"].sort(key=lambda s: (s["publish_date"], s["ticker"], s["side"], s["horizon"]))
    return added


# ------------------------- resolution -----------------------------------


def _resolve_one(sig: dict, ts) -> bool:
    """Try to resolve `sig` using the given TickerSeries. Returns True
    if the signal transitioned out of `pending`.

    Close-at-expiry semantics:
      win  iff close_at_expiry >= strike (put) or <= strike (call)
      loss iff close_at_expiry <  strike (put) or >  strike (call)
    Intraday moves and interim closes do NOT resolve; only the close
    on ``expiry_date`` does.
    """
    if sig["status"] != "pending":
        return False

    pub_d = np.datetime64(sig["publish_date"], "D")
    exp_d = np.datetime64(sig["expiry_date"], "D")
    strike = float(sig["strike"])
    is_put = sig["side"] == "put"

    # Pick out closes in the window (publish_date, expiry_date]
    mask = (ts.dates > pub_d) & (ts.dates <= exp_d)
    if not mask.any():
        return False  # No data in window yet

    window_dates = ts.dates[mask]
    window_closes = ts.close[mask]

    # Informational: running min/max across (publish_date, expiry_date]
    # so the UI can show "how close did the path get?" without using it
    # in the resolution decision.
    if is_put:
        sig["forward_close_min"] = float(window_closes.min())
    else:
        sig["forward_close_max"] = float(window_closes.max())

    # Only resolve if we have data on (or past) the expiry date. The
    # expiry_date itself is always an NYSE trading session (projected
    # via pandas_market_calendars at signal creation), so we expect
    # an exact match; if for any reason that date is missing from the
    # series (data vendor gap) we require at least one close >= expiry
    # and use the last close on-or-before expiry as the resolution.
    last_date_seen = window_dates[-1]
    if last_date_seen < exp_d:
        return False  # Still pending

    # Find close on expiry date exactly; if absent, use the latest
    # session <= expiry (defensive against vendor gaps).
    exact = np.where(window_dates == exp_d)[0]
    if len(exact):
        close_at_expiry = float(window_closes[int(exact[-1])])
    else:
        close_at_expiry = float(window_closes[-1])

    sig["close_at_expiry"] = close_at_expiry
    sig["resolved_at"] = _now_iso()
    if is_put:
        sig["status"] = "win" if close_at_expiry >= strike else "loss"
    else:
        sig["status"] = "win" if close_at_expiry <= strike else "loss"
    return True


def _resolve_pending(log: dict) -> tuple[int, int]:
    """Resolve any pending signals that can be resolved. Returns (wins, losses)
    newly-resolved this call."""
    # Group pending signals by ticker so we only load each series once.
    by_ticker: dict[str, list[dict]] = {}
    for s in log["signals"]:
        if s["status"] == "pending":
            by_ticker.setdefault(s["ticker"], []).append(s)
    new_wins = new_losses = 0
    for ticker, sigs in by_ticker.items():
        ts = load_series(ticker)
        if ts is None:
            # Ticker data disappeared (delisted?) — leave signals pending
            # so we re-try next run. Deliberate: we don't want to silently
            # drop survivorship evidence.
            continue
        for sig in sigs:
            if _resolve_one(sig, ts):
                if sig["status"] == "win":
                    new_wins += 1
                elif sig["status"] == "loss":
                    new_losses += 1
    return new_wins, new_losses


# ------------------------- summary -----------------------------------


def _summarize(log: dict) -> dict:
    def tally(signals):
        total = len(signals)
        pending = sum(1 for s in signals if s["status"] == "pending")
        wins = sum(1 for s in signals if s["status"] == "win")
        losses = sum(1 for s in signals if s["status"] == "loss")
        resolved = wins + losses
        win_rate = (wins / resolved) if resolved > 0 else None
        return {
            "total": total,
            "pending": pending,
            "resolved": resolved,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        }

    signals = log["signals"]
    put = tally([s for s in signals if s["side"] == "put"])
    call = tally([s for s in signals if s["side"] == "call"])
    overall = tally(signals)

    # First publish date (earliest) + most recent resolution
    first_pub = signals[0]["publish_date"] if signals else None
    resolved_ats = [s["resolved_at"] for s in signals if s.get("resolved_at")]
    last_resolved = max(resolved_ats) if resolved_ats else None

    return {
        **overall,
        "put":  put,
        "call": call,
        "first_publish_date": first_pub,
        "last_resolved_at":   last_resolved,
    }


# ------------------------- public entry point -----------------------------


def update_live_log(signals_path: str, log_path: str) -> dict:
    """One-shot: read today's signals.json, add any new rungs to the log,
    resolve whatever pending entries can be resolved, write the log back.
    Returns the updated log blob.
    """
    with open(signals_path, "r") as fh:
        signals_blob = json.load(fh)
    log = _load_existing_log(log_path)
    added = _append_new_signals(log, signals_blob)
    new_wins, new_losses = _resolve_pending(log)
    log["generated_at"] = _now_iso()
    log["summary"] = _summarize(log)

    print(f"live_log: added {added} new signal rungs")
    print(f"live_log: resolved +{new_wins} wins, +{new_losses} losses this run")
    s = log["summary"]
    print(
        f"live_log: total={s['total']}  pending={s['pending']}  "
        f"resolved={s['resolved']}  wins={s['wins']}  losses={s['losses']}  "
        f"win_rate={s['win_rate']}"
    )
    _atomic_write(log_path, log)
    return log


if __name__ == "__main__":
    # CLI convenience: update using the default paths.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    signals = os.path.join(repo, "spreads", "docs", "data", "signals.json")
    logf = os.path.join(repo, "spreads", "docs", "data", "live_log.json")
    if len(sys.argv) > 1:
        signals = sys.argv[1]
    if len(sys.argv) > 2:
        logf = sys.argv[2]
    update_live_log(signals, logf)
