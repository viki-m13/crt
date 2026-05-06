"""Stillpoint live log — append-only record of every signal ever
published, resolved at expiry against actual close.

This is the survivorship-bias-free scoreboard for the Stillpoint
tiers (Atomic IC, Universal IC, Robust UIC). Every cron run:

  1. Reads strategies/stillpoint/results/stillpoint_signals.json.
  2. Appends any new (publish_date, tier, ticker, horizon) combos
     to spreads/docs/data/stillpoint_live_log.json (deduped on the
     5-tuple ID).
  3. Tries to resolve each pending signal: if the ticker series
     (in docs/data/tickers/{TICKER}.json) contains a close on the
     signal's expiry_date, we look it up and grade WIN/LOSS.

Win condition (close-at-expiry, matches what the engine validated):
   K_put_short ≤ close[expiry_date] ≤ K_call_short

Signals stay in the log forever. Tickers that drop off the eligible
list still resolve at expiry. This is the honest live measurement —
a loss that happens stays a loss; a win that happens stays a win.

The webapp can render live WR alongside backtest WR so users can
SEE whether live performance matches the 95% bar.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    list_tickers, load_series,
)


_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PUB_DIR = os.path.join(_REPO_ROOT, "spreads", "docs", "data")
_LIVE_LOG_PATH = os.path.join(_PUB_DIR, "stillpoint_live_log.json")
_SIGNALS_PATH = os.path.join(_PUB_DIR, "stillpoint_signals.json")


def _signal_id(tier, ticker, horizon, publish_date):
    return f"{tier}|{ticker}|{horizon}|{publish_date}"


def _make_signal(tier, ticker, horizon, publish_date, spot, expiry_date,
                  K_put_short, K_call_short, ror_pct, claimed_wr_pct,
                  q=None, width=None, n_oos=None):
    return {
        "id": _signal_id(tier, ticker, horizon, publish_date),
        "tier": tier,
        "ticker": ticker,
        "horizon": horizon,
        "publish_date": publish_date,
        "expiry_date": expiry_date,
        "spot_at_publish": spot,
        "K_put_short": K_put_short,
        "K_call_short": K_call_short,
        "claimed_ror_pct": ror_pct,
        "claimed_wr_pct": claimed_wr_pct,
        "q": q,
        "width": width,
        "n_oos_tests": n_oos,
        "status": "pending",
        "close_at_expiry": None,
        "resolution_date": None,
    }


def _resolve_signal(signal):
    """If ticker has close[expiry_date], grade as WIN/LOSS."""
    if signal["status"] != "pending":
        return signal
    ts = load_series(signal["ticker"])
    if ts is None:
        return signal
    target = np.datetime64(signal["expiry_date"])
    matches = np.where(ts.dates == target)[0]
    if len(matches) == 0:
        # try the trading day on or after target
        idx = np.searchsorted(ts.dates, target)
        if idx >= len(ts.dates):
            return signal
        actual_date = str(ts.dates[idx])
    else:
        idx = int(matches[0])
        actual_date = signal["expiry_date"]
    close_at = float(ts.close[idx])
    Kp = signal["K_put_short"]
    Kc = signal["K_call_short"]
    if Kp <= close_at <= Kc:
        signal["status"] = "win"
    else:
        signal["status"] = "loss"
    signal["close_at_expiry"] = close_at
    signal["resolution_date"] = actual_date
    return signal


def _load_log():
    if os.path.exists(_LIVE_LOG_PATH):
        try:
            return json.load(open(_LIVE_LOG_PATH))
        except (OSError, json.JSONDecodeError):
            pass
    return {"signals": [], "summary": {}}


def _save_log(log):
    os.makedirs(_PUB_DIR, exist_ok=True)
    with open(_LIVE_LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2, default=str)


def _summarize(signals):
    by_tier = defaultdict(lambda: {"total": 0, "pending": 0, "win": 0, "loss": 0})
    for s in signals:
        t = s["tier"]
        by_tier[t]["total"] += 1
        by_tier[t][s["status"]] += 1
    summary = {"by_tier": {}, "total": len(signals)}
    for tier, stats in by_tier.items():
        resolved = stats["win"] + stats["loss"]
        wr = stats["win"] / resolved if resolved else None
        summary["by_tier"][tier] = {
            "total": stats["total"],
            "pending": stats["pending"],
            "wins": stats["win"],
            "losses": stats["loss"],
            "resolved": resolved,
            "live_win_rate": wr,
            "live_win_rate_pct": wr * 100 if wr is not None else None,
        }
    summary["resolved"] = sum(t["win"] + t["loss"] for t in by_tier.values())
    summary["pending"] = sum(t["pending"] for t in by_tier.values())
    summary["wins"] = sum(t["win"] for t in by_tier.values())
    summary["losses"] = sum(t["loss"] for t in by_tier.values())
    if summary["resolved"]:
        summary["overall_live_win_rate_pct"] = summary["wins"] / summary["resolved"] * 100
    else:
        summary["overall_live_win_rate_pct"] = None
    return summary


def main():
    if not os.path.exists(_SIGNALS_PATH):
        print(f"[err] {_SIGNALS_PATH} not found; nothing to log")
        return 1
    pub = json.load(open(_SIGNALS_PATH))
    publish_date = (pub.get("generated_at") or "")[:10]
    if not publish_date:
        print("[warn] no generated_at; using today's date")
        publish_date = datetime.utcnow().date().isoformat()

    log = _load_log()
    existing_ids = {s["id"] for s in log["signals"]}
    new_count = 0

    # Universal IC signals
    for s in pub.get("uic_signals", []):
        for r in s["ladder"]:
            sig = _make_signal(
                "uic", s["ticker"], r["horizon"], publish_date,
                spot=s["today_close"], expiry_date=r["expiry_date"],
                K_put_short=r["K_put_short"], K_call_short=r["K_call_short"],
                ror_pct=r["combined_ror_pct"],
                claimed_wr_pct=r["joint_win_rate_pct"],
                q=r.get("z_put_q"), width=r.get("width"),
                n_oos=r.get("n_test"),
            )
            if sig["id"] in existing_ids:
                continue
            log["signals"].append(sig)
            existing_ids.add(sig["id"])
            new_count += 1

    # Robust UIC signals
    for s in pub.get("ruic_signals", []):
        for r in s["ladder"]:
            sig = _make_signal(
                "ruic", s["ticker"], r["horizon"], publish_date,
                spot=s["today_close"], expiry_date=r["expiry_date"],
                K_put_short=r["K_put_short"], K_call_short=r["K_call_short"],
                ror_pct=r["combined_ror_pct"],
                claimed_wr_pct=r["pooled_wr_pct"],
                q=r.get("q_chosen"), width=r.get("width"),
                n_oos=r.get("n_test"),
            )
            if sig["id"] in existing_ids:
                continue
            log["signals"].append(sig)
            existing_ids.add(sig["id"])
            new_count += 1

    # Atomic IC signals
    for s in pub.get("ic_signals", []):
        for r in s["ladder"]:
            sig = _make_signal(
                "ic", s["ticker"], r["horizon"], publish_date,
                spot=s["today_close"], expiry_date=r["expiry_date"],
                K_put_short=r["K_put_short"], K_call_short=r["K_call_short"],
                ror_pct=r["combined_ror_pct"],
                claimed_wr_pct=r["joint_win_rate_pct"],
                q=r.get("q_chosen"), width=r.get("width"),
                n_oos=r.get("n_test"),
            )
            if sig["id"] in existing_ids:
                continue
            log["signals"].append(sig)
            existing_ids.add(sig["id"])
            new_count += 1

    # Resolve any pending signals
    resolved_count = 0
    for s in log["signals"]:
        if s["status"] == "pending":
            before = s["status"]
            _resolve_signal(s)
            if s["status"] != before:
                resolved_count += 1

    log["summary"] = _summarize(log["signals"])
    log["last_updated"] = datetime.utcnow().isoformat() + "Z"
    _save_log(log)

    print(f"Live log: {len(log['signals'])} total signals  "
          f"(+{new_count} new this run, +{resolved_count} resolved this run)")
    print(f"Live tier WRs:")
    for tier, stats in log["summary"]["by_tier"].items():
        wr = stats["live_win_rate_pct"]
        wr_str = f"{wr:.2f}%" if wr is not None else "n/a (none resolved)"
        print(f"  {tier}: {stats['wins']}W / {stats['losses']}L / "
              f"{stats['pending']}P  live={wr_str}")
    print(f"Wrote {_LIVE_LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
