"""Convenience driver: refresh data, run the research pipeline, and
publish the lean signal file into the /spreads/ webapp data directory
so the page can load it. Intended to be called daily.

Pipeline (all fail-closed):
  1. fetch_full_history.py CS_REFRESH=1 — rebuild the full-history
     panel from yfinance on one consistent adjustment basis (~3 min).
  2. fetch_optionable.py — refresh the listed-options map if it is
     older than 7 days (~8 min when it runs; otherwise instant).
  3. research.py with CS_DATA_DIR=cache_full — the v3 Sigma-Clear scan.
  4. Publish signals.json + update the append-only live log.

Set CS_SKIP_FETCH=1 to skip steps 1-2 (data already fresh).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
WEB_DATA = os.path.join(REPO, "spreads", "docs", "data")
CACHE_FULL = os.path.join(HERE, "cache_full")
OPTIONABLE = os.path.join(RESULTS, "optionable.json")
ADV = os.path.join(RESULTS, "adv.json")


def _map_age_days(path: str) -> float:
    """Age of a fetched map from its embedded as_of stamp (CI resets
    file mtimes, so mtime is unreliable)."""
    if not os.path.exists(path):
        return 999.0
    try:
        with open(path) as fh:
            as_of = json.load(fh).get("as_of")
        if as_of:
            return (time.time() - time.mktime(
                time.strptime(as_of, "%Y-%m-%dT%H:%M:%SZ"))) / 86400.0
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return 999.0


def main() -> int:
    os.makedirs(WEB_DATA, exist_ok=True)
    env = os.environ.copy()

    if env.get("CS_SKIP_FETCH") != "1":
        rc = subprocess.call(
            [sys.executable, os.path.join(HERE, "fetch_full_history.py")],
            env={**env, "CS_REFRESH": "1"},
        )
        if rc != 0:
            print(f"fetch_full_history.py exited non-zero: {rc}", file=sys.stderr)
            return rc
        # Fail-safe: if the fetch produced too few series (yfinance
        # outage / rate-limiting on the runner), abort WITHOUT running
        # the scan — better to keep yesterday's published signals than
        # to overwrite them with output from a crippled panel.
        n_cached = len([f for f in os.listdir(CACHE_FULL) if f.endswith(".json")]) \
            if os.path.isdir(CACHE_FULL) else 0
        if n_cached < 500:
            print(f"ABORT: only {n_cached} series in {CACHE_FULL} (<500); "
                  "keeping previous signals.", file=sys.stderr)
            return 1
        # Optionability + ADV maps: refresh weekly (both are sticky).
        # Age comes from the embedded as_of stamp, not file mtime — CI
        # checkouts reset mtimes on every run.
        if _map_age_days(OPTIONABLE) > 7:
            rc = subprocess.call(
                [sys.executable, os.path.join(HERE, "fetch_optionable.py")],
                env=env,
            )
            if rc != 0:
                print(f"fetch_optionable.py exited non-zero: {rc}", file=sys.stderr)
                return rc
        # ADV (underlying dollar-volume liquidity) map, weekly. Fail-soft:
        # if it can't refresh, reality.py simply skips the underlying
        # gate for missing names rather than blocking the whole scan.
        if _map_age_days(ADV) > 7:
            rc = subprocess.call(
                [sys.executable, os.path.join(HERE, "fetch_adv.py")],
                env=env,
            )
            if rc != 0:
                print(f"[WARN] fetch_adv.py exited non-zero: {rc} — "
                      "underlying-liquidity gate degraded this run.",
                      file=sys.stderr)

    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "research.py")],
        env={**env, "CS_DATA_DIR": CACHE_FULL},
    )
    if rc != 0:
        print(f"research.py exited non-zero: {rc}", file=sys.stderr)
        return rc

    # Tier 2 ("Vol-Alpha" GBM puts) — additive; a tier2 failure must
    # never block the Tier 1 publication, so this step is fail-soft.
    rc2 = subprocess.call(
        [sys.executable, os.path.join(HERE, "tier2.py"), "scan"],
        env={**env, "CS_DATA_DIR": CACHE_FULL},
    )
    if rc2 != 0:
        print(f"[WARN] tier2.py scan exited non-zero: {rc2} — "
              "publishing Tier 1 only.", file=sys.stderr)

    # Currently-open Conviction Picks (pending) — fail-soft.
    rc3 = subprocess.call(
        [sys.executable, os.path.join(HERE, "conviction_open.py")],
        env={**env, "CS_DATA_DIR": CACHE_FULL},
    )
    if rc3 != 0:
        print(f"[WARN] conviction_open.py exited non-zero: {rc3}",
              file=sys.stderr)

    src = os.path.join(RESULTS, "signals.json")
    dst = os.path.join(WEB_DATA, "signals.json")
    shutil.copyfile(src, dst)
    print(f"Published {dst}")

    # Update the live signal log — append any new signal rungs (dedup
    # on publish_date+ticker+side+horizon) and resolve any whose expiry
    # date has now been reached. Resolution must read the same fresh
    # full-history panel the scan used (the legacy docs/data/tickers
    # panel is no longer refreshed by this pipeline), so CS_DATA_DIR is
    # set BEFORE live_log/common are imported.
    os.environ["CS_DATA_DIR"] = CACHE_FULL
    sys.path.insert(0, HERE)
    from live_log import update_live_log  # noqa: E402
    log_path = os.path.join(WEB_DATA, "live_log.json")
    update_live_log(dst, log_path)

    # Also drop a last_run marker for the webapp banner.
    with open(os.path.join(WEB_DATA, "last_run.txt"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))

    # Summary for the log
    with open(dst, "r") as fh:
        blob = json.load(fh)
    s = blob.get("summary", {})
    combined = s.get("combined", {})
    put = s.get("put", {})
    call = s.get("call", {})
    print(
        f"  put:      certified={put.get('n_certified')} published={put.get('n_published')} "
        f"tests={put.get('pooled_wins',0)+put.get('pooled_losses',0)} losses={put.get('pooled_losses')}"
    )
    print(
        f"  call:     certified={call.get('n_certified')} published={call.get('n_published')} "
        f"tests={call.get('pooled_wins',0)+call.get('pooled_losses',0)} losses={call.get('pooled_losses')}"
    )
    print(
        f"  combined: certified={combined.get('n_certified')} published={combined.get('n_published')} "
        f"tests={combined.get('pooled_wins',0)+combined.get('pooled_losses',0)} losses={combined.get('pooled_losses')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
