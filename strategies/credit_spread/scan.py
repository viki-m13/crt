"""Convenience driver: run the research pipeline and publish the lean
signal file into the /spreads/ webapp data directory so the page can
load it. Intended to be called daily after the main daily_scan.
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


def main() -> int:
    os.makedirs(WEB_DATA, exist_ok=True)
    env = os.environ.copy()
    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "research.py")],
        env=env,
    )
    if rc != 0:
        print(f"research.py exited non-zero: {rc}", file=sys.stderr)
        return rc

    src = os.path.join(RESULTS, "signals.json")
    dst = os.path.join(WEB_DATA, "signals.json")
    shutil.copyfile(src, dst)
    print(f"Published {dst}")

    # Update the live signal log — append any new signal rungs (dedup
    # on publish_date+ticker+side+horizon) and resolve any whose expiry
    # date has now been reached.
    # Import lazily so the scan.py module can be imported without numpy
    # when used just for summary printing.
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
        f"  put:      elig={put.get('n_eligible')} tests={put.get('pooled_wins',0)+put.get('pooled_losses',0)} "
        f"losses={put.get('pooled_losses')}"
    )
    print(
        f"  call:     elig={call.get('n_eligible')} tests={call.get('pooled_wins',0)+call.get('pooled_losses',0)} "
        f"losses={call.get('pooled_losses')}"
    )
    print(
        f"  combined: elig={combined.get('n_eligible')} tests={combined.get('pooled_wins',0)+combined.get('pooled_losses',0)} "
        f"losses={combined.get('pooled_losses')} win_rate={combined.get('pooled_win_rate')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
