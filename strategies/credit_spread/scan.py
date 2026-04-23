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

    # Also drop a last_run marker for the webapp banner.
    with open(os.path.join(WEB_DATA, "last_run.txt"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))

    # Summary for the log
    with open(dst, "r") as fh:
        blob = json.load(fh)
    s = blob.get("summary", {})
    print(
        f"  signals={s.get('n_tickers_eligible')}  "
        f"pooled_tests={s.get('pooled_wins', 0) + s.get('pooled_losses', 0)}  "
        f"pooled_win_rate={s.get('pooled_win_rate')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
