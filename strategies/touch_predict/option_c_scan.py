"""Option C daily scan — run research + publish signals into the
spreads webapp data dir."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WEB_DATA = os.path.join(REPO, "spreads", "docs", "data")


def main() -> int:
    os.makedirs(WEB_DATA, exist_ok=True)

    # Make sure OHLCV data is fresh (v2_backfill is idempotent and fast).
    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "v2_backfill.py")]
    )
    if rc != 0:
        print(f"v2_backfill.py exited {rc}", file=sys.stderr)
        return rc

    # Run research
    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "option_c_research.py")]
    )
    if rc != 0:
        print(f"option_c_research.py exited {rc}", file=sys.stderr)
        return rc

    # Publish the output to the spreads webapp data dir
    src = os.path.join(HERE, "results", "option_c_signals.json")
    dst = os.path.join(WEB_DATA, "option_c_signals.json")
    shutil.copyfile(src, dst)
    print(f"Published {dst}")

    # Summary
    with open(dst, "r") as fh:
        blob = json.load(fh)
    s = blob.get("summary", {})
    print(
        f"  rules: {s.get('n_eligible_rules')}  "
        f"puts: {s.get('n_short_puts')}  calls: {s.get('n_short_calls')}  "
        f"live: {s.get('n_live_fires')}  "
        f"win: {s.get('overall_win_rate_pct', 0):.1f}%  "
        f"roi: {s.get('overall_roi_on_max_loss_pct', 0):.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
