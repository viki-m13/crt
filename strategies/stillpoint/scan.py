"""Stillpoint convenience driver: run research and publish to the webapp.

Reads from `strategies/stillpoint/results/stillpoint_signals.json` and
copies it into `spreads/docs/data/stillpoint_signals.json`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PUBLISH_DIR = os.path.join(_REPO_ROOT, "spreads", "docs", "data")


def main() -> int:
    rc = subprocess.call([sys.executable, os.path.join(_HERE, "research.py")])
    if rc != 0:
        return rc
    src = os.path.join(_HERE, "results", "stillpoint_signals.json")
    if not os.path.exists(src):
        print(f"[ERR] expected research output not found: {src}", file=sys.stderr)
        return 1
    os.makedirs(_PUBLISH_DIR, exist_ok=True)
    dst = os.path.join(_PUBLISH_DIR, "stillpoint_signals.json")
    shutil.copyfile(src, dst)
    print(f"Published {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
