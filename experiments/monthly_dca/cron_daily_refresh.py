"""Cron-friendly daily refresh for the v5 PIT-S&P-500 strategy webapp.

This is a thin wrapper around experiments/monthly_dca/v5/cron_daily_refresh_v5.py.
Kept at the original path so existing cron jobs and CI continue to work.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from experiments.monthly_dca.v5.cron_daily_refresh_v5 import main
    sys.exit(main())
