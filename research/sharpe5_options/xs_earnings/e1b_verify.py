#!/usr/bin/env python3
"""Verify the EDGAR acceptance-time -> market-moving-session convention.

Known ground truth (widely documented reporting habits):
  after the close : AAPL MSFT NFLX AMZN GOOGL META TSLA NVDA INTC AMD QCOM
  before the open : JPM GS MS BAC WFC C JNJ PG KO MCD UPS CAT XOM CVX WMT HD
If the convention is right, per-symbol median acceptance hour lands >=16 for the
first group and <9:30 for the second.
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

AFTER = ["AAPL", "MSFT", "NFLX", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "INTC", "AMD", "QCOM"]
BEFORE = ["JPM", "GS", "MS", "BAC", "WFC", "C", "JNJ", "PG", "KO", "MCD", "UPS", "CAT",
          "XOM", "CVX", "WMT", "HD"]


def main():
    d = pd.read_parquet(os.path.join(HERE, "earnings_8k.parquet"))
    d = d[d.filingDate >= "2019-01-01"].copy()
    acc = d.acceptance.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    d["hour"] = acc.dt.hour + acc.dt.minute / 60.0
    d["hour_utc"] = d.acceptance.dt.hour + d.acceptance.dt.minute / 60.0

    print("EDGAR acceptanceDateTime is UTC; converted to America/New_York.")
    print(f"raw-UTC medians would be {d.hour_utc.median():.2f}h vs ET {d.hour.median():.2f}h\n")
    print(f"{'expected':<14}{'sym':<8}{'n':>4}{'median_h':>10}{'p10':>8}{'p90':>8}  verdict")
    ok = bad = 0
    for grp, syms in (("AFTER close", AFTER), ("BEFORE open", BEFORE)):
        for s in syms:
            x = d[d.symbol == s]
            if not len(x):
                continue
            m, p10, p90 = x.hour.median(), x.hour.quantile(.1), x.hour.quantile(.9)
            good = (m >= 16.0) if grp == "AFTER close" else (m < 9.5)
            ok, bad = ok + good, bad + (not good)
            print(f"{grp:<14}{s:<8}{len(x):>4}{m:>10.2f}{p10:>8.2f}{p90:>8.2f}  "
                  f"{'OK' if good else '** MISMATCH **'}")
    print(f"\nconvention verified on {ok}/{ok+bad} known cases")

    h = d.hour.dropna()
    print("\nfull-sample acceptance-hour distribution:")
    print(f"  before 09:30 : {(h < 9.5).mean():6.1%}")
    print(f"  09:30-16:00  : {((h >= 9.5) & (h < 16)).mean():6.1%}")
    print(f"  16:00+       : {(h >= 16).mean():6.1%}")


if __name__ == "__main__":
    main()
