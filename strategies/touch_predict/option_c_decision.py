"""Decision script: did walk-forward calibration validate any combo
strongly enough to ship as 'Certified'?

Reads results/option_c_walkforward.json and prints the verdict.

A combo SHIPS if:
  - n ≥ 100 OOS predictions across cutoff years
  - empirical win rate ≥ 0.98
  - every yearly bucket also ≥ 0.95 (no single year had a meltdown)

If multiple combos qualify, we prefer the one with:
  - highest min_yearly_win_rate (most robust)
  - then highest n (most useful)
"""
from __future__ import annotations

import json
import os
import sys


def main(json_path: str | None = None) -> int:
    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "option_c_walkforward.json",
        )
    with open(json_path) as fh:
        d = json.load(fh)
    summary = d["summary"]

    print(f"Walk-forward results from: {json_path}")
    print(f"Cutoff years: {d['cutoff_years']}")
    print(f"Tunable settings: {d.get('tunable')}")
    print()

    # Compute min yearly win rate per combo
    enriched = {}
    for cname, s in summary.items():
        by_year = s.get("by_year", {})
        yearly_rates = []
        for y, info in by_year.items():
            if info["n"] >= 10:
                yearly_rates.append(info["w"] / info["n"])
        min_yr = min(yearly_rates) if yearly_rates else None
        enriched[cname] = {
            **s,
            "min_yearly_wr": min_yr,
            "yearly_n_buckets": len(yearly_rates),
        }

    print(f"{'combo':<14} {'n':>7} {'win%':>7} {'min_yr%':>9} {'#yrs':>5}")
    print("-" * 50)
    sorted_combos = sorted(
        enriched.items(),
        key=lambda kv: (-(kv[1]["win_rate"] or 0), -(kv[1]["min_yearly_wr"] or 0)),
    )
    for cname, s in sorted_combos:
        wr = (s["win_rate"] or 0) * 100
        myr = (s["min_yearly_wr"] or 0) * 100 if s["min_yearly_wr"] is not None else 0
        print(f"{cname:<14} {s['n']:>7} {wr:>6.2f}% {myr:>8.2f}% {s['yearly_n_buckets']:>5}")

    print()
    qualifying = [
        (cname, s) for cname, s in enriched.items()
        if s["n"] >= 100
        and (s["win_rate"] or 0) >= 0.98
        and s["min_yearly_wr"] is not None
        and s["min_yearly_wr"] >= 0.95
    ]
    if qualifying:
        # Pick best: highest min yearly WR, then n
        qualifying.sort(key=lambda kv: (-(kv[1]["min_yearly_wr"]), -kv[1]["n"]))
        best_name, best = qualifying[0]
        print(f"VERDICT: SHIP")
        print(f"  Best combo: {best_name}")
        print(f"  n = {best['n']}, win_rate = {best['win_rate']*100:.2f}%, "
              f"min_yearly = {best['min_yearly_wr']*100:.2f}%")
        return 0
    else:
        print(f"VERDICT: DO NOT SHIP — no combo meets the bar")
        print(f"  Required: n ≥ 100, win_rate ≥ 98%, every-year ≥ 95%")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
