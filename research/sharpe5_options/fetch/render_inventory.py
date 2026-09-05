#!/usr/bin/env python3
"""Render cache/inventory.json into the markdown tables used by
docs/OPTION_DATA_INVENTORY.md. Prints to stdout."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "cache", "inventory.json")))

KS = ["<0.90", "0.90-0.95", "0.95-0.98", "0.98-1.02", "1.02-1.05", "1.05-1.10", ">1.10"]
DTE = ["0-7", "8-14", "15-30", "31-60", "61-90", "90+"]


def grid_md(g, title, val="med", fmt="{:.1f}"):
    print(f"\n**{title}**\n")
    print("| K/S \\ DTE | " + " | ".join(DTE) + " |")
    print("|---|" + "---|" * len(DTE))
    for k in KS:
        cells = []
        for d in DTE:
            c = g.get(f"{k}|{d}")
            cells.append(fmt.format(c[val]) if c else "-")
        print(f"| {k} | " + " | ".join(cells) + " |")


def grid_n(g, title):
    print(f"\n**{title}**\n")
    print("| K/S \\ DTE | " + " | ".join(DTE) + " |")
    print("|---|" + "---|" * len(DTE))
    for k in KS:
        cells = []
        for d in DTE:
            c = g.get(f"{k}|{d}")
            cells.append(f"{c['n']:,}" if c else "-")
        print(f"| {k} | " + " | ".join(cells) + " |")


t = R["totals"]
n = t["rows"]
print(f"dates={R['n_dates']} {R['first']}..{R['last']} rows={n:,} "
      f"two_sided={t['two_sided']:,} ({100*t['two_sided']/n:.2f}%)")
print(f"zero_bid={t['zero_bid']:,} ({100*t['zero_bid']/n:.2f}%) "
      f"zero_ask={t['zero_ask']:,} crossed={t['crossed']:,} locked={t['locked']:,}")
print(f"iv_null={t['iv_null']:,} greek_null={t['greek_null']:,} bid_null={t['bid_null']:,}")
print(f"stale_frac={R['stale_frac']:.4f} over {R['stale_den']:,} contract-day pairs")
print(f"mean expiries per symbol-date = {R['mean_exp_per_sym_date']:.2f}")
print(f"missing business days: {len(R['missing_bdays'])}")

print("\n## per-year rows")
print("| year | chain rows | dates | median spread %% of mid | p90 |")
print("|---|---:|---:|---:|---:|")
dates_per_year = {}
for d in R["dates"]:
    dates_per_year[d[:4]] = dates_per_year.get(d[:4], 0) + 1
for y in sorted(R["per_year"]):
    by = R["by_year"].get(y) or R["by_year"].get(int(y)) or {}
    print(f"| {y} | {R['per_year'][y]:,} | {dates_per_year.get(y,0)} | "
          f"{by.get('med',float('nan')):.2f} | {by.get('p90',float('nan')):.1f} |")

print("\n## per symbol")
print("| symbol | rows | obs dates | first | last | two-sided % | zero-bid % | crossed | median spread % of mid |")
print("|---|---:|---:|---|---|---:|---:|---:|---:|")
for s in sorted(R["per_sym"], key=lambda x: -R["per_sym"][x]["rows"]):
    d = R["per_sym"][s]
    ms = R["per_sym_med_spread"].get(s)
    print(f"| {s} | {d['rows']:,} | {d['dates']} | {d['first']} | {d['last']} | "
          f"{100*d['two_sided']/d['rows']:.2f} | {100*d['zero_bid']/d['rows']:.2f} | "
          f"{d['crossed']} | {ms:.2f} |" if ms is not None else
          f"| {s} | {d['rows']:,} | {d['dates']} | {d['first']} | {d['last']} | "
          f"{100*d['two_sided']/d['rows']:.2f} | {100*d['zero_bid']/d['rows']:.2f} | {d['crossed']} | - |")

grid_md(R["grid"], "ALL SYMBOLS - median bid-ask spread as % of mid")
grid_md(R["grid"], "ALL SYMBOLS - 90th pct spread as % of mid", val="p90")
grid_md(R["grid"], "ALL SYMBOLS - median absolute spread ($)", val="med_abs", fmt="{:.2f}")
grid_n(R["grid"], "ALL SYMBOLS - sample size (two-sided quotes)")
grid_md(R["spy_grid"], "SPY ONLY - median bid-ask spread as % of mid")
grid_md(R["spy_grid"], "SPY ONLY - median absolute spread ($)", val="med_abs", fmt="{:.2f}")
grid_n(R["spy_grid"], "SPY ONLY - sample size")

print("\n## by |delta|")
print("| abs(delta) % | n | median spread % | p90 % | median $ spread | median mid |")
print("|---|---:|---:|---:|---:|---:|")
for k, v in R["by_delta"].items():
    print(f"| {k} | {v['n']:,} | {v['med']:.2f} | {v['p90']:.1f} | {v['med_abs']:.2f} | {v['med_mid']:.2f} |")

print("\n## by DTE (all moneyness)")
print("| DTE | n | median spread % | mean % | p90 % | median $ spread | median mid |")
print("|---|---:|---:|---:|---:|---:|---:|")
for k in DTE:
    v = R["by_dte"].get(k)
    if v:
        print(f"| {k} | {v['n']:,} | {v['med']:.2f} | {v['mean']:.2f} | {v['p90']:.1f} | {v['med_abs']:.2f} | {v['med_mid']:.2f} |")

print("\n## by moneyness (all DTE)")
print("| K/S | n | median spread % | mean % | p90 % | median $ spread | median mid |")
print("|---|---:|---:|---:|---:|---:|---:|")
for k in KS:
    v = R["by_ks"].get(k)
    if v:
        print(f"| {k} | {v['n']:,} | {v['med']:.2f} | {v['mean']:.2f} | {v['p90']:.1f} | {v['med_abs']:.2f} | {v['med_mid']:.2f} |")

print(f"\ncall median spread %: {R['call_med']:.2f}   put median: {R['put_med']:.2f}")
print("\n## SPY ATM (K/S 0.98-1.02) by DTE")
for k, v in R["spy_atm_by_dte"].items():
    print(f"- {k} DTE: n={v['n']:,} median {v['med']:.2f}% of mid (${v['med_abs']:.2f} on ${v['med_mid']:.2f} mid)")

v = R["vol"]
print(f"\n## volatility_history: files={v['files']} rows={v['rows']:,} symbols={len(v['syms'])}")
print("cols: " + ", ".join(v["cols"]))
print("null counts: " + ", ".join(f"{k}={x:,}" for k, x in v["nulls"].items() if x))

mb = R["missing_bdays"]
print(f"\nmissing business days ({len(mb)}), first 40: {mb[:40]}")
