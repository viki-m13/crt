"""Build the authoritative point-in-time Nasdaq-100 membership panel.

Source: jmccarrell/n100tickers (https://github.com/jmccarrell/n100tickers)
        — a community-maintained authoritative PIT NDX history sourced
        from Nasdaq's own annual reconstitution announcements and
        intra-year change press releases. Accurate coverage from
        Jan 1 2015 onward.

Output:  experiments/monthly_dca/v5/qqq_pit/ndx_pit_membership_monthly.parquet
         with columns (asof, ticker) — one row per (month-end, ticker)
         where the ticker was a NDX member on that month-end.

Honesty note: coverage starts 2015-01-01. Any v5 walk-forward split
that begins earlier (A1/A2/A3 expanding, R1 GFC, R2) cannot be tested
on PIT NDX with this dataset.
"""
from __future__ import annotations
import json
import sys
import urllib.request
import ssl
import subprocess
from pathlib import Path
from datetime import date

import yaml
import numpy as np
import pandas as pd


class _NoBoolLoader(yaml.SafeLoader):
    """SafeLoader that DOES NOT coerce 'ON'/'OFF'/'YES'/'NO' etc to bool.

    The Nasdaq-100 has constituent 'ON' (onsemi) and yaml 1.1 treats it
    as boolean True. This loader strips that resolver so all bare-word
    tickers stay as strings.
    """


# Drop the bool resolver so 'on', 'On', 'ON', 'true', 'yes', 'no' stay strings.
_NoBoolLoader.yaml_implicit_resolvers = {
    k: [(tag, regex) for tag, regex in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

ROOT = Path(__file__).resolve().parents[4]
V5_DIR = ROOT / "experiments" / "monthly_dca" / "v5"
QQQ_DIR = V5_DIR / "qqq_pit"
N100_REPO = Path("/tmp/n100tickers")
N100_YAML_DIR = N100_REPO / "src" / "nasdaq_100_ticker_history"


def ensure_n100_repo() -> None:
    if N100_REPO.exists():
        return
    print("Cloning jmccarrell/n100tickers...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/jmccarrell/n100tickers.git",
         str(N100_REPO)],
        check=True,
    )


def fetch_current_ndx() -> list[str]:
    """Fetch the current Nasdaq-100 list from Nasdaq.com's public API.

    Used as a sanity-check / cross-validation against the yaml-derived
    'membership as of today'.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100?limit=200"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                  "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        j = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return [r["symbol"] for r in j["data"]["data"]["rows"]]


def load_year_yaml(year: int) -> dict:
    path = N100_YAML_DIR / f"n100-ticker-changes-{year}.yaml"
    with open(path) as f:
        return yaml.load(f, Loader=_NoBoolLoader)


def build_pit_membership_grid() -> pd.DataFrame:
    """Reconstruct the NDX member set on every month-end 2015-01-31..2026-05-31.

    Algorithm: start with `tickers_on_Jan_1` for each year. Walk forward
    day by day applying any same-day changes. Emit the membership snapshot
    at each month-end.
    """
    # Load all yamls
    yearly = {}
    for year in range(2015, 2027):
        try:
            yearly[year] = load_year_yaml(year)
        except FileNotFoundError:
            print(f"No yaml for {year}, stopping.")
            break

    # Build daily timeline: dict[date -> set[ticker]]
    asofs = pd.date_range("2015-01-31", "2026-05-31", freq="ME")
    rows = []
    # Track running set
    current = set(yearly[2015]["tickers_on_Jan_1"])
    pending_changes = []  # list of (date, removed_set, added_set)
    # Collate all changes across all years
    for year, doc in sorted(yearly.items()):
        # Each year's yaml may override Jan 1 — verify
        jan1 = set(doc["tickers_on_Jan_1"])
        # Apply year-boundary correction: at Jan 1 of `year`, set should equal jan1
        # (We trust the yaml's Jan 1 list more than the cumulative apply.)
        # For each year, after applying that year's changes to year-1's Jan1,
        # we should reach year's Dec 31 set ≈ (year+1)'s Jan 1.
        # We'll trust the year's tickers_on_Jan_1 as the canonical state on
        # Jan 1 of that year and seed from there.
        pass
    # Build the running membership by year
    asofs_per_year = {}
    for year, doc in sorted(yearly.items()):
        running = set(doc["tickers_on_Jan_1"])
        changes = doc.get("changes") or {}
        # Pre-sort change dates
        change_dates = sorted(pd.Timestamp(d) for d in changes.keys())
        for asof in asofs:
            if asof.year != year:
                continue
            # Apply all changes on or before asof for this year
            for cd in change_dates:
                if cd > asof:
                    break
                spec = changes[cd.strftime("%Y-%m-%d")]
                removed = set(spec.get("difference") or [])
                added = set(spec.get("union") or [])
                running = (running - removed) | added
            for tk in sorted(running):
                rows.append({"asof": asof, "ticker": tk})
            # Reset running for next asof iteration in same year — we'll
            # re-apply from Jan 1 each iteration to avoid order issues
            running = set(doc["tickers_on_Jan_1"])
    return pd.DataFrame(rows)


def main():
    QQQ_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Build PIT Nasdaq-100 panel (jmccarrell/n100tickers source)")
    print("=" * 60)

    ensure_n100_repo()

    print("\n[1/3] Parsing yaml change files...")
    mem = build_pit_membership_grid()
    print(f"     → {len(mem)} (asof, ticker) rows; {mem['ticker'].nunique()} unique tickers")
    print(f"     → asof range: {mem['asof'].min().date()} → {mem['asof'].max().date()}")
    pool_per_month = mem.groupby("asof").size()
    print(f"     → pool size: median={int(pool_per_month.median())}, "
          f"min={int(pool_per_month.min())}, max={int(pool_per_month.max())}")

    print("\n[2/3] Cross-checking with live nasdaq.com API...")
    current_live = set(fetch_current_ndx())
    last_asof = mem["asof"].max()
    last_mem = set(mem[mem["asof"] == last_asof]["ticker"].tolist())
    only_live = current_live - last_mem
    only_yaml = last_mem - current_live
    print(f"     → live API: {len(current_live)} tickers")
    print(f"     → yaml last asof ({last_asof.date()}): {len(last_mem)} tickers")
    print(f"     → only in live (likely post-yaml additions): "
          f"{sorted(only_live) if only_live else 'none'}")
    print(f"     → only in yaml last asof: "
          f"{sorted(only_yaml) if only_yaml else 'none'}")

    print("\n[3/3] Loading panel data and checking coverage...")
    mr = pd.read_parquet(ROOT / "experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet")
    panel_set = set(mr.columns.tolist())
    all_ndx_tickers = set(mem["ticker"].tolist())
    have = sum(1 for t in all_ndx_tickers if t in panel_set)
    missing = sorted(all_ndx_tickers - panel_set)
    print(f"     → unique NDX tickers across 2015-2026: {len(all_ndx_tickers)}")
    print(f"     → in our price panel: {have}/{len(all_ndx_tickers)}")
    if missing:
        print(f"     → MISSING from panel ({len(missing)}): {missing[:30]}{'...' if len(missing) > 30 else ''}")

    # Save with only panel-resolvable rows for the simulator; also save raw
    mem.to_parquet(QQQ_DIR / "ndx_pit_membership_monthly_full.parquet")
    mem_panel = mem[mem["ticker"].isin(panel_set)].reset_index(drop=True)
    mem_panel.to_parquet(QQQ_DIR / "ndx_pit_membership_monthly.parquet")

    pool_panel = mem_panel.groupby("asof").size()
    print(f"\n     → panel-resolvable pool size: "
          f"median={int(pool_panel.median())}, min={int(pool_panel.min())}, "
          f"max={int(pool_panel.max())}")

    print(f"\nSaved:")
    print(f"  {QQQ_DIR / 'ndx_pit_membership_monthly_full.parquet'}  (all NDX names)")
    print(f"  {QQQ_DIR / 'ndx_pit_membership_monthly.parquet'}       (panel-resolvable subset)")


if __name__ == "__main__":
    main()
