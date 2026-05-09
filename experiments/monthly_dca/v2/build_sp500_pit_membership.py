"""Build a point-in-time S&P 500 membership panel.

Inputs:
  cache/v2/sp500_pit/sp500_hist_1996_2019.csv          — daily PIT 1996-01-02..2019-01-11
  cache/v2/sp500_pit/sp500_changes_since_2019.csv      — changes 2019-01-18..present (add,remove)
  cache/v2/sp500_pit/sp500_today.csv                   — sanity check current list

Output:
  cache/v2/sp500_pit/sp500_membership_monthly.parquet
    one row per (asof, ticker) for asof = each panel month-end 2003..2025
    columns: asof, ticker, in_sp500 (bool)

The historical csv encodes a ticker's last month as the suffix `-YYYYMM`. The
"clean" ticker that we match on is the prefix before that suffix.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"


SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:-(?P<yymm>\d{6}))?$")


def parse_ticker(tok: str) -> str:
    """Strip the `-YYYYMM` removal-month suffix if present."""
    m = SUFFIX_RE.match(tok)
    if not m:
        return tok
    base = m.group("base")
    yymm = m.group("yymm")
    # The historical CSV encodes a ticker's removal month as a -YYYYMM suffix.
    # We strip it and use the base symbol; the suffix only confirms removal.
    return base if yymm else tok


def load_historical_pit() -> pd.DataFrame:
    df = pd.read_csv(PIT / "sp500_hist_1996_2019.csv")
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for _, r in df.iterrows():
        d = r["date"]
        toks = [t.strip() for t in r["tickers"].split(",") if t.strip()]
        for t in toks:
            rows.append({"date": d, "ticker": parse_ticker(t)})
    out = pd.DataFrame(rows)
    print(f"[hist] {len(df)} dates, {len(out)} (date,ticker) rows, "
          f"{out['ticker'].nunique()} unique tickers")
    return out


def normalize_ticker(t: str) -> str:
    """Normalise to the form used in the price panel.

    The price panel uses Yahoo-style symbols: BRK.B, BF.B etc. The S&P CSV
    uses BRK.B (matching). Some tickers carry a -YYYYMM suffix in the
    historical csv that should already have been stripped.
    """
    return t.strip().upper()


def members_on_date_from_changes(
    base_date: pd.Timestamp, base_members: set[str],
    changes: pd.DataFrame, target: pd.Timestamp,
) -> set[str]:
    """Apply changes between base_date (exclusive) and target (inclusive)."""
    members = set(base_members)
    rel = changes[(changes["date"] > base_date) & (changes["date"] <= target)]
    for _, r in rel.iterrows():
        if isinstance(r["add"], str) and r["add"]:
            for t in r["add"].split(","):
                members.add(normalize_ticker(t))
        if isinstance(r["remove"], str) and r["remove"]:
            for t in r["remove"].split(","):
                members.discard(normalize_ticker(t))
    return members


def build_monthly_membership(
    panel_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """For each date in panel_dates, return the set of S&P 500 tickers."""
    hist = load_historical_pit()
    # Group history by date for fast lookup
    by_date = hist.groupby("date")["ticker"].apply(lambda s: set(map(normalize_ticker, s))).to_dict()
    hist_dates = sorted(by_date.keys())
    # Forward-fill: for any panel date <= last hist_date, find the last hist_date <= panel_date
    hist_idx = pd.DatetimeIndex(hist_dates)

    # Load post-2019 changes
    changes = pd.read_csv(PIT / "sp500_changes_since_2019.csv")
    changes["date"] = pd.to_datetime(changes["date"])
    changes = changes.sort_values("date").reset_index(drop=True)
    last_hist_date = hist_idx.max()
    last_hist_members = by_date[last_hist_date]
    print(f"[hist] last historical date: {last_hist_date.date()}, "
          f"members={len(last_hist_members)}")
    print(f"[changes] {len(changes)} change-events post {last_hist_date.date()}")

    rows = []
    for d in panel_dates:
        d = pd.Timestamp(d)
        if d <= last_hist_date:
            # find latest hist_date <= d
            pos = hist_idx.searchsorted(d, side="right") - 1
            if pos < 0:
                # before history begins
                continue
            members = by_date[hist_idx[pos]]
        else:
            members = members_on_date_from_changes(
                last_hist_date, last_hist_members, changes, d,
            )
        for t in members:
            rows.append({"asof": d, "ticker": t})
    out = pd.DataFrame(rows)
    return out


def main():
    # Build the panel month-ends. Use the predictions parquet to know which
    # asof dates we need.
    preds = pd.read_parquet(CACHE / "v2" / "ml_preds_v2.parquet")
    panel_dates = sorted(pd.to_datetime(preds["asof"].unique()))
    print(f"[panel] {len(panel_dates)} month-ends from {panel_dates[0].date()} "
          f"to {panel_dates[-1].date()}")

    member = build_monthly_membership(pd.DatetimeIndex(panel_dates))
    print(f"[member] {len(member)} (asof, ticker) rows, "
          f"{member['ticker'].nunique()} unique tickers")

    # Per-asof summary
    summary = member.groupby("asof").size().rename("n_members").to_frame()
    print("\n[summary] members per asof (head/tail):")
    print(summary.head().to_string())
    print(summary.tail().to_string())
    print(f"  mean={summary['n_members'].mean():.1f}, "
          f"min={summary['n_members'].min()}, "
          f"max={summary['n_members'].max()}")

    out_path = PIT / "sp500_membership_monthly.parquet"
    member.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    summary.to_csv(PIT / "sp500_membership_count.csv")


if __name__ == "__main__":
    main()
