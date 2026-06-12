"""Lightweight price-series backfiller for CreditFloor.

Updates only `series.dates` and `series.prices` in each
`docs/data/tickers/{TICKER}.json` file, leaving every other field
(analog outcomes, scores, etc.) untouched. CreditFloor consumes only
the close-price series, so this is enough to make its daily run
accurate without paying the cost of the full main-site scan.

Semantics
---------

  - Target end: the latest NYSE trading day on or before today (UTC).
  - Fetches adjusted close in one batched yfinance call per chunk.
  - If a ticker's existing series is already at the target end and has
    not shrunk, it is left alone (idempotent).
  - If a fetch fails for a ticker, that ticker's series is left intact
    and the run continues. Failures are counted and printed.
  - Fetches from the last existing date of the series (inclusive) so we
    normally only append; we don't rewrite historical closes on every
    run. This avoids the classic "yfinance changed a split/div
    adjustment so everything moved 0.3%" leakage into the model.
  - EXCEPTION — seam integrity check: the fetch window includes the
    last stored date, and the freshly-fetched adjusted close for that
    overlap day is compared against the stored value. If they disagree
    by more than SEAM_TOLERANCE (a split, or a large dividend/spinoff
    re-adjustment), appending would stitch incompatible adjustment
    bases together and fabricate a giant fake price jump (observed
    live: BKNG 25:1 and KLAC 10:1 splits in 2026-03 produced fake -96%
    and -90% "moves" in the stored series). In that case the ticker's
    ENTIRE series is re-fetched (period="max") on a consistent
    adjustment basis and rewritten from scratch.

Run
---

    python strategies/credit_spread/backfill_prices.py        # full universe
    CS_LIMIT=25 python .../backfill_prices.py                 # smoke test
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from common import TICKERS_DIR, list_tickers


CHUNK_SIZE = 60
RETRIES = 3
SLEEP_BETWEEN = 0.5  # seconds between chunk fetches; yfinance is rate-limited

# Max tolerated relative difference between the stored close and the
# freshly-fetched adjusted close for the SAME (overlap) day. Bigger ->
# the adjustment basis changed (split / big dividend / spinoff) and the
# series must be rebuilt from scratch instead of appended to.
SEAM_TOLERANCE = 0.03


def _read_series(path: str) -> tuple[dict, list[str], list[float]]:
    with open(path, "r") as fh:
        blob = json.load(fh)
    s = blob.get("series") or {}
    return blob, list(s.get("dates") or []), list(s.get("prices") or [])


def _write_series(path: str, blob: dict, dates: list[str], prices: list[float]) -> None:
    blob.setdefault("series", {})
    blob["series"]["dates"] = dates
    blob["series"]["prices"] = prices
    # Keep ancillary arrays (wash, final) trimmed to the same length so
    # the existing webapp doesn't choke; fill tail with None-safe values.
    for k in ("wash", "final"):
        arr = blob["series"].get(k)
        if isinstance(arr, list):
            if len(arr) < len(dates):
                arr.extend([0.0] * (len(dates) - len(arr)))
            elif len(arr) > len(dates):
                blob["series"][k] = arr[: len(dates)]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(blob, fh)
    os.replace(tmp, path)


def _latest_trading_day_ny() -> str:
    """Return the most recent NYSE trading day on or before today.

    Uses the `pandas_market_calendars` NYSE calendar so weekends,
    federal holidays, and any exchange special closures (e.g. storms,
    days of mourning) are all accounted for.
    """
    import pandas as pd
    import pandas_market_calendars as mcal

    today = pd.Timestamp(datetime.now(timezone.utc).date())
    nyse = mcal.get_calendar("NYSE")
    # Look back 14 calendar days to guarantee at least one trading day.
    sessions = nyse.valid_days(
        start_date=(today - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )
    if len(sessions) == 0:
        return today.strftime("%Y-%m-%d")
    return sessions[-1].strftime("%Y-%m-%d")


def _fetch_chunk(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Return adjusted-close DataFrame (index=date, columns=tickers).

    Uses `auto_adjust=True` so the returned "Close" is already split- and
    dividend-adjusted — matching how the original series was stored.
    """
    last_err = None
    for attempt in range(RETRIES):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            if df is None or df.empty:
                last_err = RuntimeError("empty dataframe")
                continue
            # If single ticker, columns are ('Open','High',...); wrap it.
            if len(tickers) == 1:
                close = df[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                # Multi-ticker: MultiIndex columns (field, ticker)
                close = df["Close"]
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            return close
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 ** attempt)
    print(f"  chunk fetch failed after {RETRIES} tries: {last_err}", file=sys.stderr)
    return None


def _fetch_full(ticker: str) -> tuple[list[str], list[float]] | None:
    """Re-fetch a ticker's entire history on a consistent adjustment
    basis. Used when the seam integrity check fails (split / large
    re-adjustment since the series was last written)."""
    last_err = None
    for attempt in range(RETRIES):
        try:
            df = yf.download(
                tickers=ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if df is None or df.empty:
                last_err = RuntimeError("empty dataframe")
                continue
            close = df["Close"]
            if hasattr(close, "columns"):  # MultiIndex leftover
                close = close[ticker]
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            close = close.dropna()
            close = close[close > 0]
            dates = [idx.strftime("%Y-%m-%d") for idx in close.index]
            prices = [float(v) for v in close.values]
            return dates, prices
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 ** attempt)
    print(f"  full refetch failed for {ticker}: {last_err}", file=sys.stderr)
    return None


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    target_end = _latest_trading_day_ny()
    # yfinance "end" is exclusive; fetch up to the day AFTER target.
    fetch_end = (np.datetime64(target_end, "D") + 1).astype(str)

    print(f"Backfilling {len(tickers)} tickers to close of {target_end}")

    t0 = time.time()
    updated = 0
    unchanged = 0
    failed = 0
    rebuilt = 0

    for chunk_start in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[chunk_start : chunk_start + CHUNK_SIZE]
        # Fetch start = earliest *stored last day* across the chunk so the
        # fetch window overlaps each series by one day (seam check). The
        # short-circuit below still keys off the earliest missing day.
        chunk_starts = []
        fetch_starts = []
        existing = {}
        for t in chunk:
            p = os.path.join(TICKERS_DIR, f"{t}.json")
            if not os.path.exists(p):
                failed += 1
                continue
            blob, dates, prices = _read_series(p)
            if dates:
                last = dates[-1]
                # day AFTER last recorded date (for the no-op short-circuit)
                nxt = (np.datetime64(last, "D") + 1).astype(str)
                chunk_starts.append(nxt)
                # fetch from the last recorded date itself (overlap day)
                fetch_starts.append(last)
            else:
                chunk_starts.append("2015-01-02")
                fetch_starts.append("2015-01-02")
            existing[t] = (p, blob, dates, prices)

        if not existing:
            continue

        min_start = min(chunk_starts)
        # If the earliest "next" date is already past the target, nothing to do.
        if np.datetime64(min_start, "D") > np.datetime64(target_end, "D"):
            unchanged += len(existing)
            continue

        close_df = _fetch_chunk(list(existing.keys()), min(fetch_starts), fetch_end)
        if close_df is None:
            failed += len(existing)
            continue

        for t, (p, blob, dates, prices) in existing.items():
            col = close_df[t] if t in close_df.columns else None
            if col is None:
                failed += 1
                continue
            # drop NaN rows
            col = col.dropna()
            if col.empty:
                unchanged += 1
                continue
            last_existing = dates[-1] if dates else ""

            # Seam integrity check: the freshly-fetched adjusted close
            # for the stored last day must (approximately) equal the
            # stored value. A big difference means the adjustment basis
            # moved (split/dividend/spinoff) and appending would create
            # a fake price jump — rebuild the whole series instead.
            if dates:
                overlap = col[col.index == pd.Timestamp(last_existing)]
                if len(overlap):
                    fetched_last = float(overlap.iloc[-1])
                    stored_last = float(prices[-1])
                    if stored_last > 0 and abs(fetched_last / stored_last - 1.0) > SEAM_TOLERANCE:
                        full = _fetch_full(t)
                        if full is None:
                            failed += 1
                            continue
                        f_dates, f_prices = full
                        _write_series(p, blob, f_dates, f_prices)
                        rebuilt += 1
                        print(
                            f"  [seam] {t}: stored {stored_last:.4f} vs fetched "
                            f"{fetched_last:.4f} on {last_existing} — series rebuilt "
                            f"({len(f_dates)} days)"
                        )
                        continue

            new_dates, new_prices = [], []
            for idx, val in col.items():
                ds = idx.strftime("%Y-%m-%d")
                if dates and ds <= last_existing:
                    continue
                new_dates.append(ds)
                new_prices.append(float(val))
            if not new_dates:
                unchanged += 1
                continue
            merged_dates = dates + new_dates
            merged_prices = prices + new_prices
            _write_series(p, blob, merged_dates, merged_prices)
            updated += 1

        print(
            f"  chunk {chunk_start // CHUNK_SIZE + 1}  "
            f"updated={updated} unchanged={unchanged} failed={failed} rebuilt={rebuilt}  "
            f"elapsed={time.time()-t0:.1f}s"
        )
        time.sleep(SLEEP_BETWEEN)

    print()
    print(f"Done. updated={updated} unchanged={unchanged} failed={failed} rebuilt={rebuilt}")
    return 0 if failed == 0 else 0  # non-fatal; partial success is OK


if __name__ == "__main__":
    sys.exit(main())
