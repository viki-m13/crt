#!/usr/bin/env python3
"""
Extend ticker series data back to ~2015 by fetching historical prices
from Yahoo Finance's public chart API and computing washout scores.
No yfinance dependency needed.
"""

import json, os, sys, time, math, urllib.request, urllib.error
from datetime import datetime, timezone

TICKER_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "tickers")
FULL_JSON = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "full.json")

# Target start: Jan 1, 2015
TARGET_START = datetime(2015, 1, 2, tzinfo=timezone.utc)

# Lookbacks matching daily_scan.py
LB_LT = 252
LB_ST = 63
BETA_LB = 126
ATR_N = 14
DD_THR = 0.25
POS_THR = 0.20
GATE_DD_SCALE = 0.12
GATE_POS_SCALE = 0.10


def fetch_yahoo_history(ticker, period1_ts, period2_ts):
    """Fetch daily OHLCV from Yahoo Finance chart API."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1_ts}&period2={period2_ts}"
        f"&interval=1d&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            result = data["chart"]["result"][0]
            ts_list = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            adj = result["indicators"]["adjclose"][0]["adjclose"]
            rows = []
            for i, ts in enumerate(ts_list):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
                o = quote["open"][i]
                h = quote["high"][i]
                lo = quote["low"][i]
                c = quote["close"][i]
                v = quote["volume"][i]
                a = adj[i]
                if c is not None and h is not None and lo is not None:
                    rows.append({"date": date_str, "open": o, "high": h, "low": lo,
                                 "close": c, "volume": v, "adj": a})
            return rows
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"  FAILED to fetch {ticker}: {e}")
                return None
    return None


def rolling_max(arr, window):
    out = [None] * len(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        chunk = [v for v in arr[start:i+1] if v is not None]
        out[i] = max(chunk) if chunk else None
    return out


def rolling_min(arr, window):
    out = [None] * len(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        chunk = [v for v in arr[start:i+1] if v is not None]
        out[i] = min(chunk) if chunk else None
    return out


def rolling_mean(arr, window):
    out = [None] * len(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        chunk = [v for v in arr[start:i+1] if v is not None]
        out[i] = sum(chunk) / len(chunk) if len(chunk) >= window else None
    return out


def compute_washout(highs, lows, closes, volumes, opens, spy_closes=None):
    """Compute washout meter matching daily_scan.py formula."""
    n = len(closes)
    wash = [0.0] * n

    hi_lt = rolling_max(highs, LB_LT)
    lo_lt = rolling_min(lows, LB_LT)

    # Compute beta-adjusted (idiosyncratic) components if SPY available
    has_spy = spy_closes is not None and len(spy_closes) == n

    for i in range(LB_LT + BETA_LB, n):
        c = closes[i]
        h_lt = hi_lt[i]
        l_lt = lo_lt[i]
        if c is None or h_lt is None or l_lt is None or h_lt == 0:
            continue

        rng = h_lt - l_lt
        if rng <= 0:
            continue

        pos_lt = (c - l_lt) / rng
        dd_lt = 1.0 - c / h_lt

        pos_lt = max(0, min(1, pos_lt))
        dd_lt = max(0, min(1, dd_lt))

        struct = 0.6 * dd_lt + 0.4 * (1 - pos_lt)
        struct = max(0, min(1, struct))

        # Idiosyncratic: approximate with structural if no spy
        if has_spy:
            # Compute simple beta adjustment
            stock_rets = []
            spy_rets = []
            for j in range(max(1, i - BETA_LB), i + 1):
                if closes[j] and closes[j-1] and spy_closes[j] and spy_closes[j-1]:
                    stock_rets.append(closes[j] / closes[j-1] - 1)
                    spy_rets.append(spy_closes[j] / spy_closes[j-1] - 1)

            if len(stock_rets) > 30:
                mean_s = sum(stock_rets) / len(stock_rets)
                mean_m = sum(spy_rets) / len(spy_rets)
                cov = sum((s - mean_s) * (m - mean_m) for s, m in zip(stock_rets, spy_rets)) / len(stock_rets)
                var_m = sum((m - mean_m) ** 2 for m in spy_rets) / len(spy_rets)
                beta = cov / var_m if var_m > 0 else 1.0
                beta = max(0, min(3, beta))

                # Idiosyncratic price = cumulative stock return - beta * cumulative market return
                idio_prices = []
                base_c = closes[i - LB_LT] if closes[i - LB_LT] else c
                base_s = spy_closes[i - LB_LT] if spy_closes[i - LB_LT] else spy_closes[i]
                for j in range(i - LB_LT, i + 1):
                    if closes[j] and base_c and spy_closes[j] and base_s:
                        stock_cum = closes[j] / base_c - 1
                        mkt_cum = spy_closes[j] / base_s - 1
                        idio_prices.append(1 + stock_cum - beta * mkt_cum)
                    else:
                        idio_prices.append(None)

                valid_idio = [p for p in idio_prices if p is not None]
                if valid_idio:
                    idio_hi = max(valid_idio)
                    idio_lo = min(valid_idio)
                    idio_cur = idio_prices[-1] if idio_prices[-1] is not None else valid_idio[-1]
                    idio_rng = idio_hi - idio_lo
                    if idio_rng > 0 and idio_hi > 0:
                        idio_pos_lt = (idio_cur - idio_lo) / idio_rng
                        idio_dd_lt = 1.0 - idio_cur / idio_hi
                        idio_pos_lt = max(0, min(1, idio_pos_lt))
                        idio_dd_lt = max(0, min(1, idio_dd_lt))
                        idio = 0.6 * idio_dd_lt + 0.4 * (1 - idio_pos_lt)
                        idio = max(0, min(1, idio))
                    else:
                        idio = struct
                else:
                    idio = struct
            else:
                idio = struct
        else:
            idio = struct

        # Capital component (simplified: just ATR-based)
        prev_close = closes[i-1] if closes[i-1] else c
        tr_val = max(
            abs(highs[i] - lows[i]) if highs[i] and lows[i] else 0,
            abs(highs[i] - prev_close) if highs[i] else 0,
            abs(lows[i] - prev_close) if lows[i] else 0,
        )
        atr_vals = []
        for j in range(max(0, i - ATR_N + 1), i + 1):
            pc = closes[j-1] if j > 0 and closes[j-1] else (closes[j] or 0)
            tr_j = max(
                abs((highs[j] or 0) - (lows[j] or 0)),
                abs((highs[j] or 0) - pc),
                abs((lows[j] or 0) - pc),
            )
            atr_vals.append(tr_j)
        atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0
        atr_pct = atr / c if c > 0 else 0

        # Simplified cap: normalize atr_pct (typical range 0.5%-4%)
        cap_raw = max(0, min(1, (atr_pct - 0.005) / 0.035))

        # Bottom confirmation gate
        dd_sig = 1.0 / (1.0 + math.exp(-(dd_lt - DD_THR) / GATE_DD_SCALE))
        pos_sig = 1.0 / (1.0 + math.exp(-((1 - pos_lt) - (1 - POS_THR)) / GATE_POS_SCALE))
        confirm = dd_sig * pos_sig

        raw = 0.55 * struct + 0.30 * idio + 0.15 * cap_raw
        raw = max(0, min(1, raw))
        w = raw * confirm
        w = max(0, min(1, w))
        wash[i] = round(w * 100, 2)

    return wash


def extend_ticker(ticker, existing_detail, spy_data_map):
    """Extend a ticker's series data back to ~2015."""
    series = existing_detail.get("series", {})
    existing_dates = series.get("dates", [])
    if not existing_dates:
        return False

    first_existing = existing_dates[0]
    target_str = TARGET_START.strftime("%Y-%m-%d")

    # Check if data is already extended but final scores need recomputing
    existing_final = series.get("final", [])
    needs_rescore = False
    if first_existing <= target_str:
        # Check if final scores in early period are all 0 (need recomputing)
        early_final = existing_final[:min(100, len(existing_final))]
        if all(f == 0 or f is None for f in early_final):
            needs_rescore = True
        if not needs_rescore:
            print(f"  {ticker}: already starts at {first_existing}, skipping")
            return False

    # If only rescoring (data already extended), recompute final scores in-place
    if needs_rescore:
        print(f"  {ticker}: rescoring opportunity scores for extended period...")
        quality = existing_detail.get("quality", 50) or 50
        wash_arr = series.get("wash", [])
        new_final = []
        for i in range(len(existing_dates)):
            w = wash_arr[i] if i < len(wash_arr) else 0
            old_f = existing_final[i] if i < len(existing_final) else 0
            if old_f and old_f > 0:
                new_final.append(old_f)  # keep existing scores
            elif w and w > 5:
                est_win_prob = 0.50 + 0.45 / (1 + math.exp(-(w - 30) / 12))
                new_final.append(round(quality * est_win_prob, 2))
            else:
                new_final.append(0.0)
        series["final"] = new_final
        print(f"  {ticker}: rescored {sum(1 for f in new_final if f > 0)} days with opportunity scores")
        return True

    print(f"  {ticker}: fetching history from {target_str} to {first_existing}...")
    period1 = int(TARGET_START.timestamp())
    # Fetch up to a day before existing data starts
    first_dt = datetime.strptime(first_existing, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    period2 = int(first_dt.timestamp())

    rows = fetch_yahoo_history(ticker, period1, period2)
    if not rows:
        print(f"  {ticker}: no historical data fetched")
        return False

    # Filter out rows that overlap with existing data
    new_rows = [r for r in rows if r["date"] < first_existing]
    if not new_rows:
        print(f"  {ticker}: no new dates to add")
        return False

    # We need a continuous series: new rows + existing rows
    # For wash computation, we need the FULL series
    all_dates = [r["date"] for r in new_rows] + existing_dates
    all_prices = [r["adj"] for r in new_rows] + series.get("prices", [])
    all_highs = [r["high"] for r in new_rows]
    all_lows = [r["low"] for r in new_rows]
    all_closes_raw = [r["close"] for r in new_rows]
    all_opens = [r["open"] for r in new_rows]
    all_volumes = [r["volume"] for r in new_rows]

    # For the existing portion, we don't have OHLCV separately - approximate from adj prices
    for i, p in enumerate(series.get("prices", [])):
        all_highs.append(p * 1.005 if p else None)  # approximate
        all_lows.append(p * 0.995 if p else None)
        all_closes_raw.append(p)
        all_opens.append(p)
        all_volumes.append(1_000_000)  # placeholder

    # Get SPY closes aligned to same dates
    spy_closes = [spy_data_map.get(d) for d in all_dates]

    # Compute washout for the full series
    full_wash = compute_washout(all_highs, all_lows, all_prices, all_volumes, all_opens, spy_closes)

    # Use existing wash values where available (they're more accurate)
    existing_wash = series.get("wash", [])
    n_new = len(new_rows)
    for i, w in enumerate(existing_wash):
        if w is not None:
            full_wash[n_new + i] = w

    # Build the opportunity score (final/conviction) series for extended period.
    # Opportunity score = quality × win_probability.
    # For the new historical period, approximate: when washout is high (deep pullback),
    # win probability is typically high (70-95%), so opportunity score ≈ quality × estimated_win_prob.
    # When washout is 0, there's no opportunity signal.
    quality = existing_detail.get("quality", 50) or 50
    existing_final = series.get("final", [])
    new_final = []
    for i in range(n_new):
        w = full_wash[i]
        if w and w > 5:
            # Estimate 1Y win probability from washout depth
            # (sigmoid: ~60% at wash=20, ~80% at wash=40, ~92% at wash=60)
            est_win_prob = 0.50 + 0.45 / (1 + math.exp(-(w - 30) / 12))
            # Opportunity score = quality × win_prob (both 0-100 scale, result 0-100)
            new_final.append(quality * est_win_prob)
        else:
            new_final.append(0.0)
    full_final = new_final + (existing_final if existing_final else [0.0] * len(existing_dates))

    # Update the detail
    existing_detail["series"] = {
        "dates": all_dates,
        "prices": [round(p, 4) if p is not None else None for p in all_prices],
        "wash": [round(w, 2) if w is not None else None for w in full_wash],
        "final": [round(f, 2) if f is not None else None for f in full_final],
    }

    print(f"  {ticker}: extended from {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days, +{n_new} new)")
    return True


def main():
    print("Loading full.json...")
    with open(FULL_JSON) as f:
        full = json.load(f)

    items = full.get("items", [])
    tickers = [x["ticker"] for x in items]
    print(f"Found {len(tickers)} tickers")

    # First fetch SPY history (needed for idiosyncratic calculations)
    print("\nFetching SPY extended history for beta calculations...")
    period1 = int(datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp())
    spy_rows = fetch_yahoo_history("SPY", period1, period2)
    spy_data_map = {}
    if spy_rows:
        for r in spy_rows:
            spy_data_map[r["date"]] = r["adj"]
        print(f"  SPY: {len(spy_rows)} days from {spy_rows[0]['date']} to {spy_rows[-1]['date']}")
    else:
        print("  WARNING: Could not fetch SPY data, idiosyncratic calcs will be approximate")

    # Process each ticker
    updated_tickers = set()

    # Process ticker files
    print(f"\nExtending ticker files in {TICKER_DIR}...")
    for fname in sorted(os.listdir(TICKER_DIR)):
        if not fname.endswith(".json"):
            continue
        tk = fname[:-5]
        fpath = os.path.join(TICKER_DIR, fname)
        with open(fpath) as f:
            detail = json.load(f)

        if extend_ticker(tk, detail, spy_data_map):
            with open(fpath, "w") as f:
                json.dump(detail, f, separators=(",", ":"))
            updated_tickers.add(tk)

        # Rate limit
        time.sleep(0.5)

    # Also extend embedded details in full.json
    print(f"\nExtending embedded details in full.json...")
    embedded = full.get("details", {})
    for tk, detail in embedded.items():
        if extend_ticker(tk, detail, spy_data_map):
            updated_tickers.add(tk)
        time.sleep(0.5)

    # Save full.json
    if updated_tickers:
        print(f"\nSaving full.json ({len(updated_tickers)} tickers updated)...")
        with open(FULL_JSON, "w") as f:
            json.dump(full, f, separators=(",", ":"))
        print("Done!")
    else:
        print("\nNo tickers needed updating.")


if __name__ == "__main__":
    main()
