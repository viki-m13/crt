"""Global live market-data layer. No preloaded universe, no vendor lock-in.

Every number the site displays comes from here — never from a language model.
An LLM proposes candidate tickers; this module independently verifies each one
exists and computes its risk/return profile from real price history. A ticker
the LLM invents simply 404s and is dropped, which is the hallucination guard.

Sources, in order of authority:

  1. Yahoo chart endpoint (free, keyless, GLOBAL). Verified live across NYSE,
     Nasdaq, Tokyo, XETRA, Amsterdam, Paris, Swiss, HKSE, NSE, KSE, ASX,
     Toronto and Sao Paulo. Gives name, currency, exchange, live price and
     daily history, from which volatility, drawdown, momentum and 52-week
     position are computed here rather than trusted from anyone.
  2. Alpha Vantage COMPANY_OVERVIEW (optional, needs ALPHAVANTAGE_API_KEY).
     Adds sector/PE/margins/beta — but it is US-only (a Tokyo symbol returns
     {}), so it is strictly an enrichment and never a requirement.

Standard library only: this runs in a serverless function where cold-start
time is user-visible latency.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

HDR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")}
CHART = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
AV = "https://www.alphavantage.co/query"

# Warm-lambda caches. Prices move, so quotes expire fast; identity does not.
_cache: dict = {}
_TTL = 300.0

# Instruments a "pick me a stock" product must never return.
_BANNED = ("2X", "3X", "-1X", "ULTRA", "INVERSE", "LEVERAGED", "BULL 3", "BEAR 3")


class NotFound(Exception):
    """The symbol does not exist upstream — authoritative, do not retry."""


class RateLimited(Exception):
    """Upstream throttled us. Distinct from 'no data' so callers never
    mistake a 429 for a verdict about the stock."""


def _get_json(url: str, timeout: float = 12.0, tries: int = 3):
    """Fetch with backoff. Yahoo throttles datacenter IPs aggressively
    (measured: 429 after a burst), so a bare request is not enough — but a
    429 must never be reported as 'this stock does not exist'."""
    delay = 0.6
    last = None
    for i in range(tries):
        req = urllib.request.Request(url, headers=HDR)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(url) from e
            if e.code in (429, 503, 502):
                last = RateLimited(f"{e.code}")
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if (ra or "").isdigit() else delay
                if i < tries - 1:
                    time.sleep(min(wait, 3.0))
                    delay *= 2
                continue
            last = e
            break
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(delay)
                delay *= 2
    if isinstance(last, RateLimited):
        raise last
    raise last or RuntimeError("fetch failed")


# Development-only disk cache. Set MARKET_DISK_CACHE to a directory to make
# test runs reproducible and survive upstream throttling; never set in prod,
# where staleness would be a correctness bug rather than a convenience.
_DISK = os.environ.get("MARKET_DISK_CACHE")


def _disk_path(key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return os.path.join(_DISK, safe + ".json")


def _cached(key: str):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < _TTL:
        return hit[1]
    if _DISK:
        try:
            with open(_disk_path(key)) as f:
                val = json.load(f)
            _cache[key] = (time.time(), val)
            return val
        except Exception:
            return None
    return None


def _put(key: str, val):
    _cache[key] = (time.time(), val)
    if _DISK and isinstance(val, (dict, list)):
        try:
            os.makedirs(_DISK, exist_ok=True)
            with open(_disk_path(key), "w") as f:
                json.dump(val, f)
        except Exception:
            pass
    return val


# --------------------------------------------------------------------------
# price history -> risk metrics
# --------------------------------------------------------------------------
def _stdev(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _metrics_from_closes(closes: list[float]) -> dict:
    """Realised risk/return profile. Daily log returns; 252-day annualisation."""
    out = {}
    if len(closes) < 30:
        return out
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    if len(rets) < 20:
        return out
    out["ann_vol"] = _stdev(rets) * math.sqrt(252)
    # deepest peak-to-trough over the window: what the holder actually endured
    peak, mdd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        if peak > 0:
            mdd = min(mdd, c / peak - 1.0)
    out["max_dd"] = mdd
    last = closes[-1]
    for lab, back in (("mom_1m", 21), ("mom_3m", 63), ("mom_6m", 126), ("mom_12m", 252)):
        if len(closes) > back and closes[-1 - back] > 0:
            out[lab] = last / closes[-1 - back] - 1.0
    hi, lo = max(closes), min(closes)
    out["pos_52w"] = (last - lo) / (hi - lo) if hi > lo else 0.5
    out["dd_from_high"] = last / hi - 1.0 if hi > 0 else 0.0
    # annualised realised return over the window, for a Sharpe-like read
    yrs = len(closes) / 252.0
    if yrs > 0.25 and closes[0] > 0 and last > 0:
        out["cagr_window"] = (last / closes[0]) ** (1 / yrs) - 1
        if out["ann_vol"] > 1e-6:
            out["sharpe_window"] = (out["cagr_window"] - 0.04) / out["ann_vol"]
    return out


def quote(symbol: str, rng: str = "2y") -> dict:
    """Verify a symbol exists and profile it from real history.

    Returns {"ok": False, "reason": ...} for anything invented, delisted,
    illiquid, or structurally unsuitable (leveraged/inverse products).
    """
    symbol = (symbol or "").strip().upper()
    if not symbol or len(symbol) > 16:
        return {"ok": False, "symbol": symbol, "reason": "malformed"}
    ck = f"q:{symbol}:{rng}"
    hit = _cached(ck)
    if hit is not None:
        return hit

    data, throttled = None, False
    for host in ("query1", "query2"):
        url = CHART.format(host=host, sym=urllib.parse.quote(symbol)) + \
            f"?range={rng}&interval=1d"
        try:
            data = _get_json(url)
            break
        except NotFound:
            return _put(ck, {"ok": False, "symbol": symbol, "reason": "not_found"})
        except RateLimited:
            throttled = True
            continue
        except Exception:
            continue
    if not data:
        # Never cached: a throttle is a statement about us, not about the
        # stock, and must not harden into a permanent rejection.
        return {"ok": False, "symbol": symbol,
                "reason": "rate_limited" if throttled else "unreachable",
                "retryable": True}

    res = (data.get("chart") or {}).get("result")
    if not res:
        return _put(ck, {"ok": False, "symbol": symbol, "reason": "not_found"})
    r0 = res[0]
    meta = r0.get("meta") or {}
    try:
        closes = [c for c in r0["indicators"]["quote"][0]["close"] if c is not None]
    except Exception:
        closes = []
    if len(closes) < 60:
        return _put(ck, {"ok": False, "symbol": symbol, "reason": "insufficient_history"})

    name = meta.get("longName") or meta.get("shortName") or symbol
    if any(b in name.upper() for b in _BANNED):
        return _put(ck, {"ok": False, "symbol": symbol, "reason": "leveraged_product"})

    out = {
        "ok": True,
        "symbol": symbol,
        "name": name,
        "price": meta.get("regularMarketPrice"),
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName"),
        "type": meta.get("instrumentType"),
        "high_52w": meta.get("fiftyTwoWeekHigh"),
        "low_52w": meta.get("fiftyTwoWeekLow"),
        "bars": len(closes),
    }
    out.update(_metrics_from_closes(closes))
    # a coarse liquidity/size read that works on every exchange
    try:
        vols = [v for v in r0["indicators"]["quote"][0]["volume"] if v]
        if vols and out.get("price"):
            recent = vols[-21:] if len(vols) >= 21 else vols
            out["adv_usd_local"] = (sum(recent) / len(recent)) * float(out["price"])
    except Exception:
        pass
    return _put(ck, out)


def quotes(symbols: list[str], rng: str = "2y", workers: int = 8) -> dict:
    """Verify many candidates concurrently — latency is user-visible here."""
    syms, seen = [], set()
    for s in symbols:
        s = (s or "").strip().upper()
        if s and s not in seen:
            seen.add(s)
            syms.append(s)
    if not syms:
        return {}
    out = {}
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(syms))) as ex:
        futs = {ex.submit(quote, s, rng): s for s in syms}
        for f in cf.as_completed(futs):
            s = futs[f]
            try:
                out[s] = f.result()
            except Exception as e:  # noqa: BLE001
                out[s] = {"ok": False, "symbol": s, "reason": f"error:{type(e).__name__}"}
    return out


def search(query: str, limit: int = 6) -> list[dict]:
    """Resolve a company name to listed symbols, worldwide."""
    q = (query or "").strip()
    if not q:
        return []
    ck = f"s:{q.lower()}:{limit}"
    hit = _cached(ck)
    if hit is not None:
        return hit
    url = (SEARCH + "?" + urllib.parse.urlencode(
        {"q": q, "quotesCount": limit, "newsCount": 0}))
    try:
        d = _get_json(url)
    except Exception:
        return []
    rows = []
    for it in d.get("quotes", []):
        if it.get("quoteType") != "EQUITY" or not it.get("symbol"):
            continue
        rows.append({"symbol": it["symbol"],
                     "name": it.get("shortname") or it.get("longname") or it["symbol"],
                     "exchange": it.get("exchange"),
                     "country": it.get("region")})
    return _put(ck, rows[:limit])


def enrich(symbol: str) -> dict:
    """Optional US-only fundamentals. Absent key or non-US symbol -> {}."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return {}
    ck = f"e:{symbol}"
    hit = _cached(ck)
    if hit is not None:
        return hit
    url = AV + "?" + urllib.parse.urlencode(
        {"function": "OVERVIEW", "symbol": symbol, "apikey": key})
    try:
        d = _get_json(url, timeout=10.0)
    except Exception:
        return {}
    if not isinstance(d, dict) or not d.get("Symbol"):
        return _put(ck, {})

    def num(k):
        try:
            v = d.get(k)
            return float(v) if v not in (None, "", "None", "-") else None
        except Exception:
            return None

    return _put(ck, {k: v for k, v in {
        "sector": (d.get("Sector") or "").title() or None,
        "industry": (d.get("Industry") or "").title() or None,
        "description": d.get("Description"),
        "market_cap": num("MarketCapitalization"),
        "pe": num("PERatio"),
        "peg": num("PEGRatio"),
        "dividend_yield": num("DividendYield"),
        "beta": num("Beta"),
        "profit_margin": num("ProfitMargin"),
        "revenue_growth": num("QuarterlyRevenueGrowthYOY"),
        "analyst_target": num("AnalystTargetPrice"),
    }.items() if v is not None})
