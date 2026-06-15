"""Phase F1: build a POINT-IN-TIME fundamentals fact table from SEC EDGAR
for the PIT S&P 500 universe, for an orthogonal (non-price) quality sleeve.

PIT discipline (the #1 fake-alpha trap per SECOND_SLEEVE_SCOPE.md): every
EDGAR XBRL fact carries a `filed` date. A fact is usable only on/after its
`filed` date -- NEVER its period `end`. We keep (ticker, filed, end, tag,
val) and lag strictly to `filed` downstream.

Source: data.sec.gov companyfacts (free, authoritative, no key). XBRL begins
~2009, so this sleeve covers ~2010+, which is exactly E2's weak window
(post-2010); 2003-2009 is left to the price sleeves that already dominate it.

Coverage caveat: SEC's current company_tickers.json maps ~73% of the
post-2010 PIT universe (delisted/renamed names absent). The sleeve is a
cross-sectional rank among available names each month; missing names reduce
breadth. Documented, not hidden.

Output: augmented/fundamentals_pit_facts.parquet
Cache:  <gitignored> cache/edgar_companyfacts/*.json
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
CACHE_DIR = ROOT / "experiments" / "monthly_dca" / "cache" / "edgar_companyfacts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "crt-research viktormashalov@gmail.com"}
# us-gaap tags for gross profitability + ROA + accruals (robust quality set)
TAGS = {
    "Assets", "GrossProfit", "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    "NetIncomeLoss", "OperatingIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
}


def _get(url, timeout=40):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout))


def ticker_cik_map():
    ct = _get("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in ct.values()}


def universe():
    mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    post = mem[mem["asof"] >= "2009-06-01"]["ticker"].unique()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    return sorted(set(post) & set(mr.columns))   # must be simulable


def fetch_facts(cik):
    fp = CACHE_DIR / f"{cik}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        d = _get(url)
    except Exception as e:
        fp.write_text(json.dumps({"_error": str(e)[:200]}))
        return {"_error": str(e)[:200]}
    fp.write_text(json.dumps(d))
    time.sleep(0.15)            # SEC fair-access (<10 req/s)
    return d


def extract(ticker, facts):
    rows = []
    g = facts.get("facts", {}).get("us-gaap", {})
    for tag in TAGS:
        if tag not in g:
            continue
        for unit, recs in g[tag].get("units", {}).items():
            if unit != "USD":
                continue
            for r in recs:
                if r.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    continue
                if not r.get("filed") or not r.get("end"):
                    continue
                rows.append({"ticker": ticker, "tag": tag,
                             "filed": r["filed"], "end": r["end"],
                             "start": r.get("start"), "val": r["val"],
                             "form": r["form"], "fp": r.get("fp")})
    return rows


def main():
    t2c = ticker_cik_map()
    univ = universe()
    hit = [(t, t2c[t]) for t in univ if t in t2c]
    print(f"universe simulable & post-2009: {len(univ)} | CIK hits: {len(hit)}",
          flush=True)
    all_rows = []
    for i, (t, cik) in enumerate(hit):
        f = fetch_facts(cik)
        if "_error" not in f:
            all_rows.extend(extract(t, f))
        if i % 50 == 0 or i == len(hit) - 1:
            print(f"  {i+1}/{len(hit)} {t} rows={len(all_rows)}", flush=True)
    df = pd.DataFrame(all_rows)
    df["filed"] = pd.to_datetime(df["filed"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["ticker", "tag", "filed", "end"])
    out = AUG / "fundamentals_pit_facts.parquet"
    df.to_parquet(out)
    print(f"\nsaved {len(df)} facts ({df.ticker.nunique()} tickers, "
          f"{df.filed.min().date()}..{df.filed.max().date()}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
