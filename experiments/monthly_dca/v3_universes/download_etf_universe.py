"""
Download a tradeable global ETF universe.

This is a survivorship-bias-FREE universe: every ETF in the basket has
existed continuously since its inception. No delisting risk to model.

Universe: ~50 ETFs covering:
  - US broad/factors: SPY, QQQ, DIA, IWM, IWB, IWV, MDY, VTI, RSP, USMV, MTUM, QUAL
  - US sectors: XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE, XLC
  - International: EFA, EEM, VWO, IEMG, ACWI, VXUS, VEA
  - Country: EWJ, EWG, EWU, MCHI, FXI, EWZ, INDA, EWA, EWC, EWY, EWT, EWH, EWS
  - Bonds: AGG, TLT, IEF, LQD, HYG, TIP
  - Commodity/Real: GLD, SLV, USO, GLDM, VNQ
  - Crypto-adj: GBTC (since 2013, may not be in scope)

Saves: cache/v3_universes/etf/prices.parquet

Run: python3 -m experiments.monthly_dca.v3_universes.download_etf_universe
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
DATA = ROOT / "experiments" / "monthly_dca" / "v3_universes" / "data"
OUT = CACHE / "v3_universes" / "etf"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


ETFS = [
    # US broad/factor
    "SPY", "QQQ", "DIA", "IWM", "IWB", "IWV", "MDY", "VTI", "RSP",
    "USMV", "MTUM", "QUAL", "VLUE", "SIZE",
    # US sectors (SPDR)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
    # US sub-sectors (popular)
    "SMH", "SOXX", "IBB", "XBI", "ITB", "KBE", "KRE", "OIH", "GDX",
    # International developed
    "EFA", "VEA", "IEFA", "VXUS", "ACWI", "ACWX", "VT",
    # Emerging
    "EEM", "VWO", "IEMG", "EMXC", "FM",
    # Country
    "EWJ", "EWG", "EWU", "MCHI", "FXI", "EWZ", "INDA", "EWA", "EWC", "EWY", "EWT", "EWH", "EWS",
    # Bonds
    "AGG", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB", "BND",
    # Commodity / Real assets
    "GLD", "SLV", "USO", "VNQ", "REET", "DBC",
]


def main():
    print(f"Universe: {len(ETFS)} ETFs")
    series_map: dict[str, pd.Series] = {}
    log_rows = []
    for i, tkr in enumerate(ETFS):
        if i % 10 == 0:
            print(f"  [{i}/{len(ETFS)}] {tkr}...")
        try:
            t = yf.Ticker(tkr)
            h = t.history(period="max", interval="1d", auto_adjust=True, actions=False)
            if h.empty or "Close" not in h.columns:
                log_rows.append({"ticker": tkr, "status": "empty", "n_obs": 0})
                continue
            s = h["Close"].dropna()
            if len(s) < 30:
                log_rows.append({"ticker": tkr, "status": "too_short", "n_obs": len(s)})
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
            s.name = tkr
            series_map[tkr] = s
            log_rows.append({"ticker": tkr, "status": "ok", "n_obs": len(s),
                             "first_date": str(s.index.min().date()),
                             "last_date": str(s.index.max().date())})
        except Exception as e:
            log_rows.append({"ticker": tkr, "status": "error", "n_obs": 0, "err": str(e)[:80]})
        if (i + 1) % 25 == 0:
            time.sleep(0.5)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(DATA / "etf_download_log.csv", index=False)
    print(log_df["status"].value_counts())

    if series_map:
        panel = pd.concat(series_map, axis=1).sort_index()
        panel.to_parquet(OUT / "prices.parquet")
        print(f"\nSaved {panel.shape[1]} ETF series, range {panel.index.min().date()} - {panel.index.max().date()}")


if __name__ == "__main__":
    main()
