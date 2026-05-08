"""Extend the price panel back to ~2000 via yfinance.

Two passes:
  1. For every ticker in the existing panel that doesn't go back to 2000,
     fetch yfinance history and splice in the older bars.
  2. Add a curated list of ~100 tickers that were major US-equities at some
     point but got delisted/acquired/renamed before today. Many will return
     no data via yfinance (Yahoo prunes deeply delisted symbols), but
     some - like LEH (Lehman), BSC (Bear Stearns), GMGMQ (old GM), TWX,
     CELG, ATVI, RTN, MON, EMC, SVB, FRC, BBBY, JCP - DO return data.

For each ticker, we fetch with auto_adjust=True so prices are dividend &
split adjusted in a self-consistent way. We splice older bars onto the
existing series after rescaling on the latest common date so the joint
series stays consistent.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PANEL_PATH = CACHE / "prices.parquet"
EXTENDED_PATH = CACHE / "prices_extended.parquet"


# Curated list of historically major US tickers that delisted, were acquired,
# or were renamed. We attempt all of them; yfinance only returns data for some.
# Sources: Wikipedia historical S&P 500 components, S&P 500 component changes,
# historical dot-com era / 2008 crisis / 2010s notable bankruptcies.
HISTORICAL_DELISTED = [
    # Mergers / acquisitions / spin-offs
    "TWX",     # Time Warner -> AT&T (now WBD)
    "CELG",    # Celgene -> BMY (2019)
    "MON",     # Monsanto -> Bayer (2018)
    "RTN",     # Raytheon -> RTX (2020)
    "EMC",     # EMC -> Dell (2016)
    "BRCM",    # Broadcom -> Avago (now AVGO) (2016)
    "ATVI",    # Activision Blizzard -> Microsoft (2023)
    "VMW",     # VMware -> Broadcom (2023)
    "ANTM",    # Anthem -> Elevance Health
    "WLP",     # WellPoint -> Anthem
    "FOXA",    # Old 21st Century Fox
    "TRIP",    # control: still trades
    "FB",      # Facebook -> Meta (META)
    "FBHS",
    "DOW",     # confusion: legacy DOW chemical -> DD
    "DD",      # legacy DuPont
    "DXC",
    "FOX",
    "SCG",     # SCANA -> Dominion
    "TIF",     # Tiffany -> LVMH
    "STI",     # SunTrust -> Truist (TFC)
    "BBT",     # BB&T -> Truist
    "RDC",
    "PCG.WS",
    "PCLN",    # Priceline -> Booking
    "STJ",     # St. Jude -> Abbott
    "WAG",     # Walgreens (legacy)
    "WMI",
    "CMCSK",   # Comcast K shares
    "GE.WS",
    "VIAB",    # Viacom -> ViacomCBS -> Paramount
    "WCG",
    "CXO",     # Concho Resources -> ConocoPhillips
    "RDS-A",
    "RDS-B",
    "SHL.AX",
    "MSI",     # control
    "ANR",     # Alpha Natural Resources
    "PCS",
    "AGN",     # Allergan -> AbbVie
    "ACT",     # legacy Actavis -> AGN
    "WFM",     # Whole Foods -> Amazon
    "RAI",     # Reynolds American -> BAT
    "DPS",     # Dr Pepper Snapple -> Keurig
    "ESV",     # Ensco -> Valaris
    "TROW",    # control: still trades
    "TWC",     # Time Warner Cable -> Charter
    "MJN",     # Mead Johnson -> RB
    "PVH",     # control: still trades
    "PCP",     # Precision Castparts -> Berkshire
    "BLL",
    "SE",      # control: still trades (Sea Limited)
    "TIE",
    # Dot-com era failures
    "SUNW",
    "LU",      # Lucent
    "Q",       # Qwest -> CenturyLink
    "TLAB",
    # 2008-2009 financial crisis
    "LEH",     # Lehman Brothers
    "BSC",     # Bear Stearns
    "WAMUQ",   # Washington Mutual
    "CFC",     # Countrywide
    "INDB",
    "MER",     # Merrill Lynch -> BAC
    "WB",      # Wachovia -> Wells Fargo
    "ABK",     # Ambac
    "MBI",     # MBIA (still trades?)
    "FNM",     # old Fannie symbol
    "FRE",     # old Freddie symbol
    "FNMA",    # Fannie Mae OTC
    "FMCC",    # Freddie Mac OTC
    # Retail / consumer failures
    "SHLD",    # Sears
    "SHLDQ",
    "JCP",     # JCPenney
    "JCPNQ",
    "TOYS",    # Toys R Us
    "RAD",     # Rite Aid
    "BBBY",    # Bed Bath & Beyond
    "BBBYQ",
    "HEXO",
    "REVG",
    "CIR",
    "BOND",
    "M.WS",
    "GPS",     # control: still trades
    "ANN",     # AnnTaylor / Ascena
    "ASNA",
    # Banks 2023
    "SVB",     # Silicon Valley Bank
    "SIVB",
    "SIVBQ",
    "FRC",     # First Republic
    "FRCB",
    "SBNY",
    "PACW",
    # Other notable
    "GME.WS",
    "AMC.WS",
    "MEDP",
    "NWS-A",
    "VIAC",
    "VIA",
    "BBT",
    "CLNS",
    "MNK",     # Mallinckrodt
    "MNKD",
    "GTAT",    # GT Advanced Tech
    "OFC",
    "CHK",     # Chesapeake (re-IPO'd)
    "REVH",
    "MAVS",
    # Cannabis
    "TLRY",    # control: still trades
    "ACB",     # control: still trades
    # Telecom / media
    "WCOM",    # WorldCom
    "Q",
    "CCI.WS",
    "TWTR",    # Twitter -> private (Musk)
    # Dot-com darlings
    "INTC",    # control: still trades
    "ENRN",    # Enron
    "GTW",     # Gateway computer
    "CALM",    # control
    "CMGI",
    "PALM",    # Palm -> HP
    # Energy
    "XOM",     # control
    "OAS",     # Oasis Petroleum
    "WLL",     # Whiting Petroleum
    "HK",      # Halcón
    "HCRSE",
    # Consumer finance
    "GTAT",
    "HOV",     # Hovnanian
    "LL",      # Lumber Liquidators -> LL Flooring -> bankrupt
    # Healthcare
    "ALK",     # Alkermes; control
    "TVPT",
    "ESI",     # ITT Education
    "CRC",     # California Resources
    "CHKR",
    # Notable acquisitions 2010s/2020s
    "MEDP",
    "TIF",     # Tiffany -> LVMH
    "ROST",    # control
    "SHO",     # control
    "PRGO",    # control
    "LPL",     # control
    "COL",     # Rockwell Collins -> UTX
    "UTX",     # United Technologies -> RTX
    "LLL",     # L3 -> L3Harris (LHX)
    "DPS",     # Dr Pepper Snapple
]


def _close_series(t: str, start: str = "1995-01-01") -> pd.Series | None:
    try:
        d = yf.download(t, start=start, progress=False, threads=False, auto_adjust=True)
        if d is None or d.empty:
            return None
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype("float64")
        if len(close) < 100:
            return None
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.name = t
        return close
    except Exception as e:
        return None


def extend_existing(existing: pd.DataFrame, target_first: pd.Timestamp = pd.Timestamp("2000-01-01")) -> pd.DataFrame:
    """For every ticker in the existing panel whose first valid date is later
    than target_first, fetch yfinance history and splice in the older bars.
    Splicing rescales the older series so its value at the local series'
    earliest date matches the local series, keeping adjustments consistent.

    IMPORTANT: builds a fresh panel by collecting the per-ticker series and
    concat-ing along columns. Assigning to existing[col] would silently drop
    new rows whose index is outside existing's index.
    """
    n_extended = 0
    n_skipped = 0
    n_failed = 0
    cols = list(existing.columns)
    series_map: dict[str, pd.Series] = {}
    for i, tkr in enumerate(cols):
        local = existing[tkr].dropna()
        if local.empty:
            series_map[tkr] = existing[tkr]
            continue
        if local.index.min() <= target_first:
            series_map[tkr] = existing[tkr]
            n_skipped += 1
            continue
        yf_close = _close_series(tkr, start=target_first.strftime("%Y-%m-%d"))
        if yf_close is None or yf_close.empty:
            series_map[tkr] = existing[tkr]
            n_failed += 1
            continue
        local_first = local.index.min()
        older = yf_close.loc[yf_close.index < local_first]
        if older.empty:
            series_map[tkr] = existing[tkr]
            n_skipped += 1
            continue
        if older.iloc[-1] == 0 or not np.isfinite(older.iloc[-1]):
            series_map[tkr] = existing[tkr]
            n_failed += 1
            continue
        scale = local.iloc[0] / older.iloc[-1]
        older_scaled = older * scale
        full = pd.concat([older_scaled, local])
        full = full[~full.index.duplicated(keep="last")].sort_index()
        full.name = tkr
        series_map[tkr] = full
        n_extended += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cols)}] extended={n_extended}  skipped={n_skipped}  failed={n_failed}")
    print(f"  TOTAL: extended={n_extended}  skipped={n_skipped}  failed={n_failed}")

    # Build the new DataFrame from the union of all series indices
    out = pd.concat(series_map.values(), axis=1).sort_index()
    out.index.name = existing.index.name
    return out


def add_delisted(panel: pd.DataFrame, target_first: pd.Timestamp = pd.Timestamp("2000-01-01")) -> pd.DataFrame:
    """Add delisted tickers to the panel as new columns. Only adds those for
    which yfinance returns at least 100 bars within [target_first, now]."""
    new_cols = {}
    n_added = 0
    n_failed = 0
    seen = set(panel.columns)
    for tkr in HISTORICAL_DELISTED:
        if tkr in seen:
            continue
        s = _close_series(tkr, start=target_first.strftime("%Y-%m-%d"))
        if s is None:
            n_failed += 1
            continue
        new_cols[tkr] = s
        n_added += 1
    print(f"  delisted: added={n_added}  failed={n_failed}")
    if not new_cols:
        return panel
    add_df = pd.concat(new_cols.values(), axis=1, keys=new_cols.keys())
    out = panel.join(add_df, how="outer").sort_index()
    return out


def main() -> None:
    if not PANEL_PATH.exists():
        from experiments.monthly_dca.load_data import main as build
        build()
    panel = pd.read_parquet(PANEL_PATH)
    print(f"Existing panel: {panel.shape}  date range {panel.index.min().date()} → {panel.index.max().date()}")
    target = pd.Timestamp("2000-01-01")

    print("\n=== Pass 1: extending existing tickers back to 2000 ===")
    panel = extend_existing(panel, target_first=target)
    print(f"After pass 1: {panel.shape}  date range {panel.index.min().date()} → {panel.index.max().date()}")

    print("\n=== Pass 2: adding historically delisted tickers ===")
    panel = add_delisted(panel, target_first=target)
    print(f"After pass 2: {panel.shape}  date range {panel.index.min().date()} → {panel.index.max().date()}")

    panel.to_parquet(EXTENDED_PATH, compression="zstd")
    print(f"\nWrote {EXTENDED_PATH} ({panel.shape})  size {EXTENDED_PATH.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
