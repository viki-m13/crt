"""ETF universes for the v3-on-ETFs experiment (branch O0MtP).

Two universes plus a combined one:

  * BROAD: ~210 plain (un-leveraged) ETFs covering US equity (broad / sector /
    factor / size / style), international, fixed income, commodities, real
    estate, currencies. Curated to be a representative cross-section of what a
    real investor could buy, while keeping the list at a size where the v3
    ML cross-sectional ranker can learn meaningful relative orderings each
    month-end.

  * LEVERED: ~50 leveraged & inverse ETFs (mostly 2x and 3x). Many launched
    2006-2009, so usable history is shorter.

The lists are intentionally over-inclusive of yfinance-resolvable tickers; we
filter to those that actually return data and have enough history at fetch
time.

SPY is always included for regime detection / benchmarking; SPY itself is in
EXCLUDE so the strategy never picks it.
"""
from __future__ import annotations


# Always required for regime detection and benchmark
ALWAYS_INCLUDE = ["SPY"]


# ---------------------------------------------------------------------------
# BROAD (un-leveraged) ETFs
# ---------------------------------------------------------------------------
BROAD_ETFS = sorted(set([
    # US broad / total market
    "SPY", "VOO", "IVV", "VTI", "ITOT", "SCHB", "SPLG", "RSP",
    "QQQ", "QQQM", "DIA", "OEF",
    "IWM", "IWB", "IWV", "VB", "VTWO", "IJR", "VBR", "IJS", "VIOO",
    "MDY", "IJH", "VO", "IWR", "VOE", "VOT",
    # Style / factor
    "VUG", "VTV", "IVW", "IVE", "IWF", "IWD", "MTUM", "QUAL", "USMV", "VLUE", "SIZE",
    "DGRO", "VIG", "VYM", "SCHD", "DVY", "SDY", "NOBL", "HDV", "SPHQ", "SPLV", "SPHD",
    "MOAT", "FNDX", "FNDA", "FNDB", "FNDF", "FNDE",
    # US sector
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "VGT", "VFH", "VDE", "VHT", "VCR", "VDC", "VIS", "VPU", "VAW", "VOX",
    "SMH", "SOXX", "IBB", "XBI", "ITA", "KRE", "KIE", "KBE",
    "IYR", "VNQ", "REZ", "RWR",
    "XOP", "OIH", "URA", "LIT",
    "GDX", "GDXJ", "SIL",
    "TAN", "ICLN", "PBW", "FAN", "QCLN",
    "PEJ", "JETS", "AWAY", "PHO", "FIW",
    "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",
    "FDN", "PNQI", "HACK", "BUG", "SKYY", "ROBO", "BOTZ", "AIQ", "FINX", "IPAY",
    "IGV", "IGM", "VOX", "SOCL", "ESPO", "HERO", "BETZ",
    "IEDI", "ITB", "XHB", "XRT", "XTN",
    # International — developed
    "EFA", "VEA", "IEFA", "SCHF", "EFG", "EFV", "IEMG", "VWO",
    "VEU", "VXUS", "IXUS", "ACWI", "ACWX", "ACWV", "VT",
    "EWJ", "EWG", "EWU", "EWQ", "EWP", "EWI", "EWN", "EWD", "EWL", "EWA",
    "EWC", "EWZ", "EWW", "EWY", "EWT", "EWH", "EWS", "EZA",
    "ASHR", "MCHI", "FXI", "KWEB", "INDA", "EPI", "EIDO", "THD", "EPHE",
    "EPOL", "TUR", "RSX", "VNM",
    "ILF", "AAXJ", "EEMA", "EEMV", "ESGU",
    # Fixed income — duration
    "TLT", "TLH", "IEF", "IEI", "SHY", "BIL", "VGSH", "VGIT", "VGLT", "GOVT",
    "AGG", "BND", "BIV", "BSV", "VTC", "VCSH", "VCIT", "VCLT",
    "LQD", "HYG", "JNK", "USHY", "SHYG", "SJNK", "ANGL",
    "EMB", "VWOB", "EMLC", "PCY",
    "MUB", "VTEB", "TFI", "HYD",
    "TIP", "VTIP", "STIP", "SCHP",
    "MBB", "VMBS", "MBSD",
    "BNDX", "IAGG", "BWX", "IGOV", "ISHG",
    # Commodities / real assets
    "GLD", "IAU", "SGOL", "SLV", "SIVR",
    "DBC", "GSG", "DJP", "USO", "BNO", "UNG", "DBA", "WEAT", "CORN", "SOYB",
    "PPLT", "PALL", "DBB", "JJC", "CPER", "WOOD",
    # REIT focused
    "REM", "MORT", "SRET", "ICF", "USRT", "IYR",
    # Currency / volatility
    "UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "FXF", "VXX", "VIXY",
    # Dividend / income
    "PFF", "PFFD", "VRP", "PGX", "PCEF",
    # Niche themes
    "GUNR", "BLOK", "BLCN", "IBLC",
    # Convertibles / preferreds
    "CWB", "ICVT",
    # Smart-beta value/momentum/quality
    "VLUE", "PRF", "MGV", "MGK", "RFG", "RFV", "RPV", "RPG",
    # Big-cap tilts
    "OEF", "MGC", "MGV", "MGK",
    # Healthcare / biotech depth
    "IHI", "IHF", "IXJ", "PJP", "BBH",
    # Materials / industrials depth
    "PICK", "REMX", "COPX", "IGE", "IYE", "IYM", "IYJ",
    # Tech / semis
    "FTEC", "FXL", "QTEC", "PSI",
]))


# ---------------------------------------------------------------------------
# LEVERAGED & INVERSE ETFs (2x, 3x, -1x, -2x, -3x)
# ---------------------------------------------------------------------------
LEVERAGED_ETFS = sorted(set([
    # 3x long / inverse — broad index
    "TQQQ", "SQQQ",          # NASDAQ 100 +/- 3x
    "UPRO", "SPXU",          # SPX +/- 3x
    "SPXL", "SPXS",          # SPX +/- 3x (Direxion)
    "TNA",  "TZA",           # Russell 2000 +/- 3x
    "UDOW", "SDOW",          # DJIA +/- 3x
    "UMDD", "SMDD",          # Mid-cap +/- 3x
    # 2x long / inverse — broad index
    "QLD",  "QID",           # NDX +/- 2x
    "SSO",  "SDS",            # SPX +/- 2x
    "DDM",  "DXD",            # DJIA +/- 2x
    "UWM",  "TWM",            # R2K +/- 2x
    "ROM",  "REW",            # Tech +/- 2x
    "MVV",  "MZZ",            # Mid-cap +/- 2x
    "URE",  "SRS",            # Real estate +/- 2x
    "UYG",  "SKF",            # Financials +/- 2x
    # 3x sector
    "FAS",  "FAZ",            # Financials +/- 3x
    "SOXL", "SOXS",           # Semis +/- 3x
    "TECL", "TECS",           # Tech +/- 3x
    "CURE", "RXD",            # Healthcare +/- 3x / -2x
    "DPST",                   # Regional banks 3x
    "DRN",  "DRV",            # Real estate +/- 3x
    "ERX",  "ERY",            # Energy +/- 3x (Direxion)
    "GUSH", "DRIP",           # Oil & gas E&P +/- 2x/-2x
    "NUGT", "DUST",           # Gold miners +/- 2x
    "JNUG", "JDST",           # Junior gold miners +/- 2x
    "LABU", "LABD",            # Biotech +/- 3x
    "RETL", "WEBL", "WEBS",   # Retail/Internet 3x
    "YINN", "YANG",            # FTSE China +/- 3x
    "EDC",  "EDZ",             # EM +/- 3x
    "DZK",  "DPK",             # Developed mkts +/- 3x
    "EURL",                    # Euro 3x
    # 2x commodities
    "AGQ",  "ZSL",             # Silver +/- 2x
    "UGL",  "GLL",             # Gold +/- 2x
    "BOIL", "KOLD",            # Natgas +/- 2x
    "UCO",  "SCO",              # Crude +/- 2x
    "DIG",  "DUG",              # Oil & Gas +/- 2x
    # Treasury leverage
    "TMF",  "TMV",              # 20+yr UST +/- 3x
    "TYD",  "TYO",              # 7-10yr UST +/- 3x
    "UBT",  "TBT",              # UST 2x / -2x (TBT = -2x 20yr)
    "TBF",                       # -1x 20+yr UST
    # Currency leverage
    "EUO",  "ULE",              # Euro +/- 2x
    "YCL",  "YCS",              # Yen +/- 2x
    # VIX leverage (path-decay nightmare, but real)
    "UVXY",                      # VIX 1.5x
    "SVXY",                      # VIX -0.5x
]))


def combined_universe() -> list[str]:
    return sorted(set(BROAD_ETFS) | set(LEVERAGED_ETFS) | set(ALWAYS_INCLUDE))


if __name__ == "__main__":
    print(f"BROAD: {len(BROAD_ETFS)} tickers")
    print(f"LEVERAGED: {len(LEVERAGED_ETFS)} tickers")
    print(f"COMBINED: {len(combined_universe())} tickers")
