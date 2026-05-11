"""Universe definitions for the deployment validation.

Tickers are intersected with the broader 1811-ticker universe at load time;
any name not present in that panel is silently dropped.

Sources:
  QQQ_TECH = Nasdaq-100 representative tech-heavy names (composition is
             dynamic; this is a robust modern snapshot).
  IYW      = iShares US Technology ETF top holdings (broad US tech).
  IGM      = iShares Expanded Tech sector (US, internet + software + h/w).
  TECH_BROAD = approximated tech-sector membership across mega/large/mid
               caps that have been in iShares tech ETFs in the last decade.
"""

# Nasdaq-100 (representative — modern)
QQQ_TECH = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "INTU",
    "CMCSA", "QCOM", "TXN", "AMGN", "HON", "AMAT", "BKNG", "LRCX", "ISRG",
    "REGN", "ADP", "MU", "MELI", "PANW", "KLAC", "SBUX", "MDLZ", "GILD",
    "ABNB", "VRTX", "MAR", "ADI", "SNPS", "PYPL", "ORLY", "CRWD", "FTNT",
    "ROP", "NXPI", "CHTR", "CDNS", "MNST", "WDAY", "MRVL", "KDP", "AEP",
    "ROST", "MCHP", "BIIB", "CTAS", "KHC", "ADSK", "EXC", "ODFL", "FAST",
    "CSGP", "IDXX", "DDOG", "LULU", "EA", "GEHC", "DXCM", "BKR", "CSX",
    "VRSK", "ANSS", "ZS", "TEAM", "CTSH", "TTD", "TTWO", "PCAR", "FANG",
    "MDB", "ON", "WBD", "ARM", "MRNA", "DLTR", "ILMN", "SIRI", "WBA",
    "BIDU", "JD", "OKTA", "DOCU", "ZM", "ATVI", "MTCH", "PTON", "PDD",
]

# iShares US Tech (IYW) representative composition
IYW = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "AMD", "ADBE", "CRM", "ACN",
    "NOW", "IBM", "INTU", "QCOM", "TXN", "CSCO", "INTC", "AMAT", "MU",
    "PANW", "ANET", "LRCX", "KLAC", "ADI", "FTNT", "CRWD", "SNPS", "CDNS",
    "MRVL", "NXPI", "MCHP", "GLW", "IT", "JKHY", "MSI", "ROP", "ZBRA",
    "EPAM", "GEN", "FICO", "WDAY", "TYL", "DELL", "HPQ", "HPE", "NTAP",
    "JNPR", "STX", "WDC", "FSLR", "ON", "SWKS", "QRVO", "TER", "MPWR",
    "ANSS", "TDY", "FFIV", "AKAM", "VRSN", "SMCI", "NTGR", "CIEN", "LITE",
    "VIAV", "OLED", "AEIS", "ACLS", "AMKR", "AVT", "BR", "CACC", "CCMP",
    "CDW", "CNXC", "COHR", "CYRX", "DAVA", "DOCN", "ENPH", "ENV", "ENV",
    "EPAY", "EQIX", "EVTC", "FATE", "FFIE", "FIS", "FISV", "FIVN", "FLT",
    "FN", "FORM", "FORT", "FTV", "GDDY", "GLOB", "GPN", "GPRO", "HEAR",
    "HUBS", "INFN", "INST", "JNT", "KEYS", "LFUS", "LITE", "LSCC", "MAR",
    "MCFE", "MTSI", "NLOK", "NOK", "NTNX", "NVT", "OLED", "PAYO", "PEGA",
    "PFGC", "PI", "PLAB", "PLOW", "PRGS", "PSTG", "RNG", "SAIC", "SANM",
    "SHOP", "SIMO", "SPSC", "SQ", "SSNC", "SSTI", "STM", "TDC", "TRMB",
    "TWLO", "U", "UCTT", "VEEV", "VERI", "VPG", "VRT", "VSH", "WIX",
    "WK", "XPER", "YEXT", "ZBRA", "ZUO", "BB", "PLTR", "SNOW", "DOCN",
    "DT", "MQ", "ROK",
]

# iShares Expanded Tech (IGM) — broader still, includes internet/services
IGM_EXTRA = [
    "ABNB", "UBER", "LYFT", "DASH", "RIVN", "LCID", "GRAB", "BABA",
    "JD", "PDD", "DIDI", "BIDU", "TME", "BILI", "RBLX", "U", "TWLO",
    "PINS", "SNAP", "ZG", "Z", "TRIP", "BKNG", "EXPE", "TSLA", "F",
    "GM", "RACE", "TOST", "AFRM", "PYPL", "FI", "MA", "V", "AXP",
]

# Tech-broad set: union of QQQ + IYW + IGM_EXTRA, deduped
def tech_broad():
    s = set(QQQ_TECH) | set(IYW) | set(IGM_EXTRA)
    return sorted(s)


TECH_BROAD = tech_broad()


# iShares Global Tech ETF (IXN) — modern composition snapshot.
# IXN holds ~140 names globally. Many of its non-US holdings (Samsung,
# SAP, Sony, Infineon local listings) do not have US-listed tickers in
# our price database; only US-listed names + ADRs are included below.
# Effective coverage: ~70-75% of IXN's market-cap weight via US + ADR.
IXN = [
    # US large-cap tech (heavy weight)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN", "ADBE", "AMD",
    "NOW", "INTU", "QCOM", "IBM", "TXN", "CSCO", "INTC", "AMAT", "MU",
    "PANW", "ANET", "LRCX", "KLAC", "ADI", "FTNT", "CRWD", "SNPS", "CDNS",
    "MRVL", "NXPI", "MCHP", "GLW", "MSI", "ROP", "DELL", "HPQ", "HPE",
    "NTAP", "JNPR", "STX", "WDC", "FSLR", "ON", "SWKS", "QRVO", "TER",
    "MPWR", "ANSS", "TDY", "FFIV", "AKAM", "VRSN", "SMCI", "WDAY",
    # International tech via US-listed ADRs
    "TSM",   # Taiwan Semiconductor
    "ASML",  # ASML Netherlands
    "ARM",   # ARM Holdings UK
    "SHOP",  # Shopify Canada
    "GFS",   # GlobalFoundries
    # Internet / services overlap
    "GOOGL", "GOOG", "META", "AMZN", "NFLX", "TSLA", "BIDU", "JD",
    "PDD", "BABA",
]


# Russell 1000 approximation — top US large- and mid-cap names.
# Russell 1000 has ~1000 names; not all are in our broader 1811 panel
# (some are mid-caps that came public after 2010 and aren't in v3's
# trained universe). This is a representative subset focused on the
# major sectors. Use with caution — this is NOT a PIT R1000 membership;
# it's a modern-snapshot approximation that may have survivorship bias.
RUSSELL_1000_CORE = [
    # S&P 500 mega/large caps (subset)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO",
    "TSLA", "JPM", "V", "WMT", "XOM", "JNJ", "MA", "UNH", "PG", "HD",
    "ORCL", "CVX", "MRK", "LLY", "ABBV", "KO", "PEP", "BAC", "COST",
    "ADBE", "TMO", "MCD", "CSCO", "CRM", "NFLX", "ABT", "ACN", "LIN",
    "DHR", "INTC", "TXN", "VZ", "WFC", "NKE", "DIS", "AMD", "PM", "QCOM",
    "T", "BMY", "CMCSA", "RTX", "INTU", "UPS", "LOW", "HON", "AMGN",
    "ISRG", "SPGI", "GS", "CAT", "PFE", "BLK", "IBM", "ELV", "BKNG",
    "AXP", "DE", "MS", "GILD", "C", "TJX", "PLD", "SCHW", "ADP", "MDLZ",
    "MO", "SYK", "VRTX", "REGN", "ZTS", "MMC", "CB", "ETN", "BSX", "CI",
    "FI", "AMAT", "PYPL", "PNC", "BX", "EQIX", "DUK", "AON", "CL", "PGR",
    "ICE", "SO", "USB", "TGT", "EOG", "BDX", "WM", "ITW", "MMM", "CSX",
    # Plus mid-caps from broader
    "ROK", "PSX", "PCG", "FCX", "EW", "MNST", "GD", "EMR", "FDX",
    "HCA", "CME", "APD", "MCO", "GE", "F", "NEM", "NSC", "TFC", "EXC",
    "SLB", "ATVI", "PSA", "AIG", "FIS", "MET", "ADI", "MAR", "AEP",
    "WBA", "DOW", "TRV", "GIS", "AFL", "PEG", "PRU", "DG", "ECL",
    "OXY", "WMB", "VLO", "SRE", "TWLO", "ROST", "MSCI", "STZ", "CCI",
    "DLR", "PCAR", "AZO", "ED", "PXD", "MCK", "FTNT", "KDP", "CTAS",
    "DLTR", "EBAY", "KMB", "MPC", "EL", "PAYX", "ORLY", "HUM", "WELL",
    "BIIB", "ATO", "MTB", "FAST", "VICI", "EFX", "STT", "WTW", "ANET",
    "CTSH", "ETSY", "WEC", "GLW", "PPG", "BR", "TROW", "ROL", "GWW",
    "DD", "NUE", "OKE", "LRCX", "KLAC", "MNDT", "PEG", "DTE", "EXR",
    "EIX", "MRO", "AVB", "BIO", "MTD", "DOV", "CBRE", "ESS", "WAB",
    "FE", "RMD", "SBUX", "ALL", "HSY", "DLTR", "FANG", "WTRG", "CHD",
    "CTAS", "IDXX", "PWR", "MLM", "ROST", "VMC", "OMC", "VTR", "HUBS",
    "ALGN", "BR", "OKE", "GPN", "STZ", "HBAN", "PFG", "ZBH", "WAT",
    # ARM, GFS, GEHC, NET, SNOW (newer issues)
    "ARM", "GFS", "GEHC", "NET", "SNOW", "DDOG", "DOCU", "ZM", "PLTR",
    "ABNB", "UBER", "LYFT", "DASH", "AFRM", "HOOD", "RIVN", "ZS", "TEAM",
    "WBD", "PARA",
]


def ixn_us():
    """Universe of IXN names that have US-listed equivalents in our data."""
    return sorted(set(IXN))


def russell1000_core():
    """Russell-1000-ish approximation, deduped."""
    return sorted(set(RUSSELL_1000_CORE))


IXN_US = ixn_us()
R1000 = russell1000_core()
