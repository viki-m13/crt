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
