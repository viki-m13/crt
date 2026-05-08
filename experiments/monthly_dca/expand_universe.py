"""Expand the panel with small caps, small-cap tech, and international tickers.

Curated list pulled from:
  - Russell 2000 well-known small caps
  - Small/mid-cap tech (semiconductors, software, niche hardware)
  - International ADRs (large foreign companies trading on US exchanges)
  - Sector and country ETFs

Fetches via yfinance and merges into prices_extended.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "experiments" / "monthly_dca" / "cache" / "prices_extended.parquet"


# Tickers known to be major small/mid-cap, small-cap tech, and international.
# Most exist in yfinance with multi-decade history.
EXPANSION = [
    # ===== Small-cap tech / semiconductors =====
    "AEHR", "AEIS", "AKAM", "ALGM", "ALRM", "AMBA", "AMD", "ANET", "ANSS", "ARW",
    "ASML", "AVAV", "AYX", "BAND", "BILL", "BL", "BLKB", "BOX", "BRKS", "BRO",
    "CALX", "CCJ", "CCK", "CDNS", "CEVA", "CGNX", "CHKP", "CIEN", "CLDR", "CNCE",
    "COHR", "COMM", "CORT", "COUP", "CPSI", "CRDO", "CRSR", "CRWD", "CUI", "CVLT",
    "CXM", "CYBR", "DDOG", "DOCN", "DOCU", "DOLE", "DSGX", "DT", "DUOL", "EBET",
    "EGHT", "ELF", "ENPH", "EPAM", "ESTC", "EVBG", "EVOP", "EXTR", "FFIV", "FIVN",
    "FLEX", "FLNC", "FNKO", "FORM", "FORTY", "FOUR", "FRPT", "FRSH", "FTNT", "GDS",
    "GDYN", "GEN", "GIB", "GLBE", "GLOB", "GNRC", "GOOS", "GTLB", "HALO", "HASI",
    "HCKT", "HEAR", "HIMS", "HLIT", "HPK", "HUBS", "IART", "IDCC", "IDXX", "IIIV",
    "IIVI", "IMMR", "INFN", "INMD", "INST", "INTA", "IOSP", "IPGP", "IRDM", "JAMF",
    "JKHY", "JNPR", "KARO", "KEYS", "KLIC", "KNTK", "LASR", "LITE", "LFUS", "LSCC",
    "LSPD", "LYFT", "MANH", "MARA", "MASI", "MAXR", "MCRS", "MDB", "MGNI", "MITK",
    "MKSI", "MLAB", "MNDY", "MOMO", "MOS", "MPWR", "MQ", "MRVL", "MSI", "MSTR",
    "MTCH", "MTLS", "MTSI", "MUSA", "MXL", "NABL", "NATI", "NCNO", "NCR", "NEM",
    "NET", "NEWR", "NICE", "NICK", "NLOK", "NOC", "NOK", "NOVT", "NOW", "NPO",
    "NSIT", "NTAP", "NTCT", "NTRA", "NU", "NUTX", "NVCR", "NVMI", "OII", "OKTA",
    "OLED", "OLLI", "OMCL", "ONDS", "ONTO", "OPRX", "OSIS", "PANW", "PATH", "PCOR",
    "PCTY", "PDCO", "PD", "PEGA", "PENN", "PI", "PINS", "PLAB", "PLTR", "PLUG",
    "POWI", "PRGS", "PRO", "PSTG", "PTC", "PUBM", "PYCR", "QLYS", "QRVO", "RAMP",
    "REKR", "RDFN", "RILY", "RIVN", "RMBS", "RNG", "ROKU", "RPD", "RUM", "RVNC",
    "S", "SAGE", "SANM", "SAP", "SATS", "SAY", "SCAB", "SDC", "SE", "SEDG",
    "SEMR", "SHOP", "SIMO", "SLAB", "SMAR", "SMTC", "SNPS", "SNX", "SONO", "SOXX",
    "SPLK", "SPSC", "SQ", "SSTI", "SUMO", "SVMK", "SWIR", "SYNA", "SYM", "TASK",
    "TDC", "TEAM", "TECD", "TENB", "TER", "THM", "TIXT", "TLS", "TLYS", "TPC",
    "TRMB", "TROO", "TRUE", "TS", "TSEM", "TWLO", "TWST", "TYL", "U", "UCTT",
    "UEC", "UI", "UPST", "UPWK", "VBNK", "VCEL", "VECO", "VEEV", "VERX", "VICR",
    "VPG", "VPRT", "VRT", "VVPR", "WB", "WDAY", "WIT", "WIX", "WK", "WMG",
    "WOLF", "XPER", "YEXT", "ZBRA", "ZD", "ZEN", "ZG", "ZI", "ZIM", "ZS",
    "ZWS",
    # ===== Russell 2000 / small-cap broad =====
    "AAOI", "ABG", "ABMD", "ACA", "ACAD", "ACGL", "ACIW", "ACLS", "ADM", "AEL",
    "AGYS", "AIN", "AIT", "ALEX", "ALKS", "ALSN", "AMC", "AMED", "AMK", "AMN",
    "AMR", "ANDE", "AOSL", "APEI", "APLE", "APOG", "ARCH", "ARLO", "ARWR", "ASGN",
    "ASTE", "ATEC", "ATGE", "ATKR", "ATRC", "AUB", "AVA", "AVPT", "AWI", "AXNX",
    "AXTI", "AZZ", "BANC", "BANF", "BANR", "BBCP", "BBSI", "BCC", "BCO", "BCRX",
    "BDC", "BFC", "BGC", "BGFV", "BHB", "BHE", "BHLB", "BJRI", "BKE", "BL",
    "BLBD", "BMI", "BMRC", "BMRN", "BOH", "BOOM", "BPMC", "BSIG", "BSY", "BTU",
    "BV", "BVH", "BWA", "BXC", "BZH", "CADE", "CAKE", "CAL", "CALM", "CAR",
    "CARG", "CARS", "CASH", "CATY", "CBT", "CBU", "CDLX", "CECE", "CENT", "CENTA",
    "CENX", "CERT", "CFFN", "CHGG", "CHX", "CIVI", "CLDX", "CLF", "CNK", "CNS",
    "CNX", "COLB", "COLM", "COMP", "CONN", "COOP", "CORE", "CPF", "CPLG", "CPRX",
    "CRAI", "CRC", "CRGY", "CROX", "CRS", "CRUS", "CSGS", "CSWI", "CSX", "CTRE",
    "CTS", "CUBE", "CUTR", "CVCO", "CVI", "CWST", "CXM", "CXW", "CYRX", "DBI",
    "DCO", "DDS", "DENN", "DGII", "DIN", "DIOD", "DK", "DLX", "DNUT", "DOC",
    "DPZ", "DRH", "DRQ", "DXC", "EAT", "EFC", "EFSC", "EGRX", "ELS", "ELY",
    "EME", "ENS", "ENV", "ENVA", "EPAC", "EPC", "EPRT", "ESI", "ETD", "EVH",
    "EVO", "EVRG", "EXLS", "EXPO", "EXTR", "FBNC", "FCFS", "FCN", "FFBC", "FFIN",
    "FHB", "FHN", "FIBK", "FIVE", "FIX", "FL", "FLIC", "FLO", "FLR", "FMBH",
    "FN", "FORR", "FOXF", "FRBA", "FREY", "FRGE", "FRT", "FTAI", "FTDR", "FTI",
    "FUL", "FUN", "GBX", "GCMG", "GDEN", "GDOT", "GES", "GFF", "GHC", "GIII",
    "GLPI", "GLT", "GMS", "GNW", "GO", "GOGO", "GPI", "GPMT", "GPRO", "GRBK",
    "GRPN", "GSHD", "GTLS", "GTY", "GVA", "HAFC", "HBI", "HBT", "HCI", "HEES",
    "HELE", "HFWA", "HIBB", "HIW", "HLNE", "HMN", "HMST", "HNI", "HOMB", "HONE",
    "HOPE", "HOV", "HP", "HPP", "HRB", "HRMY", "HRT", "HSII", "HTGC", "HTH",
    "HTLD", "HUBG", "HURN", "HWBK", "HWKN", "HYLN", "ICUI", "IDA", "IIPR", "IMKTA",
    "INDB", "INGN", "INGR", "INSE", "INSM", "INSP", "INVA", "IOSP", "IRT", "ISBC",
    "ITGR", "ITRI", "JACK", "JBL", "JBLU", "JBSS", "JJSF", "JOBY", "JOE", "JOUT",
    "JWN", "KAI", "KALU", "KE", "KFY", "KIDS", "KMT", "KNF", "KOP", "KPTI",
    "KRYS", "KTB", "KW", "KWR", "LAUR", "LBRT", "LC", "LE", "LEU", "LFST",
    "LGND", "LH", "LIVN", "LMND", "LNC", "LNW", "LPLA", "LRCX", "LZB", "MAC",
    "MAN", "MATX", "MBC", "MBIN", "MBUU", "MC", "MCRI", "MCY", "MDC", "MDP",
    "MDU", "MED", "MGY", "MIDD", "MLI", "MLKN", "MMI", "MNRO", "MOG-A", "MOH",
    "MOV", "MP", "MRCY", "MRTN", "MSFT", "MSGE", "MSGS", "MSM", "MTRN", "MTRX",
    "MTW", "MWA", "MYE", "MYRG", "NARI", "NATR", "NAVI", "NBR", "NEWT", "NFG",
    "NJR", "NKLA", "NMIH", "NMRK", "NOG", "NPK", "NSP", "NSSC", "NTGR", "NUS",
    "NVEE", "NWBI", "NWE", "NWN", "NX", "NXST", "OBK", "OCFC", "OCSL", "ODP",
    "OFG", "OGS", "OI", "OLN", "OPRA", "OPY", "ORI", "ORN", "OSIS", "OSPN",
    "OSTK", "OUT", "OXM", "PACK", "PAGS", "PARR", "PATK", "PAYC", "PBF", "PCH",
    "PCYO", "PDCE", "PDFS", "PEB", "PEBO", "PERI", "PETS", "PGRE", "PHIN", "PHM",
    "PHR", "PII", "PLAY", "PLNT", "PLOW", "PLPC", "PLUS", "PLXS", "PNW", "POOL",
    "POR", "POWL", "POWW", "PPBI", "PRAA", "PRDO", "PRFT", "PRG", "PRIM", "PRK",
    "PRLB", "PRMW", "PRTY", "PSEC", "PSMT", "PSN", "PSTL", "PTEN", "PTGX", "PUMP",
    "PVH", "PWR", "PZZA", "QCRH", "QNST", "QTWO", "R", "RAPT", "RBA", "RBC",
    "RCM", "RCUS", "RDN", "RDNT", "RES", "REVG", "REX", "REYN", "RGEN", "RGNX",
    "RGR", "RILYK", "ROAD", "ROCK", "ROG", "ROIC", "RPAY", "RRX", "RSI", "RUSHA",
    "RWT", "RYI", "SAFE", "SAFM", "SAH", "SAIC", "SAM", "SAMG", "SANA", "SBCF",
    "SBLK", "SBSI", "SCCO", "SCHL", "SCI", "SCOR", "SCS", "SCSC", "SCVL", "SEAS",
    "SEM", "SF", "SFBS", "SFL", "SGA", "SHAK", "SHEN", "SHO", "SIEB", "SITC",
    "SITE", "SJW", "SKIN", "SKT", "SKY", "SLG", "SLGN", "SLVM", "SM", "SMP",
    "SMPL", "SNDR", "SNDX", "SNEX", "SNFCA", "SNV", "SONO", "SPB", "SPNS", "SPRY",
    "SPT", "SR", "SRCE", "SRDX", "SRT", "SSB", "SSD", "SSP", "SSTK", "STAA",
    "STAG", "STBA", "STC", "STER", "STFC", "STGW", "STRA", "STRL", "STT", "SUM",
    "SUPN", "SVC", "SWBI", "SWX", "SXC", "SXI", "SXT", "SYBT", "SYNH", "TBI",
    "TCBI", "TCBK", "TCMD", "TCRR", "TDOC", "TDS", "TDW", "TFII", "TGNA", "THC",
    "THFF", "THG", "THRM", "THRY", "TILE", "TIPT", "TISI", "TITN", "TKR", "TMHC",
    "TNC", "TNDM", "TNL", "TR", "TREE", "TREX", "TRIP", "TRMK", "TRNS", "TROX",
    "TRTN", "TSC", "TSE", "TTC", "TTEC", "TTGT", "TTI", "TTMI", "TTNP", "TUSK",
    "TVTX", "TWI", "TWO", "TXMD", "UAA", "UCBI", "UE", "UFCS", "UFI", "UFPI",
    "UFPT", "UHT", "UIS", "UMBF", "UMH", "UNF", "UNFI", "UNIT", "UNTY", "UPLD",
    "URBN", "URI", "USB", "USNA", "UTI", "UTL", "UTZ", "UVE", "UVSP", "UVV",
    "VAC", "VBNK", "VC", "VECO", "VHI", "VIAV", "VICI", "VIRT", "VITL", "VMD",
    "VOXX", "VRA", "VRAY", "VRDN", "VREX", "VSAT", "VSCO", "VSEC", "VSH", "VSTO",
    "VTOL", "VVI", "VYGR", "WABC", "WAFD", "WB", "WD", "WDFC", "WERN", "WETF",
    "WGO", "WHD", "WIRE", "WIT", "WK", "WLDN", "WLY", "WMK", "WMS", "WNS",
    "WOR", "WPC", "WRB", "WRBY", "WRLD", "WSBC", "WSBF", "WSC", "WSFS", "WSO",
    "WSR", "WTBA", "WTI", "WTRG", "WTS", "WTTR", "WW", "WWE", "WWW", "WYY",
    "XEL", "XMTR", "XOM", "XPER", "XPO", "YETI", "YORW", "YOU", "YPF", "ZBH",
    "ZBRA", "ZD", "ZEPP", "ZIP", "ZUO", "ZWS",
    # ===== International ADRs =====
    "ABEV", "ALC", "AMX", "ASR", "AZN", "BABA", "BAP", "BBD", "BBVA", "BCH",
    "BCS", "BHP", "BIDU", "BMA", "BNS", "BP", "BSAC", "BTI", "BUD", "CCL",
    "CHL", "CHT", "CIB", "CIG", "CM", "CNH", "CNI", "COE", "COR", "CRH",
    "CS", "CSAN", "CUK", "CYBR", "DB", "DEO", "DOOO", "ECH", "EDU", "ELP",
    "EOAN", "ERIC", "ERJ", "ESLT", "FMX", "FORTY", "FRO", "GFI", "GGB", "GLOB",
    "GMAB", "GOLD", "GRMN", "GSK", "HBM", "HDB", "HMC", "HSBC", "ICL", "IDA",
    "IHG", "INDA", "ING", "IPHI", "IRM", "ITUB", "JD", "KB", "KEP", "KOF",
    "LFC", "LI", "LOGI", "LX", "LYG", "MFC", "MFG", "MMC", "MMP", "MMYT",
    "MUFG", "NIO", "NOK", "NTES", "NVO", "NVS", "ORAN", "PAGS", "PBR", "PDD",
    "PHG", "PHI", "PSO", "RBA", "RDY", "RIO", "RY", "RYAAY", "SAN", "SAP",
    "SCCO", "SE", "SHEL", "SHOP", "SIEGY", "SKM", "SLB", "SNN", "SNY", "SONY",
    "STM", "STT", "SUZ", "TAL", "TCEHY", "TCK", "TD", "TEF", "TEVA", "TLK",
    "TM", "TME", "TS", "TSEM", "TSM", "TTE", "TX", "UBS", "UL", "UMC",
    "VALE", "VIPS", "VIST", "VOD", "WCN", "WIT", "WIX", "WNS", "XPEV", "YPF",
    "YY",
    # ===== Sector / country / thematic ETFs =====
    "ACWI", "ARKG", "ARKK", "ARKQ", "ARKW", "BLOK", "DIA", "EEM", "EFA", "EWA",
    "EWC", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWM", "EWN", "EWO", "EWP",
    "EWQ", "EWS", "EWT", "EWU", "EWW", "EWY", "EWZ", "EZA", "EZU", "FAS",
    "FAZ", "FCG", "FXI", "GDX", "GDXJ", "ICLN", "IDV", "IEMG", "IGV", "IHI",
    "ITA", "ITB", "IWM", "IWN", "IYR", "JETS", "KBE", "KRE", "KWEB", "LIT",
    "MJ", "MOAT", "PALL", "PFF", "PSCT", "QQQM", "REM", "REMX", "ROBO", "RWR",
    "SCHA", "SCHB", "SCHD", "SCHE", "SCHF", "SCHG", "SCHM", "SCHV", "SDS", "SDY",
    "SH", "SLV", "SMH", "SOXL", "SOXS", "SPDN", "SPLG", "SPLV", "SPSM", "SPTM",
    "SPTM", "SPXL", "SPXS", "SPYG", "SPYV", "SRTY", "SSO", "TAN", "THD", "TLT",
    "TMV", "TNA", "TQQQ", "TZA", "UNG", "UPRO", "URA", "URTY", "USMV", "USO",
    "UVXY", "VBK", "VBR", "VCR", "VDC", "VDE", "VFH", "VGT", "VHT", "VIS",
    "VIXY", "VLUE", "VNQ", "VONE", "VOO", "VOX", "VPL", "VPU", "VTV", "VUG",
    "VWO", "VXUS", "VYM", "WEED", "XBI", "XLB", "XLC", "XLE", "XLF", "XLG",
    "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XME", "XOP", "XPH",
    "XRT", "XTL", "XTN", "YINN",
]


def fetch_close(ticker: str, start: str = "1995-01-01") -> pd.Series | None:
    try:
        d = yf.download(ticker, start=start, progress=False, threads=False, auto_adjust=True)
        if d is None or d.empty:
            return None
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype("float64")
        if len(close) < 60:  # less than ~3 months — too sparse
            return None
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.name = ticker
        return close
    except Exception:
        return None


def main() -> None:
    panel = pd.read_parquet(PANEL)
    print(f"Existing panel: {panel.shape}  range {panel.index.min().date()} → {panel.index.max().date()}")
    existing = set(panel.columns)
    new_tickers = [t for t in dict.fromkeys(EXPANSION) if t not in existing]
    print(f"New tickers to fetch: {len(new_tickers)}")

    fetched: dict[str, pd.Series] = {}
    n_fetched = 0
    n_failed = 0
    for i, t in enumerate(new_tickers):
        s = fetch_close(t, start="1995-01-01")
        if s is None:
            n_failed += 1
            continue
        fetched[t] = s
        n_fetched += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(new_tickers)}] fetched={n_fetched} failed={n_failed}")
    print(f"  TOTAL: fetched={n_fetched}, failed={n_failed}")

    if not fetched:
        print("Nothing to merge.")
        return

    add_df = pd.concat(fetched.values(), axis=1, keys=fetched.keys())
    out = panel.join(add_df, how="outer").sort_index()
    print(f"After merge: {out.shape}")

    # Forward-fill within each new ticker's [first_valid, last_valid] range
    for c in fetched.keys():
        s = out[c]
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None or lv is None:
            continue
        mask = (out.index >= fv) & (out.index <= lv)
        out.loc[mask, c] = s[mask].ffill().values

    out.to_parquet(PANEL, compression="zstd")
    size_mb = PANEL.stat().st_size / 1024 / 1024
    print(f"Wrote {PANEL} ({out.shape})  size {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
