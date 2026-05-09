"""
Download a broad universe of individual international stocks via yfinance.

Coverage:
  - Japan (TOPIX large/mid caps): suffix .T (Tokyo)
  - UK (FTSE 100 + FTSE 250): suffix .L (London)
  - Germany (DAX + MDAX): suffix .DE (XETRA)
  - France (CAC 40): suffix .PA (Paris)
  - Hong Kong (Hang Seng): suffix .HK
  - Switzerland (SMI): suffix .SW
  - Netherlands (AEX): suffix .AS
  - Australia (ASX 200): suffix .AX
  - Korea (KOSPI 200): suffix .KS
  - Canada (TSX 60): suffix .TO

These ticker lists are mostly drawn from current major-index constituents,
which means survivorship bias for delisted international names (like the
S&P 500 case before we backfilled). For honest research we'll note this
limitation; international delisted-stock data from Yahoo is far less reliable
than US data.

Saves:
  - data/intl_tickers_list.csv — the universe definition (ticker, country, name)
  - data/intl_prices.parquet — daily close panel
  - data/intl_download_log.csv — fetch attempt log

Run:
    python3 -m experiments.monthly_dca.v3_universes.download_international
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments" / "monthly_dca" / "v3_universes" / "data"
DATA.mkdir(parents=True, exist_ok=True)


# === Universe lists (current major-index constituents) ===

# Japan: TOPIX Core 30 + Nikkei 225 sample of liquid names with .T suffix
JAPAN = [
    # Nikkei 225 large caps
    "7203.T",  # Toyota
    "6758.T",  # Sony
    "9432.T",  # NTT
    "9984.T",  # SoftBank
    "8306.T",  # Mitsubishi UFJ
    "6861.T",  # Keyence
    "8316.T",  # Sumitomo Mitsui Financial
    "8035.T",  # Tokyo Electron
    "9433.T",  # KDDI
    "4063.T",  # Shin-Etsu Chemical
    "6098.T",  # Recruit Holdings
    "7974.T",  # Nintendo
    "8001.T",  # Itochu
    "8002.T",  # Marubeni
    "8031.T",  # Mitsui & Co
    "8053.T",  # Sumitomo Corp
    "8058.T",  # Mitsubishi Corp
    "9434.T",  # SoftBank Corp
    "4502.T",  # Takeda
    "4503.T",  # Astellas
    "4519.T",  # Chugai Pharma
    "4523.T",  # Eisai
    "4568.T",  # Daiichi Sankyo
    "4661.T",  # Oriental Land
    "4901.T",  # Fujifilm
    "4911.T",  # Shiseido
    "5108.T",  # Bridgestone
    "5401.T",  # Nippon Steel
    "5713.T",  # Sumitomo Metal Mining
    "5802.T",  # Sumitomo Electric
    "5938.T",  # LIXIL
    "6178.T",  # Japan Post
    "6273.T",  # SMC
    "6301.T",  # Komatsu
    "6326.T",  # Kubota
    "6367.T",  # Daikin
    "6501.T",  # Hitachi
    "6502.T",  # Toshiba
    "6503.T",  # Mitsubishi Electric
    "6594.T",  # Nidec
    "6701.T",  # NEC
    "6702.T",  # Fujitsu
    "6752.T",  # Panasonic
    "6762.T",  # TDK
    "6902.T",  # Denso
    "6954.T",  # Fanuc
    "6971.T",  # Kyocera
    "6981.T",  # Murata
    "7011.T",  # Mitsubishi Heavy
    "7201.T",  # Nissan
    "7267.T",  # Honda
    "7269.T",  # Suzuki
    "7270.T",  # Subaru
    "7733.T",  # Olympus
    "7741.T",  # Hoya
    "7751.T",  # Canon
    "7832.T",  # Bandai Namco
    "8113.T",  # Unicharm
    "8267.T",  # Aeon
    "8411.T",  # Mizuho Financial
    "8591.T",  # Orix
    "8601.T",  # Daiwa Securities
    "8604.T",  # Nomura
    "8630.T",  # SOMPO
    "8725.T",  # MS&AD Insurance
    "8750.T",  # Dai-ichi Life
    "8766.T",  # Tokio Marine
    "8801.T",  # Mitsui Fudosan
    "8802.T",  # Mitsubishi Estate
    "9020.T",  # JR East
    "9021.T",  # JR West
    "9022.T",  # JR Central
    "9101.T",  # NYK Line
    "9104.T",  # Mitsui OSK Lines
    "9201.T",  # JAL
    "9202.T",  # ANA
    "9301.T",  # Mitsubishi Logistics
    "9501.T",  # TEPCO
    "9531.T",  # Tokyo Gas
    "9602.T",  # Toho
    "9613.T",  # NTT Data
    "9735.T",  # Secom
    "9766.T",  # Konami
    "9983.T",  # Fast Retailing
    "1928.T",  # Sekisui House
    "2502.T",  # Asahi Group
    "2503.T",  # Kirin
    "2802.T",  # Ajinomoto
    "2914.T",  # JT (Japan Tobacco)
    "3382.T",  # Seven & I
    "3402.T",  # Toray Industries
    "3407.T",  # Asahi Kasei
    "3659.T",  # Nexon
    "4005.T",  # Sumitomo Chemical
    "4042.T",  # Tosoh
    "4188.T",  # Mitsubishi Chemical
    "4452.T",  # Kao
    "4507.T",  # Shionogi
    "4528.T",  # Ono Pharma
    "4543.T",  # Terumo
    "4578.T",  # Otsuka Holdings
    "4612.T",  # Nippon Paint
    "4689.T",  # LY Corp (Z Holdings)
    "4704.T",  # Trend Micro
    "4732.T",  # USS
    "4751.T",  # CyberAgent
    "4755.T",  # Rakuten
    "4901.T",  # Fujifilm (dup, removing)
    "5019.T",  # Idemitsu Kosan
    "5020.T",  # Eneos
    "5201.T",  # AGC (Asahi Glass)
    "5232.T",  # Sumitomo Osaka Cement
    "5333.T",  # NGK Insulators
    "5631.T",  # Japan Steel Works
    "5947.T",  # Rinnai
    "6062.T",  # Charm Care
    "6113.T",  # Amada
    "6201.T",  # Toyota Industries
    "6479.T",  # Minebea Mitsumi
    "6504.T",  # Fuji Electric
    "6645.T",  # Omron
    "6770.T",  # Alps Alpine
    "6841.T",  # Yokogawa Electric
    "6857.T",  # Advantest
    "6920.T",  # Lasertec
    "7211.T",  # Mitsubishi Motors
    "7259.T",  # Aisin
    "7261.T",  # Mazda
    "7272.T",  # Yamaha Motor
    "7309.T",  # Shimano
    "7459.T",  # Mediken
    "7532.T",  # Pan Pacific
    "7550.T",  # Zensho
    "7732.T",  # Topcon
    "7747.T",  # Asahi Intecc
    "7912.T",  # Dai Nippon Printing
    "8035.T",  # Tokyo Electron (dup)
    "8233.T",  # Takashimaya
    "8252.T",  # Marui
    "8331.T",  # Chiba Bank
    "8354.T",  # Fukuoka Financial
    "8355.T",  # Shizuoka Bank
    "8511.T",  # Japan Securities Finance
    "8697.T",  # Japan Exchange Group
    "8804.T",  # Tokyo Tatemono
    "9007.T",  # Odakyu
    "9008.T",  # Keio
    "9009.T",  # Keisei
    "9064.T",  # Yamato Holdings
    "9532.T",  # Osaka Gas
    "9603.T",  # H.I.S.
    "9684.T",  # Square Enix
    "9697.T",  # Capcom
    "9962.T",  # Misumi
]

# UK FTSE 100
UK = [
    "AAL.L",   # Anglo American
    "ABF.L",   # Associated British Foods
    "ADM.L",   # Admiral
    "AHT.L",   # Ashtead
    "ANTO.L",  # Antofagasta
    "AV.L",    # Aviva
    "AZN.L",   # AstraZeneca
    "BA.L",    # BAE Systems
    "BARC.L",  # Barclays
    "BATS.L",  # British American Tobacco
    "BDEV.L",  # Barratt
    "BEZ.L",   # Beazley
    "BKG.L",   # Berkeley
    "BLND.L",  # British Land
    "BME.L",   # B&M
    "BNZL.L",  # Bunzl
    "BP.L",    # BP
    "BT-A.L",  # BT Group
    "CCH.L",   # Coca-Cola HBC
    "CNA.L",   # Centrica
    "CPG.L",   # Compass
    "CRDA.L",  # Croda
    "CRH.L",   # CRH
    "CTEC.L",  # ConvaTec
    "DCC.L",   # DCC
    "DGE.L",   # Diageo
    "EXPN.L",  # Experian
    "EZJ.L",   # easyJet
    "FCIT.L",  # F&C Investment
    "FRES.L",  # Fresnillo
    "GLEN.L",  # Glencore
    "GSK.L",   # GSK
    "HIK.L",   # Hikma
    "HL.L",    # Hargreaves Lansdown
    "HLMA.L",  # Halma
    "HSBA.L",  # HSBC
    "HSX.L",   # Hiscox
    "HWDN.L",  # Howdens Joinery
    "IAG.L",   # IAG
    "ICG.L",   # Intermediate Capital
    "IHG.L",   # InterContinental Hotels
    "III.L",   # 3i
    "IMB.L",   # Imperial Brands
    "INF.L",   # Informa
    "ITRK.L",  # Intertek
    "JD.L",    # JD Sports
    "KGF.L",   # Kingfisher
    "LAND.L",  # Land Securities
    "LGEN.L",  # Legal & General
    "LLOY.L",  # Lloyds
    "LSEG.L",  # London Stock Exchange
    "MKS.L",   # Marks & Spencer
    "MNDI.L",  # Mondi
    "MNG.L",   # M&G
    "NG.L",    # National Grid
    "NWG.L",   # NatWest
    "NXT.L",   # Next
    "OCDO.L",  # Ocado
    "PHNX.L",  # Phoenix
    "PRU.L",   # Prudential
    "PSH.L",   # Pershing Square Holdings
    "PSN.L",   # Persimmon
    "PSON.L",  # Pearson
    "REL.L",   # RELX
    "RIO.L",   # Rio Tinto
    "RKT.L",   # Reckitt Benckiser
    "RMV.L",   # Rightmove
    "RR.L",    # Rolls-Royce
    "RTO.L",   # Rentokil
    "SBRY.L",  # Sainsbury's
    "SDR.L",   # Schroders
    "SGE.L",   # Sage
    "SGRO.L",  # Segro
    "SHEL.L",  # Shell
    "SMDS.L",  # Smurfit Westrock
    "SMIN.L",  # Smiths Group
    "SMT.L",   # Scottish Mortgage
    "SN.L",    # Smith & Nephew
    "SPX.L",   # Spirax
    "SSE.L",   # SSE
    "STAN.L",  # Standard Chartered
    "STJ.L",   # St. James's Place
    "SVT.L",   # Severn Trent
    "TSCO.L",  # Tesco
    "TW.L",    # Taylor Wimpey
    "ULVR.L",  # Unilever
    "UTG.L",   # Unite Group
    "UU.L",    # United Utilities
    "VOD.L",   # Vodafone
    "WEIR.L",  # Weir Group
    "WPP.L",   # WPP
    "WTB.L",   # Whitbread
]

# Germany DAX 40
GERMANY = [
    "ADS.DE",  # Adidas
    "AIR.DE",  # Airbus
    "ALV.DE",  # Allianz
    "BAS.DE",  # BASF
    "BAYN.DE", # Bayer
    "BEI.DE",  # Beiersdorf
    "BMW.DE",  # BMW
    "BNR.DE",  # Brenntag
    "CBK.DE",  # Commerzbank
    "CON.DE",  # Continental
    "DBK.DE",  # Deutsche Bank
    "DB1.DE",  # Deutsche Boerse
    "DHL.DE",  # DHL Group
    "DTE.DE",  # Deutsche Telekom
    "DTG.DE",  # Daimler Truck
    "ENR.DE",  # Siemens Energy
    "EOAN.DE", # E.ON
    "FME.DE",  # Fresenius Medical
    "FRE.DE",  # Fresenius
    "HEI.DE",  # Heidelberg Materials
    "HEN3.DE", # Henkel
    "HFG.DE",  # HelloFresh
    "HNR1.DE", # Hannover Rück
    "IFX.DE",  # Infineon
    "MBG.DE",  # Mercedes-Benz
    "MRK.DE",  # Merck KGaA
    "MTX.DE",  # MTU Aero
    "MUV2.DE", # Munich Re
    "P911.DE", # Porsche
    "PAH3.DE", # Porsche Auto Holding
    "QIA.DE",  # Qiagen
    "RHM.DE",  # Rheinmetall
    "RWE.DE",  # RWE
    "SAP.DE",  # SAP
    "SHL.DE",  # Siemens Healthineers
    "SIE.DE",  # Siemens
    "SY1.DE",  # Symrise
    "VNA.DE",  # Vonovia
    "VOW3.DE", # Volkswagen
    "ZAL.DE",  # Zalando
]

# France CAC 40
FRANCE = [
    "AC.PA",   # Accor
    "AI.PA",   # Air Liquide
    "AIR.PA",  # Airbus (also)
    "ALO.PA",  # Alstom
    "BN.PA",   # Danone
    "BNP.PA",  # BNP Paribas
    "CA.PA",   # Carrefour
    "CAP.PA",  # Capgemini
    "CS.PA",   # AXA
    "DG.PA",   # Vinci
    "DSY.PA",  # Dassault
    "EL.PA",   # EssilorLuxottica
    "EN.PA",   # Bouygues
    "ENGI.PA", # Engie
    "ERF.PA",  # Eurofins
    "GLE.PA",  # Société Générale
    "HO.PA",   # Thales
    "KER.PA",  # Kering
    "LR.PA",   # Legrand
    "MC.PA",   # LVMH
    "ML.PA",   # Michelin
    "OR.PA",   # L'Oreal
    "ORA.PA",  # Orange
    "PUB.PA",  # Publicis
    "RI.PA",   # Pernod Ricard
    "RMS.PA",  # Hermes
    "RNO.PA",  # Renault
    "SAF.PA",  # Safran
    "SAN.PA",  # Sanofi
    "SGO.PA",  # Saint-Gobain
    "STLAP.PA",# Stellantis
    "STMPA.PA",# STMicroelectronics
    "SU.PA",   # Schneider Electric
    "SW.PA",   # Sodexo
    "TEP.PA",  # Teleperformance
    "TTE.PA",  # TotalEnergies
    "URW.AS",  # Unibail (listed in Amsterdam)
    "VIE.PA",  # Veolia
    "VIV.PA",  # Vivendi
    "WLN.PA",  # Worldline
]

# Hong Kong
HK = [
    "0001.HK", # CK Hutchison
    "0002.HK", # CLP
    "0003.HK", # HK and China Gas
    "0005.HK", # HSBC
    "0006.HK", # Power Assets
    "0011.HK", # Hang Seng Bank
    "0012.HK", # Henderson Land
    "0016.HK", # Sun Hung Kai
    "0017.HK", # New World Development
    "0027.HK", # Galaxy Entertainment
    "0066.HK", # MTR
    "0083.HK", # Sino Land
    "0101.HK", # Hang Lung Properties
    "0144.HK", # China Merchants Port
    "0175.HK", # Geely
    "0241.HK", # Alibaba Health
    "0267.HK", # CITIC
    "0288.HK", # WH Group
    "0291.HK", # China Resources Beer
    "0386.HK", # Sinopec
    "0388.HK", # HKEX
    "0688.HK", # China Overseas Land
    "0700.HK", # Tencent
    "0762.HK", # China Unicom
    "0788.HK", # China Tower
    "0823.HK", # Link REIT
    "0857.HK", # PetroChina
    "0883.HK", # CNOOC
    "0939.HK", # CCB
    "0941.HK", # China Mobile
    "0992.HK", # Lenovo
    "1024.HK", # Kuaishou
    "1038.HK", # CK Infrastructure
    "1093.HK", # CSPC Pharma
    "1109.HK", # China Resources Land
    "1113.HK", # CK Asset
    "1177.HK", # Sino Biopharm
    "1211.HK", # BYD
    "1299.HK", # AIA
    "1398.HK", # ICBC
    "1810.HK", # Xiaomi
    "1876.HK", # Budweiser APAC
    "1929.HK", # Chow Tai Fook
    "1997.HK", # Wharf REIC
    "2007.HK", # Country Garden
    "2018.HK", # AAC Tech
    "2020.HK", # ANTA Sports
    "2269.HK", # WuXi Bio
    "2313.HK", # Shenzhou
    "2318.HK", # Ping An
    "2319.HK", # Mengniu
    "2331.HK", # Li Ning
    "2382.HK", # Sunny Optical
    "2388.HK", # BOC Hong Kong
    "2628.HK", # China Life
    "2688.HK", # ENN Energy
    "2899.HK", # Zijin Mining
    "3690.HK", # Meituan
    "3968.HK", # CMB
    "3988.HK", # BoC
    "9618.HK", # JD.com
    "9633.HK", # Nongfu Spring
    "9888.HK", # Baidu
    "9988.HK", # Alibaba
    "9999.HK", # NetEase
]

# Switzerland SMI
SWITZERLAND = [
    "ABBN.SW", # ABB
    "ALC.SW",  # Alcon
    "CSGN.SW", # Credit Suisse (delisted)
    "GEBN.SW", # Geberit
    "GIVN.SW", # Givaudan
    "HOLN.SW", # Holcim
    "KNIN.SW", # Kuehne + Nagel
    "LOGN.SW", # Logitech
    "LONN.SW", # Lonza
    "NESN.SW", # Nestlé
    "NOVN.SW", # Novartis
    "PGHN.SW", # Partners Group
    "RIGN.SW", # Richemont (also)
    "ROG.SW",  # Roche
    "SCMN.SW", # Swisscom
    "SGSN.SW", # SGS
    "SIKA.SW", # Sika
    "SLHN.SW", # Swiss Life
    "SOON.SW", # Sonova
    "SREN.SW", # Swiss Re
    "UBSG.SW", # UBS
    "UHR.SW",  # Swatch
    "ZURN.SW", # Zurich
]

# Australia ASX 200 (sample)
AUSTRALIA = [
    "BHP.AX",  # BHP
    "RIO.AX",  # Rio Tinto
    "CBA.AX",  # Commonwealth Bank
    "WBC.AX",  # Westpac
    "NAB.AX",  # NAB
    "ANZ.AX",  # ANZ
    "MQG.AX",  # Macquarie
    "WES.AX",  # Wesfarmers
    "WOW.AX",  # Woolworths
    "TLS.AX",  # Telstra
    "CSL.AX",  # CSL
    "FMG.AX",  # Fortescue
    "WDS.AX",  # Woodside
    "STO.AX",  # Santos
    "QAN.AX",  # Qantas
    "SUN.AX",  # Suncorp
    "QBE.AX",  # QBE
    "ALL.AX",  # Aristocrat
    "GMG.AX",  # Goodman Group
    "WPL.AX",  # Woodside Petroleum (old)
    "S32.AX",  # South32
    "NCM.AX",  # Newcrest
    "AMC.AX",  # Amcor
    "ORG.AX",  # Origin Energy
    "TCL.AX",  # Transurban
    "SCG.AX",  # Scentre
    "REA.AX",  # REA Group
    "XRO.AX",  # Xero
    "WTC.AX",  # WiseTech
    "JBH.AX",  # JB Hi-Fi
]

# Canada TSX 60
CANADA = [
    "RY.TO",   # Royal Bank
    "TD.TO",   # TD Bank
    "BNS.TO",  # Scotiabank
    "BMO.TO",  # BMO
    "CM.TO",   # CIBC
    "MFC.TO",  # Manulife
    "SLF.TO",  # Sun Life
    "ENB.TO",  # Enbridge
    "TRP.TO",  # TC Energy
    "SU.TO",   # Suncor
    "CNQ.TO",  # CNRL
    "CVE.TO",  # Cenovus
    "IMO.TO",  # Imperial Oil
    "BCE.TO",  # BCE
    "T.TO",    # Telus
    "RCI-B.TO",# Rogers
    "CNR.TO",  # CN Rail
    "CP.TO",   # CP Rail
    "L.TO",    # Loblaw
    "ATD.TO",  # Couche-Tard
    "MG.TO",   # Magna
    "DOL.TO",  # Dollarama
    "GIB-A.TO",# CGI
    "CSU.TO",  # Constellation Software
    "OTEX.TO", # Open Text
    "SHOP.TO", # Shopify
    "NTR.TO",  # Nutrien
    "POU.TO",  # Paramount Resources
    "BAM.TO",  # Brookfield
    "BN.TO",   # Brookfield Corp
    "BIP-UN.TO",# Brookfield Infra
    "FTS.TO",  # Fortis
    "EMA.TO",  # Emera
    "H.TO",    # Hydro One
    "WCN.TO",  # Waste Connections
    "ABX.TO",  # Barrick
    "AEM.TO",  # Agnico Eagle
    "GIL.TO",  # Gildan
    "MFI.TO",  # Maple Leaf Foods
    "WN.TO",   # Weston
    "QSR.TO",  # Restaurant Brands
    "IFC.TO",  # Intact
    "MRU.TO",  # Metro
    "DOO.TO",  # BRP
    "NA.TO",   # National Bank
]

# Korea KOSPI 200 sample
KOREA = [
    "005930.KS",# Samsung Electronics
    "000660.KS",# SK Hynix
    "035720.KS",# Kakao
    "035420.KS",# Naver
    "207940.KS",# Samsung Biologics
    "005380.KS",# Hyundai Motor
    "051910.KS",# LG Chem
    "005490.KS",# POSCO
    "006400.KS",# Samsung SDI
    "068270.KS",# Celltrion
    "012330.KS",# Hyundai Mobis
    "028260.KS",# Samsung C&T
    "066570.KS",# LG Electronics
    "096770.KS",# SK Innovation
    "017670.KS",# SK Telecom
    "030200.KS",# KT
    "086790.KS",# Hana Financial
    "055550.KS",# Shinhan
    "105560.KS",# KB Financial
    "316140.KS",# Woori Financial
    "003550.KS",# LG Corp
    "003670.KS",# POSCO Future M
    "010130.KS",# Korea Zinc
    "010950.KS",# S-Oil
    "032830.KS",# Samsung Life
    "036570.KS",# NCsoft
    "041510.KS",# SM Entertainment
    "047810.KS",# KAI
    "086280.KS",# Hyundai Glovis
    "090430.KS",# Amorepacific
]

# Netherlands AEX
NETHERLANDS = [
    "ASML.AS", # ASML
    "ABN.AS",  # ABN AMRO
    "ADYEN.AS",# Adyen
    "AD.AS",   # Ahold Delhaize
    "AGN.AS",  # Aegon
    "AKZA.AS", # AkzoNobel
    "ASRNL.AS",# ASR
    "DSFIR.AS",# DSM-Firmenich
    "EXO.AS",  # Exor
    "GLPG.AS", # Galapagos
    "HEIA.AS", # Heineken
    "IMCD.AS", # IMCD
    "INGA.AS", # ING
    "KPN.AS",  # KPN
    "MT.AS",   # ArcelorMittal
    "NN.AS",   # NN Group
    "PHIA.AS", # Philips
    "PRX.AS",  # Prosus
    "RAND.AS", # Randstad
    "REN.AS",  # Relx (also)
    "RDSA.AS", # Royal Dutch Shell (delisted)
    "UNA.AS",  # Unilever NV (delisted)
    "URW.AS",  # Unibail-Rodamco
    "WKL.AS",  # Wolters Kluwer
]

# Combine all
ALL_TICKERS = (
    [(t, "JP", "Japan") for t in JAPAN]
    + [(t, "GB", "UK") for t in UK]
    + [(t, "DE", "Germany") for t in GERMANY]
    + [(t, "FR", "France") for t in FRANCE]
    + [(t, "HK", "HongKong") for t in HK]
    + [(t, "CH", "Switzerland") for t in SWITZERLAND]
    + [(t, "AU", "Australia") for t in AUSTRALIA]
    + [(t, "CA", "Canada") for t in CANADA]
    + [(t, "KR", "Korea") for t in KOREA]
    + [(t, "NL", "Netherlands") for t in NETHERLANDS]
)


def main():
    print(f"Universe size: {len(ALL_TICKERS)} tickers")
    # Save universe definition
    pd.DataFrame(ALL_TICKERS, columns=["ticker", "country_iso", "country"]).drop_duplicates(
        subset=["ticker"]
    ).to_csv(DATA / "intl_tickers_list.csv", index=False)
    print("Saved intl_tickers_list.csv")

    series_map: dict[str, pd.Series] = {}
    log_rows = []
    unique_tickers = sorted(set(t for t, _, _ in ALL_TICKERS))

    for i, tkr in enumerate(unique_tickers):
        if i % 25 == 0:
            print(f"  [{i}/{len(unique_tickers)}] {tkr}...")
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
        if (i + 1) % 50 == 0:
            time.sleep(1.0)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(DATA / "intl_download_log.csv", index=False)
    print("\nDownload status counts:")
    print(log_df["status"].value_counts())

    if series_map:
        panel = pd.concat(series_map, axis=1).sort_index()
        panel.to_parquet(DATA / "intl_prices.parquet")
        print(f"\nSaved {panel.shape[1]} series to intl_prices.parquet")
        print(f"  Date range: {panel.index.min()} - {panel.index.max()}")


if __name__ == "__main__":
    main()
