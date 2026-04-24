"""Curated universe of names with liquid monthly-options markets.

These are S&P 100 + major ETFs — the tickers where a monthly 3rd-Friday
OTM call/put has tight bid-ask, real open interest, and reliable fills.
Restricted list = better trade quality at the cost of breadth.
"""

LIQUID_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "ORCL", "ADBE", "CRM", "NFLX", "INTC", "AMD", "QCOM", "CSCO",
    "TXN", "IBM", "INTU",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "BLK",
    "SCHW", "AXP", "BRK-B",
    # Healthcare
    "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "UNH", "CVS",
    "AMGN", "DHR", "BMY", "GILD",
    # Consumer
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE", "LOW",
    "SBUX", "DIS", "TGT",
    # Industrials / energy / materials
    "BA", "CAT", "UPS", "HON", "RTX", "LMT", "DE", "GE",
    "XOM", "CVX", "COP", "SLB",
    # Communication / media
    "T", "VZ", "CMCSA",
    # Real estate / utilities (large-cap)
    "NEE", "AMT", "PLD",
    # Major ETFs — the most liquid options products in the world
    "SPY", "QQQ", "IWM", "DIA", "EEM", "XLF", "XLE", "XLK", "XLV",
    "XLY", "XLI", "XLU", "XLP", "XLB", "XLC", "TLT", "GLD", "SLV",
    "USO", "UNG",
]

assert len(LIQUID_TICKERS) == len(set(LIQUID_TICKERS)), "dupes in liquid list"
