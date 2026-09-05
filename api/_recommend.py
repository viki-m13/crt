"""Turning a finished profile into one recommended, verified stock.

Pipeline, in strict order:

  1. CANDIDATES  proposed by a search-grounded model when one is configured
                 (global, current, unbounded universe), else by Yahoo's live
                 predefined screeners mapped to the profile, else by a small
                 last-resort roster. Sources are layered, so the game still
                 finishes when the internet is uncooperative.
  2. VERIFY      every proposal is checked against real price history. A
                 ticker that does not exist is dropped here, which is what
                 stops a model inventing a company.
  3. RANK        the deterministic engine scores verified candidates. The
                 model has no vote in the ordering.
  4. NARRATE     a model writes the reveal copy, constrained to the numbers
                 we measured. If no model is available, the copy is
                 assembled from those same numbers.

The invariant worth stating plainly: every figure shown to a user is
computed in _market.py from real prices. Prose is the only thing a model is
allowed to produce, and it is handed the figures rather than asked for them.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import _engine as E
import _market as M

HDR = M.HDR
SCREENER = ("https://query1.finance.yahoo.com/v1/finance/screener/"
            "predefined/saved?scrIds={scr}&count={n}")

# Live Yahoo screeners, chosen by what the profile is asking for. These are
# lists of tickers fetched at request time, not stored fundamentals.
SCREENS = {
    "safe_income": ("portfolio_anchors", "undervalued_large_caps"),
    "value": ("undervalued_large_caps", "undervalued_growth_stocks"),
    "growth": ("growth_technology_stocks", "solid_large_growth"),
    "momentum": ("day_gainers", "most_actives"),
    "small": ("aggressive_small_caps", "small_cap_gainers"),
}

# Last resort only, when both the model and the screeners are unavailable.
# Deliberately broad and global; every metric is still fetched live, so this
# is a roster of names to look up, never a source of stale numbers.
FALLBACK_ROSTER = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD",
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "MCD", "HD", "UNH", "LLY", "ABBV",
    "JPM", "V", "MA", "BRK-B", "XOM", "CVX", "CAT", "BA", "GE", "HON",
    "NFLX", "DIS", "SBUX", "NKE", "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN",
    "ASML.AS", "SAP.DE", "MC.PA", "NESN.SW", "7203.T", "6758.T", "0700.HK",
    "RELIANCE.NS", "BHP.AX", "SHOP.TO", "NVO", "TSM", "BABA", "TM",
]

THEME_HINTS = {
    "ai": ["NVDA", "MSFT", "GOOGL", "AVGO", "TSM"],
    "semis": ["NVDA", "AMD", "TSM", "ASML.AS", "AVGO", "TXN"],
    "software": ["MSFT", "CRM", "ADBE", "ORCL", "SAP.DE"],
    "cloud": ["MSFT", "AMZN", "GOOGL", "ORCL"],
    "healthcare": ["JNJ", "UNH", "LLY", "ABBV", "NVO"],
    "biotech": ["LLY", "ABBV", "NVO", "AMGN"],
    "energy": ["XOM", "CVX", "COP", "SHEL"],
    "renewables": ["NEE", "ENPH", "FSLR", "VWS.CO"],
    "finance": ["JPM", "V", "MA", "BRK-B", "GS"],
    "consumer": ["PG", "KO", "PEP", "COST", "WMT", "NESN.SW"],
    "luxury": ["MC.PA", "RMS.PA", "CFR.SW", "TPR"],
    "retail": ["WMT", "COST", "HD", "TGT"],
    "industrial": ["CAT", "HON", "GE", "UNP"],
    "defense": ["LMT", "RTX", "NOC", "GD"],
    "space": ["RKLB", "LMT", "BA"],
    "autos": ["TSLA", "TM", "7203.T", "GM", "F"],
    "travel": ["BKNG", "MAR", "DAL", "ABNB"],
    "media": ["DIS", "NFLX", "CMCSA"],
    "gaming": ["NTDOY", "EA", "TTWO", "7974.T"],
    "crypto": ["COIN", "MSTR", "HOOD"],
    "materials": ["BHP.AX", "RIO", "LIN", "VALE"],
    "realestate": ["PLD", "AMT", "O"],
    "telecom": ["T", "VZ", "TMUS"],
    "utilities": ["NEE", "DUK", "SO"],
    "food": ["KO", "PEP", "NESN.SW", "MDLZ"],
    "ecommerce": ["AMZN", "BABA", "SHOP.TO", "MELI"],
    "robotics": ["ABBNY", "6954.T", "ROK"],
}

CANDIDATE_SYSTEM = """You propose stock tickers for a matching engine. You do \
NOT decide what gets recommended — a separate engine scores your proposals \
against measured market data, and any ticker that does not exist is discarded.

Rules:
- Propose REAL, currently-listed tickers in the exact form a quote service \
expects: bare for US (NVDA), suffixed elsewhere (7203.T, ASML.AS, MC.PA, \
0700.HK, RELIANCE.NS, BHP.AX, NESN.SW, SHOP.TO).
- Any exchange in the world is fair game. Prefer names that genuinely fit the \
profile over famous ones.
- Give a spread: some obvious, some less so. Never propose leveraged or \
inverse products, and never propose an index fund as the single answer.
- Tag each with themes from this list where they apply: """ + ", ".join(E.THEMES) + """

Return ONLY JSON:
{"candidates": [{"symbol": "NVDA", "themes": ["ai","semis"], \
"why": "one short clause"}, ...]}"""


def _profile_brief(profile: dict, answers: list[dict]) -> str:
    d = profile["dims"]
    def band(v, lo, hi):
        return lo if v < 0.35 else ("mid" if v < 0.65 else hi)
    themes = sorted(profile["themes"], key=lambda t: -profile["themes"][t])
    lines = [
        f"risk appetite: {band(d['risk'], 'preserve capital', 'swing for the fences')} ({d['risk']:.2f})",
        f"holding period: {band(d['horizon'], 'months', 'decades')} ({d['horizon']:.2f})",
        f"tolerance for wild swings: {band(d['volatility'], 'low', 'high')} ({d['volatility']:.2f})",
        f"style: {band(d['value_growth'], 'cheap and unloved', 'pays up for growth')}",
        f"momentum: {band(d['momentum'], 'contrarian', 'rides winners')}",
        f"size: {band(d['size'], 'small and obscure', 'mega-cap household')}",
        f"income: {band(d['income'], 'no dividend needed', 'wants to be paid')}",
        f"concentration: {band(d['conviction'], 'spread around', 'one big idea')}",
        f"target annualised volatility: {E._target_vol(profile):.0%}",
        f"themes they leaned toward: {', '.join(themes) if themes else 'none expressed'}",
    ]
    if profile.get("avoid"):
        lines.append(f"REFUSES to own: {', '.join(profile['avoid'])}")
    picks = [f"- {a.get('q','')[:80]} -> {a.get('a','')[:60]}" for a in answers[-8:]]
    return "\n".join(lines) + "\n\nTheir actual answers:\n" + "\n".join(picks)


def _screen(scr: str, n: int = 25) -> list[str]:
    try:
        d = M._get_json(SCREENER.format(scr=scr, n=n), timeout=10.0, tries=2)
        quotes = d["finance"]["result"][0]["quotes"]
        return [q["symbol"] for q in quotes if q.get("symbol")]
    except Exception:
        return []


def _screen_keys(profile: dict) -> list[str]:
    d = profile["dims"]
    keys = []
    if d["income"] > 0.55 or d["risk"] < 0.35:
        keys.append("safe_income")
    if d["value_growth"] < 0.4:
        keys.append("value")
    if d["value_growth"] > 0.6:
        keys.append("growth")
    if d["momentum"] > 0.65:
        keys.append("momentum")
    if d["size"] < 0.4:
        keys.append("small")
    return keys or ["growth", "value"]


def candidates(profile: dict, answers: list[dict], want: int = 16) -> dict:
    """Gather candidate tickers. Returns {symbols, themes, source, notes}."""
    themes: dict[str, list[str]] = {}
    notes: list[str] = []
    symbols: list[str] = []
    source = "none"

    # 1. search-grounded model, the only source with a truly global reach
    try:
        import _llm
        if _llm.available():
            user = (_profile_brief(profile, answers) +
                    f"\n\nPropose {want} candidate tickers for this person.")
            raw, prov = _llm.complete_json(
                [{"role": "system", "content": CANDIDATE_SYSTEM},
                 {"role": "user", "content": user}],
                task="live", max_tokens=900, temperature=0.7)
            for c in (raw.get("candidates") or [])[:want * 2]:
                if not isinstance(c, dict):
                    continue
                s = str(c.get("symbol") or "").strip().upper()
                if not s or len(s) > 16 or s in symbols:
                    continue
                symbols.append(s)
                tags = [str(t).lower() for t in (c.get("themes") or [])
                        if str(t).lower() in set(E.THEMES)]
                if tags:
                    themes[s] = tags
            if symbols:
                source = f"model:{prov}"
    except Exception as e:  # noqa: BLE001
        notes.append(f"model candidates unavailable ({type(e).__name__})")

    # 2. live screeners — dynamic, but US-centric
    if len(symbols) < 8:
        got = []
        for key in _screen_keys(profile):
            for scr in SCREENS.get(key, ()):
                got.extend(_screen(scr))
        for s in got:
            if s not in symbols:
                symbols.append(s)
        if got:
            source = source if symbols and source.startswith("model") else "screener"
        else:
            notes.append("live screeners unavailable")

    # 3. theme hints and the roster, so the game always finishes
    if len(symbols) < 8:
        for t in sorted(profile["themes"], key=lambda x: -profile["themes"][x]):
            for s in THEME_HINTS.get(t, []):
                if s not in symbols:
                    symbols.append(s)
                    themes.setdefault(s, []).append(t)
        for s in FALLBACK_ROSTER:
            if s not in symbols:
                symbols.append(s)
        if source == "none":
            source = "roster"

    return {"symbols": symbols[:max(want * 2, 24)], "themes": themes,
            "source": source, "notes": notes}


NARRATE_SYSTEM = """You write the reveal screen of a stock-matching game.

You are given a person's profile and ONE stock that a scoring engine already \
chose, with real measured numbers. Your job is only to explain the match in \
a way that feels personal and earned.

Hard rules:
- Use ONLY the numbers provided. Never state a price, return, or statistic \
that is not in the data given to you.
- Do not promise performance. No "will", no "guaranteed", no price targets.
- Second person, warm, confident, specific. No hype, no emoji, no disclaimers \
(the interface handles those).

Return ONLY JSON:
{"headline": "<=60 chars, punchy, names the company",
 "why": ["<=110 chars", "<=110 chars", "<=110 chars"],
 "watch_out": "<=130 chars, the honest risk of THIS stock for THIS person"}"""


def _facts(stock: dict, profile: dict) -> str:
    f = [f"company: {stock.get('name')} ({stock['symbol']})",
         f"exchange: {stock.get('exchange')} in {stock.get('currency')}"]
    if stock.get("price") is not None:
        f.append(f"price: {stock['price']} {stock.get('currency')}")
    if stock.get("ann_vol") is not None:
        f.append(f"annualised volatility: {stock['ann_vol']:.0%} "
                 f"(this person's target is {E._target_vol(profile):.0%})")
    if stock.get("max_dd") is not None:
        f.append(f"worst drawdown in the window: {stock['max_dd']:.0%}")
    if stock.get("dd_from_high") is not None:
        f.append(f"currently {abs(stock['dd_from_high']):.0%} below its high")
    for k, lab in (("mom_12m", "12-month price change"),
                   ("mom_3m", "3-month price change")):
        if stock.get(k) is not None:
            f.append(f"{lab}: {stock[k]:+.0%}")
    if stock.get("dividend_yield"):
        f.append(f"dividend yield: {stock['dividend_yield']:.1%}")
    if stock.get("pe"):
        f.append(f"price/earnings: {stock['pe']:.0f}")
    if stock.get("sector"):
        f.append(f"sector: {stock['sector']}")
    if stock.get("themes"):
        f.append(f"themes: {', '.join(stock['themes'])}")
    f.append(f"match score: {stock.get('fit', 0):.2f} of 1.00")
    parts = stock.get("fit_parts") or {}
    if parts:
        best = sorted(parts.items(), key=lambda kv: -kv[1])[:3]
        f.append("strongest match components: " +
                 ", ".join(f"{k} {v:.2f}" for k, v in best))
    return "\n".join(f)


def _fallback_copy(stock: dict, profile: dict) -> dict:
    """Reveal copy assembled from measured numbers, used when no model is
    configured. Deliberately plain — better a true sentence than a fake one."""
    name = stock.get("name") or stock["symbol"]
    why = []
    tv = E._target_vol(profile)
    if stock.get("ann_vol") is not None:
        why.append(f"Its {stock['ann_vol']:.0%} volatility sits close to the "
                   f"{tv:.0%} ride your answers asked for.")
    if stock.get("dividend_yield") and profile["dims"]["income"] > 0.5:
        why.append(f"It pays {stock['dividend_yield']:.1%}, and you said you "
                   f"want to be paid while you wait.")
    if stock.get("mom_12m") is not None:
        d = "climbed" if stock["mom_12m"] >= 0 else "fallen"
        why.append(f"It has {d} {abs(stock['mom_12m']):.0%} over the past year, "
                   f"which matches how you answered on chasing strength.")
    if stock.get("themes"):
        why.append(f"It sits in {', '.join(stock['themes'][:2])}, the area you "
                   f"kept picking.")
    dd = stock.get("max_dd")
    watch = (f"It has dropped {abs(dd):.0%} peak-to-trough before. Assume it can "
             f"do that again." if dd is not None else
             "Every individual stock can fall a long way. Size the position accordingly.")
    return {"headline": f"You matched with {name}",
            "why": why[:3] or [f"{name} scored highest against your answers."],
            "watch_out": watch, "source": "computed"}


def narrate(stock: dict, profile: dict, answers: list[dict]) -> dict:
    try:
        import _llm
        if not _llm.available():
            return _fallback_copy(stock, profile)
        user = (_profile_brief(profile, answers) +
                "\n\nThe engine chose this stock:\n" + _facts(stock, profile))
        raw, prov = _llm.complete_json(
            [{"role": "system", "content": NARRATE_SYSTEM},
             {"role": "user", "content": user}],
            task="smart", max_tokens=600, temperature=0.75)
        head = str(raw.get("headline") or "").strip()[:80]
        why = [str(w).strip()[:140] for w in (raw.get("why") or []) if str(w).strip()][:3]
        watch = str(raw.get("watch_out") or "").strip()[:170]
        if not head or not why:
            return _fallback_copy(stock, profile)
        return {"headline": head, "why": why,
                "watch_out": watch or _fallback_copy(stock, profile)["watch_out"],
                "source": prov}
    except Exception:  # noqa: BLE001
        return _fallback_copy(stock, profile)


def recommend(profile: dict, answers: list[dict]) -> dict:
    """Full pipeline. Always returns a dict; `ok` False when nothing could be
    verified, which the UI must show honestly rather than papering over."""
    cand = candidates(profile, answers)
    verified = M.quotes(cand["symbols"], rng="2y")
    stocks = []
    for sym, q in verified.items():
        if q.get("ok"):
            q = dict(q)
            if cand["themes"].get(sym):
                q["themes"] = cand["themes"][sym]
            stocks.append(q)

    ranked, rejected = E.rank(stocks, profile, cand["themes"], limit=6)
    diag = {"proposed": len(cand["symbols"]), "verified": len(stocks),
            "ranked": len(ranked), "source": cand["source"],
            "notes": cand["notes"],
            "throttled": sum(1 for q in verified.values()
                             if q.get("reason") in ("rate_limited", "unreachable"))}
    if not ranked:
        return {"ok": False, "reason": "no_verified_candidates",
                "rejected": rejected[:8], "diagnostics": diag}

    top = ranked[0]
    # US-only enrichment, best effort, never blocking
    try:
        extra = M.enrich(top["symbol"])
        for k, v in extra.items():
            top.setdefault(k, v)
        if extra:
            top["fit"] = E.fit(top, profile, top.get("themes"))["score"]
    except Exception:  # noqa: BLE001
        pass

    copy = narrate(top, profile, answers)
    return {"ok": True, "pick": top,
            "runners_up": [{"symbol": r["symbol"], "name": r.get("name"),
                            "fit": r["fit"], "price": r.get("price"),
                            "currency": r.get("currency"),
                            "ann_vol": r.get("ann_vol")} for r in ranked[1:4]],
            "copy": copy, "archetype": None, "diagnostics": diag,
            "rejected": rejected[:6]}
