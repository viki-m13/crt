"""The recommendation engine: preference profile -> ranked real stocks.

Deliberately split in two so the interesting half can be tested without a
network or a language model:

  PROFILE   an 8-dimension vector built from quiz answers. Every answer
            carries its own weight, so a decisive answer moves a dimension
            further than a hedged one, and confidence per dimension is
            tracked explicitly — that is what tells the quiz when to stop.

  FIT       a deterministic score of one real stock against one profile,
            computed only from measured market data (realised volatility,
            drawdown, momentum, 52-week position, size, yield). No LLM
            opinion enters the number. An LLM may PROPOSE candidates and
            write the prose, but it cannot change the ranking.

The engine is intentionally opinionated about what disqualifies a stock: a
"safe" profile must never be handed a 90%-volatility microcap because the
prose sounded good. Hard gates run before scoring.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------
# Each dimension is a 0..1 latent. 0.5 = no information yet.
DIMS = (
    "risk",          # 0 capital preservation      -> 1 swing for the fences
    "horizon",       # 0 months                    -> 1 decades
    "volatility",    # 0 needs a smooth ride       -> 1 stomachs wild swings
    "value_growth",  # 0 cheap and unloved         -> 1 pay up for growth
    "momentum",      # 0 buy the dip / contrarian  -> 1 ride the winners
    "size",          # 0 small and obscure         -> 1 mega-cap household
    "income",        # 0 no dividends needed       -> 1 wants to be paid
    "conviction",    # 0 spread it around          -> 1 one big idea
)

# Themes are categorical, not a latent — a user can love two at once.
THEMES = ("ai", "semis", "software", "healthcare", "biotech", "energy",
          "renewables", "finance", "consumer", "luxury", "retail",
          "industrial", "defense", "space", "autos", "travel", "media",
          "gaming", "crypto", "materials", "realestate", "telecom",
          "utilities", "food", "ecommerce", "cloud", "robotics")


def new_profile() -> dict:
    return {
        "dims": {d: 0.5 for d in DIMS},
        # evidence mass per dimension; confidence is a function of this
        "mass": {d: 0.0 for d in DIMS},
        "themes": {},
        "avoid": [],
        "notes": [],
    }


def apply_answer(profile: dict, effects: dict) -> dict:
    """Fold one answer into the profile.

    `effects` looks like {"risk": 0.9, "horizon": 0.2, "themes": ["ai"],
    "weight": 1.0}. Dimension values are targets in 0..1; the profile moves
    toward each target by an amount that decays as evidence accumulates, so
    early answers shape the profile and later ones refine it.
    """
    p = {"dims": dict(profile["dims"]), "mass": dict(profile["mass"]),
         "themes": dict(profile["themes"]), "avoid": list(profile.get("avoid", [])),
         "notes": list(profile.get("notes", []))}
    w = float(effects.get("weight", 1.0))
    w = max(0.0, min(2.0, w))
    for d in DIMS:
        if d not in effects:
            continue
        try:
            target = float(effects[d])
        except (TypeError, ValueError):
            continue
        target = max(0.0, min(1.0, target))
        m = p["mass"][d]
        # learning rate falls as evidence accumulates: 1st answer moves a lot
        lr = w / (1.0 + m)
        lr = max(0.0, min(1.0, lr))
        p["dims"][d] += (target - p["dims"][d]) * lr
        p["mass"][d] = m + w
    for t in effects.get("themes", []) or []:
        t = str(t).lower().strip()
        if t:
            p["themes"][t] = p["themes"].get(t, 0.0) + w
    for t in effects.get("avoid_themes", []) or []:
        t = str(t).lower().strip()
        if t and t not in p["avoid"]:
            p["avoid"].append(t)
    if effects.get("note"):
        p["notes"].append(str(effects["note"])[:120])
    return p


def confidence(profile: dict) -> float:
    """0..1 overall readiness to recommend.

    Deliberately demanding on the dimensions that decide whether a
    recommendation is *safe* (risk, volatility, horizon) and relaxed about
    taste dimensions, plus a bonus once any theme is expressed. This is the
    number the game shows as a 'signal lock' meter, so it must move
    visibly with every answer but not reach 100% cheaply.
    """
    core = ("risk", "volatility", "horizon")
    taste = ("value_growth", "momentum", "size", "income", "conviction")
    # each unit of mass saturates; 2.5 units of evidence ~ 0.78 confident
    def sat(m):
        return 1.0 - math.exp(-m / 1.6)
    c_core = sum(sat(profile["mass"][d]) for d in core) / len(core)
    c_taste = sum(sat(profile["mass"][d]) for d in taste) / len(taste)
    c_theme = min(1.0, sum(profile["themes"].values()) / 2.0)
    return max(0.0, min(1.0, 0.55 * c_core + 0.30 * c_taste + 0.15 * c_theme))


def weakest_dimensions(profile: dict, k: int = 3) -> list[str]:
    """Which dimensions the next question should target (lowest evidence,
    weighted so the safety-critical ones get asked about first)."""
    prio = {"risk": 1.5, "volatility": 1.4, "horizon": 1.3, "value_growth": 1.0,
            "momentum": 0.9, "size": 0.9, "income": 0.8, "conviction": 0.7}
    order = sorted(DIMS, key=lambda d: profile["mass"][d] / prio.get(d, 1.0))
    return order[:k]


# --------------------------------------------------------------------------
# hard gates — run before any scoring
# --------------------------------------------------------------------------
def gate(stock: dict, profile: dict) -> str | None:
    """Return a rejection reason, or None if the stock is allowed.

    These are safety rails, not preferences: no amount of thematic
    enthusiasm should be able to hand a preservation-minded user a
    penny-stock-volatility name.
    """
    if not stock.get("ok"):
        return stock.get("reason", "unverified")
    if stock.get("bars", 0) < 200:
        return "too_new"
    price = stock.get("price")
    if not price or price <= 0:
        return "no_price"
    if stock.get("currency") == "USD" and price < 1.0:
        return "sub_dollar"
    vol = stock.get("ann_vol")
    if vol is None:
        return "no_vol"
    if vol > 2.5:
        return "extreme_vol"
    risk, vtol = profile["dims"]["risk"], profile["dims"]["volatility"]
    # the ceiling a user's stated risk appetite earns, in annualised vol
    ceiling = 0.22 + 0.78 * max(risk, vtol) ** 1.35
    if vol > ceiling * 1.9:
        return "too_volatile_for_profile"
    dd = stock.get("max_dd")
    if dd is not None and risk < 0.35 and dd < -0.75:
        return "drawdown_too_deep_for_profile"
    return None


# --------------------------------------------------------------------------
# fit scoring
# --------------------------------------------------------------------------
def _bell(x: float, target: float, width: float) -> float:
    """1.0 at target, decaying smoothly. Width is the 'half-comfort' span."""
    if width <= 0:
        return 0.0
    return math.exp(-0.5 * ((x - target) / width) ** 2)


def _target_vol(profile: dict) -> float:
    """Annualised volatility this profile is actually asking for."""
    risk = profile["dims"]["risk"]
    vtol = profile["dims"]["volatility"]
    blend = 0.45 * risk + 0.55 * vtol
    return 0.14 + 0.52 * blend ** 1.25          # ~14% timid -> ~66% aggressive


def fit(stock: dict, profile: dict, themes: list[str] | None = None) -> dict:
    """Score one verified stock against one profile. Returns components so
    the UI can explain the match rather than assert it."""
    d = profile["dims"]
    parts: dict[str, float] = {}

    vol = stock.get("ann_vol") or 0.0
    tvol = _target_vol(profile)
    # asymmetric: overshooting the user's risk budget is worse than undershooting
    width = tvol * (0.55 if vol > tvol else 0.75)
    parts["volatility"] = _bell(vol, tvol, max(width, 0.05))

    dd = stock.get("max_dd")
    if dd is not None:
        tolerable = -(0.12 + 0.63 * max(d["risk"], d["volatility"]) ** 1.2)
        parts["drawdown"] = 1.0 if dd >= tolerable else \
            max(0.0, 1.0 - (tolerable - dd) / 0.45)

    m12, m3 = stock.get("mom_12m"), stock.get("mom_3m")
    if m12 is not None:
        # momentum lovers want strength; contrarians want the beaten-down
        want = d["momentum"]
        strength = max(0.0, min(1.0, (m12 + 0.35) / 1.15))
        parts["momentum"] = 1.0 - abs(strength - want)
    pos = stock.get("pos_52w")
    if pos is not None:
        parts["position"] = 1.0 - abs(pos - (0.25 + 0.6 * d["momentum"]))

    mcap = stock.get("market_cap")
    if mcap:
        # log10 size mapped to 0..1 across ~$300M .. ~$3T
        s = (math.log10(max(mcap, 1e7)) - 8.5) / 4.0
        parts["size"] = 1.0 - abs(max(0.0, min(1.0, s)) - d["size"])

    dy = stock.get("dividend_yield")
    if dy is not None:
        want_y = 0.005 + 0.045 * d["income"]
        parts["income"] = _bell(dy, want_y, 0.022) if d["income"] > 0.35 else \
            (1.0 if dy <= 0.03 else 0.75)

    pe = stock.get("pe")
    if pe and pe > 0:
        # value_growth 0 -> reward low PE; 1 -> tolerate high PE
        cheapness = max(0.0, min(1.0, 1.0 - (math.log10(max(pe, 1.0)) - 0.7) / 1.1))
        parts["valuation"] = 1.0 - abs(cheapness - (1.0 - d["value_growth"]))

    # thematic affinity, from tags supplied alongside the candidate
    tags = [str(t).lower() for t in (themes or stock.get("themes") or [])]
    if profile["themes"]:
        total = sum(profile["themes"].values()) or 1.0
        hit = sum(v for k, v in profile["themes"].items() if k in tags)
        parts["theme"] = min(1.0, hit / total * 1.35)
    if any(t in tags for t in profile.get("avoid", [])):
        parts["theme"] = 0.0

    # quality/consistency of the ride, which everyone benefits from
    sh = stock.get("sharpe_window")
    if sh is not None:
        parts["consistency"] = max(0.0, min(1.0, (sh + 0.5) / 2.0))

    weights = {
        "volatility": 2.4 + 1.2 * (1.0 - d["risk"]),   # matters most to the timid
        "drawdown": 1.9 + 1.0 * (1.0 - d["risk"]),
        "momentum": 1.2,
        "position": 0.7,
        "size": 1.3,
        "income": 1.1 + 1.4 * d["income"],
        "valuation": 1.0 + 0.8 * (1.0 - d["value_growth"]),
        "theme": 2.2,
        "consistency": 1.0 + 0.8 * (1.0 - d["risk"]),
    }
    num = sum(parts[k] * weights.get(k, 1.0) for k in parts)
    den = sum(weights.get(k, 1.0) for k in parts)
    score = (num / den) if den else 0.0
    return {"score": round(score, 4), "parts": {k: round(v, 3) for k, v in parts.items()},
            "target_vol": round(tvol, 4)}


def rank(stocks: list[dict], profile: dict, themes_by_symbol: dict | None = None,
         limit: int = 10) -> tuple[list[dict], list[dict]]:
    """Rank verified candidates. Returns (ranked, rejected)."""
    themes_by_symbol = themes_by_symbol or {}
    ranked, rejected = [], []
    for s in stocks:
        reason = gate(s, profile)
        if reason:
            rejected.append({"symbol": s.get("symbol"), "reason": reason,
                             "name": s.get("name")})
            continue
        f = fit(s, profile, themes_by_symbol.get(s.get("symbol")))
        row = dict(s)
        row["fit"] = f["score"]
        row["fit_parts"] = f["parts"]
        row["target_vol"] = f["target_vol"]
        ranked.append(row)
    ranked.sort(key=lambda r: r["fit"], reverse=True)
    return ranked[:limit], rejected
