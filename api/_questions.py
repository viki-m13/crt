"""The question layer: what the game actually asks, and how it adapts.

Two sources, one contract. Every question — hand-written or model-generated —
is a dict of {id, text, options[]} where each option carries `effects` that
the engine folds into the profile. Because the contract is identical, the
game plays the same with no API key at all; the LLM adds novelty and
personalisation on top of a bank that already covers every dimension.

Design rule for questions: never ask a survey question. "Rate your risk
tolerance 1-10" is worthless — people answer aspirationally. Ask about a
concrete moment where the answer reveals behaviour instead of self-image.
"""
from __future__ import annotations

import hashlib
import json
import random

from _engine import DIMS, THEMES, weakest_dimensions

# --------------------------------------------------------------------------
# the hand-written bank. `targets` lets the selector find questions that
# resolve the dimensions still in doubt.
# --------------------------------------------------------------------------
BANK: list[dict] = [
    {
        "id": "crash_6am",
        "targets": ["risk", "conviction", "momentum"],
        "text": "It's 6am. Your biggest holding is down 22% overnight. No news, no explanation.",
        "subtext": "Be honest about what you'd actually do.",
        "options": [
            {"label": "Buy more. Same company, cheaper price.", "emoji": "🛒",
             "effects": {"risk": 0.85, "momentum": 0.15, "conviction": 0.8, "weight": 1.2}},
            {"label": "Hold and stop looking at my phone.", "emoji": "🫥",
             "effects": {"risk": 0.6, "horizon": 0.8, "conviction": 0.55, "weight": 1.0}},
            {"label": "Sell half. Sleep matters.", "emoji": "😰",
             "effects": {"risk": 0.28, "volatility": 0.25, "conviction": 0.35, "weight": 1.1}},
            {"label": "Sell it all. I'm out.", "emoji": "🚪",
             "effects": {"risk": 0.06, "volatility": 0.08, "horizon": 0.3, "weight": 1.3}},
        ],
    },
    {
        "id": "two_doors",
        "targets": ["risk", "volatility"],
        "text": "Two doors. Behind each is a real ten-year outcome.",
        "subtext": "You can only walk through one.",
        "options": [
            {"label": "Door A: 8%/yr, almost never a scary year", "emoji": "🚪",
             "effects": {"risk": 0.15, "volatility": 0.15, "weight": 1.3}},
            {"label": "Door B: 14%/yr, but two years where you lose a third", "emoji": "🎢",
             "effects": {"risk": 0.62, "volatility": 0.65, "weight": 1.2}},
            {"label": "Door C: 25%/yr, and one year you lose two-thirds", "emoji": "🌋",
             "effects": {"risk": 0.95, "volatility": 0.95, "weight": 1.3}},
        ],
    },
    {
        "id": "when_money",
        "targets": ["horizon", "income"],
        "text": "When do you actually want to touch this money?",
        "options": [
            {"label": "Within a year — I have plans for it", "emoji": "⏱️",
             "effects": {"horizon": 0.1, "risk": 0.35, "weight": 1.3}},
            {"label": "Three to five years", "emoji": "📅",
             "effects": {"horizon": 0.4, "weight": 1.1}},
            {"label": "A decade, minimum", "emoji": "🌳",
             "effects": {"horizon": 0.85, "weight": 1.2}},
            {"label": "Never. This is the money that outlives me.", "emoji": "🏛️",
             "effects": {"horizon": 1.0, "income": 0.7, "weight": 1.3}},
        ],
    },
    {
        "id": "friend_3x",
        "targets": ["risk", "conviction", "momentum"],
        "text": "A friend just tripled their money on something you passed on.",
        "subtext": "The feeling that hits you first:",
        "options": [
            {"label": "Genuine happiness. Different games.", "emoji": "😌",
             "effects": {"risk": 0.3, "conviction": 0.4, "weight": 0.9}},
            {"label": "What are they buying next?", "emoji": "👀",
             "effects": {"risk": 0.8, "momentum": 0.85, "weight": 1.1}},
            {"label": "It's probably about to crash.", "emoji": "🧐",
             "effects": {"momentum": 0.15, "value_growth": 0.2, "weight": 1.0}},
            {"label": "Physical pain.", "emoji": "🫠",
             "effects": {"risk": 0.72, "momentum": 0.75, "conviction": 0.7, "weight": 1.0}},
        ],
    },
    {
        "id": "company_type",
        "targets": ["value_growth", "size", "momentum"],
        "text": "Which of these would you rather own a piece of?",
        "options": [
            {"label": "A boring business that's printed cash for 60 years", "emoji": "🏭",
             "effects": {"value_growth": 0.15, "size": 0.85, "income": 0.7, "weight": 1.2}},
            {"label": "The company everyone already knows is winning", "emoji": "👑",
             "effects": {"value_growth": 0.75, "size": 0.9, "momentum": 0.8, "weight": 1.1}},
            {"label": "A good company having a genuinely bad year", "emoji": "🩹",
             "effects": {"value_growth": 0.2, "momentum": 0.12, "risk": 0.6, "weight": 1.2}},
            {"label": "Something most people haven't heard of yet", "emoji": "🔍",
             "effects": {"size": 0.15, "risk": 0.8, "value_growth": 0.7, "weight": 1.2}},
        ],
    },
    {
        "id": "portfolio_shape",
        "targets": ["conviction", "risk"],
        "text": "You have $10,000 to put to work today.",
        "options": [
            {"label": "All of it into one conviction", "emoji": "🎯",
             "effects": {"conviction": 0.95, "risk": 0.8, "weight": 1.2}},
            {"label": "Split across three or four ideas", "emoji": "🍰",
             "effects": {"conviction": 0.5, "risk": 0.5, "weight": 1.0}},
            {"label": "Spread thin across a dozen names", "emoji": "🌾",
             "effects": {"conviction": 0.15, "risk": 0.32, "weight": 1.1}},
        ],
    },
    {
        "id": "world_2035",
        "targets": ["value_growth"],
        "text": "Which of these is most obviously bigger in 2035?",
        "subtext": "Go with your gut.",
        "options": [
            {"label": "Machines that think", "emoji": "🤖",
             "effects": {"themes": ["ai", "semis", "software"], "value_growth": 0.8,
                         "weight": 1.0}},
            {"label": "Keeping people alive longer", "emoji": "🧬",
             "effects": {"themes": ["healthcare", "biotech"], "weight": 1.0}},
            {"label": "Powering all of it", "emoji": "⚡",
             "effects": {"themes": ["energy", "renewables", "utilities"], "weight": 1.0}},
            {"label": "The stuff people buy no matter what", "emoji": "🛒",
             "effects": {"themes": ["consumer", "food", "retail"],
                         "value_growth": 0.3, "weight": 1.0}},
        ],
    },
    {
        "id": "brand_pull",
        "targets": ["size"],
        "text": "Pick the world you find most interesting.",
        "options": [
            {"label": "Chips, cloud and code", "emoji": "💻",
             "effects": {"themes": ["semis", "cloud", "software"], "weight": 1.0}},
            {"label": "Banks, insurers and money itself", "emoji": "🏦",
             "effects": {"themes": ["finance"], "value_growth": 0.3, "weight": 1.0}},
            {"label": "Cars, rockets and robots", "emoji": "🚀",
             "effects": {"themes": ["autos", "space", "robotics", "defense"],
                         "risk": 0.7, "weight": 1.0}},
            {"label": "Handbags, hotels and good coffee", "emoji": "☕",
             "effects": {"themes": ["luxury", "travel", "consumer"], "weight": 1.0}},
        ],
    },
    {
        "id": "dividend_or_not",
        "targets": ["income", "horizon"],
        "text": "A company can either pay you $3 a year, or reinvest it to grow faster.",
        "options": [
            {"label": "Pay me. I like getting paid.", "emoji": "💵",
             "effects": {"income": 0.95, "value_growth": 0.25, "risk": 0.3, "weight": 1.3}},
            {"label": "Reinvest all of it.", "emoji": "🌱",
             "effects": {"income": 0.05, "value_growth": 0.8, "horizon": 0.8, "weight": 1.3}},
            {"label": "A bit of both is fine.", "emoji": "⚖️",
             "effects": {"income": 0.5, "weight": 0.9}},
        ],
    },
    {
        "id": "check_frequency",
        "targets": ["volatility", "horizon"],
        "text": "How often would you check the price?",
        "options": [
            {"label": "Several times a day, let's be real", "emoji": "📱",
             "effects": {"volatility": 0.6, "horizon": 0.25, "momentum": 0.7, "weight": 1.0}},
            {"label": "Once a week or so", "emoji": "📊",
             "effects": {"volatility": 0.5, "horizon": 0.6, "weight": 0.8}},
            {"label": "I'd forget I owned it", "emoji": "🧘",
             "effects": {"volatility": 0.35, "horizon": 0.9, "weight": 1.1}},
        ],
    },
    {
        "id": "red_year",
        "targets": ["volatility", "risk"],
        "text": "Your account is down 35%. It's been eleven months.",
        "subtext": "Your honest reaction:",
        "options": [
            {"label": "This is the opportunity I've been waiting for", "emoji": "🔥",
             "effects": {"risk": 0.92, "volatility": 0.9, "momentum": 0.2, "weight": 1.3}},
            {"label": "Uncomfortable, but it's the price of admission", "emoji": "😤",
             "effects": {"risk": 0.65, "volatility": 0.7, "weight": 1.1}},
            {"label": "I would not have signed up for this", "emoji": "😖",
             "effects": {"risk": 0.12, "volatility": 0.12, "weight": 1.4}},
        ],
    },
    {
        "id": "story_or_numbers",
        "targets": ["value_growth", "momentum"],
        "text": "What convinces you to buy something?",
        "options": [
            {"label": "A story I believe in", "emoji": "📖",
             "effects": {"value_growth": 0.8, "risk": 0.7, "weight": 1.0}},
            {"label": "Numbers that look too cheap", "emoji": "🧮",
             "effects": {"value_growth": 0.12, "momentum": 0.2, "weight": 1.1}},
            {"label": "It keeps going up and I stopped arguing", "emoji": "📈",
             "effects": {"momentum": 0.92, "value_growth": 0.7, "weight": 1.1}},
            {"label": "Everyone I know already uses the product", "emoji": "🧾",
             "effects": {"size": 0.8, "value_growth": 0.6, "weight": 1.0}},
        ],
    },
    {
        "id": "home_or_world",
        "targets": ["size"],
        "text": "Where should your money live?",
        "options": [
            {"label": "American giants, keep it simple", "emoji": "🇺🇸",
             "effects": {"size": 0.85, "weight": 0.9}},
            {"label": "Anywhere on earth, I don't care about the flag", "emoji": "🌍",
             "effects": {"size": 0.5, "risk": 0.6, "note": "open to global listings",
                         "weight": 0.9}},
            {"label": "Somewhere unloved and cheap", "emoji": "🗺️",
             "effects": {"value_growth": 0.1, "risk": 0.7, "size": 0.35, "weight": 1.0}},
        ],
    },
    {
        "id": "nightmare",
        "targets": ["risk", "conviction"],
        "text": "Which outcome would haunt you more?",
        "options": [
            {"label": "Losing half of what I put in", "emoji": "💀",
             "effects": {"risk": 0.15, "volatility": 0.18, "weight": 1.3}},
            {"label": "Missing the thing that went up 10x", "emoji": "🚀",
             "effects": {"risk": 0.88, "volatility": 0.85, "conviction": 0.8, "weight": 1.3}},
        ],
    },
    {
        # The bank was pushing moderate players to extremes: every momentum
        # option was either 0.12 or 0.92, so someone who is genuinely in the
        # middle had to overstate. These next questions exist to let a
        # moderate answer BE moderate.
        "id": "winner_or_laggard",
        "targets": ["momentum", "value_growth"],
        "text": "Same industry, two companies. One's up 60% this year, one's flat.",
        "options": [
            {"label": "The winner. It's winning for a reason.", "emoji": "🏆",
             "effects": {"momentum": 0.88, "value_growth": 0.75, "weight": 1.1}},
            {"label": "The flat one, if the business is still sound.", "emoji": "😐",
             "effects": {"momentum": 0.45, "value_growth": 0.45, "weight": 1.0}},
            {"label": "The laggard. That's where the value is.", "emoji": "🔻",
             "effects": {"momentum": 0.10, "value_growth": 0.2, "weight": 1.1}},
        ],
    },
    {
        "id": "windfall",
        "targets": ["risk", "horizon", "conviction"],
        "text": "An unexpected $25,000 lands in your account.",
        "options": [
            {"label": "Straight into the market, all at once", "emoji": "⚡",
             "effects": {"risk": 0.75, "conviction": 0.75, "weight": 1.0}},
            {"label": "Drip it in over a year", "emoji": "💧",
             "effects": {"risk": 0.45, "volatility": 0.42, "horizon": 0.7,
                         "conviction": 0.35, "weight": 1.0}},
            {"label": "Half invested, half in cash for now", "emoji": "⚖️",
             "effects": {"risk": 0.4, "volatility": 0.4, "conviction": 0.4,
                         "weight": 0.9}},
            {"label": "Pay things off first. Invest what's left.", "emoji": "🧾",
             "effects": {"risk": 0.2, "volatility": 0.25, "weight": 1.0}},
        ],
    },
    {
        "id": "steady_or_spicy",
        "targets": ["volatility", "income", "risk"],
        "text": "Describe the ride you're signing up for.",
        "options": [
            {"label": "Smooth. I want to barely notice it.", "emoji": "🛋️",
             "effects": {"volatility": 0.12, "risk": 0.18, "income": 0.65, "weight": 1.2}},
            {"label": "Some bumps, nothing dramatic.", "emoji": "🚗",
             "effects": {"volatility": 0.42, "risk": 0.45, "income": 0.45, "weight": 1.1}},
            {"label": "Rough is fine if it's going somewhere.", "emoji": "🛻",
             "effects": {"volatility": 0.72, "risk": 0.72, "income": 0.15, "weight": 1.1}},
            {"label": "Strap me in.", "emoji": "🎢",
             "effects": {"volatility": 0.96, "risk": 0.94, "income": 0.03, "weight": 1.2}},
        ],
    },
    {
        "id": "experience",
        "targets": ["conviction", "size", "risk"],
        "text": "How long have you been doing this?",
        "options": [
            {"label": "This is basically my first go", "emoji": "🐣",
             "effects": {"risk": 0.35, "size": 0.82, "conviction": 0.3, "weight": 1.0}},
            {"label": "A few years. I've seen one bad stretch.", "emoji": "📚",
             "effects": {"risk": 0.55, "size": 0.6, "conviction": 0.5, "weight": 0.9}},
            {"label": "Long enough to have lost real money", "emoji": "🪦",
             "effects": {"risk": 0.62, "conviction": 0.68, "size": 0.45, "weight": 1.0}},
        ],
    },
    {
        "id": "vice_or_virtue",
        "targets": [],
        "text": "Anything you flatly refuse to own?",
        "options": [
            {"label": "Crypto-adjacent anything", "emoji": "🪙",
             "effects": {"avoid_themes": ["crypto"], "weight": 0.6}},
            {"label": "Weapons makers", "emoji": "🛡️",
             "effects": {"avoid_themes": ["defense"], "weight": 0.6}},
            {"label": "Oil and gas", "emoji": "🛢️",
             "effects": {"avoid_themes": ["energy"], "weight": 0.6}},
            {"label": "Nothing's off the table", "emoji": "🤷",
             "effects": {"risk": 0.62, "weight": 0.5}},
        ],
    },
]

_BY_ID = {q["id"]: q for q in BANK}


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def pick_from_bank(profile: dict, asked: list[str], seed: str = "") -> dict | None:
    """Choose the bank question that best resolves what we still don't know."""
    weak = weakest_dimensions(profile, 4)
    weight = {d: (len(weak) - i) for i, d in enumerate(weak)}
    best, best_score = [], -1.0
    for q in BANK:
        if q["id"] in asked:
            continue
        score = sum(weight.get(t, 0) for t in q.get("targets", []))
        # theme questions stay useful even once dimensions are settled
        if not q.get("targets"):
            score += 1.0
        if not profile["themes"] and any(
                "themes" in o.get("effects", {}) for o in q["options"]):
            score += 2.5
        if score > best_score:
            best, best_score = [q], score
        elif score == best_score:
            best.append(q)
    if not best:
        return None
    rng = random.Random(seed + str(len(asked)))
    return rng.choice(best)


# --------------------------------------------------------------------------
# validation of model-generated questions — never trust raw output into the UI
# --------------------------------------------------------------------------
_VALID_DIMS = set(DIMS)
_VALID_THEMES = set(THEMES)


def _is_pictograph(ch: str) -> bool:
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF
            or o in (0xFE0F, 0x200D, 0x20E3) or 0x1F1E6 <= o <= 0x1F1FF)


def _strip_edge_emoji(s: str) -> str:
    """Remove emoji and stray whitespace from both ends of a label."""
    chars = list(s)
    while chars and (_is_pictograph(chars[0]) or chars[0].isspace()):
        chars.pop(0)
    while chars and (_is_pictograph(chars[-1]) or chars[-1].isspace()):
        chars.pop()
    return "".join(chars).strip()


def validate_question(raw: dict, asked: list[str]) -> dict | None:
    """Coerce a model's question into the contract, or reject it."""
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not (8 <= len(text) <= 180):
        return None
    opts_in = raw.get("options")
    if not isinstance(opts_in, list) or not (2 <= len(opts_in) <= 4):
        return None
    options = []
    for o in opts_in:
        if not isinstance(o, dict):
            return None
        # Models reliably repeat the emoji inside the label as well as in the
        # emoji field ("ROCKET All in on biotech ROCKET"), which renders twice
        # in the card. Strip pictographs from both ends of the label.
        label = _strip_edge_emoji(str(o.get("label") or "").strip())
        # 64 chars is what fits an option card on a 360px phone at 16px
        # without wrapping to three lines; longer labels are rejected rather
        # than truncated mid-word.
        if not (1 <= len(label) <= 64):
            return None
        eff_in = o.get("effects") or {}
        if not isinstance(eff_in, dict):
            return None
        eff: dict = {}
        for k, v in eff_in.items():
            if k in _VALID_DIMS:
                try:
                    eff[k] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    continue
            elif k in ("themes", "avoid_themes") and isinstance(v, list):
                tags = [str(x).lower().strip() for x in v]
                tags = [t for t in tags if t in _VALID_THEMES]
                if tags:
                    eff[k] = tags[:4]
        if not eff:
            return None
        eff["weight"] = 1.0
        emoji = str(o.get("emoji") or "").strip()[:4]
        options.append({"label": label, "emoji": emoji, "effects": eff})
    # an option set that says the same thing twice teaches the engine nothing
    if len({o["label"].lower() for o in options}) != len(options):
        return None
    qid = "llm_" + hashlib.sha1(text.encode()).hexdigest()[:8]
    if qid in asked:
        return None
    sub = str(raw.get("subtext") or "").strip()
    return {"id": qid, "text": text, "subtext": sub[:120] if sub else "",
            "options": options, "source": "llm"}


SYSTEM = """You write questions for a fast, addictive stock-matching game.

Your question must reveal how someone ACTUALLY behaves with money. Never ask \
them to rate themselves; put them in a concrete moment and make every option \
tempting to somebody. Be vivid, specific and a little playful.

LENGTH IS CRITICAL — this is read on a phone:
- question: under 100 characters, one sentence
- each option label: under 45 characters
Long options wrap badly and lose the player. Cut every wasted word.

VARIETY IS CRITICAL. Do not reuse the SHAPE of an earlier question, not just \
its words. "A sum of money arrives, where do you put it?" is one shape — used \
once, it is finished. Same for "someone tips you about a small company".

Rotate through genuinely different situations: a position already losing \
money, a decision someone else made that you have to react to, a choice \
between two companies you can see, how you'd feel a year from now, what you'd \
tell a friend, what you would refuse to own, how you behaved last time, a \
trade-off with no money in it at all. Vary the industry too.

Return ONLY a JSON object:
{"text": "the question", "subtext": "optional short line or empty",
 "options": [{"label": "...", "emoji": "X", "effects": {...}}, ...]}

2-4 options. Each option's `effects` maps dimensions to the value that answer \
implies, 0..1:
  risk 0=preserve capital 1=swing for the fences
  horizon 0=months 1=decades
  volatility 0=needs a smooth ride 1=stomachs wild swings
  value_growth 0=cheap and unloved 1=pays up for growth
  momentum 0=contrarian 1=rides winners
  size 0=small and obscure 1=mega-cap household name
  income 0=no dividends needed 1=wants to be paid
  conviction 0=spread it around 1=one big idea
Optionally "themes": [...] or "avoid_themes": [...] from this list:
""" + ", ".join(THEMES) + """

Set only the 1-3 dimensions the answer genuinely reveals.

Use the FULL range, not just 0 and 1. A hedged or middle-of-the-road option \
must carry middle values like 0.45 — forcing every option to an extreme makes \
moderate players look extreme and produces a wrong recommendation.

Do not restate a question already asked. No preamble, no code fences, JSON only."""


def llm_question(profile: dict, asked_texts: list[str], asked_ids: list[str],
                 answers: list[dict]):
    """Ask the fast model for a fresh question. Returns (question, provider)
    or (None, error-string) — callers fall back to the bank."""
    import _llm

    focus = weakest_dimensions(profile, 3)
    story = []
    for a in answers[-6:]:
        story.append(f"Q: {a.get('q', '')[:90]} -> chose: {a.get('a', '')[:70]}")
    themes = ", ".join(sorted(profile["themes"])) or "none yet"
    user = (
        f"Answers so far:\n" + ("\n".join(story) or "(none yet — this is question 1)") +
        f"\n\nThemes they've leaned toward: {themes}"
        f"\nDimensions still unknown, ask about these: {', '.join(focus)}"
        f"\nAlready used — do not repeat any of these SHAPES or subjects:\n"
        + ("\n".join(f"  - {t[:80]}" for t in asked_texts[-10:]) or "  (none yet)")
        + f"\n\nWrite question #{len(answers) + 1}.")
    try:
        raw, prov = _llm.complete_json(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user}],
            task="fast", max_tokens=600, temperature=0.95)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"
    q = validate_question(raw, asked_ids)
    if not q:
        return None, "validation_failed"
    return q, prov


# --------------------------------------------------------------------------
# archetype — the shareable identity revealed with the pick
# --------------------------------------------------------------------------
# Each archetype is a PROTOTYPE profile plus the dimensions that define it.
# Nearest-prototype rather than first-match-wins: an ordered rule list
# mislabels people whose profile satisfies two rules at once (a decades-long
# compounder was coming back as "The Surfer" purely because momentum was
# tested later in the list).
ARCHETYPES = [
    # income is named here on purpose: without it, this prototype wins against
    # The Landlord for income-hungry players simply by not being measured on
    # the dimension that defines them.
    ("The Vault", "Capital comes home intact. That's the whole job.",
     {"risk": 0.08, "volatility": 0.10, "horizon": 0.55, "income": 0.45}),
    ("The Landlord", "You want the asset to pay rent while you sleep.",
     {"income": 0.92, "risk": 0.30, "volatility": 0.30}),
    ("The Compounder", "Boring, relentless, decades long. The cheat code.",
     {"horizon": 0.92, "risk": 0.50, "volatility": 0.45, "momentum": 0.5}),
    ("The Bargain Hunter", "You want it broken, cheap, and fixable.",
     {"value_growth": 0.10, "momentum": 0.12, "risk": 0.60}),
    ("The Surfer", "You don't argue with a wave. You ride it.",
     {"momentum": 0.92, "risk": 0.70, "volatility": 0.70, "horizon": 0.35}),
    ("The Sniper", "One target. Full conviction. No hedging.",
     {"conviction": 0.93, "risk": 0.75, "volatility": 0.7}),
    ("The Moonshot", "You'd rather miss rent than miss the 10x.",
     {"risk": 0.97, "volatility": 0.93, "horizon": 0.30, "income": 0.05}),
    ("The Explorer", "The good stuff is where nobody's looking.",
     {"size": 0.10, "risk": 0.70, "value_growth": 0.55}),
    ("The Balancer", "Enough risk to matter, enough sense to sleep.",
     {d: 0.5 for d in DIMS}),
]


def archetype(profile: dict) -> dict:
    """Nearest prototype, measured only on the dimensions that define each
    archetype, normalised so a 3-dimension archetype is not penalised
    against an 8-dimension one."""
    d = profile["dims"]
    best, best_dist = None, 1e9
    for name, blurb, proto in ARCHETYPES:
        dist = sum(abs(d.get(k, 0.5) - v) for k, v in proto.items()) / len(proto)
        # the catch-all should only win when nothing else is close
        if name == "The Balancer":
            dist += 0.06
        if dist < best_dist:
            best, best_dist = (name, blurb), dist
    return {"name": best[0], "blurb": best[1], "distance": round(best_dist, 3)}
