"""HTTP surface for the stock-matching game.

  POST /api/quiz {"action":"next","answers":[...],"asked":[...]}
       -> the next question, plus how locked-in the profile now is
  POST /api/quiz {"action":"pick","answers":[...]}
       -> one verified stock, ranked by the engine, with reveal copy
  GET  /api/quiz?action=health
       -> which providers are configured (no secrets echoed)

Stateless by design: the client carries the answer history, so any lambda can
serve any turn and there is no session store to expire mid-game. Because the
client is therefore in a position to lie, every incoming effect is re-validated
and clamped here before it can touch the profile — the client can replay its
own game, but it cannot invent a preference the engine will not sanity-check.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _engine as E          # noqa: E402
import _questions as Q       # noqa: E402
import _recommend as R       # noqa: E402

MAX_ANSWERS = 40
MIN_QUESTIONS = 6
MAX_QUESTIONS = 14
CONFIDENCE_TARGET = 0.82


def _clean_effects(raw) -> dict:
    """Trust nothing from the wire. Unknown keys vanish, values are clamped."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if k in E.DIMS:
            try:
                out[k] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                continue
        elif k in ("themes", "avoid_themes") and isinstance(v, list):
            tags = [str(x).lower().strip() for x in v][:4]
            tags = [t for t in tags if t in set(E.THEMES)]
            if tags:
                out[k] = tags
        elif k == "note":
            out["note"] = str(v)[:120]
    try:
        out["weight"] = max(0.0, min(2.0, float(raw.get("weight", 1.0))))
    except (TypeError, ValueError):
        out["weight"] = 1.0
    return out


def _rebuild(answers: list) -> tuple[dict, list, list, list]:
    """Replay the client's history into a profile it cannot forge."""
    profile = E.new_profile()
    clean, asked_ids, asked_texts = [], [], []
    for a in answers[:MAX_ANSWERS]:
        if not isinstance(a, dict):
            continue
        eff = _clean_effects(a.get("effects"))
        if not eff:
            continue
        profile = E.apply_answer(profile, eff)
        qid = str(a.get("id") or "")[:64]
        qtext = str(a.get("q") or "")[:200]
        clean.append({"q": qtext, "a": str(a.get("a") or "")[:120], "id": qid})
        if qid:
            asked_ids.append(qid)
        if qtext:
            asked_texts.append(qtext)
    return profile, clean, asked_ids, asked_texts


def handle_next(body: dict) -> dict:
    profile, answers, asked_ids, asked_texts = _rebuild(body.get("answers") or [])
    conf = E.confidence(profile)
    n = len(answers)

    done = (n >= MAX_QUESTIONS) or (n >= MIN_QUESTIONS and conf >= CONFIDENCE_TARGET)
    if done:
        return {"done": True, "confidence": round(conf, 3), "asked": n,
                "archetype": Q.archetype(profile)}

    question, source = None, "bank"
    if os.environ.get("QUIZ_DISABLE_LLM") != "1":
        question, source = Q.llm_question(profile, asked_texts, asked_ids, answers)
        if question is None:
            source = f"bank ({source})"
    if question is None:
        question = Q.pick_from_bank(profile, asked_ids, seed=str(n))
        if question is None:                       # bank exhausted
            return {"done": True, "confidence": round(conf, 3), "asked": n,
                    "archetype": Q.archetype(profile)}
        question = {k: question[k] for k in
                    ("id", "text", "subtext", "options") if k in question}
        question["source"] = "bank"
    else:
        question["source"] = source

    # progress the UI can trust: whichever bound will actually end the game
    by_conf = conf / CONFIDENCE_TARGET if CONFIDENCE_TARGET else 1.0
    by_count = n / MAX_QUESTIONS
    return {"done": False, "question": question,
            "confidence": round(conf, 3),
            "progress": round(max(0.0, min(0.99, max(by_conf, by_count))), 3),
            "asked": n, "min_questions": MIN_QUESTIONS,
            "archetype_preview": Q.archetype(profile)}


def handle_pick(body: dict) -> dict:
    profile, answers, _, _ = _rebuild(body.get("answers") or [])
    if len(answers) < 3:
        return {"ok": False, "reason": "not_enough_answers"}
    result = R.recommend(profile, answers)
    result["archetype"] = Q.archetype(profile)
    result["confidence"] = round(E.confidence(profile), 3)
    result["profile"] = {"dims": {k: round(v, 3) for k, v in profile["dims"].items()},
                         "themes": profile["themes"],
                         "avoid": profile.get("avoid", []),
                         "target_vol": round(E._target_vol(profile), 4)}
    return result


def handle_health() -> dict:
    import _llm
    return {"ok": True,
            "providers": _llm.available(),
            "llm_disabled": os.environ.get("QUIZ_DISABLE_LLM") == "1",
            "bank_questions": len(Q.BANK),
            "dimensions": list(E.DIMS)}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        action = (q.get("action", ["health"])[0] or "health").lower()
        if action == "health":
            self._respond(200, handle_health())
        else:
            self._respond(400, {"error": "use POST for next/pick"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 200_000:
                self._respond(413, {"error": "payload too large"})
                return
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode() or "{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception as e:  # noqa: BLE001
            self._respond(400, {"error": f"bad request: {type(e).__name__}"})
            return

        action = str(body.get("action") or "next").lower()
        try:
            if action == "next":
                self._respond(200, handle_next(body))
            elif action == "pick":
                self._respond(200, handle_pick(body))
            elif action == "health":
                self._respond(200, handle_health())
            else:
                self._respond(400, {"error": f"unknown action '{action}'"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._respond(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)
