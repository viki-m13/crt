"""HTTP surface for the ISO / AMT planner.

  POST /api/iso {"shares":10000,"strike":2,"fmv":52,"income":200000,...}
       -> the crossover, the cost of exercising everything, a ladder, and a
          plain-language explanation
  GET  /api/iso?action=constants
       -> the tax constants in force, with their source, so a user can check
          them against the IRS rather than trust us

Every number is computed in _tax.py. The model is handed those numbers and
allowed only to explain them — measured head-to-head, that division is not
cosmetic: given the same figures, a fast model wrote "you keep the full
spread as a tax-free gain if you hold past the required period", which is
wrong (a qualifying disposition is taxed as long-term capital gain, not
exempt). Prose is generated; arithmetic never is.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _tax as T  # noqa: E402

MAX_SHARES = 50_000_000
MAX_MONEY = 1e9

EXPLAIN_SYSTEM = """You explain an ISO exercise decision to a smart person who \
is not a tax professional.

Every figure has ALREADY been computed and is given to you. Use only those \
figures and arithmetic on them. Never introduce a tax rate, threshold or \
dollar amount that is not in the input, and never estimate one.

Be accurate about the mechanics: exercising an ISO and holding creates an AMT \
preference item, not ordinary income. Selling after the qualifying holding \
period (2 years from grant, 1 year from exercise) means the gain is taxed as \
LONG-TERM CAPITAL GAIN — it is never tax-free. AMT paid on an ISO generally \
creates a credit usable against regular tax in later years.

Three short paragraphs, second person, plain language:
  1. what they can do this year without owing AMT, and what it costs
  2. what exercising everything at once would cost, and the trade-off
  3. what this calculation leaves out (state tax, AMT credit carryforward) \
and which direction each would move the answer

Write PLAIN TEXT only. No markdown, no asterisks, no bold, no headings, no \
bullet points, and no advice to consult a professional — the interface \
handles all of that."""


def _num(v, default=0.0, lo=0.0, hi=MAX_MONEY):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return max(lo, min(hi, x))


def _facts(p: dict, body: dict) -> str:
    status = "married filing jointly" if body["status"] == "mfj" else "single"
    ex = p["exercise_all"]
    lines = [
        f"Tax year {T.TAX_YEAR}, federal only. Filing status: {status}.",
        f"Ordinary income: ${body['income']:,.0f}.",
        f"ISO grant: {p['shares_total']:,} options at ${body['strike']:,.2f} "
        f"strike, current fair market value ${body['fmv']:,.2f}, "
        f"spread ${p['spread_per_share']:,.2f} per share.",
        f"Regular federal tax before any exercise: ${p['baseline_tax']:,.0f}.",
        f"Shares exercisable this year with ZERO AMT: {p['amt_free_shares']:,}.",
        f"Cash cost to exercise those: ${p['amt_free_exercise_cost']:,.0f}.",
        f"Bargain element at that point: ${p['amt_free_bargain_element']:,.0f}.",
        f"If ALL {p['shares_total']:,} are exercised and held: cash cost "
        f"${ex['exercise_cost']:,.0f}, AMT owed ${ex['amt_owed']:,.0f}, "
        f"total federal tax ${ex['total_federal_tax']:,.0f}.",
        f"Cost of the next dollar of bargain element at that point: "
        f"{ex['marginal_amt_rate']:.0%}.",
    ]
    if p["years_to_exercise_all_amt_free"]:
        lines.append(f"Years to exercise the whole grant AMT-free at this "
                     f"income: {p['years_to_exercise_all_amt_free']}.")
    return "\n".join(lines)


def explain(p: dict, body: dict) -> dict:
    """Prose for the result. Absent a model, say nothing rather than guess."""
    try:
        import _llm
        if not _llm.available():
            return {"text": None, "source": "none"}
        text, prov = _llm.complete(
            [{"role": "system", "content": EXPLAIN_SYSTEM},
             {"role": "user", "content": _facts(p, body)}],
            task="smart", max_tokens=700, temperature=0.4)
        return {"text": text.strip()[:2600], "source": prov}
    except Exception:  # noqa: BLE001
        return {"text": None, "source": "unavailable"}


def handle_calc(body: dict) -> dict:
    status = str(body.get("status") or "single").lower()
    if status not in T.FILING_STATUSES:
        status = "single"
    shares = int(_num(body.get("shares"), 0, 0, MAX_SHARES))
    strike = _num(body.get("strike"), 0.0, 0.0, 1e6)
    fmv = _num(body.get("fmv"), 0.0, 0.0, 1e6)
    income = _num(body.get("income"), 0.0)
    ltcg = _num(body.get("ltcg"), 0.0)
    itemized = bool(body.get("itemized"))
    ded = body.get("deduction")
    deduction = _num(ded, T.STANDARD_DEDUCTION[status]) if ded not in (None, "") \
        else T.STANDARD_DEDUCTION[status]

    if shares <= 0 or fmv <= 0:
        return {"ok": False, "reason": "need a share count and a current price"}
    if fmv <= strike:
        return {"ok": False, "reason": "underwater",
                "message": "The current price is at or below your strike, so "
                           "there is no bargain element and no AMT to plan "
                           "around."}

    kw = dict(status=status, ltcg=ltcg, deduction=deduction, itemized=itemized)
    p = T.plan(shares, strike, fmv, income, **kw)
    rows = T.ladder(shares, strike, fmv, income, points=8, **kw)
    echo = {"shares": shares, "strike": strike, "fmv": fmv, "income": income,
            "status": status, "ltcg": ltcg, "deduction": deduction,
            "itemized": itemized}

    return {"ok": True, "inputs": echo, "plan": p, "ladder": rows,
            "explanation": explain(p, echo),
            "constants": handle_constants(),
            "assumptions": [
                "Federal tax only — no state income tax or state AMT. "
                "California in particular has its own AMT and would raise "
                "this bill materially.",
                "No AMT credit carryforward from prior years, and no credit "
                "for the AMT paid here (which usually recovers some of it in "
                "later years).",
                "Assumes shares are exercised and HELD past year end. Selling "
                "in the same calendar year is a disqualifying disposition and "
                "is taxed differently.",
                "No Net Investment Income Tax, QSBS, 83(b) election, or other "
                "AMT preference items.",
            ]}


def handle_constants() -> dict:
    return {"tax_year": T.TAX_YEAR, "source": T.SOURCE,
            "amt_exemption": T.AMT_EXEMPTION,
            "amt_phaseout_start": T.AMT_PHASEOUT_START,
            "amt_phaseout_rate": T.AMT_PHASEOUT_RATE,
            "amt_rate_low": T.AMT_RATE_LOW, "amt_rate_high": T.AMT_RATE_HIGH,
            "amt_28_threshold": T.AMT_28_THRESHOLD,
            "standard_deduction": T.STANDARD_DEDUCTION}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        action = (q.get("action", ["constants"])[0] or "constants").lower()
        if action == "constants":
            self._respond(200, handle_constants())
        else:
            self._respond(400, {"error": "POST to calculate"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100_000:
                self._respond(413, {"error": "payload too large"})
                return
            body = json.loads((self.rfile.read(n) if n else b"{}").decode() or "{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception as e:  # noqa: BLE001
            self._respond(400, {"error": f"bad request: {type(e).__name__}"})
            return
        try:
            self._respond(200, handle_calc(body))
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
