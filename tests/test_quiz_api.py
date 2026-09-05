#!/usr/bin/env python3
"""End-to-end validation of the quiz API (api/quiz.py) with no network.

Plays complete games as scripted personas — the persona picks whichever
offered option best matches its true preferences, exactly as a real player
would — and asserts on what the product promises: the game terminates, the
profile converges on the truth, hostile payloads cannot steer it, and the
recommendation pipeline degrades honestly when market data is unavailable.

The market layer is stubbed so results depend only on engine logic. Live
network behaviour is covered separately by test_market_live.py.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "api"))
os.environ["QUIZ_DISABLE_LLM"] = "1"          # deterministic: bank questions only

import _engine as E      # noqa: E402
import _market as M      # noqa: E402
import _recommend as R   # noqa: E402
import quiz as API       # noqa: E402

FAILURES: list[str] = []
CHECKS = [0]


def check(cond, label, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(f"{label} {detail}")
        print(f"  FAIL  {label} {detail}")
    else:
        print(f"  ok    {label}")


# --------------------------------------------------------------------------
# stub market: a small global universe with realistic, hand-set metrics
# --------------------------------------------------------------------------
def _mk(sym, name, vol, dd, mom, pos, cap, dy=None, pe=None, sh=None, cur="USD"):
    return {"ok": True, "symbol": sym, "name": name, "bars": 500, "price": 100.0,
            "currency": cur, "exchange": "TEST", "ann_vol": vol, "max_dd": dd,
            "mom_12m": mom, "mom_3m": mom / 4, "pos_52w": pos, "market_cap": cap,
            "dividend_yield": dy, "pe": pe, "sharpe_window": sh}


STUB = {
    "KO":     _mk("KO", "Coca-Cola", 0.16, -0.18, 0.08, 0.55, 3.0e11, 0.031, 24, 0.5),
    "PG":     _mk("PG", "Procter & Gamble", 0.14, -0.16, 0.05, 0.48, 3.5e11, 0.025, 23, 0.4),
    "JNJ":    _mk("JNJ", "Johnson & Johnson", 0.15, -0.20, 0.06, 0.50, 3.6e11, 0.030, 17, 0.4),
    "NESN.SW": _mk("NESN.SW", "Nestle", 0.15, -0.22, 0.03, 0.40, 2.4e11, 0.033, 19, 0.3, "CHF"),
    "MSFT":   _mk("MSFT", "Microsoft", 0.26, -0.32, 0.22, 0.72, 3.1e12, 0.007, 34, 0.9),
    "NVDA":   _mk("NVDA", "NVIDIA", 0.52, -0.66, 0.85, 0.88, 4.4e12, 0.001, 45, 1.4),
    "ASML.AS": _mk("ASML.AS", "ASML", 0.40, -0.45, 0.30, 0.70, 3.0e11, 0.008, 38, 0.9, "EUR"),
    "TSLA":   _mk("TSLA", "Tesla", 0.65, -0.73, 0.35, 0.60, 9.0e11, None, 90, 0.5),
    "XOM":    _mk("XOM", "Exxon Mobil", 0.24, -0.35, 0.04, 0.42, 4.8e11, 0.034, 13, 0.3),
    "PFE":    _mk("PFE", "Pfizer", 0.23, -0.58, -0.12, 0.18, 1.5e11, 0.062, 11, -0.2),
    "SMALL":  _mk("SMALL", "Tiny Widgets", 0.95, -0.82, 1.2, 0.90, 4.0e8, None, None, 0.8),
}
STUB_THEMES = {"NVDA": ["ai", "semis"], "ASML.AS": ["ai", "semis"],
               "MSFT": ["ai", "cloud", "software"], "KO": ["consumer", "food"],
               "PG": ["consumer"], "NESN.SW": ["consumer", "food"],
               "JNJ": ["healthcare"], "PFE": ["healthcare", "biotech"],
               "XOM": ["energy"], "TSLA": ["autos"], "SMALL": ["industrial"]}

_market_mode = {"mode": "ok"}


def fake_quotes(symbols, rng="2y", workers=8):
    if _market_mode["mode"] == "dead":
        return {s: {"ok": False, "symbol": s, "reason": "rate_limited",
                    "retryable": True} for s in symbols}
    out = {}
    for s in symbols:
        out[s] = STUB.get(s, {"ok": False, "symbol": s, "reason": "not_found"})
    return out


M.quotes = fake_quotes
M.enrich = lambda s: {}
R.M = M
# candidate generation must not reach the network either
R.candidates = lambda profile, answers, want=16: {
    "symbols": list(STUB.keys()), "themes": STUB_THEMES,
    "source": "stub", "notes": []}


# --------------------------------------------------------------------------
# personas: a scoring function over an option's effects
# --------------------------------------------------------------------------
PERSONAS = {
    "retiree": {"risk": 0.05, "volatility": 0.1, "horizon": 0.7, "income": 0.95,
                "value_growth": 0.2, "momentum": 0.3, "size": 0.9, "conviction": 0.2},
    "degen": {"risk": 0.97, "volatility": 0.95, "horizon": 0.3, "income": 0.02,
              "value_growth": 0.9, "momentum": 0.9, "size": 0.3, "conviction": 0.95},
    "compounder": {"risk": 0.55, "volatility": 0.5, "horizon": 0.95, "income": 0.3,
                   "value_growth": 0.6, "momentum": 0.5, "size": 0.8, "conviction": 0.5},
    "contrarian": {"risk": 0.65, "volatility": 0.6, "horizon": 0.7, "income": 0.4,
                   "value_growth": 0.1, "momentum": 0.05, "size": 0.45, "conviction": 0.7},
}


def persona_choose(persona: dict, question: dict) -> int:
    """Pick the option whose effects sit closest to the persona's truth."""
    best_i, best_d = 0, 1e9
    for i, o in enumerate(question["options"]):
        eff = o["effects"]
        dims = [k for k in eff if k in E.DIMS]
        if not dims:
            d = 0.5                      # a pure-theme option: neutral distance
        else:
            d = sum(abs(eff[k] - persona[k]) for k in dims) / len(dims)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def play(persona_name: str, max_turns: int = 30):
    persona = PERSONAS[persona_name]
    answers: list[dict] = []
    turns = 0
    while turns < max_turns:
        res = API.handle_next({"answers": answers})
        if res.get("done"):
            return answers, res
        q = res["question"]
        i = persona_choose(persona, q)
        opt = q["options"][i]
        answers.append({"id": q["id"], "q": q["text"], "a": opt["label"],
                        "effects": opt["effects"]})
        turns += 1
    raise AssertionError(f"{persona_name}: game did not terminate in {max_turns} turns")


def main():
    print("=" * 78)
    print("QUIZ API VALIDATION (no network, bank questions only)")
    print("=" * 78)

    print("\n--- 1. health ---")
    h = API.handle_health()
    check(h["ok"] and h["bank_questions"] >= 12, "health reports a stocked question bank",
          f"{h['bank_questions']} questions")
    check(len(h["dimensions"]) == 8, "eight profile dimensions exposed")

    print("\n--- 2. first turn from a cold start ---")
    first = API.handle_next({"answers": []})
    check(not first["done"], "a fresh game is not immediately done")
    check(first["question"]["options"], "first question has options")
    check(first["confidence"] == 0.0, "confidence starts at zero")
    check(all(k in first["question"] for k in ("id", "text", "options")),
          "question matches the client contract")

    print("\n--- 3. full games terminate and converge ---")
    results = {}
    for name in PERSONAS:
        answers, done = play(name)
        results[name] = (answers, done)
        n = len(answers)
        check(API.MIN_QUESTIONS <= n <= API.MAX_QUESTIONS,
              f"{name}: game length within bounds", f"{n} questions")
        check(done["confidence"] >= 0.5, f"{name}: ends reasonably confident",
              f"conf={done['confidence']:.2f}")
        print(f"       {name:11} {n} questions, conf={done['confidence']:.2f}, "
              f"archetype={done['archetype']['name']}")

    print("\n--- 4. the profile learns what the persona actually is ---")
    for name, (answers, _) in results.items():
        prof, _, _, _ = API._rebuild(answers)
        truth = PERSONAS[name]
        err = sum(abs(prof["dims"][d] - truth[d]) for d in ("risk", "volatility")) / 2
        check(err < 0.3, f"{name}: recovered risk/volatility within 0.3",
              f"err={err:.2f} (risk {prof['dims']['risk']:.2f} vs {truth['risk']:.2f})")
    # Whole-profile recovery, not just the two safety dimensions. This is the
    # regression guard on question-bank coverage: when the bank offered only
    # extreme options, a moderate persona was forced to overstate and its
    # error here ran to 0.17 with the wrong archetype falling out.
    worst_mean = 0.0
    for name, (answers, _) in results.items():
        prof, _, _, _ = API._rebuild(answers)
        truth = PERSONAS[name]
        errs = {d: abs(prof["dims"][d] - truth[d]) for d in E.DIMS}
        mean_err = sum(errs.values()) / len(errs)
        worst_mean = max(worst_mean, mean_err)
        worst_dim = max(errs.items(), key=lambda kv: kv[1])
        check(mean_err < 0.16, f"{name}: whole profile recovered accurately",
              f"mean abs err={mean_err:.3f}, worst={worst_dim[0]} {worst_dim[1]:.2f}")
    check(worst_mean < 0.16, "every persona recovered within tolerance",
          f"worst mean err={worst_mean:.3f}")

    arch = {n: r[1]["archetype"]["name"] for n, r in results.items()}
    check(arch["compounder"] == "The Compounder",
          "a decades-horizon player is labelled The Compounder", arch["compounder"])
    check(arch["retiree"] in ("The Landlord", "The Vault"),
          "an income-first player gets an income archetype", arch["retiree"])
    check(arch["contrarian"] == "The Bargain Hunter",
          "a value-first player is labelled The Bargain Hunter", arch["contrarian"])
    check(arch["degen"] in ("The Moonshot", "The Sniper", "The Surfer"),
          "a maximal-risk player gets an aggressive archetype", arch["degen"])
    check(len(set(arch.values())) >= 3, "archetypes discriminate between personas",
          str(arch))

    p_ret, _, _, _ = API._rebuild(results["retiree"][0])
    p_deg, _, _, _ = API._rebuild(results["degen"][0])
    check(p_ret["dims"]["risk"] < p_deg["dims"]["risk"] - 0.3,
          "retiree ends far more risk-averse than the degen",
          f"{p_ret['dims']['risk']:.2f} vs {p_deg['dims']['risk']:.2f}")
    check(E._target_vol(p_ret) < E._target_vol(p_deg),
          "target volatility ordered correctly across personas",
          f"{E._target_vol(p_ret):.0%} vs {E._target_vol(p_deg):.0%}")

    print("\n--- 5. recommendations fit the persona ---")
    picks = {}
    for name, (answers, _) in results.items():
        out = API.handle_pick({"answers": answers})
        check(out.get("ok"), f"{name}: a recommendation was produced")
        if not out.get("ok"):
            continue
        pick = out["pick"]
        picks[name] = pick["symbol"]
        print(f"       {name:11} -> {pick['symbol']:9} {pick['name'][:22]:24} "
              f"fit={pick['fit']:.3f} vol={pick['ann_vol']:.0%} "
              f"(target {out['profile']['target_vol']:.0%}) "
              f"[{out['archetype']['name']}]")
        check("headline" in out["copy"] and out["copy"]["why"],
              f"{name}: reveal copy present")
        check(len(out["runners_up"]) >= 1, f"{name}: runners-up offered")

    check(picks.get("retiree") in ("KO", "PG", "JNJ", "NESN.SW", "XOM"),
          "retiree matched to a defensive dividend name", str(picks.get("retiree")))
    check(picks.get("degen") in ("NVDA", "TSLA", "ASML.AS", "SMALL"),
          "degen matched to a high-octane name", str(picks.get("degen")))
    check(picks.get("retiree") != picks.get("degen"),
          "opposite personas get opposite recommendations")
    ret_vol = STUB[picks["retiree"]]["ann_vol"] if picks.get("retiree") in STUB else 9
    deg_vol = STUB[picks["degen"]]["ann_vol"] if picks.get("degen") in STUB else 0
    check(ret_vol < deg_vol, "the cautious persona's pick is genuinely calmer",
          f"{ret_vol:.0%} vs {deg_vol:.0%}")

    print("\n--- 6. hostile and malformed input ---")
    forged = [{"id": "x", "q": "q", "a": "a",
               "effects": {"risk": 99, "volatility": -5, "themes": ["not_a_theme"],
                           "evil": 1, "weight": 1000}}] * 4
    prof, clean, _, _ = API._rebuild(forged)
    check(all(0.0 <= prof["dims"][d] <= 1.0 for d in E.DIMS),
          "forged out-of-range effects are clamped into 0..1")
    check("evil" not in prof["dims"], "unknown dimensions are discarded")
    check(prof["themes"] == {}, "invented themes are rejected")
    out = API.handle_pick({"answers": forged})
    check(out.get("ok") is True, "a forged history still yields a safe recommendation")
    check(STUB.get(out["pick"]["symbol"], {}).get("ann_vol", 9) < 2.5,
          "forged 'max risk' cannot unlock an unsafe instrument")

    check(API.handle_pick({"answers": []}).get("reason") == "not_enough_answers",
          "an empty history is refused rather than guessed at")
    junk = API._rebuild([{"nope": 1}, "string", None, {"effects": "bad"}])[0]
    check(E.confidence(junk) == 0.0, "junk answers contribute nothing")
    big = API.handle_next({"answers": [{"id": f"q{i}", "q": "x", "a": "y",
                                        "effects": {"risk": 0.5}}
                                       for i in range(100)]})
    check(big.get("done") is True, "an over-long history terminates instead of looping")

    print("\n--- 7. honest degradation when market data is unavailable ---")
    _market_mode["mode"] = "dead"
    out = API.handle_pick({"answers": results["compounder"][0]})
    check(out.get("ok") is False, "no verified data means no recommendation is invented")
    check(out.get("reason") == "no_verified_candidates", "the reason is explicit",
          str(out.get("reason")))
    check(out["diagnostics"]["throttled"] > 0, "diagnostics report the throttling")
    _market_mode["mode"] = "ok"

    print("\n--- 8. determinism ---")
    a1, _ = play("compounder")
    a2, _ = play("compounder")
    check([x["id"] for x in a1] == [x["id"] for x in a2],
          "the same persona plays the same game")
    o1 = API.handle_pick({"answers": a1})["pick"]["symbol"]
    o2 = API.handle_pick({"answers": a2})["pick"]["symbol"]
    check(o1 == o2, "the same answers produce the same pick", f"{o1} == {o2}")

    print("\n" + "=" * 78)
    print(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
