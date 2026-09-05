#!/usr/bin/env python3
"""Head-to-head on the two jobs this site actually gives a model.

  python scripts/compare_llm.py

Judged on what matters per job, not on vibes:
  ISO EXPLAIN — factual accuracy against numbers we computed ourselves. A
      model that invents a tax figure is disqualifying, so the check is
      whether every number in the output appears in the input.
  QUIZ QUESTION — validity against the game's schema, spread of the effect
      values (extremes-only distorts profiles), and latency.

Requires keys in the environment; skips any provider that has none.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "api"))

import _engine as E      # noqa: E402
import _llm              # noqa: E402
import _questions as Q   # noqa: E402
import _tax as T         # noqa: E402

SCENARIO = dict(shares=10_000, strike=2.0, fmv=52.0, income=200_000, status="single")


def iso_prompt():
    p = T.plan(SCENARIO["shares"], SCENARIO["strike"], SCENARIO["fmv"],
               SCENARIO["income"], status=SCENARIO["status"])
    facts = (
        f"Filing status: single. Wage income: ${SCENARIO['income']:,.0f}.\n"
        f"ISO grant: {p['shares_total']:,} options, ${SCENARIO['strike']:.2f} "
        f"strike, ${SCENARIO['fmv']:.2f} current fair market value, "
        f"${p['spread_per_share']:.2f} spread per share.\n"
        f"Shares exercisable this year with ZERO AMT: {p['amt_free_shares']:,}\n"
        f"Cost to exercise those: ${p['amt_free_exercise_cost']:,.0f}\n"
        f"If all {p['shares_total']:,} are exercised and held: AMT owed "
        f"${p['exercise_all']['amt_owed']:,.0f} on top of "
        f"${p['baseline_tax']:,.0f} regular tax, and the cost of the next "
        f"dollar of bargain element is "
        f"{p['exercise_all']['marginal_amt_rate']:.0%}.\n"
        f"Years to exercise the whole grant AMT-free at this income: "
        f"{p['years_to_exercise_all_amt_free']}\n"
        f"Tax year {T.TAX_YEAR}. Federal only — no state tax, no AMT credit "
        f"carryforward.")
    system = (
        "You explain an ISO exercise decision to a smart person who is not a "
        "tax professional. You are given figures that have ALREADY been "
        "computed. Use only those figures — never introduce a number that is "
        "not in the input, and never estimate one. Three short paragraphs, "
        "plain language, second person. Say what the trade-off is and what "
        "the omissions (state tax, AMT credit) could change. No disclaimers "
        "about consulting professionals; the interface handles that.")
    return system, facts, p


NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in(text: str) -> set:
    out = set()
    for m in NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def run_iso(provider: str):
    system, facts, p = iso_prompt()
    os.environ["QUIZ_PROVIDER_SMART"] = provider
    t0 = time.time()
    try:
        text, used = _llm.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": facts}],
            task="smart", max_tokens=700, temperature=0.4)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:90]}"}
    dt = time.time() - t0
    if used != provider:
        return {"ok": False, "error": f"routed to {used}, not {provider}"}

    # NOTE ON THIS METRIC: the first version counted any number absent from
    # the input as "invented", and it got the verdict exactly backwards —
    # it flagged Claude for $20,000, $31,350 and $191,110, all of which are
    # correct arithmetic on supplied figures (shares x strike, shares x
    # spread, regular tax + AMT), while passing a model that had written a
    # materially wrong sentence containing only quoted numbers. Derived
    # values are therefore allowed, and a wrong CLAIM is something only
    # reading the output catches. The metric is a filter, not a judge.
    allowed = numbers_in(facts) | {
        float(T.TAX_YEAR), 26.0, 28.0, 0.26, 0.28, 50.0, 100.0, 1.0, 2.0, 3.0,
        float(p["amt_free_shares"]), float(p["shares_total"])}
    derived = [
        p["shares_total"] * SCENARIO["strike"],              # cost of all
        p["amt_free_shares"] * p["spread_per_share"],        # BE at crossover
        p["amt_free_shares"] * SCENARIO["fmv"],              # stock value
        p["shares_total"] * SCENARIO["fmv"],
        p["baseline_tax"] + p["exercise_all"]["amt_owed"],   # total tax
        p["shares_total"] * SCENARIO["strike"]
        + p["baseline_tax"] + p["exercise_all"]["amt_owed"],  # out of pocket
    ]
    allowed |= {round(d, 2) for d in derived}
    # tolerate rounded restatements of any figure we supplied or derived
    for a in list(allowed):
        allowed.add(round(a / 1000.0, 1))
        allowed.add(round(a, 0))
        allowed.add(round(a / 1000.0))
    invented = {n for n in numbers_in(text) if n not in allowed and n > 3}
    return {"ok": True, "seconds": round(dt, 2), "chars": len(text),
            "invented": sorted(invented)[:6], "n_invented": len(invented),
            "text": text}


def run_quiz(provider: str, n: int = 4):
    os.environ["QUIZ_PROVIDER_FAST"] = provider
    prof = E.new_profile()
    prof = E.apply_answer(prof, {"risk": 0.7, "volatility": 0.6, "weight": 1.0})
    answers, texts = [], []
    lat, valid, mids, tot, lens = [], 0, 0, 0, []
    for _ in range(n):
        t0 = time.time()
        q, src = Q.llm_question(prof, texts, [], answers)
        lat.append(time.time() - t0)
        if not q:
            continue
        valid += 1
        texts.append(q["text"])
        lens.append(len(q["text"]))
        for o in q["options"]:
            lens.append(len(o["label"]))
            for k, v in o["effects"].items():
                if k != "weight" and isinstance(v, float):
                    tot += 1
                    if 0.15 < v < 0.85:
                        mids += 1
    return {"valid": f"{valid}/{n}",
            "mean_seconds": round(sum(lat) / max(len(lat), 1), 2),
            "moderate_pct": round(100 * mids / tot) if tot else 0,
            "max_len": max(lens) if lens else 0,
            "sample": texts[0] if texts else None}


def main():
    have = _llm.available()
    print("=" * 78)
    print("PROVIDER COMPARISON — configured:", have or "none")
    print("=" * 78)
    if len(have) < 1:
        print("no providers configured")
        return 1

    print("\n### JOB 1 — explaining an ISO/AMT result (quality-critical)\n")
    iso = {}
    for p in have:
        r = run_iso(p)
        iso[p] = r
        if not r.get("ok"):
            print(f"  {p:12} FAILED — {r['error']}")
            continue
        verdict = "clean" if r["n_invented"] == 0 else \
            f"INVENTED {r['n_invented']} figure(s): {r['invented']}"
        print(f"  {p:12} {r['seconds']:>5.2f}s  {r['chars']:>4} chars  {verdict}")
    for p, r in iso.items():
        if r.get("ok"):
            print(f"\n--- {p} ---\n{r['text'][:900]}")

    print("\n\n### JOB 2 — writing a quiz question (latency-critical)\n")
    for p in have:
        try:
            r = run_quiz(p)
            print(f"  {p:12} valid {r['valid']}  {r['mean_seconds']:>5.2f}s  "
                  f"moderate effects {r['moderate_pct']:>3}%  "
                  f"longest line {r['max_len']}c")
            if r["sample"]:
                print(f"               e.g. {r['sample'][:74]}")
        except Exception as e:  # noqa: BLE001
            print(f"  {p:12} FAILED {type(e).__name__}: {str(e)[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
