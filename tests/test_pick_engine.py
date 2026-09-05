#!/usr/bin/env python3
"""Validation suite for the stock-match engine (api/_engine.py).

Runs with plain `python tests/test_pick_engine.py` — no pytest, no network,
no LLM. Everything asserted here is a property the product depends on:
profile mechanics, safety gates, fit shape, persona outcomes, determinism
and degradation. Synthetic stocks are used deliberately, so a failure means
the ENGINE changed behaviour rather than the market moving.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import _engine as E  # noqa: E402

FAILURES: list[str] = []
CHECKS = [0]


def check(cond: bool, label: str, detail: str = ""):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(f"{label}{(' — ' + detail) if detail else ''}")
        print(f"  FAIL  {label} {detail}")
    else:
        print(f"  ok    {label}")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------
# synthetic but realistic stock fixtures (numbers chosen to mirror real ones)
# --------------------------------------------------------------------------
def mk(sym, name, vol, dd, mom12, pos, mcap=None, dy=None, pe=None,
       sharpe=None, bars=500, price=100.0, cur="USD"):
    return {"ok": True, "symbol": sym, "name": name, "bars": bars, "price": price,
            "currency": cur, "ann_vol": vol, "max_dd": dd, "mom_12m": mom12,
            "mom_3m": mom12 / 4, "pos_52w": pos, "market_cap": mcap,
            "dividend_yield": dy, "pe": pe, "sharpe_window": sharpe}


UNIVERSE = {
    "KO":    mk("KO", "Coca-Cola", 0.16, -0.18, 0.08, 0.55, 3.0e11, 0.031, 24, 0.5),
    "JNJ":   mk("JNJ", "Johnson & Johnson", 0.15, -0.20, 0.06, 0.50, 3.6e11, 0.030, 17, 0.4),
    "PG":    mk("PG", "Procter & Gamble", 0.14, -0.16, 0.05, 0.48, 3.5e11, 0.025, 23, 0.4),
    "MSFT":  mk("MSFT", "Microsoft", 0.26, -0.32, 0.22, 0.72, 3.1e12, 0.007, 34, 0.9),
    "AAPL":  mk("AAPL", "Apple", 0.28, -0.31, 0.15, 0.65, 3.4e12, 0.005, 30, 0.7),
    "NVDA":  mk("NVDA", "NVIDIA", 0.52, -0.66, 0.85, 0.88, 4.4e12, 0.001, 45, 1.4),
    "TSLA":  mk("TSLA", "Tesla", 0.65, -0.73, 0.35, 0.60, 9.0e11, None, 90, 0.5),
    "ARM":   mk("ARM", "Arm Holdings", 0.58, -0.55, 0.40, 0.70, 1.4e11, None, 95, 0.6),
    "XOM":   mk("XOM", "Exxon Mobil", 0.24, -0.35, 0.04, 0.42, 4.8e11, 0.034, 13, 0.3),
    "PFE":   mk("PFE", "Pfizer", 0.23, -0.58, -0.12, 0.18, 1.5e11, 0.062, 11, -0.2),
    "SMALL": mk("SMALL", "Tiny Widgets", 0.95, -0.82, 1.20, 0.90, 4.0e8, None, None, 0.8),
    "PENNY": mk("PENNY", "Penny Co", 1.60, -0.93, 2.0, 0.95, 6.0e7, None, None, 0.4,
                price=0.42),
    "NEW":   mk("NEW", "Just IPO'd", 0.70, -0.40, 0.50, 0.80, 8.0e9, None, None, 0.9,
                bars=120),
}


def build(answers):
    p = E.new_profile()
    for a in answers:
        p = E.apply_answer(p, a)
    return p


CONSERVATIVE = build([
    {"risk": 0.08, "volatility": 0.10, "weight": 1.0},
    {"horizon": 0.75, "income": 0.85, "weight": 1.0},
    {"value_growth": 0.25, "size": 0.9, "weight": 1.0},
    {"themes": ["consumer", "food"], "weight": 1.0},
])
AGGRESSIVE = build([
    {"risk": 0.95, "volatility": 0.92, "weight": 1.0},
    {"horizon": 0.35, "momentum": 0.9, "weight": 1.0},
    {"value_growth": 0.9, "size": 0.4, "conviction": 0.9, "weight": 1.0},
    {"themes": ["ai", "semis"], "weight": 1.0},
])
INCOME = build([
    {"risk": 0.25, "volatility": 0.25, "weight": 1.0},
    {"income": 0.95, "horizon": 0.8, "weight": 1.0},
    {"value_growth": 0.15, "size": 0.8, "weight": 1.0},
])
BALANCED = build([
    {"risk": 0.5, "volatility": 0.5, "weight": 1.0},
    {"horizon": 0.7, "momentum": 0.55, "weight": 1.0},
    {"size": 0.75, "value_growth": 0.55, "weight": 1.0},
])


def section(t):
    print(f"\n--- {t} ---")


def main():
    print("=" * 78)
    print("STOCK-MATCH ENGINE VALIDATION")
    print("=" * 78)

    # ---------------------------------------------------------------- profile
    section("1. profile mechanics")
    p0 = E.new_profile()
    check(all(close(p0["dims"][d], 0.5) for d in E.DIMS),
          "fresh profile is neutral on every dimension")
    check(E.confidence(p0) == 0.0, "fresh profile has zero confidence")

    p1 = E.apply_answer(p0, {"risk": 1.0, "weight": 1.0})
    check(p1["dims"]["risk"] > 0.7, "first answer moves its dimension decisively",
          f"risk={p1['dims']['risk']:.3f}")
    check(all(close(p1["dims"][d], 0.5) for d in E.DIMS if d != "risk"),
          "an answer touches only the dimensions it names")
    check(p0["dims"]["risk"] == 0.5, "apply_answer does not mutate its input")

    p2 = E.apply_answer(p1, {"risk": 1.0, "weight": 1.0})
    first_step = p1["dims"]["risk"] - 0.5
    second_step = p2["dims"]["risk"] - p1["dims"]["risk"]
    check(second_step < first_step,
          "learning rate decays as evidence accumulates",
          f"{first_step:.3f} -> {second_step:.3f}")

    p_conf = p0
    confs = []
    for _ in range(6):
        p_conf = E.apply_answer(p_conf, {"risk": 0.7, "volatility": 0.6,
                                         "horizon": 0.6, "themes": ["ai"],
                                         "weight": 1.0})
        confs.append(E.confidence(p_conf))
    check(all(confs[i] < confs[i + 1] for i in range(len(confs) - 1)),
          "confidence rises monotonically with evidence")
    check(confs[-1] < 1.0, "confidence never saturates to a false certainty",
          f"after 6 answers: {confs[-1]:.3f}")

    weak = E.weakest_dimensions(p1, 3)
    check("risk" not in weak, "the answered dimension is no longer the weakest")
    check(set(weak) <= set(E.DIMS), "weakest dimensions are valid dimension names")

    hedged = E.apply_answer(p0, {"risk": 1.0, "weight": 0.3})
    decisive = E.apply_answer(p0, {"risk": 1.0, "weight": 1.0})
    check(decisive["dims"]["risk"] > hedged["dims"]["risk"],
          "a decisive answer moves further than a hedged one")

    clamped = E.apply_answer(p0, {"risk": 99.0, "weight": 50.0})
    check(0.0 <= clamped["dims"]["risk"] <= 1.0, "out-of-range input is clamped")
    junk = E.apply_answer(p0, {"risk": "not-a-number", "weight": 1.0})
    check(close(junk["dims"]["risk"], 0.5), "malformed answer value is ignored")

    # ------------------------------------------------------------------ gates
    section("2. safety gates")
    check(E.gate({"ok": False, "reason": "not_found"}, BALANCED) == "not_found",
          "unverified symbol is rejected")
    check(E.gate(UNIVERSE["NEW"], BALANCED) == "too_new",
          "insufficient history is rejected")
    check(E.gate(UNIVERSE["PENNY"], AGGRESSIVE) in ("sub_dollar", "extreme_vol"),
          "sub-dollar penny stock is rejected even for the boldest profile")
    check(E.gate(UNIVERSE["NVDA"], CONSERVATIVE) == "too_volatile_for_profile",
          "high-vol name is gated out of a preservation profile")
    check(E.gate(UNIVERSE["NVDA"], AGGRESSIVE) is None,
          "the same name is allowed for an aggressive profile")
    check(E.gate(UNIVERSE["KO"], CONSERVATIVE) is None,
          "a defensive name passes for a conservative profile")
    check(E.gate({"ok": True, "bars": 400, "price": 50, "currency": "USD"},
                 BALANCED) == "no_vol",
          "missing volatility is rejected rather than assumed")

    # -------------------------------------------------------------------- fit
    section("3. fit scoring shape")
    tv_c = E._target_vol(CONSERVATIVE)
    tv_a = E._target_vol(AGGRESSIVE)
    check(tv_c < 0.25, "conservative profile targets low volatility", f"{tv_c:.1%}")
    check(tv_a > 0.5, "aggressive profile targets high volatility", f"{tv_a:.1%}")
    check(tv_a > tv_c, "target volatility is ordered by risk appetite")

    f_ko_c = E.fit(UNIVERSE["KO"], CONSERVATIVE)["score"]
    f_nv_c = E.fit(UNIVERSE["NVDA"], CONSERVATIVE)["score"]
    check(f_ko_c > f_nv_c, "conservative: defensive name outscores a volatile one",
          f"KO={f_ko_c:.3f} NVDA={f_nv_c:.3f}")
    f_ko_a = E.fit(UNIVERSE["KO"], AGGRESSIVE)["score"]
    f_nv_a = E.fit(UNIVERSE["NVDA"], AGGRESSIVE, ["ai", "semis"])["score"]
    check(f_nv_a > f_ko_a, "aggressive: the volatile thematic name wins",
          f"NVDA={f_nv_a:.3f} KO={f_ko_a:.3f}")

    themed = E.fit(UNIVERSE["NVDA"], AGGRESSIVE, ["ai", "semis"])["score"]
    unthemed = E.fit(UNIVERSE["NVDA"], AGGRESSIVE, ["utilities"])["score"]
    check(themed > unthemed, "matching the user's theme raises the score",
          f"{themed:.3f} vs {unthemed:.3f}")

    p_avoid = E.apply_answer(AGGRESSIVE, {"avoid_themes": ["crypto"], "weight": 1.0})
    avoided = E.fit(UNIVERSE["NVDA"], p_avoid, ["crypto"])
    check(avoided["parts"].get("theme") == 0.0,
          "an explicitly avoided theme zeroes thematic credit")

    f_hi_y = E.fit(UNIVERSE["PFE"], INCOME)["parts"].get("income", 0)
    f_no_y = E.fit(UNIVERSE["MSFT"], INCOME)["parts"].get("income", 0)
    check(f_hi_y > f_no_y, "income profile rewards a real yield",
          f"PFE={f_hi_y:.2f} MSFT={f_no_y:.2f}")

    mom_p = build([{"momentum": 0.95, "risk": 0.7, "volatility": 0.7, "weight": 1.0}])
    con_p = build([{"momentum": 0.05, "risk": 0.7, "volatility": 0.7, "weight": 1.0}])
    check(E.fit(UNIVERSE["NVDA"], mom_p)["parts"]["momentum"] >
          E.fit(UNIVERSE["PFE"], mom_p)["parts"]["momentum"],
          "momentum profile prefers the strong performer")
    check(E.fit(UNIVERSE["PFE"], con_p)["parts"]["momentum"] >
          E.fit(UNIVERSE["NVDA"], con_p)["parts"]["momentum"],
          "contrarian profile prefers the beaten-down name")

    check(0.0 <= f_ko_c <= 1.0 and 0.0 <= f_nv_a <= 1.0,
          "fit scores stay inside 0..1")

    # --------------------------------------------------------------- personas
    section("4. end-to-end persona ranking")
    stocks = list(UNIVERSE.values())
    themes = {"NVDA": ["ai", "semis"], "ARM": ["ai", "semis"], "MSFT": ["ai", "cloud"],
              "AAPL": ["consumer"], "KO": ["consumer", "food"], "PG": ["consumer"],
              "JNJ": ["healthcare"], "PFE": ["healthcare", "biotech"],
              "XOM": ["energy"], "TSLA": ["autos"], "SMALL": ["industrial"],
              "PENNY": ["crypto"], "NEW": ["ai"]}

    top_c, rej_c = E.rank(stocks, CONSERVATIVE, themes, limit=5)
    names_c = [r["symbol"] for r in top_c]
    print(f"    conservative -> {names_c}")
    check(names_c[0] in ("KO", "PG", "JNJ"),
          "conservative persona is matched to a defensive blue chip", str(names_c[:3]))
    check("NVDA" not in names_c and "PENNY" not in names_c,
          "conservative persona never sees the speculative names")
    check(any(r["reason"] == "sub_dollar" or r["reason"] == "extreme_vol"
              for r in rej_c), "penny stock appears in the rejection log")

    top_a, _ = E.rank(stocks, AGGRESSIVE, themes, limit=5)
    names_a = [r["symbol"] for r in top_a]
    print(f"    aggressive   -> {names_a}")
    check(names_a[0] in ("NVDA", "ARM"),
          "aggressive AI persona is matched to an AI name", str(names_a[:3]))

    top_i, _ = E.rank(stocks, INCOME, themes, limit=5)
    names_i = [r["symbol"] for r in top_i]
    print(f"    income       -> {names_i}")
    check(names_i[0] in ("KO", "PG", "JNJ", "XOM", "PFE"),
          "income persona is matched to a dividend payer", str(names_i[:3]))
    top_syms = set(names_i[:3])
    check(all(UNIVERSE[s].get("dividend_yield") for s in top_syms if s in UNIVERSE),
          "every top income match actually pays a dividend")

    check(names_c != names_a, "different personas produce different recommendations")

    # ------------------------------------------------------- robustness
    section("5. determinism and degradation")
    r1, _ = E.rank(stocks, BALANCED, themes, limit=5)
    r2, _ = E.rank(list(reversed(stocks)), BALANCED, themes, limit=5)
    check([x["symbol"] for x in r1] == [x["symbol"] for x in r2],
          "ranking is independent of input order")
    check(E.fit(UNIVERSE["KO"], CONSERVATIVE)["score"] ==
          E.fit(UNIVERSE["KO"], CONSERVATIVE)["score"], "fit is deterministic")

    sparse = {"ok": True, "symbol": "SPARSE", "name": "Sparse Data Co", "bars": 400,
              "price": 40.0, "currency": "USD", "ann_vol": 0.25, "max_dd": -0.3}
    fs = E.fit(sparse, BALANCED)
    check(0.0 <= fs["score"] <= 1.0, "a stock with only core metrics still scores",
          f"score={fs['score']:.3f} on {len(fs['parts'])} components")
    check("size" not in fs["parts"] and "income" not in fs["parts"],
          "missing fields are omitted rather than imputed")

    empty, rej = E.rank([], BALANCED, {}, limit=5)
    check(empty == [] and rej == [], "empty candidate list is handled")
    allbad, rej_all = E.rank([UNIVERSE["NEW"], UNIVERSE["PENNY"]], CONSERVATIVE,
                             {}, limit=5)
    check(allbad == [] and len(rej_all) == 2,
          "when everything is gated out, nothing is recommended")

    # neutral profile must still produce a sane, non-degenerate ranking
    top_n, _ = E.rank(stocks, E.new_profile(), themes, limit=3)
    check(len(top_n) >= 3, "a neutral profile still yields candidates")
    print(f"    neutral      -> {[r['symbol'] for r in top_n]}")

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
