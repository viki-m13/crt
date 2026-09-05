#!/usr/bin/env python3
"""Validation of the ISO/AMT engine (api/_tax.py).

Every headline case is worked out by hand in the comments so the assertion
can be checked with a pencil rather than taken on faith. If a constant
changes next year, these numbers should be recomputed by hand too — a test
that merely records whatever the code currently prints protects nothing.

    python tests/test_amt.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import _tax as T  # noqa: E402

FAIL: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    if cond:
        print(f"  ok    {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label} {detail}")


def near(a, b, tol=1.0):
    return abs(a - b) <= tol


def main():
    print("=" * 78)
    print(f"ISO / AMT ENGINE — tax year {T.TAX_YEAR}")
    print(f"source: {T.SOURCE}")
    print("=" * 78)

    # ---------------------------------------------------------------- constants
    print("\n--- 1. constants match the published 2026 figures ---")
    check(T.AMT_EXEMPTION["single"] == 90_100, "AMT exemption single = $90,100")
    check(T.AMT_EXEMPTION["mfj"] == 140_200, "AMT exemption MFJ = $140,200")
    check(T.AMT_PHASEOUT_START["single"] == 500_000, "phaseout starts $500k single")
    check(T.AMT_PHASEOUT_START["mfj"] == 1_000_000, "phaseout starts $1M MFJ")
    check(T.AMT_PHASEOUT_RATE == 0.50, "OBBBA phaseout rate is 50c per dollar")
    check(T.AMT_28_THRESHOLD == 244_500, "28% AMT rate starts at $244,500")
    check(T.STANDARD_DEDUCTION["single"] == 16_100, "standard deduction single")

    # The published phaseout ENDPOINTS are a consequence of the rate, so they
    # are a genuine cross-check on the rate rather than a restatement of it:
    # single 90,100 / 0.5 = 180,200 above 500,000 -> exhausted at 680,200.
    check(near(T.amt_exemption(680_200, "single"), 0.0),
          "single exemption fully phased out at $680,200 (published figure)")
    check(near(T.amt_exemption(1_280_400, "mfj"), 0.0),
          "MFJ exemption fully phased out at $1,280,400 (published figure)")
    check(near(T.amt_exemption(600_000, "single"), 40_100),
          "single exemption at $600k AMTI = $40,100",
          "90,100 - 0.5 x 100,000")

    # ------------------------------------------------------------ regular tax
    print("\n--- 2. regular tax, hand-computed ---")
    # Single, $200,000 wages, standard deduction 16,100 -> taxable 183,900
    #   10% x 12,400                    =  1,240.00
    #   12% x (50,400 - 12,400)=38,000  =  4,560.00
    #   22% x (105,700-50,400)=55,300   = 12,166.00
    #   24% x (183,900-105,700)=78,200  = 18,768.00
    #                             total = 36,734.00
    r = T.regular_tax(200_000, 0, 16_100, "single")
    check(near(r["taxable_income"], 183_900), "taxable income 200k single = 183,900")
    check(near(r["tax"], 36_734), "regular tax on 183,900 = $36,734",
          f"got {r['tax']:,.2f}")

    # MFJ, $400,000 wages, standard deduction 32,200 -> taxable 367,800
    #   10% x 24,800                     =  2,480.00
    #   12% x (100,800-24,800)=76,000    =  9,120.00
    #   22% x (211,400-100,800)=110,600  = 24,332.00
    #   24% x (367,800-211,400)=156,400  = 37,536.00
    #                              total = 73,468.00
    r2 = T.regular_tax(400_000, 0, 32_200, "mfj")
    check(near(r2["tax"], 73_468), "regular tax MFJ 367,800 taxable = $73,468",
          f"got {r2['tax']:,.2f}")

    # capital gains stack ABOVE ordinary income rather than starting at zero
    g = T.regular_tax(200_000, 50_000, 16_100, "single")
    check(near(g["tax"] - r["tax"], 50_000 * 0.15),
          "LTCG stacked above ordinary income taxed at 15%",
          f"delta {g['tax']-r['tax']:,.0f} vs 7,500")

    # ------------------------------------------------------------------- AMT
    print("\n--- 3. AMT, hand-computed ---")
    # Single, 200k wages, NO ISO. AMTI = 200,000 (standard deduction is not
    # allowed for AMT, so it is added back).
    #   exemption 90,100 (AMTI below 500k)
    #   base 109,900 ; all below 244,500 so 26%
    #   TMT = 28,574 < regular 36,734 -> no AMT
    z = T.compute(200_000, 0, status="single")
    check(near(z["amti"], 200_000), "AMTI adds back the standard deduction")
    check(near(z["tentative_minimum_tax"], 28_574), "TMT = $28,574",
          f"got {z['tentative_minimum_tax']:,.2f}")
    check(z["amt_owed"] == 0, "no AMT when TMT is below regular tax")

    # Crossover: TMT must reach regular tax 36,734.
    #   0.26 x (AMTI - 90,100) = 36,734 -> AMTI = 231,384.62
    #   bargain element = 231,384.62 - 200,000 = 31,384.62
    x = T.crossover_bargain_element(200_000, status="single")
    check(near(x, 31_384.62, 2.0), "crossover bargain element = $31,384",
          f"got {x:,.2f}")
    check(T.compute(200_000, x - 50, status="single")["amt_owed"] == 0,
          "just below the crossover owes no AMT")
    check(T.compute(200_000, x + 500, status="single")["amt_owed"] > 0,
          "just above the crossover owes AMT")

    # Above the crossover the AMT cost of a dollar is 26% (outside phaseout)
    mr = T.marginal_amt_rate(200_000, x + 50_000, status="single")
    check(near(mr, 0.26, 0.005), "marginal AMT rate is 26% outside the phaseout",
          f"got {mr:.3f}")

    print("\n--- 4. the OBBBA phaseout trap ---")
    # Inside the phaseout each extra $1 of AMTI also destroys $0.50 of
    # exemption, so the AMT base grows by $1.50 and the marginal cost is
    # 1.5 x 26% = 39%, or 1.5 x 28% = 42% above the 28% threshold.
    inside = T.marginal_amt_rate(450_000, 120_000, status="single")
    check(near(inside, 0.42, 0.01),
          "inside the phaseout the marginal rate is ~42%, not 28%",
          f"got {inside:.3f}")
    outside = T.marginal_amt_rate(200_000, 60_000, status="single")
    check(outside < inside - 0.10,
          "the phaseout band is markedly more expensive than outside it",
          f"{outside:.2f} vs {inside:.2f}")

    # AMTI 600k single: exemption 40,100, base 559,900
    #   26% x 244,500                 = 63,570
    #   28% x (559,900-244,500)=315,400 = 88,312
    #                             TMT = 151,882
    t = T.tentative_minimum_tax(600_000, 0, "single")
    check(near(t["tmt"], 151_882), "TMT at AMTI 600k single = $151,882",
          f"got {t['tmt']:,.2f}")

    print("\n--- 5. monotonicity and sanity ---")
    prev = -1.0
    mono = True
    for be in range(0, 900_000, 25_000):
        v = T.compute(250_000, be, status="single")["amt_owed"]
        if v < prev - 1e-6:
            mono = False
        prev = v
    check(mono, "AMT owed never decreases as the bargain element grows")
    check(T.compute(250_000, 0, status="single")["amt_owed"] == 0,
          "zero bargain element means zero AMT for a normal earner")
    check(T.crossover_bargain_element(3_000_000, status="single") >= 0,
          "a very high earner still returns a defined crossover")
    hi = T.compute(3_000_000, 0, status="single")
    check(near(hi["amt_exemption"], 0.0),
          "exemption is fully gone for a $3M earner")

    print("\n--- 6. filing status behaves correctly ---")
    xs = T.crossover_bargain_element(200_000, status="single")
    xm = T.crossover_bargain_element(200_000, status="mfj")
    check(xm > xs, "MFJ can exercise more AMT-free than single at equal income",
          f"MFJ ${xm:,.0f} vs single ${xs:,.0f}")
    try:
        T.compute(100_000, 0, status="nonsense")
        check(False, "invalid filing status is rejected")
    except ValueError:
        check(True, "invalid filing status is rejected")

    print("\n--- 7. the planner ---")
    # 10,000 ISOs, $2 strike, $52 FMV -> $50 spread, $500,000 bargain element
    p = T.plan(10_000, 2.0, 52.0, 200_000, status="single")
    check(near(p["spread_per_share"], 50.0), "spread per share = $50")
    check(p["amt_free_shares"] == int(31_384.62 // 50),
          f"AMT-free shares = {int(31_384.62 // 50)}",
          f"got {p['amt_free_shares']}")
    check(near(p["amt_free_exercise_cost"], p["amt_free_shares"] * 2.0),
          "exercise cost is shares x strike")
    check(p["exercise_all"]["amt_owed"] > 100_000,
          "exercising all 10,000 at once triggers a large AMT bill",
          f"${p['exercise_all']['amt_owed']:,.0f}")
    check(p["years_to_exercise_all_amt_free"] is not None
          and p["years_to_exercise_all_amt_free"] > 1,
          "a multi-year ladder is required",
          f"{p['years_to_exercise_all_amt_free']} years (capped at 10)")

    rows = T.ladder(10_000, 2.0, 52.0, 200_000, status="single", points=5)
    check(len(rows) == 5, "ladder returns the requested number of rows")
    check(all(rows[i]["amt_owed"] <= rows[i + 1]["amt_owed"] + 1e-6
              for i in range(len(rows) - 1)), "ladder AMT is non-decreasing")

    print("\n--- 8. degenerate inputs ---")
    check(T.plan(0, 2.0, 52.0, 200_000)["amt_free_shares"] == 0,
          "zero shares yields zero")
    under = T.plan(1_000, 60.0, 52.0, 200_000)     # underwater options
    check(under["spread_per_share"] == 0.0, "underwater options have zero spread")
    check(under["exercise_all"]["amt_owed"] == 0,
          "underwater options create no AMT")
    check(T.compute(0, 0, status="single")["total_federal_tax"] == 0,
          "zero income means zero tax")
    neg = T.compute(-5_000, -1_000, status="single")
    check(neg["total_federal_tax"] == 0, "negative inputs are floored, not crashed")

    print("\n" + "=" * 78)
    print(f"{N[0] - len(FAIL)}/{N[0]} checks passed")
    if FAIL:
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
