"""US federal tax engine for ISO / AMT planning. Pure arithmetic, no network.

Correctness is the entire product here, so this module is deliberately boring:
every constant is sourced and dated, every rule is written out rather than
folded into a magic number, and the accompanying tests hand-check the results
against arithmetic anyone can redo on paper.

2026 constants are from IRS Rev. Proc. 2025-32, which incorporates the changes
made by P.L. 119-21 (the One Big Beautiful Bill Act). OBBBA matters a great
deal here and is why most ISO calculators on the web are now wrong: from 2026
the AMT exemption phaseout reverts to $500,000 / $1,000,000 AND the phaseout
rate doubles from 25c to 50c per dollar. That doubling creates a band where
each extra dollar of AMT income costs 39-42 cents rather than 26-28.

SCOPE — what this does NOT model, and must say so plainly to the user:
  * state income tax or state AMT (California's own AMT is a large omission
    for the exact people this is built for)
  * AMT credit carryforward from prior years (which often recovers AMT paid
    on ISOs in later years)
  * Net Investment Income Tax (3.8%)
  * QSBS exclusions, disqualifying dispositions, 83(b) elections
  * phase-outs of other credits and deductions, or any AMT preference item
    other than the ISO bargain element

It is a planning tool for the ISO/AMT decision, not a tax return.
"""
from __future__ import annotations

SOURCE = ("IRS Rev. Proc. 2025-32 (2026 inflation adjustments, incorporating "
          "P.L. 119-21 / OBBBA)")
TAX_YEAR = 2026

# (upper bound of bracket, marginal rate). None = no upper bound.
ORDINARY_BRACKETS = {
    "single": [(12_400, 0.10), (50_400, 0.12), (105_700, 0.22),
               (201_775, 0.24), (256_225, 0.32), (640_600, 0.35), (None, 0.37)],
    "mfj": [(24_800, 0.10), (100_800, 0.12), (211_400, 0.22),
            (403_550, 0.24), (512_450, 0.32), (768_700, 0.35), (None, 0.37)],
}

STANDARD_DEDUCTION = {"single": 16_100, "mfj": 32_200}

# Long-term capital gains / qualified dividends: (upper bound of taxable
# income at which this rate applies, rate)
LTCG_BRACKETS = {
    "single": [(49_450, 0.00), (545_500, 0.15), (None, 0.20)],
    "mfj": [(98_900, 0.00), (613_700, 0.15), (None, 0.20)],
}

AMT_EXEMPTION = {"single": 90_100, "mfj": 140_200}
AMT_PHASEOUT_START = {"single": 500_000, "mfj": 1_000_000}
AMT_PHASEOUT_RATE = 0.50            # OBBBA: doubled from 0.25 for 2026
AMT_RATE_LOW, AMT_RATE_HIGH = 0.26, 0.28
AMT_28_THRESHOLD = 244_500          # applies to all filing statuses

FILING_STATUSES = ("single", "mfj")


def _stack_tax(amount: float, brackets, start_at: float = 0.0) -> float:
    """Progressive tax on `amount`, entering the schedule at `start_at`.

    `start_at` lets capital gains be stacked ON TOP of ordinary income, which
    is how the code actually works — gains are taxed in the brackets that
    remain above the ordinary income, not from zero.
    """
    if amount <= 0:
        return 0.0
    tax = 0.0
    lower = 0.0
    remaining = amount
    for upper, rate in brackets:
        cap = float("inf") if upper is None else float(upper)
        # portion of this bracket that sits above start_at
        band_lo = max(lower, start_at)
        band_hi = cap
        if band_hi > band_lo:
            taxable_here = min(remaining, band_hi - band_lo)
            if taxable_here > 0:
                tax += taxable_here * rate
                remaining -= taxable_here
                if remaining <= 1e-9:
                    break
        lower = cap
        if lower == float("inf"):
            break
    return tax


def regular_tax(ordinary_income: float, ltcg: float, deduction: float,
                status: str) -> dict:
    """Regular federal income tax, with capital gains stacked above ordinary
    income and taxed at preferential rates."""
    brackets = ORDINARY_BRACKETS[status]
    ltcg_brackets = LTCG_BRACKETS[status]
    total_income = max(0.0, ordinary_income) + max(0.0, ltcg)
    taxable = max(0.0, total_income - max(0.0, deduction))
    # the deduction is applied against ordinary income first
    ord_taxable = max(0.0, min(taxable, max(0.0, ordinary_income) - max(0.0, deduction)))
    gain_taxable = max(0.0, taxable - ord_taxable)

    tax_ord = _stack_tax(ord_taxable, brackets)
    tax_gain = _stack_tax(gain_taxable, ltcg_brackets, start_at=ord_taxable)
    return {"taxable_income": taxable, "ordinary_taxable": ord_taxable,
            "gain_taxable": gain_taxable, "tax": tax_ord + tax_gain,
            "tax_ordinary": tax_ord, "tax_gains": tax_gain}


def amt_exemption(amti: float, status: str) -> float:
    """Exemption after the OBBBA phaseout (50c per dollar from 2026)."""
    base = AMT_EXEMPTION[status]
    start = AMT_PHASEOUT_START[status]
    if amti <= start:
        return float(base)
    return max(0.0, base - AMT_PHASEOUT_RATE * (amti - start))


def tentative_minimum_tax(amti: float, ltcg: float, status: str) -> dict:
    """TMT on a given AMTI. Capital gains keep their preferential rates
    inside the AMT calculation too, so they are carved out of the 26/28%
    base rather than taxed at AMT rates."""
    exemption = amt_exemption(amti, status)
    base = max(0.0, amti - exemption)
    gains = max(0.0, min(ltcg, base))
    ordinary_base = max(0.0, base - gains)

    low = min(ordinary_base, AMT_28_THRESHOLD)
    high = max(0.0, ordinary_base - AMT_28_THRESHOLD)
    tax = low * AMT_RATE_LOW + high * AMT_RATE_HIGH
    # gains sit above the ordinary AMT base in the capital-gains schedule
    tax += _stack_tax(gains, LTCG_BRACKETS[status], start_at=ordinary_base)
    return {"exemption": exemption, "amt_base": base, "tmt": tax,
            "gains_in_base": gains}


def compute(ordinary_income: float, bargain_element: float, *,
            status: str = "single", ltcg: float = 0.0,
            deduction: float | None = None,
            itemized: bool = False) -> dict:
    """Full picture for one scenario.

    `bargain_element` is (FMV at exercise - strike) x shares exercised and
    held past year end. AMTI adds it to income; regular tax ignores it.
    """
    if status not in FILING_STATUSES:
        raise ValueError(f"status must be one of {FILING_STATUSES}")
    ordinary_income = max(0.0, float(ordinary_income))
    bargain_element = max(0.0, float(bargain_element))
    ltcg = max(0.0, float(ltcg))
    if deduction is None:
        deduction = STANDARD_DEDUCTION[status]
    deduction = max(0.0, float(deduction))

    reg = regular_tax(ordinary_income, ltcg, deduction, status)

    # AMTI: start from income, then remove only deductions the AMT allows.
    # The standard deduction is not allowed at all; itemized deductions are
    # allowed here except SALT, which this tool does not attempt to split
    # out — so an itemized filer should enter their AMT-allowable amount.
    amt_deduction = deduction if itemized else 0.0
    amti = ordinary_income + ltcg + bargain_element - amt_deduction
    amti = max(0.0, amti)

    tmt = tentative_minimum_tax(amti, ltcg, status)
    amt_owed = max(0.0, tmt["tmt"] - reg["tax"])

    return {
        "status": status, "tax_year": TAX_YEAR,
        "ordinary_income": ordinary_income, "ltcg": ltcg,
        "bargain_element": bargain_element,
        "deduction": deduction, "itemized": itemized,
        "regular_taxable_income": reg["taxable_income"],
        "regular_tax": reg["tax"],
        "amti": amti,
        "amt_exemption": tmt["exemption"],
        "amt_base": tmt["amt_base"],
        "tentative_minimum_tax": tmt["tmt"],
        "amt_owed": amt_owed,
        "total_federal_tax": reg["tax"] + amt_owed,
    }


def crossover_bargain_element(ordinary_income: float, *, status: str = "single",
                              ltcg: float = 0.0, deduction: float | None = None,
                              itemized: bool = False,
                              tolerance: float = 1.0) -> float:
    """Largest bargain element that still owes ZERO additional AMT.

    Solved by bisection rather than algebra: the exemption phaseout and the
    26/28% break make the function piecewise linear, and a search is both
    simpler to reason about and impossible to get subtly wrong when the
    brackets change next year. AMT owed is non-decreasing in the bargain
    element, so bisection is valid.
    """
    kw = dict(status=status, ltcg=ltcg, deduction=deduction, itemized=itemized)
    if compute(ordinary_income, 0.0, **kw)["amt_owed"] > 0:
        return 0.0                      # already in AMT before any exercise
    lo, hi = 0.0, 1_000.0
    # expand until AMT appears
    while compute(ordinary_income, hi, **kw)["amt_owed"] <= 0:
        lo = hi
        hi *= 2
        if hi > 1e9:
            return lo
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        if compute(ordinary_income, mid, **kw)["amt_owed"] > 0:
            hi = mid
        else:
            lo = mid
    return lo


def marginal_amt_rate(ordinary_income: float, bargain_element: float,
                      *, status: str = "single", step: float = 1_000.0,
                      **kw) -> float:
    """Cost of the NEXT dollar of bargain element. Inside the OBBBA phaseout
    band this exceeds the headline 26/28% — that band is the trap worth
    showing people."""
    a = compute(ordinary_income, bargain_element, status=status, **kw)
    b = compute(ordinary_income, bargain_element + step, status=status, **kw)
    return (b["amt_owed"] - a["amt_owed"]) / step


def plan(shares: int, strike: float, fmv: float, ordinary_income: float, *,
         status: str = "single", ltcg: float = 0.0,
         deduction: float | None = None, itemized: bool = False,
         max_years: int = 10) -> dict:
    """The question people actually have: how many can I exercise now with no
    AMT, what does exercising everything cost, and how long to do it all
    AMT-free at this income?"""
    spread = max(0.0, float(fmv) - float(strike))
    shares = max(0, int(shares))
    kw = dict(status=status, ltcg=ltcg, deduction=deduction, itemized=itemized)

    cross_be = crossover_bargain_element(ordinary_income, **kw)
    free_shares = int(cross_be // spread) if spread > 0 else shares
    free_shares = min(free_shares, shares)

    all_be = spread * shares
    all_in = compute(ordinary_income, all_be, **kw)
    at_free = compute(ordinary_income, spread * free_shares, **kw)
    baseline = compute(ordinary_income, 0.0, **kw)

    years = None
    if spread > 0 and free_shares > 0:
        years = -(-shares // free_shares)          # ceiling division
        if years > max_years:
            years = max_years
    return {
        "spread_per_share": spread,
        "shares_total": shares,
        "crossover_bargain_element": cross_be,
        "amt_free_shares": free_shares,
        "amt_free_exercise_cost": free_shares * float(strike),
        "amt_free_bargain_element": free_shares * spread,
        "baseline_tax": baseline["total_federal_tax"],
        "tax_at_amt_free": at_free["total_federal_tax"],
        "exercise_all": {
            "bargain_element": all_be,
            "exercise_cost": shares * float(strike),
            "amt_owed": all_in["amt_owed"],
            "total_federal_tax": all_in["total_federal_tax"],
            "marginal_amt_rate": marginal_amt_rate(
                ordinary_income, all_be, **kw),
        },
        "years_to_exercise_all_amt_free": years,
        "source": SOURCE, "tax_year": TAX_YEAR,
    }


def ladder(shares: int, strike: float, fmv: float, ordinary_income: float,
           *, points: int = 8, **kw) -> list[dict]:
    """A few sample exercise sizes, for a table the user can scan."""
    shares = max(0, int(shares))
    spread = max(0.0, float(fmv) - float(strike))
    out = []
    if shares == 0 or spread <= 0:
        return out
    for i in range(1, points + 1):
        n = max(1, round(shares * i / points))
        r = compute(ordinary_income, spread * n, **kw)
        out.append({"shares": n, "bargain_element": spread * n,
                    "exercise_cost": n * float(strike),
                    "amt_owed": r["amt_owed"],
                    "total_federal_tax": r["total_federal_tax"]})
    return out
