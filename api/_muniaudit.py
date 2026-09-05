"""Benchmark an advisor's municipal bond fills against the MSRB tape.

The measurement this implements is the one validated in research/FINDINGS.md:
compare a customer's fill to what OTHER customers paid or received for the
SAME bond on the SAME day, on the SAME side. Same-bond/same-day cancels
coupon, maturity, credit and the day's rate move, so what is left is
execution.

Three rules keep the output honest, and each exists because the alternative
is a number that looks authoritative and is not:

  1. NO BENCHMARK, NO VERDICT. If the tape has no comparable print for that
     bond and day, the trade is returned as "not assessable". It is never
     compared to a nearby day, an interpolated curve or a matrix price —
     those are exactly the fictions this product exists to replace.

  2. THE INSTITUTIONAL PRINT IS THE YARDSTICK, WHERE ONE EXISTS. That is
     what the validation measured (0.156 points after the intraday-drift
     control). Where no institutional print exists, the comparison falls
     back to the day's customer prints and SAYS SO, because a retail-only
     benchmark measures something weaker.

  3. DIRECTION IS EXPLICIT. A buyer is hurt by paying more; a seller is hurt
     by receiving less. Both are reported as a positive "cost" so a portfolio
     total means something. Getting this backwards would show a bad sale as a
     win, which is the single easiest way to discredit the whole report.

Standard library plus pandas only; no network. The caller supplies prints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RETAIL_MAX = 100_000
INSTIT_MIN = 1_000_000
# 1 point of par = 1% of face. On $100,000 that is $1,000 — the conversion
# that was wrong by 100x in the first draft of the research script.
POINT_VALUE_PER_PAR = 0.01


@dataclass
class Trade:
    cusip: str
    date: str
    side: str          # "buy" or "sell", from the CLIENT's perspective
    par: float
    price: float
    description: str = ""

    def normalised_side(self) -> str:
        s = (self.side or "").strip().lower()
        if s in ("buy", "b", "bought", "purchase", "purchased"):
            return "buy"
        if s in ("sell", "s", "sold", "sale"):
            return "sell"
        raise ValueError(f"side must be buy or sell, got {self.side!r}")


@dataclass
class Verdict:
    trade: Trade
    assessable: bool
    reason: str = ""
    benchmark: float | None = None
    benchmark_kind: str = ""          # "institutional" or "customer"
    cost_points: float | None = None  # positive = client did worse
    cost_dollars: float | None = None
    percentile: float | None = None   # where the fill sat among same-side prints
    n_prints: int = 0
    day_low: float | None = None
    day_high: float | None = None
    notes: list = field(default_factory=list)


def _par_weighted(prints: list[dict]) -> float | None:
    num = sum(p["price"] * p["par"] for p in prints if p.get("par"))
    den = sum(p["par"] for p in prints if p.get("par"))
    return (num / den) if den else None


def _emma_side_for(client_side: str) -> str:
    """EMMA encodes the DEALER's action.

    A client BUYING is filled by a dealer selling to a customer -> 'S'.
    A client SELLING is filled by a dealer purchasing from a customer -> 'P'.
    Comparing a client's buy against 'P' prints would compare it to the wrong
    side of the spread and manufacture a penalty that is not there.
    """
    return "S" if client_side == "buy" else "P"


def audit_trade(trade: Trade, day_prints: list[dict]) -> Verdict:
    """`day_prints`: [{price, par, side}] for this bond on this trade date."""
    try:
        side = trade.normalised_side()
    except ValueError as e:
        return Verdict(trade, False, reason=str(e))
    if trade.par <= 0 or trade.price <= 0:
        return Verdict(trade, False, reason="trade needs a positive par and price")

    want = _emma_side_for(side)
    same = [p for p in day_prints
            if p.get("side") == want and p.get("price") and p.get("par")]
    if not same:
        return Verdict(trade, False, n_prints=len(day_prints),
                       reason="no comparable customer print on the tape that "
                              "day — not assessable, and deliberately not "
                              "estimated from a nearby day")

    prices = sorted(p["price"] for p in same)
    v = Verdict(trade, True, n_prints=len(same),
                day_low=prices[0], day_high=prices[-1])

    instit = [p for p in same if p["par"] >= INSTIT_MIN]
    if instit:
        v.benchmark = _par_weighted(instit)
        v.benchmark_kind = "institutional"
    else:
        v.benchmark = _par_weighted(same)
        v.benchmark_kind = "customer"
        v.notes.append(
            "No institutional-size print that day, so this compares against "
            "other customer trades. That is a weaker yardstick and usually "
            "understates the true cost.")

    if v.benchmark is None:
        return Verdict(trade, False, reason="could not form a benchmark")

    # positive = the client did worse than the benchmark
    diff = (trade.price - v.benchmark) if side == "buy" \
        else (v.benchmark - trade.price)
    v.cost_points = diff
    v.cost_dollars = diff * POINT_VALUE_PER_PAR * trade.par

    # How many other customers did BETTER than this client that day: paid
    # less if buying, received more if selling. A high percentile is a bad
    # fill, which is the direction the note below reads in.
    if side == "buy":
        better = sum(1 for p in prices if p < trade.price)
    else:
        better = sum(1 for p in prices if p > trade.price)
    v.percentile = 100.0 * better / len(prices)
    if side == "buy":
        v.notes.append(f"{v.percentile:.0f}% of same-day customer buyers paid "
                       f"less than you did.")
    else:
        v.notes.append(f"{v.percentile:.0f}% of same-day customer sellers "
                       f"received more than you did.")
    return v


def audit_portfolio(verdicts: list[Verdict]) -> dict:
    """Totals across a client's trades. Only assessable trades are counted,
    and the unassessable ones are reported rather than silently dropped."""
    ok = [v for v in verdicts if v.assessable and v.cost_dollars is not None]
    skipped = [v for v in verdicts if not v.assessable]
    total_par = sum(v.trade.par for v in ok)
    total_cost = sum(v.cost_dollars for v in ok)
    worse = [v for v in ok if v.cost_dollars > 0]
    inst = [v for v in ok if v.benchmark_kind == "institutional"]
    return {
        "trades_submitted": len(verdicts),
        "trades_assessed": len(ok),
        "trades_not_assessable": len(skipped),
        "par_assessed": total_par,
        "total_cost_dollars": total_cost,
        "cost_as_pct_of_par": (total_cost / total_par * 100) if total_par else None,
        "trades_worse_than_benchmark": len(worse),
        "share_worse": (len(worse) / len(ok)) if ok else None,
        "worst": max(ok, key=lambda v: v.cost_dollars) if ok else None,
        "benchmarked_institutionally": len(inst),
        "median_cost_points": (
            sorted(v.cost_points for v in ok)[len(ok) // 2] if ok else None),
    }


def to_dict(v: Verdict) -> dict:
    return {
        "cusip": v.trade.cusip, "date": v.trade.date, "side": v.trade.side,
        "par": v.trade.par, "price": v.trade.price,
        "description": v.trade.description,
        "assessable": v.assessable, "reason": v.reason,
        "benchmark": round(v.benchmark, 4) if v.benchmark is not None else None,
        "benchmark_kind": v.benchmark_kind,
        "cost_points": round(v.cost_points, 4) if v.cost_points is not None else None,
        "cost_dollars": round(v.cost_dollars, 2) if v.cost_dollars is not None else None,
        "percentile": round(v.percentile, 1) if v.percentile is not None else None,
        "n_prints": v.n_prints,
        "day_low": v.day_low, "day_high": v.day_high,
        "notes": v.notes,
    }
