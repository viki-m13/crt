"""Muni dislocation scanner and the bid price to place.

Implements the KEYSTONE-XL rules exactly as validated in viki-m13/bonds
(munis/research/FINDINGS.md). The definitions below are copied from that
work rather than reinvented — an approximation would not inherit its
validation, which is the only reason any of these numbers mean anything.

THE RULES, verbatim from the validated strategy:

  mid            a signal-only reference price: the inter-dealer print if
                 there is one, else the midpoint of the day's customer-buy
                 and customer-sell prints, else whichever side exists. It is
                 NOT an executable price and must never be shown as one.
  trailing       60-calendar-day rolling median of mid, shifted one day so
                 today's print cannot inform today's signal.
  dislocation    a customer-buy print >= 3.0 points BELOW that trailing
                 median. Below 3 points the edge does not survive the
                 matched-random control (2pt: +0.68%, 3pt: +1.99%).
  LIMIT          pay no more than the latest prior mid + 0.25 points.
  liquidity      the bond must have printed on >= 8 distinct days in the
                 trailing 90, checked with no forward information.

The limit cap is the part worth understanding before selling anything built
on it: it was designed and validated ENTIRELY on corporate bonds and carried
to munis unchanged, so for munis it is an out-of-sample rule. It is also
where most of the edge lives — adding it roughly doubles in-sample CAGR
(+4.35% to +7.84%) and halves drawdown. Knowing which bond to buy is worth
much less than knowing what to pay for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DISCOUNT_PTS = 3.0        # locked: the level that survives the control
LIMIT_CAP = 0.25          # locked: transferred from corporates, unchanged
MIN_ACTIVE_DAYS = 8       # liquidity gate, trailing 90 calendar days
LOOKBACK_DAYS = 90
MEDIAN_WINDOW_DAYS = 60
MIN_MEDIAN_OBS = 5


@dataclass
class DayPrints:
    """One bond, one date, already aggregated from the tape."""
    date: str
    buy: float | None = None       # customer bought (EMMA side S)
    sell: float | None = None      # customer sold (EMMA side P)
    dealer: float | None = None    # inter-dealer (EMMA side D)
    buy_par: float = 0.0
    sell_par: float = 0.0

    @property
    def mid(self) -> float | None:
        """Signal reference only — never quote this as tradeable."""
        if self.dealer is not None:
            return self.dealer
        if self.buy is not None and self.sell is not None:
            return (self.buy + self.sell) / 2.0
        return self.buy if self.buy is not None else self.sell


@dataclass
class Signal:
    security_id: str
    date: str
    description: str = ""
    dislocated: bool = False
    reason: str = ""
    buy_price: float | None = None
    trailing_median: float | None = None
    discount_pts: float | None = None
    prior_mid: float | None = None
    limit_price: float | None = None      # the number the user actually wants
    active_days: int = 0
    notes: list = field(default_factory=list)


def _days_between(a: str, b: str) -> int:
    import datetime as dt
    return abs((dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days)


def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def evaluate(security_id: str, history: list[DayPrints], asof: str,
             description: str = "") -> Signal:
    """Assess one bond as of `asof`. `history` may include `asof` itself;
    everything used for the signal is strictly BEFORE it."""
    s = Signal(security_id=security_id, date=asof, description=description)
    hist = sorted(history, key=lambda d: d.date)
    prior = [d for d in hist if d.date < asof]
    today = next((d for d in hist if d.date == asof), None)

    if today is None or today.buy is None:
        s.reason = "no customer-buy print today — nothing to act on"
        return s
    s.buy_price = today.buy

    # liquidity gate, trailing 90 days, no forward information
    recent = [d for d in prior if _days_between(d.date, asof) <= LOOKBACK_DAYS]
    s.active_days = len({d.date for d in recent})
    if s.active_days < MIN_ACTIVE_DAYS:
        s.reason = (f"too illiquid — printed on {s.active_days} days in the "
                    f"trailing {LOOKBACK_DAYS}, needs {MIN_ACTIVE_DAYS}")
        return s

    window = [d.mid for d in prior
              if _days_between(d.date, asof) <= MEDIAN_WINDOW_DAYS
              and d.mid is not None]
    if len(window) < MIN_MEDIAN_OBS:
        s.reason = (f"not enough recent marks to form a trend "
                    f"({len(window)} of {MIN_MEDIAN_OBS} needed)")
        return s
    s.trailing_median = _median(window)
    s.discount_pts = s.trailing_median - today.buy

    # the latest prior mid is what the limit is set from
    prior_with_mid = [d for d in prior if d.mid is not None]
    if prior_with_mid:
        s.prior_mid = prior_with_mid[-1].mid
        s.limit_price = s.prior_mid + LIMIT_CAP

    if s.discount_pts < DISCOUNT_PTS:
        s.reason = (f"only {s.discount_pts:.2f} points below its own trend; "
                    f"the edge does not survive the control below "
                    f"{DISCOUNT_PTS:.0f}")
        return s

    s.dislocated = True
    s.reason = (f"printed {s.discount_pts:.2f} points below its trailing "
                f"{MEDIAN_WINDOW_DAYS}-day median")
    if s.limit_price is not None:
        if today.buy <= s.limit_price:
            s.notes.append(
                f"Today's customer-buy of {today.buy:.3f} is inside the "
                f"{s.limit_price:.3f} limit — this is the tradeable case.")
        else:
            s.notes.append(
                f"Today's customer-buy of {today.buy:.3f} is ABOVE the "
                f"{s.limit_price:.3f} limit. The validated rule declines this "
                f"trade rather than chasing it; paying up is where the edge "
                f"was lost in testing.")
    return s


def price_verdict(quoted: float, s: Signal) -> dict:
    """What to do about a price you are being shown right now.

    This is the question an advisor actually asks — not "is this bond
    interesting" but "they want 102.40, do I hit it?"
    """
    out = {"quoted": quoted, "limit": s.limit_price,
           "trailing_median": s.trailing_median}
    if s.limit_price is None:
        out["verdict"] = "unknown"
        out["message"] = ("No recent mid on the tape for this bond, so there "
                          "is no defensible reference price. Decline or ask "
                          "for comparable prints.")
        return out
    gap = quoted - s.limit_price
    out["over_limit_pts"] = gap
    out["over_limit_per_100k"] = gap * 1000.0
    if quoted <= s.limit_price:
        out["verdict"] = "within limit"
        out["message"] = (f"At {quoted:.3f} you are inside the {s.limit_price:.3f} "
                          f"limit set by the last mid plus {LIMIT_CAP}.")
    else:
        out["verdict"] = "too rich"
        out["message"] = (f"At {quoted:.3f} you are {gap:.3f} points above the "
                          f"{s.limit_price:.3f} limit — about "
                          f"${gap*1000:,.0f} per $100,000 of par. The tested "
                          f"rule does not pay this.")
    if s.trailing_median is not None:
        out["vs_trend_pts"] = s.trailing_median - quoted
    return out


def scan(bonds: dict, asof: str) -> list[Signal]:
    """bonds: {security_id: (description, [DayPrints])}. Returns only the
    dislocated ones, cheapest first. An empty list is a real answer — this
    strategy is dormant unless somebody is forced to sell."""
    hits = []
    for six, (desc, hist) in bonds.items():
        s = evaluate(six, hist, asof, desc)
        if s.dislocated:
            hits.append(s)
    hits.sort(key=lambda s: -(s.discount_pts or 0))
    return hits
