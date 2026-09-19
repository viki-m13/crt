"""AHE: Adaptive Hard-Evidence Horizon selector.

Research-only core. It does not create events; upstream event extractors must provide
causal, timestamped records. AHE decides whether an event family has enough matured
historical evidence to issue a >95% endpoint-direction forecast and which >=30-session
horizon to use. Losing forecasts are never extended.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

HORIZONS = (30, 60, 90, 126, 180, 252)
TARGET = 0.95


@dataclass(frozen=True)
class Event:
    event_id: str
    ticker: str
    family: str
    issue_i: int
    reference_price: float
    strength: float = 1.0


@dataclass(frozen=True)
class Outcome:
    event_id: str
    horizon: int
    exit_i: int
    resolved: bool
    higher: bool | None


@dataclass(frozen=True)
class Decision:
    ticker: str
    family: str
    horizon: int | None
    reason: str
    observed_precision: float | None
    effective_blocks: int


def wilson_lower(successes: int, n: int, z: float = 2.576) -> float:
    """99% one-sided-ish conservative screen; clustering handled outside."""
    if n <= 0:
        return 0.0
    p = successes / n
    den = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / den)


def matured_family_history(events: Iterable[Event], outcomes: Iterable[Outcome],
                            family: str, asof_i: int, horizon: int):
    by_id = {e.event_id: e for e in events if e.family == family}
    rows = []
    for o in outcomes:
        if o.horizon != horizon or o.event_id not in by_id or o.exit_i >= asof_i:
            continue
        # Missing matured outcomes count as failures for the precision screen.
        rows.append((by_id[o.event_id], bool(o.higher) if o.resolved else False))
    return rows


def independent_blocks(rows, horizon: int) -> tuple[int, int, float]:
    """Collapse correlated event rows into coarse time blocks before certification."""
    if not rows:
        return 0, 0, 0.0
    buckets: dict[int, list[bool]] = {}
    for e, y in rows:
        buckets.setdefault(e.issue_i // (2 * horizon), []).append(y)
    # A block only counts successful if every forecast in it succeeded. This is
    # intentionally harsher than treating same-date stocks as independent trials.
    block_y = [all(v) for v in buckets.values()]
    successes = sum(block_y)
    n = len(block_y)
    return successes, n, successes / n if n else 0.0


def choose_horizon(event: Event, events: Iterable[Event], outcomes: Iterable[Outcome],
                   *, minimum_blocks: int = 20, target: float = TARGET) -> Decision:
    """Earliest horizon that has already demonstrated >target precision robustly.

    The rule is causal: only outcomes with exit_i < event.issue_i are visible. If no
    horizon clears the gate, AHE abstains. It never moves an existing deadline.
    """
    best = None
    diagnostics = []
    for h in HORIZONS:
        rows = matured_family_history(events, outcomes, event.family, event.issue_i, h)
        successes, blocks, precision = independent_blocks(rows, h)
        lower = wilson_lower(successes, blocks)
        diagnostics.append((h, precision, blocks, lower))
        if blocks >= minimum_blocks and precision > target and lower > target:
            best = (h, precision, blocks)
            break
    if best is None:
        detail = "; ".join(f"{h}d p={p:.3f} blocks={b} lb={lb:.3f}"
                           for h, p, b, lb in diagnostics)
        return Decision(event.ticker, event.family, None, f"ABSTAIN: {detail}", None, 0)
    h, precision, blocks = best
    return Decision(event.ticker, event.family, h,
                    "ISSUE: earliest matured horizon clearing the 95% evidence gate",
                    precision, blocks)


def rank_candidates(candidates: Iterable[tuple[Event, Decision]]):
    """One stock only: prioritize stronger evidence, then shorter horizon, then event strength."""
    valid = [(e, d) for e, d in candidates if d.horizon is not None]
    return sorted(valid, key=lambda x: (-x[1].observed_precision,
                                        -x[1].effective_blocks,
                                        x[1].horizon,
                                        -x[0].strength,
                                        x[0].ticker))
