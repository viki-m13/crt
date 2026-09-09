"""Analog matching: find past stretches that look like now, show what followed.

The technique is old and the chart is familiar — normalise the last L days of
a stock to 100, sweep history for the windows whose shape matches, and plot
what each of those did over the following H days. What makes it honest or
dishonest is entirely in the details, and there are four that decide it:

1.  MATCHES MUST PRE-DATE THE OUTCOME THEY CLAIM, AND MUST NOT OVERLAP THE
    QUERY. A window ending at `s` is only usable at time `t` if `s + H <= t`
    — otherwise the analog's own future is inside the period we are
    pretending to forecast. Within the query's own series that is not
    sufficient: a match ending exactly H bars ago has a "future" that IS the
    right half of the window we are matching on, so the chart would draw
    recent history as its own sequel. The rule is therefore the stricter
    `s + H < t - L`: the analog and everything that followed it must be over
    before the query window even begins.

    When the library holds series other than the query, admissibility can
    only be judged by DATE, because bar indices in another series mean
    nothing in ours. If `as_of_index` is set and `dates` was not supplied,
    those series are skipped entirely rather than waved through — the gate
    fails closed, and `Forecast.skipped_unverifiable` records it.

2.  MATCHES MUST NOT OVERLAP EACH OTHER. Adjacent windows are nearly the same
    window. Take the top five by raw distance and you get one analog counted
    five times, an agreement score of ~1.0, and a confidence reading that is
    pure double-counting. `_dedupe` keeps matches at least `min_gap` apart.

3.  SHAPE, NOT LEVEL. Matching raw prices matches expensive stocks to
    expensive stocks. We match the normalised cumulative log-return path,
    divided by the window's own volatility, so a quiet mega-cap can match a
    volatile small-cap when the *pattern* is the same.

4.  THE FORECAST IS A DISTRIBUTION, NOT A LINE. Analogs disagree, and how
    much they disagree is the only thing here that carries information about
    reliability. We return every path plus the spread, never a single number
    dressed up as a prediction.

What this module does NOT do is claim accuracy. Measured out-of-sample
performance lives in research/analog/ and is reported to the user verbatim,
including when it is unimpressive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

LOOKBACK = 120          # trading days of shape we match on (~6 months)
HORIZON = 60            # trading days forward we show (~3 months)
TOP_K = 5               # analogs returned, matching the reference chart
MIN_GAP = 60            # trading days two matches must be apart


@dataclass
class Analog:
    """One historical window that resembles the query, and what followed."""
    symbol: str
    start: str                       # date of the first bar of the match
    end: str                         # date of the last bar of the match
    distance: float                  # lower is a closer match
    correlation: float               # of the two normalised cumulative paths
    path: list[float] = field(default_factory=list)      # matched window, =100
    forward: list[float] = field(default_factory=list)   # what followed, =100
    forward_return: float = 0.0      # total return over the forward window


@dataclass
class Forecast:
    """The analogs and the distribution they imply. Not a prediction."""
    symbol: str
    as_of: str
    lookback: int
    horizon: int
    query_path: list[float] = field(default_factory=list)
    analogs: list[Analog] = field(default_factory=list)
    median_path: list[float] = field(default_factory=list)
    low_path: list[float] = field(default_factory=list)
    high_path: list[float] = field(default_factory=list)
    agreement: float = 0.0           # share of analogs agreeing on direction
    median_return: float = 0.0
    searched: int = 0
    skipped_unverifiable: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.analogs)


# ---------------------------------------------------------------- shape ----

def _log_returns(closes) -> list[float]:
    out = []
    for a, b in zip(closes, closes[1:]):
        if a and b and a > 0 and b > 0:
            out.append(math.log(b / a))
        else:
            out.append(float("nan"))
    return out


def _shape(rets) -> list[float] | None:
    """Cumulative path of a return window, scaled by its own volatility.

    Dividing by the window's standard deviation is what lets a 12%-vol name
    match a 45%-vol name on pattern rather than on amplitude. A window with
    no variation has no shape and is rejected rather than divided by zero.
    """
    if any(r != r for r in rets):          # NaN
        return None
    n = len(rets)
    if n < 2:
        return None
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return None
    out, acc = [], 0.0
    for r in rets:
        acc += r / sd
        out.append(acc)
    return out


def _distance(a: list[float], b: list[float]) -> float:
    """Root mean squared gap between two normalised paths."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def _rebase(closes) -> list[float]:
    """Index a price series to 100 at its first bar."""
    base = closes[0]
    if not base:
        return [100.0] * len(closes)
    return [100.0 * c / base for c in closes]


def _dedupe(hits: list[tuple], min_gap: int) -> list[tuple]:
    """Keep the best match, then only matches far enough from every keeper.

    Without this the top five are five shifts of one window: the same event
    counted five times, which inflates agreement and makes the confidence
    number meaningless.
    """
    kept: list[tuple] = []
    for h in sorted(hits, key=lambda x: x[0]):
        d, sym, i = h[0], h[1], h[2]
        if all(not (sym == k[1] and abs(i - k[2]) < min_gap) for k in kept):
            kept.append(h)
    return kept


# ---------------------------------------------------------------- search ---

def find_analogs(query_closes, library, *, as_of_index: int | None = None,
                 lookback: int = LOOKBACK, horizon: int = HORIZON,
                 top_k: int = TOP_K, min_gap: int = MIN_GAP,
                 symbol: str = "", dates=None) -> Forecast:
    """Find the windows in `library` most like the last `lookback` bars.

    library : {symbol: (dates, closes)} — every series to search, including
              the query's own history.
    as_of_index : index into the QUERY series treated as "today". Bars after
              it are invisible. Validation passes this; live use leaves it
              None, which means the end of the series.

    The look-ahead rule is enforced here and only here: a candidate window
    ending at position j in a library series is admissible only if the whole
    of its forward window also lies in the past relative to `as_of`. For the
    query's own series that is a position test; for other symbols it is a
    date test, because their indices mean nothing to ours.
    """
    dates = list(dates or [])
    closes = list(query_closes)
    end = len(closes) if as_of_index is None else int(as_of_index) + 1
    if end > len(closes):
        end = len(closes)
    hist = closes[:end]
    if len(hist) < lookback + 1:
        return Forecast(symbol=symbol, as_of="", lookback=lookback,
                        horizon=horizon,
                        reason=f"needs {lookback + 1} bars, has {len(hist)}")

    as_of_date = dates[end - 1] if len(dates) >= end else ""
    q_rets = _log_returns(hist[-(lookback + 1):])
    q_shape = _shape(q_rets)
    if q_shape is None:
        return Forecast(symbol=symbol, as_of=as_of_date, lookback=lookback,
                        horizon=horizon, reason="query window is flat or has gaps")

    hits, searched, skipped = [], 0, []
    for sym, (sym_dates, sym_closes) in library.items():
        sym_dates = list(sym_dates)
        sym_closes = list(sym_closes)
        # A foreign series can only be gated by date. Without dates we cannot
        # prove a window predates as_of, so we refuse to use it at all.
        if sym != symbol and as_of_index is not None and not as_of_date:
            skipped.append(sym)
            continue
        rets = _log_returns(sym_closes)
        n = len(sym_closes)
        # j indexes the LAST bar of a candidate match window
        for j in range(lookback, n - horizon):
            # --- the admissibility gate -----------------------------------
            if sym == symbol:
                # the match AND its sequel must finish before the query
                # window opens, so a match never explains itself
                if j + horizon >= end - 1 - lookback:
                    break
            elif as_of_date and sym_dates[j + horizon] >= as_of_date:
                # different series: compare by date, indices are not aligned
                break
            searched += 1
            sh = _shape(rets[j - lookback:j])
            if sh is None:
                continue
            # only the key is retained: holding every candidate's full
            # `lookback`-long shape costs ~170 MB at the permitted maximum
            # and times the request out. The top few are recomputed below.
            hits.append((_distance(q_shape, sh), sym, j))

    if not hits:
        return Forecast(symbol=symbol, as_of=as_of_date, lookback=lookback,
                        horizon=horizon, searched=searched,
                        skipped_unverifiable=skipped,
                        reason="no admissible history to match against")

    out: list[Analog] = []
    for d, sym, j in _dedupe(hits, min_gap)[:top_k]:
        sym_dates, sym_closes = library[sym]
        s = _shape(_log_returns(sym_closes)[j - lookback:j])
        window = list(sym_closes[j - lookback:j + 1])
        fwd = list(sym_closes[j:j + horizon + 1])
        out.append(Analog(
            symbol=sym,
            start=str(sym_dates[j - lookback]), end=str(sym_dates[j]),
            distance=round(d, 4),
            correlation=round(_corr(q_shape, s), 4),
            path=[round(x, 3) for x in _rebase(window)],
            forward=[round(x, 3) for x in _rebase(fwd)],
            forward_return=round(fwd[-1] / fwd[0] - 1.0, 5) if fwd[0] else 0.0))

    fc = Forecast(symbol=symbol, as_of=as_of_date, lookback=lookback,
                  horizon=horizon, searched=searched, analogs=out,
                  skipped_unverifiable=skipped,
                  query_path=[round(x, 3)
                              for x in _rebase(hist[-(lookback + 1):])])

    # the distribution, step by step across the forward window
    steps = min(len(a.forward) for a in out)
    med, lo, hi = [], [], []
    for k in range(steps):
        col = sorted(a.forward[k] for a in out)
        med.append(round(_quantile(col, 0.5), 3))
        lo.append(round(min(col), 3))
        hi.append(round(max(col), 3))
    fc.median_path, fc.low_path, fc.high_path = med, lo, hi
    fc.median_return = round(med[-1] / med[0] - 1.0, 5) if med and med[0] else 0.0
    ups = sum(1 for a in out if a.forward_return > 0)
    fc.agreement = round(max(ups, len(out) - ups) / len(out), 4)
    return fc


def _quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def to_dict(fc: Forecast) -> dict:
    return {
        "ok": fc.ok,
        "symbol": fc.symbol,
        "as_of": fc.as_of,
        "lookback": fc.lookback,
        "horizon": fc.horizon,
        "searched": fc.searched,
        "skipped_unverifiable": fc.skipped_unverifiable,
        "reason": fc.reason,
        "query_path": fc.query_path,
        "median_path": fc.median_path,
        "low_path": fc.low_path,
        "high_path": fc.high_path,
        "median_return": fc.median_return,
        "agreement": fc.agreement,
        "analogs": [{
            "symbol": a.symbol, "start": a.start, "end": a.end,
            "distance": a.distance, "correlation": a.correlation,
            "path": a.path, "forward": a.forward,
            "forward_return": a.forward_return,
        } for a in fc.analogs],
    }
