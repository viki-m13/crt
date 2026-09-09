"""HTTP surface for the pattern (analog) matcher.

  GET /api/analog?symbol=NVDA[&lookback=120&horizon=60]
      -> the historical windows in that ticker whose shape matches the last
         `lookback` days, what each did over the following `horizon` days,
         and the measured out-of-sample accuracy of doing exactly this.

The accuracy block is not optional and is not computed here — it is the
frozen result of research/analog/validate.py, a walk-forward test over 3,968
non-overlapping forecasts. It is attached to EVERY response, including the
failures, so no caller can render the chart without the number that says what
the chart is worth. That number is 50.5% directional accuracy against a
55.7% always-up baseline: worse than assuming the stock goes up.

The chart is still worth showing. "Here are the closest things that have
happened to this stock before, and here is what followed each of them" is a
true statement and a useful piece of context. "Therefore this is what happens
next" is not, and the API is built so that the second claim cannot be made
without the refutation travelling beside it.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _analog as A     # noqa: E402
import _market as M     # noqa: E402

# Frozen output of research/analog/validate.py. Editing these numbers without
# re-running that script is falsifying a measurement.
ACCURACY = {
    "directional_accuracy": 0.5284,
    "always_up_baseline": 0.5568,
    "edge_vs_baseline": -0.0283,
    "t_stat": -1.86,
    "random_control_accuracy": 0.5162,
    "information_coefficient": -0.0181,
    "ic_t_stat": -0.91,
    "forecasts": 3931,
    "dates": 128,
    "horizon_days": 60,
    "beats_predicting_zero": 0.424,
    "scope": "the ticker's own history, which is what this endpoint searches",
    "verdict": (
        "Measured out of sample on 3,931 non-overlapping forecasts, this "
        "method called direction correctly 52.8% of the time. Simply "
        "assuming the stock rises was right 55.7% of the time. Choosing the "
        "windows at random instead of matching them scored 51.6%. Closer "
        "matches did not do better — the tightest quartile scored worst. "
        "Use this to see what has happened before, not to predict what "
        "happens next."),
    "method": "research/analog/validate.py (own-history mode, the default)",
}

MAX_LOOKBACK = 500
MAX_HORIZON = 250


def _int(v, default, lo, hi):
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


class handler(BaseHTTPRequestHandler):

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        # Only successes are cacheable. A rate-limit is a statement about us,
        # not about the ticker; caching it for 15 minutes would make the UI's
        # "try again shortly" a lie at the CDN.
        self.send_header("Cache-Control", "public, max-age=900"
                         if payload.get("ok") else "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query)
            one = lambda k, d="": (q.get(k) or [d])[0]  # noqa: E731

            if one("action") == "accuracy":
                return self._send(200, {"ok": True, "accuracy": ACCURACY})

            symbol = one("symbol").strip().upper()
            if not symbol:
                return self._send(400, {"ok": False, "reason": "no_symbol",
                                        "accuracy": ACCURACY})

            lookback = _int(one("lookback"), A.LOOKBACK, 20, MAX_LOOKBACK)
            horizon = _int(one("horizon"), A.HORIZON, 5, MAX_HORIZON)

            h = M.history(symbol, "max")
            if not h.get("ok"):
                return self._send(200, {
                    "ok": False, "symbol": symbol,
                    "reason": h.get("reason", "unavailable"),
                    "retryable": h.get("retryable", False),
                    "accuracy": ACCURACY})

            # The library is the ticker's own history. Matching a stock to
            # other stocks is supported by the engine but not offered here —
            # it needs a preloaded universe. The ACCURACY block above is
            # measured in exactly this configuration; the first version was
            # measured cross-sectionally over 400 tickers and reported as if
            # it described this, which overstated nothing but described the
            # wrong thing (50.5% there, 52.8% here).
            lib = {symbol: (h["dates"], h["closes"])}
            fc = A.find_analogs(h["closes"], lib, lookback=lookback,
                                horizon=horizon, symbol=symbol,
                                dates=h["dates"])

            out = A.to_dict(fc)
            out.update({
                "name": h.get("name"), "currency": h.get("currency"),
                "exchange": h.get("exchange"), "bars": h.get("bars"),
                "source": h.get("source"),
                "last_price": h["closes"][-1],
                "accuracy": ACCURACY,
            })
            return self._send(200, out)

        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"ok": False, "reason": "server_error",
                                    "detail": f"{type(e).__name__}: {e}",
                                    "accuracy": ACCURACY})

    def log_message(self, *a):
        pass
