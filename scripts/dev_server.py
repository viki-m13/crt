#!/usr/bin/env python3
"""Local dev server mirroring the Vercel routing for the quiz game.

  python scripts/dev_server.py [port]

Serves docs/ statically and routes /api/quiz to the real handler functions,
so what is exercised locally is the same code that runs in production.

DEV_STUB_MARKET=1 replaces the market layer with fixtures. That exists purely
so the reveal screen can be designed and screenshotted while the upstream
quote API is throttling this IP — it is never used in production and prints a
loud banner so a stubbed run is never mistaken for a live one.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
sys.path.insert(0, os.path.join(ROOT, "api"))

import _market as M     # noqa: E402
import iso as ISO       # noqa: E402
import quiz as API      # noqa: E402

if os.environ.get("DEV_STUB_MARKET") == "1":
    import _recommend as R

    FIXTURES = {
        "NVDA": ("NVIDIA Corporation", 230.36, "USD", "NasdaqGS", 0.52, -0.66,
                 0.42, 0.88, 4.4e12, 0.001, 45.0, ["ai", "semis"]),
        "MSFT": ("Microsoft Corporation", 512.10, "USD", "NasdaqGS", 0.24, -0.31,
                 0.18, 0.74, 3.7e12, 0.007, 36.0, ["ai", "cloud", "software"]),
        "KO": ("Coca-Cola Company", 71.20, "USD", "NYSE", 0.15, -0.17,
               0.06, 0.52, 3.0e11, 0.031, 24.0, ["consumer", "food"]),
        "JNJ": ("Johnson & Johnson", 168.44, "USD", "NYSE", 0.15, -0.21,
                0.09, 0.61, 3.6e11, 0.030, 17.0, ["healthcare"]),
        "NESN.SW": ("Nestle S.A.", 78.31, "CHF", "Swiss", 0.15, -0.24,
                    -0.02, 0.33, 2.4e11, 0.033, 19.0, ["consumer", "food"]),
        "ASML.AS": ("ASML Holding N.V.", 1467.40, "EUR", "Amsterdam", 0.39, -0.44,
                    0.28, 0.71, 3.0e11, 0.008, 38.0, ["ai", "semis"]),
        "7203.T": ("Toyota Motor Corporation", 3081.0, "JPY", "Tokyo", 0.22, -0.28,
                   0.11, 0.58, 3.4e13, 0.024, 11.0, ["autos"]),
        "MC.PA": ("LVMH Moet Hennessy Louis Vuitton", 429.10, "EUR", "Paris",
                  0.27, -0.42, -0.08, 0.24, 2.1e11, 0.021, 22.0, ["luxury"]),
    }

    def _stub_quotes(symbols, rng="2y", workers=8):
        out = {}
        for s in symbols:
            f = FIXTURES.get(s)
            if not f:
                out[s] = {"ok": False, "symbol": s, "reason": "not_found"}
                continue
            name, px, cur, exch, vol, dd, mom, pos, cap, dy, pe, th = f
            out[s] = {"ok": True, "symbol": s, "name": name, "price": px,
                      "currency": cur, "exchange": exch, "bars": 500,
                      "ann_vol": vol, "max_dd": dd, "mom_12m": mom,
                      "mom_3m": mom / 3, "pos_52w": pos, "market_cap": cap,
                      "dividend_yield": dy, "pe": pe, "sharpe_window": 0.8,
                      "dd_from_high": -(1 - pos) * 0.3, "themes": th}
        return out

    M.quotes = _stub_quotes
    M.enrich = lambda s: {}
    R.M = M
    R.candidates = lambda profile, answers, want=16: {
        "symbols": list(FIXTURES), "themes": {k: v[11] for k, v in FIXTURES.items()},
        "source": "dev-stub", "notes": ["DEV STUB — not live market data"]}
    print("!" * 70)
    print("!! DEV_STUB_MARKET=1 — market data is FAKE. Layout testing only.")
    print("!" * 70)

MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".json": "application/json",
        ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
        ".txt": "text/plain; charset=utf-8", ".xml": "application/xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/quiz"):
            self._send(200, API.handle_health())
            return
        if path.startswith("/api/iso"):
            self._send(200, ISO.handle_constants())
            return
        rel = path.lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        full = os.path.normpath(os.path.join(DOCS, rel))
        if not full.startswith(DOCS):
            self._send(403, {"error": "forbidden"})
            return
        if not os.path.isfile(full):
            alt = full + ".html"
            if os.path.isfile(alt):
                full = alt
            else:
                self._send(404, "not found", "text/plain")
                return
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f:
            data = f.read()
        self._send(200, data, MIME.get(ext, "application/octet-stream"))

    def do_POST(self):
        path = urlparse(self.path).path
        if not (path.startswith("/api/quiz") or path.startswith("/api/iso")):
            self._send(404, {"error": "no such endpoint"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})
            return
        if path.startswith("/api/iso"):
            try:
                self._send(200, ISO.handle_calc(body))
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        action = str(body.get("action") or "next").lower()
        try:
            if action == "next":
                self._send(200, API.handle_next(body))
            elif action == "pick":
                self._send(200, API.handle_pick(body))
            else:
                self._send(200, API.handle_health())
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"serving {DOCS} + /api/quiz on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
