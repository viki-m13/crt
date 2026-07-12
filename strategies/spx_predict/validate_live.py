"""Validate the pricing model against LIVE quoted option prices, for
both SPY and SPX (index) options.

For each underlying:
  1. Pull the real option chain (yfinance) and pick the listed expiry
     closest to the strategy's 63-trading-session (~91 calendar day)
     tenor.
  2. Find the real listed strikes closest to 0.97*spot (sell leg) and
     0.94*spot (buy leg).
  3. Record real bid/ask/mid, Yahoo-implied vol, open interest for both
     legs; compute the real spread credit at MID and at NATURAL (sell at
     bid, buy at ask — the worst-case fill).
  4. Price the SAME strikes and SAME calendar tenor with the v2 model
     surface and compare: model credit vs real credit, model leg IVs vs
     market leg IVs.

Verdict logic: the backtest is CONSERVATIVE if the credit it books
(model mid minus 3% slippage) is at or below what the live market
actually pays (natural credit). Emits spx/docs/data/live_validation.json
(append-style: keeps a short history of past checks) and prints a
summary. Fail-soft: any error emits ok=false rather than raising.

Run in CI (Yahoo blocks unauthenticated sandboxes).
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("spxsig", os.path.join(HERE, "signal.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

OUT = os.path.join(HERE, "..", "..", "spx", "docs", "data", "live_validation.json")
TARGET_CAL_DAYS = 91          # ~63 trading sessions
DIV_YIELD = {"SPY": 0.0, "^SPX": 0.013}   # SPY panel is TR (q folded into carry)


def _mid(row):
    b, a = float(row.get("bid") or 0), float(row.get("ask") or 0)
    if b > 0 and a > 0:
        return (b + a) / 2
    lp = float(row.get("lastPrice") or 0)
    return lp if lp > 0 else None


def _leg(tab, target_strike):
    """Nearest listed strike with a live two-sided market."""
    tab = tab.copy()
    tab["dist"] = (tab["strike"] - target_strike).abs()
    for _, r in tab.sort_values("dist").head(6).iterrows():
        b, a = float(r.get("bid") or 0), float(r.get("ask") or 0)
        if b > 0 and a > 0:
            return {"strike": float(r["strike"]), "bid": b, "ask": a,
                    "mid": (b + a) / 2,
                    "iv": round(float(r.get("impliedVolatility") or 0), 4),
                    "oi": int(r.get("openInterest") or 0),
                    "volume": int(r.get("volume") or 0)}
    return None


def validate_underlying(mkt, symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    exps = list(t.options)
    if not exps:
        return {"symbol": symbol, "ok": False, "error": "no expirations"}
    today = dt.date.today()
    exp = min(exps, key=lambda e: abs((dt.date.fromisoformat(e) - today).days
                                      - TARGET_CAL_DAYS))
    dte = (dt.date.fromisoformat(exp) - today).days
    ch = t.option_chain(exp)
    puts = ch.puts.sort_values("strike").reset_index(drop=True)
    # spot: prefer the quote from the chain fetch
    spot = None
    try:
        spot = float(t.fast_info["last_price"])
    except Exception:  # noqa: BLE001
        pass
    if not spot or spot <= 0:
        h = t.history(period="5d")
        spot = float(h["Close"].iloc[-1])

    sell = _leg(puts, spot * 0.97)
    buy = _leg(puts, spot * 0.94)
    if not sell or not buy or sell["strike"] <= buy["strike"]:
        return {"symbol": symbol, "ok": False, "error": "legs not found",
                "spot": spot, "expiry": exp}
    width = sell["strike"] - buy["strike"]
    real_mid = sell["mid"] - buy["mid"]
    real_natural = sell["bid"] - buy["ask"]

    # ---- model pricing of the SAME strikes / tenor on the v2 surface ----
    i = mkt.n - 1
    T = dte / 365.25
    r = float(mkt.rate[i])
    q = DIV_YIELD.get(symbol, 0.0)
    F = spot * math.exp((r - q) * T)
    s_atm = float(mkt.atm_iv(i))

    def model_leg(K):
        s_leg = max(s_atm * (1.0 + S.BETA * math.log(F / K)), 0.03)
        return S.bs_put_F(F, K, T, s_leg, r), s_leg
    v1, iv1 = model_leg(sell["strike"])
    v2, iv2 = model_leg(buy["strike"])
    model_mid = v1 - v2
    model_booked = model_mid * (1 - S.SLIP)   # what the backtest books

    return {
        "symbol": symbol, "ok": True, "spot": round(spot, 2),
        "expiry": exp, "dte_calendar": dte,
        "sell_leg": sell, "buy_leg": buy, "width": round(width, 2),
        "real_credit_mid": round(real_mid, 2),
        "real_credit_natural": round(real_natural, 2),
        "real_credit_mid_pct_width": round(real_mid / width, 4),
        "model_atm_iv": round(s_atm, 4),
        "model_leg_ivs": [round(iv1, 4), round(iv2, 4)],
        "market_leg_ivs": [sell["iv"], buy["iv"]],
        "model_credit_mid": round(model_mid, 2),
        "model_credit_booked": round(model_booked, 2),
        "model_vs_real_mid": round(model_mid / real_mid, 3) if real_mid > 0 else None,
        "booked_vs_natural": round(model_booked / real_natural, 3) if real_natural > 0 else None,
        "model_conservative": bool(model_booked <= real_natural),
        "bidask_pct_of_mid": [round((sell["ask"] - sell["bid"]) / sell["mid"], 3),
                              round((buy["ask"] - buy["bid"]) / buy["mid"], 3)],
    }


def main() -> int:
    mkt = S.Market()
    checks = []
    for sym in ("SPY", "^SPX"):
        try:
            checks.append(validate_underlying(mkt, sym))
        except Exception as exc:  # noqa: BLE001
            checks.append({"symbol": sym, "ok": False, "error": str(exc)[:200]})

    # keep a short history so the page can show consistency over time
    path = os.path.abspath(OUT)
    hist = []
    if os.path.exists(path):
        try:
            hist = json.load(open(path)).get("history", [])
        except Exception:  # noqa: BLE001
            hist = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hist = ([{"as_of": stamp, "checks": checks}] + hist)[:30]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"as_of": stamp, "checks": checks, "history": hist}, fh, indent=2)

    print(f"live validation {stamp}")
    for c in checks:
        if not c.get("ok"):
            print(f"  {c['symbol']}: FAILED — {c.get('error')}")
            continue
        print(f"  {c['symbol']}: spot {c['spot']} exp {c['expiry']} ({c['dte_calendar']}d)  "
              f"legs {c['sell_leg']['strike']}/{c['buy_leg']['strike']}")
        print(f"    real credit mid={c['real_credit_mid']} natural={c['real_credit_natural']}  "
              f"model mid={c['model_credit_mid']} booked={c['model_credit_booked']}  "
              f"conservative={c['model_conservative']}")
        print(f"    leg IVs market={c['market_leg_ivs']} model={c['model_leg_ivs']}  "
              f"OI={[c['sell_leg']['oi'], c['buy_leg']['oi']]}  "
              f"bid-ask %mid={c['bidask_pct_of_mid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
