# Signal — the stock-matching game

The homepage (`docs/index.html` + `game.js` + `game.css`) plays a short,
adaptive quiz and matches the player to one real, verified stock. This
directory is the engine behind it.

## Shape of it

```
_market.py      live global market data + verification    (no preloaded universe)
_engine.py      preference profile, safety gates, fit scoring   (no LLM, no network)
_questions.py   question bank, model-written questions, archetypes
_recommend.py   candidates -> verify -> rank -> narrate
_llm.py         provider-agnostic model access
quiz.py         the HTTP surface
```

The division that matters: **an LLM proposes candidates and writes prose; it
never decides the ranking and never supplies a number.** Every figure shown to
a player is computed in `_market.py` from real prices. A ticker a model
invents simply fails verification and is dropped.

## Configuration

All optional — with no keys at all the game still plays from the built-in
question bank, but it cannot verify a recommendation (see below).

| Variable | What it buys |
|---|---|
| `CEREBRAS_API_KEY` | Fast adaptive questions. Currently in use. |
| `FINNHUB_API_KEY` | **Market data that works from a datacenter.** |
| `TWELVEDATA_API_KEY` | Same, second choice (8 req/min is tight). |
| `ANTHROPIC_API_KEY` | Better reveal copy; also a question fallback. |
| `PERPLEXITY_API_KEY` | Search-grounded, current candidate proposals. |
| `ALPHAVANTAGE_API_KEY` | US-only extras: sector, P/E, margins. |
| `QUIZ_DISABLE_LLM=1` | Force the deterministic bank (used by tests). |

Key names are matched case-insensitively and under common misspellings, so a
variable entered by hand in a dashboard does not silently mean "no provider".

### The market-data problem, stated plainly

Yahoo's chart endpoint is the only free source with genuinely global coverage
— verified across NYSE, Nasdaq, Tokyo, XETRA, Amsterdam, Paris, Zurich,
Hong Kong, Mumbai, Seoul, Sydney, Toronto and São Paulo. **It rate-limits
datacenter IPs, including Vercel's.** Measured in production: 16 candidates
proposed, 0 verified, 16 throttled. Stooq, the other keyless option, now
answers a bot-check page instead of CSV.

So a keyed fallback is required for the reveal to work in production.
Finnhub's free tier (60 calls/minute, global) fits the burst one game
creates; Twelve Data's free tier (8/minute) is tight but works.

`GET /api/quiz?action=probe` reports which sources answer **from the host
serving traffic** — the only place that question can be answered.

When nothing can be verified the game says so and recommends nothing. That is
deliberate: a made-up price is worse than no answer.

## Endpoints

```
GET  /api/quiz?action=health   configured providers, bank size
GET  /api/quiz?action=probe    which market sources are reachable from here
POST /api/quiz {"action":"next","answers":[...]}   the next question
POST /api/quiz {"action":"pick","answers":[...]}   the recommendation
```

Stateless: the client carries the answer history and the server replays it,
re-validating and clamping every effect, so a client can repeat its own game
but cannot forge a preference past the safety gates.

## Running and testing locally

```bash
python scripts/dev_server.py 8099          # mirrors the Vercel routing
DEV_STUB_MARKET=1 python scripts/dev_server.py 8099   # fixtures, for layout work

python tests/test_pick_engine.py    # 45 checks: profile, gates, fit, personas
python tests/test_quiz_api.py       # 59 checks: full games, hostile input, degradation
python tests/test_market_live.py    # live coverage; SKIPs when upstream throttles
python scripts/shoot_game.py        # plays the game in a browser at 5 viewports
                                    # and audits overflow, tap targets, clipping
```

The first two need no network and no API key, and are the ones to trust when
changing the engine. `test_market_live.py` reports SKIP rather than FAIL under
throttling, so a rate limit is never mistaken for a regression.

## Notes for whoever changes this next

- Question options must use the **full 0..1 range**. An early bank offered
  only extremes, which forced moderate players to overstate and moved
  recovered profiles by 0.17, handing them the wrong archetype and a worse
  match. `test_quiz_api.py` guards this now.
- Archetypes are nearest-prototype, not first-match-wins, and every prototype
  must name the dimensions that define it — a prototype that stays silent on
  a dimension wins players it should lose.
- Reveal copy must describe the relationship that actually holds. It once
  shipped "its 15% volatility sits close to the 33% ride you asked for".
