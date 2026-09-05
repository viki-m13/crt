"""Provider-agnostic LLM access, routed by what each task actually needs.

Three task classes, because they have genuinely different requirements:

  FAST   generating the next quiz question. Latency is the product here — a
         two-second pause between questions kills the game feel — and the
         task is easy. Cerebras first (fastest inference available), then
         Groq, then a small Anthropic model.
  LIVE   proposing candidate tickers. Needs CURRENT knowledge of what is
         happening in markets, so a search-grounded provider wins.
         Perplexity Sonar first (search built in, returns citations), then
         Anthropic, then anything else.
  SMART  writing the final recommendation prose. Quality matters once per
         game, so the best available reasoning model goes first.

Every provider is optional. With no keys at all the caller falls back to the
built-in deterministic question bank, and the site still works end to end —
that is a hard requirement, not a nicety.

Standard library only (serverless cold start).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

TIMEOUT_FAST = 12.0
TIMEOUT_SMART = 45.0


class LLMUnavailable(Exception):
    """No provider could answer. Callers must degrade, never fabricate."""


# --------------------------------------------------------------------------
# provider adapters — each returns plain text or raises
# --------------------------------------------------------------------------
# Several providers sit behind Cloudflare, which blocks the default
# "Python-urllib/3.x" signature outright (403, error code 1010). Sending an
# explicit User-Agent is what makes these APIs reachable at all from a plain
# stdlib client — verified against Cerebras, where its absence is a hard 403.
UA = "signal-quiz/1.0 (+https://github.com/viki-m13/crt)"


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    headers = dict(headers)
    headers.setdefault("User-Agent", UA)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _openai_chat(base: str, key: str, model: str, messages: list, *,
                 max_tokens: int, temperature: float, timeout: float,
                 extra: dict | None = None) -> str:
    # Reasoning models (gpt-oss, qwen3) emit a `reasoning` field before
    # `content`, and when the budget runs out mid-thought they return a
    # message with NO content key at all. Indexing blindly turned that into
    # a KeyError that looked like a provider outage, so the budget is raised
    # for these models and the read is defensive.
    reasoning = any(t in model.lower() for t in ("gpt-oss", "qwen", "-r1", "thinking"))
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens * 3 if reasoning else max_tokens,
               "temperature": temperature}
    if extra:
        payload.update(extra)
    d = _post(base, payload, {"Content-Type": "application/json",
                              "Authorization": f"Bearer {key}"}, timeout)
    choices = d.get("choices") or []
    if not choices:
        raise ValueError("no choices returned")
    msg = choices[0].get("message") or {}
    text = msg.get("content")
    if not text:
        raise ValueError(
            f"empty content (finish_reason={choices[0].get('finish_reason')})")
    return text


def _anthropic(key: str, model: str, messages: list, *, max_tokens: int,
               temperature: float, timeout: float) -> str:
    system = "".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    payload = {"model": model, "max_tokens": max_tokens,
               "temperature": temperature, "messages": turns}
    if system:
        payload["system"] = system
    d = _post("https://api.anthropic.com/v1/messages", payload,
              {"Content-Type": "application/json", "x-api-key": key,
               "anthropic-version": "2023-06-01"}, timeout)
    return "".join(b.get("text", "") for b in d.get("content", []))


# name -> (env var, callable(key, messages, **kw) -> text, default model)
def _providers() -> dict:
    return {
        "cerebras": ("CEREBRAS_API_KEY",
                     lambda k, m, **kw: _openai_chat(
                         "https://api.cerebras.ai/v1/chat/completions", k,
                         # verified against the account's /v1/models list;
                         # llama-3.3-70b is NOT available and 404s
                         os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"), m, **kw)),
        "groq": ("GROQ_API_KEY",
                 lambda k, m, **kw: _openai_chat(
                     "https://api.groq.com/openai/v1/chat/completions", k,
                     os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), m, **kw)),
        "perplexity": ("PERPLEXITY_API_KEY",
                       lambda k, m, **kw: _openai_chat(
                           "https://api.perplexity.ai/chat/completions", k,
                           os.environ.get("PERPLEXITY_MODEL", "sonar"), m, **kw)),
        "anthropic": ("ANTHROPIC_API_KEY",
                      lambda k, m, **kw: _anthropic(
                          k, os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                          m, **kw)),
        "openai": ("OPENAI_API_KEY",
                   lambda k, m, **kw: _openai_chat(
                       "https://api.openai.com/v1/chat/completions", k,
                       os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), m, **kw)),
    }


PREFERENCE = {
    "fast": ("cerebras", "groq", "anthropic", "openai", "perplexity"),
    "live": ("perplexity", "anthropic", "openai", "cerebras", "groq"),
    "smart": ("anthropic", "openai", "perplexity", "cerebras", "groq"),
}


# Environment variables are case-sensitive and easy to misname when set by
# hand in a dashboard. Rather than fail silently with "no provider
# configured", each provider accepts a few plausible spellings of its key.
_ALIASES = {
    "CEREBRAS_API_KEY": ("CEREBRAS_API_KEY", "CEREBRAS_KEY", "CEREBRAS",
                         "CEREBRUS_API_KEY", "CEREBRUS_KEY", "CEREBRUS"),
    "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "CLAUDE_API_KEY"),
    "PERPLEXITY_API_KEY": ("PERPLEXITY_API_KEY", "PERPLEXITY_KEY", "PPLX_API_KEY"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY", "OPENAI_KEY"),
    "GROQ_API_KEY": ("GROQ_API_KEY", "GROQ_KEY"),
}


def _key_for(env: str) -> str | None:
    """Find a provider key under any accepted spelling, in any case."""
    for name in _ALIASES.get(env, (env,)):
        for variant in (name, name.lower(), name.upper()):
            v = os.environ.get(variant)
            if v and v.strip():
                return v.strip()
    return None


def available() -> list[str]:
    return [n for n, (env, _) in _providers().items() if _key_for(env)]


def complete(messages: list, *, task: str = "fast", max_tokens: int = 700,
             temperature: float = 0.8, timeout: float | None = None) -> tuple[str, str]:
    """Return (text, provider_name). Raises LLMUnavailable if none succeed."""
    provs = _providers()
    order = PREFERENCE.get(task, PREFERENCE["fast"])
    # an explicit override always wins, e.g. QUIZ_PROVIDER_FAST=groq
    forced = os.environ.get(f"QUIZ_PROVIDER_{task.upper()}")
    if forced:
        order = (forced,) + tuple(o for o in order if o != forced)
    t = timeout or (TIMEOUT_FAST if task == "fast" else TIMEOUT_SMART)
    errors = []
    for name in order:
        entry = provs.get(name)
        if not entry:
            continue
        env, fn = entry
        key = _key_for(env)
        if not key:
            continue
        try:
            txt = fn(key, messages, max_tokens=max_tokens,
                     temperature=temperature, timeout=t)
            if txt and txt.strip():
                return txt, name
            errors.append(f"{name}:empty")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:160]
            except Exception:
                pass
            errors.append(f"{name}:{e.code}:{detail}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}:{type(e).__name__}")
    raise LLMUnavailable("; ".join(errors) or "no provider configured")


# --------------------------------------------------------------------------
# JSON coaxing — models wrap JSON in prose or fences no matter how you ask
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str):
    """Best-effort extraction of a JSON object/array from model output."""
    if not text:
        raise ValueError("empty")
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = _FENCE.search(t)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # first balanced {...} or [...] in the stream
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(t)):
                c = t[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except Exception:
                            break
            start = t.find(opener, start + 1)
    raise ValueError("no JSON found")


def complete_json(messages: list, **kw):
    """Ask for JSON and insist on getting it, with one corrective retry."""
    text, prov = complete(messages, **kw)
    try:
        return parse_json(text), prov
    except ValueError:
        retry = list(messages) + [
            {"role": "assistant", "content": text[:500]},
            {"role": "user", "content":
             "That was not valid JSON. Reply with ONLY the JSON object, "
             "no prose, no code fences."}]
        kw2 = dict(kw)
        kw2["temperature"] = 0.2
        text2, prov = complete(retry, **kw2)
        return parse_json(text2), prov
