"""Model adapters: everything that can answer a prompt.

Spec strings accepted by ``make_adapter``:
    oracle | naive | mock | anthropic[:model] | openai[:model]

API models (anthropic, openai) receive ONLY the prompt text. The reference solvers (oracle, naive)
also receive each question's operation chain, and the simulated model (mock) may look at the
instance it is simulating; neither ever talks to the network.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .models import Instance
from .seeds import derive_seed
from .solver import forecast_estimate, solve
from .util import sha256_hex


@dataclass
class Task:
    """What an adapter may know besides the prompt. API adapters ignore all of it."""

    instance_id: str
    variant: str
    questions: list[dict]  # qid, chain, answer_type, unit, pool -- no answers
    forecast: dict | None = None  # {"qid": "F1", "rule": {...}}
    hidden: Instance | None = None  # only for the simulated mock model; never sent anywhere


@dataclass
class Completion:
    text: str
    meta: dict = field(default_factory=dict)


class AdapterError(RuntimeError):
    pass


class Adapter:
    name = "adapter"
    kind = "reference"  # reference | simulated | api
    concurrent = False  # API adapters can be called from several threads

    def complete(self, prompt: str, task: Task) -> Completion:
        raise NotImplementedError


class OracleAdapter(Adapter):
    """Rule-based upper bound: applies every reading rule to the digest text."""

    name = "oracle"

    def complete(self, prompt: str, task: Task) -> Completion:
        answers = solve(prompt, task.questions, mode="oracle")
        if task.forecast:
            answers[task.forecast["qid"]] = forecast_estimate(prompt, task.forecast["rule"])
        return Completion(json.dumps({"answers": answers}))


class NaiveAdapter(Adapter):
    """Rule-based baseline that ignores deduplication, corrections and retractions."""

    name = "naive"

    def complete(self, prompt: str, task: Task) -> Completion:
        answers = solve(prompt, task.questions, mode="naive")
        if task.forecast:
            answers[task.forecast["qid"]] = 0.5
        return Completion(json.dumps({"answers": answers}))


class MockAdapter(Adapter):
    """A SIMULATED model with built-in benchmark contamination, for demos without API keys.

    It reads the digest like the oracle, then gets each answer right with probability
        0.92  public template, original wording    (it has "memorised" these)
        0.80  public template, paraphrased wording
        0.62  new template                          (it has to actually reason)
    and otherwise errs, usually by falling for a trap when the question has one. All randomness
    is seeded by the exact prompt text, so any surface change (a metamorphic variant) produces
    different mistakes, which the consistency check detects. About one reply in five is formatted
    sloppily to exercise the robust JSON parser. It is NOT a language model.
    """

    name = "mock"
    kind = "simulated"
    P_CORRECT = {("public", True): 0.92, ("public", False): 0.80, ("new", True): 0.62, ("new", False): 0.62}

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def complete(self, prompt: str, task: Task) -> Completion:
        rng = random.Random(derive_seed(self.seed, "mock", sha256_hex(prompt)))
        truth = solve(prompt, task.questions, mode="oracle")
        fallen: dict[str, list] = {}
        if task.hidden is not None:
            for trap in task.hidden.traps:
                for qid, value in trap.affected.items():
                    fallen.setdefault(qid, []).append(value)
        answers: dict[str, Any] = {}
        for q in task.questions:
            qid = q["qid"]
            if rng.random() < self.P_CORRECT[(q["pool"], task.variant == "v0")]:
                answers[qid] = truth[qid]
            elif qid in fallen and rng.random() < 0.75:
                answers[qid] = rng.choice(fallen[qid])
            else:
                answers[qid] = _perturb(q, truth[qid], rng, task)
        if task.forecast:
            answers[task.forecast["qid"]] = round(rng.uniform(0.05, 0.6), 2)
        return Completion(_sloppy_json(answers, rng))


def _perturb(q: dict, value: Any, rng: random.Random, task: Task) -> Any:
    answer_type = q["answer_type"]
    if value is None:
        return None
    if answer_type == "count":
        wrong = max(0, value + rng.choice((-2, -1, 1, 2)))
        return wrong if wrong != value else value + 1
    if answer_type == "event_id":
        refs = sorted({r.ref for r in task.hidden.reports} - {value}) if task.hidden else []
        return rng.choice(refs) if refs else f"{value}x"
    if answer_type == "magnitude":
        return round(value + rng.choice((-0.4, -0.3, 0.3, 0.4)), 2)
    if answer_type in ("depth", "distance"):
        return round(value + rng.choice((-1, 1)) * max(2.0, abs(value) * rng.uniform(0.2, 0.4)), 1)
    if answer_type == "region":
        term = next(s for s in q["chain"] if s["op"] == "compare_regions")
        return rng.choice([x for x in (term["a"], term["b"], "tie") if x != value])
    if answer_type == "utc_time":
        hh, mm = (int(x) for x in value.split(":"))
        total = (hh * 60 + mm + rng.choice((-40, -15, -7, 7, 15, 40))) % 1440
        return f"{total // 60:02d}:{total % 60:02d}"
    return value


def _sloppy_json(answers: dict, rng: random.Random) -> str:
    clean = json.dumps({"answers": answers})
    roll = rng.random()
    if roll < 0.80:
        return clean
    if roll < 0.90:
        return f"Here are my answers based on the digest:\n```json\n{json.dumps({'answers': answers}, indent=2)}\n```"
    if roll < 0.95:  # numbers as strings and a trailing comma
        body = ", ".join(f'"{k}": "{v}"' for k, v in answers.items())
        return '{"answers": {' + body + ",}}"
    if roll < 0.98:
        return str({"answers": answers})  # Python repr: single quotes, None
    return clean[: len(clean) // 2]  # a truncated reply


# ------------------------------------------------------------------------------ API models

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 529}
# Room for models that think before answering: with 4096 tokens, a reasoning model can spend the
# whole budget thinking and return no answer at all. Raise it with `cli.py run --max-tokens`.
DEFAULT_MAX_TOKENS = 16000


def is_truncated(record: dict) -> bool:
    """A stored reply that stopped at the output-token limit (Anthropic or OpenAI wording)."""
    meta = record.get("meta") or {}
    return meta.get("stop_reason") == "max_tokens" or meta.get("finish_reason") == "length"


def token_usage(record: dict) -> tuple[int | None, int | None]:
    """(input, output) tokens of a stored reply as the API reported them (Anthropic or OpenAI wording).
    Anthropic counts cached prompt tokens separately, so they are added back in; output tokens include
    any thinking the model did. (None, None) when the reply carries no usage, e.g. a reference solver."""
    usage = (record.get("meta") or {}).get("usage") or {}
    if "input_tokens" in usage or "output_tokens" in usage:
        prompt = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return prompt, usage.get("output_tokens")
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return None, None


def _post_json(session: Any, url: str, headers: dict, body: dict, timeout: float, *, retries: int = 4,
               drop_on_400: tuple[str, ...] = ()) -> dict:
    body = dict(body)
    for attempt in range(retries + 1):
        try:
            response = session.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise AdapterError(f"request failed: {exc}") from exc
            time.sleep(min(30, 2 ** attempt))
            continue
        if response.status_code == 400:
            dropped = [k for k in drop_on_400 if k in body and k in response.text]
            if dropped:  # some models reject e.g. a temperature setting; retry without it
                for k in dropped:
                    body.pop(k)
                continue
        if response.status_code in RETRY_STATUS and attempt < retries:
            try:
                wait = float(response.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            time.sleep(min(60.0, wait))
            continue
        if response.status_code >= 400:
            raise AdapterError(f"HTTP {response.status_code}: {response.text[:500]}")
        try:
            return response.json()
        except ValueError as exc:  # a 200 with a broken body: retry like a server error
            if attempt == retries:
                raise AdapterError(f"invalid JSON in response: {exc}") from exc
            time.sleep(min(30, 2 ** attempt))
    raise AdapterError("gave up after retries")


class AnthropicAdapter(Adapter):
    kind = "api"
    concurrent = True
    URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-opus-5-5"

    def __init__(self, model: str | None = None, *, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float | None = 0.0,
                 timeout: float = 600.0, session: Any = None) -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AdapterError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("EVALGEN_ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self.name = f"anthropic:{self.model}"
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.session = session or requests

    def complete(self, prompt: str, task: Task) -> Completion:
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post_json(self.session, self.URL, headers, body, self.timeout, drop_on_400=("temperature",))
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return Completion(text, {"model": data.get("model"), "stop_reason": data.get("stop_reason"),
                                 "usage": data.get("usage")})


class OpenAIAdapter(Adapter):
    kind = "api"
    concurrent = True
    DEFAULT_MODEL = "gpt-4.1"

    def __init__(self, model: str | None = None, *, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float | None = 0.0,
                 timeout: float = 600.0, session: Any = None) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise AdapterError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("EVALGEN_OPENAI_MODEL", self.DEFAULT_MODEL)
        self.name = f"openai:{self.model}"
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.url = f"{base}/chat/completions"  # OPENAI_BASE_URL also allows compatible local servers
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.session = session or requests

    def complete(self, prompt: str, task: Task) -> Completion:
        body: dict[str, Any] = {"model": self.model, "max_completion_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        data = _post_json(self.session, self.url, headers, body, self.timeout, drop_on_400=("temperature",))
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        return Completion(text, {"model": data.get("model"), "finish_reason": choice.get("finish_reason"),
                                 "usage": data.get("usage")})


def make_adapter(spec: str, seed: int = 0, max_tokens: int = DEFAULT_MAX_TOKENS) -> Adapter:
    name, _, model = spec.strip().partition(":")
    name = name.lower()
    if name == "oracle":
        return OracleAdapter()
    if name == "naive":
        return NaiveAdapter()
    if name == "mock":
        return MockAdapter(seed)
    if name == "anthropic":
        return AnthropicAdapter(model or None, max_tokens=max_tokens)
    if name == "openai":
        return OpenAIAdapter(model or None, max_tokens=max_tokens)
    raise AdapterError(f"unknown model spec {spec!r} (use oracle, naive, mock, anthropic[:model], openai[:model])")
