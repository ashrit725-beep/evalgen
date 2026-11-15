"""Model adapters: everything that can answer a prompt.

Spec strings accepted by ``make_adapter``:
    oracle | naive | mock[:options] | anthropic[:model] | openai[:model]

The simulated model's options give it a penalty on held-out sets: ``mock:held_out=0.3`` (held-out
domains and genres), ``mock:held_out_domains=0.3`` or ``mock:held_out_genres=0.3``, joined with "+"
for both kinds with different penalties. Each lowers its chance of a right answer by that much, but
a chance stops at 0: where the penalties that apply add up to more than a chance (0.62 on a new
template), they lower it by less, and `run` says so (MockAdapter.note).

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

from .domains import get_domain
from .mix import HELD_OUT
from .models import Instance
from .ops import Step, terminal_step
from .scoring import AnswerType, get_answer_type
from .seeds import SeedLog
from .solver import forecast_estimate, solve
from .sources import proxies_for
from .util import sha256_hex


@dataclass
class Task:
    """What an adapter may know besides the prompt. API adapters ignore all of it."""

    instance_id: str
    variant: str
    questions: list[dict]  # qid, chain, answer_type, unit, pool -- no answers
    forecast: dict | None = None  # {"qid": "F1", "rule": {...}}
    hidden: Instance | None = None  # only for the simulated mock model; never sent anywhere
    # What a non-legacy run records (all None in a legacy run, whose prompts are read as 0.2.0 read
    # them): the instance's domain and split, and the rendering's genre and that genre's split.
    domain: str | None = None
    genre: str | None = None
    split: str | None = None  # "public" or "held_out"
    genre_split: str | None = None  # "public" or "held_out"


@dataclass
class Completion:
    text: str
    meta: dict = field(default_factory=dict)


class AdapterError(RuntimeError):
    pass


class UnknownModel(AdapterError, ValueError):
    """A model spec that names no model, or gives its model options it does not take: `run` refuses it
    before any model runs, and the command line exits with status 2."""


class MissingKey(AdapterError):
    """An API model whose key is not set in the environment: `run` skips it, with a note."""


class Adapter:
    name = "adapter"
    kind = "reference"  # reference | simulated | api
    concurrent = False  # API adapters can be called from several threads

    def complete(self, prompt: str, task: Task) -> Completion:
        raise NotImplementedError

    def note(self) -> str | None:
        """Something a run should know about this adapter before it answers anything, or None."""
        return None


class OracleAdapter(Adapter):
    """Rule-based upper bound: applies every reading rule to the digest text."""

    name = "oracle"

    def complete(self, prompt: str, task: Task) -> Completion:
        answers = solve(prompt, task.questions, mode="oracle", domain=task.domain, genre=task.genre)
        if task.forecast:
            answers[task.forecast["qid"]] = forecast_estimate(prompt, task.forecast["rule"], domain=task.domain,
                                                              genre=task.genre)
        return Completion(json.dumps({"answers": answers}))


class NaiveAdapter(Adapter):
    """Rule-based baseline that ignores deduplication, corrections and retractions."""

    name = "naive"

    def complete(self, prompt: str, task: Task) -> Completion:
        answers = solve(prompt, task.questions, mode="naive", domain=task.domain, genre=task.genre)
        if task.forecast:
            answers[task.forecast["qid"]] = 0.5
        return Completion(json.dumps({"answers": answers}))


class MockAdapter(Adapter):
    """A SIMULATED model with built-in benchmark contamination, for demos without API keys.

    It reads the digest like the oracle, then gets each answer right with probability
        0.92  public template, original wording    (it has "memorised" these)
        0.80  public template, paraphrased wording
        0.62  new template, or any other pool       (it has to actually reason)
    and otherwise errs, usually by falling for a trap when the question has one. All randomness
    is seeded by the exact prompt text, so any surface change (a metamorphic variant) produces
    different mistakes, which the consistency check detects. About one reply in five is formatted
    sloppily to exercise the robust JSON parser. It is NOT a language model.

    It can also be given a penalty on held-out sets, as a model tuned to the public domains or genres
    would have: `held_out_domains` is taken off each probability above for an instance whose domain is
    held out (Task.split), and `held_out_genres` for a rendering whose genre is held out
    (Task.genre_split). A probability stops at 0, so penalties that add up to more than one of them
    (0.62, on a new template) take less off it than they say (see note). Without a penalty (the
    default) it draws exactly as before, on every run.
    """

    name = "mock"
    kind = "simulated"
    P_CORRECT = {("public", True): 0.92, ("public", False): 0.80, ("new", True): 0.62, ("new", False): 0.62}
    P_UNSEEN = 0.62  # a pool it has no entry for: as unseen as a new template

    def __init__(self, seed: int = 0, *, held_out_domains: float = 0.0, held_out_genres: float = 0.0) -> None:
        for value in (held_out_domains, held_out_genres):
            if not 0.0 <= value <= 1.0:
                raise UnknownModel(f"a held-out penalty is a probability between 0 and 1, not {value!r}")
        self.seed = seed
        self.held_out_domains, self.held_out_genres = held_out_domains, held_out_genres
        self.name = _mock_name(held_out_domains, held_out_genres)

    def penalty(self, task: Task) -> float:
        """How much less likely a right answer is on this task: the penalty of each held-out set it is in."""
        return ((self.held_out_domains if task.split == HELD_OUT else 0.0)
                + (self.held_out_genres if task.genre_split == HELD_OUT else 0.0))

    def note(self) -> str | None:
        """When the penalties can add up to more than a probability of a right answer, which then stops
        at 0: on those questions the mock loses less than its penalty, so a run that calibrates the gaps
        with it injects smaller gaps than the penalties. None when no probability can fall below 0."""
        lowest = min(min(self.P_CORRECT.values()), self.P_UNSEEN)
        if self.held_out_domains + self.held_out_genres <= lowest + 1e-9:
            return None
        return (f"{self.name}: a chance of a right answer stops at 0, so where its penalties add up to more than "
                f"{lowest:g} (its chance on a new template) they take off less than their size, and the gaps they "
                "inject are smaller than the penalties")

    def complete(self, prompt: str, task: Task) -> Completion:
        # Through a SeedLog, as every draw is: the run's "mock" seed (manifest.json) and the prompt's SHA-256
        # (each stored reply's prompt_sha256) give this generator again.
        rng = SeedLog(self.seed).rng("mock", sha256_hex(prompt))
        truth = solve(prompt, task.questions, mode="oracle", domain=task.domain, genre=task.genre)
        fallen: dict[str, list] = {}
        if task.hidden is not None:
            for trap in task.hidden.traps:
                for qid, value in trap.affected.items():
                    fallen.setdefault(qid, []).append(value)
        penalty = self.penalty(task)
        answers: dict[str, Any] = {}
        for q in task.questions:
            qid = q["qid"]
            if rng.random() < self.P_CORRECT.get((q["pool"], task.variant == "v0"), self.P_UNSEEN) - penalty:
                answers[qid] = truth[qid]
            elif qid in fallen and rng.random() < 0.75:
                answers[qid] = rng.choice(fallen[qid])
            else:
                answers[qid] = _perturb(q, truth[qid], rng, task)
        if task.forecast:
            answers[task.forecast["qid"]] = round(rng.uniform(0.05, 0.6), 2)
        return Completion(_sloppy_json(answers, rng))


MOCK_OPTIONS = ("held_out", "held_out_domains", "held_out_genres")


def _mock_name(held_out_domains: float, held_out_genres: float) -> str:
    """A simulated model's name, written as its spec: "mock", or with its penalties ("mock:held_out=0.3")."""
    if held_out_domains == held_out_genres:
        return "mock" if not held_out_domains else f"mock:held_out={held_out_domains:g}"
    return "mock:" + "+".join(f"{option}={value:g}" for option, value in (
        ("held_out_domains", held_out_domains), ("held_out_genres", held_out_genres)) if value)


def _mock(seed: int, options: str) -> MockAdapter:
    """A simulated model from the options of its spec: none, or option=penalty pairs joined with "+", each
    penalty set once (held_out sets both)."""
    penalties = {"held_out_domains": 0.0, "held_out_genres": 0.0}
    given: set[str] = set()
    for part in filter(None, (piece.strip() for piece in options.split("+"))):
        option, equals, value = (piece.strip() for piece in part.partition("="))
        try:
            amount = float(value)
        except ValueError:
            amount = None
        if option not in MOCK_OPTIONS or not equals or amount is None:
            raise UnknownModel(f"unknown mock option {part!r} (use held_out=P, held_out_domains=P or "
                               "held_out_genres=P, joined with +)")
        for key in ("held_out_domains", "held_out_genres") if option == "held_out" else (option,):
            if key in given:
                raise UnknownModel(f"the mock option {part!r} sets {key} a second time")
            given.add(key)
            penalties[key] = amount
    return MockAdapter(seed, **penalties)


def _perturb(q: dict, value: Any, rng: random.Random, task: Task) -> Any:
    """A wrong answer to the question. The 0.2.0 answer types keep their 0.2.0 draws; any other type
    (a domain's own) is perturbed according to its kind (scoring.AnswerType)."""
    answer_type = q["answer_type"]
    if value is None:
        return None
    if answer_type == "count":
        return _other_count(value, rng)
    if answer_type == "event_id":
        return _other_id(value, rng, task)
    if answer_type == "magnitude":
        return round(value + rng.choice((-0.4, -0.3, 0.3, 0.4)), 2)
    if answer_type in ("depth", "distance"):
        return round(value + rng.choice((-1, 1)) * max(2.0, abs(value) * rng.uniform(0.2, 0.4)), 1)
    if answer_type == "region":
        return _other_group(q, value, rng)
    if answer_type == "utc_time":
        return _other_clock(value, rng)
    known = get_answer_type(answer_type)
    if known.kind == "count":
        return _other_count(value, rng)
    if known.kind == "id":
        return _other_id(value, rng, task)
    if known.kind == "group":
        return _other_group(q, value, rng)
    if known.kind == "clock":
        return _other_clock(value, rng)
    if known.kind == "number":
        return _other_number(q, value, known, rng, task)
    raise ValueError(f"the mock cannot get a {known.kind} answer ({answer_type}) wrong")


def _other_number(q: dict, value: float, known: AnswerType, rng: random.Random, task: Task) -> float:
    """A wrong number in the question's unit and to its places, as a depth or a distance is got wrong
    (a fifth to two fifths off) but scaled to the unit: at least three times the type's tolerance and
    two steps of the places away, so that it is scored wrong even after rounding, in au as in km."""
    places = _places(q, task)
    abs_tol, rel_tol = known.tolerance_in(q["unit"])
    sign = rng.choice((-1, 1))
    shift = max(3 * max(abs_tol, rel_tol * abs(value)), 2 * 10.0 ** -places, abs(value) * rng.uniform(0.2, 0.4))
    return round(value + sign * shift, places)


def _other_count(value: int, rng: random.Random) -> int:
    wrong = max(0, value + rng.choice((-2, -1, 1, 2)))
    return wrong if wrong != value else value + 1


def _other_id(value: str, rng: random.Random, task: Task) -> str:
    refs = sorted({r.ref for r in task.hidden.reports} - {value}) if task.hidden else []
    return rng.choice(refs) if refs else f"{value}x"


def _other_group(q: dict, value: str, rng: random.Random) -> str:
    term = next(s for s in q["chain"] if Step(s).kind == "compare_groups")
    return rng.choice([x for x in (term["a"], term["b"], "tie") if x != value])


def _other_clock(value: str, rng: random.Random) -> str:
    hh, mm = (int(x) for x in value.split(":"))
    total = (hh * 60 + mm + rng.choice((-40, -15, -7, 7, 15, 40))) % 1440
    return f"{total // 60:02d}:{total % 60:02d}"


def _places(q: dict, task: Task) -> int:
    """The decimal places of a numeric answer: the question's rounding, or else (an answer that is
    a stored value) the places its field is stored to."""
    rounding = next((s["ndigits"] for s in q["chain"] if s["op"] == "round"), None)
    if rounding is not None:
        return rounding
    return get_domain(task.domain).field(Step(terminal_step(q["chain"])).field).decimals


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
    any thinking the model did. (None, None) when the reply carries no usage, e.g. a reference solver; a
    count that is not an integer (a reply edited by hand) counts as not reported."""
    meta = record.get("meta")
    usage = meta.get("usage") if isinstance(meta, dict) else None
    if not isinstance(usage, dict):
        return None, None

    def count(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if "input_tokens" in usage or "output_tokens" in usage:
        prompt = sum(count(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return prompt, count("output_tokens")
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return count("prompt_tokens"), count("completion_tokens")
    return None, None


def _unexpected(data: Any, why: str) -> AdapterError:
    """A 200 response whose body is not the shape the API documents: the call failed, and the next `run`
    asks the prompt again (an OpenAI-compatible server or a proxy may answer with anything)."""
    return AdapterError(f"unexpected response body ({why}): {_shown(data)}")


def _shown(data: Any) -> str:
    try:
        return json.dumps(data)[:300]
    except (TypeError, ValueError, RecursionError):  # nested too deep to write back
        return f"a {type(data).__name__}"


def _kept(value: Any) -> Any:
    """A value of a response body as a stored reply keeps it (its model, stop reason, usage): as it is when
    it can be written back as JSON, else None (a server can nest it too deep for that)."""
    try:
        json.dumps(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return value


def _text_parts(parts: Any, data: Any, what: str) -> str:
    """The text of a list of content blocks or parts ({"type": "text", "text": "..."}), one per line, the
    others (thinking, tool use) left out; AdapterError when it is no such list."""
    if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
        raise _unexpected(data, f"{what} is not a list of objects")
    texts = [part.get("text") for part in parts if part.get("type") == "text"]
    if not all(isinstance(text, str) for text in texts):
        raise _unexpected(data, f"a text block of {what} has no text")
    return "\n".join(texts)


class _Untrusting:
    """What an API adapter posts through unless it is given a session: like requests.post, a session of its own
    for each call (calls from several threads share nothing), but one with trust_env off, as every live request
    is (sources.http_get). So it sends no credentials from a netrc file, which requests would add as HTTP Basic
    auth to every host its 'default' entry matches (to api.anthropic.com beside the key, and in place of
    OpenAI's Bearer key), on the first request and again on a redirect; and it reads no CA bundle or proxy
    settings from the environment, going through a proxy only where the proxy variables name one
    (sources.proxies_for)."""

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        with requests.Session() as session:
            session.trust_env = False
            return session.post(url, proxies=proxies_for(url), **kwargs)


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
        except (ValueError, RecursionError) as exc:  # a 200 with a broken body (or nested too deep): retry it
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
            raise MissingKey("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("EVALGEN_ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self.name = f"anthropic:{self.model}"
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.session = session or _Untrusting()

    def complete(self, prompt: str, task: Task) -> Completion:
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post_json(self.session, self.URL, headers, body, self.timeout, drop_on_400=("temperature",))
        if not isinstance(data, dict):
            raise _unexpected(data, "not an object")
        text = _text_parts(data.get("content"), data, "content")  # text blocks, each on its own line
        return Completion(text, {"model": _kept(data.get("model")), "stop_reason": _kept(data.get("stop_reason")),
                                 "usage": _kept(data.get("usage"))})


class OpenAIAdapter(Adapter):
    kind = "api"
    concurrent = True
    DEFAULT_MODEL = "gpt-4.1"

    def __init__(self, model: str | None = None, *, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float | None = 0.0,
                 timeout: float = 600.0, session: Any = None) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingKey("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("EVALGEN_OPENAI_MODEL", self.DEFAULT_MODEL)
        self.name = f"openai:{self.model}"
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.url = f"{base}/chat/completions"  # OPENAI_BASE_URL also allows compatible local servers
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.session = session or _Untrusting()

    def complete(self, prompt: str, task: Task) -> Completion:
        body: dict[str, Any] = {"model": self.model, "max_completion_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        data = _post_json(self.session, self.url, headers, body, self.timeout, drop_on_400=("temperature",))
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise _unexpected(data, "no choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _unexpected(data, "the choice has no message object")
        content = message.get("content")
        if isinstance(content, list):  # some compatible servers send a list of content parts
            content = _text_parts(content, data, "the message content")
        elif content is not None and not isinstance(content, str):
            raise _unexpected(data, "the message content is not text")
        return Completion(content or "", {"model": _kept(data.get("model")),
                                          "finish_reason": _kept(choice.get("finish_reason")),
                                          "usage": _kept(data.get("usage"))})


def make_adapter(spec: str, seed: int = 0, max_tokens: int = DEFAULT_MAX_TOKENS) -> Adapter:
    name, _, model = spec.strip().partition(":")
    name = name.lower()
    if name in ("oracle", "naive") and model.strip():
        raise UnknownModel(f"the {name} model takes no options: {spec!r}")
    if name == "oracle":
        return OracleAdapter()
    if name == "naive":
        return NaiveAdapter()
    if name == "mock":
        return _mock(seed, model)
    if name == "anthropic":
        return AnthropicAdapter(model or None, max_tokens=max_tokens)
    if name == "openai":
        return OpenAIAdapter(model or None, max_tokens=max_tokens)
    raise UnknownModel(f"unknown model spec {spec!r} (use oracle, naive, mock[:options], anthropic[:model], "
                       "openai[:model])")
