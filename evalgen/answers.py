"""Robust parsing of model replies into {question id: answer}.

Models are asked for one strict JSON object. We record how well they complied:
  strict   the whole reply is that JSON object
  lenient  the answers had to be recovered: code fences, prose or reasoning around the JSON,
           comments, trailing commas, single quotes, Python literals, or as a last resort
           "Q1: 7" lines (the last mention of each question wins)
  failed   nothing usable
When a reply contains several JSON objects, the one covering the most expected question IDs wins
(ties: the one wrapped in "answers", then the later one). If no single object covers them all,
e.g. one object per question, the objects are merged in order. Answer *values* are normalised in
scoring.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

_QID = re.compile(r"^\s*([QqFf])?\s*(\d+)\s*$")
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_SCRAPE = re.compile(
    r"\**[\"']?\b([QF]\d+)\b[\"']?\**\s*[:=][\s*]*"
    r"(\"[^\"\n]*\"|'[^'\n]*'"  # a quoted value
    r"|\d{1,2}[:h]\d{2}(?::\d{2})?"  # a clock time, before plain numbers
    r"|[-+]?\d+(?:\.\d+)?(?![\d:])"  # a number
    r"|[A-Za-z][^\n}\"]*?(?=\s*(?:,\s*[\"']?\**[QF]\d+\b|[}\n]|$)))"  # bare text, commas allowed
)


@dataclass
class ParsedReply:
    answers: dict[str, Any] = field(default_factory=dict)
    status: str = "failed"
    error: str | None = None


def normalize_qid(key: Any) -> str | None:
    m = _QID.match(str(key))
    if not m:
        return None
    return f"{(m.group(1) or 'Q').upper()}{int(m.group(2))}"


def _answers_from(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    inner = obj.get("answers", obj)
    if isinstance(inner, list):  # [{"id": "Q1", "answer": 3}, ...]
        inner = {item.get("id") or item.get("qid"): item.get("answer") for item in inner if isinstance(item, dict)}
    if not isinstance(inner, dict):
        return None
    out = {}
    for key, value in inner.items():
        qid = normalize_qid(key)
        if qid:
            out[qid] = value
    return out or None


def _json_objects(text: str) -> list[tuple[int, str]]:
    """Every top-level balanced {...} substring with its position (a brace matcher that respects strings)."""
    found, depth, start, in_string, escaped = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                found.append((start, text[start:i + 1]))
    return found


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments outside JSON strings."""
    out, i, in_string, escaped = [], 0, False, False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
        elif ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _repairs(candidate: str) -> Iterator[str]:
    yield candidate
    fixed = _strip_comments(candidate)
    fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    yield fixed
    pythonic = re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", fixed)))
    yield pythonic
    if "'" in pythonic:
        yield pythonic.replace("'", '"')  # single-quoted keys and strings


def _scalar(token: str) -> Any:
    token = token.strip()
    if token[:1] in "\"'":
        return token[1:-1]
    for cast in (int, float):
        try:
            return cast(token)
        except ValueError:
            pass
    return token


def _scrape(text: str) -> dict[str, Any]:
    """'Q1: 7' lines over the whole reply, the last mention of each question winning; what follows
    the last '{' (a final answer cut off mid-JSON) overrides only the questions it contains."""

    def scan(source: str) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for m in _SCRAPE.finditer(source):
            qid = normalize_qid(m.group(1))
            if qid:
                found[qid] = _scalar(m.group(2))
        return found

    found = scan(text)
    if "{" in text:
        found.update(scan(text[text.rfind("{") + 1:]))
    return found


def parse_reply(text: str | None, expected: Sequence[str] | None = None) -> ParsedReply:
    if not text or not text.strip():
        return ParsedReply(error="empty reply")
    try:
        answers = _answers_from(json.loads(text.strip()))
        if answers:
            return ParsedReply(answers, "strict")
    except json.JSONDecodeError:
        pass
    wanted = set(expected or ())

    def coverage(answers: dict) -> int:
        return len(set(answers) & wanted) if wanted else len(answers)

    candidates = [(m.start(), m.group(1)) for m in _FENCE.finditer(text)] + _json_objects(text)
    candidates += _json_objects(_strip_comments(text))
    parsed: list[tuple[int, bool, dict]] = []  # (position, wrapped in "answers", answers)
    for position, candidate in candidates:
        for attempt in _repairs(candidate):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            answers = _answers_from(obj)
            if answers:
                parsed.append((position, isinstance(obj, dict) and "answers" in obj, answers))
                break
    if parsed:
        best = dict(max(parsed, key=lambda p: (coverage(p[2]), p[1], p[0]))[2])
        if wanted and coverage(best) < len(wanted):  # e.g. one JSON object per question: merge them
            merged: dict[str, Any] = {}
            for _, _, answers in sorted(parsed, key=lambda p: p[0]):
                merged.update(answers)
            if coverage(merged) > coverage(best):
                best = merged
        return ParsedReply(best, "lenient")
    scraped = _scrape(text)
    if scraped:
        return ParsedReply(scraped, "lenient", "recovered by line scraping")
    return ParsedReply(error="no JSON object with answers found")
