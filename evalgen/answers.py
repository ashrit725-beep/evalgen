"""Robust parsing of model replies into {question id: answer}.

Models are asked for one strict JSON object. We record how well they complied:
  strict   the whole reply is that JSON object, with every question in it once
  lenient  the answers had to be recovered: code fences, prose or reasoning around the JSON,
           comments, trailing commas, single quotes, Python literals, a question given twice, or as
           a last resort "Q1: 7" lines (the last mention of each question wins)
  failed   nothing usable
When a reply contains several JSON objects, those covering the most expected question IDs are the
answer: of them, those wrapped in "answers" (the shape the prompt asks for) when there are any, so that
working notes keyed by question ID beside the answer are not read as one. If no single object covers
them all, e.g. one object per question, all the objects are merged. A question that the objects used (or
one object, by giving it twice) answer with different values is ambiguous: its answer is the list of
those values, which the scorer reads as no answer unless they all read the same, as it reads "7 or 8".
Two complete answers that disagree are a hedge, whichever comes last. A value that is the prompt's own
placeholder ("<event ID>", "HH:MM") is not an answer. A scraped line keeps its whole value ("7 or 8",
"5,088.8 km") for the scorer to read. A value that runs into the end of a reply cut off inside its final
JSON object may be cut short ("2217903" cut to "22"): it never replaces an earlier mention of its
question, and is read as written (as 0.2.0 read it) when there is none, unless it holds no digit or
letter (a lone "-"). Answer *values* are normalised in scoring.py.

Replies are untrusted input: parse_reply never raises, takes time linear in the length of the reply,
and gives the same answers on every Python version. JSON nested deeper than any answer object is not
parsed (how deep json.loads can go depends on the Python version and the stack), and a number that has
no exact JSON reading on every version (an integer of more than 1000 digits, NaN, Infinity, 1e999) is
kept as the text it was written as, so that what `score` records is always standard JSON.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

_QID = re.compile(r"([QqFf]?)\s*(\d{1,6})")  # matched whole, on the stripped key
_MAX_DEPTH = 64  # an answer object is nested a few levels deep: {"answers": [{"id": "Q1", "answer": [..]}]}
_MAX_INT_DIGITS = 1000  # json.loads refuses longer integers on Python 3.11+ (sys.int_info), not on 3.9
_PLACEHOLDER = re.compile(r"\s*(?:<[^<>\n]*>|hh\s*:\s*mm)\s*", re.IGNORECASE)  # the prompt's "<event ID>", "HH:MM"
_PLAIN_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_MENTION = re.compile(r"\b([QqFf]\d{1,6})\b[\"'`\u201d\u2019]?\**[ \t]*(?::|=>?|->)")  # "Q1:", **Q1:**, "Q1" =>
_QUOTES = {'"': '"', "'": "'", "`": "`", "\u201c": "\u201d", "\u2018": "\u2019"}
_LEADING = re.compile(r"[\s*]*")
_FAIL = object()


@dataclass
class ParsedReply:
    answers: dict[str, Any] = field(default_factory=dict)
    status: str = "failed"
    error: str | None = None


def normalize_qid(key: Any) -> str | None:
    """"Q1", "q1", " 1 ", 1 or "F1" as "Q1"/"F1"; None for anything else (a list, an object, a number
    of more than six digits)."""
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        key = str(key) if 0 <= key < 10 ** 6 else ""
    if not isinstance(key, str):
        return None
    m = _QID.fullmatch(key.strip())
    if not m:
        return None
    return f"{(m.group(1) or 'Q').upper()}{int(m.group(2))}"


# ------------------------------------------------------------------------------------ JSON


class _Object(dict):
    """A JSON object that keeps every key it was given, in order, a repeated key included."""

    pairs: list = []


def _object(pairs: list) -> _Object:
    obj = _Object(pairs)
    obj.pairs = pairs
    return obj


def _integer(text: str) -> Any:
    return int(text) if len(text) <= _MAX_INT_DIGITS else text


def _float(text: str) -> Any:
    value = float(text)
    return value if math.isfinite(value) else text


_STRUCTURE = re.compile(r'[\\"\[\]{}]')


def _too_deep(text: str) -> bool:
    """Whether arrays and objects nest deeper than _MAX_DEPTH in `text` (outside its strings)."""
    depth, in_string, skip = 0, False, -1
    for m in _STRUCTURE.finditer(text):
        i, ch = m.start(), m.group()
        if i < skip:
            continue
        if in_string:
            if ch == "\\":
                skip = i + 2
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > _MAX_DEPTH:
                return True
        elif ch in "]}":
            depth = max(0, depth - 1)
    return False


def _load(text: str) -> Any:
    """The JSON value of `text`, or _FAIL. Never raises (see the module docstring)."""
    if _too_deep(text):
        return _FAIL
    try:
        return json.loads(text, object_pairs_hook=_object, parse_int=_integer, parse_float=_float,
                          parse_constant=str)
    except (ValueError, RecursionError):  # JSONDecodeError is a ValueError
        return _FAIL


def _pairs(obj: dict) -> list:
    return obj.pairs if isinstance(obj, _Object) else list(obj.items())


def _answers_from(obj: Any) -> dict[str, list] | None:
    """Every value an answer object gives each question, {qid: [value, ...]} in order, or None: an object
    of question IDs, the same wrapped in "answers" (several "answers" keys: all of them), or a list of
    {"id": "Q1", "answer": 3} items, wrapped or not. Placeholders are no answer."""
    if isinstance(obj, dict):
        inners = [value for key, value in _pairs(obj) if key == "answers"] if "answers" in obj else [obj]
    elif isinstance(obj, list):
        inners = [obj]
    else:
        return None
    out: dict[str, list] = {}
    for inner in inners:
        if isinstance(inner, list):
            given = [(item.get("id") or item.get("qid"), item.get("answer")) for item in inner if isinstance(item, dict)]
        elif isinstance(inner, dict):
            given = _pairs(inner)
        else:
            continue
        for key, value in given:
            qid = normalize_qid(key)
            if qid and not (isinstance(value, str) and _PLACEHOLDER.fullmatch(value)):
                out.setdefault(qid, []).append(value)
    return out or None


def _value(values: list) -> Any:
    """The answer of a question given `values`: the one value, or the list of the different values it was
    given (a value that is a list counting as each of its items), which the scorer reads as one answer only
    when they all read the same."""
    if len(values) == 1:
        return values[0]
    distinct, seen = [], set()
    for value in values:
        for item in value if isinstance(value, list) else [value]:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                distinct.append(item)
    return distinct[0] if len(distinct) == 1 else distinct


def _combined(objects: list[dict[str, list]]) -> tuple[dict[str, Any], list[str]]:
    """Answer objects taken together, and the questions they give different values."""
    given: dict[str, list] = {}
    for answers in objects:
        for qid, values in answers.items():
            given.setdefault(qid, []).extend(values)
    combined = {qid: _value(values) for qid, values in given.items()}
    return combined, [qid for qid, values in given.items() if len(values) > 1 and isinstance(combined[qid], list)]


_OBJECT_SCAN = re.compile(r'["\\{}]')


def _json_objects(text: str) -> list[tuple[int, int]]:
    """The (start, end) of every top-level balanced {...} substring (a brace matcher that respects strings)."""
    found, depth, start, in_string, skip = [], 0, 0, False, -1
    for m in _OBJECT_SCAN.finditer(text):
        i, ch = m.start(), m.group()
        if i < skip:
            continue
        if in_string:
            if ch == "\\":
                skip = i + 2
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                found.append((start, i + 1))
    return found


def _fences(text: str) -> list[tuple[int, int]]:
    """The (start, end) of the content of every ```json ... ``` code fence."""
    found, i = [], text.find("```")
    while i >= 0:
        end = text.find("```", i + 3)
        if end < 0:
            break
        start = i + 3
        if text.startswith(("json", "JSON"), start):
            start += 4
        found.append((start, end))
        i = text.find("```", end + 3)
    return found


_COMMENT_SCAN = re.compile(r'["\\]|//|/\*')


def _blank_comments(text: str) -> str:
    """`text` with its // and /* */ comments outside JSON strings blanked out: replaced by spaces, their
    line breaks kept, so that every position in it is the same position in `text`."""
    out, last, i, in_string = [], 0, 0, False
    while True:
        m = _COMMENT_SCAN.search(text, i)
        if not m:
            break
        token, at = m.group(), m.start()
        if in_string:
            if token == '"':
                in_string = False
            i = at + 2 if token != '"' else at + 1  # an escaped character, or "//" and "/*" inside a string
            continue
        if token == '"':
            in_string, i = True, at + 1
        elif token == "\\":
            i = at + 1
        else:
            end = text.find("\n", at) if token == "//" else text.find("*/", at + 2)
            end = len(text) if end < 0 else end if token == "//" else end + 2
            out += [text[last:at], re.sub(r"[^\n]", " ", text[at:end])]
            last = i = end
    out.append(text[last:])
    return "".join(out)


def _repairs(candidate: str) -> Iterator[str]:
    yield candidate
    fixed = _blank_comments(candidate)
    fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    yield fixed
    pythonic = re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", fixed)))
    yield pythonic
    if "'" in pythonic:
        yield pythonic.replace("'", '"')  # single-quoted keys and strings


def _objects(text: str) -> list[tuple[dict[str, list], bool]]:
    """The answer objects in a reply, in order, each with whether it is wrapped in "answers": in its code
    fences, and every top-level object, found in the reply and in the reply with its comments blanked out
    (a comment can hide a brace from the first), each span parsed once, with the first repair that parses."""
    spans = sorted(set(_fences(text)) | set(_json_objects(text)) | set(_json_objects(_blank_comments(text))))
    found = []
    for start, end in spans:
        for attempt in _repairs(text[start:end]):
            obj = _load(attempt)
            if obj is not _FAIL:  # a repair only mends what does not parse
                answers = _answers_from(obj)
                if answers:
                    found.append((answers, isinstance(obj, dict) and "answers" in obj))
                break
    return found


# ------------------------------------------------------------------------------- "Q1: 7" lines


class _Positions:
    """The positions of one character in a text, for the next one at or after a position."""

    def __init__(self, text: str, char: str) -> None:
        self.at = [m.start() for m in re.finditer(re.escape(char), text)]

    def next(self, position: int, default: int) -> int:
        i = bisect.bisect_left(self.at, position)
        return self.at[i] if i < len(self.at) else default


def _scalar(token: str) -> Any:
    """A scraped value without its quotes and markdown, as a number when the whole of it is one."""
    if _PLAIN_NUMBER.fullmatch(token) and len(token) <= _MAX_INT_DIGITS:
        if token.lstrip("+-").isdigit():
            return int(token)
        value = float(token)
        return value if math.isfinite(value) else token
    return token


def _opening(text: str, start: int, floor: int) -> int:
    """Where the mention of a question that starts at `start` begins, its quote or markdown included
    ('"Q2":', '**Q2:**'), so that the value before it ends there."""
    while start > floor and text[start - 1] in "\"'`*“‘ \t":
        start -= 1
    return start


def _unquoted(raw: str) -> Any:
    """A scraped value without the markdown, separators and quotes around it; None when there is none, or
    when it is a list or an object (not a value a line gives)."""
    value = raw.strip().strip("*`").strip()
    if len(value) >= 2 and _QUOTES.get(value[0]) == value[-1]:
        value = value[1:-1].strip()
    else:  # separators after it, and the opening quote of a key cut off after it ('7, "')
        value = value.rstrip(" \t,;.\"'`*“‘").strip("*`").strip()
    if not value or value[0] in "[{":
        return None
    return _scalar(value)


def _scrape(text: str) -> dict[str, Any]:
    """'Q1: 7' lines over the whole reply, the last mention of each question winning. A value ends at the
    end of its line, at a '}' or where the next question begins; a quoted value at its closing quote, unless
    more than punctuation follows it there ('"nc75012345" or "pt75975146"' is read whole). A value that runs
    into the end of a reply cut off inside its final JSON object (its last '{' is never closed) may be cut
    short ("2217903" cut to "22"), so it never replaces an earlier mention of its question; with none, it is
    read as written, as 0.2.0 read it (a reply cut off in its only answer object: "183." is 183), unless it
    holds no digit or letter ("-", the start of a negative number). A quoted value cut off there is dropped."""
    n, mentions = len(text), list(_MENTION.finditer(text))
    starts = [m.start() for m in mentions]
    lines, braces = _Positions(text, "\n"), _Positions(text, "}")
    closers = {close: _Positions(text, close) for close in set(_QUOTES.values())}
    last_open = text.rfind("{")
    cut_from = last_open if last_open >= 0 and text.find("}", last_open) < 0 else n
    found: dict[str, Any] = {}
    consumed = 0
    for k, m in enumerate(mentions):
        if m.start() < consumed:  # inside the quoted value of a question before it
            continue
        limit = min(_opening(text, starts[k + 1], m.end()) if k + 1 < len(starts) else n, braces.next(m.end(), n))
        i = _LEADING.match(text, m.end(), limit).end()  # the value may start on the next line
        if i >= limit:
            continue
        cut = m.start() > cut_from
        qid = normalize_qid(m.group(1))
        line_end = lines.next(i, n)
        stop = min(limit, line_end)
        close = _QUOTES.get(text[i])
        value: Any = None
        if close is not None:
            quote_end = closers[close].next(i + 1, n)
            if quote_end < line_end:
                after = quote_end + 1
                j = bisect.bisect_left(starts, after)
                stop = min(line_end, braces.next(after, n), _opening(text, starts[j], after) if j < len(starts) else n)
                rest = text[after:stop].lstrip(" \t*")
                if not rest or rest[0] in ",;.)]}":  # it ends there: '"nc75012345", "Q2": ...'
                    value, stop = text[i + 1:quote_end].strip(), after
            elif cut and line_end == n:
                continue  # a quoted value cut off at the end of the reply
        consumed = stop
        if value is None:
            raw = text[i:stop]
            if stop == n:  # the last value of the reply: no key cut off after it ('28, "Q3"'), and maybe cut short
                raw, ended = _before_cut_key(raw)
                if (cut and not ended and raw.rstrip("\"'`*“‘")[-1:] not in (" ", "\t", "\r")
                        and (qid in found or not _LETTER_OR_DIGIT.search(raw))):
                    continue  # it runs into the end of a reply cut off inside its final object
            if raw.strip().strip("*`").strip().rstrip(",;") in ("null", "None"):
                value = _NULL  # JSON's null (Python's None): no answer
            else:
                value = _unquoted(raw)
        if value is None or value == "" or (isinstance(value, str) and _PLACEHOLDER.fullmatch(value)):
            continue
        if qid:
            found[qid] = None if value is _NULL else value
    return found


_CUT_KEY = re.compile(r"[\s\"'`*“‘”’]*(?:[QqFf]\d*[\s\"'`*”’]*)?")
_LETTER_OR_DIGIT = re.compile(r"[^\W_]")
_NULL = object()


def _before_cut_key(raw: str) -> tuple[str, bool]:
    """The last value of a reply without what follows its comma when that is only the start of a key ('28, "Q3"'
    is 28), and whether a comma ended it there."""
    comma = raw.rfind(",")
    if comma >= 0 and _CUT_KEY.fullmatch(raw, comma + 1):
        return raw[:comma], True
    return raw, False


# ------------------------------------------------------------------------------------ parse


def parse_reply(text: Any, expected: Sequence[str] | None = None) -> ParsedReply:
    if not isinstance(text, str) or not text.strip():
        return ParsedReply(error="empty reply")
    whole, nested = _load(text.strip()), False
    if isinstance(whole, str):  # one JSON string: the object written inside it (double-encoded)
        whole, nested = _load(whole.strip()), True
    given = None if whole is _FAIL else _answers_from(whole)
    if given:
        answers, ambiguous = _combined([given])
        repeated = sorted(qid for qid, values in given.items() if len(values) > 1)
        strict = isinstance(whole, dict) and not nested and not repeated
        return ParsedReply(answers, "strict" if strict else "lenient", _repeated_note(repeated, ambiguous))
    wanted = set(expected or ())

    def coverage(answers: dict) -> int:
        return len(set(answers) & wanted) if wanted else len(answers)

    objects = _objects(text)
    if objects:
        most = max(coverage(answers) for answers, _ in objects)
        chosen = [answers for answers, _ in objects if coverage(answers) == most]
        # Of those, the ones in the shape the prompt asks for: working notes keyed by question ID beside them
        # ('Working: {"Q1": "3 PT + 4 NC"}') are no answer. Wrapped ones that disagree are still a hedge.
        wrapped = [answers for answers, shaped in objects if shaped and coverage(answers) == most]
        chosen = wrapped or chosen
        if wanted and most < len(wanted):  # e.g. one JSON object per question: merge them
            if len(set().union(*(answers for answers, _ in objects)) & wanted) > most:
                chosen = [answers for answers, _ in objects]
        answers, ambiguous = _combined(chosen)
        return ParsedReply(answers, "lenient", _repeated_note([], ambiguous))
    scraped = _scrape(text)
    if scraped:
        return ParsedReply(scraped, "lenient", "recovered by line scraping")
    return ParsedReply(error="no JSON object with answers found")


def _repeated_note(repeated: list[str], different: list[str]) -> str | None:
    """What the parse noted about questions given more than once (None: nothing). A question given
    different values is scored only if they all read the same (scoring.read_answer)."""
    notes = []
    if different:
        notes.append(f"given different values: {', '.join(sorted(different))}")
    same = [qid for qid in repeated if qid not in different]
    if same:
        notes.append(f"given more than once: {', '.join(same)}")
    return "; ".join(notes) or None
