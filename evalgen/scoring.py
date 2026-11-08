"""Automatic scoring: answer normalisation, per-type correctness, consistency, summary statistics.

Scoring rules (public, like everything else):
  count             exact integer match
  event_id          exact match (case-insensitive)
  region            exact match after case/space normalisation; "tie" (or equal/same/...) for ties
  magnitude         |answer - key| <= 0.05
  depth             |answer - key| <= max(0.3, 2% of key)   (in the unit the question asked for)
  distance          |answer - key| <= max(1.5, 3% of key)   (in the unit the question asked for)
  utc_time          within 1 minute (HH:MM, 24-hour)
  probability       not scored for accuracy; Brier score once the forecast resolves
A string answer is read only when it is unambiguous: "7 events" is 7 and "5,088.8 km" is 5088.8,
but a string holding several different values ("7 or 8", "3-4", two event IDs) counts as no answer.

Answer types live in a registry (``ANSWER_TYPES``). A type's kind (count, id, number, group, clock
or probability) says how its answers are read; the type adds its tolerance, ID pattern, number
parser or group suffix. The types above are built in, and a domain registers its own with
``register_answer_type``: one per field, plus the types of its record IDs and its groups.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from .models import EVENT_ID_PATTERN

TIME_TOLERANCE_MIN = 1
TIE_WORDS = {"tie", "tied", "equal", "same", "neither", "both", "draw", "even"}
WORD_NUMBERS = {word: i for i, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_GROUPED = re.compile(r"(?<=\d)[,   ](?=\d{3}(?!\d))")  # thousands separators: 5,088.8
_HEDGE = re.compile(r"\b(?:or|between|either)\b|\d\s*(?:-|–|to)\s*\d", re.IGNORECASE)
_FRACTION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:/|in|out of)\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_CLOCK = re.compile(r"(\d{1,2})[:h](\d{2})")


def _number(value: Any) -> float | None:
    """A single unambiguous number, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip().lower().replace("−", "-")  # typographic minus sign: "−1.1 km"
        if text in WORD_NUMBERS:
            return float(WORD_NUMBERS[text])
        text = _GROUPED.sub("", text)
        if _HEDGE.search(text):
            return None
        numbers = {float(x) for x in _NUMBER.findall(text)}
        return numbers.pop() if len(numbers) == 1 else None
    return None


def _parse_probability(value: Any) -> float | None:
    """A number, a percentage ("15%"), a fraction ("1 in 3") or a decimal comma ("0,3")."""
    if not isinstance(value, str):
        return _number(value)
    fraction = _FRACTION.match(value)
    if fraction:
        return float(fraction.group(1)) / float(fraction.group(2)) if float(fraction.group(2)) else None
    x = _number(value.replace(",", ".") if re.fullmatch(r"\s*\d+,\d+\s*", value) else value)
    return x / 100.0 if x is not None and "%" in value else x


# ------------------------------------------------------------------------- answer types


@dataclass(frozen=True)
class AnswerType:
    """How the answers of one type are read and compared.

    kind        count | id | number | group | clock | probability
    tolerance   number: (abs, rel), right when |answer - key| <= max(abs, rel × |key|);
                clock: (minutes, 0), counted around midnight
    id_pattern  id: the regex that picks the ID out of a longer answer, matched in lower case
    parse       count, number, probability: a raw answer as one number, or None when ambiguous
    suffix      group: a trailing word an answer may add, in lower case ("Nevada region" is "Nevada")
    """

    name: str
    kind: str
    tolerance: tuple[float, float] = (0.0, 0.0)
    id_pattern: str | None = None
    parse: Callable[[Any], float | None] = _number
    suffix: str | None = None


def _read_count(t: AnswerType, value: Any) -> int | None:
    x = t.parse(value)
    return int(round(x)) if x is not None and abs(x - round(x)) < 1e-6 else None


def _read_number(t: AnswerType, value: Any) -> float | None:
    return t.parse(value)


def _read_probability(t: AnswerType, value: Any) -> float | None:
    x = t.parse(value)
    return x if x is not None and 0.0 <= x <= 1.0 else None


def _read_id(t: AnswerType, value: Any) -> str | None:
    text = str(value).strip().lower()
    ids = {m.group(0) for m in re.finditer(t.id_pattern, text)} if t.id_pattern else set()
    if len(ids) == 1:
        return ids.pop()
    return None if ids else (text or None)


def _read_group(t: AnswerType, value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value)).strip().strip(".\"'").casefold()
    if text.startswith("the "):
        text = text[4:]
    if text in TIE_WORDS or re.search(r"\btied?\b", text):
        return "tie"
    if re.search(r"\bor\b", text):
        return None  # "Nevada or Utah" is not an answer
    if t.suffix:
        text = re.sub(r"\s+" + re.escape(t.suffix) + "$", "", text)
    return text or None


def _read_clock(t: AnswerType, value: Any) -> int | None:
    clocks = {(int(h), int(m)) for h, m in _CLOCK.findall(str(value))}
    if len(clocks) != 1:
        return None
    hh, mm = clocks.pop()
    return hh * 60 + mm if hh < 24 and mm < 60 else None


_READERS: dict[str, Callable[[AnswerType, Any], Any]] = {
    "count": _read_count,
    "id": _read_id,
    "number": _read_number,
    "group": _read_group,
    "clock": _read_clock,
    "probability": _read_probability,
}

ANSWER_TYPES: dict[str, AnswerType] = {t.name: t for t in (
    AnswerType("count", "count"),
    AnswerType("event_id", "id", id_pattern=EVENT_ID_PATTERN),
    AnswerType("magnitude", "number", (0.05, 0.0)),
    AnswerType("depth", "number", (0.3, 0.02)),
    AnswerType("distance", "number", (1.5, 0.03)),
    AnswerType("region", "group", suffix="region"),
    AnswerType("utc_time", "clock", (TIME_TOLERANCE_MIN, 0)),
    AnswerType("probability", "probability", parse=_parse_probability),
)}
# The numeric types and their (abs, rel) tolerance. Trap placement uses it too, to keep a misread
# answer clear of the scoring edge.
TOLERANCE: dict[str, tuple[float, float]] = {
    name: t.tolerance for name, t in ANSWER_TYPES.items() if t.kind == "number"
}


def register_answer_type(answer_type: AnswerType) -> AnswerType:
    """Add an answer type. Registering an identical type again is harmless; registering a different
    type under a name that is taken is an error."""
    if answer_type.kind not in _READERS:
        raise ValueError(f"unknown answer kind {answer_type.kind!r}")
    known = ANSWER_TYPES.get(answer_type.name)
    if known is not None and known != answer_type:
        raise ValueError(f"answer type {answer_type.name!r} is already registered differently")
    ANSWER_TYPES[answer_type.name] = answer_type
    if answer_type.kind == "number":
        TOLERANCE[answer_type.name] = answer_type.tolerance
    return answer_type


def normalize(answer_type: str, value: Any) -> Any:
    """Bring an answer (from a model, or the key) into a comparable canonical form, or None.
    An answer of an unregistered type is compared as it is."""
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != 1:
            return None  # several values offered: ambiguous
        value = value[0]
    known = ANSWER_TYPES.get(answer_type)
    return _READERS[known.kind](known, value) if known else value


def equivalent(answer_type: str, a: Any, b: Any, reference: Any = None) -> bool:
    """Are two *normalised* answers the same, up to the scoring tolerance?"""
    if a is None or b is None:
        return a is None and b is None
    known = ANSWER_TYPES.get(answer_type)
    if known is not None and known.kind == "number":
        abs_tol, rel_tol = known.tolerance
        scale = abs(reference) if reference is not None else max(abs(a), abs(b))
        return abs(a - b) <= max(abs_tol, rel_tol * scale) + 1e-9
    if known is not None and known.kind == "clock":
        diff = abs(a - b) % 1440
        return min(diff, 1440 - diff) <= known.tolerance[0]
    return a == b


def is_correct(answer_type: str, key: Any, given: Any) -> bool:
    k, g = normalize(answer_type, key), normalize(answer_type, given)
    return k is not None and g is not None and equivalent(answer_type, k, g, reference=k)


def same_answer(answer_type: str, a: Any, b: Any) -> bool:
    """Raw-value comparison used by the generator (e.g. does a trap change an answer?)."""
    return equivalent(answer_type, normalize(answer_type, a), normalize(answer_type, b))


# ----------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def difference_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference of two proportions with a normal-approximation 95% interval."""
    if n1 == 0 or n2 == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    gap = p1 - p2
    return (gap, gap - z * se, gap + z * se)


def stratified_difference(strata: list[tuple[int, int, int, int]], z: float = 1.96) -> tuple[float, float, float, float] | None:
    """Difference of two proportions compared like for like: within each stratum (k1, n1, k2, n2),
    weighted by the stratum's share of all questions. Strata that only one side has are left out.
    Returns (difference, ci_low, ci_high, share of questions covered), or None."""
    usable = [s for s in strata if s[1] and s[3]]
    total = sum(s[1] + s[3] for s in strata)
    covered = sum(s[1] + s[3] for s in usable)
    if not usable:
        return None
    gap = variance = 0.0
    for k1, n1, k2, n2 in usable:
        w = (n1 + n2) / covered
        gap += w * (k1 / n1 - k2 / n2)
        a1, a2 = (k1 + 0.5) / (n1 + 1), (k2 + 0.5) / (n2 + 1)  # keeps the variance positive in perfect strata
        variance += w * w * (a1 * (1 - a1) / n1 + a2 * (1 - a2) / n2)
    se = math.sqrt(variance)
    return gap, gap - z * se, gap + z * se, covered / total


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar test for paired answers: do the questions that exactly one of two models
    got right split evenly between them? (Binomial test on the discordant pairs, p = 1/2.)"""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_difference_ci(both: int, a_only: int, b_only: int, neither: int,
                         z: float = 1.96) -> tuple[float, float, float]:
    """Accuracy of A minus accuracy of B on the same questions, with the Agresti-Min 95% interval
    (half a count added to each cell of the paired 2x2 table). Returns (difference, low, high)."""
    n = both + a_only + b_only + neither
    if n == 0:
        return (0.0, -1.0, 1.0)
    b, c, m = a_only + 0.5, b_only + 0.5, n + 2
    centre = (b - c) / m
    se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / m)) / m
    return ((a_only - b_only) / n, max(-1.0, centre - z * se), min(1.0, centre + z * se))


def brier(probability: float, outcome: int) -> float:
    return (probability - outcome) ** 2
