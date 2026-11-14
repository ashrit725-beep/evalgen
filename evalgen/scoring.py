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
parser or group words. The types above are built in. Every other type belongs to a domain, which
declares it (``Domain.answer_types``: one per field, plus the types of its record IDs and its
groups), and registering the domain registers its types (see evalgen.domains). A type that is not
registered yet makes ``get_answer_type`` discover the built-in domains first, so a fresh process
scores any of their types; a type that no domain declares is an error, never compared as raw values.
"""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Mapping

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

    kind            count | id | number | group | clock | probability
    tolerance       number: (abs, rel), right when |answer - key| <= max(abs, rel × |key|), in the
                    unit the question asked for; clock: (minutes, 0), counted around midnight
    unit_tolerance  number: the (abs, rel) of answers asked in a particular unit, by unit symbol, for
                    units whose values are on another scale (inches of mercury beside hectopascals)
    id_pattern      id: the regex that picks the ID out of a longer answer, matched in lower case
    parse           count, number, probability: a raw answer as one number, or None when ambiguous
    suffixes        group: the trailing words an answer may add, one of them, in lower case ("Nevada
                    region" is "Nevada"): a domain's group label, and every one its instances can draw
    """

    name: str
    kind: str
    tolerance: tuple[float, float] = (0.0, 0.0)
    unit_tolerance: Mapping[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
    id_pattern: str | None = None
    parse: Callable[[Any], float | None] = _number
    suffixes: tuple[str, ...] = ()

    def tolerance_in(self, unit: str | None) -> tuple[float, float]:
        """The (abs, rel) tolerance of an answer given in `unit` (None: a unitless answer)."""
        return self.unit_tolerance.get(unit, self.tolerance) if unit is not None else self.tolerance


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
    text = re.sub(r"\s+", " ", str(value)).strip().strip(".\"'\u201c\u201d\u2018\u2019").casefold()  # “” ‘’ too
    if text.startswith("the "):
        text = text[4:]
    if text in TIE_WORDS or re.search(r"\btied?\b", text):
        return "tie"
    if re.search(r"\bor\b", text):
        return None  # "Nevada or Utah" is not an answer
    if t.suffixes:  # one group word that ends the answer, the longest: "north sea sea area" is "north sea"
        text = re.sub(r"\s+(?:" + "|".join(re.escape(word) for word in t.suffixes) + ")$", "", text)
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
    AnswerType("region", "group", suffixes=("region",)),
    AnswerType("utc_time", "clock", (TIME_TOLERANCE_MIN, 0)),
    AnswerType("probability", "probability", parse=_parse_probability),
)}


def answer_type_conflict(answer_type: AnswerType) -> str | None:
    """Why `answer_type` cannot be registered, or None when it can."""
    if answer_type.kind not in _READERS:
        return f"unknown answer kind {answer_type.kind!r}"
    known = ANSWER_TYPES.get(answer_type.name)
    if known is not None and known != answer_type:
        return f"answer type {answer_type.name!r} is already registered differently"
    return None


def register_answer_type(answer_type: AnswerType) -> AnswerType:
    """Add an answer type. Registering an identical type again is harmless; registering a different
    type under a name that is taken is an error."""
    conflict = answer_type_conflict(answer_type)
    if conflict is not None:
        raise ValueError(conflict)
    ANSWER_TYPES[answer_type.name] = answer_type
    return answer_type


def get_answer_type(name: str) -> AnswerType:
    """The answer type called `name`. A name that is not registered yet first has the built-in
    domains discovered (registering a domain registers its types); a name that is still unknown
    raises KeyError."""
    known = ANSWER_TYPES.get(name)
    if known is None:
        from .domains import all_domains  # imported here: evalgen.domains imports this module

        all_domains()
        known = ANSWER_TYPES.get(name)
    if known is None:
        raise KeyError(f"unknown answer type {name!r}")
    return known


def normalize(answer_type: str, value: Any) -> Any:
    """Bring an answer (from a model, or the key) into a comparable canonical form, or None."""
    return read_answer(get_answer_type(answer_type), value)


def read_answer(answer_type: AnswerType, value: Any) -> Any:
    """normalize, for an answer type given as itself (one that need not be registered)."""
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != 1:
            return None  # several values offered: ambiguous
        value = value[0]
    return _READERS[answer_type.kind](answer_type, value)


def equivalent(answer_type: str, a: Any, b: Any, reference: Any = None, *, unit: str | None = None) -> bool:
    """Are two *normalised* answers the same, up to the scoring tolerance of answers in `unit`?"""
    known = get_answer_type(answer_type)
    if a is None or b is None:
        return a is None and b is None
    if known.kind == "number":
        abs_tol, rel_tol = known.tolerance_in(unit)
        scale = abs(reference) if reference is not None else max(abs(a), abs(b))
        return abs(a - b) <= max(abs_tol, rel_tol * scale) + 1e-9
    if known.kind == "clock":
        diff = abs(a - b) % 1440
        return min(diff, 1440 - diff) <= known.tolerance[0]
    return a == b


def is_correct(answer_type: str, key: Any, given: Any, *, unit: str | None = None) -> bool:
    """Is `given` right, for a question whose key is `key` and that asked for an answer in `unit`?"""
    k, g = normalize(answer_type, key), normalize(answer_type, given)
    return k is not None and g is not None and equivalent(answer_type, k, g, reference=k, unit=unit)


def same_answer(answer_type: str, a: Any, b: Any, *, unit: str | None = None) -> bool:
    """Raw-value comparison used by the generator (e.g. does a trap change an answer?)."""
    return equivalent(answer_type, normalize(answer_type, a), normalize(answer_type, b), unit=unit)


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
    """Difference of two proportions, p1 - p2, with the normal-approximation (Wald) 95% interval, the
    difference ± z sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2). The interval has zero width where neither side
    varies (each side at 0 or at 1), and it can pass ±1 with few rows or extreme rates; report.py cuts it to
    ±1 off the legacy path. A side without rows gives (0.0, -1.0, 1.0), 0.2.0's, which a legacy report prints
    as a gap of 0 and a report off the legacy path as n/a."""
    if n1 == 0 or n2 == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    gap = p1 - p2
    return (gap, gap - z * se, gap + z * se)


def stratified_difference(strata: list[tuple[int, int, int, int]], z: float = 1.96) -> tuple[float, float, float, float] | None:
    """Difference of two proportions compared like for like: the difference within each stratum (k1, n1,
    k2, n2), weighted by its share of the rows in the strata that both sides have, w = (n1 + n2) / their
    rows (the weights sum to 1); strata that only one side has are left out. The estimate uses each
    stratum's own proportions, k / n. Its variance is the sum of w² (a1 (1 - a1) / n1 + a2 (1 - a2) / n2),
    each proportion taken as a = (k + 0.5) / (n + 1), half a right and half a wrong answer added, so that a
    stratum whose sides are all right or all wrong keeps a positive variance; the 95% interval is the
    estimate ± z times its square root (it can pass ±1: report.py cuts it to ±1 off the legacy path).
    Returns (difference, ci_low, ci_high, share of the rows in the strata compared), or None when no
    stratum has both sides."""
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


def design_effect(rows: Iterable[tuple[Hashable, bool, bool]]) -> float:
    """How much more a difference of two proportions varies because its rows come in clusters whose rows
    go together, as the questions and renderings of one instance do: Kish's design effect, the ratio of
    the difference's cluster-robust (sandwich) variance, with the clusters as the independent units and
    the small-sample factor G / (G - 1) over its G clusters, to its variance with every row independent.
    Each row is (cluster, on the first side, correct). Answers that do not cluster give a ratio near 1,
    which falls below 1 by chance about half the time; the design effect is never below 1, so that this
    never narrows an interval. It is 1 when either side has no rows, when either variance is 0, and with
    fewer than two clusters, which say nothing about the variance between clusters."""
    rows = list(rows)
    counts = {True: [0, 0], False: [0, 0]}
    for _, first, correct in rows:
        counts[bool(first)][0] += bool(correct)
        counts[bool(first)][1] += 1
    (k1, n1), (k2, n2) = counts[True], counts[False]
    if not n1 or not n2:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    independent = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    totals: dict[Hashable, float] = {}
    for cluster, first, correct in rows:
        residual = (bool(correct) - p1) / n1 if first else -(bool(correct) - p2) / n2
        totals[cluster] = totals.get(cluster, 0.0) + residual
    g = len(totals)
    if g < 2 or independent <= 0:
        return 1.0
    # math.fsum, exactly rounded: the built-in sum() of floats adds them one by one up to Python 3.11 and with
    # compensation from 3.12, so the last digits of every gap interval that this widens differed between the two.
    return max(1.0, g / (g - 1) * math.fsum(t * t for t in totals.values()) / independent)


def widened(estimate: float, low: float, high: float, factor: float) -> tuple[float, float]:
    """The interval (low, high) around `estimate`, each half `factor` times as wide (an interval of a
    difference of proportions may then pass ±1; report.py cuts it to ±1)."""
    return estimate - factor * (estimate - low), estimate + factor * (high - estimate)


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
