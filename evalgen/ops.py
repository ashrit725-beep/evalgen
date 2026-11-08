"""Question operations and the code-only answer key.

A question's meaning is an *operation chain*, a list of small steps such as

    [{"op": "dedupe"},
     {"op": "mag_at_least", "value": 3.1},
     {"op": "in_region", "region": "Nevada"},
     {"op": "count"}]

The answer key is simply ``execute(chain, snapshot_events)``: plain Python over the frozen
snapshot, with no LLM and no text parsing. The same executor also powers the reference solvers
(fed with what they parsed from the digest text) and trap grading (fed with a digest read the
wrong way, see ``rules.Mishandle``). On digest data, dedupe applies the reading rules of the
records' domain (``resolver``: its ``Domain.resolve``; the earthquake rules by default).

Chains always have the shape: dedupe, zero or more filters, one terminal, optional post-steps
(convert, round).

The operations are generic over domains (see ``evalgen.domains``), and ``KINDS`` defines each kind
once. A step that reads a numeric field names it, and a value given in a unit says which:

    {"op": "at_least", "field": "temp", "value": 20.5, "unit": "°C"}

The 0.2.0 earthquake operations are aliases of generic kinds on fixed fields (``ALIASES``):
``mag_at_least`` is ``at_least`` on "mag", ``depth_shallower_than`` is ``below`` on "depth",
``in_region`` is ``in_group``, and so on. Earthquake chains keep their 0.2.0 dicts: aliases are
resolved on every read (``Step``) and never written back, so stored chains, their descriptions and
their answers are exactly as before. Records are read only through ``values``, ``group``, ``time``,
``lat``, ``lon`` and, in terminals, ``id``: earthquake Events and Reports provide these as well as
the generic Records and RecordReports.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .domains.base import Domain, RecordLike, ResolvedRecord
from .geo import haversine_km
from .models import Notice, Report
from .rules import Mishandle, Resolver, naive_events, resolve
from .units import LENGTH, Unit, find_unit
from .util import plain_number, utc_iso

EPS = 1e-9

# The 0.2.0 operation names, in their 0.2.0 order. FILTER_OPS is also the population that
# questions.compose_new_templates samples twin templates from, so its order is part of the output.
FILTER_OPS = (
    "mag_at_least",
    "mag_below",
    "depth_shallower_than",
    "depth_at_least",
    "within_distance",
    "time_between",
    "in_region",
)
TERMINAL_OPS = (
    "count",
    "argmax_mag",
    "rank_mag",
    "max_mag",
    "max_depth",
    "min_depth",
    "nearest_to",
    "nearest_distance",
    "compare_regions",
    "time_of_largest",
)

Units = Mapping[str, Sequence[Unit]]  # each generic field's units by key: a domain's `units`
Predicate = Callable[[RecordLike], bool]


@dataclass
class DigestData:
    """Reports and notices as read from a digest, before the reading rules are applied."""

    reports: list[Report]
    notices: list[Notice]


@dataclass(frozen=True)
class Alias:
    """A 0.2.0 operation name: a generic kind on a fixed field, complete with that field's units and
    its 0.2.0 answer type, so that an alias chain never needs a domain."""

    kind: str
    field: str | None = None
    units: tuple[Unit, ...] = ()  # what the step's 'unit' and a following convert refer to
    answer_type: str | None = None  # terminals
    keys: Mapping[str, str] = dataclasses.field(default_factory=dict)  # generic parameter -> 0.2.0 key


ALIASES: dict[str, Alias] = {
    "mag_at_least": Alias("at_least", "mag"),
    "mag_below": Alias("below", "mag"),
    "depth_shallower_than": Alias("below", "depth", LENGTH),
    "depth_at_least": Alias("at_least", "depth", LENGTH),
    "in_region": Alias("in_group", keys={"group": "region"}),
    "argmax_mag": Alias("argmax", "mag", answer_type="event_id"),
    "rank_mag": Alias("rank", "mag", answer_type="event_id"),
    "max_mag": Alias("max", "mag", answer_type="magnitude"),
    "max_depth": Alias("max", "depth", LENGTH, answer_type="depth"),
    "min_depth": Alias("min", "depth", LENGTH, answer_type="depth"),
    "compare_regions": Alias("compare_groups", answer_type="region"),
    "time_of_largest": Alias("time_of_max", "mag", answer_type="utc_time"),
}
# Answer types of generic ID and group answers when no domain is given: the 0.2.0 ones.
DEFAULT_ID_TYPE, DEFAULT_GROUP_TYPE = "event_id", "region"
DISTANCE_KINDS = ("within_distance", "nearest_distance")  # radii and distances: km, given in km or miles


class Step:
    """One chain step read generically: a 0.2.0 name resolves to its kind and field, and parameters
    are read under their generic names. The stored dict is never modified."""

    __slots__ = ("raw", "alias", "kind", "field", "_units")

    def __init__(self, raw: Mapping[str, Any], units: Units | None = None) -> None:
        self.raw = raw
        self.alias = ALIASES.get(raw["op"])
        self.kind: str = self.alias.kind if self.alias else raw["op"]
        self.field: str | None = self.alias.field if self.alias else raw.get("field")
        self._units = units

    @property
    def op(self) -> str:
        """The name as stored: a 0.2.0 alias or a generic kind."""
        return self.raw["op"]

    def key(self, name: str) -> str:
        """The stored name of a generic parameter: 'region' for the group of an in_region step."""
        return self.alias.keys.get(name, name) if self.alias else name

    def __getitem__(self, name: str) -> Any:
        return self.raw[self.key(name)]

    def get(self, name: str) -> Any:
        return self.raw.get(self.key(name))

    def units(self) -> Sequence[Unit]:
        """The units of this step's threshold or radius, or of its answer."""
        if self.alias is not None:
            return self.alias.units
        if self.kind in DISTANCE_KINDS:
            return LENGTH
        if self.field is None:
            return ()
        if self._units is None or self.field not in self._units:
            raise ValueError(f"the units of field {self.field!r} are unknown: pass the domain's units")
        return self._units[self.field]

    def unit(self, symbol: str) -> Unit:
        return find_unit(self.units(), symbol)


def distance_km(step: Mapping[str, Any], record: RecordLike) -> float:
    return haversine_km(step["lat"], step["lon"], record.lat, record.lon)


def by_field(rows: Iterable[ResolvedRecord], field: str) -> list[ResolvedRecord]:
    """Largest value first; ties broken by earlier time, then ID (the guards make ties impossible)."""
    return sorted(rows, key=lambda r: (-r.values[field], r.time, r.id))


# ------------------------------------------------------------------------------- filters
# Each builds the predicate for one step, so thresholds are converted once per step.


def canonical_threshold(s: Step) -> float:
    """A threshold filter's value in its field's canonical unit (a value without a unit is canonical)."""
    symbol = s.get("unit")
    return s["value"] if symbol is None else s.unit(symbol).to_canonical(s["value"])


def _at_least(s: Step) -> Predicate:
    field, limit = s.field, canonical_threshold(s) - EPS
    return lambda r: r.values[field] >= limit


def _below(s: Step) -> Predicate:
    field, limit = s.field, canonical_threshold(s) - EPS
    return lambda r: r.values[field] < limit


def _within_distance(s: Step) -> Predicate:
    limit = s.unit(s["unit"]).to_canonical(s["radius"]) + EPS
    return lambda r: distance_km(s.raw, r) <= limit


def _time_between(s: Step) -> Predicate:
    start, end = s["start"], s["end"]
    return lambda r: start <= r.time < end


def _in_group(s: Step) -> Predicate:
    group = s["group"]
    return lambda r: r.group == group


# ----------------------------------------------------------------------------- terminals


def _count(s: Step, rows: Sequence[ResolvedRecord]) -> int:
    return len(rows)


def _argmax(s: Step, rows: Sequence[ResolvedRecord]) -> str:
    return by_field(rows, s.field)[0].id


def _rank(s: Step, rows: Sequence[ResolvedRecord]) -> str | None:
    ordered = by_field(rows, s.field)
    return ordered[s["k"] - 1].id if len(ordered) >= s["k"] else None


def _max(s: Step, rows: Sequence[ResolvedRecord]) -> float:
    field = s.field
    return max(r.values[field] for r in rows)


def _min(s: Step, rows: Sequence[ResolvedRecord]) -> float:
    field = s.field
    return min(r.values[field] for r in rows)


def _nearest_to(s: Step, rows: Sequence[ResolvedRecord]) -> str:
    return min(rows, key=lambda r: (distance_km(s.raw, r), r.id)).id


def _nearest_distance(s: Step, rows: Sequence[ResolvedRecord]) -> float:
    return min(distance_km(s.raw, r) for r in rows)


def _compare_groups(s: Step, rows: Sequence[ResolvedRecord]) -> str:
    a = sum(1 for r in rows if r.group == s["a"])
    b = sum(1 for r in rows if r.group == s["b"])
    return s["a"] if a > b else s["b"] if b > a else "tie"


def _time_of_max(s: Step, rows: Sequence[ResolvedRecord]) -> str:
    """UTC HH:MM of the record with the largest value, truncated to the minute."""
    return datetime.fromtimestamp(by_field(rows, s.field)[0].time, tz=timezone.utc).strftime("%H:%M")


# ---------------------------------------------------------------------------- post-steps


def _convert(s: Step, value: Any, term: Step | None) -> Any:
    """From the canonical unit of the terminal's answer into the unit asked for."""
    if term is None:
        raise ValueError("convert needs a terminal step before it")
    return term.unit(s["to"]).from_canonical(value)


def _round(s: Step, value: Any, term: Step | None) -> Any:
    return round(value, s["ndigits"])


# ---------------------------------------------------------------------------- describing


def _threshold_text(s: Step, sign: str) -> str:
    unit = s.get("unit")
    return f"{s.field} {sign} {plain_number(s['value'])}" + ("" if unit is None else f" {unit}")


def _call(s: Step, *args: str) -> str:
    """'rank_mag(k=2)': a 0.2.0 name already says its field, a generic step shows it: 'rank(mag, k=2)'."""
    shown = ([s.field] if s.field is not None and s.alias is None else []) + list(args)
    return f"{s.op}({', '.join(shown)})" if shown else s.op


def _at_point(s: Step) -> str:
    return _call(s, f"{s['lat']:.3f}", f"{s['lon']:.3f}")


def _within_text(s: Step) -> str:
    return f"within {plain_number(s['radius'])} {s['unit']} of ({s['lat']:.3f}, {s['lon']:.3f})"


def _window_text(s: Step) -> str:
    return f"time in [{utc_iso(s['start'])}, {utc_iso(s['end'])})"


# --------------------------------------------------------------------------------- kinds


@dataclass(frozen=True)
class Kind:
    """One generic operation: its place in a chain, what it does and how a description shows it."""

    role: str  # "dedupe", "filter", "terminal" or "post"
    describe: Callable[[Step], str]
    test: Callable[[Step], Predicate] | None = None  # filters: the predicate of one step
    run: Callable[[Step, Sequence[ResolvedRecord]], Any] | None = None  # terminals: the answer
    apply: Callable[[Step, Any, Step | None], Any] | None = None  # post-steps: (step, value, terminal step)
    answer: str | None = None  # terminals: count | id | field | distance | group | utc_time
    on_empty: bool = False  # terminals: also answer when no record passes (the others give None)


KINDS: dict[str, Kind] = {
    "dedupe": Kind("dedupe", _call),
    "at_least": Kind("filter", lambda s: _threshold_text(s, "≥"), test=_at_least),
    "below": Kind("filter", lambda s: _threshold_text(s, "<"), test=_below),
    "within_distance": Kind("filter", _within_text, test=_within_distance),
    "time_between": Kind("filter", _window_text, test=_time_between),
    "in_group": Kind("filter", lambda s: f"{s.key('group')} = {s['group']}", test=_in_group),
    "count": Kind("terminal", _call, run=_count, answer="count", on_empty=True),
    "argmax": Kind("terminal", _call, run=_argmax, answer="id"),
    "rank": Kind("terminal", lambda s: _call(s, f"k={s['k']}"), run=_rank, answer="id"),
    "max": Kind("terminal", _call, run=_max, answer="field"),
    "min": Kind("terminal", _call, run=_min, answer="field"),
    "nearest_to": Kind("terminal", _at_point, run=_nearest_to, answer="id"),
    "nearest_distance": Kind("terminal", _at_point, run=_nearest_distance, answer="distance"),
    "compare_groups": Kind("terminal", lambda s: _call(s, f"{s['a']} vs {s['b']}"), run=_compare_groups,
                           answer="group", on_empty=True),
    "time_of_max": Kind("terminal", _call, run=_time_of_max, answer="utc_time"),
    "convert": Kind("post", lambda s: f"convert → {s['to']}", apply=_convert),
    "round": Kind("post", lambda s: f"round({s['ndigits']})", apply=_round),
}


def role(op: str) -> str | None:
    """'dedupe', 'filter', 'terminal' or 'post' for an op name (a 0.2.0 alias or a generic kind),
    or None for an unknown op."""
    alias = ALIASES.get(op)
    kind = KINDS.get(alias.kind if alias else op)
    return kind.role if kind else None


# ----------------------------------------------------------------------------- execution


def _answer(kind: Kind, step: Step, rows: Sequence[ResolvedRecord]) -> Any:
    return kind.run(step, rows) if rows or kind.on_empty else None


def passes(step: Mapping[str, Any], record: RecordLike, units: Units | None = None) -> bool:
    s = Step(step, units)
    kind = KINDS.get(s.kind)
    if kind is None or kind.role != "filter":
        raise ValueError(f"not a filter op: {s.op!r}")
    return kind.test(s)(record)


def terminal(step: Mapping[str, Any], rows: Sequence[ResolvedRecord]) -> Any:
    s = Step(step)
    kind = KINDS.get(s.kind)
    if kind is None or kind.role != "terminal":
        raise ValueError(f"not a terminal op: {s.op!r}")
    return _answer(kind, s, rows)


def _dedupe(rows: Any, mishandle: Mishandle | None, resolver: Resolver) -> list[Any]:
    if isinstance(rows, DigestData):
        return resolver(rows.reports, rows.notices, mishandle)
    return list(rows)  # snapshot records are already one row per real thing


def _as_events(rows: Any) -> list[Any]:
    if isinstance(rows, DigestData):  # dedupe was skipped: every report line counts
        return naive_events(rows.reports)
    return list(rows)


def rows_after_filters(chain: Sequence[dict], data: Any, *, mishandle: Mishandle | None = None,
                       skip: Sequence[str] = (), units: Units | None = None,
                       resolver: Resolver = resolve) -> list[Any]:
    """The records that survive dedupe and every filter (the question's 'evidence')."""
    rows = data
    for raw in chain:
        if raw["op"] in skip:
            continue
        step = Step(raw, units)
        kind = KINDS.get(step.kind)
        if step.kind == "dedupe":
            rows = _dedupe(rows, mishandle, resolver)
        elif kind is not None and kind.role == "filter":
            test = kind.test(step)
            rows = [r for r in _as_events(rows) if test(r)]
    return _as_events(rows)


def execute(chain: Sequence[dict], data: Any, *, mishandle: Mishandle | None = None,
            skip: Sequence[str] = (), units: Units | None = None, resolver: Resolver = resolve) -> Any:
    """Run an operation chain over snapshot events (answer key) or parsed digest data (solvers).

    `units` gives the units of generic fields (a domain's ``units``); 0.2.0 names carry their own.
    `resolver` gives the reading rules that dedupe applies to digest data: the records' domain's
    ``resolve`` (the default is the earthquake rules)."""
    rows = rows_after_filters(chain, data, mishandle=mishandle, skip=skip, units=units, resolver=resolver)
    value: Any = None
    term: Step | None = None
    for raw in chain:
        if raw["op"] in skip:
            continue
        step = Step(raw, units)
        kind = KINDS.get(step.kind)
        if kind is None:
            raise ValueError(f"unknown op: {step.op!r}")
        if kind.role == "terminal":
            value, term = _answer(kind, step, rows), step
        elif kind.role == "post" and value is not None:
            value = kind.apply(step, value, term)
    return value


def terminal_step(chain: Sequence[dict]) -> dict:
    return next(s for s in chain if role(s["op"]) == "terminal")


def answer_spec(chain: Sequence[dict], domain: Domain | None = None) -> tuple[str, str | None]:
    """(answer_type, unit) of a chain, e.g. ("depth", "mi").

    A 0.2.0 name keeps its 0.2.0 answer type. A generic terminal takes its type from the domain: its
    field's answer type, or the domain's ID or group type. Without a domain, IDs are event IDs and
    groups are regions, as in 0.2.0."""
    term = Step(terminal_step(chain))
    answer_type = term.alias.answer_type if term.alias else _answer_type(term, domain)
    unit = next((s["to"] for s in chain if s["op"] == "convert"), None)
    return answer_type, unit


def _answer_type(term: Step, domain: Domain | None) -> str:
    answer = KINDS[term.kind].answer
    if answer == "field":
        if domain is None:
            raise ValueError(f"the answer type of {term.op} on {term.field!r} comes from its domain")
        return domain.field(term.field).answer_type
    if answer == "id":
        return domain.id_answer_type if domain else DEFAULT_ID_TYPE
    if answer == "group":
        return domain.group_answer_type if domain else DEFAULT_GROUP_TYPE
    return answer  # count, distance and utc_time read the same in every domain


def describe_step(step: Mapping[str, Any]) -> str:
    s = Step(step)
    kind = KINDS.get(s.kind)
    return kind.describe(s) if kind else s.op


def describe(chain: Sequence[dict]) -> str:
    return " → ".join(describe_step(s) for s in chain)
