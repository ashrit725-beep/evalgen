"""The domain interface: what one source of records declares so that evalgen can build tests on it.

A *domain* is one kind of record (earthquakes, weather observations, close approaches of near-Earth
objects, ...): its vocabulary, its numeric fields and their units, a seeded synthetic generator,
optionally a live source, its ID scheme, the wording of its traps and its reading rules. Operations,
templates, traps, genres and scoring are written once against this interface.

    Unit          a unit a value is printed in, and its affine map to the field's canonical unit
    FieldSpec     one numeric field: labels, units, answer type, guard margins and threshold draws
    Spatial       distance questions in a domain whose records have a location: radii and guards
    Lexicon       how questions word a domain's records and operations
    Record        one real thing in a generic domain: the ground truth, like an earthquake Event
    RecordReport  one digest line about a record, like an earthquake Report
    Glossary      a prompt's data dictionary of a domain's records (GlossaryField: one numeric field)
    LiveAdapter   a live source, fetched once, frozen and hashed
    Forecaster    delayed-resolution questions about a live source, graded once their window has passed
    TrapVocab     how a domain words its traps: re-reporting sources, notes, reasons and notices
    Domain        the base class; each domain module exports an instance as ``DOMAIN``

Earthquakes keep their 0.2.0 record types (``models.Event`` and ``models.Report``), which expose the
same generic names as Record and RecordReport: ``values``, ``group`` and ``source``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol, Sequence

from ..geo import KM_PER_MILE
from ..scoring import AnswerType, answer_type_conflict

if TYPE_CHECKING:
    from ..models import Notice
    from ..rules import Mishandle


@dataclass(frozen=True)
class Unit:
    """A unit a field's value can be printed or asked in: canonical = value × scale + offset.

    Scale and offset are relative to the field's canonical unit, so one symbol can mean different
    things in different fields: km is the canonical unit of a depth, but 1 / 149597870.7 of the
    canonical au of a miss distance. The offset makes the map affine, as °F needs. A unit whose values
    are on another scale than the field's stored ones (inches of mercury beside hectopascals) sets its
    own precision and scoring tolerance; the others use the field's."""

    symbol: str  # printed after a number: "km", "mi", "°F"
    words: str  # written in a question: "km", "miles", "degrees Fahrenheit"
    scale: float = 1.0
    offset: float = 0.0
    decimals: int | None = None  # places of answers and restated thresholds in this unit (None: the field's)
    tolerance: tuple[float, float] | None = None  # (abs, rel) of answers in this unit (None: the field's)

    def to_canonical(self, value: float) -> float:
        return value * self.scale + self.offset

    def from_canonical(self, value: float) -> float:
        return (value - self.offset) / self.scale

    def places(self, default: int) -> int:
        """Decimal places of a value written in this unit: its own, or else `default` (the field's)."""
        return default if self.decimals is None else self.decimals


KM = Unit("km", "km")
MI = Unit("mi", "miles", KM_PER_MILE)
LENGTH = (KM, MI)  # depths, radii and distances: stored in km, printed in km or miles


def find_unit(units: Iterable[Unit], symbol: str) -> Unit:
    """The unit printed as `symbol` among `units`."""
    for unit in units:
        if unit.symbol == symbol:
            return unit
    raise ValueError(f"unknown unit {symbol!r}")


class ExactValues:
    """Printing a field's stored values so that a reader recovers them exactly (FieldSpec, and
    GlossaryField for a reader that knows only the prompt). Stored values have `decimals` places in
    the canonical unit. Converting a printed value back is rarely exact (most tenths of a degree
    Celsius have no decimal form in Fahrenheit that converts back to them), so a printed value is
    converted and then rounded to the stored places."""

    decimals: int

    def decode(self, printed: float, unit: Unit | None = None) -> float:
        """The stored value that a number printed in `unit` (None: the canonical unit) stands for."""
        return round(printed if unit is None else unit.to_canonical(printed), self.decimals)

    def encode(self, value: float, unit: Unit | None = None) -> str:
        """A stored value printed in `unit` (None: the canonical unit), with the fewest decimals that
        decode back to it. A value with more places than the field stores has no such form."""
        shown = value if unit is None else unit.from_canonical(value)
        for places in range(16):
            text = f"{shown:.{places}f}"
            if self.decode(float(text), unit) == value:
                return text
        raise ValueError(f"{value!r} cannot be printed in {unit.symbol if unit else 'its canonical unit'} so "
                         f"that it reads back exactly: is it stored to {self.decimals} decimal places?")


@dataclass(frozen=True)
class FieldSpec(ExactValues):
    """One numeric field of a domain's records, and what the generator needs to ask about it.

    Values are stored in `canonical_unit` (None for a unitless field such as a magnitude), to
    `decimals` places, and can be printed or asked in any of `units`. Thresholds are drawn from the
    `quantiles` range of an instance's values, moved by up to ± `jitter` (canonical units), asked in
    a unit drawn from `units` and rounded to `threshold_step` there, or to `coarse_step` half of the
    time when it is set. No record may lie within `margin` (abs, rel, in canonical units) of a
    threshold, and ranked answers need neighbours at least `rank_gap` apart. Answers that report this
    field are converted to a drawn unit, rounded to the unit's places (``Unit.places``) and scored as
    `answer_type`, within the unit's tolerance or else `tolerance` (abs, rel).
    """

    key: str  # name in a record's values: "mag", "depth"
    label: str  # name in text: "magnitude", "depth"
    canonical_unit: str | None
    units: tuple[Unit, ...]
    decimals: int  # decimal places of stored values and of rounded answers
    answer_type: str
    tolerance: tuple[float, float]
    margin: tuple[float, float]
    rank_gap: float
    quantiles: tuple[float, float]
    threshold_step: float
    # Generic filter and terminal kinds that apply to it. The order of `filters` is the order in which
    # twin templates sample them (see questions.filter_ops).
    filters: tuple[str, ...]  # "at_least", "below"
    terminals: tuple[str, ...]  # "argmax", "rank", "max", "min", "time_of_max"
    qualifier: str | None = None  # qualifier stored with the value and revised with it: "mag_type"
    signed: bool = False  # values can be negative (a depth above sea level)
    correctable: bool = False  # correction notices can revise it
    jitter: float = 0.0  # thresholds move up to this far (canonical units) from the drawn quantile
    coarse_step: float | None = None  # half of the thresholds are rounded to this step instead
    positive_threshold: bool = False  # a threshold must be above zero in the unit it is asked in


@dataclass(frozen=True)
class Spatial:
    """Distance questions in a domain whose records have a location (lat, lon): the radii a
    within_distance question asks about, and the guards that keep distance answers unambiguous.
    Distances are great-circle km, asked in km or miles (LENGTH) and answered to `decimals` places."""

    radii: Mapping[str, tuple[int, ...]]  # within_distance radii by unit symbol
    margin: tuple[float, float]  # (km, rel): no record lies within max(km, rel × radius) of a radius
    nearest_gap: tuple[float, float]  # (km, rel): the runner-up lies max(km, rel × nearest) further away
    nearest_max_km: float  # the nearest record lies at most this far from the question's point
    point_offset_km: tuple[float, float]  # a question's point lies this far from a random record
    point_label: str  # names a question's point: "station {source}-{number}"
    decimals: int = 1


@dataclass(frozen=True)
class Lexicon:
    """How questions word a domain, keyed by the op names of its templates (see questions.op_name).

    A filter clause attaches to `records` ("earthquakes of magnitude 3 or greater"); a question
    without filters is about `all_records`. Wordings are str.format templates. Filter clauses use
    {value} (a threshold or radius with its unit), {place}, {start}, {end} and {group} (a group's
    name); questions use {records} (the records with their clauses), {unit}, {nth}, {place}, {a} and
    {b}. Every wording of an op is equally likely."""

    records: str  # "earthquakes"
    all_records: str  # "earthquakes in the digest"
    filters: Mapping[str, tuple[str, ...]]
    questions: Mapping[str, tuple[str, ...]]


class RecordLike(Protocol):
    """What filters read from a row. Events, Reports, Records and RecordReports all qualify."""

    @property
    def time(self) -> int: ...

    @property
    def lat(self) -> float | None: ...

    @property
    def lon(self) -> float | None: ...

    @property
    def values(self) -> Mapping[str, float]: ...

    @property
    def group(self) -> str: ...


class ResolvedRecord(RecordLike, Protocol):
    """A record after the reading rules (an Event or a Record): terminals also read its ID."""

    @property
    def id(self) -> str: ...


@dataclass(frozen=True)
class Record:
    """One real thing in a generic domain: a row of the frozen snapshot (the ground truth)."""

    id: str
    time: int  # Unix seconds, UTC
    lat: float | None  # None in domains without a location
    lon: float | None
    group: str
    source: str  # who reported it (the domain's source_label names the role)
    values: dict[str, float]  # canonical units, by FieldSpec key
    qualifiers: dict[str, str] = field(default_factory=dict)  # e.g. a value's scale or method
    place: str | None = None


@dataclass
class RecordReport:
    """One digest line about a record: what some source said about it."""

    ref: str  # the identifier printed on the line
    source: str
    status: str  # "reviewed" or "preliminary"
    time: int
    lat: float | None
    lon: float | None
    group: str
    values: dict[str, float]  # canonical units, by FieldSpec key
    qualifiers: dict[str, str] = field(default_factory=dict)
    place: str | None = None
    same_as: str | None = None  # printed cross-reference to the record this line duplicates
    # Generator-side bookkeeping. Never rendered into the digest text.
    role: str = "primary"  # primary | duplicate | conflict | phantom
    trap_id: str | None = None


@dataclass(frozen=True)
class GlossaryField(ExactValues):
    """One numeric field as a glossary defines it: its label, its units with their conversions and
    the places its values are stored to, which is what a reader needs to read every printed value
    back exactly (see ExactValues.decode)."""

    key: str  # its key in record values and in question chains: "depth"
    label: str  # its name in the text: "depth"
    canonical_unit: str | None  # the unit values are stored in (None: unitless)
    units: tuple[Unit, ...]  # the units its values may be printed in
    decimals: int  # places of stored values, in the canonical unit


@dataclass(frozen=True)
class Glossary:
    """A prompt's data dictionary: what the records are called, how their ID, source and group are
    labelled, and the label, units and precision of each numeric field (``Domain.glossary``).
    Non-legacy prompts print it (prompt assembly, plan phase 1b), so that a reader, and a genre's
    parser, learn every label and unit from the prompt itself: it is the only way to read a domain
    whose surface words are drawn per instance. Legacy (0.2.0) prompts print none. It never names the
    domain's registry name, which a held-out plugin keeps private."""

    noun: str
    plural: str
    id_label: str
    source_label: str
    group_label: str
    fields: tuple[GlossaryField, ...] = ()

    @property
    def units(self) -> dict[str, tuple[Unit, ...]]:
        """Each field's units by key, as chains refer to them (like ``Domain.units``)."""
        return {f.key: f.units for f in self.fields}


class LiveAdapter(Protocol):
    """A live source. The harness fetches it once, freezes and SHA-256 hashes the raw bytes, then
    parses them; replaying the frozen bytes reproduces the snapshot. When the source cannot be
    reached, the domain's synthetic generator is the fallback."""

    raw_filename: str  # the frozen copy in snapshot/, e.g. "raw_feed.geojson"
    user_agent: str  # sent with every request (some services refuse requests without one)

    def fetch(self) -> tuple[bytes, str]:
        """Download the source once: (raw bytes, URL). Raises on any network or HTTP error."""
        ...

    def parse(self, raw: bytes) -> tuple[list[Record], int, int]:
        """The frozen bytes as (records, compiled_at, window_hours), without the clock or network."""
        ...


@dataclass(frozen=True)
class Forecaster:
    """Delayed-resolution questions about a live source: asked when a digest is compiled and graded
    once the window they ask about has passed, so their answers cannot be known in advance."""

    source: str  # the snapshot source whose instances ask one: "usgs"
    # (records, rng, compiled_at, window_hours): an instance's forecast question and its resolution rule
    make: Callable[[Sequence[Any], random.Random, int, int], dict[str, Any]]
    text: Callable[[dict[str, Any]], str]  # the question as a prompt words it
    resolve: Callable[..., dict[str, Any]]  # (forecasts, now[, fetch]): grades, in place, those that can be


@dataclass(frozen=True)
class TrapVocab:
    """How a domain words its traps."""

    rereporting_sources: tuple[str, ...]  # sources that re-report other sources' records: "us", "at"
    duplicate_notes: tuple[str, ...]  # notes that point a re-report at its record: "same event as"
    retraction_reasons: tuple[str, ...]  # why a report was withdrawn: "quarry blast"
    retraction_notices: tuple[str, ...]  # phrases that announce a withdrawn report: "withdrawn"
    correction_notices: tuple[str, ...]  # phrases that announce a revised value: "revised to"


REQUIRED_ATTRIBUTES = ("name", "noun", "plural", "id_label", "source_label", "group_label", "fields", "roles",
                       "lexicon", "id_pattern", "id_answer_type", "group_answer_type", "trap_vocab")


class Domain:
    """One source of records. A domain module subclasses this, sets the class attributes, implements
    the methods and exports an instance as ``DOMAIN`` (see evalgen.domains), which the registry
    checks with ``validate`` before anything is built from it."""

    # Vocabulary, as questions and digests word it.
    name: str  # registry key: "earthquakes"
    noun: str  # one record: "earthquake"
    plural: str  # "earthquakes"
    id_label: str  # "event ID"
    source_label: str  # who reports records: "network"
    group_label: str  # what records are grouped by: "region"

    fields: tuple[FieldSpec, ...]
    roles: Mapping[str, str]  # field keys in the 0.2.0 roles: {"primary": "mag", "secondary": "depth"}
    spatial: Spatial | None = None  # set when records have a location
    lexicon: Lexicon
    # The names this domain's chains give generic kinds, by (kind, field): earthquakes keep their 0.2.0
    # names (ops.ALIASES). Any other kind is stored generically, e.g. {"op": "at_least", "field": "temp"}.
    op_names: Mapping[tuple[str, str | None], str] = MappingProxyType({})
    live: LiveAdapter | None = None
    forecaster: Forecaster | None = None  # set when instances from a live source also ask forecasts

    id_pattern: str  # regex of a record ID in lower-cased text; the scorer uses it to find IDs too
    id_answer_type: str  # answer type naming a record (argmax, rank, nearest_to): "event_id"
    group_answer_type: str  # answer type naming a group (compare_groups): "region"
    trap_vocab: TrapVocab
    # Set by a domain whose surface words (record noun, labels, unit words, ID prefix, group names) are
    # drawn per instance; it implements draw_surface and bind.
    has_surface: bool = False

    @property
    def has_location(self) -> bool:
        """Records have lat/lon, so within_distance, nearest_to and nearest_distance apply."""
        return self.spatial is not None

    @property
    def units(self) -> dict[str, tuple[Unit, ...]]:
        """Each field's units by key: what a step's 'unit' and a convert step refer to."""
        return {spec.key: spec.units for spec in self.fields}

    def field(self, key: str) -> FieldSpec:
        for spec in self.fields:
            if spec.key == key:
                return spec
        raise KeyError(f"{self.name} has no field {key!r}")

    def answer_types(self) -> tuple[AnswerType, ...]:
        """How the scorer reads the answers about this domain's records: one type per numeric field
        (with the field's tolerance, and that of every unit that sets its own), the type of a record
        ID and the type of a group. Registering the domain registers them (see evalgen.domains)."""
        numbers = tuple(AnswerType(spec.answer_type, "number", spec.tolerance,
                                   unit_tolerance={u.symbol: u.tolerance for u in spec.units
                                                   if u.tolerance is not None})
                        for spec in self.fields)
        return (*numbers, AnswerType(self.id_answer_type, "id", id_pattern=self.id_pattern),
                AnswerType(self.group_answer_type, "group", suffix=self.group_label.casefold()))

    def validate(self) -> None:
        """Check, before anything is built from the domain, that it is complete and consistent:
          - it sets every attribute of the interface;
          - each field's canonical unit is one of its units, with scale 1 and offset 0, and no
            threshold margin is smaller than the rounding of a threshold restated in another unit;
          - each op name is the 0.2.0 alias of the kind and field it names (no other name runs),
            reading that field in the domain's units and answering with the domain's types;
          - P01-P12 bind to its primary and secondary fields, and its lexicon words every filter a
            template may use and every terminal the templates end in;
          - its answer types agree with each other and with the registered ones.
        Raises ValueError naming every problem."""
        missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(self, name)]
        if missing:
            raise ValueError(f"{type(self).__name__} does not set {', '.join(missing)}")
        problems = self._unit_problems() + self._op_problems() + self._answer_type_problems()
        if problems:
            raise ValueError(f"domain {self.name!r}: " + "; ".join(problems))

    def _unit_problems(self) -> list[str]:
        problems = []
        for spec in self.fields:
            canonical = [u for u in spec.units if u.symbol == spec.canonical_unit]
            if len({u.symbol for u in spec.units}) != len(spec.units):
                problems.append(f"{spec.key} lists a unit symbol twice")
            if spec.canonical_unit is None and spec.units:
                problems.append(f"{spec.key} has units but no canonical unit")
            elif spec.canonical_unit is not None and not canonical:
                problems.append(f"{spec.key} is stored in {spec.canonical_unit}, which is not one of its units")
            elif canonical and (canonical[0].scale, canonical[0].offset) != (1.0, 0.0):
                problems.append(f"the canonical unit of {spec.key} must have scale 1 and offset 0")
            if spec.filters:
                problems += _margin_problems(spec.key, spec.margin[0], spec.units, spec.decimals)
        if self.spatial is not None:
            problems += _margin_problems("distance", self.spatial.margin[0], LENGTH, self.spatial.decimals)
        return problems

    def _op_problems(self) -> list[str]:
        from ..ops import ALIASES, KINDS  # imported here: both modules import this one
        from ..questions import filter_ops, public_templates

        problems = []
        keys = {spec.key for spec in self.fields}
        for (kind, key), name in self.op_names.items():
            alias = ALIASES.get(name)
            if alias is None or (alias.kind, alias.field) != (kind, key):
                problems.append(f"op name {name!r} is not the 0.2.0 alias of {kind}({key or ''})")
                continue
            if key is not None and key not in keys:
                problems.append(f"op name {name!r} reads {key!r}, which is not one of its fields")
                continue
            answer = KINDS[kind].answer
            wanted = {"field": self.field(key).answer_type if key else None, "id": self.id_answer_type,
                      "group": self.group_answer_type}.get(answer, answer)
            if alias.answer_type is not None and alias.answer_type != wanted:
                problems.append(f"op name {name!r} answers as {alias.answer_type!r}, not as {wanted!r}")
            if key is not None and alias.units != self.field(key).units:
                problems.append(f"op name {name!r} reads {key} in other units than the field's")
        try:
            public = public_templates(self)
        except (KeyError, ValueError) as exc:
            return problems + [f"P01-P12 do not bind to its fields ({exc})"]
        unworded = sorted({op for op in filter_ops(self) if op not in self.lexicon.filters}
                          | {t.terminal for t in public if t.terminal not in self.lexicon.questions})
        if unworded:
            problems.append(f"its lexicon has no wording for {', '.join(unworded)}")
        return problems

    def _answer_type_problems(self) -> list[str]:
        problems = []
        declared: dict[str, AnswerType] = {}
        for answer_type in self.answer_types():
            if declared.setdefault(answer_type.name, answer_type) != answer_type:
                problems.append(f"it declares answer type {answer_type.name!r} twice, differently")
            conflict = answer_type_conflict(answer_type)
            if conflict is not None:
                problems.append(conflict)
        return problems

    def draw_surface(self, rng: random.Random) -> dict[str, Any]:
        """The surface words of one instance, drawn from its own stream (seed path "instance", i,
        "surface", which only non-legacy runs create). Only a domain with `has_surface` draws them."""
        raise NotImplementedError(f"{type(self).__name__} draws no surface words")

    def bind(self, surface: Mapping[str, Any] | None) -> Domain:
        """The domain as one instance words it: a copy that uses the drawn `surface` words wherever its
        own would appear (labels, lexicon, unit words, IDs, glossary). A domain without surface words
        words every instance alike, so it is its own binding."""
        return self

    def glossary(self) -> Glossary:
        """The data dictionary that non-legacy prompts print: what the records, their IDs, sources
        and groups are called, and each field's label, units with their conversions and precision.
        Bind a drawn surface first to get the glossary of one instance."""
        numeric = tuple(GlossaryField(s.key, s.label, s.canonical_unit, s.units, s.decimals) for s in self.fields)
        return Glossary(self.noun, self.plural, self.id_label, self.source_label, self.group_label, numeric)

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[ResolvedRecord]:
        """`n` seeded records inside the window that ends at `compiled_at`; never reads the clock."""
        raise NotImplementedError(f"{type(self).__name__} has no synthetic generator")

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID in `source`'s style, added to `taken`."""
        raise NotImplementedError(f"{type(self).__name__} has no ID scheme")

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID shaped like `example`, added to `taken`, so planted lines carry no tell-tale ID."""
        raise NotImplementedError(f"{type(self).__name__} has no ID scheme")

    # The generic versions of the two methods below, shared by every domain except earthquakes, come
    # with the generic record path (plan phase 1b).
    def rules_text(self) -> str:
        """The reading rules printed in every prompt."""
        raise NotImplementedError(f"{type(self).__name__} has no reading rules")

    def resolve(self, reports: Sequence[Any], notices: Sequence[Notice],
                mishandle: Mishandle | None = None) -> list[ResolvedRecord]:
        """The reading rules in code: one record per real thing (see rules.resolve). Every reader of
        a digest uses it: the oracle, trap grading and the generator's own checks."""
        raise NotImplementedError(f"{type(self).__name__} has no reading rules")


def _margin_problems(name: str, margin: float, units: Sequence[Unit], decimals: int) -> list[str]:
    """A threshold restated in another unit is rounded to that unit's places, so it moves by up to half
    their last digit; the guard margin must cover the move, or the restatement could cross a record."""
    problems = []
    for unit in units:
        shift = 0.5 * 10 ** -unit.places(decimals) * unit.scale  # in the canonical unit
        if margin < shift:
            problems.append(f"the {name} margin {margin:g} is smaller than the rounding of a threshold "
                            f"restated in {unit.symbol} ({shift:g})")
    return problems
