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
    LiveAdapter   a live source, fetched once, frozen and hashed
    TrapVocab     how a domain words its traps: re-reporting sources, notes, reasons and notices
    Domain        the base class; each domain module exports an instance as ``DOMAIN``

Earthquakes keep their 0.2.0 record types (``models.Event`` and ``models.Report``), which expose the
same generic names as Record and RecordReport: ``values``, ``group`` and ``source``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, Sequence

from ..geo import KM_PER_MILE

if TYPE_CHECKING:
    from ..models import Notice
    from ..rules import Mishandle


@dataclass(frozen=True)
class Unit:
    """A unit a field's value can be printed or asked in: canonical = value × scale + offset.

    Scale and offset are relative to the field's canonical unit, so one symbol can mean different
    things in different fields: km is the canonical unit of a depth, but 1 / 149597870.7 of the
    canonical au of a miss distance. The offset makes the map affine, as °F needs."""

    symbol: str  # printed after a number: "km", "mi", "°F"
    words: str  # written in a question: "km", "miles", "degrees Fahrenheit"
    scale: float = 1.0
    offset: float = 0.0

    def to_canonical(self, value: float) -> float:
        return value * self.scale + self.offset

    def from_canonical(self, value: float) -> float:
        return (value - self.offset) / self.scale


KM = Unit("km", "km")
MI = Unit("mi", "miles", KM_PER_MILE)
LENGTH = (KM, MI)  # depths, radii and distances: stored in km, printed in km or miles


def find_unit(units: Iterable[Unit], symbol: str) -> Unit:
    """The unit printed as `symbol` among `units`."""
    for unit in units:
        if unit.symbol == symbol:
            return unit
    raise ValueError(f"unknown unit {symbol!r}")


@dataclass(frozen=True)
class FieldSpec:
    """One numeric field of a domain's records, and what the generator needs to ask about it.

    Values are stored in `canonical_unit` (None for a unitless field such as a magnitude) and can be
    printed or asked in any of `units`. Thresholds are drawn from the `quantiles` range of an
    instance's values, moved by up to ± `jitter` (canonical units), asked in a unit drawn from
    `units` and rounded to `threshold_step` there, or to `coarse_step` half of the time when it is
    set. No record may lie within `margin` (abs, rel, in canonical units) of a threshold, and ranked
    answers need neighbours at least `rank_gap` apart. Answers that report this field are converted
    to a drawn unit, rounded to `decimals` and scored as `answer_type`, within `tolerance` (abs, rel).
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
class TrapVocab:
    """How a domain words its traps."""

    rereporting_sources: tuple[str, ...]  # sources that re-report other sources' records: "us", "at"
    duplicate_notes: tuple[str, ...]  # notes that point a re-report at its record: "same event as"
    retraction_reasons: tuple[str, ...]  # why a report was withdrawn: "quarry blast"
    retraction_notices: tuple[str, ...]  # phrases that announce a withdrawn report: "withdrawn"
    correction_notices: tuple[str, ...]  # phrases that announce a revised value: "revised to"


class Domain:
    """One source of records. A domain module subclasses this, sets the class attributes, implements
    the methods and exports an instance as ``DOMAIN`` (see evalgen.domains)."""

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

    id_pattern: str  # regex of a record ID in lower-cased text; the scorer uses it to find IDs too
    id_answer_type: str  # answer type naming a record (argmax, rank, nearest_to): "event_id"
    group_answer_type: str  # answer type naming a group (compare_groups): "region"
    trap_vocab: TrapVocab

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

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[ResolvedRecord]:
        """`n` seeded records inside the window that ends at `compiled_at`; never reads the clock."""
        raise NotImplementedError(f"{type(self).__name__} has no synthetic generator")

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID in `source`'s style, added to `taken`."""
        raise NotImplementedError(f"{type(self).__name__} has no ID scheme")

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID shaped like `example`, added to `taken`, so planted lines carry no tell-tale ID."""
        raise NotImplementedError(f"{type(self).__name__} has no ID scheme")

    # The generic versions of the three methods below, shared by every domain except earthquakes,
    # come with the generic record path (plan phase 1b).
    def rules_text(self) -> str:
        """The reading rules printed in every prompt."""
        raise NotImplementedError(f"{type(self).__name__} has no reading rules")

    def glossary(self, surface: Mapping[str, Any] | None = None) -> str:
        """The data dictionary printed in non-legacy prompts: field labels, units and conversions."""
        raise NotImplementedError(f"{type(self).__name__} has no glossary")

    def resolve(self, reports: Sequence[Any], notices: Sequence[Notice],
                mishandle: Mishandle | None = None) -> list[ResolvedRecord]:
        """The reading rules in code: one record per real thing (see rules.resolve)."""
        raise NotImplementedError(f"{type(self).__name__} has no reading rules")
