"""The domain interface: what one source of records declares so that evalgen can build tests on it.

A *domain* is one kind of record (earthquakes, weather observations, close approaches of near-Earth
objects, ...): its vocabulary, its numeric fields and their units, a seeded synthetic generator,
optionally a live source, its ID scheme, the wording of its traps and its reading rules. Operations,
templates, traps, genres and scoring are written once against this interface.

    FieldSpec     one numeric field: labels, units, answer type, guard margins and threshold draws
    Spatial       distance questions in a domain whose records have a location: radii and guards
    Lexicon       how questions word a domain's records and operations
    Record        one real thing in a generic domain: the ground truth, like an earthquake Event
    RecordReport  one digest line about a record, like an earthquake Report
    Glossary      a prompt's data dictionary of a domain's records (GlossaryField: one numeric field)
    LiveAdapter   a live source, fetched once, frozen and hashed
    Forecaster    delayed-resolution questions about a live source, graded once their window has passed
    TrapVocab     how a domain words its traps: re-reporting sources, notes, reasons and notices
    Surface       the words one instance uses for a domain whose surface is drawn per instance
    Domain        the base class; each domain module exports an instance as ``DOMAIN``

Units, and the exact codec of the numbers printed in them, live in evalgen.units.
``standard_lexicon`` words the questions of any domain from its own vocabulary and field labels, and
``record_rules`` writes the reading rules of any domain that prints none of its own.

Earthquakes keep their 0.2.0 record types (``models.Event`` and ``models.Report``), which expose the
same generic names as Record and RecordReport: ``values``, ``group`` and ``source``.
"""
from __future__ import annotations

import copy
import random
import textwrap
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

from ..scoring import AnswerType, answer_type_conflict
from ..units import LENGTH, ExactValues, Unit

if TYPE_CHECKING:
    from ..models import Notice
    from ..rules import Mishandle


@dataclass(frozen=True)
class FieldSpec(ExactValues):
    """One numeric field of a domain's records, and what the generator needs to ask about it.

    Values are stored in `canonical_unit` (None for a unitless field such as a magnitude), to
    `decimals` places, and can be printed or asked in any of `units`. Thresholds are drawn from the
    `quantiles` range of an instance's values, moved by up to ± `jitter` (canonical units), asked in
    a unit drawn from `units` and rounded to that unit's step (``Unit.step``, else `threshold_step`)
    there, or to `coarse_step` half of the time when it is set. No record may lie within `margin`
    (abs, rel, in canonical units) of a threshold, and ranked answers need neighbours at least
    `rank_gap` apart. Answers that report this field are converted to a drawn unit, rounded to the
    unit's places (``Unit.places``) and scored as `answer_type`: within `tolerance` (abs, rel) in the
    canonical unit, and in any other unit within its own tolerance or else the canonical one converted
    into it (``unit_tolerances``), so that a tolerance means the same in au as in km.
    """

    key: str  # name in a record's values: "mag", "depth"
    label: str  # name in text: "magnitude", "depth"
    canonical_unit: str | None
    units: tuple[Unit, ...]
    decimals: int  # decimal places of stored values and of rounded answers
    answer_type: str
    tolerance: tuple[float, float]  # (abs in the canonical unit, rel) of answers
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
    # False keeps `tolerance` as it is in every unit (0.2.0 depths: 0.3 km or 0.3 miles), which only
    # units of about the same size may share (see Domain.validate).
    tolerance_scales: bool = True

    def unit_tolerances(self) -> dict[str, tuple[float, float]]:
        """The (abs, rel) tolerance of answers asked in each unit whose tolerance is not the field's own:
        the unit's own tolerance, or else the absolute tolerance converted into the unit (1 km is
        1 / 149597870.7 au). A relative tolerance is the same in every unit of a scale."""
        out = {}
        for unit in self.units:
            if unit.tolerance is not None:
                out[unit.symbol] = unit.tolerance
            elif self.tolerance_scales and unit.scale != 1.0:
                out[unit.symbol] = (self.tolerance[0] / unit.scale, self.tolerance[1])
        return out


@dataclass(frozen=True)
class Spatial:
    """Distance questions in a domain whose records have a location (lat, lon): the radii a
    within_distance question asks about, and the guards that keep distance answers unambiguous.
    Distances are great-circle km, asked in km or miles (LENGTH) and answered to `decimals` places.
    Latitudes and longitudes are stored to `coord_decimals` places, which digests print exactly."""

    radii: Mapping[str, tuple[int, ...]]  # within_distance radii by unit symbol
    margin: tuple[float, float]  # (km, rel): no record lies within max(km, rel × radius) of a radius
    nearest_gap: tuple[float, float]  # (km, rel): the runner-up lies max(km, rel × nearest) further away
    nearest_max_km: float  # the nearest record lies at most this far from the question's point
    point_offset_km: tuple[float, float]  # a question's point lies this far from a random record
    point_label: str  # names a question's point: "station {source}-{number}"
    decimals: int = 1
    coord_decimals: int = 3


@dataclass(frozen=True)
class Lexicon:
    """How questions word a domain, keyed by the op names of its templates (see questions.op_name).

    A filter clause attaches to `records` ("earthquakes of magnitude 3 or greater"); a question
    without filters is about `all_records`. Wordings are str.format templates. Filter clauses use
    {value} (a threshold or radius with its unit), {place}, {start}, {end} and {group} (a group's
    name); questions use {records} (the records with their clauses), {unit}, {rounding} (the rounding
    asked for, as a sentence: "Round to 1 decimal place."), {nth}, {place}, {a} and {b}. Every wording
    of an op is equally likely. ``standard_lexicon`` words any domain from its own words."""

    records: str  # "earthquakes"
    all_records: str  # "earthquakes in the digest"
    filters: Mapping[str, tuple[str, ...]]
    questions: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {"records": self.records, "all_records": self.all_records,
                "filters": {op: list(wordings) for op, wordings in self.filters.items()},
                "questions": {op: list(wordings) for op, wordings in self.questions.items()}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Lexicon:
        return cls(d["records"], d["all_records"], {op: tuple(w) for op, w in d["filters"].items()},
                   {op: tuple(w) for op, w in d["questions"].items()})


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
    status: str  # one of its domain's statuses: "reviewed" or "preliminary"
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
    """One numeric field as a glossary defines it: its key, its label, its units with their
    conversions (Unit.for_readers) and the places its values are stored to, which is what a reader
    needs to read every printed value back exactly (see evalgen.units), and the key of the qualifier
    printed with its values, if it has one."""

    key: str  # its key in record values, in question chains and in log lines: "depth"
    label: str  # its name in the text: "depth"
    canonical_unit: str | None  # the unit values are stored in (None: unitless)
    units: tuple[Unit, ...]  # the units its values may be printed in
    decimals: int  # places of stored values, in the canonical unit
    qualifier: str | None = None  # the key of the qualifier printed with its values: "probe"


@dataclass(frozen=True)
class Glossary:
    """A prompt's data dictionary: what the records are called, how their IDs look, how their source
    and group are labelled, which status settles a record's values, whether records have a position
    (and to how many places), and the key, label, units and precision of each numeric field
    (``Domain.glossary``). Non-legacy prompts print it (evalgen.genres.base.render_glossary), so that a
    reader, and a genre's parser, learn every word, unit and ID pattern from the prompt itself: it is
    the only way to read a domain whose surface words are drawn per instance. Legacy (0.2.0) prompts
    print none. It never names the domain's registry name, which a held-out plugin keeps private."""

    noun: str
    plural: str
    id_label: str
    id_pattern: str  # a record ID, as a regex over lower-cased text (Domain.id_pattern)
    source_label: str
    group_label: str
    statuses: tuple[str, str]  # the status that settles a record's values, then the one that does not
    coord_decimals: int | None  # places of latitudes and longitudes; None: records have no position
    fields: tuple[GlossaryField, ...] = ()

    @property
    def located(self) -> bool:
        """Records have a position (lat, lon)."""
        return self.coord_decimals is not None

    @property
    def units(self) -> dict[str, tuple[Unit, ...]]:
        """Each field's units by key, as chains refer to them (like ``Domain.units``)."""
        return {f.key: f.units for f in self.fields}


class LiveAdapter(Protocol):
    """A live source. The harness fetches it once, freezes and SHA-256 hashes the raw bytes, then
    parses them; replaying the frozen bytes reproduces the snapshot (sources.build_domain_snapshot).
    When the source cannot be reached, or gives fewer than `min_records` usable records, the
    domain's synthetic generator is the fallback."""

    source: str  # the snapshot's source when it comes from here: "usgs"
    raw_filename: str  # the frozen copy in snapshot/, e.g. "raw_feed.geojson"
    user_agent: str  # sent with every request (some services refuse requests without one)
    min_records: int  # a feed with fewer usable records than this is not used

    def fetch(self) -> tuple[bytes, str]:
        """Download the source once: (raw bytes, URL). Raises on any network or HTTP error."""
        ...

    def parse(self, raw: bytes) -> tuple[list[ResolvedRecord], int, int]:
        """The frozen bytes as (records, compiled_at, window_hours), without the clock or network.
        Raises ValueError (or KeyError) on bytes it cannot read."""
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

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TrapVocab:
        return cls(**{name: tuple(words) for name, words in d.items()})


VOCABULARY = ("noun", "plural", "id_label", "source_label", "group_label")  # a domain's words for its records


@dataclass(frozen=True)
class Surface:
    """The words one instance uses for a domain whose surface is drawn per instance (Domain.draw_surface):
    what its records, IDs, sources, groups and fields are called, its units, the prefix of its IDs,
    the names of its groups and sources, its trap words, chatter and question wording. Whatever a
    surface leaves unset keeps the domain's own. An instance stores it as a plain dict (``to_dict``);
    ``Domain.bind`` applies it, and the bound domain renames the snapshot's records (``Domain.dress``).

    The snapshot keeps the domain's own names, so one snapshot serves every instance: a record's
    group and source are renamed through `groups` and `sources`, and its ID gets `id_prefix` in place
    of the domain's own (Domain.id_prefix)."""

    noun: str | None = None
    plural: str | None = None
    id_label: str | None = None
    source_label: str | None = None
    group_label: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)  # field key -> its label
    units: Mapping[str, tuple[Unit, ...]] = field(default_factory=dict)  # field key -> its units, canonical first
    id_prefix: str | None = None
    groups: Mapping[str, str] = field(default_factory=dict)  # a group as the snapshot names it -> its name here
    sources: Mapping[str, str] = field(default_factory=dict)  # a source as the snapshot names it -> its name here
    trap_vocab: TrapVocab | None = None
    chatter: tuple[str, ...] | None = None  # chatter lines (see Domain.chatter)
    lexicon: Lexicon | None = None  # None: the domain words the drawn words itself (Domain.surface_lexicon)

    def to_dict(self) -> dict[str, Any]:
        """The surface as JSON, without what it leaves unset."""
        d: dict[str, Any] = {name: getattr(self, name) for name in (*VOCABULARY, "id_prefix")
                             if getattr(self, name) is not None}
        for name in ("labels", "groups", "sources"):
            if getattr(self, name):
                d[name] = dict(getattr(self, name))
        if self.units:
            d["units"] = {key: [unit.to_dict() for unit in units] for key, units in self.units.items()}
        if self.trap_vocab is not None:
            d["trap_vocab"] = {name: list(words) for name, words in asdict(self.trap_vocab).items()}
        if self.chatter is not None:
            d["chatter"] = list(self.chatter)
        if self.lexicon is not None:
            d["lexicon"] = self.lexicon.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Surface:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"a surface has no {', '.join(sorted(unknown))}")
        return cls(
            **{name: d.get(name) for name in (*VOCABULARY, "id_prefix")},
            labels=dict(d.get("labels", {})),
            units={key: tuple(Unit.from_dict(u) for u in units) for key, units in d.get("units", {}).items()},
            groups=dict(d.get("groups", {})),
            sources=dict(d.get("sources", {})),
            trap_vocab=None if d.get("trap_vocab") is None else TrapVocab.from_dict(d["trap_vocab"]),
            chatter=None if d.get("chatter") is None else tuple(d["chatter"]),
            lexicon=None if d.get("lexicon") is None else Lexicon.from_dict(d["lexicon"]),
        )

    def dress(self, record: Any, id_prefix: str | None) -> Any:
        """`record` (a Record of the snapshot) as this surface names it: its group and source renamed,
        and `id_prefix`, the domain's own prefix, replaced by the surface's at the start of its ID. A
        surface that renames groups (or sources) renames every one, so that no name of the domain's own
        shows through and no two are renamed alike: one it has no name for is an error."""
        if self.id_prefix is None and not self.groups and not self.sources:
            return record
        if not isinstance(record, Record):
            raise TypeError(f"a surface renames Records, not {type(record).__name__}s")
        record_id = record.id
        if self.id_prefix is not None:
            if id_prefix is None or not record_id.startswith(id_prefix):
                raise ValueError(f"{record_id!r} does not start with the domain's ID prefix {id_prefix!r}")
            record_id = self.id_prefix + record_id[len(id_prefix):]
        return replace(record, id=record_id, group=_renamed(self.groups, record.group, "group"),
                       source=_renamed(self.sources, record.source, "source"))


def _renamed(names: Mapping[str, str], name: str, what: str) -> str:
    """`name` as a surface's `names` rename it (unrenamed when the surface renames none)."""
    if not names:
        return name
    if name not in names:
        raise ValueError(f"the surface has no name for the {what} {name!r}")
    return names[name]


# Chatter of a domain that has none of its own: lines that carry no data, worded from its vocabulary
# ({noun}, {plural}, {id_label}, {source_label}, {group_label}) and filled in by digest.make_chatter
# ({station}, {h1}, {h2}). None of them reads like a report or a notice.
GENERIC_CHATTER = (
    "The {source_label} {station} was offline from {h1} to {h2} UTC for maintenance; nothing it sent then is "
    "included.",
    "Editor's note: entries are reproduced exactly as each {source_label} filed them, time conventions included.",
    "Late {plural} may be listed in the next edition instead.",
    "Reminder: each {source_label} reports in its own units; read the unit printed with every value.",
    "This digest is compiled automatically and has not been checked by an editor.",
)

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
    live: LiveAdapter | None = None  # its live source, used off the legacy path (sources.build_domain_snapshot)
    forecaster: Forecaster | None = None  # set when instances from a live source also ask forecasts
    synthetic_window_hours: int = 24  # the time window its synthetic snapshots cover

    id_pattern: str  # regex of a record ID in lower-cased text; the scorer uses it to find IDs too
    id_prefix: str | None = None  # how every record ID begins, for a domain whose surface draws another
    id_answer_type: str  # answer type naming a record (argmax, rank, nearest_to): "event_id"
    group_answer_type: str  # answer type naming a group (compare_groups): "region"
    # The status of a report that settles its record's values, then that of a report that does not (a
    # re-report, or an early report the settling one supersedes), as digests print them.
    statuses: tuple[str, str] = ("reviewed", "preliminary")
    trap_vocab: TrapVocab
    chatter: tuple[str, ...] = GENERIC_CHATTER  # lines without data that digests mix in (digest.make_chatter)
    # Set by a domain whose surface words (record noun, labels, unit words, ID prefix, group names) are
    # drawn per instance; it implements draw_surface, and bind applies the drawn Surface.
    has_surface: bool = False
    # Set on the copy that bind returns: the surface it is bound to, and the domain it was bound from.
    surface: Surface | None = None
    unbound: Domain | None = None
    # Near twins: real records as close to each other as a re-report can be to its record (a station's
    # next observation, an aftershock). Off the legacy path a digest draws, for `twin_share` of its
    # records, the record of the snapshot with the same twin_key nearest in time, when it lies within
    # `twin_seconds` (pipeline.sample_with_slots). None: the domain's records have no near twins.
    twin_seconds: int | None = None
    twin_share: float = 0.0

    @property
    def has_location(self) -> bool:
        """Records have lat/lon, so within_distance, nearest_to and nearest_distance apply."""
        return self.spatial is not None

    @property
    def units(self) -> dict[str, tuple[Unit, ...]]:
        """Each field's units by key: what a step's 'unit' and a convert step refer to."""
        return {spec.key: spec.units for spec in self.fields}

    @property
    def words(self) -> dict[str, str]:
        """Its vocabulary by name (VOCABULARY), as chatter lines fill it in."""
        return {name: getattr(self, name) for name in VOCABULARY}

    def field(self, key: str) -> FieldSpec:
        for spec in self.fields:
            if spec.key == key:
                return spec
        raise KeyError(f"{self.name} has no field {key!r}")

    def answer_types(self) -> tuple[AnswerType, ...]:
        """How the scorer reads the answers about this domain's records: one type per numeric field
        (with the field's tolerance, and its tolerance in every other unit), the type of a record ID
        and the type of a group. Registering the domain registers them (see evalgen.domains)."""
        numbers = tuple(AnswerType(spec.answer_type, "number", spec.tolerance, unit_tolerance=spec.unit_tolerances())
                        for spec in self.fields)
        return (*numbers, AnswerType(self.id_answer_type, "id", id_pattern=self.id_pattern),
                AnswerType(self.group_answer_type, "group", suffix=self.group_label.casefold()))

    def validate(self) -> None:
        """Check, before anything is built from the domain, that it is complete and consistent:
          - it sets every attribute of the interface;
          - each field's canonical unit is one of its units, with scale 1 and offset 0, and no
            threshold margin is smaller than the rounding of a threshold restated in another unit;
          - each field's tolerance makes sense in every unit an answer can be asked in;
          - fields have keys and labels of their own, and a unit symbol has one wording throughout;
          - it has two status words and every kind of trap word that digests print;
          - each op name is the 0.2.0 alias of the kind and field it names (no other name runs),
            reading that field in the domain's units and answering with the domain's types;
          - P01-P12 bind to its primary and secondary fields, and its lexicon words every filter a
            template may use and every terminal the templates end in;
          - its answer types agree with each other and with the registered ones.
        Raises ValueError naming every problem."""
        missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(self, name)]
        if missing:
            raise ValueError(f"{type(self).__name__} does not set {', '.join(missing)}")
        problems = (self._unit_problems() + self._word_problems() + self._op_problems()
                    + self._answer_type_problems())
        if problems:
            raise ValueError(f"domain {self.name!r}: " + "; ".join(problems))

    def _word_problems(self) -> list[str]:
        """Words a digest prints that a reader must be able to tell apart, or that genres draw from."""
        problems = []
        for what in ("key", "label"):
            names = [getattr(spec, what) for spec in self.fields]
            if len(set(names)) != len(names):
                problems.append(f"two fields have the same {what}")
        wordings: dict[str, str] = {}
        for unit in [u for spec in self.fields for u in spec.units] + (list(LENGTH) if self.has_location else []):
            if wordings.setdefault(unit.symbol, unit.words) != unit.words:
                problems.append(f"the unit {unit.symbol} is worded both {wordings[unit.symbol]!r} and {unit.words!r}")
        if len(self.statuses) != 2 or len(set(self.statuses)) != 2 or not all(self.statuses):
            problems.append(f"its statuses {self.statuses!r} are not two different words")
        problems += [f"its trap vocabulary has no {name.replace('_', ' ')}"
                     for name, words in asdict(self.trap_vocab).items() if not words]
        return problems

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
            problems += _tolerance_problems(spec)
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
        "surface", which only non-legacy runs create), as a Surface's dict (Surface.to_dict). Only a
        domain with `has_surface` draws them."""
        raise NotImplementedError(f"{type(self).__name__} draws no surface words")

    def bind(self, surface: Mapping[str, Any] | None) -> Domain:
        """The domain as one instance words it: a copy that uses the drawn `surface` (a Surface's dict)
        wherever its own words would appear: its vocabulary, field labels and units, ID prefix, trap
        words, chatter, question wording (the surface's own, or else ``surface_lexicon``) and glossary.
        The copy renames the snapshot's records with ``dress``. Without a surface the domain is its own
        binding: a domain without surface words words every instance alike.

        Raises ValueError when the bound domain would not hold together: a word for a field it does
        not have, an ID prefix it has none to replace, two groups or two sources renamed alike, a unit
        whose answers the scorer would read with the wrong tolerance, or a guard margin that no longer
        covers the rounding of a threshold restated in another unit."""
        if surface is None:
            return self
        if self.unbound is not None:  # a surface replaces another: bind the domain itself
            return self.unbound.bind(surface)
        drawn = Surface.from_dict(surface)
        bound = copy.copy(self)
        bound.surface, bound.unbound = drawn, self
        for name in VOCABULARY:
            if getattr(drawn, name) is not None:
                setattr(bound, name, getattr(drawn, name))
        bound.fields = tuple(_bound_field(spec, drawn) for spec in self.fields)
        if drawn.id_prefix is not None:
            bound.id_prefix = drawn.id_prefix
        if drawn.trap_vocab is not None:
            bound.trap_vocab = drawn.trap_vocab
        if drawn.chatter is not None:
            bound.chatter = drawn.chatter
        bound.lexicon = bound.surface_lexicon() if drawn.lexicon is None else drawn.lexicon
        problems = (self._surface_problems(drawn) + bound._unit_problems() + bound._word_problems()
                    + bound._op_problems())
        if problems:
            raise ValueError(f"domain {self.name!r} cannot be bound to its surface: " + "; ".join(problems))
        return bound

    def _surface_problems(self, drawn: Surface) -> list[str]:
        keys = {spec.key for spec in self.fields}
        problems = [f"the surface words a field {key!r} that it does not have"
                    for key in sorted((set(drawn.labels) | set(drawn.units)) - keys)]
        if drawn.id_prefix is not None and self.id_prefix is None:
            problems.append("the surface draws an ID prefix, but its IDs have no prefix to replace")
        for name in ("groups", "sources"):
            names = list(getattr(drawn, name).values())
            if len(set(names)) != len(names):
                problems.append(f"the surface gives two {name} the same name")
        for spec in self.fields:
            units = drawn.units.get(spec.key)
            if units is not None and bool(units) != bool(spec.units):
                problems.append(f"the surface gives {spec.key} {'units' if units else 'no units'}, which it "
                                f"{'does not have' if units else 'has'}")
            for unit in units or ():
                problems += _drawn_unit_problems(spec, unit)
        return problems

    def surface_lexicon(self) -> Lexicon:
        """The question wording of this domain bound to a surface that brings none of its own: the
        standard wording of the bound words. A domain that words its own questions and draws surface
        words overrides it."""
        return standard_lexicon(self)

    def prompt_lexicon(self) -> Lexicon:
        """The question wording of prompts off the legacy path: the domain's lexicon (as bound to an
        instance's surface). Earthquakes, whose legacy prompts keep the 0.2.0 wording, override it."""
        return self.lexicon

    def dress(self, record: Any) -> Any:
        """A record of the snapshot as one instance shows it: renamed by the surface the domain is bound
        to (its ID prefix, group and source, see Surface.dress), or as it is when the domain is not
        bound to one. Questions, keys and digests are all made from dressed records."""
        if self.surface is None or self.unbound is None:
            return record
        return self.surface.dress(record, self.unbound.id_prefix)

    def sample_group(self, record: Any) -> str:
        """The group a digest's records are drawn by: digests focus on a few groups
        (pipeline.sample_events), so that questions about one have substance."""
        return record.group

    def sample_weight(self, record: Any) -> float:
        """How much likelier than others a record is to be drawn into a digest (1: all alike)."""
        return 1.0

    def twin_key(self, record: Any) -> Any:
        """What a record's near twins share (see twin_seconds): by default its group and its place, so
        a station's observations are each other's twins. None: the record has no twin."""
        return (record.group, record.place)

    def planted_value(self, spec: FieldSpec, value: float, like: float, peers: Sequence[float] = ()) -> float:
        """A value of `spec` that the trap builders plant, `value`, written as a real report would write it,
        never below zero unless the field is signed, and never -0.0. `like` is the real value it copies or
        stands for (a re-report's record's), and `peers` the values of the same field that the same source
        files in the digest. By default a source writes every value of a field to the same places, which
        its values show: the most places of `like` and `peers` (one value alone can end in zeros by chance).
        A domain whose precision belongs to the record (an object's H, a five-minute observation's whole
        degrees) writes to `like`'s own places, and one whose real values lie on a coarser grid (whole knots,
        hundredths of an inch) puts planted ones on it."""
        places = max(stored_places(v, spec.decimals) for v in (like, *peers))
        return round(value if spec.signed else max(0.0, value), places) + 0.0

    def rereport_ref(self, record: Any, slot: Any, rng: random.Random, taken: set[str]) -> str:
        """The reference of its own under which a source re-reports `record` in `slot`'s place (a record
        drawn with the digest but not kept, filed by that source): by default the slot's own ID, which no
        other line of the digest carries, so that a re-report's reference is a real ID of that source. A
        domain whose IDs tell something about the record (an object's designation) mimics the record's
        ID instead."""
        return slot.id

    def glossary(self, surface: Mapping[str, Any] | None = None) -> Glossary:
        """The data dictionary that non-legacy prompts print, in the words of `surface` (a drawn
        Surface's dict; None: the domain's own words, or those of the surface it is bound to): what the
        records, their IDs, sources and groups are called, the ID pattern, the status words, the
        precision of positions, and each field's key, label, units with their conversions, precision
        and qualifier."""
        domain = self.bind(surface)
        numeric = tuple(GlossaryField(s.key, s.label, s.canonical_unit, tuple(u.for_readers() for u in s.units),
                                      s.decimals, s.qualifier) for s in domain.fields)
        coord_decimals = None if domain.spatial is None else domain.spatial.coord_decimals
        return Glossary(domain.noun, domain.plural, domain.id_label, domain.id_pattern, domain.source_label,
                        domain.group_label, tuple(domain.statuses), coord_decimals, numeric)

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[ResolvedRecord]:
        """`n` seeded records inside the window that ends at `compiled_at`; never reads the clock."""
        raise NotImplementedError(f"{type(self).__name__} has no synthetic generator")

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID in `source`'s style, added to `taken`."""
        raise NotImplementedError(f"{type(self).__name__} has no ID scheme")

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID shaped like `example`, added to `taken`, so planted lines carry no tell-tale ID."""
        raise NotImplementedError(f"{type(self).__name__} has no ID scheme")

    def rules_text(self) -> str:
        """The reading rules printed in every prompt: the genre-neutral rules of every record domain,
        in this domain's words (``record_rules``). A domain with rules of its own overrides it."""
        return record_rules(self)

    @property
    def qualifier_keys(self) -> dict[str, str]:
        """The key each field's qualifier is stored under in a record's qualifiers, by field key."""
        return {spec.key: spec.qualifier for spec in self.fields if spec.qualifier is not None}

    def resolve(self, reports: Sequence[Any], notices: Sequence[Notice],
                mishandle: Mishandle | None = None) -> list[ResolvedRecord]:
        """The reading rules in code: one record per real thing (see rules.resolve). Every reader of
        a digest uses it: the oracle, trap grading and the generator's own checks. The generic rules
        settle a record by its report of the first of its statuses ("reviewed") and store a corrected
        qualifier under its field's qualifier key."""
        from ..rules import resolve  # imported here: rules imports this module

        return resolve(reports, notices, mishandle, authoritative=self.statuses[0], qualifiers=self.qualifier_keys)

    def to_generic(self, item: Any) -> Any:
        """A report or notice of this domain's digests in the form that genres write and read: a
        RecordReport, or a Notice whose correction names its field (field, new_value, old_value,
        qualifier). A Record domain's are in that form already; a domain whose digests hold record
        types of its own maps them (earthquakes: 0.2.0 Reports and magnitude corrections).
        ``from_generic`` is its inverse."""
        return item

    def from_generic(self, item: Any) -> Any:
        """A RecordReport or Notice that a genre read from a digest, as this domain's digests hold it:
        the inverse of ``to_generic``."""
        return item


def stored_places(value: float, decimals: int) -> int:
    """The decimal places a stored value is written to: the fewest, up to `decimals`, that hold it
    (15.0 has 0, 15.4 has 1, 21.35 has 2)."""
    for places in range(decimals):
        if round(value, places) == value:
            return places
    return decimals


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


def _tolerance_problems(spec: FieldSpec) -> list[str]:
    """A field's tolerance must mean the same in every unit an answer can be asked in: a relative
    tolerance means nothing on a scale whose zero is elsewhere (°F beside °C), and one absolute number
    cannot serve units of very different sizes (au and km) unless it scales."""
    problems = []
    for unit in spec.units:
        if unit.tolerance is not None:
            continue
        if unit.offset and spec.tolerance[1]:
            problems.append(f"{spec.key} has a relative tolerance, which means nothing in {unit.symbol}, whose zero "
                            f"is not that of {spec.canonical_unit}: give {unit.symbol} a tolerance of its own")
        if not spec.tolerance_scales and not 0.5 <= unit.scale <= 2.0:
            problems.append(f"the tolerance of {spec.key} is one number in {spec.canonical_unit} and in "
                            f"{unit.symbol}, units of very different sizes: let it scale, or give {unit.symbol} "
                            "a tolerance of its own")
    return problems


def _bound_field(spec: FieldSpec, drawn: Surface) -> FieldSpec:
    """A field as a surface words it: its drawn label, and its drawn units (the first one canonical)."""
    bound = replace(spec, label=drawn.labels.get(spec.key, spec.label))
    units = drawn.units.get(spec.key)
    if units is None:
        return bound
    return replace(bound, units=units, canonical_unit=units[0].symbol if units else None)


def _drawn_unit_problems(spec: FieldSpec, unit: Unit) -> list[str]:
    """The scorer reads an answer in a unit with the tolerance registered for that unit's symbol when
    the domain was loaded (FieldSpec.unit_tolerances), or else the field's own. A unit drawn for one
    instance must be scored as it should be: the domain declares it, or the field's tolerance holds in
    it unchanged (a relative tolerance holds in every unit of a scale)."""
    registered = spec.unit_tolerances().get(unit.symbol, spec.tolerance)
    wanted = replace(spec, units=(unit,)).unit_tolerances().get(unit.symbol, spec.tolerance)
    if registered == wanted:
        return []
    return [f"answers in the drawn unit {unit.symbol} would be scored within {registered}, not {wanted}: declare "
            f"{unit.symbol} among the units of {spec.key}, or give {spec.key} a relative tolerance only"]


# The wordings of a within_distance filter off the legacy path. Each names the distance before its number,
# so that joined to a threshold in a length unit ("with a depth of 16 km or more and ...") it cannot be
# read as the upper end of a range of that field, as the 0.2.0 wording "no more than {value} from
# {place}" can.
DISTANCE_WORDINGS = ("within {value} of {place}", "at a distance of no more than {value} from {place}")


def standard_lexicon(domain: Domain) -> Lexicon:
    """Wording for every operation a domain's templates use, made from the domain's own words: its
    noun, plural, ID label and group label, and each field's label. A domain can word its questions
    with it instead of writing a lexicon, and a domain whose words are drawn per instance is worded
    with it (Domain.surface_lexicon). A question about a field with units asks for the answer in a
    unit and ends with the rounding its chain applies ({unit} and {rounding})."""
    from ..questions import op_name  # imported here: questions imports this module

    noun, id_label, group = (_literal(word) for word in (domain.noun, domain.id_label, domain.group_label))
    filters: dict[str, tuple[str, ...]] = {}
    questions: dict[str, tuple[str, ...]] = {
        op_name(domain, "count"): ("How many distinct {records} are there?", "Count the distinct {records}."),
        op_name(domain, "compare_groups"): (
            f"Counting only {{records}}, which {group} has more of them: {{a}} or {{b}}? Answer \"tie\" if the "
            "counts are equal.",
            f"Which {group} has more {{records}}: {{a}} or {{b}}? Say \"tie\" if they are equal."),
    }
    for spec in domain.fields:
        label = _literal(spec.label)
        as_unit, in_unit, rounding = (", in {unit},", " in {unit}", " {rounding}") if spec.units else ("", "", "")
        wordings = {
            "at_least": (f"whose {label} is at least {{value}}", f"with {label} at or above {{value}}"),
            "below": (f"whose {label} is below {{value}}", f"with {label} under {{value}}"),
            "argmax": (f"Among the {{records}}, which one has the highest {label}? Give its {id_label}.",
                       f"What is the {id_label} of the one with the highest {label} among the {{records}}?"),
            "rank": (f"Among the {{records}}, which one has the {{nth}}-highest {label}? Give its {id_label}.",
                     f"Rank the {{records}} by {label}, highest first: which one is {{nth}}? Give its {id_label}."),
            "max": (f"What is the highest {label}{as_unit} among the {{records}}?{rounding}",
                    f"Among the {{records}}, what is the highest {label}{in_unit}?{rounding}"),
            "min": (f"What is the lowest {label}{as_unit} among the {{records}}?{rounding}",
                    f"Among the {{records}}, what is the lowest {label}{in_unit}?{rounding}"),
            "time_of_max": (f"At what time (UTC, HH:MM) was the highest {label} among the {{records}} recorded?",
                            f"When, in UTC (HH:MM), was the {noun} with the highest {label} among the {{records}} "
                            "recorded?"),
        }
        filters.update((op_name(domain, kind, spec.key), wordings[kind]) for kind in spec.filters)
        questions.update((op_name(domain, kind, spec.key), wordings[kind]) for kind in spec.terminals)
    filters[op_name(domain, "time_between")] = ("recorded between {start} and {end}",
                                                "dated from {start} up to {end}")
    filters[op_name(domain, "in_group")] = ("in {group}", f"whose {group} is {{group}}")
    if domain.has_location:
        filters[op_name(domain, "within_distance")] = DISTANCE_WORDINGS
        questions[op_name(domain, "nearest_to")] = (
            f"Which of the {{records}} is closest to {{place}}? Give its {id_label}.",
            f"Among the {{records}}, give the {id_label} of the one nearest to {{place}}.")
        questions[op_name(domain, "nearest_distance")] = (
            "How far from {place}, in {unit}, is the closest of the {records}? {rounding}",
            "Among the {records}, what is the distance in {unit} from {place} to the nearest one? {rounding}")
    return Lexicon(records=domain.plural, all_records=f"{domain.plural} in the digest", filters=filters,
                   questions=questions)


def _literal(text: str) -> str:
    """`text` inside a str.format template, standing for itself."""
    return text.replace("{", "{{").replace("}", "}}")


RULES_WIDTH = 100  # record_rules wraps each rule at this many columns

# What every genre needs a reader to know, whatever else a domain's rules say: a message's own time is
# not a record's, and a quote or read-back repeats what is printed elsewhere, while a forwarded or
# relayed message is the report or notice it carries. Some genres deliver every notice that way (an
# email thread forwards each one), so a forward must count. record_rules prints them as rules 6 and 7;
# the earthquake domain adds them to the 0.2.0 rules in its non-legacy prompts.
REPEAT_RULE = ("A quoted or read-back repeat of a report or notice printed elsewhere in the digest is not a new "
               "report or notice. A forwarded or relayed report or notice is not a repeat: it counts, once.")


def message_time_rule(record: str) -> str:
    """The rule on message times, about `record` ("a weather observation", "an earthquake")."""
    return f"The time at which a message was sent, forwarded or relayed is never the time of {record}."


def numbered_rules(rules: Sequence[str], start: int = 1) -> str:
    """Rules numbered from `start`, each wrapped at RULES_WIDTH columns under its own text, one per
    line and ending in a newline, as every reading-rules text prints them."""
    return "".join(textwrap.fill(f"{number}. {rule}", width=RULES_WIDTH, subsequent_indent=" " * len(f"{number}. "),
                                 break_long_words=False, break_on_hyphens=False) + "\n"
                   for number, rule in enumerate(rules, start=start))


def record_rules(domain: Domain) -> str:
    """The reading rules of a domain that prints none of its own, in its words: which report settles a
    record, how re-reports, corrections and retractions read, units, positions and times, and that
    nothing else in a digest carries data. They are genre-neutral: they say what a notice, a note, a
    repeat, a forward and a message time are in any kind of document, and they quote every phrase the
    domain's notices print (its trap vocabulary), so that no notice is a gotcha."""
    noun, id_label, source, group = domain.noun, domain.id_label, domain.source_label, domain.group_label
    final, draft = (f'"{status}"' for status in domain.statuses)
    vocab = domain.trap_vocab
    position = (" Positions are latitude then longitude in decimal degrees; signed values are negative for south "
                "and west. Distances are great-circle distances on a sphere of radius 6371 km."
                if domain.has_location else "")
    rules = (
        f"Every {noun} has exactly one {id_label}: the ID printed on its {final} report. A {noun} may appear in "
        f"several reports: a {draft} report from its own {source}, or a re-report from another {source}, either "
        f"under the same {id_label} or under its own reference with a note naming the {id_label} it repeats. "
        f"Reports with the same {id_label}, or that name it in such a note, are the same {noun}. Count each {noun} "
        f"once and refer to it by its {id_label}.",
        f"If reports of the same {noun} disagree, use the values from its {final} report: {draft} values are "
        "superseded.",
        "A notice is a message about an earlier report, not a report itself. A correction (a CORRECTION notice, a "
        f"notice saying that a value was revised, or any notice saying {_alternatives(vocab.correction_notices)}) "
        f"replaces the stated value for the named {noun}, even on a {final} report.",
        f"A retraction (a RETRACTION notice, a notice saying that a report was not a real {noun}, or any notice "
        f"saying {_alternatives(vocab.retraction_notices)}) means the named report was not a real {noun}: ignore "
        "that report completely.",
        "Each value may be given in any unit the glossary lists for its field: convert it as the glossary says. "
        "Values are compared as plain numbers once rules 2 and 3 have settled them." + position,
        "Times may be in UTC or in another time zone with its UTC offset stated. A report's time is when its "
        f"{noun} was recorded; a notice's time is when the notice was issued. {message_time_rule(f'a {noun}')} A "
        "time window includes its start and excludes its end.",
        REPEAT_RULE,
        f"A {group} in a question is the {group} printed on the report, matched exactly as written.",
        "Lines that are neither reports nor notices (headers, section titles, status messages, editor's notes, "
        "reminders, greetings, signatures) carry no data and change nothing.",
    )
    return "Reading rules:\n" + numbered_rules(rules)


def _alternatives(phrases: Sequence[str]) -> str:
    """'"a"', '"a" or "b"', '"a", "b" or "c"': phrases quoted, as a list of alternatives."""
    quoted = [f'"{phrase}"' for phrase in phrases]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]
