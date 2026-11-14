"""Plain data records that flow through one evaluation cycle.

Event      one real earthquake: a row of the frozen snapshot (the ground truth)
Report     one report of an earthquake as a digest prints it (what some network said about it): a
           line, a table row, a form, a radio message, ...
Notice     a correction or retraction notice, in whatever form its genre prints it: a NOTICE, UPDATE
           or CORRECTION line, a forwarded email, a pinned chat message, a footnote, an amendment
           form, a dispatch broadcast
Trap       a deliberately injected pitfall whose correct handling is known in advance
Question   one composed question; its operation chain is its machine-readable meaning
Rendering  one textual presentation of an instance (original or a metamorphic variant)
Instance   everything needed to ask, answer and grade one test item

Event and Report also answer to the generic names that evalgen.domains uses for every record type
(``values``, ``group``, ``source``). The fields added in 0.3.0 are optional and left out of the JSON
while unset, as they are on the legacy path, so a legacy instance (--mix legacy, the 0.2.0 run)
serializes exactly as in 0.2.0. Off the legacy path every instance records its domain and split, and
each rendering its genre: earthquakes in the classic genre too (--mix earthquakes:classic), whose
prompts also print the glossary and the rules of every genre.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

# Event IDs look like USGS ones: nc75012345, us7000tvs3, aka2026vupcsv (always lower case).
EVENT_ID_PATTERN = r"[a-z]{2,4}\d[0-9a-z]{3,14}"
EVENT_ID_RE = re.compile(EVENT_ID_PATTERN)


class _GenericNames:
    """The generic record names of evalgen.domains, mapped onto the earthquake attributes. They are
    read-only properties, not dataclass fields, so constructors, asdict and the JSON are unchanged."""

    mag: float
    depth_km: float
    region: str
    network: str

    @property
    def values(self) -> dict[str, float]:
        """The numeric fields by generic key: 'mag', and 'depth' in km."""
        return {"mag": self.mag, "depth": self.depth_km}

    @property
    def group(self) -> str:
        return self.region

    @property
    def source(self) -> str:
        return self.network


@dataclass(frozen=True)
class Event(_GenericNames):
    id: str
    time: int  # Unix seconds, UTC
    lat: float
    lon: float
    depth_km: float
    mag: float
    mag_type: str
    region: str
    network: str
    place: str | None = None


@dataclass
class Report(_GenericNames):
    ref: str  # the identifier printed on the line
    network: str
    status: str  # "reviewed" or "preliminary"
    time: int
    lat: float
    lon: float
    depth_km: float
    mag: float
    mag_type: str
    region: str
    place: str | None = None
    same_as: str | None = None  # printed cross-reference: "same event as <event id>"
    # Generator-side bookkeeping. Never rendered into the digest text.
    role: str = "primary"  # primary | duplicate | conflict | phantom
    trap_id: str | None = None


@dataclass
class Notice:
    kind: str  # "retraction" or "correction"
    target: str  # ID of the report or event concerned
    time: int
    reason: str | None = None  # retractions
    new_mag: float | None = None  # corrections
    old_mag: float | None = None
    mag_type: str | None = None
    trap_id: str | None = None
    # A correction in a generic domain: the field it revises, the new and old value (canonical
    # units) and the new qualifier. Earthquake corrections use new_mag, old_mag and mag_type instead.
    field: str | None = None
    new_value: float | None = None
    old_value: float | None = None
    qualifier: str | None = None

    OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = ("field", "new_value", "old_value", "qualifier")


@dataclass
class Trap:
    trap_id: str
    kind: str  # duplicate | retraction | conflicting_magnitude (earthquakes) | conflicting_value (other domains)
    form: str  # same_id | cross_ref | phantom | correction | second_report
    target: str
    description: str
    mishandles: list[dict[str, list]]  # the careless readings of this trap (see rules.Mishandle)
    affected: dict[str, Any] = field(default_factory=dict)  # qid -> answer if ONLY this trap is misread
    fallen: dict[str, list] = field(default_factory=dict)  # qid -> every wrong answer explained by misreading
    #                                                        this trap (alone or together with other traps)


@dataclass
class Question:
    qid: str
    template_id: str
    pool: str  # "public" or "new"
    chain: list[dict[str, Any]]
    # A registered answer type (scoring.get_answer_type): the 0.2.0 ones (count, event_id, magnitude, depth,
    # distance, region, utc_time) or a domain's own (Domain.answer_types: weather_temperature, neo_approach_id, ...).
    answer_type: str
    unit: str | None = None
    evidence: list[str] = field(default_factory=list)  # IDs of events that pass the filters
    # Code-computed difficulty features, used to compare the template pools like for like:
    # trap_exposure = how many traps change this answer when misread (capped at 2);
    # filters_matter = whether dropping every filter would change the answer.
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class Rendering:
    variant: str  # "v0" is the original; "v1", "v2", ... are metamorphic re-renderings
    seed: int
    style: dict[str, Any]
    digest: str
    question_text: dict[str, str]
    prompt: str
    prompt_sha256: str
    genre: str | None = None  # the genre it was rendered in; unset on the legacy (classic) path

    OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = ("genre",)


@dataclass
class Instance:
    instance_id: str
    seed: int
    compiled_at: int
    window_hours: int
    event_ids: list[str]
    reports: list[Report]
    notices: list[Notice]
    chatter: list[str]
    traps: list[Trap]
    questions: list[Question]
    answer_key: dict[str, Any]
    renderings: list[Rendering]
    forecast: dict[str, Any] | None = None
    # Set only in non-legacy runs: the domain, whether it is public or held out, and the drawn
    # surface words of a domain that has them.
    domain: str | None = None
    split: str | None = None  # "public" or "held_out"
    surface: dict[str, Any] | None = None
    # Off the legacy path, for each chatter line, the time after which a genre may post it: the last minute of the
    # window it tells of, or None (digest.timed_chatter), so that a rendering can be drawn again from the instance.
    chatter_after: list[int | None] | None = None

    OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = ("domain", "split", "surface", "chatter_after")

    def to_dict(self) -> dict[str, Any]:
        """asdict, without the optional keys that are unset (here and in every notice and rendering)."""
        d = _without_unset(asdict(self), Instance.OPTIONAL_KEYS)
        d["notices"] = [_without_unset(n, Notice.OPTIONAL_KEYS) for n in d["notices"]]
        d["renderings"] = [_without_unset(r, Rendering.OPTIONAL_KEYS) for r in d["renderings"]]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instance":
        """An instance as to_dict wrote it. Its reports are earthquake Reports, or the RecordReports of
        a generic domain (whose lines carry their values by field key)."""
        return cls(
            instance_id=d["instance_id"],
            seed=d["seed"],
            compiled_at=d["compiled_at"],
            window_hours=d["window_hours"],
            event_ids=list(d["event_ids"]),
            reports=[_report(r) for r in d["reports"]],
            notices=[Notice(**n) for n in d["notices"]],
            chatter=list(d["chatter"]),
            traps=[Trap(**t) for t in d["traps"]],
            questions=[Question(**q) for q in d["questions"]],
            answer_key=dict(d["answer_key"]),
            renderings=[Rendering(**r) for r in d["renderings"]],
            forecast=d.get("forecast"),
            domain=d.get("domain"),
            split=d.get("split"),
            surface=d.get("surface"),
            chatter_after=d.get("chatter_after"),
        )

    def question(self, qid: str) -> Question:
        return next(q for q in self.questions if q.qid == qid)


def _without_unset(d: dict[str, Any], optional: tuple[str, ...]) -> dict[str, Any]:
    """`d` without the optional keys whose value is None, in the same order."""
    return {k: v for k, v in d.items() if v is not None or k not in optional}


def _report(d: dict[str, Any]) -> Any:
    """A digest line from its JSON: a generic RecordReport (it has values), else an earthquake Report."""
    if "values" not in d:
        return Report(**d)
    from .domains.base import RecordReport  # imported here: evalgen.domains imports this module

    return RecordReport(**d)
