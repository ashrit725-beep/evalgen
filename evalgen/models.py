"""Plain data records that flow through one evaluation cycle.

Event      one real earthquake: a row of the frozen snapshot (the ground truth)
Report     one digest line describing an earthquake (what some network said about it)
Notice     a CORRECTION or RETRACTION line in a digest
Trap       a deliberately injected pitfall whose correct handling is known in advance
Question   one composed question; its operation chain is its machine-readable meaning
Rendering  one textual presentation of an instance (original or a metamorphic variant)
Instance   everything needed to ask, answer and grade one test item
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Event IDs look like USGS ones: nc75012345, us7000tvs3, aka2026vupcsv (always lower case).
EVENT_ID_PATTERN = r"[a-z]{2,4}\d[0-9a-z]{3,14}"
EVENT_ID_RE = re.compile(EVENT_ID_PATTERN)


@dataclass(frozen=True)
class Event:
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
class Report:
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


@dataclass
class Trap:
    trap_id: str
    kind: str  # duplicate | retraction | conflicting_magnitude
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
    answer_type: str  # count | event_id | magnitude | depth | distance | region | utc_time
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instance":
        return cls(
            instance_id=d["instance_id"],
            seed=d["seed"],
            compiled_at=d["compiled_at"],
            window_hours=d["window_hours"],
            event_ids=list(d["event_ids"]),
            reports=[Report(**r) for r in d["reports"]],
            notices=[Notice(**n) for n in d["notices"]],
            chatter=list(d["chatter"]),
            traps=[Trap(**t) for t in d["traps"]],
            questions=[Question(**q) for q in d["questions"]],
            answer_key=dict(d["answer_key"]),
            renderings=[Rendering(**r) for r in d["renderings"]],
            forecast=d.get("forecast"),
        )

    def question(self, qid: str) -> Question:
        return next(q for q in self.questions if q.qid == qid)
