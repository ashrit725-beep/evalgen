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
        a generic domain (whose lines carry their values by field key). An instance file of a run received
        from someone else may hold any JSON, and run, score and report index it, so JSON that is not of the
        shape to_dict writes raises ValueError, saying what is wrong (instance_problem)."""
        problem = instance_problem(d)
        if problem is not None:
            raise ValueError(problem)
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
        """The question `qid` of this instance; KeyError when it has none (not a bare StopIteration, which a
        generator around the call would turn into a RuntimeError)."""
        for q in self.questions:
            if q.qid == qid:
                return q
        raise KeyError(f"{self.instance_id} has no question {qid!r}")

    def unregistered(self) -> list[str]:
        """What this instance names that no registry of this process knows: its domain, the genres of its
        renderings and the answer types of its questions (evalgen.domains, evalgen.genres, evalgen.scoring),
        each as "domain 'x'", "genre 'y'" or "answer type 'z'". run, score and report look each of them up,
        once the run's plugins are loaded; an instance file received from someone else may name anything."""
        from .domains import get_domain  # imported here: evalgen.domains imports this module
        from .genres import get_genre
        from .scoring import get_answer_type

        named = (("domain", [] if self.domain is None else [self.domain], get_domain),
                 ("genre", [r.genre for r in self.renderings if r.genre is not None], get_genre),
                 ("answer type", [q.answer_type for q in self.questions], get_answer_type))
        unknown = []
        for kind, names, lookup in named:
            for name in dict.fromkeys(names):
                try:
                    lookup(name)
                except KeyError:
                    unknown.append(f"{kind} {name!r}")
        return unknown


def check_registered(instances: list[Instance], run_dir: Any) -> None:
    """Refuse (util.RunDirError) the instances of the run in `run_dir` when one names a domain, a genre or
    an answer type that this process does not know (Instance.unregistered), once the run's plugins are
    loaded: run, score and report look them up, and would otherwise stop on a KeyError midway."""
    from pathlib import Path

    from .util import RunDirError

    for inst in instances:
        unknown = inst.unregistered()
        if unknown:
            raise RunDirError(f"{Path(run_dir) / 'instances' / (inst.instance_id + '.json')} names {', '.join(unknown)}, "
                              "which neither evalgen nor the run's plugins define: it is not an instance that evalgen "
                              "wrote for this run")


_SPLITS = ("public", "held_out")  # an instance's split (evalgen.mix.SPLITS)


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _texts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _chain_problem(chain: Any) -> str | None:
    """What is wrong with a question's chain as the solvers, the scorer and the report read it: a list of
    steps, each an object whose op is one evalgen knows, with a terminal step among them."""
    from .ops import role  # imported here: ops imports the domains, which import this module

    if not isinstance(chain, list) or not all(isinstance(step, dict) for step in chain):
        return "is not a list of steps"
    ops = [step.get("op") for step in chain]
    if not all(isinstance(op, str) and role(op) is not None for op in ops):
        return f"names an op that evalgen does not know ({', '.join(map(repr, ops))})"
    if not any(role(op) == "terminal" for op in ops):
        return "has no terminal step"
    return None


def instance_problem(d: Any) -> str | None:
    """What keeps `d` (JSON as read from an instance file) from being an instance as to_dict writes it, or
    None: the keys that run, score, report and resolve index, in the types evalgen writes them, and the
    questions that traps and renderings name being the instance's own. Names that a registry must know
    (domain, genres, answer types) are checked once a run's plugins are loaded (check_registered)."""
    if not isinstance(d, dict):
        return "it is not a JSON object"
    if not isinstance(d.get("instance_id"), str):
        return "its instance_id is not text"
    for key in ("seed", "compiled_at", "window_hours"):
        if not _whole(d.get(key)):
            return f"its {key} is not a whole number"
    for key in ("event_ids", "chatter"):
        if not _texts(d.get(key)):
            return f"its {key} are not a list of texts"
    if not isinstance(d.get("reports"), list) or not all(isinstance(r, dict) and isinstance(r.get("ref"), str)
                                                         for r in d["reports"]):
        return "its reports are not objects, each with the ID it prints (ref)"
    if not isinstance(d.get("notices"), list) or not all(isinstance(n, dict) for n in d["notices"]):
        return "its notices are not objects"
    questions = d.get("questions")
    if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
        return "its questions are not objects"
    qids = [q.get("qid") for q in questions]
    if not _texts(qids) or len(set(qids)) != len(qids):
        return "its questions are not named by distinct QIDs"
    for q in questions:
        for key in ("template_id", "pool", "answer_type"):
            if not isinstance(q.get(key), str) or not q[key]:
                return f"question {q['qid']} has no {key} as text"
        if not isinstance(q.get("unit"), (str, type(None))) or not _texts(q.get("evidence", [])):
            return f"question {q['qid']} has a unit or evidence that is not text"
        problem = _chain_problem(q.get("chain"))
        if problem is not None:
            return f"the chain of question {q['qid']} {problem}"
        features = q.get("features", {})
        if not isinstance(features, dict) or not _whole(features.get("trap_exposure", 0)) or not isinstance(
                features.get("filters_matter", True), bool):
            return f"the features of question {q['qid']} are not a trap exposure and whether its filters matter"
    if not isinstance(d.get("answer_key"), dict) or not all(qid in d["answer_key"] for qid in qids):
        return "it holds no answer key for each of its questions"
    traps = d.get("traps")
    if not isinstance(traps, list) or not all(isinstance(t, dict) for t in traps):
        return "its traps are not objects"
    for t in traps:
        if not all(isinstance(t.get(key), str) for key in ("trap_id", "kind", "form", "target", "description")):
            return "a trap has no ID, kind, form, target or description as text"
        if not isinstance(t.get("mishandles"), list):
            return f"trap {t['trap_id']} lists no careless readings"
        for key in ("affected", "fallen"):
            named = t.get(key, {})
            if not isinstance(named, dict) or not set(named) <= set(qids):
                return f"trap {t['trap_id']} names in its {key} what is not a question of the instance"
        if not all(isinstance(answers, list) for answers in t.get("fallen", {}).values()):
            return f"trap {t['trap_id']} gives its fallen answers otherwise than as lists"
    renderings = d.get("renderings")
    if not isinstance(renderings, list) or not renderings or not all(isinstance(r, dict) for r in renderings):
        return "its renderings are not objects"
    for r in renderings:
        if not (isinstance(r.get("variant"), str) and _whole(r.get("seed")) and isinstance(r.get("style"), dict)
                and all(isinstance(r.get(key), str) for key in ("digest", "prompt", "prompt_sha256"))
                and isinstance(r.get("genre"), (str, type(None)))):
            return "a rendering has no variant, seed, style, digest or prompt of the types evalgen writes"
        texts = r.get("question_text")
        if not isinstance(texts, dict) or not all(isinstance(texts.get(qid), str) for qid in qids):
            return f"rendering {r['variant']} words not every question of the instance as text"
    forecast = d.get("forecast")
    if forecast is not None and not (isinstance(forecast, dict) and isinstance(forecast.get("qid"), str)
                                     and isinstance(forecast.get("rule"), dict)):
        return "its forecast has no QID and rule"
    if not isinstance(d.get("domain"), (str, type(None))) or d.get("split") not in (None, *_SPLITS):
        return "its domain or split is not one that evalgen writes"
    if not isinstance(d.get("surface"), (dict, type(None))):
        return "its surface words are not an object"
    after = d.get("chatter_after")
    if after is not None and not (isinstance(after, list) and all(a is None or _whole(a) for a in after)):
        return "its chatter times are not whole numbers"
    return None


def _without_unset(d: dict[str, Any], optional: tuple[str, ...]) -> dict[str, Any]:
    """`d` without the optional keys whose value is None, in the same order."""
    return {k: v for k, v in d.items() if v is not None or k not in optional}


def _report(d: dict[str, Any]) -> Any:
    """A digest line from its JSON: a generic RecordReport (it has values), else an earthquake Report."""
    if "values" not in d:
        return Report(**d)
    from .domains.base import RecordReport  # imported here: evalgen.domains imports this module

    return RecordReport(**d)
