"""Plant traps whose correct handling is known in advance.

Besides one REVIEWED report per real earthquake, every digest contains:
  duplicate              another network re-reports a quake (same event ID, or its own reference
                         plus "same event as <id>") with slightly different, PRELIMINARY values
  retraction             a phantom report that a later notice RETRACTS (quarry blast, ...)
  conflicting_magnitude  a magnitude that disagrees with the truth, settled either by a CORRECTION
                         notice or by a REVIEWED report next to a PRELIMINARY one

Each trap records its careless readings (``mishandles``). Re-running every question under them
gives ``affected`` (questions whose answer changes when only this trap is misread, with that wrong
answer) and ``fallen`` (every wrong answer a reader reaches by misreading this trap, alone or
together with other traps). Placement is rejection-sampled so that most traps are load-bearing
and their wrong answers sit clearly outside the scoring tolerance; a trap that changes nothing is
still planted, as plain noise.

On the legacy path the 0.2.0 builders plant every line, as 0.2.0 did (its bytes are pinned). Their
planted lines copy the style of the real ones (place notes that agree with their coordinates, the
network's magnitude precision, IDs that fit the network's own ID sequence), but not where real lines
lie: a re-report sits 3 to 40 s and 1 to 8 km from its quake, filed by one of the networks that
re-report others' events (us, at, pt, ew), of which only us files events of its own, a second report at
its quake's very time and place, and a conflicting magnitude within 0.2 of an asked threshold. So in a
legacy digest these features, which the reading rules never mention, pick out planted lines without
their notes, statuses or notices: a reader who merges reports within 40 s, keeping the commoner
network's, closes about a fifth of the gap between the naive reader and the oracle.

Off the legacy path, a planted report stands in for a *slot*: a record drawn with the digest, exactly
as its records are, but not kept (pipeline.sample_with_slots; see the slot builders below). A phantom is
a slot filed as it is, and a re-report or a second report is a slot filed under a real record's ID, or
under its own with a note naming one, so its time, place, source, values, their precision and its ID are
a real record's, and nothing that its ID says about its record contradicts it (a close approach is
re-reported under its own ID only where its object makes no other approach in the digest, which would
share its H); a correction shows a value of a slot, or, for a value tied to the record's near twins (a
station's temperature), one that agrees with them as a real one does (_twin_values). Only the signals the
rules name set a planted line apart: a shared ID or a note naming one, the status, and the notices. Each
trap also draws the scopes of a real record of the digest, how many of the counting questions' scopes and
how many of the others' it lies in, and for each question how far a real record that decides another
question leads the next one (``TrapConfig.match_scope``), and its planted report lies in as many and leads as
far, so that placing traps where they change answers neither puts planted reports in counting questions'
scopes more often than real ones nor has them pass the reports they take a question from by less than real
ones lead (nor by just about what the record they pass leads the next by, see _Leads). A digest's traps
are planned before its records, each kind on its own, with as many records less as they add reports
(draw_plan): how many reports a digest plants varies widely, none about as often as any number in a
question's scope, so that no share or number taken off a count does better than the count, and none,
one or two retractions, none the likeliest, so that the count left by the status rule, which counts every
phantom in a question's scope, is the likeliest right count. A notice is issued later than any genre sends
the report it names (NOTICE_DELAY).

Off the legacy path, placement is also guarded (``TrapConfig.guarded``): a careless reading counts
only where the misread digest passes the same guards as the key (questions.guards_ok): no record,
planted ones included, within a threshold's margin, no tie or near tie in a ranked or nearest answer,
no record at a time window's edge. Only then is the wrong answer the one that a reader who makes that
mistake gets from the text at its printed precision, rather than one decided by the stored values'
last digit or by a tie-break. A placement whose every reading passes is preferred to any other, so a
trap is planted unguarded only when no guarded placement turned up, and its affected questions are
then only those its guarded readings change. Placing traps where their readings change answers and
hold is itself a selection, which matching a real record's scopes and leads undoes in part: a planted
report is still a little likelier than a real one to hold its group's highest value or to lie clear of a
threshold (AUCs up to about 0.6), though no reader that drops reports by such a feature does better than
the naive one.

Placement, ranking and scoring are the same for every domain. What a trap looks like depends on the
domain's records, so the builders are looked up per domain (``builders_for``), and they take the
domain's trap vocabulary (retraction reasons), statuses, reading rules and record hooks
(Domain.rereport_values, planted_value) from the domain itself. Earthquakes keep the 0.2.0 builders on
the legacy path. Every domain off it, and every plugin domain, gets the same three traps, with the same
careless readings:
  duplicate          a slot repeats a record (its ID, or the slot's own with a note naming the ID), with
                     the slot's values and the domain's second status
  retraction         a phantom report, a slot, that a later notice retracts
  conflicting_value  a wrong value of a field (a correctable one when there is one), settled by a
                     correction notice or by the settling report next to an unsettled one, a slot's
(earthquakes keep their kind's 0.2.0 name, conflicting_magnitude).
"""
from __future__ import annotations

import functools
import itertools
import math
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from .domains.base import Domain, FieldSpec, Record, RecordReport, folded, stored_places
from .domains.earthquakes import EARTHQUAKES, Earthquakes
from .geo import COMPASS16, compass_point, destination, haversine_km, initial_bearing
from .models import Event, Notice, Question, Report, Trap
from .ops import EPS, DigestData, Step, canonical_threshold, distance_km, execute, passes, role, rows_after_filters
from .questions import Context, guards_ok
from .rules import Mishandle
from .scoring import get_answer_type, normalize, same_answer
from .sources import COORD_DECIMALS
from .units import Unit
from .util import plain_number

_PLACE = re.compile(r"^(\d+(?:\.\d+)?) km ([NSEW]{1,3}) of (.+)$")


# ------------------------------------------------------------------- off the legacy path: the digest's plan
# Off the legacy path a digest's traps are planned before its records are drawn (draw_plan; pipeline.build_instance,
# seed path "instance", i, "plan"), and the digest keeps one record less for each report its traps will add
# (pipeline.sample_with_slots): it prints as many reports however many of them are planted, and neither its size
# nor any count of what it prints tells how many are.
#
# How many reports are planted varies widely from digest to digest. With two or three re-reports, a conflict and
# a retraction or more in every digest (and one more re-report or conflict wherever one changed no answer), the
# planted reports were a steady share of a digest, about a sixth of its reports, and of the reports in any
# question's scope, since they lie in questions' scopes as real records do (TrapConfig.match_scope). A reader who
# took that share off every count (round(0.84 x the naive count)), reading no status, ID, note or notice, was
# right on half the counting questions where the naive reader was on a third, and closed a tenth of the gap
# between the naive reader and the oracle, as much on held-out domains and genres. Now every kind is drawn on
# its own, so that how many of one kind a reader sees tells nothing of how many of another there are:
#   retractions     none, one or two (RETRACTION_WEIGHTS), none the likeliest: the status rule counts every
#                   phantom in a question's scope, so its count is then the likeliest right count;
#   cross-refs      re-reports under IDs of their own, with a note naming their record: none in 45% of the
#                   digests, else one to three alike (CROSS_REF_WEIGHTS);
#   same-ID lines   reports under their records' own IDs: none in 30% of the digests, else one to eight alike
#                   (SAME_ID_WEIGHTS), each a second report (a conflict) one time in five and else a re-report;
#   corrections     none to three (CORRECTION_WEIGHTS), which add no report.
# A digest then holds no planted report about one time in sixteen, and up to thirteen, and the planted reports in
# a question's scope number none about as often as any other number, whatever its scope (and so do the phantoms
# and cross-refs that a reader who merges reports by ID alone still counts): neither a share nor a number taken
# off a count does much better than the count (exactly: by at most 5 points of the counting questions whose
# scope holds a given share of the records, and by none below a third; by 20 points before). The weights are this family's (none in a share of
# digests, else a number from one up drawn alike) that leave about the least such gain with about as many planted
# reports as before (5.0 a digest). A digest prints every report its plan adds: the second reports are planted
# first, each about a record free of traps, then the re-reports, which re-report a record that re-reports alone
# carry where no record is free (_virtual_pair: in the heaviest plans of the smallest digests), then the
# retractions, and the corrections last, which need a record free of traps too but add no report. Corrections,
# which change no count, make up the traps that change an answer (two a digest; the plan before planted 0.6).
RETRACTION_WEIGHTS = (0.5, 0.25, 0.25)
CROSS_REF_WEIGHTS = (0.45, 0.55 / 3, 0.55 / 3, 0.55 / 3)
SAME_ID_WEIGHTS = (0.3,) + (0.7 / 8,) * 8
SECOND_REPORT_SHARE = 0.2
CORRECTION_WEIGHTS = (0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class TrapPlan:
    """How many traps of each kind and form a digest off the legacy path holds (draw_plan)."""

    retractions: int = 0
    cross_refs: int = 0  # re-reports under IDs of their own, with a note naming their record
    same_ids: int = 0  # re-reports under their records' own IDs
    second_reports: int = 0  # conflicts settled by the settling report beside a second report under its ID
    corrections: int = 0  # conflicts settled by a correction notice

    @property
    def lines(self) -> int:
        """The reports its traps add to the digest (a correction adds none)."""
        return self.retractions + self.cross_refs + self.same_ids + self.second_reports


def draw_plan(rng: random.Random) -> TrapPlan:
    """A digest's traps off the legacy path, each kind drawn on its own (see the plan above)."""
    def drawn(weights: Sequence[float]) -> int:
        return rng.choices(range(len(weights)), weights=weights, k=1)[0]

    retractions, cross_refs, same_id = drawn(RETRACTION_WEIGHTS), drawn(CROSS_REF_WEIGHTS), drawn(SAME_ID_WEIGHTS)
    second_reports = sum(rng.random() < SECOND_REPORT_SHARE for _ in range(same_id))
    return TrapPlan(retractions=retractions, cross_refs=cross_refs, same_ids=same_id - second_reports,
                    second_reports=second_reports, corrections=drawn(CORRECTION_WEIGHTS))


@dataclass(frozen=True)
class TrapConfig:
    duplicates: tuple[int, int] = (2, 3)  # inclusive range per instance (the 0.2.0 plan)
    retractions: int = 1
    conflicts: int = 1
    candidates: int = 25  # placements tried per trap, looking for a load-bearing, clean one (SLOT_CANDIDATES off it)
    id_style: str = "random"  # "random" (synthetic IDs) or "sequence" (fit a real feed's ID sequence)
    # Off the legacy path: a careless reading counts only where the misread digest passes the key's
    # guards, and a placement whose every reading does is preferred (see the module docstring). False is
    # the 0.2.0 placement, which the legacy path keeps.
    guarded: bool = False
    # Off the legacy path: a load-bearing planted report must change an answer, so it lies in more of the
    # questions' scopes (it passes all of a question's filters), and in more of the counting questions' ones,
    # than a real record does, and one that changes the record a question names, or its value, by taking its
    # place leads the next report by less than a real one does. So each trap draws the scopes of a real record of
    # the digest, how many of the counting questions' and how many of the others' it lies in, and for each
    # question how far a real record that decides another question leads the next one (_Leads), and prefers a
    # placement whose readings all hold, whose planted report lies in exactly as many counting questions' scopes,
    # and in as many of the others' give or take one, and whose planted report, where it decides a question,
    # leads as far give or take a factor of two, to any placement of a higher rank (_fit). False on the legacy
    # path, which draws nothing for it.
    match_scope: bool = False
    # Off the legacy path: the digest's traps, planned before its records (draw_plan), in place of the 0.2.0
    # plan above (duplicates, retractions, conflicts, and a form drawn for each).
    plan: Optional[TrapPlan] = None


# Off the legacy path a trap tries this many placements: matching a real record's scopes and leads (TrapConfig.
# match_scope) leaves fewer of them that change an answer, and twice the 0.2.0 number finds one about as often.
SLOT_CANDIDATES = 50


@dataclass
class _Candidate:
    trap: Trap
    add_reports: list[Any]  # Reports, or the RecordReports of a generic domain
    add_notices: list[Notice]
    replace_at: tuple[int, Any] | None = None
    slot: str | None = None  # the ID of the slot whose place its planted report takes


@dataclass
class _Site:
    """What a trap builder reads while one digest is being planted, and the bookkeeping it updates."""

    domain: Domain
    events: list[Any]  # the digest's records: Events, or the Records of a generic domain
    questions: list[Question]
    rng: random.Random
    compiled_at: int
    window_start: int
    id_style: str
    reports: list[Any]  # the digest so far: a conflict may replace a settling report in it
    taken: set[str]  # every ID in use, including the IDs drawn for candidates that were not chosen
    used: set[str]  # records that already carry a duplicate or a conflict
    conflicted: set[str] = field(default_factory=set)  # records that carry a conflict (_virtual_pair)
    # Off the legacy path: the records drawn with the digest but not kept (pipeline.sample_with_slots), as
    # generic Records, and the IDs of those whose place a chosen trap has taken; the digest's records as
    # generic Records; and whether a record with a place stands at a fixed site (_fixed).
    slots: list[Record] = field(default_factory=list)
    filled: set[str] = field(default_factory=set)
    tried: set[str] = field(default_factory=set)  # the slots the placements of the trap being planted tried
    records: list[Record] = field(default_factory=list)
    fixed: bool = True
    correction: bool = False  # the form the conflict being planted drew (_slot_conflict)
    form: str | None = None  # the form the re-report being planted was planned in (None: drawn, _slot_duplicate)
    pairs: list[tuple[int, dict[str, float]]] | None = None  # the real twin pairs (_twin_pairs), once read


# Optional, not "X | None": this alias is evaluated at import, and Python 3.9 has no such unions.
Builder = Callable[[str, _Site], Optional[_Candidate]]


@dataclass(frozen=True)
class Builders:
    """How one domain's digests are planted: the report that settles a record, and one builder per
    kind of trap, which proposes a placement (or None when this attempt finds no room for one)."""

    primary: Callable[[Any], Any]
    duplicate: Builder
    retraction: Builder
    conflict: Builder


def primary_report(e: Event) -> Report:
    return Report(ref=e.id, network=e.network, status="reviewed", time=e.time, lat=e.lat, lon=e.lon,
                  depth_km=e.depth_km, mag=e.mag, mag_type=e.mag_type, region=e.region, place=e.place)


def _apply(reports: list[Report], notices: list[Notice], c: _Candidate) -> tuple[list[Report], list[Notice]]:
    reports = list(reports)
    if c.replace_at is not None:
        index, report = c.replace_at
        reports[index] = report
    return reports + c.add_reports, list(notices) + c.add_notices


@dataclass(frozen=True)
class _Reading:
    """One question answered under one careless reading of a trap."""

    question: Question
    answer: Any
    guarded: bool  # the misread digest passes the question's guards (always, in the 0.2.0 placement)


def _single_readings(trap: Trap, questions: list[Question], data: DigestData, domain: Domain,
                     time_margin: int | None = None) -> list[_Reading]:
    """Every question under each careless reading of this one trap, with the domain's reading rules for
    everything the reader does not misread, readings in the trap's order and questions in theirs.

    With a `time_margin` (guarded placement), a reading is guarded when the misread digest passes the
    question's guards (questions.guards_ok, time windows kept that far from every record), as the
    snapshot passes them for the key: no record within a threshold's margin, no tie or near tie in a
    ranked or nearest answer. Its answer is then the one a reader who makes that mistake gets from the
    text, at the precision the digest prints. Without one (the 0.2.0 placement), every reading counts."""
    units = domain.units
    readings = []
    for reading in trap.mishandles:
        mishandle = Mishandle.from_dict(reading)
        records = None if time_margin is None else domain.resolve(data.reports, data.notices, mishandle)
        for q in questions:
            answer = execute(q.chain, data, mishandle=mishandle, units=units, resolver=domain.resolve)
            guarded = records is None or guards_ok(q.chain, records, domain=domain, time_margin=time_margin)
            readings.append(_Reading(q, answer, guarded))
    return readings


def _effects(readings: Sequence[_Reading], key: Mapping[str, Any]) -> dict:
    """qid -> wrong answer, for questions a reader gets wrong when only this trap is misread: the first
    guarded reading of each question that differs from the key."""
    out = {}
    for r in readings:
        q = r.question
        if r.guarded and q.qid not in out and not same_answer(q.answer_type, r.answer, key[q.qid], unit=q.unit):
            out[q.qid] = r.answer
    return out


def _clear_of_tolerance_edge(readings: Sequence[_Reading], key: Mapping[str, Any]) -> bool:
    """A misread numeric answer should be exactly right or clearly wrong (beyond twice the scoring
    tolerance), so that rounding in the digest cannot move a careless reader across the edge."""
    for r in readings:
        q = r.question
        answer_type = get_answer_type(q.answer_type)
        if answer_type.kind != "number":
            continue
        a, k = normalize(q.answer_type, r.answer), normalize(q.answer_type, key[q.qid])
        if a is None or k is None or a == k:
            continue
        abs_tol, rel_tol = answer_type.tolerance_in(q.unit)
        if abs(a - k) <= 2 * max(abs_tol, rel_tol * abs(k)):
            return False
    return True


def _rank(readings: Sequence[_Reading], key: Mapping[str, Any], guarded: bool) -> int:
    """How good a placement is. The 0.2.0 ranks: 2 when it is load-bearing and every misread answer is
    clear of the scoring tolerance's edge, 1 when it is load-bearing, 0 when it changes nothing. A
    guarded placement whose readings all pass the guards takes its 0.2.0 rank plus one (so at most
    GUARDED_TOP), and any other guarded placement 0: a trap that changes nothing beats one whose
    careless readings no reader of the text can be held to."""
    rank = (2 if _clear_of_tolerance_edge(readings, key) else 1) if _effects(readings, key) else 0
    if not guarded:
        return rank
    return rank + 1 if all(r.guarded for r in readings) else 0


TOP, GUARDED_TOP = 2, 3  # the best rank of a placement, 0.2.0 and guarded: the first one found is taken


class _Scopes:
    """Which of the questions' scopes a row lies in (the questions all of whose filters it passes): how many
    of the counting questions' (whose answer is a count) and how many of the others'."""

    def __init__(self, questions: Sequence[Question], units: Mapping[str, Sequence[Unit]]) -> None:
        self.filters = [[raw for raw in q.chain if role(raw["op"]) == "filter"] for q in questions]
        self.counting = [any(Step(raw).kind == "count" for raw in q.chain) for q in questions]
        self.units = units

    def count(self, row: Any) -> tuple[int, int]:
        inside = [all(passes(step, row, self.units) for step in steps) for steps in self.filters]
        return (sum(1 for i, c in zip(inside, self.counting) if i and c),
                sum(1 for i, c in zip(inside, self.counting) if i and not c))


def _planted(candidate: _Candidate) -> Any:
    """The planted report of a placement: the one it adds, or the settling report it shows a wrong value on."""
    return candidate.add_reports[0] if candidate.add_reports else candidate.replace_at[1]


def _composition(scope: tuple[int, int], target: tuple[int, int]) -> tuple[bool, bool]:
    """How the scopes of a placement's planted report, (counting questions', others'), fit a trap's target:
    in exactly as many counting questions' scopes, and in as many of the others' give or take one."""
    return scope[0] == target[0], abs(scope[1] - target[1]) <= 1


def _fit(rank: int, scope: tuple[int, int], target: tuple[int, int], top: int, leads_like: bool = True) -> tuple:
    """How well a placement of `rank`, whose planted report lies in `scope` of the questions' scopes, fits a
    trap that draws `target` (TrapConfig.match_scope): one whose readings all hold first (rank 1 and up),
    then by whether every planted report leads the next report as a real one does wherever it decides a
    question (`leads_like`, _Leads), then by its scopes (_composition), then by rank, then by how far its
    scopes are from the target.

    Its scopes come before its rank. A planted report that changes a count lies in that count's scope, and
    one that changes another answer must hold its scope's highest or lowest value, so placements ranked
    first lay planted reports in counting questions' scopes half as often again as real records lie there
    (0.46 to 0.53 such scopes a report, against 0.30 to 0.32), and removing one changed the naive answer
    of more questions than removing a real record does (an AUC of 0.60 to 0.69). Its leads come first of all:
    a planted report that takes the record a question names, or its value, from a real one must pass it, but
    need pass it only by the question's gap, so placements ranked first led the report they passed by half as
    much as a real record that decides a question leads the next one (an AUC of 0.27 in weather, where a
    station's next observation is the likeliest to pass it, and of 0.18 to 0.23 for the report nearest to a
    question's place in earthquakes). A placement whose planted report decides no question leads as a real
    one alike."""
    distance = -abs(scope[0] - target[0]) - abs(scope[1] - target[1])
    return (rank >= 1, leads_like, *_composition(scope, target), rank, distance)


def _best_fit(target: tuple[int, int], top: int) -> tuple:
    """The fit at which the search for a placement stops (_fit)."""
    return _fit(top, target, target, top)


def _matched_placement(candidates: Sequence[_Candidate | None], target: tuple[int, int], scopes: _Scopes,
                       rank_of: Callable[[_Candidate], int], top: int,
                       leads_like: Callable[[_Candidate], bool] = lambda candidate: True,
                       ) -> tuple[tuple, int, _Candidate] | None:
    """The placement of `candidates` (None: a placement that found no room) that fits `target` best (_fit),
    with its fit and rank. Placements are ranked in the order of their leads' and scopes' fit, the first
    proposed first, and the search stops once no placement left can beat the best one: ranking a placement
    reads every question under every careless reading, and its leads and scopes are read at a glance."""
    ordered = sorted(((leads_like(c), *_composition(scopes.count(_planted(c)), target), -i, c)
                      for i, c in enumerate(candidates) if c is not None), key=lambda item: item[:4], reverse=True)
    best: tuple[tuple, int, _Candidate] | None = None
    for *glance, _, candidate in ordered:
        if best is not None and best[0][0] and tuple(glance) < best[0][1:1 + len(glance)]:
            break  # neither it nor any after it can beat the best placement, whatever their rank
        rank = rank_of(candidate)
        fit = _fit(rank, scopes.count(_planted(candidate)), target, top, glance[0])
        if best is None or fit > best[0]:
            best = (fit, rank, candidate)
        if fit == _best_fit(target, top):
            break
    return best


LEAD_KINDS = ("argmax", "max", "min", "time_of_max", "rank")
LEAD_FACTOR = 2.0  # a planted report that decides a question leads the next one this close to a real one's


class _Leads:
    """How far the report that decides a question leads the next one (TrapConfig.match_scope), as a reader who
    takes every report for a record of its own sees it: for a question that names a record or a value of one
    (its highest or lowest value, the k-th), the report that holds it, and how far it leads the next report of
    the question's scope, in shares of the field's spread over the digest's records. A report its scope holds
    alone leads infinitely far, as a real one alone in its scope does.

    The real records' own leads, the key's, are what a real report that decides a question leads by. For each
    question, each trap draws the lead by which a real record decides *another* question, one that a
    different record decides, of the same kind where the digest asks one (else of any of these kinds), and
    prefers a placement whose planted report, wherever it decides a question, leads within a factor of
    LEAD_FACTOR of the lead drawn for that question, and in which every report planted before still does. A
    question for which no other question can lend a lead draws none, and its deciders are not held to one.

    The lead lent is never the question's own, nor that of a question its own decider decides. A planted report
    that takes the record a question names passes the key's decider, so the next report's lead over the third
    is the key's own lead: matched to that very lead (as the lead first drawn from the kind's own questions was,
    in a digest that asks one question of a kind), a planted decider led the next by about what the next led
    the third by, within a factor of two for 58-72% of planted deciders against 31-36% of real ones (an AUC of
    0.28 to 0.36 for |log2| of the ratio of the two leads), though no reader gained by it."""

    def __init__(self, questions: Sequence[Question], records: Sequence[Any], units: Mapping[str, Sequence[Unit]],
                 domain: Domain) -> None:
        self.units = units
        self.terms = []  # each question of a LEAD_KINDS kind: its number, its filters and its terminal
        for number, q in enumerate(questions):
            term = Step(next(raw for raw in q.chain if role(raw["op"]) == "terminal"), units)
            if term.kind in LEAD_KINDS:
                self.terms.append((number, [raw for raw in q.chain if role(raw["op"]) == "filter"], term))
        self.spread = {spec.key: _spread(records, spec) for spec in domain.fields} if records else {}
        self.real = self.decided(records) if records else []
        self.planted: list[tuple[int, dict[int, float]]] = []  # each planted report's index, and its trap's leads

    def decided(self, rows: Sequence[Any]) -> list[tuple[int, str, float, int]]:
        """(the question's number, its kind, the lead, the index of the row that decides it) for each question
        that one of `rows` (generic records or reports) decides."""
        out = []
        for number, filters, term in self.terms:
            inside = [i for i, r in enumerate(rows) if all(passes(step, r, self.units) for step in filters)]
            if not inside:
                continue
            key, low = term.field, term.kind == "min"
            ordered = sorted(inside, key=lambda i: (rows[i].values[key] if low else -rows[i].values[key], i))
            k = term["k"] - 1 if term.kind == "rank" else 0
            if len(ordered) <= k:
                continue
            gap = (abs(rows[ordered[k]].values[key] - rows[ordered[k + 1]].values[key]) / self.spread[key]
                   if len(ordered) > k + 1 else math.inf)
            out.append((number, term.kind, gap, ordered[k]))
        return out

    def draw(self, rng: random.Random) -> dict[int, float]:
        """For one trap, a lead for each question the key's records decide: the lead by which a real record
        decides another question, decided by another record, of the same kind where there is one."""
        targets = {}
        for number, kind, _, at in self.real:
            others = [(other, lead) for q, other, lead, by in self.real if q != number and by != at]
            pool = [lead for other, lead in others if other == kind] or [lead for _, lead in others]
            if pool:
                targets[number] = rng.choice(pool)
        return targets

    def like(self, rows: Sequence[Any], index: int, target: Mapping[int, float]) -> bool:
        """Whether, in the digest `rows`, the row numbered `index` (planted with the leads `target`) and every
        report planted before lead, wherever they decide a question, within a factor of LEAD_FACTOR of their
        trap's lead for that question (both infinite, or both finite)."""
        wanted = dict(self.planted)
        wanted[index] = dict(target)
        for number, _, lead, at in self.decided(rows):
            want = wanted.get(at, {}).get(number)
            if want is None:
                continue
            if math.isinf(lead) or math.isinf(want):
                if math.isinf(lead) != math.isinf(want):
                    return False
                continue
            lead, want = max(lead, 1e-9), max(want, 1e-9)
            if not 1 / LEAD_FACTOR <= lead / want <= LEAD_FACTOR:
                return False
        return True


def score_traps(traps: list[Trap], questions: list[Question], key: dict, reports: list[Report],
                notices: list[Notice], *, domain: Domain = EARTHQUAKES, time_margin: int | None = None) -> None:
    """Fill trap.affected (misread alone) and trap.fallen (misread alone or with any other traps).

    With a `time_margin` (guarded placement), `affected` keeps only the questions whose misread
    digest passes their guards (_single_readings): a trap's exposure, its catch rate and the strata
    of every gap count only careless readings that a reader of the text is held to. `fallen` keeps
    every misread answer that is not exactly the key, as computed from the stored values, including
    ones the scorer would still accept, so that a careless reader whose rounded answer lands just past
    the edge is attributed to the trap. Answers are deduplicated by exact value.

    Off the legacy path (with a `time_margin`), the joint misreadings of each question are those of the
    traps whose careless readings change the records it reads (the rows that pass its filters): a trap
    acts on records of its own, so one that changes none of them changes the question's answer in no
    combination, and every wrong answer of the others' combinations is its too, as a reader who also
    misreads it gets that answer. That is what reading every combination of every trap gives, at a cost
    that does not double with every trap a digest holds (_fallen_by_question)."""
    data = DigestData(reports, notices)
    units = domain.units
    for trap in traps:
        trap.affected = _effects(_single_readings(trap, questions, data, domain, time_margin), key)
        trap.fallen = {}
    if time_margin is not None:
        _fallen_by_question(traps, questions, key, data, domain)
        return
    choices = [[None] + list(range(len(t.mishandles))) for t in traps]  # None = read correctly
    for combo in itertools.product(*choices):
        misread = [i for i, c in enumerate(combo) if c is not None]
        if not misread:
            continue
        mishandle = Mishandle.merged(traps[i].mishandles[combo[i]] for i in misread)
        for q in questions:
            answer = execute(q.chain, data, mishandle=mishandle, units=units, resolver=domain.resolve)
            normal = normalize(q.answer_type, answer)
            if normal == normalize(q.answer_type, key[q.qid]):
                continue
            for i in misread:
                bucket = traps[i].fallen.setdefault(q.qid, [])
                if all(normalize(q.answer_type, seen) != normal for seen in bucket):
                    bucket.append(answer)


def _fallen_by_question(traps: list[Trap], questions: list[Question], key: Mapping[str, Any], data: DigestData,
                        domain: Domain) -> None:
    """Trap.fallen off the legacy path (see score_traps): for each question, every combination of the
    careless readings of the traps that change the rows it reads, each wrong answer given to every trap
    misread in it and to every trap that changes none of those rows."""
    units, resolver = domain.units, domain.resolve
    for q in questions:
        def rows(mishandle: Mishandle | None) -> list[Any]:
            return rows_after_filters(q.chain, data, mishandle=mishandle, units=units, resolver=resolver)

        read = rows(None)
        relevant = [i for i, t in enumerate(traps)
                    if any(rows(Mishandle.from_dict(m)) != read for m in t.mishandles)]
        bystanders = [i for i in range(len(traps)) if i not in relevant]
        choices = [[None] + list(range(len(traps[i].mishandles))) for i in relevant]  # None = read correctly
        for combo in itertools.product(*choices):
            misread = [(i, c) for i, c in zip(relevant, combo) if c is not None]
            if not misread:
                continue
            mishandle = Mishandle.merged(traps[i].mishandles[c] for i, c in misread)
            answer = execute(q.chain, data, mishandle=mishandle, units=units, resolver=resolver)
            normal = normalize(q.answer_type, answer)
            if normal == normalize(q.answer_type, key[q.qid]):
                continue
            for i in [i for i, _ in misread] + bystanders:
                bucket = traps[i].fallen.setdefault(q.qid, [])
                if all(normalize(q.answer_type, seen) != normal for seen in bucket):
                    bucket.append(answer)


def threshold_values(questions: Sequence[Question], field: str) -> list[float]:
    """The thresholds the questions ask about on a field, in question order and then chain order: the
    value of every at_least or below step on it, named by its 0.2.0 alias or generically."""
    values = []
    for q in questions:
        for raw in q.chain:
            step = Step(raw)
            if step.kind in ("at_least", "below") and step.field == field:
                values.append(step["value"])
    return values


def _pick_unused(events: list[Event], used: set[str], rng: random.Random) -> Event | None:
    pool = [e for e in events if e.id not in used]
    return rng.choice(pool) if pool else None


def _sequence_ref(network: str, site: _Site, when: int) -> str:
    """A new ID that fits the network's own ID sequence at time `when`, as a real report would.

    Numeric IDs (ci41571656, nc75462032) get an unused value between the IDs of the events just before
    and after `when`, on the network's own step (e.g. NC IDs that all end in 2 or 7). Alphanumeric,
    time-coded IDs (aka2026vupcsv, us7000tvs3) keep the nearest-in-time neighbour's ID except its last
    two characters."""
    rng, taken, domain = site.rng, site.taken, site.domain
    same = sorted((e for e in site.events if e.network == network), key=lambda e: (e.time, e.id))
    if not same:
        return domain.new_id(network, rng, taken)
    letters = re.match(r"[a-z]+", same[0].id).group(0)
    bodies = [e.id[len(letters):] for e in same if e.id.startswith(letters)]
    if len(bodies) == len(same) and all(b.isdigit() for b in bodies) and len({len(b) for b in bodies}) == 1:
        width = len(bodies[0])
        values = [int(b) for b in bodies]
        ordered = sorted(set(values))
        step = functools.reduce(math.gcd, [b - a for a, b in zip(ordered, ordered[1:])], 0) or 1
        before = [v for e, v in zip(same, values) if e.time <= when]
        after = [v for e, v in zip(same, values) if e.time > when]
        lo = max(before) if before else min(values) - step
        hi = min(after) if after else max(values) + step
        lo, hi = min(lo, hi), max(lo, hi)
        if (hi - lo) // step <= 2000:  # take a free value on the network's step, nearest the time neighbours
            free = [v for v in range(lo - 8 * step, hi + 8 * step + 1, step)
                    if v > 0 and f"{letters}{v:0{width}d}" not in taken]
            if free:
                middle = (lo + hi) / 2
                free.sort(key=lambda v: (abs(v - middle), v))
                ref = f"{letters}{rng.choice(free[:3]):0{width}d}"
                taken.add(ref)
                return ref
        else:
            for _ in range(200):
                ref = f"{letters}{lo + step * rng.randint(1, (hi - lo) // step - 1):0{width}d}"
                if ref not in taken:
                    taken.add(ref)
                    return ref
    nearest = min(same, key=lambda e: (abs(e.time - when), e.id))
    for _ in range(200):
        tail = "".join(rng.choice("0123456789") if c.isdigit() else rng.choice("abcdefghijklmnopqrstuvwxyz")
                       for c in nearest.id[-2:])
        ref = nearest.id[:-2] + tail
        if ref not in taken and re.fullmatch(domain.id_pattern, ref):
            taken.add(ref)
            return ref
    return domain.mimic_id(nearest.id, rng, taken)


def _fresh_ref(source: str, site: _Site, when: int) -> str:
    """A new ID for a planted report, generated the way real IDs look in this mode."""
    if site.id_style == "sequence" and any(e.source == source for e in site.events):
        return _sequence_ref(source, site, when)
    return site.domain.new_id(source, site.rng, site.taken)


def _implied_town(event: Event) -> tuple[str, float, float] | None:
    """'8 km E of Lower Lake, CA' at (lat, lon) puts Lower Lake 8 km west of (lat, lon)."""
    m = _PLACE.match(event.place or "")
    if not m:
        return None
    lat, lon = destination(event.lat, event.lon, float(m.group(1)), (COMPASS16.index(m.group(2)) * 22.5 + 180) % 360)
    return m.group(3), lat, lon


def _place_for(anchor: Event, lat: float, lon: float, events: list[Event]) -> str | None:
    """A feed-style place note for a point near `anchor`, consistent with the real notes.

    The town's position is estimated from every real note that names it (averaging away their
    rounding), and the new note gives the distance and 16-point bearing from there to (lat, lon)."""
    anchor_town = _implied_town(anchor)
    if anchor_town is None:
        return anchor.place  # e.g. "Southern Alaska": no town to measure from
    estimates = [t for t in map(_implied_town, events) if t is not None and t[0] == anchor_town[0]]
    town_lat = sum(t[1] for t in estimates) / len(estimates)
    town_lon = sum(t[2] for t in estimates) / len(estimates)
    km = round(haversine_km(town_lat, town_lon, lat, lon))
    return f"{km} km {compass_point(initial_bearing(town_lat, town_lon, lat, lon))} of {anchor_town[0]}"


def _decimals(events: list[Event], network: str) -> int:
    """Magnitude precision a network uses in this digest (USGS networks often print 2 decimals)."""
    return 2 if any(round(e.mag, 1) != e.mag for e in events if e.network == network) else 1


def _rereporters(site: _Site, network: str) -> list[str]:
    """The domain's re-reporting networks other than `network`, in the vocabulary's order."""
    return [n for n in site.domain.trap_vocab.rereporting_sources if n != network]


def _duplicate(trap_id: str, site: _Site) -> _Candidate | None:
    rng, events = site.rng, site.events
    e = _pick_unused(events, site.used, rng)
    if e is None:
        return None
    net = rng.choice(_rereporters(site, e.network))
    form = rng.choice(("same_id", "cross_ref"))
    when = min(site.compiled_at - 1, max(site.window_start, e.time + rng.choice((-1, 1)) * rng.randint(3, 40)))
    ref = e.id if form == "same_id" else _fresh_ref(net, site, when)
    lat, lon = destination(e.lat, e.lon, rng.uniform(1.0, 8.0), rng.uniform(0.0, 360.0))
    dup = Report(
        ref=ref, network=net, status="preliminary", time=when, lat=round(lat, COORD_DECIMALS),
        lon=round(lon, COORD_DECIMALS),
        depth_km=round(max(0.0, e.depth_km + rng.uniform(-3.0, 3.0)), 1),
        mag=round(max(0.0, e.mag + rng.choice((-0.3, -0.2, -0.1, 0.1, 0.2, 0.3))), 2),
        mag_type=e.mag_type, region=e.region, place=_place_for(e, lat, lon, events) if e.place else None,
        same_as=None if form == "same_id" else e.id, role="duplicate", trap_id=trap_id,
    )
    how = "under the same event ID" if form == "same_id" else f"as {ref} with a 'same event as' note"
    description = (f"{net.upper()} re-reported {e.id} {how}: preliminary M{dup.mag:g}, depth {dup.depth_km:g} km "
                   f"(reviewed: M{e.mag:g}, {e.depth_km:g} km)")
    trap = Trap(trap_id, "duplicate", form, e.id, description,
                mishandles=[Mishandle(split=((ref, net, "preliminary"),)).to_dict()])
    return _Candidate(trap, [dup], [])


def _phantom(trap_id: str, site: _Site) -> _Candidate:
    rng, events, compiled_at = site.rng, site.events, site.compiled_at
    anchor = rng.choice(events)
    same_network = sorted(e.mag for e in events if e.network == anchor.network)
    base = same_network[int(rng.uniform(0.2, 0.95) * (len(same_network) - 1))]
    mag = round(min(max(0.0, base + rng.uniform(-0.2, 0.2)), same_network[-1] + 0.3), _decimals(events, anchor.network))
    lat, lon = destination(anchor.lat, anchor.lon, rng.uniform(3.0, 40.0), rng.uniform(0.0, 360.0))
    depth = round(max(min(e.depth_km for e in events), anchor.depth_km + rng.uniform(-5.0, 5.0)), 1)
    when = rng.randrange(site.window_start + 60, compiled_at - 600)
    ref = _fresh_ref(anchor.network, site, when)
    phantom = Report(ref=ref, network=anchor.network, status="reviewed", time=when, lat=round(lat, COORD_DECIMALS),
                     lon=round(lon, COORD_DECIMALS), depth_km=depth, mag=mag, mag_type=anchor.mag_type, region=anchor.region,
                     place=_place_for(anchor, lat, lon, events), role="phantom", trap_id=trap_id)
    reason = rng.choice(site.domain.trap_vocab.retraction_reasons)
    notice = Notice(kind="retraction", target=ref, time=min(compiled_at - 60, when + rng.randint(600, 5400)),
                    reason=reason, trap_id=trap_id)
    description = f"phantom report {ref} (M{mag:g}, {anchor.region}) later retracted: {reason}"
    trap = Trap(trap_id, "retraction", "phantom", ref, description,
                mishandles=[Mishandle(keep_retracted=(ref,)).to_dict()])
    return _Candidate(trap, [phantom], [notice])


def _conflict(trap_id: str, site: _Site) -> _Candidate | None:
    rng, events, compiled_at = site.rng, site.events, site.compiled_at
    e = _pick_unused(events, site.used, rng)
    if e is None:
        return None
    decimals = _decimals(events, e.network)
    step = 0.01 if decimals == 2 else 0.1
    thresholds = threshold_values(site.questions, "mag")
    if thresholds and rng.random() < 0.7:  # aim the wrong value across a threshold that is asked about
        x = rng.choice(thresholds)
        wrong = x - step * rng.randint(1, round(0.2 / step)) if e.mag >= x else x + step * rng.randint(0, round(0.2 / step))
    else:
        wrong = e.mag + rng.choice((-1, 1)) * rng.uniform(0.45, 0.85)
    wrong = round(max(0.0, wrong), decimals)
    if abs(wrong - e.mag) < 0.05:
        return None
    if rng.random() < 0.5:  # form 1: the reviewed report shows the wrong value, a CORRECTION fixes it
        index = next(i for i, r in enumerate(site.reports) if r.ref == e.id and r.role == "primary")
        shown = replace(site.reports[index], mag=wrong, role="conflict", trap_id=trap_id)
        issued = min(compiled_at - 30, max(e.time + 30, e.time + rng.randint(900, 7200)))
        notice = Notice(kind="correction", target=e.id, time=issued, new_mag=e.mag, old_mag=wrong,
                        mag_type=e.mag_type, trap_id=trap_id)
        description = f"{e.id} filed as M{wrong:g} on its reviewed report, then CORRECTED to M{e.mag:g}"
        trap = Trap(trap_id, "conflicting_magnitude", "correction", e.id, description,
                    mishandles=[Mishandle(ignore_correction_for=(e.id,)).to_dict()])
        return _Candidate(trap, [], [notice], replace_at=(index, shown))
    # form 2: a PRELIMINARY report with the same ID disagrees with the REVIEWED one. A careless reader
    # either takes its magnitude, or counts it as a separate quake.
    net = e.network if rng.random() < 0.5 else rng.choice(_rereporters(site, e.network))
    prelim = Report(ref=e.id, network=net, status="preliminary", time=e.time, lat=e.lat, lon=e.lon,
                    depth_km=e.depth_km, mag=wrong, mag_type=e.mag_type, region=e.region, place=e.place,
                    role="conflict", trap_id=trap_id)
    description = f"{e.id} also reported as M{wrong:g} (preliminary, {net.upper()}) beside the reviewed M{e.mag:g}"
    trap = Trap(trap_id, "conflicting_magnitude", "second_report", e.id, description,
                mishandles=[Mishandle(prefer_preliminary_for=(e.id,)).to_dict(),
                            Mishandle(split=((e.id, net, "preliminary"),)).to_dict()])
    return _Candidate(trap, [prelim], [])


# --------------------------------------------------------- the Records of every other domain


def record_report(record: Record, status: str) -> RecordReport:
    """The report of a record that settles its values, filed under `status` (its domain's first)."""
    return RecordReport(ref=record.id, source=record.source, status=status, time=record.time, lat=record.lat,
                        lon=record.lon, group=record.group, values=dict(record.values),
                        qualifiers=dict(record.qualifiers), place=record.place)


def _spread(records: Sequence[Record], spec: FieldSpec) -> float:
    """How far apart a field's values lie in the digest (at least one step of its stored places)."""
    values = [r.values[spec.key] for r in records]
    return max(max(values) - min(values), 10.0 ** -spec.decimals)


def _shown(values: Mapping[str, float], domain: Domain) -> str:
    """Values for a trap's description: "water temperature 12.3 °C, wave height 1.85 m"."""
    return ", ".join(f"{spec.label} {plain_number(values[spec.key])}"
                     + ("" if spec.canonical_unit is None else f" {spec.canonical_unit}") for spec in domain.fields)


def _canonical_thresholds(questions: Sequence[Question], key: str, units: Mapping[str, Sequence[Unit]]) -> list[float]:
    """The thresholds the questions ask about on a field, in its canonical unit (see threshold_values)."""
    steps = [Step(raw, units) for q in questions for raw in q.chain]
    return [canonical_threshold(s) for s in steps if s.kind in ("at_least", "below") and s.field == key]


def _across(threshold: float, value: float, spec: FieldSpec, rng: random.Random, reach: float) -> float:
    """A stored value on the other side of `threshold` (canonical) from `value`: below it when `value`
    passes an "at least" filter there, at or above it when it does not (filters compare as ops does,
    with EPS). It lies at least the field's guard margin from the threshold, as every record of the key
    does (questions.guards_ok), so a careless reader who takes it sees it across the threshold at any
    printed precision, and up to `reach` further."""
    step = 10.0 ** -spec.decimals
    margin = max(spec.margin[0], spec.margin[1] * abs(threshold))
    steps = rng.randint(0, max(0, round(reach / step) - 1))
    direction = -1 if value >= threshold - EPS else 1
    wrong = round(threshold, spec.decimals)
    while (wrong >= threshold - EPS) != (direction > 0) or abs(wrong - threshold) < margin:
        wrong = round(wrong + direction * step, spec.decimals)
    return wrong + direction * steps * step


# ------------------------------------------------- off the legacy path: planted reports in slots' places
# Off the legacy path every domain's traps are planted by these builders, earthquakes included (their 0.2.0
# Events and Reports are read and written in the generic form, Domain.to_generic / from_generic), and so
# are the traps of every plugin domain. A planted report stands where a real record could have stood: it
# is a *slot*, a record drawn with the digest but not kept (pipeline.sample_with_slots), which is drawn
# exactly as the kept records are, and it changes only what its trap needs:
#   phantom        the slot as it is (its time, position, place, group, source, values with their precision,
#                  qualifiers and ID), under the settling status, retracted by a later notice;
#   re-report      the slot under the second status, filed under its own ID with a note naming a real
#                  record of its group, the one nearest to it in time, or under the ID of a real record of
#                  its group that is not one of the slot's neighbours (_record_for), at its place only where
#                  the domain's IDs name it (Domain.ids_name_place) and only for a record alone in the
#                  sequence its ID numbers it in (Domain.id_place: an approach of an object with no other),
#                  with what the slot gives for the record (Domain.rereport_values: its own values by
#                  default, a close approach's with its object's H as another catalogue fits it);
#   second report  the same, under the record's ID, with a value that disagrees with the record's;
#   correction     the record's own settling report, showing a value of a slot of its group (for a field tied to
#                  the record's near twins, a value that agrees with its twin: _twin_values), which a notice
#                  corrects.
# So a planted report has the neighbours, in time, space and value, that a real record has (real records of
# a domain with near twins, Domain.twin_seconds, are sometimes close indeed), even once the reports that
# share its ID are set aside; its source files real records as often as planted ones; and its values and
# ID come from the real records' own ranges and precision: nothing but the signals the reading rules name
# (a shared ID or a note naming it, the status, the notices) sets a planted line apart. When no slot is
# left (a pool hardly larger than a digest, or a direct call without slots), a builder plants a report made
# beside a real record instead (_virtual_slot), which keeps the real neighbour distances but draws its
# values, so it is only statistically like a real one.


def _generic(item: Any, domain: Domain) -> Record:
    """A record of a digest (an Event, or a generic Record) as a generic Record."""
    if isinstance(item, Record):
        return item
    return Record(id=item.id, time=item.time, lat=item.lat, lon=item.lon, group=item.group, source=item.source,
                  values=dict(item.values),
                  qualifiers={key: getattr(item, key) for key in domain.qualifier_keys.values()}, place=item.place)


def _fixed(site: _Site, record: Record) -> bool:
    """Whether `record` stands at a fixed site, a place every report of it gives (a weather station): a
    generic record with a place. An earthquake's place is a note on where it struck."""
    return site.fixed and record.place is not None


def _with_siblings(site: _Site) -> set[str]:
    """The real records whose ID numbers them in a sequence (Domain.id_place: an object's approaches) that
    holds another record, slot or report of the digest: a report filed under such a record's ID would
    have to agree with its siblings, and so copy what it shares with them (an object's H), which no real
    report beside its record does. Empty for a domain whose IDs number nothing."""
    place = site.domain.id_place
    if all(place(r.id) is None for r in site.records):
        return set()
    members: dict[Any, set[str]] = {}
    refs = [r.id for r in (*site.records, *site.slots)] + [g.ref for g in map(site.domain.to_generic, site.reports)]
    for ref in refs:
        at = place(ref)
        if at is not None:
            members.setdefault(at[0], set()).add(ref.casefold())
    return {r.id for r in site.records if place(r.id) is not None and len(members[place(r.id)[0]]) > 1}


def _colocated(site: _Site, same_id: bool) -> bool:
    """Whether a report filed under its record's own ID must stand at the record's place: where the domain's
    IDs name their record's place (Domain.ids_name_place) and its records stand at fixed sites."""
    return same_id and site.fixed and site.domain.ids_name_place


def _slot_pair(site: _Site, *, same_id: bool) -> tuple[Record, Record] | None:
    """A slot not taken yet, drawn at random among those that can report a record, and the record it
    reports, which carries no trap yet (_record_for). None when no slot can.

    A report stands where its slot stands, wherever its record is, except that one filed under its
    record's own ID (`same_id`) in a domain whose IDs name their record's place (Domain.ids_name_place)
    stands at that place, as its ID says, so its record must stand at the slot's place. Such a report can
    then never be the only one at its place, as a real record can; its slot is drawn with a weight of one
    over the number of real records at its place, so that it joins a place of few reports more often than
    one of many. (No built-in domain's IDs name a place: a weather observation's ID is an opaque serial.) A
    report under its record's own ID reports only a record whose ID shares its sequence with no other
    record, slot or report (_with_siblings: an approach whose object makes no other approach in the digest
    or its slots), as one beside a sibling would have to repeat what the siblings share. Each placement of
    a trap draws among the slots that its earlier placements have not tried while one of them can report
    a record, so that the placements cover the digest's slots rather than draw some of them again."""
    free = [s for s in site.slots if s.id not in site.filled]
    barred = _with_siblings(site) if same_id else set()
    colocated = _colocated(site, same_id)
    options: list[tuple[Record, list[Record]]] = []
    weights: list[float] = []
    for pool in ([s for s in free if s.id not in site.tried], free):
        for slot in pool:
            fixed = colocated and _fixed(site, slot)
            mates = [r for r in site.records if r.id not in site.used and r.id not in barred
                     and r.group == slot.group
                     and (not colocated or r.place == slot.place or not (fixed or _fixed(site, r)))]
            if mates:
                options.append((slot, mates))
                weights.append(1.0 / sum(r.place == slot.place for r in site.records) if fixed else 1.0)
        if options:
            break
    if not options:
        return None
    slot, mates = site.rng.choices(options, weights=weights, k=1)[0]
    site.tried.add(slot.id)
    return slot, _record_for(site, slot, mates, same_id=same_id)


def _record_for(site: _Site, slot: Record, mates: Sequence[Record], *, same_id: bool) -> Record:
    """The record of `mates` (real records of the slot's group that can carry its trap) that a report in
    `slot`'s place reports.

    Under an ID of its own, with a note naming the record, it is the record nearest to the slot in time
    (ties: the earlier, then the smaller ID). Under the record's own ID it is a record drawn at random
    among those that are not the slot's own neighbours, while another is left: not the real record of its
    group nearest to it in time (at its place, where the report must stand there: _colocated), not one
    that shares its twin key within the domain's twin window (Domain.twin_key, twin_seconds), and not one at
    its place, where it stands somewhere else. A reader who sets aside the reports that share its ID (they
    are one record, rule 1) then sees around the report exactly the neighbours its slot has, as around a
    real record: its nearest report, the twins about it and the reports at its place are still there. Had
    it reported its nearest record, the report would have had, once that one was set aside, only the
    neighbours of a record further from everything (an AUC of about 0.6 for the gap to its nearest other
    report of its group)."""
    def gap(r: Record) -> tuple:
        return abs(r.time - slot.time), r.time, r.id

    nearest = min(mates, key=gap)
    if not same_id:
        return nearest
    domain = site.domain
    colocated = _colocated(site, same_id) and _fixed(site, slot)
    near = [r for r in site.records if r.group == slot.group and (not colocated or r.place == slot.place)]
    neighbours = {min(near, key=gap).id} if near else set()
    if domain.twin_seconds is not None:
        key = domain.twin_key(slot)
        neighbours |= {r.id for r in site.records
                       if abs(r.time - slot.time) <= domain.twin_seconds and domain.twin_key(r) == key}
    if not colocated and slot.place is not None:
        neighbours |= {r.id for r in site.records if r.place == slot.place}
    pool = [r for r in mates if r.id not in neighbours] or [r for r in mates if r.id != nearest.id] or list(mates)
    return site.rng.choice(pool)


def _neighbour(record: Record, records: Sequence[Record]) -> Record | None:
    """The record of `records` other than `record` nearest to it in time (ties: the earlier, then the
    smaller ID), or None."""
    others = [r for r in records if r.id != record.id]
    return min(others, key=lambda r: (abs(r.time - record.time), r.time, r.id)) if others else None


def _peers(record: Record, spec: FieldSpec, site: _Site) -> list[float]:
    """The values of `spec` that `record`'s source files in the digest: what its precision is read from."""
    return [r.values[spec.key] for r in site.records if r.source == record.source]


def _virtual_slot(site: _Site, anchor: Record, *, earliest: int, latest: int, at_place: bool) -> Record:
    """When no slot is left: a record that could have been drawn beside `anchor`, a real record. A donor, a
    real record of the anchor's group (at its place, at a fixed site, when `at_place`) drawn at random,
    lends it its source, its qualifiers, the precision of its values and the shape of its ID; it lies as far
    from the anchor in time as the donor lies from its nearest neighbour (before or after the anchor, within
    [earliest, latest]); at a fixed site, at the anchor's place when `at_place` and at the donor's else, and
    elsewhere as far from the anchor as the donor is from that neighbour, in a random direction; and its
    values are drawn from the group's (a value of the group moved by up to 5% of the field's spread in the
    digest, folded back into the range the group's values cover), not copied from any record."""
    rng, domain, records = site.rng, site.domain, site.records
    fixed = _fixed(site, anchor)
    donors = ([r for r in records if r.group == anchor.group and (not (fixed and at_place) or r.place == anchor.place)]
              or [anchor])
    donor = rng.choice(donors)
    near = _neighbour(donor, records)
    gap = abs(near.time - donor.time) if near is not None else rng.randint(60, 3600)
    times = [t for t in (anchor.time - gap, anchor.time + gap) if earliest <= t <= latest]
    when = rng.choice(times) if times else min(max(anchor.time, earliest), latest)
    lat, lon, place = (donor.lat, donor.lon, donor.place) if fixed else (anchor.lat, anchor.lon, anchor.place)
    spatial = domain.spatial
    if spatial is not None and not fixed:
        km = haversine_km(donor.lat, donor.lon, near.lat, near.lon) if near is not None else rng.uniform(1.0, 40.0)
        lat, lon = destination(anchor.lat, anchor.lon, km, rng.uniform(0.0, 360.0))
        lat, lon = round(lat, spatial.coord_decimals) + 0.0, round(lon, spatial.coord_decimals) + 0.0
        place = _place_for(anchor, lat, lon, records) if anchor.place else None
    peers = [r for r in records if r.group == anchor.group]
    pool = peers if len(peers) >= 3 else records
    values = {}
    for spec in domain.fields:
        observed = sorted(r.values[spec.key] for r in pool)
        drawn = observed[int(rng.random() * (len(observed) - 1))] + rng.uniform(-0.05, 0.05) * _spread(records, spec)
        drawn = folded(drawn, observed[0], observed[-1])  # never a value no real record of the group comes near
        values[spec.key] = domain.planted_value(spec, drawn, donor.values[spec.key], _peers(donor, spec, site))
    ref = (_fresh_ref(donor.source, site, when) if isinstance(domain, Earthquakes)
           else domain.mimic_id(donor.id, rng, site.taken))
    return Record(id=ref, time=when, lat=lat, lon=lon, group=anchor.group, source=donor.source, values=values,
                  qualifiers=dict(donor.qualifiers), place=place)


def _virtual_pair(site: _Site, *, same_id: bool) -> tuple[Record, Record] | None:
    """When no slot is left: a record that carries no trap yet, drawn at random (for a report under its own ID,
    one without siblings: see _slot_pair), and a slot made beside it (at its place, where the report must
    stand there). A re-report planned for a digest whose every record carries a trap already re-reports a record
    that only re-reports carry (two sources may re-report one record; a conflict is never planted beside a
    re-report), so that a digest prints every report its plan adds (draw_plan). None when there is no such
    record."""
    barred = _with_siblings(site) if same_id else set()
    record = _pick_unused(site.records, site.used | barred, site.rng)
    if record is None and site.form is not None:
        record = _pick_unused(site.records, site.conflicted | barred, site.rng)
    if record is None:
        return None
    return _virtual_slot(site, record, earliest=site.window_start, latest=site.compiled_at - 1,
                         at_place=_colocated(site, same_id)), record


def _filed_as(slot: Record, record: Record, site: _Site, *, ref: str, same_as: str | None, status: str, role: str,
              trap_id: str) -> RecordReport:
    """`slot` filed as a report of `record`: under `ref` (and the note `same_as`) and `status`, with what the
    slot gives for the record (Domain.rereport_values); its own time, position, place, source and qualifiers."""
    values = site.domain.rereport_values(record, slot, site.rng, same_id=ref == record.id)
    return RecordReport(ref=ref, source=slot.source, status=status, time=slot.time, lat=slot.lat, lon=slot.lon,
                        group=record.group, values=values, qualifiers=dict(slot.qualifiers), place=slot.place,
                        same_as=same_as, role=role, trap_id=trap_id)


def _slot_duplicate(trap_id: str, site: _Site) -> _Candidate | None:
    """A re-report in the form its trap was planned in (_Site.form; drawn for each placement without a plan)."""
    rng, domain = site.rng, site.domain
    form = site.form or rng.choice(("same_id", "cross_ref"))
    pair = _slot_pair(site, same_id=form == "same_id") or _virtual_pair(site, same_id=form == "same_id")
    if pair is None:
        return None
    slot, e = pair
    status = domain.statuses[1]
    ref = e.id if form == "same_id" else slot.id  # a cross-reference: the slot's own ID, with a note naming the record
    dup = _filed_as(slot, e, site, ref=ref, same_as=None if form == "same_id" else e.id, status=status,
                    role="duplicate", trap_id=trap_id)
    how = f"under the same {domain.id_label}" if form == "same_id" else f"as {ref} with a note naming it"
    description = (f"{slot.source} re-reported {e.id} {how}: {status} {_shown(dup.values, domain)} "
                   f"({domain.statuses[0]}: {_shown(e.values, domain)})")
    trap = Trap(trap_id, "duplicate", form, e.id, description,
                mishandles=[Mishandle(split=((ref, slot.source, status),)).to_dict()])
    return _Candidate(trap, [domain.from_generic(dup)], [], slot=slot.id)


def _virtual_phantom(site: _Site) -> Record:
    """When no slot is left: a phantom's slot made beside a real record drawn at random (_virtual_slot)."""
    anchor = site.rng.choice(site.records)
    return _virtual_slot(site, anchor, earliest=site.window_start + 60, latest=site.compiled_at - 60 - NOTICE_DELAY[0],
                         at_place=True)


# Off the legacy path a notice about a report is issued 92 minutes to 3 hours after the report's time (a retraction
# always so: its phantom lies that long before the compilation; a correction so but for one of a record that close
# to the compilation, which is issued before it), later than any genre sends a report (a form is sent up to 90
# minutes and 59 seconds after its record's time). So no genre sends a planted report sooner because a notice about
# it follows: a form, whose sent time is kept before any notice about its record, was sent a median 21 minutes after
# a phantom's time when retractions came 10 to 90 minutes after it, against 44 minutes after a real record's (an
# AUC of 0.28 to 0.29, and 0.36 for corrected records), and every genre now sends planted reports as it sends real
# ones. (The 0.2.0 builders keep their own times.)
NOTICE_DELAY = (92 * 60, 3 * 3600)


def _slot_phantom(trap_id: str, site: _Site) -> _Candidate:
    """A slot filed as it is, under its settling status, which a later notice retracts. Its retraction is
    issued 92 minutes to 3 hours after it (NOTICE_DELAY), before the digest is compiled, so the slot lies
    before then."""
    rng, domain = site.rng, site.domain
    open_slots = [s for s in site.slots if s.id not in site.filled
                  and site.window_start + 60 <= s.time <= site.compiled_at - 60 - NOTICE_DELAY[0]]
    open_slots = [s for s in open_slots if s.id not in site.tried] or open_slots  # untried first (_slot_pair)
    slot = rng.choice(open_slots) if open_slots else _virtual_phantom(site)
    site.tried.add(slot.id)
    phantom = RecordReport(ref=slot.id, source=slot.source, status=domain.statuses[0], time=slot.time, lat=slot.lat,
                           lon=slot.lon, group=slot.group, values=dict(slot.values), qualifiers=dict(slot.qualifiers),
                           place=slot.place, role="phantom", trap_id=trap_id)
    reason = rng.choice(domain.trap_vocab.retraction_reasons)
    notice = Notice(kind="retraction", target=slot.id,
                    time=min(site.compiled_at - 60, slot.time + rng.randint(*NOTICE_DELAY)), reason=reason,
                    trap_id=trap_id)
    description = f"phantom report {slot.id} ({_shown(slot.values, domain)}, {slot.group}) later retracted: {reason}"
    trap = Trap(trap_id, "retraction", "phantom", slot.id, description,
                mishandles=[Mishandle(keep_retracted=(slot.id,)).to_dict()])
    return _Candidate(trap, [domain.from_generic(phantom)], [notice], slot=slot.id)


NEAREST_DONORS = 3  # a correction's wrong value is the value of one of the slots nearest its record (_wrong_value)
TWIN_PAIRS = 8  # a tied value is moved as one of the real twin pairs this near its gap moved (_twin_values)


def _as_record(item: Any, domain: Domain) -> Record:
    """A record, or a report of the digest, as a generic Record (a report's ID is its ref)."""
    if isinstance(item, Record):
        return item
    g = domain.to_generic(item)
    return Record(id=g.ref, time=g.time, lat=g.lat, lon=g.lon, group=g.group, source=g.source, values=dict(g.values),
                  qualifiers=dict(g.qualifiers), place=g.place)


def _twin_pairs(site: _Site) -> list[tuple[int, dict[str, float]]]:
    """The real twin pairs among the digest's records and slots: two records with one twin key (Domain.twin_key)
    within twice the domain's twin window of each other, each pair as the seconds between them and how far each
    value moved from the earlier to the later (in time order, then ID order). Read once a digest."""
    if site.pairs is None:
        domain = site.domain
        window = 2 * (domain.twin_seconds or 0)
        members: dict[Any, list[Record]] = {}
        for r in sorted((*site.records, *site.slots), key=lambda r: (r.time, r.id)):
            key = domain.twin_key(r)
            if key is not None:
                members.setdefault(key, []).append(r)
        site.pairs = [(b.time - a.time, {k: b.values[k] - a.values[k] for k in a.values if k in b.values})
                      for twins in members.values() for a, b in itertools.combinations(twins, 2)
                      if b.time - a.time <= window]
    return site.pairs


def _twin_values(e: Record, spec: FieldSpec, site: _Site) -> list[float] | None:
    """For a field whose values a record shares, give or take, with its near twins (Domain.twin_fields): the
    values that `e`'s settling report can show and still agree with its twins as its own value does. Its twin
    is the report of the digest, or the free slot, nearest to it in time that shares its twin key (a station's
    other observation, an object's other approach) within twice the domain's twin window, under another ID;
    each value is that twin's moved as one of the real twin pairs of the digest's records and slots moved over
    about as long a time (the TWIN_PAIRS pairs whose gap is nearest to the gap between `e` and its twin first),
    so that a wrong value lies from its twin as a real value lies from its own: a station's temperature never
    jumps from its last hour's as no real one does, nor an object's speed past the Moon from its speed past
    Earth. None when the field is not tied or `e` has no twin (any value of its group will do); no value when
    no twin pair is there to move it by."""
    domain = site.domain
    if spec.key not in domain.twin_fields or domain.twin_seconds is None:
        return None
    key = domain.twin_key(e)
    if key is None:
        return None
    window = 2 * domain.twin_seconds
    free = [s for s in site.slots if s.id not in site.filled]
    near = [r for r in (*(_as_record(r, domain) for r in site.reports), *free)
            if r.id.casefold() != e.id.casefold() and abs(r.time - e.time) <= window and domain.twin_key(r) == key]
    if not near:
        return None
    twin = min(near, key=lambda r: (abs(r.time - e.time), r.time, r.id))
    gap, sign = abs(e.time - twin.time), 1 if e.time >= twin.time else -1
    pairs = sorted((p for p in _twin_pairs(site) if spec.key in p[1]), key=lambda p: abs(p[0] - gap))
    return [twin.values[spec.key] + sign * moved[spec.key] for _, moved in pairs[:TWIN_PAIRS]]


def _wrong_value(e: Record, spec: FieldSpec, site: _Site) -> float | None:
    """The wrong value of `spec` that `e`'s settling report shows (a correction fixes it): the value of a slot
    that no trap has taken, of `e`'s group when one is (of any group else), written at `e`'s precision
    (Domain.planted_value). It is thus a real value of the group, which no line of the digest prints, and,
    taken from one of the slots nearest to `e` (at its place first, then nearest in time) whose value has as
    many places as `e`'s, one that fits the rest of `e`'s report as a real value would (in whole degrees in a
    five-minute observation and in tenths else). Where the field's values are tied to a record's twins'
    (Domain.twin_fields: a station's temperature to its observations an hour away, an object's speed past the
    Moon to its speed past Earth) and `e` has a twin, it is a value that agrees with that twin as `e`'s own
    does instead (_twin_values; None when none differs from `e`'s by more than twice the tolerance), and
    without a twin a slot value of its group from within twice the twin window of its time. A value of a slot
    of the group, which was often another station's, jumped a median 4.5 °C from its station's nearest
    other report, where real reports jump 1 °C (an AUC of 0.84 to 0.90), and a corrected velocity left its
    object's other approach by more than any real one does (0.75 to 0.81).

    Up to seven times in ten it is aimed across a threshold that the questions ask about on the field: a
    slot value on the other side of it from `e`'s, at least the field's guard margin beyond it (filters
    compare as ops does, with EPS), so that a careless reader sees it across the threshold at any printed
    precision. Aimed values come from one side of a threshold, so the aim is taken from the side with the
    larger share of the group's values only (share there / share here) ** 2 as often: then a wrong value
    lies on each side of the threshold, and as far beyond it, as often as a real value does. Without a slot
    value to take, it is a value across a threshold (_across) or one moved by a share of the field's spread,
    folded back into the range of the digest's values of the field, so that it is never one no real record
    comes near (a miss distance of 0 km)."""
    rng, domain = site.rng, site.domain
    value, spread = e.values[spec.key], _spread(site.events, spec)
    free = [s for s in site.slots if s.id not in site.filled]
    tied = _twin_values(e, spec, site)
    if tied is not None:
        real = [r.values[spec.key] for r in (*site.records, *site.slots)]
        pool = [folded(v, min(real), max(real)) for v in tied]  # nearest gap first
    else:
        donors = sorted([s for s in free if s.group == e.group] or free,
                        key=lambda s: (s.place != e.place, abs(s.time - e.time), s.id))
        if spec.key in domain.twin_fields and domain.twin_seconds is not None:  # at about its time
            donors = [s for s in donors if abs(s.time - e.time) <= 2 * domain.twin_seconds] or donors
        places = stored_places(value, spec.decimals)
        donors = [s for s in donors if stored_places(s.values[spec.key], spec.decimals) == places] or donors
        pool = [s.values[spec.key] for s in donors]  # nearest first
    peers = _peers(e, spec, site)

    def written(v: float) -> float:
        return domain.planted_value(spec, v, value, peers)

    def beyond(x: float, v: float) -> bool:
        """`v` lies on the other side of the threshold `x` from `e`'s value, at least the guard margin from it."""
        side = v < x - EPS if value >= x - EPS else v >= x - EPS  # e passes "at least x": its wrong value does not
        return side and abs(v - x) >= max(spec.margin[0], spec.margin[1] * abs(x))

    thresholds = _canonical_thresholds(site.questions, spec.key, domain.units)
    aimed = [x for x in thresholds if any(beyond(x, v) for v in pool)]  # only across a threshold slot values cross
    if aimed:
        x = rng.choice(aimed)
        there = [v for v in pool if beyond(x, v)]
        here = sum((v >= x - EPS) == (value >= x - EPS) for v in pool) + 1  # e's side, e included
        if rng.random() < 0.7 * min(1.0, (len(there) / here) ** 2):
            wrong = written(rng.choice(there[:NEAREST_DONORS]))
            if beyond(x, wrong):
                return wrong
    tolerance = max(spec.tolerance[0], spec.tolerance[1] * abs(value))
    options = [v for v in pool if abs(written(v) - value) > 2 * tolerance]
    if options:
        return written(rng.choice(options[:NEAREST_DONORS]))
    if tied is not None:
        return None  # no value that agrees with its twin differs from its own: this placement finds no room
    least, most = min(r.values[spec.key] for r in site.records), max(r.values[spec.key] for r in site.records)
    if thresholds and rng.random() < 0.7:
        return written(folded(_across(rng.choice(thresholds), value, spec, rng, reach=0.1 * spread), least, most))
    return written(folded(value + rng.choice((-1, 1)) * rng.uniform(0.2, 0.45) * spread, least, most))


def _slot_conflict(trap_id: str, site: _Site) -> _Candidate | None:
    """A conflicting value in the form the trap drew (_Site.correction): the settling report shows a wrong
    value and a correction fixes it, or a second report under the same ID, in a slot's place, gives values
    of its own (Domain.rereport_values) that disagree with the record's on the trap's field. The form is
    drawn once per trap, not per placement, so that which form a digest gets never depends on which form
    finds a placement more easily in it (a correction adds no line, and digests of either form must not
    differ in size)."""
    rng, domain = site.rng, site.domain
    spec = rng.choice([s for s in domain.fields if s.correctable] or list(domain.fields))
    kind = "conflicting_magnitude" if isinstance(domain, Earthquakes) else "conflicting_value"
    unit = "" if spec.canonical_unit is None else f" {spec.canonical_unit}"
    final, draft = domain.statuses
    if spec.correctable and site.correction:  # the settling report shows the wrong value, a correction fixes it
        e = _pick_unused(site.records, site.used, rng)
        if e is None:
            return None
        value = e.values[spec.key]
        wrong = _wrong_value(e, spec, site)
        if wrong is None or abs(wrong - value) <= max(spec.tolerance[0], spec.tolerance[1] * abs(value)):
            return None  # no room for a wrong value, or a careless reader would still be right
        index = next(i for i, r in enumerate(site.reports) if r.ref == e.id and r.role == "primary")
        primary = domain.to_generic(site.reports[index])
        shown = replace(primary, values={**primary.values, spec.key: wrong}, role="conflict", trap_id=trap_id)
        issued = min(site.compiled_at - 30, e.time + rng.randint(*NOTICE_DELAY))  # see NOTICE_DELAY
        notice = Notice(kind="correction", target=e.id, time=issued, trap_id=trap_id, field=spec.key, new_value=value,
                        old_value=wrong, qualifier=e.qualifiers.get(spec.qualifier) if spec.qualifier else None)
        description = (f"{e.id} filed with {spec.label} {plain_number(wrong)}{unit} on its {final} report, then "
                       f"CORRECTED to {plain_number(value)}{unit}")
        trap = Trap(trap_id, kind, "correction", e.id, description,
                    mishandles=[Mishandle(ignore_correction_for=(e.id,)).to_dict()])
        return _Candidate(trap, [], [domain.from_generic(notice)], replace_at=(index, domain.from_generic(shown)))
    # A second report under the same ID, in a slot's place. A careless reader either takes its values, or
    # counts it as a record of its own.
    pair = _slot_pair(site, same_id=True) or _virtual_pair(site, same_id=True)
    if pair is None:
        return None
    slot, e = pair
    value = e.values[spec.key]
    second = _filed_as(slot, e, site, ref=e.id, same_as=None, status=draft, role="conflict", trap_id=trap_id)
    wrong = second.values[spec.key]
    if abs(wrong - value) <= max(spec.tolerance[0], spec.tolerance[1] * abs(value)):
        return None  # a careless reader would still be right
    description = (f"{e.id} also reported with {spec.label} {plain_number(wrong)}{unit} ({draft}, {slot.source}) "
                   f"beside the {final} {plain_number(value)}{unit}")
    trap = Trap(trap_id, kind, "second_report", e.id, description,
                mishandles=[Mishandle(prefer_preliminary_for=(e.id,)).to_dict(),
                            Mishandle(split=((e.id, slot.source, draft),)).to_dict()])
    return _Candidate(trap, [domain.from_generic(second)], [], slot=slot.id)


EARTHQUAKE_BUILDERS = Builders(primary_report, _duplicate, _phantom, _conflict)  # the 0.2.0 builders


def builders_for(domain: Domain, *, slotted: bool = False) -> Builders:
    """The builders that plant traps in a domain's digests. On the legacy path (without `slotted`) an
    earthquake domain, whose records are 0.2.0 Events and Reports, keeps the 0.2.0 builders. Every other
    domain, which has no legacy path, and every domain off it (`slotted`), is planted in the place of its
    slots by the same builders, whose settling reports carry the domain's first status."""
    if isinstance(domain, Earthquakes) and not slotted:
        return EARTHQUAKE_BUILDERS
    primary = (primary_report if isinstance(domain, Earthquakes)
               else functools.partial(record_report, status=domain.statuses[0]))
    return Builders(primary, _slot_duplicate, _slot_phantom, _slot_conflict)


def inject_traps(events: list[Event], questions: list[Question], rng: random.Random, *, compiled_at: int,
                 window_hours: int, config: TrapConfig = TrapConfig(), domain: Domain = EARTHQUAKES,
                 slots: Sequence[Any] = ()) -> tuple[list[Report], list[Notice], list[Trap]]:
    """Build the digest's reports and notices: one settling report per record (under the domain's first
    status: REVIEWED for earthquakes), plus the traps. Each trap is the best of up to `config.candidates`
    placements (_rank), guarded off the legacy path, where the planted reports take the places of `slots`,
    records drawn with the digest but not kept."""
    build = builders_for(domain, slotted=config.guarded)
    units = domain.units
    site = _Site(domain, events, questions, rng, compiled_at, window_start=compiled_at - window_hours * 3600,
                 id_style=config.id_style, reports=[build.primary(e) for e in events],
                 taken={e.id for e in events} | {s.id for s in slots}, used=set(),
                 slots=[_generic(s, domain) for s in slots], records=[_generic(e, domain) for e in events],
                 fixed=not isinstance(domain, Earthquakes))
    notices: list[Notice] = []
    traps: list[Trap] = []
    key = {q.qid: execute(q.chain, events, units=units) for q in questions}
    time_margin = Context(compiled_at, window_hours).time_margin if config.guarded else None
    top = GUARDED_TOP if config.guarded else TOP
    builders = {"duplicate": build.duplicate, "retraction": build.retraction, "conflict": build.conflict}
    kinds: list[tuple[str, str | None]]  # each trap's kind, and its form when it is planned (None: drawn)
    if config.plan is None:  # the 0.2.0 plan
        kinds = [(kind, None) for kind in ["duplicate"] * rng.randint(*config.duplicates)
                 + ["retraction"] * config.retractions + ["conflict"] * config.conflicts]
    else:  # off the legacy path: the digest's own plan (draw_plan), the re-reports' forms in a drawn order, and the
        # traps that add a report first, the second reports (which need a record free of traps) before the re-reports
        # (which may share one: _virtual_pair) and the corrections last (they need one too, but add no report), so
        # that the digest prints every report its plan adds
        plan = config.plan
        duplicates = ["same_id"] * plan.same_ids + ["cross_ref"] * plan.cross_refs
        rng.shuffle(duplicates)
        kinds = ([("conflict", "second_report")] * plan.second_reports + [("duplicate", form) for form in duplicates]
                 + [("retraction", "phantom")] * plan.retractions + [("conflict", "correction")] * plan.corrections)
    scopes = _Scopes(questions, units) if config.match_scope else None
    real_scopes = [] if scopes is None else [scopes.count(e) for e in events]
    leads = _Leads(questions, site.records, units, domain) if config.match_scope else None
    for number, (kind, form) in enumerate(kinds, start=1):
        trap_id = f"T{number}"
        best: tuple[tuple, int, _Candidate] | None = None
        # Off the legacy path, a scope and a lead drawn from the real records' (see TrapConfig.match_scope), and
        # a re-report's or a conflict's form, planned (or drawn) once for the trap (_slot_duplicate,
        # _slot_conflict).
        target = rng.choice(real_scopes) if real_scopes else None
        lead = leads.draw(rng) if leads is not None and target is not None else {}
        site.form = form if kind == "duplicate" else None
        if kind == "conflict" and config.guarded:
            site.correction = form == "correction" if form is not None else rng.random() < 0.5
        site.tried = set()

        def rank_of(candidate: _Candidate) -> int:
            trial = DigestData(*_apply(site.reports, notices, candidate))
            return _rank(_single_readings(candidate.trap, questions, trial, domain, time_margin), key, config.guarded)

        def planted_at(candidate: _Candidate) -> int:
            """Where a placement's planted report stands in the digest: the settling report it shows a wrong
            value on, or the first report it adds."""
            return candidate.replace_at[0] if candidate.replace_at is not None else len(site.reports)

        def leads_like(candidate: _Candidate) -> bool:
            if leads is None or not (lead or leads.planted):
                return True
            trial, _ = _apply(site.reports, notices, candidate)
            return leads.like([domain.to_generic(r) for r in trial], planted_at(candidate), lead)

        if target is None:  # the 0.2.0 search: placements in turn, until one of the top rank turns up
            for _ in range(config.candidates):
                candidate = builders[kind](trap_id, site)
                if candidate is None:
                    continue
                rank = rank_of(candidate)
                if best is None or rank > best[1]:
                    best = ((rank,), rank, candidate)
                if rank == top:
                    break
        else:
            best = _matched_placement([builders[kind](trap_id, site) for _ in range(config.candidates)], target,
                                      scopes, rank_of, top, leads_like)
        if best is None:
            continue
        chosen = best[2]
        if leads is not None and lead:
            leads.planted.append((planted_at(chosen), lead))
        site.reports, notices = _apply(site.reports, notices, chosen)
        traps.append(chosen.trap)
        site.used.add(chosen.trap.target)
        if kind == "conflict":
            site.conflicted.add(chosen.trap.target)
        if chosen.slot is not None:
            site.filled.add(chosen.slot)
    reports = site.reports
    # authoritative, on the final digest
    score_traps(traps, questions, key, reports, notices, domain=domain, time_margin=time_margin)
    problems = check_resolution(reports, notices, events, domain=domain)
    if problems:
        raise AssertionError("trap injection broke the reading-rules invariant: " + "; ".join(problems))
    return reports, notices, traps


def check_resolution(reports: list[Report], notices: list[Notice], events: list[Event], *,
                     domain: Domain = EARTHQUAKES) -> list[str]:
    """Generator invariant: applying the reading rules to the digest recovers the snapshot exactly."""
    got = {e.id: e for e in domain.resolve(reports, notices)}
    want = {e.id: e for e in events}
    problems = []
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    if missing:
        problems.append(f"missing events {missing}")
    if extra:
        problems.append(f"unexpected events {extra}")
    for event_id in sorted(set(want) & set(got)):
        if got[event_id] != want[event_id]:
            problems.append(f"{event_id} resolves to {got[event_id]} instead of {want[event_id]}")
    return problems
