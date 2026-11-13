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
still planted, as plain noise. Planted lines copy the style of the real ones: place notes that
agree with their coordinates, the network's magnitude precision, and IDs that fit the network's
own ID sequence, so the format gives nothing away.

Off the legacy path, placement is also guarded (``TrapConfig.guarded``): a careless reading counts
only where the misread digest passes the same guards as the key (questions.guards_ok): no record,
planted ones included, within a threshold's margin, no tie or near tie in a ranked or nearest answer,
no record at a time window's edge. Only then is the wrong answer the one that a reader who makes that
mistake gets from the text at its printed precision, rather than one decided by the stored values'
last digit or by a tie-break. A placement whose every reading passes is preferred to any other, so a
trap is planted unguarded only when no guarded placement turned up, and its affected questions are
then only those its guarded readings change. The 0.2.0 placement of the legacy path is unchanged.

Placement, ranking and scoring are the same for every domain. What a trap looks like depends on the
domain's records, so the builders are looked up per domain (``builders_for``), and they take the
domain's trap vocabulary (re-reporting sources, retraction reasons), ID scheme, statuses and reading
rules from the domain itself. Earthquakes keep the 0.2.0 builders. The Records of every other domain
get the same three traps, with the same careless readings:
  duplicate          a re-reporting source repeats a record (its ID, or its own reference with a note
                     naming the ID), with slightly different values and the domain's second status
  retraction         a phantom report, plausible for its group, that a later notice retracts
  conflicting_value  a wrong value of a field (a correctable one when there is one), settled by a
                     correction notice or by the settling report next to an unsettled one
Values move by a share of the spread of their field in the digest, so the traps scale to every field,
and a wrong value aimed across a threshold lands at least the field's guard margin beyond it.
"""
from __future__ import annotations

import functools
import itertools
import math
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from .domains.base import Domain, FieldSpec, Record, RecordReport
from .domains.earthquakes import EARTHQUAKES, Earthquakes
from .geo import COMPASS16, compass_point, destination, haversine_km, initial_bearing
from .models import Event, Notice, Question, Report, Trap
from .ops import EPS, DigestData, Step, canonical_threshold, execute
from .questions import Context, guards_ok
from .rules import Mishandle
from .scoring import get_answer_type, normalize, same_answer
from .sources import COORD_DECIMALS
from .units import Unit
from .util import plain_number

_PLACE = re.compile(r"^(\d+(?:\.\d+)?) km ([NSEW]{1,3}) of (.+)$")


@dataclass(frozen=True)
class TrapConfig:
    duplicates: tuple[int, int] = (2, 3)  # inclusive range per instance
    retractions: int = 1
    conflicts: int = 1
    candidates: int = 25  # placements tried per trap, looking for a load-bearing, clean one
    id_style: str = "random"  # "random" (synthetic IDs) or "sequence" (fit a real feed's ID sequence)
    # Off the legacy path: a careless reading counts only where the misread digest passes the key's
    # guards, and a placement whose every reading does is preferred (see the module docstring). False is
    # the 0.2.0 placement, which the legacy path keeps.
    guarded: bool = False


@dataclass
class _Candidate:
    trap: Trap
    add_reports: list[Any]  # Reports, or the RecordReports of a generic domain
    add_notices: list[Notice]
    replace_at: tuple[int, Any] | None = None


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


def score_traps(traps: list[Trap], questions: list[Question], key: dict, reports: list[Report],
                notices: list[Notice], *, domain: Domain = EARTHQUAKES, time_margin: int | None = None) -> None:
    """Fill trap.affected (misread alone) and trap.fallen (misread alone or with any other traps).

    With a `time_margin` (guarded placement), `affected` keeps only the questions whose misread
    digest passes their guards (_single_readings): a trap's exposure, its catch rate and the strata
    of every gap count only careless readings that a reader of the text is held to. `fallen` keeps
    every misread answer that is not exactly the key, as computed from the stored values, including
    ones the scorer would still accept, so that a careless reader whose rounded answer lands just past
    the edge is attributed to the trap. Answers are deduplicated by exact value."""
    data = DigestData(reports, notices)
    units = domain.units
    for trap in traps:
        trap.affected = _effects(_single_readings(trap, questions, data, domain, time_margin), key)
        trap.fallen = {}
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
    """How far apart a field's values lie in the digest (at least one step of its stored places): planted
    values move by shares of it, so a trap means as much for a wave height as for a temperature."""
    values = [r.values[spec.key] for r in records]
    return max(max(values) - min(values), 10.0 ** -spec.decimals)


def _stored(value: float, spec: FieldSpec) -> float:
    """A planted value as the field stores it: to its places, and never below zero unless it is signed."""
    return round(value if spec.signed else max(0.0, value), spec.decimals) + 0.0  # + 0.0: never -0.0


def _position(anchor: Record, km: tuple[float, float], site: _Site) -> tuple[float | None, float | None]:
    """Where a planted report puts itself: nowhere in a domain without locations, at its anchor when the
    anchor names its place (a station does not move), and else `km` away from it in a random direction."""
    spatial = site.domain.spatial
    if spatial is None:
        return None, None
    if anchor.place is not None:
        return anchor.lat, anchor.lon
    lat, lon = destination(anchor.lat, anchor.lon, site.rng.uniform(*km), site.rng.uniform(0.0, 360.0))
    return round(lat, spatial.coord_decimals) + 0.0, round(lon, spatial.coord_decimals) + 0.0


def _shown(values: Mapping[str, float], domain: Domain) -> str:
    """Values for a trap's description: "water temperature 12.3 °C, wave height 1.85 m"."""
    return ", ".join(f"{spec.label} {plain_number(values[spec.key])}"
                     + ("" if spec.canonical_unit is None else f" {spec.canonical_unit}") for spec in domain.fields)


def _record_duplicate(trap_id: str, site: _Site) -> _Candidate | None:
    rng, records, domain = site.rng, site.events, site.domain
    e = _pick_unused(records, site.used, rng)
    if e is None:
        return None
    source = rng.choice(_rereporters(site, e.source))
    form = rng.choice(("same_id", "cross_ref"))
    when = min(site.compiled_at - 1, max(site.window_start, e.time + rng.choice((-1, 1)) * rng.randint(3, 40)))
    ref = e.id if form == "same_id" else _fresh_ref(source, site, when)
    lat, lon = _position(e, (1.0, 8.0), site)
    values = {spec.key: _stored(e.values[spec.key] + rng.uniform(-0.08, 0.08) * _spread(records, spec), spec)
              for spec in domain.fields}
    status = domain.statuses[1]
    dup = RecordReport(ref=ref, source=source, status=status, time=when, lat=lat, lon=lon, group=e.group,
                       values=values, qualifiers=dict(e.qualifiers), place=e.place,
                       same_as=None if form == "same_id" else e.id, role="duplicate", trap_id=trap_id)
    how = f"under the same {domain.id_label}" if form == "same_id" else f"as {ref} with a note naming it"
    description = (f"{source} re-reported {e.id} {how}: {status} {_shown(values, domain)} "
                   f"({domain.statuses[0]}: {_shown(e.values, domain)})")
    trap = Trap(trap_id, "duplicate", form, e.id, description,
                mishandles=[Mishandle(split=((ref, source, status),)).to_dict()])
    return _Candidate(trap, [dup], [])


def _record_phantom(trap_id: str, site: _Site) -> _Candidate:
    """A report of a record that never was, with values typical of its group's records, retracted later."""
    rng, records, domain = site.rng, site.events, site.domain
    anchor = rng.choice(records)
    peers = [r for r in records if r.group == anchor.group]
    pool = peers if len(peers) >= 3 else records
    values = {}
    for spec in domain.fields:
        observed = sorted(r.values[spec.key] for r in pool)
        spread = _spread(records, spec)
        value = observed[int(rng.uniform(0.2, 0.95) * (len(observed) - 1))] + rng.uniform(-0.05, 0.05) * spread
        values[spec.key] = _stored(min(max(value, observed[0]), observed[-1] + 0.05 * spread), spec)
    lat, lon = _position(anchor, (3.0, 40.0), site)
    when = rng.randrange(site.window_start + 60, site.compiled_at - 600)
    ref = _fresh_ref(anchor.source, site, when)
    phantom = RecordReport(ref=ref, source=anchor.source, status=domain.statuses[0], time=when, lat=lat, lon=lon,
                           group=anchor.group, values=values, qualifiers=dict(anchor.qualifiers), place=anchor.place,
                           role="phantom", trap_id=trap_id)
    reason = rng.choice(domain.trap_vocab.retraction_reasons)
    notice = Notice(kind="retraction", target=ref, time=min(site.compiled_at - 60, when + rng.randint(600, 5400)),
                    reason=reason, trap_id=trap_id)
    description = f"phantom report {ref} ({_shown(values, domain)}, {anchor.group}) later retracted: {reason}"
    trap = Trap(trap_id, "retraction", "phantom", ref, description,
                mishandles=[Mishandle(keep_retracted=(ref,)).to_dict()])
    return _Candidate(trap, [phantom], [notice])


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


def _record_conflict(trap_id: str, site: _Site) -> _Candidate | None:
    rng, records, domain = site.rng, site.events, site.domain
    e = _pick_unused(records, site.used, rng)
    if e is None:
        return None
    spec = rng.choice([s for s in domain.fields if s.correctable] or list(domain.fields))
    value, spread = e.values[spec.key], _spread(records, spec)
    thresholds = _canonical_thresholds(site.questions, spec.key, domain.units)
    if thresholds and rng.random() < 0.7:  # aim the wrong value across a threshold that is asked about
        wrong = _across(rng.choice(thresholds), value, spec, rng, reach=0.1 * spread)
    else:
        wrong = value + rng.choice((-1, 1)) * rng.uniform(0.2, 0.45) * spread
    wrong = _stored(wrong, spec)
    if abs(wrong - value) <= max(spec.tolerance[0], spec.tolerance[1] * abs(value)):
        return None  # a careless reader would still be right
    unit = "" if spec.canonical_unit is None else f" {spec.canonical_unit}"
    final, draft = domain.statuses
    if spec.correctable and rng.random() < 0.5:  # the settling report shows the wrong value, a correction fixes it
        index = next(i for i, r in enumerate(site.reports) if r.ref == e.id and r.role == "primary")
        primary = site.reports[index]
        shown = replace(primary, values={**primary.values, spec.key: wrong}, role="conflict", trap_id=trap_id)
        issued = min(site.compiled_at - 30, max(e.time + 30, e.time + rng.randint(900, 7200)))
        notice = Notice(kind="correction", target=e.id, time=issued, trap_id=trap_id, field=spec.key, new_value=value,
                        old_value=wrong, qualifier=e.qualifiers.get(spec.qualifier) if spec.qualifier else None)
        description = (f"{e.id} filed with {spec.label} {plain_number(wrong)}{unit} on its {final} report, then "
                       f"CORRECTED to {plain_number(value)}{unit}")
        trap = Trap(trap_id, "conflicting_value", "correction", e.id, description,
                    mishandles=[Mishandle(ignore_correction_for=(e.id,)).to_dict()])
        return _Candidate(trap, [], [notice], replace_at=(index, shown))
    # A second report under the same ID disagrees with the settling one. A careless reader either takes
    # its value, or counts it as a record of its own.
    source = e.source if rng.random() < 0.5 else rng.choice(_rereporters(site, e.source))
    second = RecordReport(ref=e.id, source=source, status=draft, time=e.time, lat=e.lat, lon=e.lon, group=e.group,
                          values={**e.values, spec.key: wrong}, qualifiers=dict(e.qualifiers), place=e.place,
                          role="conflict", trap_id=trap_id)
    description = (f"{e.id} also reported with {spec.label} {plain_number(wrong)}{unit} ({draft}, {source}) beside the "
                   f"{final} {plain_number(value)}{unit}")
    trap = Trap(trap_id, "conflicting_value", "second_report", e.id, description,
                mishandles=[Mishandle(prefer_preliminary_for=(e.id,)).to_dict(),
                            Mishandle(split=((e.id, source, draft),)).to_dict()])
    return _Candidate(trap, [second], [])


EARTHQUAKE_BUILDERS = Builders(primary_report, _duplicate, _phantom, _conflict)  # the 0.2.0 builders


def builders_for(domain: Domain) -> Builders:
    """The builders that plant traps in a domain's digests. Earthquake domains (whose records are 0.2.0
    Events and Reports) keep the 0.2.0 builders; every other domain's Records are planted by the generic
    builders, whose settling reports carry the domain's first status."""
    if isinstance(domain, Earthquakes):
        return EARTHQUAKE_BUILDERS
    return Builders(functools.partial(record_report, status=domain.statuses[0]), _record_duplicate, _record_phantom,
                    _record_conflict)


def inject_traps(events: list[Event], questions: list[Question], rng: random.Random, *, compiled_at: int,
                 window_hours: int, config: TrapConfig = TrapConfig(),
                 domain: Domain = EARTHQUAKES) -> tuple[list[Report], list[Notice], list[Trap]]:
    """Build the digest's reports and notices: one settling report per record (under the domain's first
    status: REVIEWED for earthquakes), plus the traps. Each trap is the best of up to `config.candidates`
    placements (_rank), guarded off the legacy path."""
    build = builders_for(domain)
    units = domain.units
    site = _Site(domain, events, questions, rng, compiled_at, window_start=compiled_at - window_hours * 3600,
                 id_style=config.id_style, reports=[build.primary(e) for e in events], taken={e.id for e in events},
                 used=set())
    notices: list[Notice] = []
    traps: list[Trap] = []
    key = {q.qid: execute(q.chain, events, units=units) for q in questions}
    time_margin = Context(compiled_at, window_hours).time_margin if config.guarded else None
    top = GUARDED_TOP if config.guarded else TOP
    builders = {"duplicate": build.duplicate, "retraction": build.retraction, "conflict": build.conflict}
    plan = (["duplicate"] * rng.randint(*config.duplicates) + ["retraction"] * config.retractions
            + ["conflict"] * config.conflicts)
    for number, kind in enumerate(plan, start=1):
        trap_id = f"T{number}"
        best: tuple[int, _Candidate] | None = None
        for _ in range(config.candidates):
            candidate = builders[kind](trap_id, site)
            if candidate is None:
                continue
            trial = DigestData(*_apply(site.reports, notices, candidate))
            rank = _rank(_single_readings(candidate.trap, questions, trial, domain, time_margin), key, config.guarded)
            if best is None or rank > best[0]:
                best = (rank, candidate)
            if rank == top:
                break
        if best is None:
            continue
        chosen = best[1]
        site.reports, notices = _apply(site.reports, notices, chosen)
        traps.append(chosen.trap)
        site.used.add(chosen.trap.target)
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
