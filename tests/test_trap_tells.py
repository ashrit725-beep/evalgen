"""Off the legacy path, a planted report is told from a real one only by the signals the reading rules name:
a shared ID or a note naming one, the status, and the notices (traps.py). Every planted report stands in
for a *slot*, a record drawn with the digest but not kept, so its time, place, source, values, precision
and ID are a real record's. These tests pin that, and measure it as a red team would, on a small sample of
every built-in domain in every genre, read from its prompt text: no feature the rules never mention (how
close a report is to another in time, space or value, how often its source files, the precision of its
values, which questions' scopes it lies in and how many answers it changes) tells planted reports from
real ones, and no reader that merges or drops reports by such a feature, or that takes a learned number
off the status rule's counts, without reading a status, an ID, a note or a notice, does better than the
naive reader (or the status rule). The same holds for a plugin domain, which gets the same builders.

The full measurement (1,008 fresh instances per sample, bootstrap intervals by domain and genre, a learned
combination of every feature, the held-out toy plugin and the recorded CAD response) is described under
Traps in docs/DESIGN.md. Before the slots, merging reports within 40 s and keeping the commoner source
closed 38% of the gap between the naive reader and the oracle; before the second round, a close approach
re-reported under its ID gave its object an H that its other approach did not have, planted reports lay
in counting questions' scopes half as often again as real ones, and the status rule's count minus a learned
offset closed 14% of that gap in weather; before the third round, planted reports were a steady share of every
digest (one factor off every naive count closed a tenth of the gap, on held-out domains and genres too), a
corrected temperature or velocity disagreed with its station's or its object's other reports, a planted report
that decided a question led the next by half as much as a real one (and almost never decided one about a
place), and a phantom's form was sent sooner than a real one; on these samples such readers close none of it."""
from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pytest

from evalgen.domains.base import Record
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.domains.invented import INVENTED
from evalgen.domains.neo import NEO, designation_of
from evalgen.domains.weather import WEATHER
from evalgen.genres import get_genre
from evalgen.geo import haversine_km
from evalgen.ops import DigestData, Step, execute, passes, role
from evalgen.pipeline import GenerateConfig, build_instance
from evalgen.questions import Context, compose_new_templates
from evalgen.rules import Mishandle
from evalgen.scoring import is_correct, normalize
from evalgen.seeds import SeedLog
from evalgen.solver import _reading, solve
from evalgen.sources import build_domain_snapshot
from evalgen.traps import (CORRECTION_WEIGHTS, CROSS_REF_WEIGHTS, EARTHQUAKE_BUILDERS, NOTICE_DELAY, RETRACTION_WEIGHTS,
                           SAME_ID_WEIGHTS, TrapConfig, builders_for, draw_plan)

DOMAINS = (EARTHQUAKES, WEATHER, NEO, INVENTED)
GENRES = ("classic", "news", "email", "chat", "table", "form", "radio")
SEED, INSTANCES = 61, 28  # four instances of each domain in each genre
LOW, HIGH = 0.3, 0.7  # a feature whose AUC lies outside this band on the sample tells planted reports apart
PLANTED = ("dup_same", "dup_xref", "second", "phantom", "corr")
REMOVABLE = ("dup_same", "dup_xref", "second", "phantom")  # the planted reports a reader must set aside
SAME_ID = ("dup_same", "second")  # filed under their record's own ID


def generic(record, domain) -> Record:
    """A snapshot record as a generic Record (an earthquake Event's magnitude type is its qualifier)."""
    if isinstance(record, Record):
        return record
    return Record(id=record.id, time=record.time, lat=record.lat, lon=record.lon, group=record.group,
                  source=record.source, values=dict(record.values), qualifiers={"mag_type": record.mag_type},
                  place=record.place)


def stored_places(value: float, decimals: int) -> int:
    """The decimal places a stored value is written to: the fewest, up to `decimals`, that hold it."""
    return next((places for places in range(decimals) if round(value, places) == value), decimals)


def auc(planted: list[float], real: list[float]) -> float:
    """P(a planted value > a real one) + 1/2 P(tie): the Mann-Whitney AUC."""
    ranked = sorted([(v, 1) for v in planted] + [(v, 0) for v in real])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mean_rank = (i + j + 1) / 2  # ranks i+1 .. j
        rank_sum += mean_rank * sum(flag for _, flag in ranked[i:j])
        i = j
    n, m = len(planted), len(real)
    return (rank_sum - n * (n + 1) / 2) / (n * m)


def digests_of(domain, seed: int, n: int, genres=GENRES) -> list[dict]:
    """The first `n` instances of a run about `domain` off the legacy path, the i-th in genres[i % len], built
    as the pipeline builds them, with the snapshot and the reports its prompt prints as the genre's parser
    reads them back, each labelled with what was planted (labels are read only to sort reports for the
    measurement, never by a reader)."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    cfg = GenerateConfig(seed=seed, variants=0)
    out = []
    for i in range(n):
        genre = genres[i % len(genres)]
        genre = get_genre(genre) if isinstance(genre, str) else genre
        inst = build_instance(i, snapshot, new, log, cfg, ctx, used, usage, domain=domain, genre_plan=(genre,))
        bound = domain.bind(inst.surface)
        forms = {t.trap_id: t.form for t in inst.traps}
        labels = {}
        for r in map(bound.to_generic, inst.reports):
            label = {"primary": "real", "phantom": "phantom"}.get(r.role)
            if r.role == "duplicate":
                label = "dup_same" if forms[r.trap_id] == "same_id" else "dup_xref"
            elif r.role == "conflict":
                label = "corr" if r.status == bound.statuses[0] else "second"
            labels[(r.ref.casefold(), r.source.casefold(), r.status.casefold())] = label
        data, reader, units = _reading(inst.renderings[0].prompt, inst.domain, genre.name)
        reports = [reader.to_generic(r) for r in data.reports]
        out.append({"domain": domain, "bound": bound, "inst": inst, "snapshot": snapshot, "data": data, "units": units,
                    "genre": genre.name, "reports": reports,
                    "labels": [labels[(r.ref.casefold(), r.source.casefold(), r.status.casefold())] for r in reports]})
    return out


@pytest.fixture(scope="module")
def sample():
    return [digest for domain in DOMAINS for digest in digests_of(domain, SEED, INSTANCES)]


def rows(sample, feature, kinds=PLANTED):
    """(planted values, real values) of `feature`(report, others, digest) over the sample."""
    planted, real = [], []
    for digest in sample:
        reports = digest["reports"]
        for i, (r, label) in enumerate(zip(reports, digest["labels"])):
            value = feature(r, reports[:i] + reports[i + 1:], digest)
            if value is None:
                continue
            if label == "real":
                real.append(value)
            elif label in kinds:
                planted.append(value)
    return planted, real


def slots_of(digest) -> dict:
    """The snapshot's records that the digest does not keep (its slots, and the records of no slot), by ID."""
    bound, kept = digest["bound"], set(digest["inst"].event_ids)
    return {r.id: generic(r, bound) for r in map(bound.dress, digest["snapshot"].events) if r.id not in kept}


def naive_answers(digest, keep) -> list:
    """The naive reader's answers (every report it keeps is a record of its own; notices are ignored), each as
    the scorer normalizes it, over the reports of the digest numbered `keep`."""
    data, units, bound = digest["data"], digest["units"], digest["bound"]
    native = [data.reports[i] for i in keep]
    return [normalize(q.answer_type, execute(q.chain, DigestData(native, []), skip=("dedupe",), units=units,
                                             resolver=bound.resolve)) for q in digest["inst"].questions]


# ----------------------------------------------------------------------- what a planted report is


def test_the_sample_reads_every_built_in_domain_in_every_genre(sample):
    assert Counter((d["domain"].name, d["genre"]) for d in sample) == {(domain.name, genre): 4 for domain in DOMAINS
                                                                       for genre in GENRES}
    assert all("real" in d["labels"] and set(d["labels"]) & set(REMOVABLE) for d in sample)
    assert {label for d in sample for label in d["labels"]} == {"real", *PLANTED}


def test_a_planted_report_stands_in_for_a_record_drawn_but_not_kept(sample):
    """A phantom is a record of the snapshot that the digest does not keep, filed as it is; a re-report
    under an ID of its own is such a record with a note; a re-report or a second report under its record's
    ID gives such a record's time, place, source and values (a close approach keeps its object's H)."""
    counts = Counter()
    for digest in sample:
        bound, inst = digest["bound"], digest["inst"]
        spare = slots_of(digest)
        by_place = {(r.time, r.source, r.lat, r.lon, r.place): r for r in spare.values()}
        for report in map(bound.to_generic, inst.reports):
            if report.role == "phantom" or (report.role == "duplicate" and report.same_as is not None):
                slot = spare[report.ref]  # its own ID is a real record's, which the digest does not keep
            elif report.role in ("duplicate", "conflict") and report.status == bound.statuses[1]:
                slot = by_place[(report.time, report.source, report.lat, report.lon, report.place)]
            else:
                continue
            counts[report.role] += 1
            assert (report.time, report.source, report.group, report.lat, report.lon, report.place) == (
                slot.time, slot.source, slot.group, slot.lat, slot.lon, slot.place)
            assert report.qualifiers == slot.qualifiers
            same = {k: v for k, v in report.values.items() if not (bound is NEO and k == "h" and report.ref != slot.id)}
            assert same == {k: slot.values[k] for k in same}
    assert counts["phantom"] >= len(sample) // 2 and counts["duplicate"] >= 2 * len(sample) and counts["conflict"] > 0


def test_a_corrected_value_is_a_real_value_of_the_group_not_one_just_past_a_threshold(sample):
    """The wrong value that a correction fixes is the value of a record of the group that the digest does
    not keep (written to the corrected record's places), not one placed just past an asked threshold. (A
    field whose values a record shares with its near twins, Domain.twin_fields, is corrected to a value that
    agrees with them instead: see the next test.)"""
    taken, corrections = 0, 0
    for digest in sample:
        bound, inst = digest["bound"], digest["inst"]
        for notice in map(bound.to_generic, inst.notices):
            if notice.kind != "correction" or notice.field in bound.twin_fields:
                continue
            corrections += 1
            record = next(generic(r, bound) for r in map(bound.dress, digest["snapshot"].events) if r.id == notice.target)
            spec = bound.field(notice.field)
            places = stored_places(notice.new_value, spec.decimals)
            spare = [r for r in slots_of(digest).values() if r.group == record.group]
            taken += any(round(r.values[spec.key], places) == notice.old_value for r in spare)
    assert corrections >= 20 and taken >= 0.8 * corrections, (taken, corrections)


def test_a_corrected_value_agrees_with_its_records_twins(sample):
    """Where a record's values are tied to its near twins' (Domain.twin_fields: a station's temperature to its
    observations hours away, an object's speed past the Moon to its speed past Earth), a correction shows a
    value that agrees with the record's nearest twin as a real value agrees with its own (traps._twin_values),
    not a value of any slot of its group: a corrected temperature, often another station's, jumped a median
    4.5 °C from its station's nearest other report where real ones jump 1 °C (an AUC of 0.84 to 0.90), and a
    corrected velocity left its object's other approach by more than any real one does (0.75 to 0.81). An
    object passes the Moon within a km/s of its speed past Earth, and so does every corrected approach."""
    weather = [d for d in sample if d["bound"] is WEATHER] + digests_of(WEATHER, 17, 32, genres=("classic",))

    def jump(r, others, digest):  # to its station's nearest other report within 3 h, per root hour
        at = [o for o in others if o.place == r.place and o.ref.casefold() != r.ref.casefold()
              and abs(o.time - r.time) <= 3 * 3600]
        if not at:
            return None
        o = min(at, key=lambda o: abs(o.time - r.time))
        return abs(r.values["temp"] - o.values["temp"]) / math.sqrt(abs(o.time - r.time) / 3600 + 0.25)

    planted, real = rows(weather, jump, ("corr",))
    assert len(planted) >= 20 and 0.3 <= auc(planted, real) <= 0.7, (len(planted), auc(planted, real))
    median = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    assert median(planted) <= 1.6 * median(real), (median(planted), median(real))
    neo = [d for d in sample if d["bound"] is NEO] + digests_of(NEO, 17, 32, genres=("classic",))
    siblings = 0
    for digest in neo:
        for r, label in zip(digest["reports"], digest["labels"]):
            if label not in ("real", "corr"):
                continue
            mates = [o for o in digest["reports"] if designation_of(o.ref) == designation_of(r.ref)
                     and o.ref.casefold() != r.ref.casefold()]
            for o in mates:
                siblings += label == "corr"
                assert abs(r.values["v_rel"] - o.values["v_rel"]) <= 1.0 + 0.01, (label, r, o)
    assert siblings > 0


def test_planted_values_lie_where_real_ones_do(sample):
    """No planted value is one no real record has (TB2: a miss distance of 0 km, a velocity of 0.29 km/s
    where no record is slower than 1.3): every value of a phantom or a re-report is a value of a record of
    the snapshot in its group, and a field that no record has at 0 has no planted 0 either. (The stored
    reports: earthquakes in the classic genre print the 0.2.0 digest, at 0.2.0 precision.)"""
    for digest in sample:
        bound = digest["bound"]
        pool: dict = {}
        for r in map(bound.dress, digest["snapshot"].events):
            for key, value in r.values.items():
                pool.setdefault((r.group, key), set()).add(value)
        lows = {f.key: min(r.values[f.key] for r in digest["snapshot"].events) for f in bound.fields}
        for r in map(bound.to_generic, digest["inst"].reports):
            for key, value in r.values.items():
                if r.role == "phantom" or (r.role == "duplicate" and r.same_as is not None):
                    assert value in pool[(r.group, key)], (r.role, r.ref, key, value)
                if r.role != "primary" and lows[key] > 0:
                    assert value > 0, (r.role, r.ref, key)


# ------------------------------------------------------------------ features the rules never mention


def test_proximity_in_time_and_space_tells_nothing(sample):
    def near(r, others, digest):  # another report within 40 s (and 8.5 km, where reports have a position)
        located = digest["bound"].spatial is not None
        return float(any(abs(o.time - r.time) <= 40 and (not located or haversine_km(o.lat, o.lon, r.lat, r.lon) <= 8.5)
                         for o in others))

    def gap(r, others, digest):  # seconds to the nearest other report of its group
        return min((abs(o.time - r.time) for o in others if o.group == r.group), default=None)

    for feature in (near, gap):
        for kinds in (("dup_same", "dup_xref"), ("second",), ("phantom",)):
            planted, real = rows(sample, feature, kinds)
            assert LOW <= auc(planted, real) <= HIGH, (feature.__name__, kinds, auc(planted, real))


def twin_key(r, digest):
    """What a report's near twins share (Domain.twin_key, on the printed report): a quake's region, a station,
    an object's designation, an agency's depot."""
    name = digest["domain"].name
    return {"earthquakes": r.group, "neo": designation_of(r.ref), "invented": (r.group, r.source)}.get(
        name, (r.group, r.place))


def test_a_report_under_its_records_id_has_its_slots_neighbours(sample):
    """Set aside the reports that share its ID (they are one record, rule 1), and a report filed under its
    record's own ID has around it the neighbours its slot has, as a real record has its own: its record is
    not its slot's nearest one, nor a twin of it (traps._record_for). It was, before the second round, and
    the gap to the nearest of its twins (reports that share its twin key) was then a tell: an AUC of 0.57
    to 0.66, 0.6 in all."""
    def gap(r, others, digest):  # to the nearest report that shares its twin key, with another ID
        key = twin_key(r, digest)
        return min((abs(o.time - r.time) for o in others if twin_key(o, digest) == key
                    and o.ref.casefold() != r.ref.casefold()), default=10.0 ** 9)

    planted, real = rows(sample, gap, SAME_ID)
    assert 0.42 <= auc(planted, real) <= 0.58, auc(planted, real)
    # Its record is the real record of its group nearest to it in time only where no other could carry the trap.
    nearest, same_id = 0, 0
    for digest in sample:
        real_reports = [r for r, label in zip(digest["reports"], digest["labels"]) if label == "real"]
        for r, label in zip(digest["reports"], digest["labels"]):
            if label in SAME_ID:
                same_id += 1
                mates = [o for o in real_reports if o.group == r.group]
                nearest += min(mates, key=lambda o: (abs(o.time - r.time), o.time)).ref.casefold() == r.ref.casefold()
    assert nearest < 0.2 * same_id, (nearest, same_id)


def test_sources_tell_nothing(sample):
    def files(r, others, digest):  # how many other reports its source files in the digest
        return float(sum(o.source == r.source for o in others))

    def alone(r, others, digest):  # its source files no other report of its group
        return float(not any(o.source == r.source and o.group == r.group for o in others))

    def relay(r, others, digest):  # its source is one of the domain's public re-reporting names
        vocab = {s.casefold() for s in digest["bound"].trap_vocab.rereporting_sources}
        return float(r.source.casefold() in vocab) if digest["bound"].surface is None else None

    for feature in (files, alone, relay):
        planted, real = rows(sample, feature, ("dup_same", "dup_xref", "second"))
        assert LOW <= auc(planted, real) <= HIGH, (feature.__name__, auc(planted, real))


def test_every_source_that_re_reports_files_records_of_its_own():
    """The synthetic worlds off the legacy path: the networks, feeds, catalogues and agencies that re-report
    others' records file records of their own (T2: relay names alone closed 38% of the gap in weather)."""
    compiled_at = int(datetime(2026, 6, 1, 6, tzinfo=timezone.utc).timestamp())
    for domain in DOMAINS:
        records = domain.synthetic(random.Random(5), 2000, compiled_at, domain.synthetic_window_hours)
        filed = Counter(r.source for r in records)
        relays = set(domain.trap_vocab.rereporting_sources) - ({"ew"} if domain is EARTHQUAKES else set())
        assert all(filed[source] >= 20 for source in relays), (domain.name, filed)


def test_precision_and_ranges_tell_nothing(sample):
    def places(key):
        def feature(r, others, digest):
            spec = next((s for s in digest["bound"].fields if s.key == key), None)
            return None if spec is None else float(stored_places(r.values[key], spec.decimals))
        feature.__name__ = f"places_{key}"
        return feature

    for key in ("mag", "depth", "temp", "wind", "pressure", "dist", "v_rel", "h", "f1", "f2"):
        planted, real = rows(sample, places(key))
        if len(planted) >= 10:
            assert LOW <= auc(planted, real) <= HIGH, (key, auc(planted, real))
    # A numbered designation, or one older than two years, is never printed with an H to three places
    # (_draw_object); a planted one never is either, and its number lies in the range real ones come from.
    for digest in sample:
        if digest["bound"] is not NEO:
            continue
        year = datetime.fromtimestamp(digest["inst"].compiled_at, tz=timezone.utc).year
        for r, label in zip(digest["reports"], digest["labels"]):
            designation = designation_of(r.ref)
            body = designation.lstrip("ACDIPX") if not designation.isdigit() else designation
            found = re.match(r"(\d{4})[A-Y]", body)
            old = body.isdigit() or (found is not None and int(found.group(1)) < year - 2)
            assert not (old and stored_places(r.values["h"], 3) == 3), (label, r.ref, r.values["h"])
            if body.isdigit():
                assert 3000 <= int(body) < 650_000, (label, r.ref)


def test_closeness_in_value_tells_nothing(sample):
    def nearest(r, others, digest):  # the largest per-field gap to the closest report of its group, in spreads
        fields = digest["bound"].fields
        spread = {f.key: max(max(o.values[f.key] for o in digest["reports"])
                             - min(o.values[f.key] for o in digest["reports"]), 1e-9) for f in fields}
        gaps = [max(abs(o.values[f.key] - r.values[f.key]) / spread[f.key] for f in fields)
                for o in others if o.group == r.group]
        return min(gaps, default=None)

    for kinds in (("dup_same", "dup_xref"), ("phantom",), ("second",)):
        planted, real = rows(sample, nearest, kinds)
        assert LOW <= auc(planted, real) <= HIGH, (kinds, auc(planted, real))


# ---------------------------------------------------------------- the questions a planted report affects


def scopes_of(r, digest, counting: bool | None = None) -> float:
    """How many of the questions' scopes (all of a question's filters pass) a report lies in: of the counting
    questions (whose answer is a count), of the others, or of all (None)."""
    units, total = digest["units"], 0
    for q in digest["inst"].questions:
        if counting is not None and any(Step(s).kind == "count" for s in q.chain) != counting:
            continue
        total += all(passes(s, r, units) for s in q.chain if role(s["op"]) == "filter")
    return float(total)


def test_planted_reports_lie_in_as_many_question_scopes_as_real_ones(sample):
    """In all, and in the counting questions' scopes and in the others' apart: a placement chosen to change
    an answer lay in counting questions' scopes half as often again as a real record (traps._fit), which
    matching only the number of scopes left as it was (an AUC of about 0.6 for those scopes)."""
    for counting in (None, True, False):
        for kinds in (PLANTED, ("dup_same", "dup_xref"), ("phantom",)):
            planted, real = rows(sample, lambda r, others, digest: scopes_of(r, digest, counting), kinds)
            assert 0.44 <= auc(planted, real) <= 0.56, (counting, kinds, auc(planted, real))


def test_removing_a_planted_report_changes_as_many_answers_as_removing_a_real_one(sample):
    """How many of the naive reader's answers change when a report alone is set aside: a planted report that
    lay in more counting questions' scopes than a real one changed more of them (an AUC of 0.60 to 0.69 before
    the second round), and a reader could have dropped the reports that matter most."""
    planted, real = [], []
    for digest in sample:
        everything = list(range(len(digest["reports"])))
        base = naive_answers(digest, everything)
        for i, label in enumerate(digest["labels"]):
            changed = sum(a != b for a, b in zip(base, naive_answers(digest, everything[:i] + everything[i + 1:])))
            (real if label == "real" else planted if label in REMOVABLE else []).append(changed)
    assert 0.4 <= auc(planted, real) <= 0.6, auc(planted, real)


LEAD_TERMINALS = ("argmax", "max", "min", "time_of_max", "rank", "nearest_to", "nearest_distance")


def deciders(digest) -> list[tuple[str, int, float]]:
    """(the kind of question, the report that decides it, how far it leads the next) for each question whose
    naive answer one report decides (every report a record of its own): the largest (smallest: min) value of
    its field in scope, the k-th, or the report nearest its place; the lead in shares of the field's spread
    over the digest's reports, or in km."""
    reports, units = digest["reports"], digest["units"]
    out = []
    for q in digest["inst"].questions:
        term = Step(next(s for s in q.chain if role(s["op"]) == "terminal"), units)
        if term.kind not in LEAD_TERMINALS:
            continue
        inside = [i for i, r in enumerate(reports)
                  if all(passes(s, r, units) for s in q.chain if role(s["op"]) == "filter")]
        if len(inside) < 2:
            continue
        if term.kind in ("nearest_to", "nearest_distance"):
            d = sorted((haversine_km(term.raw["lat"], term.raw["lon"], reports[i].lat, reports[i].lon), i)
                       for i in inside)
            out.append((term.kind, d[0][1], d[1][0] - d[0][0]))
            continue
        key = term.field
        values = [r.values[key] for r in reports]
        spread = max(max(values) - min(values), 1e-9)
        ordered = sorted(inside, key=lambda i: (reports[i].values[key] if term.kind == "min" else -reports[i].values[key], i))
        k = term["k"] - 1 if term.kind == "rank" else 0
        if len(ordered) > k + 1:
            out.append((term.kind, ordered[k], abs(reports[ordered[k]].values[key] - reports[ordered[k + 1]].values[key])
                        / spread))
    return out


def test_a_report_that_decides_a_question_leads_the_next_as_a_real_one_does(sample):
    """A planted report that changes the record a question names, or its value, must pass the real record
    that holds it; placed where it changes an answer, it passed it by little (a station's next observation
    passes its last one by a degree), and the report a question about a place was drawn near was a real record,
    so that real reports decided those questions, by tens or hundreds of km, and planted ones almost never.
    Each trap now draws how far a real record that decides a question leads the next one, of each kind of
    question, and prefers a placement whose planted report, where it decides one, leads within a factor of two
    of it (traps._Leads); and a question about a place is asked about a place near a report of the digest that
    then decides it, planted or real (questions.reanchor_nearest, and the test that follows). (Before: an AUC
    of 0.27 in weather for how far a report that decides a question leads the next, and of 0.18 to 0.23 for
    the report nearest a place in earthquakes.)"""
    planted, real, nearest = [], [], Counter()
    for digest in sample:
        for kind, i, lead in deciders(digest):
            label = digest["labels"][i]
            if kind in ("nearest_to", "nearest_distance"):
                nearest[label in REMOVABLE] += 1
                continue
            (planted if label in REMOVABLE else real if label == "real" else []).append(lead)
    assert len(planted) >= 40 and 0.38 <= auc(planted, real) <= 0.62, (len(planted), auc(planted, real))
    assert nearest[True] > 0 and nearest[False] > 0, nearest  # the report nearest a place is sometimes planted


@pytest.fixture(scope="module")
def places(sample):
    """The sample's digests of the domains whose records have a position, and those of three more seeds."""
    return [digest for digest in sample if digest["domain"].spatial is not None] + [
        digest for domain in DOMAINS if domain.spatial is not None for seed in (SEED + 1, SEED + 2, SEED + 3)
        for digest in digests_of(domain, seed, INSTANCES)]


def test_a_place_lies_near_the_report_that_decides_it_planted_or_real(places):
    """A question about the record nearest to a place is asked about a place drawn near a report of the digest
    that then decides it, a record's or a planted one, each kind about as often as its share of the reports the
    question asks about, with a label drawn for that report and no bound on how far the key's nearest record
    lies (questions.reanchor_nearest). So the report nearest to a place lies in the band of a drawn place, and
    is named by its label (where a label names a source), as often planted as real; and planted reports decide
    such questions, and lead the next, more nearly as real ones do. (Before: a place drawn near any report kept
    the label drawn for a record, and was decided by a report in that band, and named by its label, less often
    when the report was planted, with AUCs of 0.28 to 0.38 in weather; and the key's nearest record had to lie
    within Spatial.nearest_max_km, which bounded how far a planted report that decides such a question leads
    the next and kept most of them from deciding one: in weather, an AUC of 0.38 for that lead, and planted
    reports deciding a fifth as many of these questions as their share of the reports asked about, where they
    now decide half as many, and in earthquakes four fifths.)"""
    band, named, leads, decided, asked = defaultdict(list), defaultdict(list), defaultdict(list), Counter(), Counter()
    for digest in places:
        spatial = digest["domain"].spatial
        reports, units, labels = digest["reports"], digest["units"], digest["labels"]
        lo, hi = spatial.point_offset_km
        for q in digest["inst"].questions:
            term = Step(next(s for s in q.chain if role(s["op"]) == "terminal"), units)
            if term.kind not in ("nearest_to", "nearest_distance"):
                continue
            inside = [i for i, r in enumerate(reports)
                      if all(passes(s, r, units) for s in q.chain if role(s["op"]) == "filter")]
            d = sorted((haversine_km(term.raw["lat"], term.raw["lon"], reports[i].lat, reports[i].lon), i)
                       for i in inside)
            i = d[0][1]
            planted = labels[i] in REMOVABLE
            band[planted].append(lo - 0.6 <= d[0][0] <= hi + 0.6)  # coordinates are printed to 0.001°
            named[planted].append(bool(re.search(rf"\b{re.escape(reports[i].source.upper())}\b",
                                                 str(term.raw["label"]).upper())))
            leads[planted].append(d[1][0] - d[0][0] if len(d) > 1 else math.inf)
            decided[planted] += 1
            for j in inside:
                asked[labels[j] in REMOVABLE] += 1
    rate = {k: {p: sum(v[p]) / len(v[p]) for p in (True, False)} for k, v in (("band", band), ("named", named))}
    assert all(abs(r[True] - r[False]) <= 0.1 for r in rate.values()) and rate["band"][False] >= 0.9, rate
    share = decided[True] / sum(decided.values()) / (asked[True] / sum(asked.values()))
    assert decided[True] >= 10 and share >= 0.4, (decided, asked, share)  # (0.57 on this sample)
    assert 0.3 <= auc(leads[True], leads[False]) <= 0.7, auc(leads[True], leads[False])  # (0.33; before, 0.22)


def test_a_notice_follows_its_report_later_than_any_genre_sends_it(sample, monkeypatch):
    """A notice about a report is issued 92 minutes to 3 hours after the report's time (traps.NOTICE_DELAY;
    a correction of a record that close to the compilation just before it), later than a form is sent, so no
    genre sends a planted report sooner because a notice about it follows: a form, sent before any notice
    about its record, was sent a median 21 minutes after a phantom's time when retractions came 10 to 90
    minutes after it, against 44 after a real record's (an AUC of 0.28 to 0.29)."""
    for digest in sample:
        inst = digest["inst"]
        final = digest["bound"].statuses[0]
        times = {r.ref: r.time for r in map(digest["bound"].to_generic, inst.reports) if r.status == final}
        for notice in inst.notices:
            gap = notice.time - times[notice.target]
            assert NOTICE_DELAY[0] <= gap <= NOTICE_DELAY[1] or (
                notice.kind == "correction" and notice.time == inst.compiled_at - 30), (notice, gap)
    from evalgen.genres import form

    calls = []
    original = form._sent_time

    def recording(rng, when, deadline):
        calls.append((when, deadline))
        return original(rng, when, deadline)

    monkeypatch.setattr(form, "_sent_time", recording)
    capped = Counter()
    for domain in DOMAINS:  # (a phantom lies at least 92 minutes before the compilation, as a report at its time)
        phantoms = {r.time for digest in digests_of(domain, 23, 12, genres=("form",))
                    for r in digest["inst"].reports if r.role == "phantom"}
        for when, deadline in calls:
            if when in phantoms:
                capped[deadline - when <= 90 * 60 + 59] += 1
        calls.clear()
    assert capped[False] > 0 and capped[True] == 0, capped


def test_the_status_rule_count_is_the_likeliest_right_count(sample):
    """A reader who drops every report of the second status (the status rule) and counts what is left counts
    every phantom in a question's scope, as a phantom is settled as it is. With a retraction in every digest
    and its phantom in counting questions' scopes as often as a load-bearing trap's, that count was one too
    many more often than it was right, and taking one off it (an offset learned per template) gained 9% of
    the gap between the naive reader and the oracle beyond the status rule, 14% in weather. A digest now
    holds none, one or two retractions, none the likeliest (traps.RETRACTION_WEIGHTS), drawn apart from its
    other traps (traps.draw_plan): the status count is right more often than it is off by any number, and a
    learned offset gains nothing."""
    offsets, rows_ = Counter(), []
    for digest in sample:
        reports, draft = digest["reports"], digest["bound"].statuses[1]
        settled = [i for i, r in enumerate(reports) if r.status != draft]
        status, naive = naive_answers(digest, settled), naive_answers(digest, list(range(len(reports))))
        inst = digest["inst"]
        for q, counted, careless in zip(inst.questions, status, naive):
            key = normalize(q.answer_type, inst.answer_key[q.qid])
            is_count = any(Step(s).kind == "count" for s in q.chain)
            if is_count:
                offsets[counted - key] += 1
            rows_.append({"half": int(inst.instance_id[-3:]) % 2, "count": is_count, "tpl": (digest["domain"].name,
                          q.template_id), "status": counted, "naive": careless, "key": key})
    assert offsets[0] == max(offsets.values()) and offsets[0] > 1.5 * offsets[1], offsets
    retractions = Counter(sum(t.kind == "retraction" for t in d["inst"].traps) for d in sample)
    assert RETRACTION_WEIGHTS[0] == max(RETRACTION_WEIGHTS) and max(retractions) <= len(RETRACTION_WEIGHTS) - 1
    # An offset per domain and template, learned on half the instances, applied to the status counts of the
    # others, gains less than a twentieth of the naive reader's gap to the oracle beyond the status rule.
    learned: dict = defaultdict(Counter)
    for row in rows_:
        if row["half"] == 0 and row["count"]:
            learned[row["tpl"]][row["status"] - row["key"]] += 1
    test = [row for row in rows_ if row["half"] == 1]

    def shifted(row):  # the status rule's answer, a count less the offset learned for its template
        offset = learned[row["tpl"]].most_common(1)[0][0] if row["count"] and learned[row["tpl"]] else 0
        return row["status"] - offset if row["count"] else row["status"]

    status_right = sum(row["status"] == row["key"] for row in test)
    shifted_right = sum(shifted(row) == row["key"] for row in test)
    naive_right = sum(row["naive"] == row["key"] for row in test)
    assert (shifted_right - status_right) / (len(test) - naive_right) < 0.05, (shifted_right, status_right)


# ------------------------------------------------------------------------------- what IDs imply


def test_a_close_approach_is_re_reported_under_its_id_only_where_its_object_has_no_other(sample):
    """An approach ID names its object and numbers its approaches in time order, and every approach of an
    object prints the object's one H. A report filed under an approach's own ID beside another approach of
    the object would have to give that H, the very H of its record, a copy no two real reports of a body
    share, or another, which no real approach does (an AUC of 0 for the sibling's H before the second round),
    and it could fall out of the order of the approaches' numbers. So it is filed only for an approach whose
    object makes no other approach in the digest or its slots (traps._with_siblings, Domain.id_place), with
    the object's H as another catalogue fits it. Every object of a digest then has one H and its approaches
    lie in the order of their numbers, and no planted report copies the H of another report of its body."""
    def copied(r, others, digest):  # another report of its group gives exactly its H
        return float(any(o.group == r.group and o.values["h"] == r.values["h"] for o in others))

    neo = [d for d in sample if d["bound"] is NEO] + digests_of(NEO, 7, 28, genres=("classic",))
    same_id = 0
    for digest in neo:
        objects: dict = defaultdict(list)
        for r, label in zip(digest["reports"], digest["labels"]):
            n = int(r.ref.casefold().rpartition("-ca")[2])
            objects[designation_of(r.ref)].append((n, r.time, r.values["h"], label))
        for reports in objects.values():
            numbers = {n for n, _, _, _ in reports}
            if any(label in SAME_ID for _, _, _, label in reports):
                same_id += 1
                assert len(numbers) == 1, reports  # its object makes no other approach in the digest
            else:
                assert len({h for _, _, h, _ in reports}) == 1, reports
            for (n1, t1, _, _), (n2, t2, _, _) in itertools.combinations(reports, 2):
                assert n1 == n2 or (n1 < n2) == (t1 < t2), reports
    assert same_id >= 40
    planted, real = rows(neo, copied, PLANTED)
    assert LOW <= auc(planted, real) <= HIGH, auc(planted, real)


def test_a_digest_prints_as_many_reports_however_many_are_planted(sample):
    """A digest keeps one record less for each report its traps will add (pipeline.sample_with_slots,
    traps.TrapPlan.lines), so its reports, records and planted ones together, number from events_min to
    events_max however many are planted, and neither its size nor the reports beside a planted one tell how
    many are: with a phantom count that varied on top of the records, a phantom lay in a larger digest than a
    real record did (an AUC of 0.64 to 0.66 for the reports beside it), and a re-report count on top of them
    would do the same for re-reports."""
    cfg = GenerateConfig(seed=SEED)
    planted_reports = Counter()
    for digest in sample:
        assert cfg.events_min <= len(digest["reports"]) <= cfg.events_max, len(digest["reports"])
        planted_reports[sum(label in REMOVABLE for label in digest["labels"])] += 1
    assert len(planted_reports) >= 6 and max(planted_reports) >= 8, planted_reports
    for kinds in (("phantom",), ("dup_same", "dup_xref"), ("second",)):
        planted, real = rows(sample, lambda r, others, digest: float(len(others)), kinds)
        assert 0.4 <= auc(planted, real) <= 0.6, (kinds, auc(planted, real))


def test_a_weather_re_report_is_not_pinned_to_its_observations_station(sample):
    """An observation ID is an opaque serial: it names no station, so a re-report filed under it stands where
    its slot stands, in its observation's region, and not always at its observation's station (when IDs began
    with the station's identifier it had to, and could never be the only report there, as a real observation
    often is: AUCs of about 0.37 for the other observations at its station, and 0.40 for the time to the
    nearest one)."""
    weather = [d for d in sample if d["bound"] is WEATHER]
    moved = 0
    for digest in weather:
        settled = {r.ref: r for r, label in zip(digest["reports"], digest["labels"]) if label == "real"}
        for r, label in zip(digest["reports"], digest["labels"]):
            if label in SAME_ID:
                assert r.group == settled[r.ref].group
                moved += r.place != settled[r.ref].place
    assert moved > 0

    def at_station(r, others, digest):  # other reports at its station, with or without its own ID
        return float(sum(o.place == r.place for o in others))

    def at_station_other_id(r, others, digest):
        return float(sum(o.place == r.place and o.ref.casefold() != r.ref.casefold() for o in others))

    def gap_at_station(r, others, digest):
        return min((abs(o.time - r.time) for o in others if o.place == r.place), default=10.0 ** 9)

    def gap_at_station_other_id(r, others, digest):
        return min((abs(o.time - r.time) for o in others if o.place == r.place
                    and o.ref.casefold() != r.ref.casefold()), default=10.0 ** 9)

    for feature, band in ((at_station, 0.15), (at_station_other_id, 0.12), (gap_at_station, 0.12),
                          (gap_at_station_other_id, 0.12)):
        planted, real = rows(weather, feature, SAME_ID)
        assert abs(auc(planted, real) - 0.5) <= band, (feature.__name__, auc(planted, real))


# ----------------------------------------------------------------------------- how traps are planned


def test_a_digest_holds_the_traps_its_plan_draws(sample):
    """A digest's traps are planned before its records (traps.draw_plan), each kind on its own: none to two
    retractions, none or one to three re-reports under IDs of their own, none or one to eight reports under
    their records' own IDs (a fifth of them second reports), none to three corrections. Every planned trap
    finds a place (a re-report may share its record with another re-report where no record is left free of
    traps), and no trap follows another: a planned re-report or conflict that changed no answer was followed by
    one more of its kind, which made how many reports a digest plants depend on its questions. About as many
    traps change an answer as before (three a digest, against 3.1 to 3.5)."""
    load_bearing, forms = [], Counter()
    for digest in sample:
        traps = digest["inst"].traps
        kinds = Counter(t.form for t in traps)
        forms.update(kinds)
        assert kinds["phantom"] <= len(RETRACTION_WEIGHTS) - 1 and kinds["cross_ref"] <= len(CROSS_REF_WEIGHTS) - 1
        assert kinds["same_id"] + kinds["second_report"] <= len(SAME_ID_WEIGHTS) - 1
        assert kinds["correction"] <= len(CORRECTION_WEIGHTS) - 1
        assert [t.trap_id for t in traps] == [f"T{i}" for i in range(1, len(traps) + 1)]  # every planned trap placed
        load_bearing.append(sum(1 for t in traps if t.affected))
    assert set(forms) == {"same_id", "cross_ref", "phantom", "second_report", "correction"}
    assert sum(load_bearing) / len(load_bearing) >= 2.7, sum(load_bearing) / len(load_bearing)


def test_how_many_reports_a_digest_plants_varies_widely_and_each_kind_on_its_own():
    """Planted reports were a steady share of a digest (two or three re-reports, a conflict and a retraction
    or more in every one), and lay in questions' scopes as real records do, so that a reader who took that
    share off every count, round(0.84 x the naive count), reading no status, ID, note or notice, was right on
    about 20 points more of the counting questions than the naive reader, and closed a tenth of the gap between
    the naive reader and the oracle, as much on held-out domains and genres. Now how many a digest plants
    varies widely (none in about one digest in sixteen, up to thirteen), each kind is drawn on its own, and for
    any share of a digest's records that a counting question's scope holds, the planted reports in it are none
    about as often as any other number: within 5 points for the naive reader (every planted report; none of it
    below a scope of a third of the digest), 3 for a reader who merges reports by ID alone (the
    cross-referenced re-reports and the phantoms), and exactly for the status rule (the phantoms), so that no
    number or share taken off a count does much better than the count itself (traps.draw_plan)."""
    rng = random.Random(1)
    plans = [draw_plan(rng) for _ in range(20000)]
    lines = Counter(p.lines for p in plans)
    assert 0.055 <= lines[0] / len(plans) <= 0.075 and max(lines) == 13
    assert 4.7 <= sum(p.lines for p in plans) / len(plans) <= 5.3
    for a, b in itertools.combinations(("retractions", "cross_refs", "same_ids", "corrections"), 2):
        x, y = [getattr(p, a) for p in plans], [getattr(p, b) for p in plans]
        assert abs(_correlation(x, y)) < 0.03, (a, b)

    def pmf(weights):
        return [w / sum(weights) for w in weights]

    def plus(p, q):
        out = [0.0] * (len(p) + len(q) - 1)
        for i, x in enumerate(p):
            for j, y in enumerate(q):
                out[i + j] += x * y
        return out

    def thinned(p, s):  # how many of them lie in a scope that holds a share s of the digest's records
        return [sum(p[t] * math.comb(t, c) * s ** c * (1 - s) ** (t - c) for t in range(c, len(p)))
                for c in range(len(p))]

    phantoms, cross_refs = pmf(RETRACTION_WEIGHTS), pmf(CROSS_REF_WEIGHTS)
    every = plus(plus(phantoms, cross_refs), pmf(SAME_ID_WEIGHTS))
    for residual, bound in ((every, 0.05), (plus(phantoms, cross_refs), 0.03), (phantoms, 0.0)):
        for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9):
            in_scope = thinned(residual, s)
            assert max(in_scope) - in_scope[0] <= bound + 1e-12, (s, in_scope)
            assert s >= 1 / 3 or max(in_scope) == in_scope[0], (s, in_scope)


def _correlation(x, y) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def test_joint_misreadings_are_those_of_every_combination_of_traps(sample):
    """Off the legacy path a trap's fallen answers come from the combinations of only the traps that change a
    question's rows (traps.score_traps), which give, answer for answer, what every combination of every trap
    gives."""
    from evalgen.traps import score_traps

    for digest in sample[::9]:
        inst, bound = digest["inst"], digest["bound"]
        data = DigestData(inst.reports, inst.notices)
        traps = inst.traps
        if len(traps) > 8:
            continue
        want: dict = defaultdict(lambda: defaultdict(set))
        for combo in itertools.product(*[[None] + list(range(len(t.mishandles))) for t in traps]):
            misread = [i for i, c in enumerate(combo) if c is not None]
            if not misread:
                continue
            mishandle = Mishandle.merged(traps[i].mishandles[combo[i]] for i in misread)
            for q in inst.questions:
                answer = normalize(q.answer_type, execute(q.chain, data, mishandle=mishandle, units=bound.units,
                                                          resolver=bound.resolve))
                if answer != normalize(q.answer_type, inst.answer_key[q.qid]):
                    for i in misread:
                        want[traps[i].trap_id][q.qid].add(answer)
        stored = {t.trap_id: {qid: {normalize(inst.question(qid).answer_type, a) for a in answers}
                              for qid, answers in t.fallen.items()} for t in traps}
        assert stored == {t.trap_id: dict(want[t.trap_id]) for t in traps}, inst.instance_id
        score_traps(traps, inst.questions, inst.answer_key, inst.reports, inst.notices, domain=bound,
                    time_margin=Context(inst.compiled_at, inst.window_hours).time_margin)  # and again, alike
        assert {t.trap_id: {qid: {normalize(inst.question(qid).answer_type, a) for a in answers}
                            for qid, answers in t.fallen.items()} for t in traps} == stored


# --------------------------------------------------------------------------------- shortcut readers


def _score(digest, keep) -> list[bool]:
    data, units, bound = digest["data"], digest["units"], digest["bound"]
    native = [data.reports[i] for i in keep]
    return [is_correct(q.answer_type, digest["inst"].answer_key[q.qid],
                       execute(q.chain, DigestData(native, []), skip=("dedupe",), units=units, resolver=bound.resolve),
                       unit=q.unit) for q in digest["inst"].questions]


def _merged(reports, linked, pick=None) -> list[int]:
    """One report of each cluster of reports linked to one another (single linkage): by default the one
    whose source files the most reports (then the first printed)."""
    group = list(range(len(reports)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            if linked(reports[i], reports[j]):
                group[root(j)] = root(i)
    files = Counter(r.source for r in reports)
    clusters: dict[int, list[int]] = {}
    for i in range(len(reports)):
        clusters.setdefault(root(i), []).append(i)
    pick = pick or (lambda c: max(c, key=lambda i: (files[reports[i].source], -i)))
    return sorted(pick(c) for c in clusters.values())


def readings_of(sample) -> dict:
    """Questions right, per reader, over the sample, and per domain."""
    scores: Counter = Counter()
    for digest in sample:
        reports, bound = digest["reports"], digest["bound"]
        located = bound.spatial is not None
        spread = {f.key: max(max(r.values[f.key] for r in reports) - min(r.values[f.key] for r in reports), 1e-9)
                  for f in bound.fields}

        def close(a, b, seconds=40, same_group=False):
            return (abs(a.time - b.time) <= seconds and (not same_group or a.group == b.group)
                    and (not located or haversine_km(a.lat, a.lon, b.lat, b.lon) <= 8.5))

        def similar(a, b):
            return a.group == b.group and all(abs(a.values[k] - b.values[k]) <= 0.08 * s for k, s in spread.items())

        def same_id(a, b):
            return a.ref.casefold() == b.ref.casefold()

        def same_place(a, b):  # one site's reports within an hour (a weather station's, say)
            return a.place is not None and a.place == b.place and abs(a.time - b.time) <= 3600

        everything = list(range(len(reports)))
        readers = {
            "naive": everything,
            "proximity": _merged(reports, close),
            "proximity_hour": _merged(reports, lambda a, b: close(a, b, 3600, True)),
            "similarity": _merged(reports, similar),
            "same_place": _merged(reports, same_place),
            "status": [i for i, r in enumerate(reports) if r.status != bound.statuses[1]],
            # Reports that share an ID are one record (rule 1); which of them is real only the status says.
            "id_first": _merged(reports, same_id, pick=min),
            "id_latest": _merged(reports, same_id, pick=lambda c: max(c, key=lambda i: (reports[i].time, -i))),
            "id_status": _merged(reports, same_id, pick=lambda c: next(
                (i for i in c if reports[i].status == bound.statuses[0]), c[0])),
        }
        for name, keep in readers.items():
            got = sum(_score(digest, keep))
            scores[name] += got
            scores[(digest["domain"].name, name)] += got
        specs = [{"qid": q.qid, "chain": q.chain} for q in digest["inst"].questions]
        oracle = solve(digest["inst"].renderings[0].prompt, specs, domain=digest["inst"].domain, genre=digest["genre"])
        got = sum(is_correct(q.answer_type, digest["inst"].answer_key[q.qid], oracle[q.qid], unit=q.unit)
                  for q in digest["inst"].questions)
        scores["oracle"] += got
        scores[(digest["domain"].name, "oracle")] += got
    return scores


def closed(scores, name, domain=None) -> float:
    """The share of the naive reader's gap to the oracle that a reader closes."""
    at = (lambda n: scores[n]) if domain is None else (lambda n: scores[(domain, n)])
    return (at(name) - at("naive")) / (at("oracle") - at("naive"))


def test_no_shortcut_reads_the_digest_better_than_the_naive_reader(sample):
    """Merging reports that lie within 40 s (and 8.5 km) of each other, or within an hour in their group or at
    their site, or whose every value lies within 8% of its field's spread of another's in their group,
    keeping the commoner source, and reading the rest as the naive reader does, closes at most a tenth of
    the gap between the naive reader and the oracle in all (a fifth in any domain), while the intended
    status rule (a report of the second status settles nothing) closes far more. Of reports that share an
    ID, keeping the first printed or the latest does no better than a coin would: only the status says
    which is real."""
    scores = readings_of(sample)
    assert scores["oracle"] > scores["naive"]
    for name in ("proximity", "proximity_hour", "similarity", "same_place"):
        assert closed(scores, name) <= 0.1, (name, closed(scores, name))
        for domain in DOMAINS:
            # (28 instances a domain: a reader that the 0.2.0 placement let through closed 0.35 or more)
            assert closed(scores, name, domain.name) <= 0.2, (name, domain.name, closed(scores, name, domain.name))
    assert closed(scores, "status") >= 0.25, closed(scores, "status")
    merged = (closed(scores, "id_first") + closed(scores, "id_latest")) / 2
    assert closed(scores, "id_status") - merged >= 0.04, scores


# ------------------------------------------------------------------------ plugin domains, the legacy path


def test_a_plugin_domain_gets_the_same_placement(toys):
    """A plugin domain's traps are planted by the same builders off the legacy path: its phantoms are records
    of its snapshot that the digest does not keep, and merging near-simultaneous reports gains nothing."""
    sample = digests_of(toys.toy, 9, 12, genres=("classic", "news", "table"))
    for digest in sample:
        spare = slots_of(digest)
        for report in map(digest["bound"].to_generic, digest["inst"].reports):
            if report.role == "phantom":
                assert report.ref in spare and spare[report.ref].values == report.values
    scores = readings_of_simple(sample)
    assert closed(scores, "proximity") <= 0.1 and closed(scores, "status") >= 0.25, scores


def test_no_share_taken_off_a_count_carries_to_a_held_out_domain_and_genre(sample, toys, monkeypatch):
    """A reader who takes a share off every count (round(a x the naive count)), with one factor fitted on the
    built-in domains, reading no status, ID, note or notice, closed 9.7 to 10% of the gap between the naive
    reader and the oracle there (a = 0.84), and 11 to 13% on the held-out toy domain and the toy-telex genre:
    planted reports were a steady share of every digest (traps.draw_plan). Now the factor fitted on the
    built-in sample closes less than a twentieth of the gap on it, and less than a tenth on the toy domain, in
    classic and in the held-out genre."""
    from pathlib import Path

    from evalgen import genres

    monkeypatch.setattr(genres, "REGISTRY", genres.Registry())
    telex = genres.load_plugin(Path(__file__).resolve().parent / "toy_genre.py")
    toy = digests_of(toys.toy, 29, 48, genres=("classic", telex))
    # (each domain on a seed of its own: a digest's plan is drawn on its instance's seed path, which the module's
    # sample shares between its domains, so that it holds 28 plans four times over)
    public = [d for seed, domain in zip((31, 37, 41, 43), DOMAINS) for d in digests_of(domain, seed, 24)]

    def answers(digests):  # (a counting question?, the naive answer, the question, the key) of each question
        out = []
        for digest in digests:
            data, units, bound, inst = digest["data"], digest["units"], digest["bound"], digest["inst"]
            for q in inst.questions:
                naive = execute(q.chain, DigestData(data.reports, []), skip=("dedupe",), units=units,
                                resolver=bound.resolve)
                out.append((any(Step(s).kind == "count" for s in q.chain), naive, q, inst.answer_key[q.qid]))
        return out

    public, held_out = answers(public), answers(toy)
    grid = [round(1 - 0.01 * i, 2) for i in range(31)]
    fitted = max(grid, key=lambda a: (sum(round(a * n) == k for c, n, _, k in public if c), a))
    for rows_, most in ((public, 0.05), (held_out, 0.1)):
        right = lambda scaled: sum(is_correct(q.answer_type, k, round(fitted * n) if scaled and c else n, unit=q.unit)  # noqa: E731
                                   for c, n, q, k in rows_)
        naive_right = right(False)
        assert (right(True) - naive_right) / (len(rows_) - naive_right) < most, (fitted, right(True), naive_right)


def readings_of_simple(sample) -> Counter:
    """The naive, proximity and status readers and the oracle, over a sample (readings_of without the rest)."""
    scores: Counter = Counter()
    for digest in sample:
        reports, bound = digest["reports"], digest["bound"]
        located = bound.spatial is not None

        def close(a, b):
            return abs(a.time - b.time) <= 40 and (not located or haversine_km(a.lat, a.lon, b.lat, b.lon) <= 8.5)

        for name, keep in (("naive", list(range(len(reports)))), ("proximity", _merged(reports, close)),
                           ("status", [i for i, r in enumerate(reports) if r.status != bound.statuses[1]])):
            scores[name] += sum(_score(digest, keep))
        specs = [{"qid": q.qid, "chain": q.chain} for q in digest["inst"].questions]
        oracle = solve(digest["inst"].renderings[0].prompt, specs, domain=digest["inst"].domain, genre=digest["genre"])
        scores["oracle"] += sum(is_correct(q.answer_type, digest["inst"].answer_key[q.qid], oracle[q.qid], unit=q.unit)
                                for q in digest["inst"].questions)
    return scores


def test_the_legacy_path_keeps_the_020_placement():
    """Legacy output is the 0.2.0 output byte for byte (tests/test_regression_020.py), so the legacy path
    keeps the 0.2.0 builders and their tells (a re-report 3 to 40 s from its quake, traps.py, DESIGN.md),
    and its plan: one retraction in every digest, no scopes matched, no trap following another (and no seed
    path of the retractions drawn off it, tests/test_generic_records.py)."""
    assert builders_for(EARTHQUAKES) is EARTHQUAKE_BUILDERS
    assert builders_for(EARTHQUAKES, slotted=True) is not EARTHQUAKE_BUILDERS
    legacy = TrapConfig()
    assert legacy.retractions == 1 and not (legacy.match_scope or legacy.guarded)
    assert RETRACTION_WEIGHTS[0] == max(RETRACTION_WEIGHTS)  # off it, none is the likeliest number


# ------------------------------------------------------------------------------- the neo generator


def test_neo_moon_approaches_pile_up_at_no_distance():
    """A Moon approach's distance is drawn inside its range, not clamped to its end: no miss distance is
    shared by more synthetic approaches than chance gives (the clamp put 5% of Moon approaches at 44879 km)."""
    compiled_at = int(datetime(2026, 6, 1, 6, tzinfo=timezone.utc).timestamp())
    records = NEO.synthetic(random.Random(3), 6000, compiled_at, 720)
    assert Counter(r.values["dist"] for r in records).most_common(1)[0][1] <= 3


def test_neo_designations_never_postdate_the_compilation():
    """An object is found around its approach, never after the digest is compiled: across New Year, an
    approach in December carries the old year's designations (it did all carry the new year's)."""
    compiled_at = int(datetime(2026, 1, 1, 6, tzinfo=timezone.utc).timestamp())
    records = NEO.synthetic(random.Random(1), 2000, compiled_at, 720)
    halves = "ABCDEFGHJKLMNOPQRSTUVWXY"
    latest = (2026, 0)  # compiled in the first half of January 2026
    december = [r for r in records if datetime.fromtimestamp(r.time, tz=timezone.utc).year == 2025]
    new_year = 0
    for r in records:
        found = re.fullmatch(r"(?:[ACDIPX])?(\d{4})([A-Y])[A-Z]\d*", designation_of(r.id))
        if found:
            assert (int(found.group(1)), halves.index(found.group(2))) <= latest, r.id
            new_year += r in december and int(found.group(1)) == 2026
    assert len(december) > 500 and new_year < 0.05 * len(december)
