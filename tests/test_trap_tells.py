"""Off the legacy path, a planted report is told from a real one only by the signals the reading rules name:
a shared ID or a note naming one, the status, and the notices (traps.py). Every planted report stands in
for a *slot*, a record drawn with the digest but not kept, so its time, place, source, values, precision
and ID are a real record's. These tests pin that, and measure it as a red team would, on a small sample of
every built-in domain in every genre, read from its prompt text: no feature the rules never mention (how
close a report is to another in time, space or value, how often its source files, the precision of its
values, how many questions' scopes it lies in) tells planted reports from real ones, and no reader that
merges or drops reports by such a feature, without reading a status, an ID, a note or a notice, does
better than the naive reader. The same holds for a plugin domain, which gets the same builders.

The full measurement (1,008 fresh instances per sample, bootstrap intervals by domain and genre, the
held-out toy plugin and the recorded CAD response) is described under Traps in docs/DESIGN.md. Before
this placement, merging reports within 40 s and keeping the commoner source closed 38% of the gap between
the naive reader and the oracle; on these samples such readers close none of it."""
from __future__ import annotations

import random
import re
from collections import Counter
from datetime import datetime, timezone

import pytest

from evalgen.domains.base import Record
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.domains.invented import INVENTED
from evalgen.domains.neo import NEO, designation_of
from evalgen.domains.weather import WEATHER
from evalgen.genres import get_genre
from evalgen.geo import haversine_km
from evalgen.ops import DigestData, execute, passes, role
from evalgen.pipeline import GenerateConfig, build_instance
from evalgen.questions import Context, compose_new_templates
from evalgen.scoring import is_correct
from evalgen.seeds import SeedLog
from evalgen.solver import _reading, solve
from evalgen.sources import build_domain_snapshot
from evalgen.traps import EARTHQUAKE_BUILDERS, builders_for

DOMAINS = (EARTHQUAKES, WEATHER, NEO, INVENTED)
GENRES = ("classic", "news", "email", "chat", "table", "form", "radio")
SEED, INSTANCES = 61, 14  # two instances of each domain in each genre
LOW, HIGH = 0.3, 0.7  # a feature whose AUC lies outside this band on the sample tells planted reports apart
PLANTED = ("dup_same", "dup_xref", "second", "phantom", "corr")


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


# ----------------------------------------------------------------------- what a planted report is


def test_the_sample_reads_every_built_in_domain_in_every_genre(sample):
    assert Counter((d["domain"].name, d["genre"]) for d in sample) == {(domain.name, genre): 2 for domain in DOMAINS
                                                                       for genre in GENRES}
    assert all(set(d["labels"]) >= {"real", "phantom"} for d in sample)


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
    assert counts["phantom"] >= len(sample) and counts["duplicate"] >= 2 * len(sample) and counts["conflict"] > 0


def test_a_corrected_value_is_a_real_value_of_the_group_not_one_just_past_a_threshold(sample):
    """The wrong value that a correction fixes is the value of a record of the group that the digest does
    not keep (written to the corrected record's places), not one placed just past an asked threshold."""
    taken, corrections = 0, 0
    for digest in sample:
        bound, inst = digest["bound"], digest["inst"]
        for notice in map(bound.to_generic, inst.notices):
            if notice.kind != "correction":
                continue
            corrections += 1
            record = next(generic(r, bound) for r in map(bound.dress, digest["snapshot"].events) if r.id == notice.target)
            spec = bound.field(notice.field)
            places = stored_places(notice.new_value, spec.decimals)
            spare = [r for r in slots_of(digest).values() if r.group == record.group]
            taken += any(round(r.values[spec.key], places) == notice.old_value for r in spare)
    assert corrections >= 10 and taken >= 0.8 * corrections, (taken, corrections)


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


def test_planted_reports_lie_in_as_many_question_scopes_as_real_ones(sample):
    def scopes(r, others, digest):
        units = digest["units"]
        return float(sum(all(passes(s, r, units) for s in q.chain if role(s["op"]) == "filter")
                         for q in digest["inst"].questions))

    # Placing traps where they change an answer puts planted reports in more scopes than real ones (an AUC
    # of 0.64 over 1,008 instances before each trap drew its scope from the real records'), so the band is
    # narrower than for the other features.
    for kinds in (PLANTED, ("dup_same", "dup_xref"), ("phantom",)):
        planted, real = rows(sample, scopes, kinds)
        assert 0.4 <= auc(planted, real) <= 0.6, (kinds, auc(planted, real))


def test_a_trap_that_draws_no_scope_is_followed_by_another_of_its_kind(sample):
    """A trap whose planted report lies in no question's scope changes no answer; another of its kind
    follows it, so a digest keeps as many traps that change answers as it had before scopes were matched
    (about four, of 2-3 re-reports, a retraction and a conflict)."""
    load_bearing, extra = [], 0
    for digest in sample:
        traps = digest["inst"].traps
        kinds = Counter(t.kind for t in traps)
        conflicts = sum(n for k, n in kinds.items() if k.startswith("conflicting"))
        assert kinds["duplicate"] >= 2 and kinds["retraction"] >= 1 and conflicts >= 1
        assert [t.trap_id for t in traps] == [f"T{i}" for i in range(1, len(traps) + 1)]
        extra += max(0, kinds["duplicate"] - 3) + kinds["retraction"] - 1 + conflicts - 1  # at least this many
        load_bearing.append(sum(1 for t in traps if t.affected))
    assert extra > 0 and sum(load_bearing) / len(load_bearing) >= 3.0, (extra, sum(load_bearing) / len(load_bearing))


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

        everything = list(range(len(reports)))
        readers = {
            "naive": everything,
            "proximity": _merged(reports, close),
            "proximity_hour": _merged(reports, lambda a, b: close(a, b, 3600, True)),
            "similarity": _merged(reports, similar),
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
    """Merging reports that lie within 40 s (and 8.5 km) of each other, or within an hour in their group,
    or whose every value lies within 8% of its field's spread of another's in their group, keeping the
    commoner source, and reading the rest as the naive reader does, closes at most a tenth of the gap
    between the naive reader and the oracle in all (a fifth in any domain), while the intended status rule
    (a report of the second status settles nothing) closes far more. Of reports that share an ID, keeping
    the first printed or the latest does no better than a coin would: only the status says which is real."""
    scores = readings_of(sample)
    assert scores["oracle"] > scores["naive"]
    for name in ("proximity", "proximity_hour", "similarity"):
        assert closed(scores, name) <= 0.1, (name, closed(scores, name))
        for domain in DOMAINS:
            # (14 instances a domain: a reader that the 0.2.0 placement let through closed 0.35 or more)
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
    keeps the 0.2.0 builders and their tells: a re-report 3 to 40 s from its quake (traps.py, DESIGN.md)."""
    assert builders_for(EARTHQUAKES) is EARTHQUAKE_BUILDERS
    assert builders_for(EARTHQUAKES, slotted=True) is not EARTHQUAKE_BUILDERS


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
