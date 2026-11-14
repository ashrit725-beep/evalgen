"""Fixes from the 0.3.0 audit of question wording and guards, genre prose, and form and time details.

  - Every genre but classic writes an earthquake's 0.2.0 records in their generic form and reads them
    back through one map (Domain.to_generic, Domain.from_generic), so no genre reads an input in a way of
    its own; an earthquake correction of a field other than the magnitude is an error, never ignored.
  - Off the legacy path a question's answer belongs to its own record alone: the runner-up of a max, a
    min, a time of the largest value or a nearest distance is scored wrong, and a max or min record is
    never printed exactly halfway between two answers of the asked places. A question's point is named
    by its number alone, never by the source of the record it is drawn near (which is nearly always the
    answer's). A quantity of 1 takes the singular, and the groups of a comparison are quoted.
  - Prose places a close approach near its body, never in it; a classic prose line says its record was
    recorded at its time; the earthquake glossary names the magnitude type in words.
  - A weather observation's local time is its station's own, daylight saving time included.
  - A form's sent time is drawn alike for every form of a rendering, so it never tells whether a notice
    names the form's record.

Legacy questions and digests keep the 0.2.0 guards, labels and wording (the regression test pins them).
"""
from __future__ import annotations

import ast
import copy
import random
import re
from dataclasses import replace

import pytest

from evalgen import pipeline
from evalgen.domains import get_domain
from evalgen.domains.base import RecordReport
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import get_genre
from evalgen.genres.base import RenderContext, domain_items
from evalgen.models import Event, Instance, Notice, Report
from evalgen.ops import Step, by_field, distance_km, execute, rows_after_filters, terminal_step
from evalgen.pipeline import GenerateConfig, build_instance
from evalgen.questions import Context, compose_new_templates, guards_ok
from evalgen.report import load_instances
from evalgen.rules import resolve
from evalgen.scoring import is_correct
from evalgen.seeds import SeedLog
from evalgen.sources import build_domain_snapshot
from evalgen.units import encode, find_unit

from conftest import ROOT, quiet

GENERIC_GENRES = ("news", "email", "chat", "table", "form", "radio")  # every genre that writes records generically
DOMAINS = ("earthquakes", "weather", "neo", "invented")
GENRES = ("classic", *GENERIC_GENRES)


def instances_of(domain, genre: str, seed: int, n: int = 1) -> list[Instance]:
    """The first instances of a run about `domain` with every rendering in `genre`, built as the pipeline
    builds them off the legacy path."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    plan = (get_genre(genre),) * 3
    return [build_instance(i, snapshot, new, log, GenerateConfig(seed=seed), ctx, used, usage, domain=domain,
                           genre_plan=plan) for i in range(n)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Small runs off the legacy path, one per built-in domain, each instance in a genre of its own: what
    the run-wide checks below read."""
    root = tmp_path_factory.mktemp("audit")
    made = {}
    for number, domain in enumerate(DOMAINS):
        cfg = GenerateConfig(seed=900 + number, instances=12, variants=1, mix=",".join(f"{domain}:{g}" for g in GENRES))
        made[domain] = load_instances(pipeline.generate(cfg, root / domain, log=quiet))
    return made


def context(inst: Instance, domain, *, reports=None, notices=None, seed: int = 1, genre=None) -> RenderContext:
    """A rendering context of `inst` (its own reports and notices unless others are given), styled by
    `genre`'s own draw."""
    rng = random.Random(seed)
    style = genre.choose_style(rng, domain)
    return RenderContext(inst.reports if reports is None else reports, inst.notices if notices is None else notices,
                         inst.chatter, inst.compiled_at, inst.window_hours, style, rng, domain, domain.glossary())


# ------------------------------------------------------------- one map for an earthquake's records


def test_no_genre_but_classic_builds_earthquake_records_itself():
    """The genres map an earthquake's 0.2.0 Reports and corrections through Domain.to_generic and
    Domain.from_generic: none constructs a models.Report, or a 0.2.0 correction, of its own (classic is
    the 0.2.0 code)."""
    for path in sorted((ROOT / "evalgen" / "genres").glob("*.py")):
        if path.name in ("classic.py", "__init__.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        built = [ast.unparse(node.func) for node in calls if isinstance(node.func, ast.Name) and node.func.id == "Report"]
        legacy_notices = [node for node in calls
                          if any(k.arg in ("new_mag", "old_mag", "mag_type") for k in node.keywords)]
        assert not built and not legacy_notices, path.name


def test_the_one_map_reads_every_edge_case_as_from_generic_does():
    base = dict(ref="nc75012345", source="nc", status="reviewed", time=1_780_000_000, lat=38.0, lon=-122.0,
                group="Northern California", values={"mag": 3.1, "depth": 6.4})
    no_type = RecordReport(**base, qualifiers={})
    extra = RecordReport(**base, qualifiers={"mag_type": "md", "probe": "x"})
    no_position = RecordReport(**{**base, "lat": None, "lon": None}, qualifiers={"mag_type": "md"})
    depth_fix = Notice(kind="correction", target="nc75012345", time=1_780_000_100, field="depth", new_value=7.0,
                       old_value=6.4)
    assert domain_items(EARTHQUAKES, [no_type, extra]) == []  # no earthquake report: it changes nothing
    (report,) = domain_items(EARTHQUAKES, [no_position])
    assert isinstance(report, Report) and (report.lat, report.lon, report.mag_type) == (None, None, "md")
    assert domain_items(EARTHQUAKES, [depth_fix]) == [depth_fix]  # passed on as it is: resolve refuses it


@pytest.mark.parametrize("name", GENERIC_GENRES)
def test_every_genre_writes_and_reads_an_earthquake_correction_of_depth_alike(name):
    """A correction of a field that no 0.2.0 correction holds (the depth) is written and read back in
    its generic form by every genre alike (news refused to print it, email and chat lost it), and the
    reading rules then refuse it rather than ignore it."""
    genre = get_genre(name)
    (inst,) = instances_of(EARTHQUAKES, name, seed=61)
    target = next(r for r in inst.reports if r.role == "primary")
    fix = Notice(kind="correction", target=target.ref, time=inst.compiled_at - 120, field="depth",
                 new_value=round(target.depth_km + 3.0, 1), old_value=target.depth_km)
    notices = [*inst.notices, fix]
    ctx = context(inst, EARTHQUAKES, notices=notices, genre=genre)
    text = genre.render(ctx)  # checked to read back exactly
    data = genre.parse(text, EARTHQUAKES.glossary(), EARTHQUAKES)
    assert fix in data.notices
    with pytest.raises(ValueError, match="no new magnitude"):
        resolve(data.reports, data.notices)


@pytest.mark.parametrize("name", GENERIC_GENRES)
def test_every_genre_refuses_an_earthquake_report_without_a_magnitude_type(name):
    """No genre reads back a report that is no earthquake report (news and form used to rebuild it with
    no magnitude type, the others dropped it), so every genre refuses to print one."""
    genre = get_genre(name)
    (inst,) = instances_of(EARTHQUAKES, name, seed=62)
    reports = [replace(inst.reports[0], mag_type=None), *inst.reports[1:]]
    with pytest.raises(ValueError):
        genre.render(context(inst, EARTHQUAKES, reports=reports, genre=genre))


def test_an_earthquake_correction_of_another_field_is_an_error():
    report = Report(ref="nc75012345", network="nc", status="reviewed", time=1_780_000_000, lat=38.0, lon=-122.0,
                    depth_km=6.4, mag=3.1, mag_type="md", region="Northern California")
    depth_fix = Notice(kind="correction", target="nc75012345", time=1_780_000_100, field="depth", new_value=7.0,
                       old_value=6.4)
    with pytest.raises(ValueError, match="'depth'"):
        resolve([report], [depth_fix])
    magnitude_fix = Notice(kind="correction", target="nc75012345", time=1_780_000_100, new_mag=3.4, old_mag=3.1,
                           mag_type="ml")
    (event,) = resolve([report], [magnitude_fix])
    assert (event.mag, event.mag_type, event.depth_km) == (3.4, "ml", 6.4)


# ------------------------------------------------------------ answers of their own record alone


def quake(event_id: str, *, depth: float = 10.0, mag: float = 3.0, time: int = 1_780_000_000, lat: float = 38.0,
          lon: float = -122.0) -> Event:
    return Event(id=event_id, time=time, lat=lat, lon=lon, depth_km=depth, mag=mag, mag_type="md",
                 region="Northern California", network="nc")


def deepest(unit: str) -> list[dict]:
    return [{"op": "dedupe"}, {"op": "max_depth"}, {"op": "convert", "to": unit}, {"op": "round", "ndigits": 1}]


def test_a_max_whose_runner_up_is_scored_right_is_refused_off_the_legacy_path():
    """80.5 km is 50.0 mi, and the runner-up's 79.9 km is 49.6 mi, which the depth tolerance accepts: a
    reader who took the wrong quake would be scored right. Legacy questions keep the 0.2.0 guard."""
    close = [quake("nc1", depth=80.5), quake("nc2", depth=79.9), quake("nc3", depth=20.0)]
    assert is_correct("depth", execute(deepest("mi"), close), execute(deepest("mi"), close[1:]), unit="mi")
    assert guards_ok(deepest("mi"), close) and not guards_ok(deepest("mi"), close, strict=True)
    clear = [quake("nc1", depth=80.5), quake("nc2", depth=60.0), quake("nc3", depth=20.0)]
    assert guards_ok(deepest("mi"), clear, strict=True)
    tied = [quake("nc1", depth=80.5), quake("nc2", depth=80.5), quake("nc3", depth=20.0)]
    assert guards_ok(deepest("mi"), tied, strict=True)  # a tie answers alike, whichever quake is taken


def test_a_max_printed_halfway_between_two_answers_is_refused_off_the_legacy_path():
    """82.8 km is printed 51.45 mi (82.8 / 1.609344 = 51.4495...): rounded to one place it reads 51.4 or
    51.5, whichever way a reader rounds the printed half. Asked in km it is printed 82.8, exactly."""
    records = [quake("nc1", depth=82.8), quake("nc2", depth=30.0)]
    assert encode(82.8, 1, find_unit(EARTHQUAKES.field("depth").units, "mi")) == "51.45"
    assert guards_ok(deepest("mi"), records) and not guards_ok(deepest("mi"), records, strict=True)
    assert guards_ok(deepest("km"), records, strict=True)


def test_a_time_of_the_largest_whose_runner_up_is_a_minute_away_is_refused_off_the_legacy_path():
    largest = [{"op": "dedupe"}, {"op": "time_of_largest"}]
    near = [quake("nc1", mag=4.0, time=1_780_000_000), quake("nc2", mag=3.5, time=1_780_000_050)]
    assert guards_ok(largest, near) and not guards_ok(largest, near, strict=True)
    apart = [quake("nc1", mag=4.0, time=1_780_000_000), quake("nc2", mag=3.5, time=1_780_000_600)]
    assert guards_ok(largest, apart, strict=True)


def runner_up(q, records, domain):
    """The record a reader who misses a value question's own record answers from (None: no other one)."""
    term = Step(terminal_step(q.chain), domain.units)
    rows = rows_after_filters(q.chain, records, units=domain.units)
    if term.kind in ("max", "min"):
        pick = max if term.kind == "max" else min
        top = pick(r.values[term.field] for r in rows)
        return pick((r for r in rows if r.values[term.field] != top), key=lambda r: r.values[term.field], default=None)
    if term.kind == "time_of_max":
        return by_field(rows, term.field)[1] if len(rows) > 1 else None
    ordered = sorted(rows, key=lambda r: (distance_km(term.raw, r), r.id))
    return ordered[1] if len(ordered) > 1 else None


@pytest.mark.parametrize("name", DOMAINS)
def test_no_value_answer_off_the_legacy_path_is_shared_by_another_record(runs, name):
    """In every built-in domain, no max, min, time of the largest value or nearest distance accepts the
    answer of its runner-up record, and no max or min record is printed halfway between two answers."""
    asked = 0
    for inst in runs[name]:
        domain = get_domain(inst.domain).bind(inst.surface)
        records = domain.resolve(inst.reports, inst.notices)
        for q in inst.questions:
            term = Step(terminal_step(q.chain), domain.units)
            if term.kind not in ("max", "min", "time_of_max", "nearest_distance"):
                continue
            asked += 1
            other = runner_up(q, records, domain)
            if other is not None:
                wrong = execute(q.chain, [other], units=domain.units)
                assert not is_correct(q.answer_type, inst.answer_key[q.qid], wrong, unit=q.unit), (inst.instance_id,
                                                                                                    q.qid)
            if term.kind in ("max", "min") and q.unit is not None:
                spec = domain.field(term.field)
                unit = find_unit(spec.units, q.unit)
                stored = execute([s for s in q.chain if s["op"] not in ("convert", "round")], records,
                                 units=domain.units)
                printed = encode(stored, spec.decimals, unit)
                halfway = re.fullmatch(rf"-?\d+\.\d{{{unit.places(spec.decimals)}}}5", printed)
                assert halfway is None, (inst.instance_id, q.qid, printed)
    assert asked >= 10


@pytest.mark.parametrize("name", ("earthquakes", "weather"))
def test_a_point_label_off_the_legacy_path_names_no_source(runs, name):
    """A question's point lies near a record drawn at random, which is nearly always the nearest record:
    a label that named its source (0.2.0's "station NC-42", weather's "COOP site 51 near KPIT") gave the
    answer's source away. Off the legacy path a label is the point's number alone."""
    labels = 0
    for inst in runs[name]:
        sources = {r.source.upper() for r in inst.reports}
        for q in inst.questions:
            for step in q.chain:
                if "label" in step:
                    labels += 1
                    assert not any(re.search(rf"\b{re.escape(s)}\b", step["label"]) for s in sources), step["label"]
    assert labels >= 5


def test_a_domain_whose_point_label_names_a_source_is_refused(toys):
    toys.toy.validate()
    domain = copy.copy(toys.toy)
    domain.spatial = replace(toys.toy.spatial, point_label="buoy {source}-{number}")
    with pytest.raises(ValueError, match="point label"):
        domain.validate()


# ------------------------------------------------------------------------------ question wording


def test_a_quantity_of_one_takes_the_singular_off_the_legacy_path():
    """ "colder than 1 degrees Celsius" read as a slip; off the legacy path a quantity of exactly 1 takes
    its unit's words for one (Unit.one). The 0.2.0 wording, which legacy prompts keep, is unchanged."""
    from evalgen.domains.weather import WEATHER
    from evalgen.models import Question
    from evalgen.questions import phrase_question

    def question(unit: str, value: float) -> Question:
        return Question("Q1", "P01", "public", [{"op": "dedupe"}, {"op": "below", "field": "temp", "value": value,
                                                                   "unit": unit}, {"op": "count"}], "count")

    for unit, one in (("°C", "1 degree Celsius"), ("°F", "1 degree Fahrenheit"), ("K", "1 kelvin")):
        texts = [phrase_question(question(unit, 1.0), random.Random(seed), domain=WEATHER) for seed in range(8)]
        assert all(one in text and f"{one}s" not in text for text in texts), texts
        assert all("-1 degree" in phrase_question(question(unit, -1.0), random.Random(seed), domain=WEATHER)
                   for seed in range(3)) or unit == "K"
        assert all("1.5 degrees" in phrase_question(question(unit, 1.5), random.Random(seed), domain=WEATHER)
                   for seed in range(3)) or unit == "K"
    shallow = Question("Q1", "P06", "public", [{"op": "dedupe"}, {"op": "depth_shallower_than", "value": 1.0,
                                                                  "unit": "mi"}, {"op": "count"}], "count")
    legacy = {phrase_question(shallow, random.Random(seed)) for seed in range(8)}
    genre = {phrase_question(shallow, random.Random(seed), lexicon=EARTHQUAKES.prompt_lexicon()) for seed in range(8)}
    assert all("1 miles" in text for text in legacy) and all("1 mile" in text and "1 miles" not in text
                                                              for text in genre)
    assert {text.replace("1 mile", "1 miles") for text in genre} == legacy  # the same draws, word for word


@pytest.mark.parametrize("name", DOMAINS)
def test_no_question_off_the_legacy_path_writes_one_with_a_plural(runs, name):
    for inst in runs[name]:
        for rendering in inst.renderings:
            for text in rendering.question_text.values():
                assert not re.search(r"(?<![\d.])-?1 (?:degrees|kelvins|miles|kilometres|knots|metres|hectopascals|"
                                     r"inches|astronomical units|lunar distances)\b", text), text


def test_the_groups_of_a_comparison_are_quoted_off_the_legacy_path():
    """ "which region has more of them: Honshu, Japan or Southern Alaska?" reads as three choices. Off the
    legacy path both names are quoted, as the answer must give one exactly as written."""
    from evalgen.models import Question
    from evalgen.questions import phrase_question

    compare = Question("Q1", "P05", "public", [{"op": "dedupe"}, {"op": "mag_at_least", "value": 1.7},
                                               {"op": "compare_regions", "a": "Honshu, Japan", "b": "Southern Alaska"}],
                       "region")
    legacy = {phrase_question(compare, random.Random(seed)) for seed in range(8)}
    genre = {phrase_question(compare, random.Random(seed), lexicon=EARTHQUAKES.prompt_lexicon()) for seed in range(8)}
    assert all("Honshu, Japan or Southern Alaska" in text for text in legacy)
    assert all('"Honshu, Japan" or "Southern Alaska"' in text for text in genre)
    assert {text.replace('"', "") for text in genre} == {text.replace('"', "") for text in legacy}


@pytest.mark.parametrize("name", DOMAINS)
def test_every_comparison_off_the_legacy_path_quotes_its_groups(runs, name):
    compared = 0
    for inst in runs[name]:
        for q in inst.questions:
            term = terminal_step(q.chain)
            if Step(term).kind != "compare_groups":
                continue
            compared += 1
            for rendering in inst.renderings:
                text = rendering.question_text[q.qid]
                assert f'"{term[Step(term).key("a")]}"' in text and f'"{term[Step(term).key("b")]}"' in text, text
    assert compared >= 1


# -------------------------------------------------------------------------- genre prose and glossary


BODIES = ("Earth", "Moon", "Venus", "Mars", "Mercury")


@pytest.mark.parametrize("name", GENRES)
def test_a_body_approached_is_never_written_as_a_place(name):
    """A close approach is recorded near the body it passes, never "in Moon" (Neo.group_preposition), and
    every genre still reads its digest back exactly (render checks it)."""
    near = 0
    for inst in instances_of(get_domain("neo"), name, seed=71, n=2):
        for rendering in inst.renderings:
            assert not re.search(rf"\b[Ii]n (?:{'|'.join(BODIES)})\b", rendering.digest), name
            assert not re.search(rf"^(?:Reports from|Across) (?:{'|'.join(BODIES)})$", rendering.digest, re.M)
            near += len(re.findall(rf"\b[Nn]ear (?:{'|'.join(BODIES)})\b", rendering.digest))
    assert near > 0 or name in ("table", "form", "radio")  # these print the body under its label


def test_a_classic_prose_line_gives_the_time_its_record_was_recorded():
    """The prose line opened with a time and then said that a source "lists" the record, which read as
    the time of the listing; it now says the record was recorded then, and names its source after."""
    (inst,) = instances_of(get_domain("weather"), "classic", seed=72)
    lines = [line for r in inst.renderings for line in r.digest.splitlines() if " — " in line and "Listed by" in line]
    assert lines and not any(" lists " in r.digest for r in inst.renderings)
    pattern = re.compile(r"^.+ — weather observation \S+ \((?:verified|preliminary)\) was recorded in .+: .+\. Listed by "
                         r"\S+\.")
    assert all(pattern.match(line) for line in lines), lines[:2]


def test_the_earthquake_glossary_names_the_magnitude_type_in_words():
    """The 0.2.0 digest prints a magnitude's type as "(md)", "MD 3.2" or "type=md", never under its key
    "mag_type", so the glossary names it in words too, and reads back exactly."""
    from evalgen.genres.base import parse_glossary, render_glossary

    text = render_glossary(EARTHQUAKES.glossary())
    assert 'its "mag_type" ("magnitude type") is given with each value' in text
    assert parse_glossary(text) == EARTHQUAKES.glossary()
    assert parse_glossary(text).fields[0].qualifier_label == "magnitude type"


# ---------------------------------------------------------------------------------- local times


def test_a_weather_station_keeps_its_own_time_zone():
    """Digests print a located record's local time with its offset stated. The nominal offset of a
    longitude put Detroit at UTC-06:00 in winter and Atlanta at UTC-06:00 in summer; a weather station's
    local time is that of its zone, with US daylight saving time (Phoenix keeps none)."""
    import calendar

    from evalgen.domains.base import Record
    from evalgen.domains.weather import STATION_BY_ID, WEATHER

    def offset(station_id: str, *when: int) -> int:
        s = STATION_BY_ID[station_id]
        record = Record(id=f"{station_id}-12345", time=calendar.timegm((*when, 0)), lat=s.lat, lon=s.lon,
                        group=s.region, source=station_id, values={}, place=s.name)
        return WEATHER.utc_offset(record)

    assert offset("KDTW", 2026, 2, 4, 1, 21) == -5 and offset("KDTW", 2026, 7, 4, 12, 0) == -4
    assert offset("KATL", 2026, 7, 30, 22, 2) == -4 and offset("KOKC", 2026, 5, 27, 17, 13) == -5
    assert offset("KPHX", 2026, 1, 15, 12, 0) == offset("KPHX", 2026, 7, 15, 12, 0) == -7
    assert offset("KSEA", 2026, 3, 8, 9, 59) == -8 and offset("KSEA", 2026, 3, 8, 10, 0) == -7  # 2:00 PST
    assert offset("KSEA", 2026, 11, 1, 8, 59) == -7 and offset("KSEA", 2026, 11, 1, 9, 0) == -8  # 2:00 PDT


@pytest.mark.parametrize("name", GENRES)
def test_every_genre_writes_a_weather_observation_in_its_stations_local_time(name):
    """Wherever a genre writes an observation's time in local time, the offset is its station's, never the
    nominal offset of its longitude where the two differ. (Observations at stations of two zones can share a
    UTC second, a five-minute mark say, and the one in the zone whose offset is the other's nominal one then
    prints the very text the nominal offset would: that text is not counted against the other.)"""
    from evalgen.domains.weather import WEATHER
    from evalgen.genres.base import OFFSET_TIME_FORMATS, format_time
    from evalgen.geo import utc_offset_hours

    real = nominal = 0
    for seed in (73, 75):
        for inst in instances_of(WEATHER, name, seed=seed, n=2):
            own_texts = {format_time(r.time, fmt, WEATHER.utc_offset(r)) for r in inst.reports
                         for fmt in OFFSET_TIME_FORMATS}
            for rendering in inst.renderings:
                for r in inst.reports:
                    own, solar = WEATHER.utc_offset(r), utc_offset_hours(r.lon)
                    for fmt in OFFSET_TIME_FORMATS:
                        real += format_time(r.time, fmt, own) in rendering.digest
                        text = format_time(r.time, fmt, solar)
                        nominal += own != solar and text not in own_texts and text in rendering.digest
    assert nominal == 0 and real > 0


# ------------------------------------------------------------------------------------ form times


def form_delays(domain, seed: int, n: int) -> list[tuple[int, list[int], list[int]]]:
    """For every rendering of the first `n` instances in the form genre: the shortest time from a record to
    a notice about it, then the printed delays (sent time minus the record's time) of the report forms
    whose record a notice names, and of the others."""
    from evalgen.genres.base import parse_time

    sent_line = re.compile(r"(?:Form sent|Sent to desk|Relayed to desk|Forwarded)(?:: +| \.+ )(?P<time>.+?)(?: +│)?$",
                           re.M)
    out = []
    form = get_genre("form")
    for inst in instances_of(domain, "form", seed=seed, n=n):
        notified = {}
        for notice in inst.notices:
            notified[notice.target] = min(notified.get(notice.target, notice.time), notice.time)
        gaps = [notified[ref] - r.time for r in inst.reports for ref in {r.ref, r.same_as} if ref in notified]
        shortest = min(gap for gap in gaps if gap > 1)
        glossary = domain.bind(inst.surface).glossary()
        for rendering in inst.renderings:
            named, other = [], []
            for block in rendering.digest.split("\n\n")[1:]:
                found = sent_line.search(block)
                data = form.parse(block, glossary, domain)
                if found is None or len(data.reports) != 1:
                    continue
                (report,) = data.reports
                planted = report.ref in notified or (report.same_as or "") in notified
                (named if planted else other).append(parse_time(found["time"]) - report.time)
            out.append((shortest, named, other))
    return out


@pytest.mark.parametrize("name", ("weather", "invented"))
def test_a_forms_sent_time_does_not_tell_whether_a_notice_names_its_record(name):
    """Forms of retracted and corrected records were sent sooner than the others: only their delay was cut
    at the notice about them (a median of about 18 minutes against 45: a planted form was sent sooner than a
    real one of its rendering in 70-75% of pairs). Every form of a rendering now shares one longest delay,
    below the shortest time from any record to a notice about it, so that within a rendering, as a reader
    sees it, planted and real reports' forms are sent alike, and none after a notice about its record.
    (Across renderings the shared limit is lower where more notices are printed, so pooled delays would
    mix that in.)"""
    renderings = form_delays(get_domain(name), seed=74, n=12)
    assert sum(len(named) for _, named, _ in renderings) >= 20
    assert all(0 < delay < shortest for shortest, named, other in renderings for delay in named + other)
    pairs = [(a, b) for _, named, other in renderings for a in named for b in other]
    sooner = sum((a < b) + 0.5 * (a == b) for a, b in pairs) / len(pairs)  # 0.5: no tell
    assert len(pairs) >= 500 and 0.38 < sooner < 0.62, sooner
