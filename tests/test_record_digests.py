"""The generic record path, end to end, on the test-only toy domains (toy_domain.py, loaded as a plugin):
the exact codec of values in units, the shared field formats, the glossary a non-legacy prompt prints,
traps planted in Records, the classic genre for any schema, non-legacy prompts, the text-only oracle,
and whole runs (generate, oracle, naive, mock, score, report) with and without a location and with a
surface drawn per instance. Legacy (0.2.0) output is pinned by test_regression_020.py."""
from __future__ import annotations

import copy
import random
import re
import sys
from collections import Counter
from dataclasses import replace

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.adapters import Task, _perturb
from evalgen.digest import LINE_PRESETS, NOTICE_PLACEMENTS
from evalgen.domains import get_domain
from evalgen.domains.base import Glossary, GlossaryField, RecordReport, TrapVocab, record_rules
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import all_genres, detect_genre
from evalgen.genres.base import (COORD_FORMATS, TIME_FORMATS, RenderContext, format_coords, format_time,
                                 parse_coords, parse_glossary, parse_time, render_glossary)
from evalgen.genres.classic import (CLASSIC, RECORD_COORD_PRESETS, RECORD_HEADERS, RECORD_LAYOUTS,
                                    RECORD_TIME_PRESETS, UNIT_PRESETS)
from evalgen.models import Instance, Notice
from evalgen.ops import DigestData, execute
from evalgen.pipeline import GenerateConfig, build_instance, question_specs
from evalgen.prompt import domain_answer_spec, domain_placeholder, domain_unit_words
from evalgen.questions import Context, compose_new_templates
from evalgen.report import load_instances
from evalgen.rules import Mishandle
from evalgen.scoring import AnswerType, is_correct, normalize, register_answer_type, same_answer
from evalgen.seeds import SeedLog
from evalgen.solver import extract_digest, solve
from evalgen.sources import build_domain_snapshot, record_problems
from evalgen.traps import builders_for, check_resolution, primary_report
from evalgen.units import MAX_PLACES, Unit, decode, encode
from evalgen.util import read_json, write_json

from conftest import TOY_PLUGIN, quiet

COMPILED_AT = 1_780_000_000
FAHRENHEIT = Unit("°F", "degrees Fahrenheit", 5 / 9, -160 / 9)
AU = Unit("au", "astronomical units", 149597870.7)
LD = Unit("LD", "lunar distances", 384400.0)


def instances_of(domain, seed: int, n: int = 3, genres=(CLASSIC,) * 3) -> list[Instance]:
    """The first instances of a run about `domain` off the legacy path, built as the pipeline builds them."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    return [build_instance(i, snapshot, new, log, GenerateConfig(seed=seed), ctx, used, usage, domain=domain,
                           genre_plan=genres) for i in range(n)]


def printed(reports, notices) -> tuple[list, list]:
    """Reports and notices as a digest prints them: without the generator's bookkeeping."""
    return ([replace(r, role="primary", trap_id=None) for r in reports], [replace(n, trap_id=None) for n in notices])


def same_items(got, want) -> bool:
    """Equal as multisets (by ==; the items are not hashable)."""
    remaining = list(want)
    for item in got:
        if item not in remaining:
            return False
        remaining.remove(item)
    return not remaining


# ------------------------------------------------------------------------------- the codec


def test_the_codec_writes_each_stored_value_with_the_fewest_decimals_that_read_back():
    cases = [  # (value, stored places, unit, text)
        (21.0, 1, None, "21"), (21.0, 1, FAHRENHEIT, "69.8"), (-30.7, 1, FAHRENHEIT, "-23.3"),
        (0.0, 1, FAHRENHEIT, "32"),
        (-17.8, 1, FAHRENHEIT, "0"),  # -0.04 °F, never written "-0"
        (4.27, 2, None, "4.27"), (3185000.0, 0, AU, "0.02129041"), (3185000.0, 0, LD, "8.28564"),
    ]
    for value, places, unit, text in cases:
        assert encode(value, places, unit) == text and decode(text, places, unit) == value
    for value in (-0.0, 0.0, -0.04):  # zero is never written "-0"
        assert encode(round(value, 1), 1) == "0"
    checked = 0
    for unit in (None, FAHRENHEIT, AU, LD, Unit("kt", "knots", 1.852), Unit("x", "x", 1 / 12, 3.5)):
        for places in (0, 1, 2, 3):
            for n in range(-3000, 3000, 37):
                value = round(n * 10.0 ** -places * 7.3, places)
                text = encode(value, places, unit)
                assert decode(text, places, unit) == value and "e" not in text, (value, unit, text)
                assert not (float(text) == 0 and text.startswith("-")), text
                checked += 1
    assert checked == 6 * 4 * 163
    with pytest.raises(ValueError, match="stored to 1 decimal places"):
        encode(21.05, 1, FAHRENHEIT)  # more places than are stored: no printed form reads back
    assert MAX_PLACES == 15


def test_times_read_back_to_the_second_in_every_format_and_offset():
    for ts in (COMPILED_AT, COMPILED_AT + 43_199, 1_767_225_599, 1_772_323_200, 1_709_164_800):  # day and year ends
        for fmt in TIME_FORMATS:
            for offset in (-12, -5, 0, 5, 14):
                text = format_time(ts, fmt, offset)
                assert parse_time(text) == ts, (fmt, offset, text)
    assert parse_time("2026-02-30T10:00:00Z") is None and parse_time("10:00 UTC on 3 May 2026") is None
    assert parse_time("x 2026-02-03T10:00:00Z") is None  # a whole slot, never a time somewhere inside text


def test_positions_read_back_exactly():
    for lat, lon in ((56.512, -3.104), (-0.0, 179.999), (-89.5, -0.004), (0.001, 0.0)):
        for fmt in COORD_FORMATS:
            text = format_coords(lat, lon, fmt, 3)
            assert parse_coords(text, 3) == (lat, lon) and "-0" not in text.replace("-0.", ""), text
    assert format_coords(-33.43, -71.712, "hemi", 3) == "33.43°S, 71.712°W"
    assert parse_coords("(56.5, 3.1) extra", 3) is None


# ------------------------------------------------------------------------------- glossary


def test_the_glossary_reads_back_exactly_for_every_domain_and_surface(toys):
    surfaces = [toys.dressed.draw_surface(random.Random(seed)) for seed in range(6)]
    glossaries = [EARTHQUAKES.glossary(), toys.toy.glossary(), toys.flat.glossary()]
    glossaries += [toys.dressed.glossary(surface) for surface in surfaces]
    for glossary in glossaries:
        text = render_glossary(glossary)
        assert text.startswith("<glossary>\n") and text.endswith("\n</glossary>\n")
        assert parse_glossary(f"rules\n\n{text}\n<digest>\n...\n</digest>") == glossary
    assert len({g.fields for g in glossaries[3:]}) > 1  # each instance's own labels and unit words
    for surface in surfaces:
        bound = toys.dressed.bind(surface)
        assert toys.dressed.glossary(surface) == bound.glossary() == bound.glossary(surface)
        assert {f.label for f in bound.glossary().fields} == set(surface["labels"].values())
    toy = toys.toy.glossary()
    assert (toy.noun, toy.id_pattern, toy.statuses, toy.coord_decimals, toys.flat.glossary().coord_decimals) == (
        "reading", toys.toy.id_pattern, ("reviewed", "preliminary"), 3, None)
    temp = toy.fields[0]
    assert (temp.key, temp.qualifier, temp.units) == ("temp", "probe", (Unit("°C", "degrees Celsius"), FAHRENHEIT))
    assert toy.units == {"temp": temp.units, "wave": toy.fields[1].units}


def test_the_glossary_quotes_any_word_and_refuses_what_it_cannot_print():
    units = (Unit("u{2}", "unités, \"deux\""), Unit("v", "v", 2.5, -1.25))
    fields = (GlossaryField("k:1", 'label "q"', "u{2}", units, 2, "how?"), GlossaryField("n", "n", None, (), 0))
    odd = Glossary('rec "x"', "recs; {all}", "ref: id", r'r-\d{2}"\.', "a | b", "dépôt", ("ok (final)", "draft\nnext"),
                   4, fields)
    assert parse_glossary(render_glossary(odd)) == odd
    for word in ("<digest>", "</digest>"):
        with pytest.raises(ValueError, match="cannot be printed"):
            render_glossary(replace(odd, plural=word))
    assert parse_glossary("no glossary here") is None
    with pytest.raises(ValueError, match="unreadable glossary line"):
        parse_glossary(render_glossary(odd).replace("- Sources:", "- Origins:"))


@pytest.mark.parametrize("name", ["earthquakes", "weather", "neo", "invented"])
def test_the_glossary_states_an_id_pattern_that_every_printed_id_matches(name):
    """A domain's ID pattern describes its IDs lower-cased, and digests print IDs as their sources file
    them: a weather observation's in capitals ("WX-482137"), an invented surface's in the letter case it drew
    ("KX/40512"). So the glossary says that the pattern ignores letter case, which makes it true of every
    ID a digest prints, planted ones included; and parse_glossary reads that line back exactly."""
    domain = get_domain(name)
    ids: list[str] = []
    for inst in instances_of(domain, seed=31, n=4):
        glossary = domain.bind(inst.surface).glossary()
        text = render_glossary(glossary)
        assert f"; IDs match the pattern {glossary.id_pattern}, ignoring letter case.\n" in text
        assert parse_glossary(text) == glossary and glossary.id_pattern == domain.id_pattern
        named = ([r.ref for r in inst.reports] + [r.same_as for r in inst.reports if r.same_as]
                 + [n.target for n in inst.notices])
        assert all(ref in rendering.digest for rendering in inst.renderings for ref in named)  # printed as filed
        ids += named
    assert ids and all(re.fullmatch(domain.id_pattern, ref, flags=re.IGNORECASE) for ref in ids)
    cased = [ref for ref in ids if not re.fullmatch(domain.id_pattern, ref)]  # what the bare pattern misses
    assert bool(cased) == (name in ("weather", "invented"))


# ------------------------------------------------------------------------------- traps


@pytest.mark.parametrize("name", ["toy", "toy-flat", "toy-dressed"])
def test_record_traps_are_planted_and_read_as_earthquake_traps_are(toys, name):
    domain = get_domain(name)
    kinds, forms = Counter(), Counter()
    snapshot = build_domain_snapshot(domain, "synthetic", SeedLog(12))[0]
    for inst in instances_of(domain, seed=12, n=5):
        bound = domain.bind(inst.surface)
        by_id = {r.id: r for r in map(bound.dress, snapshot.events)}
        records = [by_id[record_id] for record_id in inst.event_ids]
        assert check_resolution(inst.reports, inst.notices, records, domain=bound) == []
        assert all(isinstance(r, RecordReport) for r in inst.reports)
        kinds.update(t.kind for t in inst.traps)
        forms.update((t.kind, t.form) for t in inst.traps)
        # (a digest off the legacy path holds the traps its plan draws, of each kind none or more: traps.draw_plan)
        assert {t.kind for t in inst.traps} <= {"duplicate", "retraction", "conflicting_value"}
        data = DigestData(inst.reports, inst.notices)
        vocab = bound.trap_vocab
        for trap in inst.traps:
            planted = [r for r in inst.reports if r.trap_id == trap.trap_id]
            readings = [Mishandle.from_dict(m) for m in trap.mishandles]
            if trap.kind == "duplicate":  # a record the digest does not keep, filed as a re-report
                (dup,) = planted
                assert dup.status == "preliminary" and dup.source in {r.source for r in by_id.values()}
                assert (dup.same_as is None) == (trap.form == "same_id")
                if dup.same_as is not None:  # under an ID of its own, a real record's
                    assert dup.ref in by_id and dup.ref not in inst.event_ids
                # Under its record's own ID it reads as a second report does, and records both its readings
                # while the record has no other unsettled report (traps._slot_duplicate).
                prefer = [Mishandle(prefer_preliminary_for=(trap.target,))] if trap.form == "same_id" and sum(
                    r.status == "preliminary" and (r.same_as or r.ref) == trap.target for r in inst.reports) == 1 else []
                assert readings == [*prefer, Mishandle(split=((dup.ref, dup.source, "preliminary"),))]
            elif trap.kind == "retraction":
                (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                assert notice.reason in vocab.retraction_reasons and planted[0].role == "phantom"
                assert readings == [Mishandle(keep_retracted=(notice.target,))] and notice.target == planted[0].ref
            elif trap.form == "correction":
                (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                spec = bound.field(notice.field)
                target = next(r for r in records if r.id == trap.target)
                assert spec.correctable and notice.qualifier == target.qualifiers[spec.qualifier]
                assert planted[0].values[spec.key] == notice.old_value != notice.new_value
                assert readings == [Mishandle(ignore_correction_for=(trap.target,))]
            else:
                (second,) = planted
                assert (second.ref, second.status) == (trap.target, "preliminary")
                assert readings == [Mishandle(prefer_preliminary_for=(trap.target,)),
                                    Mishandle(split=((trap.target, second.source, "preliminary"),))]
        everything = Mishandle.merged(t.mishandles[-1] for t in inst.traps)  # every trap misread at once ...
        for q in inst.questions:  # ... is what the naive reader does, as with earthquakes
            assert (execute(q.chain, data, mishandle=everything, units=bound.units, resolver=bound.resolve)
                    == execute(q.chain, data, skip=("dedupe",), units=bound.units, resolver=bound.resolve))
        for trap in inst.traps:
            for qid, fallen in trap.affected.items():  # each careless reading reproduces its wrong answer
                q = inst.question(qid)
                misread = [execute(q.chain, data, mishandle=Mishandle.from_dict(m), units=bound.units,
                                   resolver=bound.resolve) for m in trap.mishandles]
                assert fallen in misread and not same_answer(q.answer_type, fallen, inst.answer_key[qid], unit=q.unit)
            for qid, answers in trap.fallen.items():  # never exactly the key (traps.score_traps: a joint misreading
                q = inst.question(qid)               # the scorer would still accept is kept, as test_matrix reads it)
                key = normalize(q.answer_type, inst.answer_key[qid])
                assert answers and all(normalize(q.answer_type, a) != key for a in answers)
        for q in inst.questions:  # the key, and the reading rules of the domain applied to the digest
            assert execute(q.chain, data, units=bound.units, resolver=bound.resolve) == inst.answer_key[q.qid]
    assert forms[("duplicate", "same_id")] and forms[("duplicate", "cross_ref")] and kinds["retraction"]
    assert forms[("conflicting_value", "correction")] and forms[("conflicting_value", "second_report")]
    assert builders_for(domain).primary is not builders_for(EARTHQUAKES).primary == primary_report


def test_a_domain_plants_and_settles_with_its_own_statuses_and_words(toys):
    class Verified(toys.module.Toy):
        statuses = ("verified", "provisional")
        trap_vocab = TrapVocab(rereporting_sources=("relay",), duplicate_notes=("copy of",),
                               retraction_reasons=("ledger echo",), retraction_notices=("struck out",),
                               correction_notices=("amended to",))

    domain = domains.REGISTRY.register(Verified("toy-verified", located=False))
    statuses = Counter()
    for inst in instances_of(domain, seed=4, n=3):
        statuses.update(r.status for r in inst.reports)
        for rendering in inst.renderings:
            assert "reviewed" not in rendering.digest and "preliminary" not in rendering.digest
            for word in ('"verified" report', '"provisional"', '"struck out"', '"amended to"'):
                assert word in rendering.prompt.replace("\n   ", " ")
            assert solve(rendering.prompt, question_specs(inst), domain=domain.name) == inst.answer_key
    assert set(statuses) == {"verified", "provisional"}
    assert domain.resolve([RecordReport("by-00001", "a", "provisional", 1, None, None, "g", {"temp": 1.0, "wave": 1.0}),
                           RecordReport("by-00001", "b", "verified", 1, None, None, "g", {"temp": 2.0, "wave": 1.0})],
                          [])[0].values["temp"] == 2.0  # the verified report settles the values


def test_records_must_be_stored_as_digests_print_them(toys):
    good = toys.toy.synthetic(random.Random(1), 3, COMPILED_AT, 24)
    bad = [replace(good[0], lat=56.5123), replace(good[1], qualifiers={"probe": "hull", "colour": "red"}), good[2]]
    problems = "; ".join(record_problems(toys.toy, bad))
    assert f"{good[0].id} is at (56.5123, {good[0].lon}), not to 3 decimal places" in problems
    assert f"{good[1].id} has qualifiers colour, which no field has" in problems
    assert record_problems(toys.toy, good) == []


def test_validation_names_words_a_digest_could_not_tell_apart(toys):
    module = toys.module

    def variant(**attributes):
        return type("Variant", (module.Toy,), attributes)("variant", located=True)

    cases = [
        (variant(fields=(module.TEMPERATURE, replace(module.WAVE, label="water temperature"))),
         "two fields have the same label"),
        (variant(fields=(module.TEMPERATURE, replace(module.WAVE, units=(module.METRE, Unit("km", "kilometres", 1000.0,
                                                                                             decimals=4)))),
                 ), "the unit km is worded both 'kilometres' and 'km'"),
        (variant(statuses=("final", "final")), "are not two different words"),
        (variant(trap_vocab=replace(module.Toy.trap_vocab, correction_notices=())),
         "its trap vocabulary has no correction notices"),
    ]
    for domain, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            domain.validate()


# ---------------------------------------------------------------------- the classic genre

STYLE_MARKERS = {"bulletin": re.compile(r"^\[.+\] .+ \| "), "prose": re.compile(r"^.+? — .+ was recorded .+ Listed by "),
                 "wire": re.compile(r"^.+? WIRE \["), "log": re.compile(r" ref=.+ src=")}
NOTICE_LINE = re.compile(r"^(?P<keyword>NOTICE|RETRACTION|UPDATE|CORRECTION|Revised) \[")


def styles(domain, seed: int):
    """Every combination of line and time presets, with the other presets taken in turn."""
    rng = random.Random(seed)
    for i, (lines, time) in enumerate((a, b) for a in sorted(LINE_PRESETS) for b in sorted(RECORD_TIME_PRESETS)):
        style = {"units": sorted(UNIT_PRESETS)[i % 3], "time": time}
        if domain.has_location:
            style["coords"] = sorted(RECORD_COORD_PRESETS)[i % 3]
        style.update(lines=lines, layout=RECORD_LAYOUTS[i % 3], notices=NOTICE_PLACEMENTS[i % 2],
                     header=i % len(RECORD_HEADERS))
        yield style, random.Random(rng.random())


@pytest.mark.parametrize("name", ["toy", "toy-flat", "toy-dressed"])
def test_classic_reads_back_every_report_and_notice_exactly_in_every_style(toys, name):
    domain = get_domain(name)
    seen_styles, seen_formats = Counter(), Counter()
    for inst in instances_of(domain, seed=27, n=3):
        bound = domain.bind(inst.surface)
        glossary = bound.glossary()
        reports, notices = printed(inst.reports, inst.notices)
        for style, rng in styles(bound, inst.seed):
            ctx = RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                rng, bound, glossary)
            text = CLASSIC.render(ctx)
            data = CLASSIC.parse(text, glossary, bound)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style
            assert CLASSIC.sniff(text) and detect_genre(text) is CLASSIC
            for line in text.split("\n"):
                seen_styles.update(s for s, marker in STYLE_MARKERS.items() if marker.search(line))
                seen_formats.update(fmt for fmt in TIME_FORMATS if any(
                    parse_time(piece) is not None for piece in _time_slots(line, fmt)))
            for chatter in inst.chatter:  # lines without data change nothing
                assert CLASSIC.parse(chatter, glossary, bound) == DigestData([], [])
    assert set(seen_styles) == set(STYLE_MARKERS)
    assert set(seen_formats) == set(TIME_FORMATS)


@pytest.mark.parametrize("ids_name_place", [False, True])
def test_place_notes_and_missing_qualifiers_read_back_and_named_sites_stay_put(toys, ids_name_place):
    """A re-report stands where a record that the digest does not keep stands (its slot); one filed under
    its record's own ID stands at its record's named site only where the domain's IDs name it
    (Domain.ids_name_place), so that it is not pinned there otherwise (traps._slot_pair)."""
    class Placed(toys.module.Toy):
        """Toy readings, two in three at a named buoy (a place note with a comma and parentheses), and
        one in five without the probe that took its temperature."""

        def synthetic(self, rng, n, compiled_at, window_hours):
            records = super().synthetic(rng, n, compiled_at, window_hours)
            return [replace(r, place=f"{r.group} buoy {i % 7}, (moored)" if i % 3 else None,
                            qualifiers={} if i % 5 == 0 else r.qualifiers) for i, r in enumerate(records)]

    domain = Placed("toy-placed", located=True)
    domain.ids_name_place = ids_name_place
    domain = domains.REGISTRY.register(domain)
    sites, moved = 0, 0
    for inst in instances_of(domain, seed=6, n=4):
        glossary = domain.glossary()
        reports, notices = printed(inst.reports, inst.notices)
        assert any(r.place for r in reports) and any(not r.qualifiers for r in reports)
        for style, rng in styles(domain, inst.seed):
            text = CLASSIC.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at,
                                                inst.window_hours, style, rng, domain, glossary))
            data = CLASSIC.parse(text, glossary, domain)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style
        for rendering in inst.renderings:
            assert solve(rendering.prompt, question_specs(inst), domain=domain.name) == inst.answer_key
        settled = {r.ref: r for r in inst.reports if r.role == "primary"}
        snapshot = {(r.time, r.source, r.lat, r.lon, r.place) for r in build_domain_snapshot(
            domain, "synthetic", SeedLog(6))[0].events}
        for r in inst.reports:  # a re-report under its record's ID stands where a real record stands (a record
            # the digest does not keep): at its record's named site, where the IDs name it
            anchor = settled.get(r.same_as or r.ref)
            if r.role == "duplicate" and anchor is not None and anchor.place is not None and r.same_as is None:
                assert (r.time, r.source, r.lat, r.lon, r.place) in snapshot
                assert r.place == anchor.place or not ids_name_place
                sites += 1
                moved += r.place != anchor.place
    assert sites > 0 and (moved == 0) == ids_name_place


def _time_slots(line: str, fmt: str) -> list[str]:
    """The pieces of a line that a time in `fmt` could fill (enough to tell which formats a digest uses)."""
    pattern = {"iso": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "iso_space": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC",
               "rfc": r"\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} \+0000",
               "prose_utc": r"\d{2}:\d{2}:\d{2} UTC on \d{1,2} [A-Z][a-z]+ \d{4}",
               "iso_offset": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:00",
               "local12": r"\d{1,2}:\d{2}:\d{2} [AP]M local time \(UTC[+-]\d{2}:00\) on \w{3} \d{1,2}, \d{4}",
               "local24": r"\d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:00\) \d{1,2} \w{3} \d{4}"}[fmt]
    return re.findall(pattern, line)


def test_every_notice_wording_is_named_by_the_reading_rules(toys):
    """As for earthquakes: whatever wording and phrase a notice is drawn with, the rules name it."""
    for domain in (toys.toy, toys.dressed.bind(toys.dressed.draw_surface(random.Random(1)))):
        rules = " ".join(record_rules(domain).split()).lower()
        vocab = domain.trap_vocab
        words = {"retraction": ("retraction", "was not a real", *vocab.retraction_notices),
                 "correction": ("correction", "revised", *vocab.correction_notices)}
        for kind, phrases in words.items():
            assert all(phrase.lower() in rules for phrase in phrases), (kind, phrases)
        glossary = domain.glossary()
        record = toys.flat.synthetic(random.Random(2), 1, COMPILED_AT, 24)[0]
        notices = [Notice("retraction", record.id, COMPILED_AT - 99, reason="probe fault"),
                   Notice("correction", record.id, COMPILED_AT - 9, field="temp", new_value=1.5, old_value=-2.0,
                          qualifier="spar")]
        wordings = set()
        for seed in range(150):
            ctx = RenderContext([], notices, [], COMPILED_AT, 24, CLASSIC.choose_style(random.Random(seed), domain),
                                random.Random(seed), domain, glossary)
            wordings.update(line for line in CLASSIC.render(ctx).split("\n") if NOTICE_LINE.match(line))
        forms = set()
        for line in wordings:
            (notice,) = CLASSIC.parse(line, glossary, domain).notices  # each wording reads back on its own
            assert replace(notice, time=0) in [replace(n, time=0) for n in notices], line
            assert any(phrase.lower() in line.lower() for phrase in words[notice.kind]), line
            if notice.kind == "correction":  # every correction wording gives its value after a quoted phrase
                assert any(f" {phrase} " in line for phrase in vocab.correction_notices), line
            forms.add((notice.kind, NOTICE_LINE.match(line)["keyword"]))
        assert forms == {("retraction", "NOTICE"), ("retraction", "RETRACTION"), ("retraction", "UPDATE"),
                         ("correction", "CORRECTION"), ("correction", "UPDATE"), ("correction", "Revised")}
        assert len(wordings) >= 20


def test_classic_refuses_a_rendering_it_cannot_read_back(toys):
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    style = {"units": "mixed", "time": "mixed", "coords": "mixed", "lines": "bulletin", "layout": "by_time",
             "notices": "end", "header": 0}
    awkward = [replace(r, group="North | Sea (St. Kilda: west) @ 56°N") for r in inst.reports]  # printable: the
    CLASSIC.render(RenderContext(awkward, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                 random.Random(0), toys.toy, toys.toy.glossary()))  # patterns find the real slots
    split = [replace(r, group="North\nSea") for r in inst.reports]  # no line can hold a line break
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        CLASSIC.render(RenderContext(split, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                     random.Random(0), toys.toy, toys.toy.glossary()))
    with pytest.raises(ValueError, match="prints none"):
        CLASSIC.parse(inst.renderings[0].digest, None, toys.toy)


def test_no_other_genre_claims_a_record_digest(toys):
    inst = instances_of(toys.flat, seed=8, n=1)[0]
    for rendering in inst.renderings:
        assert [g.name for g in all_genres() if g.sniff(rendering.digest)] == ["classic"]
        assert rendering.style["header"] in range(len(RECORD_HEADERS)) and "coords" not in rendering.style
    assert inst.renderings[1].style != inst.renderings[0].style  # a variant in the same genre avoids the original's


# ------------------------------------------------------------------------- prompts and oracle


def test_non_legacy_prompts_word_every_answer_type_from_its_kind(toys):
    domain = toys.toy
    specs = {t.name: domain_answer_spec(t.name, t.kind == "number", domain) for t in domain.answer_types()}
    assert specs == {"toy_temperature": "a number in {unit}, rounded as the question says",
                     "toy_wave_height": "a number in {unit}, rounded as the question says",
                     "toy_reading_id": "one reading ID, as a string",
                     "toy_area": "one sea area exactly as written in the digest, or \"tie\""}
    assert [domain_answer_spec(t, False, domain) for t in ("count", "utc_time", "probability", "toy_temperature")] == [
        "an integer", "a UTC time string \"HH:MM\" (24-hour clock; seconds are dropped, not rounded)",
        "a probability between 0 and 1", "a number"]
    assert [domain_placeholder(t, domain) for t in ("toy_reading_id", "toy_area", "distance")] == [
        "\"<reading ID>\"", "\"<sea area or tie>\"", "<number>"]
    assert domain_unit_words(domain) == {"°C": "degrees Celsius", "°F": "degrees Fahrenheit", "m": "metres",
                                         "ft": "feet", "km": "km", "mi": "miles"}
    assert "km" not in domain_unit_words(toys.flat)
    braced = copy.copy(domain)
    braced.id_label = "ref {x}"  # words stand for themselves in a description
    assert domain_answer_spec("toy_reading_id", False, braced).format(unit="") == "one ref {x}, as a string"
    with pytest.raises(KeyError, match="unknown answer type 'colour'"):
        domain_answer_spec("colour", False, domain)


def test_the_text_only_oracle_reads_any_genre_through_the_prompts_glossary(toys):
    inst = instances_of(toys.dressed, seed=9, n=1)[0]
    bound = toys.dressed.bind(inst.surface)
    for rendering in inst.renderings:
        assert parse_glossary(rendering.prompt) == bound.glossary()  # the instance's own words
        assert solve(rendering.prompt, question_specs(inst), domain=inst.domain) == inst.answer_key
        assert solve(rendering.prompt, question_specs(inst), domain=inst.domain, genre="classic") == inst.answer_key
        unglossed = re.sub(r"<glossary>.*?</glossary>\n\n", "", rendering.prompt, flags=re.DOTALL)
        with pytest.raises(ValueError, match="prints none"):  # no other way to read a drawn surface
            solve(unglossed, question_specs(inst), domain=inst.domain)


# ----------------------------------------------------------------------------- whole runs


@pytest.fixture(scope="module")
def toy_runs(tmp_path_factory):
    """generate (mix "<toy>:classic") -> run (oracle, naive, mock) -> score -> report, for four instances of
    the toy domain with a position, of toy-flat without one, and of toy-dressed with a surface per instance."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        toy = domains.load_plugin(TOY_PLUGIN)
        module = sys.modules[type(toy).__module__]
        for variant in (module.FLAT, module.DRESSED):
            domains.REGISTRY.register(variant)
        runs = {}
        for name, seed in (("toy", 31), ("toy-flat", 32), ("toy-dressed", 33)):
            run_dir = pipeline.generate(GenerateConfig(seed=seed, instances=4, mix=f"{name}:classic"),
                                        tmp_path_factory.mktemp("runs") / name, log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[name] = run_dir
        yield runs


def test_record_runs_are_generated_with_a_glossary_in_every_prompt(toy_runs, tmp_path):
    for name, run in toy_runs.items():
        manifest = read_json(run / "manifest.json")
        assert manifest["config"]["mix"] == f"{name}:classic" and manifest["snapshot"]["domain"] == name
        check = manifest["self_check"]
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        for inst in load_instances(run):
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["classic"] * 3
            bound = get_domain(name).bind(inst.surface)
            for rendering in inst.renderings:
                prompt = rendering.prompt
                assert prompt.count("<glossary>") == 1 and parse_glossary(prompt) == bound.glossary()
                assert prompt.index(bound.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
                assert extract_digest(prompt) == rendering.digest and "<glossary>" not in rendering.digest
                data = CLASSIC.parse(rendering.digest, bound.glossary(), bound)
                assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_the_oracle_scores_100_percent_and_reproduces_every_key_exactly(toy_runs):
    for name, run in toy_runs.items():
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "classic")}
        for inst in load_instances(run):
            for rendering in inst.renderings:  # read by sniffing, as the generator's self-check reads it
                assert solve(rendering.prompt, question_specs(inst), domain=name) == inst.answer_key


def test_the_naive_reader_falls_for_the_traps_and_only_for_them(toy_runs):
    for name, run in toy_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        # (every kind a run's digests hold, in this order: a run of a few digests may hold no retraction)
        assert {"duplicate", "conflicting_value"} <= set(naive["by_kind"])
        assert list(naive["by_kind"]) == [k for k in ("duplicate", "retraction", "conflicting_value")
                                          if k in naive["by_kind"]]
        assert models["naive"]["accuracy"]["rate"] < 1.0 == models["oracle"]["accuracy"]["rate"]
        instances = load_instances(run)
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], row


def test_the_mocks_mistakes_on_record_types_are_real_mistakes(toy_runs, answer_registry):
    """The simulated model errs on some of a toy run's questions and not on others, over every rendering (its
    accuracy, of the 15 or so original renderings' questions alone, came out 100% by chance once the traps'
    placement changed what toy-dressed's prompts print: its errors are seeded by the prompt text), and every
    wrong answer it gives is wrong for the scorer."""
    for name, run in toy_runs.items():
        mock = {m["model"]: m for m in read_json(run / "report.json")["models"]}["mock"]
        rows = read_json(run / "scores" / "mock.json")["rows"]
        assert 0 < sum(row["correct"] for row in rows) < len(rows) and mock["format"]["replies"] == 12
        for inst in load_instances(run):
            task = Task(inst.instance_id, "v1", question_specs(inst), hidden=inst, domain=name, genre="classic")
            for q in question_specs(inst):
                for seed in range(15):
                    key = inst.answer_key[q["qid"]]
                    wrong = _perturb(q, key, random.Random(seed), task)
                    assert not is_correct(q["answer_type"], key, wrong, unit=q["unit"]), (q, key, wrong)
    # A number in a unit of another scale is missed by a share of itself, not by a fixed amount.
    register_answer_type(AnswerType("miss_distance", "number", (0.0, 0.01)))
    q = {"answer_type": "miss_distance", "unit": "au", "chain": [{"op": "round", "ndigits": 6}]}
    for seed in range(30):
        wrong = _perturb(q, 0.021294, random.Random(seed), Task("i", "v0", []))
        assert 0.4 * 0.021294 < wrong < 1.6 * 0.021294 and not is_correct("miss_distance", 0.021294, wrong)


def test_record_runs_report_their_domain_and_traps(toy_runs):
    for name, run in toy_runs.items():
        data = read_json(run / "report.json")
        assert [m["model"] for m in data["models"]] == ["oracle", "naive", "mock"]
        kinds = list(data["run"]["traps"]["by_kind"])  # (a run of a few digests may hold no retraction)
        assert {"duplicate", "conflicting_value"} <= set(kinds)
        assert kinds == [k for k in ("duplicate", "retraction", "conflicting_value") if k in kinds]
        plural = get_domain(name).bind(load_instances(run)[0].surface).plural
        markdown = (run / "report.md").read_text(encoding="utf-8")
        assert "conflicting value" in markdown and ("readings" in markdown or plural in markdown)
        assert "separate quake" not in markdown and ("separate reading" in markdown) == (name != "toy-dressed")
