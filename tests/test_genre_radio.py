"""The radio genre (evalgen/genres/radio.py): a digest as the transcript of a radio reporting net, for
earthquakes and for the test-only toy domains (with and without a position, and with a surface drawn
per instance). Every report and notice reads back exactly in every style; readbacks repeat reports
and are never new ones; transmission times, procedure and announcements change nothing; every
broadcast is named by the domain's reading rules; sniffing tells transcripts from classic digests and
other formats; and whole runs (mix "toy:radio" and "earthquakes:radio", synthetic and a frozen USGS
feed) are read by the text-only oracle at 100%, with the naive reader falling for the traps and only
for them."""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from dataclasses import replace

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.domains import get_domain
from evalgen.domains.base import RecordReport
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import detect_genre
from evalgen.genres import radio
from evalgen.genres.base import RenderContext, parse_glossary
from evalgen.genres.classic import CLASSIC, RECORD_COORD_PRESETS, RECORD_TIME_PRESETS, UNIT_PRESETS
from evalgen.genres.radio import RADIO
from evalgen.models import Instance, Notice, Report
from evalgen.ops import DigestData
from evalgen.pipeline import GenerateConfig, build_instance, question_specs
from evalgen.questions import Context, compose_new_templates
from evalgen.report import load_instances
from evalgen.rules import READING_RULES
from evalgen.seeds import SeedLog
from evalgen.solver import extract_digest, solve
from evalgen.sources import build_domain_snapshot
from evalgen.util import read_json, write_json

from conftest import TOY_PLUGIN, quiet
from test_usgs_forecast import fake_feed

DOMAINS = ("earthquakes", "toy", "toy-flat", "toy-dressed")
DIMENSIONS = {  # every preset of every dimension of a radio style
    "units": sorted(UNIT_PRESETS), "unit_names": sorted(radio.UNIT_NAMES), "time": sorted(RECORD_TIME_PRESETS),
    "coords": sorted(RECORD_COORD_PRESETS), "separator": sorted(radio.SEPARATORS), "stamps": list(radio.STAMPS),
    "calls": sorted(radio.CALL_SIGNS), "control": list(radio.CONTROL_SIGNS), "readbacks": sorted(radio.READBACKS),
    "layout": list(radio.LAYOUTS), "notices": list(radio.NOTICE_PLACEMENTS), "header": list(range(len(radio.HEADERS))),
}
# A readback, and a report as its station reads it out, with the readout itself in the group "body".
READBACK = re.compile(r", (?:I read back|reading back|read back follows): (?P<body>.+?)"
                      r"(?:\. Over\.|\. Is that correct\? Over\.)$")
REPORT = re.compile(r", (?:report follows|.+ report follows|message follows): (?P<body>.+?)"
                    r"(?:\. Over\.|, over\.|\. How copy\? Over\.)$")


def instances_of(domain, seed: int, n: int = 2) -> list[Instance]:
    """The first instances of a run about `domain` in the radio genre, built as the pipeline builds them
    (each of their renderings is checked to read back exactly as it is written)."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    return [build_instance(i, snapshot, new, log, GenerateConfig(seed=seed), ctx, used, usage, domain=domain,
                           genre_plan=(RADIO,) * 3) for i in range(n)]


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


def styles(domain):
    """Twelve styles that take every preset of every dimension in turn, each dimension out of step with
    the others."""
    for i in range(12):
        style = {key: values[(i + k) % len(values)] for k, (key, values) in enumerate(DIMENSIONS.items())}
        if not domain.has_location:
            del style["coords"]
        yield style


def render(inst: Instance, domain, style, seed: int) -> str:
    return RADIO.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                      style, random.Random(seed), domain, domain.glossary()))


# ---------------------------------------------------------------------------- round trips


@pytest.mark.parametrize("name", DOMAINS)
def test_radio_reads_back_every_report_and_notice_exactly_in_every_style(toys, name):
    domain = get_domain(name)
    seen, forms = Counter(), set()
    for seed in (0, 1, 2):
        for inst in instances_of(domain, seed):
            bound = domain.bind(inst.surface)
            glossary = bound.glossary()
            reports, notices = printed(inst.reports, inst.notices)
            assert all(isinstance(r, Report if name == "earthquakes" else RecordReport) for r in reports)
            forms.update(t.form for t in inst.traps)
            for rendering in inst.renderings:  # the styles drawn for the instance
                data = RADIO.parse(rendering.digest, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices)
                assert rendering.genre == "radio" and detect_genre(rendering.digest) is RADIO
            for k, style in enumerate(styles(bound)):  # and every preset of every dimension
                text = render(inst, bound, style, seed * 100 + k)
                data = RADIO.parse(text, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), style
                assert RADIO.sniff(text) and not CLASSIC.sniff(text) and detect_genre(text) is RADIO
                seen.update(style.items())
    assert all(seen[(key, value)] for key, values in DIMENSIONS.items() for value in values
               if key != "coords" or domain.has_location)
    # re-reports under the same ID and with a note naming it, retracted phantoms, corrected values and
    # second reports all read back
    assert forms == {"same_id", "cross_ref", "phantom", "correction", "second_report"}


def test_drawn_styles_take_every_preset_and_a_variant_avoids_the_originals(toys):
    for domain in (EARTHQUAKES, toys.flat):
        seen = Counter()
        for seed in range(80):
            rng = random.Random(seed)
            original = RADIO.choose_style(rng, domain)
            variant = RADIO.choose_style(rng, domain, avoid=original)
            assert set(original) == set(variant) == set(DIMENSIONS) - (set() if domain.has_location else {"coords"})
            assert all(variant[key] != original[key] for key in original), (original, variant)
            seen.update(original.items())
        assert all(seen[(key, value)] for key in original for value in DIMENSIONS[key])


def test_awkward_words_read_back_and_unprintable_ones_are_refused(toys):
    """Free words that hold a separator or a delimiter of the readout (a region "Honshu, Japan", a place
    with commas and parentheses) still read back: the readout takes another separator, and the patterns
    find the real slots. A word that no line can hold is refused, never misread."""
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    glossary = toys.toy.glossary()
    awkward = [replace(r, group="North; Sea / Dogger, (west) @ 56°N", place="buoy 3, (moored); time of reading x")
               for r in inst.reports]
    for k, style in enumerate(styles(toys.toy)):
        text = RADIO.render(RenderContext(awkward, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                          style, random.Random(k), toys.toy, glossary))
        assert same_items(RADIO.parse(text, glossary, toys.toy).reports, printed(awkward, [])[0])
    split = [replace(r, group="North\nSea") for r in inst.reports]
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        RADIO.render(RenderContext(split, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                   next(styles(toys.toy)), random.Random(0), toys.toy, glossary))
    dressed = instances_of(toys.dressed, seed=6, n=1)[0]
    with pytest.raises(ValueError, match="prints none"):  # drawn words can only be read from the prompt
        RADIO.parse(dressed.renderings[0].digest, None, toys.dressed.bind(dressed.surface))
    flat = instances_of(toys.flat, seed=7, n=1)[0]  # a domain's own words are what its prompt prints
    assert RADIO.parse(flat.renderings[0].digest, None, toys.flat) == RADIO.parse(flat.renderings[0].digest,
                                                                                    toys.flat.glossary(), toys.flat)


# ------------------------------------------------------------------- what is not a report


def test_readbacks_repeat_reports_word_for_word_and_are_never_new_reports(toys):
    for domain in (EARTHQUAKES, toys.toy):
        for inst in instances_of(domain, seed=8):
            glossary = domain.glossary()
            for k, style in enumerate(styles(domain)):
                text = render(inst, domain, style, k)
                lines = text.split("\n")
                bodies = [m["body"] for m in map(REPORT.search, lines) if m]
                readbacks = [i for i, line in enumerate(lines) if READBACK.search(line)]
                assert len(bodies) == len(inst.reports)
                if style["readbacks"] == "always":
                    assert len(readbacks) == len(bodies)
                for i in readbacks:  # control repeats a report just passed, word for word
                    body = READBACK.search(lines[i])["body"]
                    assert body in bodies and any(body == m["body"] for m in map(REPORT.search, lines[i - 3:i]) if m)
                data = RADIO.parse(text, glossary, domain)
                without = "\n".join(line for i, line in enumerate(lines) if i not in readbacks)
                assert RADIO.parse(without, glossary, domain) == data  # readbacks add nothing ...
                doubled = "\n".join(lines + [lines[i] for i in readbacks])
                assert RADIO.parse(doubled, glossary, domain) == data  # ... however often they come
                only = "\n".join(line for line in lines if not REPORT.search(line))
                assert RADIO.parse(only, glossary, domain).reports == []  # a readback alone is no report


def test_transmission_times_procedure_and_announcements_change_nothing(toys):
    for domain in (EARTHQUAKES, toys.toy, toys.flat):
        inst = instances_of(domain, seed=9, n=1)[0]
        glossary = domain.glossary()
        for k, style in enumerate(styles(domain)):
            text = render(inst, domain, style, k)
            data = RADIO.parse(text, glossary, domain)
            lines = text.split("\n")
            margin = re.compile(r"^(?:\d{2}:\d{2}:\d{2}|\d{4}Z) ")
            if style["stamps"] != "none":
                assert radio.STAMP_LINE in lines[:3] and all(margin.match(line) for line in lines[4:])
            for shifted in ("23:59:59 ", "0000Z ", ""):  # a message's time is never a record's time
                assert RADIO.parse("\n".join(margin.sub(shifted, line) for line in lines), glossary, domain) == data
            ids = {r.ref for r in inst.reports} | {n.target for n in inst.notices}
            carrying = [line for line in lines if any(ref in line for ref in ids)]
            assert len(carrying) < len(lines) - 3  # call-ups, acknowledgements, radio checks, announcements ...
            assert RADIO.parse("\n".join(carrying), glossary, domain) == data  # ... carry no data
            for line in lines:
                if "Information only:" in line or "For information:" in line or "Net announcement:" in line:
                    assert RADIO.parse(line, glossary, domain) == DigestData([], [])
            for chatter in inst.chatter:
                assert RADIO.parse(chatter, glossary, domain) == DigestData([], [])
                assert chatter in text  # announced by control


# The words the 0.2.0 rules use to identify each kind of notice (as tests/test_reading_rules.py).
RULE_WORDS = {"retraction": ("retraction", "retracted", "withdrawn"), "correction": ("correction", "revised")}
FORMS = {"retraction": ("; disregard it", "; strike it from the log", "RETRACTION, time of issue"),
         "correction": (" is revised to ", " now reads ", " amended from ")}


def test_every_broadcast_is_named_by_the_reading_rules_and_reads_back_alone(toys):
    """Whatever wording and phrase a broadcast is drawn with, the domain's reading rules name it: the
    0.2.0 rules for earthquakes (their prompts keep them), the genre-neutral rules for any other domain."""
    dressed = toys.dressed.bind(toys.dressed.draw_surface(random.Random(1)))
    for domain in (EARTHQUAKES, toys.toy, dressed):
        rules = " ".join(domain.rules_text().split()).lower()
        if domain is EARTHQUAKES:
            assert domain.rules_text() is READING_RULES
            words = RULE_WORDS
            notices = [Notice("retraction", "us7000abcd", 1_780_000_000, reason="quarry blast"),
                       Notice("correction", "us7000abcd", 1_780_000_900, new_mag=4.2, old_mag=4.57, mag_type="mb")]
        else:
            vocab = domain.trap_vocab
            words = {"retraction": ("retraction", "was not a real", *map(str.lower, vocab.retraction_notices)),
                     "correction": ("correction", "revised", *map(str.lower, vocab.correction_notices))}
            ref = f"{domain.id_prefix}12345"
            notices = [Notice("retraction", ref, 1_780_000_000, reason="probe fault"),
                       Notice("correction", ref, 1_780_000_900, field="temp", new_value=1.5, old_value=-2.0,
                              qualifier="spar")]
        assert all(word in rules for phrases in words.values() for word in phrases)
        glossary, forms, broadcasts = domain.glossary(), set(), set()
        for seed in range(60):
            rng = random.Random(seed)
            ctx = RenderContext([], notices, [], 1_780_003_600, 24, RADIO.choose_style(rng, domain), rng, domain,
                                glossary)
            broadcasts.update(line for line in RADIO.render(ctx).split("\n") if "All stations, all stations" in line
                              and "net is open" not in line)
        for line in broadcasts:
            (notice,) = RADIO.parse(line, glossary, domain).notices  # each broadcast reads back on its own
            assert notice in notices, line
            assert any(word in line.lower() for word in words[notice.kind]), line
            forms.update((notice.kind, form) for form in FORMS[notice.kind] if form in line)
        assert forms == {(kind, form) for kind, wordings in FORMS.items() for form in wordings}


# ----------------------------------------------------------------------------- sniffing

OTHER_FORMATS = (
    "",
    "id,time,group,value\nby-12345,2026-04-12T06:12:19Z,North Sea,12.3\n",
    "| ID | time | value |\n|---|---|---|\n| by-12345 | 2026-04-12T06:12:19Z | 12.3 |\n",
    "From: ops@example.org\nTo: desk@example.org\nSubject: readings\n\nHello,\nreading by-12345 is in, over.\n",
    "[06:12] kim: reading by-12345 is in, over.\n[06:13] lee: copy, out.\n",
    "BUOYS REPORT ROUGH SEAS\nABERDEEN, 12 April (Wire) - Operators reported ... Control out.\n",
    "RADIO TRANSCRIPT: READINGS NET ON CHANNEL 3\nNo traffic was heard.\n",  # a title and no transmission
)


def test_sniff_rejects_classic_digests_and_other_formats(toys, instances):
    for inst in instances[:3]:  # legacy (0.2.0) digests
        for rendering in inst.renderings:
            assert not RADIO.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for domain in (toys.toy, toys.dressed, EARTHQUAKES):  # classic digests of records, and of earthquakes off the
        log = SeedLog(2)                                    # legacy path
        snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
        new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
        inst = build_instance(0, snapshot, new, log, GenerateConfig(seed=2), Context(snapshot.compiled_at,
                              snapshot.window_hours), set(), {}, domain=domain, genre_plan=(CLASSIC,) * 3)
        for rendering in inst.renderings:
            assert not RADIO.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for text in OTHER_FORMATS:
        assert not RADIO.sniff(text), text
    transcript = instances_of(toys.toy, seed=10, n=1)[0].renderings[0].digest
    assert RADIO.sniff(transcript) and RADIO.sniff("\n\n" + transcript)
    assert not RADIO.sniff("\n".join("> " + line for line in transcript.split("\n")))  # quoted in an email


# --------------------------------------------------------------- earthquakes and their types


def test_earthquake_reports_and_notices_map_to_the_generic_form_and_back(instances, toys):
    kinds = Counter()
    for inst in instances:
        for item in [*inst.reports, *inst.notices]:
            generic = EARTHQUAKES.to_generic(item)
            assert EARTHQUAKES.from_generic(generic) == item
            if isinstance(item, Report):
                assert isinstance(generic, RecordReport) and generic.values == {"mag": item.mag, "depth": item.depth_km}
                assert (generic.source, generic.group, generic.qualifiers) == (item.network, item.region,
                                                                                {"mag_type": item.mag_type})
            elif item.kind == "correction":
                assert (generic.field, generic.new_value, generic.old_value, generic.qualifier) == (
                    "mag", item.new_mag, item.old_mag, item.mag_type) and generic.new_mag is None
            kinds[type(item).__name__ if isinstance(item, Report) else item.kind] += 1
    assert kinds["Report"] and kinds["retraction"] and kinds["correction"]
    record = instances_of(toys.toy, seed=11, n=1)[0]
    for item in [*record.reports, *record.notices]:  # a Record domain's are generic already
        assert toys.toy.to_generic(item) is item and toys.toy.from_generic(item) is item
    with pytest.raises(ValueError, match="not an earthquake report"):
        EARTHQUAKES.from_generic(record.reports[0])


# ----------------------------------------------------------------------------- whole runs


@pytest.fixture(scope="module")
def radio_runs(tmp_path_factory):
    """generate (mix "<domain>:radio") -> run (oracle, naive, mock) -> score -> report, for earthquakes
    (synthetic, and a frozen USGS feed with place notes and forecasts) and the three toy domains."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        toy = domains.load_plugin(TOY_PLUGIN)
        module = sys.modules[type(toy).__module__]
        for variant in (module.FLAT, module.DRESSED):
            domains.REGISTRY.register(variant)
        root = tmp_path_factory.mktemp("radio-runs")
        feed = root / "all_day.geojson"
        feed.write_bytes(fake_feed()[0])
        configs = {name: GenerateConfig(seed=60 + i, instances=3, mix=f"{name}:radio")
                   for i, name in enumerate(DOMAINS)}
        configs["usgs"] = GenerateConfig(seed=11, instances=3, mix="earthquakes:radio", source="usgs",
                                         snapshot_file=str(feed))
        runs = {}
        for label, cfg in configs.items():
            run_dir = pipeline.generate(cfg, root / label, log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[label] = run_dir
        yield runs


def test_radio_runs_print_a_glossary_and_read_back_exactly(radio_runs, tmp_path):
    for label, run in radio_runs.items():
        manifest = read_json(run / "manifest.json")
        check = manifest["self_check"]
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        name = "earthquakes" if label == "usgs" else label
        assert manifest["config"]["mix"] == f"{name}:radio" and manifest["snapshot"]["domain"] == name
        for inst in load_instances(run):
            bound = get_domain(name).bind(inst.surface)
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["radio"] * 3
            assert inst.renderings[1].style != inst.renderings[0].style != inst.renderings[2].style
            for rendering in inst.renderings:
                prompt, digest = rendering.prompt, rendering.digest
                assert parse_glossary(prompt) == bound.glossary() and extract_digest(prompt) == digest
                assert prompt.index(bound.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
                data = RADIO.parse(digest, bound.glossary(), bound)
                assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
                assert detect_genre(digest) is RADIO
                if label == "usgs":
                    assert "F1. Forecast" in prompt and "location " in digest
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()
        if name != "earthquakes":  # the genre-neutral rules name readbacks and message times
            rules = " ".join(get_domain(name).rules_text().split())
            assert "read-back repeat" in rules and "The time at which a message was sent" in rules


def test_the_oracle_scores_100_percent_and_reproduces_every_key(radio_runs):
    for label, run in radio_runs.items():
        name = "earthquakes" if label == "usgs" else label
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "radio")}
        for inst in load_instances(run):
            for rendering in inst.renderings:  # read by sniffing, and with the genre named
                for genre in (None, "radio"):
                    assert solve(rendering.prompt, question_specs(inst), domain=name, genre=genre) == inst.answer_key
        if label == "usgs":
            forecasts = read_json(run / "forecasts.json")
            assert len(forecasts) == 3 and all(f["status"] == "pending" for f in forecasts)


def test_the_naive_reader_falls_for_the_traps_and_only_for_them(radio_runs):
    for label, run in radio_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        assert models["naive"]["accuracy"]["rate"] < 1.0 == models["oracle"]["accuracy"]["rate"]
        assert models["mock"]["format"]["replies"] == 9  # the simulated model reads every transcript too
        instances = load_instances(run)
        kinds = {t.kind for inst in instances for t in inst.traps}
        conflict = "conflicting_magnitude" if label in ("earthquakes", "usgs") else "conflicting_value"
        assert kinds == {"duplicate", "retraction", conflict}
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], row
