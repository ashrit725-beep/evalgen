"""The form genre (evalgen/genres/form.py): one field or incident report form per report and one
amendment form per notice, for earthquakes and for the test-only toy domains of toy_domain.py (loaded as
a plugin), with and without a position and with a surface drawn per instance. Every rendering reads back
to exactly its reports and notices in every style; the genre recognises its own digests and no others;
a form's own noise (serial numbers, codes, message times, blank fields) carries no data; amendments are
notices the reading rules name; and whole runs give the oracle 100% and the naive reader its traps."""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from dataclasses import replace

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.domains import get_domain
from evalgen.domains.base import RecordReport, record_rules
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import all_genres, detect_genre
from evalgen.genres import form as form_module
from evalgen.genres.base import RenderContext, id_pattern, parse_glossary, parse_time
from evalgen.genres.classic import CLASSIC
from evalgen.genres.form import (AMENDMENTS_HEAD, BLANKS, CHECKBOXES, COORD_PRESETS, FORM, FORM_PRESETS, FRAME_NAMES,
                                 FRAMES, FRAMES_BY_NAME, HEADERS, LAYOUTS, NOTICE_PLACEMENTS, SECTION_HEADS,
                                 STATUS_ORDERS, TIME_PRESETS, UNIT_PRESETS)
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

DOMAINS = ("toy", "toy-flat", "toy-dressed", "earthquakes")
# How each frame's heading lines begin (inside a box for the boxed frame).
HEADINGS = {"boxed": re.compile(r"^│ (FIELD REPORT FORM|AMENDMENT FORM)"),
            "ruled": re.compile(r"^(INCIDENT REPORT — FORM|AMENDMENT — FORM)"),
            "numbered": re.compile(r"^(REPORT FORM|AMENDMENT FORM).* · no\. "),
            "dotted": re.compile(r"^(OBSERVATION REPORT FORM|AMENDMENT FORM).*   Page 1 of 1$")}
# A form's own time: when it was sent, relayed or forwarded (never its record's time).
MESSAGE_TIME = re.compile(r"(?:Form sent|Sent to desk|Relayed to desk|Forwarded)(?:: +| \.+ )(?P<time>.+?)(?: +│)?$",
                          re.MULTILINE)


def instances_of(domain, seed: int, n: int = 3) -> list[Instance]:
    """The first instances of a run about `domain` in the form genre, built as the pipeline builds them."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    return [build_instance(i, snapshot, new, log, GenerateConfig(seed=seed), ctx, used, usage, domain=domain,
                           genre_plan=(FORM,) * 3) for i in range(n)]


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


def style_options(domain) -> dict[str, set]:
    """Every value that each key of a style can take for `domain`'s records."""
    options = {"forms": set(FORM_PRESETS), "units": set(UNIT_PRESETS), "time": set(TIME_PRESETS),
               "layout": set(LAYOUTS), "boxes": set(CHECKBOXES), "blank": set(BLANKS), "statuses": set(STATUS_ORDERS),
               "notices": set(NOTICE_PLACEMENTS), "header": set(range(len(HEADERS)))}
    return {**options, "coords": set(COORD_PRESETS)} if domain.has_location else options


def styles(domain, seed: int):
    """Every frame preset with every time preset, the other dimensions taken in turn, so that each value
    of every dimension is drawn."""
    rng = random.Random(seed)
    for i, (forms, time) in enumerate((a, b) for a in sorted(FORM_PRESETS) for b in sorted(TIME_PRESETS)):
        style = {"forms": forms, "units": sorted(UNIT_PRESETS)[i % 3], "time": time}
        if domain.has_location:
            style["coords"] = sorted(COORD_PRESETS)[i % 3]
        style.update(layout=LAYOUTS[i % 3], boxes=sorted(CHECKBOXES)[i % 4], blank=sorted(BLANKS)[i % 4],
                     statuses=STATUS_ORDERS[i % 2], notices=NOTICE_PLACEMENTS[i // 2 % 2], header=i % len(HEADERS))
        yield style, random.Random(rng.random())


def render(inst: Instance, domain, style, rng, *, reports=None, notices=None, glossary=None) -> str:
    return FORM.render(RenderContext(inst.reports if reports is None else reports,
                                     inst.notices if notices is None else notices, inst.chatter, inst.compiled_at,
                                     inst.window_hours, style, rng, domain, glossary))


def form_blocks(text: str) -> list[str]:
    """The forms of a digest, each as one string (blocks of more than one line, after the header)."""
    return [block for block in text.split("\n\n")[1:] if "\n" in block]


def frame_of(block: str) -> str:
    """The frame a form is drawn in, by its heading."""
    return next(frame for frame, heading in HEADINGS.items() if any(map(heading.match, block.split("\n"))))


# ------------------------------------------------------------------------------ round trips


@pytest.mark.parametrize("seed", [21, 52, 87])
@pytest.mark.parametrize("name", DOMAINS)
def test_forms_read_back_every_report_and_notice_exactly_in_every_style(toys, name, seed):
    domain = get_domain(name)
    frames, seen = Counter(), {key: set() for key in style_options(domain)}
    for inst in instances_of(domain, seed=seed, n=3):
        bound = domain.bind(inst.surface)
        glossary = bound.glossary()
        reports, notices = printed(inst.reports, inst.notices)
        assert all(isinstance(r, Report if name == "earthquakes" else RecordReport) for r in reports)
        for style, rng in styles(bound, inst.seed):
            text = render(inst, bound, style, rng, glossary=glossary)
            data = FORM.parse(text, glossary, domain)  # the oracle names the registry's domain, never a bound one
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style
            assert FORM.sniff(text) and detect_genre(text) is FORM and not CLASSIC.sniff(text)
            for line in text.split("\n"):
                frames.update(frame for frame, heading in HEADINGS.items() if heading.match(line))
            for key, value in style.items():
                seen[key].add(value)
            for chatter in inst.chatter:  # lines without data change nothing
                assert FORM.parse(chatter, glossary, domain) == DigestData([], [])
    assert set(frames) == set(FRAME_NAMES) and seen == style_options(domain)


@pytest.mark.parametrize("family", [("toy", "toy-flat", "toy-dressed"), ("earthquakes",)])
def test_every_kind_of_trap_is_printed_and_read_back(toys, family):
    """Re-reports under the same ID and under their own reference with a note, retracted phantoms, and
    wrong values settled by a correction or by the settling report beside a second one."""
    forms = Counter()
    for name in family:
        domain = get_domain(name)
        for inst in instances_of(domain, seed=33, n=4):
            bound = domain.bind(inst.surface)
            forms.update(("conflict" if t.kind.startswith("conflicting") else t.kind, t.form) for t in inst.traps)
            for rendering in inst.renderings:
                assert rendering.genre == "form" and parse_glossary(rendering.prompt) == bound.glossary()
                data = FORM.parse(rendering.digest, parse_glossary(rendering.prompt), domain)
                for trap in inst.traps:  # every planted line and notice is among those read back
                    planted = printed([r for r in inst.reports if r.trap_id == trap.trap_id],
                                      [n for n in inst.notices if n.trap_id == trap.trap_id])
                    assert all(r in data.reports for r in planted[0]) and all(n in data.notices for n in planted[1])
            assert inst.renderings[1].style != inst.renderings[0].style  # a variant avoids the original's look
    assert set(forms) == {("duplicate", "same_id"), ("duplicate", "cross_ref"), ("retraction", "phantom"),
                          ("conflict", "correction"), ("conflict", "second_report")}


def test_legacy_earthquake_reports_read_back_as_020_records(instances):
    """The 0.2.0 records of a legacy run (models.Report, magnitude corrections) in forms: read back
    into the same types, through the domain's own glossary when the prompt prints none."""
    kinds = Counter()
    for inst in instances:
        reports, notices = printed(inst.reports, inst.notices)
        kinds.update(n.kind for n in inst.notices)
        for style, rng in list(styles(EARTHQUAKES, inst.seed))[::3]:
            text = render(inst, EARTHQUAKES, style, rng)
            for glossary in (None, EARTHQUAKES.glossary()):
                data = FORM.parse(text, glossary, EARTHQUAKES)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), style
            assert all(type(r) is Report for r in data.reports)
            assert all(n.field is None and (n.kind == "retraction") == (n.new_mag is None) for n in data.notices)
    assert set(kinds) == {"retraction", "correction"}


def test_place_notes_awkward_words_and_missing_qualifiers_read_back(toys):
    """Place notes with commas, colons, parentheses and double spaces, groups and sources with pipes, box
    glyphs and dots, and values without their qualifier, all in every frame; earthquakes with feed-style
    place notes too."""

    class Placed(toys.module.Toy):
        def synthetic(self, rng, n, compiled_at, window_hours):
            records = super().synthetic(rng, n, compiled_at, window_hours)
            return [replace(r, place=f"{r.group} buoy {i % 7}, (moored): see  log" if i % 3 else None,
                            qualifiers={} if i % 5 == 0 else r.qualifiers) for i, r in enumerate(records)]

    domain = domains.REGISTRY.register(Placed("toy-placed", located=True))
    for inst in instances_of(domain, seed=6, n=2):
        awkward = [replace(r, group=f"{r.group} │ zone: 4 ... (west)  ── end", source=f"{r.source} | relay.")
                   for r in inst.reports]
        glossary = domain.glossary()
        assert any(r.place for r in awkward) and any(not r.qualifiers for r in awkward)
        for style, rng in styles(domain, inst.seed):
            data = FORM.parse(render(inst, domain, style, rng, reports=awkward), glossary, domain)
            assert same_items(data.reports, printed(awkward, [])[0]), style
    inst = instances_of(EARTHQUAKES, seed=7, n=1)[0]
    placed = [replace(r, place=f"{i % 40} km NE of Lower Lake, CA") for i, r in enumerate(inst.reports)]
    for style, rng in list(styles(EARTHQUAKES, 3))[:5]:
        data = FORM.parse(render(inst, EARTHQUAKES, style, rng, reports=placed), EARTHQUAKES.glossary(), EARTHQUAKES)
        assert same_items(data.reports, printed(placed, [])[0])


def test_forms_refuse_what_they_cannot_read_back(toys):
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    style, rng = next(styles(toys.toy, 1))
    split = [replace(r, group="North\nSea") for r in inst.reports]  # no field of a form can hold a line break
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        render(inst, toys.toy, style, rng, reports=split)
    blank = BLANKS[style["blank"]]
    unprintable = [replace(r, place=blank) for r in inst.reports]  # a place that reads as a field left blank
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        render(inst, toys.toy, style, random.Random(1), reports=unprintable)
    with pytest.raises(ValueError, match="prints none"):  # only the glossary says how a toy record is worded
        FORM.parse(inst.renderings[0].digest, None, toys.toy)


def test_a_field_left_blank_in_a_row_of_two_shows_a_line(toys):
    """A boxed form puts two fields on some rows. Left empty, the first would put the row's second label
    where its own value belongs ("Cross-reference:      Location note: Dogger Bank"), so in a row of two a
    blank field shows a line even when the rendering leaves blanks empty; a field alone on its row may
    stay empty. Every form still reads back exactly."""
    boxed = FRAMES_BY_NAME["boxed"]
    lines = boxed.lay_out("FIELD REPORT FORM", "004417", [[("Cross-reference", ""), ("Location note", "Dogger Bank")],
                                                         [("Wave height", "1.85 m"), ("Probe", "")],
                                                         [("Form sent", "")]])
    rows = [line[2:-2].rstrip() for line in lines[3:-1]]
    assert rows == ["Cross-reference: ________  Location note: Dogger Bank",
                    "Wave height: 1.85 m        Probe: ________", "Form sent:"]
    # a row whose first field shows nothing before the next label, or whose last label ends the row
    first_empty, last_empty = re.compile(r"[^:]+: {2,}\S.*"), re.compile(r".*\S {2,}[^ :][^:]*:")
    blanks = Counter()
    for domain, ghost in ((toys.toy, "zz-00001"), (toys.flat, "zz-00001"), (EARTHQUAKES, "zz00000001")):
        for inst in instances_of(domain, seed=58, n=3):
            # A notice about an ID that no form carries: its amendment names no original form.
            amended = [*inst.notices, Notice(kind="retraction", target=ghost, time=inst.compiled_at - 600)]
            for style, rng in list(styles(domain, inst.seed))[:4]:
                style = {**style, "forms": "boxed", "blank": "empty"}
                text = render(inst, domain, style, rng, notices=amended)
                reports, notices = printed(inst.reports, amended)
                data = FORM.parse(text, domain.glossary(), domain)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), style
                for block in form_blocks(text):
                    if frame_of(block) != "boxed":
                        continue
                    for row in (line[2:-2].rstrip() for line in block.split("\n")[3:-1]):
                        assert not first_empty.fullmatch(row) and not last_empty.fullmatch(row), row
                        blanks.update(cell.split(":")[0] for cell in re.split(r" {2,}", row)
                                      if cell.endswith(f": {form_module.PAIRED_BLANK}"))
    assert {"Cross-reference", "Location note", "Original form no."} <= set(blanks)  # in both columns


def test_renderings_are_deterministic_and_variants_change_every_drawn_look(toys):
    inst = instances_of(toys.toy, seed=9, n=1)[0]
    style, _ = next(styles(toys.toy, 2))
    assert render(inst, toys.toy, style, random.Random(4)) == render(inst, toys.toy, style, random.Random(4))
    for domain in (toys.toy, toys.flat, EARTHQUAKES):
        drawn: dict[str, set] = {}
        for seed in range(60):
            original = FORM.choose_style(random.Random(seed), domain)
            variant = FORM.choose_style(random.Random(seed + 1000), domain, avoid=original)
            for key, value in original.items():
                drawn.setdefault(key, set()).add(value)
                if key not in ("notices", "header"):
                    assert variant[key] != value, key
        assert drawn == style_options(domain)  # "coords" only for records with a position
    assert FRAME_NAMES == tuple(FRAMES_BY_NAME) == tuple(frame.name for frame in FRAMES)
    assert all(set(weights) == set(FRAME_NAMES) and abs(sum(weights.values()) - 1) < 1e-9
               for weights in FORM_PRESETS.values())


# ------------------------------------------------------------------------- sniffing and noise


def test_sniff_accepts_every_form_digest_and_nothing_else(toys, instances):
    inst = instances_of(toys.flat, seed=3, n=1)[0]
    for style, rng in styles(toys.flat, 4):
        text = render(inst, toys.flat, style, rng)
        assert FORM.sniff(text) and FORM.sniff("\n\n  " + text)
        assert [g.name for g in all_genres() if g.sniff(text)] == ["form"]
        header, body = text.split("\n\n", 1)
        assert not FORM.sniff(body) and not FORM.sniff(header.split("\n")[0] + "\n\n" + body)
    for legacy in instances[:4]:  # the 0.2.0 digests of earthquakes ...
        for rendering in legacy.renderings:
            assert not FORM.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    classic = instances_of(toys.toy, seed=3, n=1)[0]  # ... and the classic digests of any other domain
    for style, rng in [(CLASSIC.choose_style(random.Random(s), toys.toy), random.Random(s)) for s in range(12)]:
        text = CLASSIC.render(RenderContext(classic.reports, classic.notices, classic.chatter, classic.compiled_at,
                                            classic.window_hours, style, rng, toys.toy, toys.toy.glossary()))
        assert not FORM.sniff(text) and detect_genre(text) is CLASSIC
    others = [
        "", "\n\n", "FIELD REPORT FORMS — READINGS",
        "From: desk@example.org\nTo: editors@example.org\nSubject: Fwd: report forms\n\nReading ID: by-12345",
        "[06:12] ana: FIELD REPORT FORMS — READINGS\n[06:13] ben: copy",
        "reading_id,operator,status\nby-12345,ukmo,reviewed",
        "| Reading ID | Operator |\n|---|---|\n| by-12345 | ukmo |",
        "NORTH SEA, 19 March (Wire) — Field report forms show calm seas.\n\nReadings were taken at dawn.",
        "UKMO-3 to Base, over.\nBase, copy that, over.",
        "READINGS DIGEST (edition 326-A)\nCompiled 06:00:00 UTC on 20 March 2026; covers the 24 hours before "
        "compilation. Reports are reproduced as received.",
    ]
    assert not any(FORM.sniff(text) for text in others)


def test_headings_noise_and_damaged_forms_change_nothing(toys):
    """Headers, section titles, the amendments heading and chatter are no forms, and a form with a row
    missing is not read at all, rather than misread."""
    inst = instances_of(toys.toy, seed=12, n=1)[0]
    glossary = toys.toy.glossary()
    words = {"source": "ukmo", "source_label": "operator", "Source_label": "Operator"}
    lines = [*(head.format(**words) for head in SECTION_HEADS), AMENDMENTS_HEAD, *inst.chatter]
    lines += [line.format(TITLE="READINGS", title="readings", batch="1-A", window="24 hours", compiled="x",
                          source_label="operator") for header in HEADERS for line in header]
    assert FORM.parse("\n\n".join(lines), glossary, toys.toy) == DigestData([], [])
    for style, rng in list(styles(toys.toy, 5))[:6]:
        text = render(inst, toys.toy, style, rng)
        whole = FORM.parse(text, glossary, toys.toy)
        for block in form_blocks(text)[:8]:
            rows = block.split("\n")
            damaged = text.replace(block, "\n".join(rows[:3] + rows[4:]))  # a row of the form torn out
            data = FORM.parse(damaged, glossary, toys.toy)
            assert len(data.reports) + len(data.notices) == len(whole.reports) + len(whole.notices) - 1
            assert all(r in whole.reports for r in data.reports) and all(n in whole.notices for n in data.notices)


def test_a_forms_own_noise_carries_no_data(toys, monkeypatch):
    """Serial numbers, codes and batch numbers never look like a record ID; each report form carries one
    message time (sent, relayed, forwarded), after its record's time and before the compilation and any
    notice about the record; and moving every message time changes nothing that is read."""
    for domain in (toys.toy, toys.dressed, EARTHQUAKES):
        for inst in instances_of(domain, seed=14, n=2):
            bound = domain.bind(inst.surface)
            glossary = bound.glossary()
            named = ({r.ref for r in inst.reports} | {r.same_as for r in inst.reports if r.same_as}
                     | {n.target for n in inst.notices})
            deadlines: dict[str, int] = {}
            for n in inst.notices:
                deadlines[n.target] = min(deadlines.get(n.target, inst.compiled_at), n.time)
            for style, rng in list(styles(bound, inst.seed))[:6]:
                text = render(inst, bound, style, rng, glossary=glossary)
                assert {m.group(0) for m in re.finditer(id_pattern(glossary), text)} == named
                for block in form_blocks(text):
                    sent = [parse_time(m["time"]) for m in MESSAGE_TIME.finditer(block)]
                    data = FORM.parse(block, glossary, domain)
                    (item,) = data.reports + data.notices
                    if isinstance(item, Notice):
                        assert sent == []
                        continue
                    (when,) = sent
                    limit = min(deadlines.get(ref, inst.compiled_at) for ref in (item.ref, item.same_as))
                    assert item.time < when < limit or item.time == when == limit - 1, block
    rng = random.Random(3)
    for _ in range(500):
        when = rng.randrange(1_780_000_000, 1_780_086_400)
        deadline = when + rng.choice((1, 2, 30, 600, 7200, 86400))
        assert when <= form_module._sent_time(rng, when, deadline) < deadline

    inst = instances_of(toys.toy, seed=15, n=1)[0]
    reports, notices = printed(inst.reports, inst.notices)
    monkeypatch.setattr(form_module, "_sent_time", lambda rng, when, deadline: when + 9 * 86400)
    for style, rng in list(styles(toys.toy, 6))[:5]:
        text = render(inst, toys.toy, style, rng)  # every message time nine days after its record's
        assert sorted(parse_time(m["time"]) for m in MESSAGE_TIME.finditer(text)) == sorted(
            r.time + 9 * 86400 for r in inst.reports)
        data = FORM.parse(text, toys.toy.glossary(), toys.toy)
        assert same_items(data.reports, reports) and same_items(data.notices, notices)


def test_every_amendment_is_a_notice_the_reading_rules_name(toys):
    """Whatever frame and phrase an amendment form is drawn with, it is a CORRECTION or RETRACTION notice
    (which both rules texts name), a correction says that the value was revised, a retraction withdraws the
    report in a phrase of the domain's own, and neither uses the other's words."""
    record = toys.flat.synthetic(random.Random(2), 1, 1_780_000_000, 24)[0]
    cases = [(toys.toy, record_rules(toys.toy),
              [Notice("retraction", record.id, 1_780_000_100, reason="probe fault"),
               Notice("correction", record.id, 1_780_000_200, field="temp", new_value=1.5, old_value=-2.0,
                      qualifier="spar"),
               Notice("correction", record.id, 1_780_000_300, field="wave", new_value=1.25, old_value=2.5)]),
             (EARTHQUAKES, READING_RULES,
              [Notice("retraction", "us7000abcd", 1_780_000_100, reason="quarry blast"),
               Notice("correction", "us7000abcd", 1_780_000_200, new_mag=4.2, old_mag=4.5, mag_type="mb")])]
    words = {"retraction": ("retraction", "retracted", "withdrawn"), "correction": ("correction", "revised")}
    for domain, rules, notices in cases:
        rules = " ".join(rules.split())
        assert "a CORRECTION notice" in rules and "a RETRACTION notice" in rules and "a value was revised" in rules
        vocab, glossary = domain.trap_vocab, domain.glossary()
        seen = Counter()
        for seed in range(80):
            style = FORM.choose_style(random.Random(seed), domain)
            ctx = RenderContext([], notices, [], 1_780_003_600, 24, style, random.Random(seed), domain, glossary)
            for block in form_blocks(FORM.render(ctx)):
                (notice,) = FORM.parse(block, glossary, domain).notices  # each amendment reads back on its own
                other = "correction" if notice.kind == "retraction" else "retraction"
                assert notice.kind.upper() in block and not any(w in block.lower() for w in words[other]), block
                if notice.kind == "retraction":
                    assert any(f" {phrase}. Disregard it." in block for phrase in vocab.retraction_notices), block
                else:
                    assert " was revised; use the corrected entry." in block
                seen[notice.kind, frame_of(block)] += 1
        assert set(seen) == {(kind, frame) for kind in ("retraction", "correction") for frame in FRAME_NAMES}


# ----------------------------------------------------------------------------- whole runs

MIXES = ("toy:form", "toy-flat:form", "toy-dressed:form", "earthquakes:form")


@pytest.fixture(scope="module")
def form_runs(tmp_path_factory):
    """generate (mix "<domain>:form") -> run (oracle, naive, mock) -> score -> report, for four instances
    of the toy domain with a position, of toy-flat without one, of toy-dressed with a surface per instance,
    and of earthquakes."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        toy = domains.load_plugin(TOY_PLUGIN)
        module = sys.modules[type(toy).__module__]
        for variant in (module.FLAT, module.DRESSED):
            domains.REGISTRY.register(variant)
        runs = {}
        for mix, seed in zip(MIXES, (41, 42, 43, 44)):
            run_dir = pipeline.generate(GenerateConfig(seed=seed, instances=4, mix=mix),
                                        tmp_path_factory.mktemp("runs") / mix.replace(":", "-"), log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[mix] = run_dir
        yield runs


def test_form_runs_print_a_glossary_and_read_back_every_rendering(form_runs, tmp_path):
    for mix, run in form_runs.items():
        name = mix.split(":")[0]
        manifest = read_json(run / "manifest.json")
        assert manifest["config"]["mix"] == mix and manifest["snapshot"]["domain"] == name
        check = manifest["self_check"]  # the oracle's agreement, and each digest sniffed as a form digest
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        for inst in load_instances(run):
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["form"] * 3
            domain = get_domain(name)
            bound = domain.bind(inst.surface)
            for rendering in inst.renderings:
                prompt = rendering.prompt
                assert prompt.count("<glossary>") == 1 and parse_glossary(prompt) == bound.glossary()
                assert prompt.index(bound.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
                assert extract_digest(prompt) == rendering.digest and detect_genre(rendering.digest) is FORM
                data = FORM.parse(rendering.digest, parse_glossary(prompt), domain)
                assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_the_oracle_scores_100_percent_on_form_runs(form_runs):
    for mix, run in form_runs.items():
        name = mix.split(":")[0]
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "form")}
        for inst in load_instances(run):
            for rendering in inst.renderings:
                for genre in ("form", None):  # named, or recognised by sniffing
                    assert solve(rendering.prompt, question_specs(inst), domain=name, genre=genre) == inst.answer_key


def test_the_naive_reader_falls_for_the_traps_of_form_runs_and_only_for_them(form_runs):
    for mix, run in form_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        conflict = "conflicting_magnitude" if mix.startswith("earthquakes") else "conflicting_value"
        assert list(naive["by_kind"]) == ["duplicate", "retraction", conflict]
        assert models["naive"]["accuracy"]["rate"] < 1.0 == models["oracle"]["accuracy"]["rate"]
        assert 0 < models["mock"]["accuracy_all_renderings"]["rate"] < 1  # (13 originals can all come out right)
        instances = load_instances(run)
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], row
