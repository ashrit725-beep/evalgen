"""The table genre (evalgen/genres/table.py): a digest as a CSV or markdown table of reports, for the
earthquake domain's 0.2.0 records and for the Records of any other domain (the test-only toy domains,
with and without a position, and with a surface drawn per instance).

Every style reads back to exactly its reports and notices; sniff recognises every style and nothing
else (legacy and generic classic digests, other formats); every footnote wording is named by the
domain's reading rules; and whole runs in the table genre (generate, oracle, naive, mock, score,
report) give the oracle 100% and the naive reader exactly the traps' wrong answers. Legacy (0.2.0)
output is pinned by test_regression_020.py."""
from __future__ import annotations

import random
import re
import sys
from dataclasses import replace

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.domains import get_domain
from evalgen.domains.base import Domain, RecordReport, record_rules
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import all_genres, detect_genre
from evalgen.genres.base import RenderContext, parse_glossary
from evalgen.genres.classic import CLASSIC, UNIT_PRESETS
from evalgen.genres.table import (CAPTIONS, CASES, CELL_LAYOUTS, COLUMN_ORDERS, CORRECTION_KEYWORDS, DELIMITERS,
                                  FENCES, FORMATS, NOTE_MARKS, POSITIONS, QUALIFIER_PLACES, QUOTING,
                                  RETRACTION_KEYWORDS, ROW_ORDERS, TABLE, TIME_STYLES, UNIT_PLACES)
from evalgen.models import Instance, Notice, Report
from evalgen.ops import DigestData
from evalgen.pipeline import GenerateConfig, build_instance, question_specs
from evalgen.questions import Context, compose_new_templates
from evalgen.report import load_instances
from evalgen.rules import READING_RULES
from evalgen.seeds import SeedLog
from evalgen.solver import extract_digest, solve
from evalgen.sources import build_domain_snapshot
from evalgen.util import read_json

from conftest import TOY_PLUGIN, quiet

DOMAINS = ["earthquakes", "toy", "toy-flat", "toy-dressed"]

# What each key of a table's look can be: in both formats, and in each format of its own.
CHOICES = {"case": CASES, "units": UNIT_PLACES, "unit_mix": UNIT_PRESETS, "time": TIME_STYLES, "position": POSITIONS,
           "qualifiers": QUALIFIER_PLACES, "columns": COLUMN_ORDERS, "rows": ROW_ORDERS, "notes": NOTE_MARKS}
FORMAT_CHOICES = {"csv": {"delimiter": DELIMITERS, "quoting": QUOTING, "fence": FENCES},
                  "markdown": {"cells": CELL_LAYOUTS}}


def instances_of(domain: Domain, seed: int, n: int = 3, genre=TABLE) -> list[Instance]:
    """The first instances of a run about `domain` in one genre, built as the pipeline builds them."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    return [build_instance(i, snapshot, new, log, GenerateConfig(seed=seed), ctx, used, usage, domain=domain,
                           genre_plan=(genre,) * 3) for i in range(n)]


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


def every_style(domain: Domain) -> list[dict]:
    """Styles that take, between them, every option of every key of a table's look, in both formats:
    style i of a format takes option i + k of its k-th key (wrapping round), so that options also meet
    in different combinations."""
    styles = []
    for fmt, own in FORMAT_CHOICES.items():
        keys = {**own, **CHOICES}
        for i in range(max(len(options) for options in keys.values())):
            style = {"format": fmt}
            for k, (key, options) in enumerate(keys.items()):
                if key != "position" or domain.has_location:
                    style[key] = sorted(options)[(i + k) % len(options)]
            style["caption"] = i % len(CAPTIONS)
            styles.append(style)
    return styles


def render(inst: Instance, style: dict, domain: Domain, seed: int = 0, glossary=None) -> str:
    return TABLE.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                      style, random.Random(seed), domain, glossary or domain.glossary()))


# ------------------------------------------------------------------------------- round trips


@pytest.mark.parametrize("name", DOMAINS)
def test_tables_read_back_every_report_and_notice_in_every_style(toys, name):
    domain = get_domain(name)
    every = {**CHOICES, **FORMAT_CHOICES["csv"], **FORMAT_CHOICES["markdown"], "format": FORMATS}
    if not domain.has_location:
        del every["position"]
    used: dict[str, set] = {key: set() for key in every}
    for inst in instances_of(domain, seed=27):
        bound = domain.bind(inst.surface)
        glossary = bound.glossary()
        reports, notices = printed(inst.reports, inst.notices)
        drawn = [TABLE.choose_style(random.Random(seed), bound) for seed in range(6)]
        for n, style in enumerate(every_style(bound) + drawn):
            text = render(inst, style, bound, seed=n, glossary=glossary)
            data = TABLE.parse(text, glossary, bound)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style
            assert all(isinstance(r, Report if name == "earthquakes" else RecordReport) for r in data.reports)
            assert [g.name for g in all_genres() if g.sniff(text)] == ["table"] and detect_genre(text) is TABLE
            for key, value in style.items():
                if key in used:
                    used[key].add(value)
        for chatter in inst.chatter:  # lines without data change nothing
            assert TABLE.parse(chatter, glossary, bound) == DigestData([], [])
    assert {key: sorted(values) for key, values in used.items()} == {key: sorted(options)
                                                                     for key, options in every.items()}


def test_the_test_styles_are_the_styles_choose_style_draws(toys):
    for domain in (EARTHQUAKES, toys.flat):
        drawn = {}
        for seed in range(30):
            style = TABLE.choose_style(random.Random(seed), domain)
            drawn.setdefault(style["format"], set(style))
        assert {style["format"]: set(style) for style in every_style(domain)} == drawn
        assert ("position" in drawn["csv"]) == domain.has_location


def test_each_choice_shows_in_the_table(toys):
    """A spot check that every key of the look changes what the reader sees."""
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    styles = every_style(toys.toy)
    csv_style = {**styles[0], "delimiter": "comma", "quoting": "minimal", "fence": "bare"}
    markdown_style = next(style for style in styles if style["format"] == "markdown")

    def table(base, **changes) -> str:
        return render(inst, {**base, **changes}, toys.toy, seed=3)

    def header(text: str) -> str:
        return next(line for line in text.split("\n") if "status" in line.lower() and not line.startswith("#"))

    assert not any(line.startswith("|") for line in table(csv_style).split("\n"))
    assert re.search(r"^\|(-{3,}\|)+$", table(markdown_style, cells="compact"), re.MULTILINE)
    assert re.search(r"^\| -+ \| .*-+: \|", table(markdown_style, cells="padded"), re.MULTILINE)
    assert "\t" in header(table(csv_style, delimiter="tab", quoting="minimal"))
    assert header(table(csv_style, delimiter="semicolon", quoting="all")).startswith('"')
    assert table(csv_style, fence="fenced").startswith("```csv\n") and table(csv_style, fence="fenced").endswith("```")
    for case, label in (("plain", "reading ID"), ("capitalized", "Reading ID"), ("upper", "READING ID")):
        assert label in header(table(csv_style, case=case))
    assert re.search(r"water temperature \(°[CF]\)", header(table(csv_style, units="parentheses", case="plain")))
    assert re.search(r"wave height \[(m|ft)\]", header(table(csv_style, units="brackets", case="plain")))
    assert re.search(r",-?\d+(\.\d+)? °[CF],", table(csv_style, units="cell", qualifiers="column"))
    assert "water temperature," in header(table(csv_style, units="cell", case="plain", delimiter="comma"))
    assert "time (UTC)" in header(table(csv_style, time="utc_column", case="plain"))
    assert re.search(r"[+-]\d{2}:00", table(csv_style, time="local"))
    assert "latitude" in header(table(csv_style, position="columns", case="plain"))
    assert re.search(r"\d°[NS], \d", table(csv_style, position="hemi"))
    assert "probe" in header(table(csv_style, qualifiers="column", case="plain"))
    assert re.search(r"\((hull|spar|drifter)\)", table(csv_style, qualifiers="inline"))
    assert header(table(csv_style, columns="source_first", case="plain")).startswith("operator")
    assert "# " in table(csv_style, rows="by_source") and "Rows from" in table(markdown_style, rows="by_source")
    for mark, opening in (("bracket", "[^1]: "), ("number", "1. "), ("paren", "(1) "), ("word", "Note 1: ")):
        assert f"\n{opening}" in table(markdown_style, notes=mark)
    # (A caption whose compilation time is written with a comma is a quoted CSV cell.)
    assert {table(csv_style, caption=i).split("\n")[0].lstrip('"')[:6] for i in range(len(CAPTIONS))} == {
        "Table ", "Readin", "Export"}


def test_a_variant_avoids_every_choice_of_the_original(toys):
    for domain in (EARTHQUAKES, toys.flat):
        for seed in range(20):
            original = TABLE.choose_style(random.Random(seed), domain)
            variant = TABLE.choose_style(random.Random(seed + 100), domain, avoid=original)
            assert variant["format"] != original["format"]
            for key in set(original) & set(variant) - {"caption", "format"}:
                assert variant[key] != original[key], key


def test_earthquake_tables_read_back_020_records_with_places(toys):
    """USGS reports name a place ("9 km N of Delta, B.C., MX"); 0.2.0 corrections give a new magnitude."""
    inst = instances_of(EARTHQUAKES, seed=8, n=1)[0]
    places = ["9 km N of Delta, B.C., MX", "55 km W of Nanwalek, Alaska", "Kermadec Islands | region; (offshore)",
              None]
    reports = [replace(r, place=places[i % len(places)]) for i, r in enumerate(inst.reports)]
    want_reports, want_notices = printed(reports, inst.notices)
    for n, style in enumerate(every_style(EARTHQUAKES)):
        text = TABLE.render(RenderContext(reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                          style, random.Random(n), EARTHQUAKES, EARTHQUAKES.glossary()))
        assert "place" in text.lower()
        for glossary in (EARTHQUAKES.glossary(), None):  # a prompt without one is read in the domain's words
            data = TABLE.parse(text, glossary, EARTHQUAKES)
            assert same_items(data.reports, want_reports) and same_items(data.notices, want_notices)
    correction = Notice("correction", inst.reports[0].ref, inst.compiled_at - 99, new_mag=4.27, old_mag=3.9,
                        mag_type="mww")
    for seed, style in enumerate(every_style(EARTHQUAKES)):  # every wording of a 0.2.0 correction, without a glossary
        text = TABLE.render(RenderContext(inst.reports, [correction], [], inst.compiled_at, inst.window_hours, style,
                                          random.Random(seed), EARTHQUAKES))
        assert TABLE.parse(text, None, EARTHQUAKES).notices == [correction]
    assert isinstance(TABLE.parse(text, None, EARTHQUAKES).reports[0], Report)


def test_awkward_words_read_back_and_unprintable_ones_are_refused(toys):
    class Placed(toys.module.Toy):
        """Toy readings, two in three at a named buoy, and one in five without the probe of its temperature."""

        def synthetic(self, rng, n, compiled_at, window_hours):
            records = super().synthetic(rng, n, compiled_at, window_hours)
            return [replace(r, place=f'{r.group} buoy {i % 7}, "B" | (moored); east' if i % 3 else None,
                            qualifiers={} if i % 5 == 0 else r.qualifiers) for i, r in enumerate(records)]

    domain = domains.REGISTRY.register(Placed("toy-placed", located=True))
    for inst in instances_of(domain, seed=6, n=2):
        reports, notices = printed(inst.reports, inst.notices)
        assert any(r.place for r in reports) and any(not r.qualifiers for r in reports)
        awkward = [replace(r, group=f"{r.group}, \"Outer\" | West; \\ 50%") for r in inst.reports]
        for n, style in enumerate(every_style(domain)):
            for given, want in ((inst.reports, reports), (awkward, printed(awkward, [])[0])):
                text = render(replace(inst, reports=given), style, domain, seed=n)
                data = TABLE.parse(text, domain.glossary(), domain)
                assert same_items(data.reports, want) and same_items(data.notices, notices), style
        for rendering in inst.renderings:
            assert solve(rendering.prompt, question_specs(inst), domain=domain.name) == inst.answer_key
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    style = every_style(toys.toy)[0]
    split = [replace(r, group="North\nSea") for r in inst.reports]  # no row can hold a line break
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        TABLE.render(RenderContext(split, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                   random.Random(0), toys.toy, toys.toy.glossary()))
    glossary = toys.toy.glossary()
    clash = replace(glossary, fields=(replace(glossary.fields[0], label="status"), glossary.fields[1]))
    with pytest.raises(ValueError, match="does not name one column"):  # a field labelled like a fixed column
        render(inst, style, toys.toy, glossary=clash)
    with pytest.raises(ValueError, match="prints none"):  # words drawn per instance are only in the glossary
        TABLE.parse(inst.renderings[0].digest, None, toys.dressed)


# ----------------------------------------------------------------------------------- sniffing

OTHER_FORMATS = (
    "Quake rattles Nevada\n\nRENO (Wire) — A small earthquake shook western Nevada on Tuesday, officials said.",
    "From: desk@example.org\nSubject: Fwd: reports\n\n> status, network, region, time, depth, magnitude\nThanks,\nAl",
    "[06:12] ana: status update, nothing new\n[06:13] ben: ok, thanks, see you, bye, later, all",
    '{"status": "reviewed", "id": "us7000abcd", "region": "Nevada", "time": 1, "depth": 2, "mag": 3}',
    "| ID | status | region |\n|---|---|---|\n| us7000abcd | reviewed | Nevada |",  # too few columns
    "| event ID | network | status | time | region | depth | magnitude |\n"  # no separator row
    "| us7000abcd | us | reviewed | x | y | 1 | 2 |",
    "event ID,network,state,time,region,depth,magnitude\nus7000abcd,us,reviewed,x,y,1,2",  # no "status" column
    "event ID,network,status,status,region,depth,magnitude\nus7000abcd,us,reviewed,x,y,1,2",  # a label twice
    "# event ID,network,status,time,region,depth,magnitude",  # a comment
    "",
)


def test_sniff_recognises_tables_and_nothing_else(toys, instances):
    for inst in instances:  # legacy (0.2.0) classic digests
        for rendering in inst.renderings:
            assert not TABLE.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for domain in (EARTHQUAKES, toys.toy, toys.flat, toys.dressed):  # classic digests of the generic path
        for inst in instances_of(domain, seed=12, n=2, genre=CLASSIC):
            for rendering in inst.renderings:
                assert not TABLE.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for text in OTHER_FORMATS:
        assert not TABLE.sniff(text), text
    inst = instances_of(toys.toy, seed=12, n=1)[0]
    for style in every_style(toys.toy):  # what makes a table: its header row (and separator row)
        text = render(inst, style, toys.toy)
        assert TABLE.sniff(text) and not CLASSIC.sniff(text)
        if style["format"] == "markdown":
            separator = re.compile(r"\|[ :|-]+")
            assert not TABLE.sniff("\n".join(line for line in text.split("\n") if not separator.fullmatch(line)))
        else:
            header = next(line for line in text.split("\n") if "status" in line.lower() and not line.startswith("#"))
            assert TABLE.sniff(header) and not TABLE.sniff(text.replace(header, "# " + header))


# ---------------------------------------------------------------------------- notice wordings


def test_every_footnote_wording_is_named_by_the_reading_rules(toys):
    """Whatever keyword, wording and phrase a footnote is drawn with, the domain's reading rules name it:
    the 0.2.0 rules (by the words test_reading_rules.py checks them for) for earthquakes, and every
    other domain's own rules (with its trap vocabulary)."""
    record = instances_of(toys.flat, seed=2, n=1)[0].reports[0]
    quake = Report("us7000abcd", "us", "reviewed", 1770530400, 1.0, 2.0, 5.0, 4.5, "mb", "Nevada")
    cases = [(EARTHQUAKES, READING_RULES, {"retraction": ("retraction", "retracted", "withdrawn"),
                                           "correction": ("correction", "revised")},
              [Notice("retraction", quake.ref, 1770530500, reason="quarry blast"),
               Notice("correction", quake.ref, 1770530600, new_mag=4.2, old_mag=4.5, mag_type="mb")], [quake])]
    for domain in (toys.flat, toys.dressed.bind(toys.dressed.draw_surface(random.Random(1)))):
        vocab = domain.trap_vocab
        cases.append((domain, record_rules(domain),
                      {"retraction": ("retraction", "was not a real", *vocab.retraction_notices),
                       "correction": ("correction", "revised", *vocab.correction_notices)},
                      [Notice("retraction", record.ref, record.time + 99, reason="probe fault"),
                       Notice("correction", record.ref, record.time + 9, field="temp", new_value=1.5, old_value=-2.0,
                              qualifier="spar")],
                      [replace(record, group="Ostra Sound", source="ostr") if domain.surface else record]))
    for domain, rules, words, notices, reports in cases:
        rules = " ".join(rules.split()).lower()
        assert all(word.lower() in rules for kind in words.values() for word in kind)
        keywords = set()
        for seed in range(80):
            style = {**TABLE.choose_style(random.Random(seed), domain), "format": "markdown", "cells": "compact",
                     "notes": "word"}
            text = TABLE.render(RenderContext(reports, notices, [], 1770530400 + 3600, 24, style,
                                              random.Random(seed), domain, domain.glossary()))
            footnotes = re.findall(r"^Note \d+: (.+)$", text, re.MULTILINE)
            assert len(footnotes) == 2 and TABLE.parse(text, domain.glossary(), domain).notices != []
            for footnote, notice in zip(footnotes, TABLE.parse(text, domain.glossary(), domain).notices):
                assert any(word.lower() in footnote.lower() for word in words[notice.kind]), footnote
                keywords.add((notice.kind, next(k for k in RETRACTION_KEYWORDS + CORRECTION_KEYWORDS
                                                if footnote.startswith(k + " (") or footnote.startswith(k + ","))))
        assert keywords == {*(("retraction", k) for k in RETRACTION_KEYWORDS),
                            *(("correction", k) for k in CORRECTION_KEYWORDS)}


# ----------------------------------------------------------------------------- whole runs


@pytest.fixture(scope="module")
def table_runs(tmp_path_factory):
    """generate (mix "<domain>:table") -> run (oracle, naive, mock) -> score -> report, for four
    instances each of earthquakes, of the toy domain with a position, of toy-flat without one, and of
    toy-dressed with a surface drawn per instance."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        toy = domains.load_plugin(TOY_PLUGIN)
        module = sys.modules[type(toy).__module__]
        for variant in (module.FLAT, module.DRESSED):
            domains.REGISTRY.register(variant)
        runs = {}
        for name, seed in zip(DOMAINS, (61, 62, 63, 64)):
            run_dir = pipeline.generate(GenerateConfig(seed=seed, instances=4, mix=f"{name}:table"),
                                        tmp_path_factory.mktemp("runs") / name, log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[name] = run_dir
        yield runs


def test_table_runs_print_tables_that_read_back_exactly(table_runs):
    for name, run in table_runs.items():
        manifest = read_json(run / "manifest.json")
        check = manifest["self_check"]
        assert manifest["config"]["mix"] == f"{name}:table"
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        for inst in load_instances(run):
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["table"] * 3
            bound = get_domain(name).bind(inst.surface)
            reports, notices = printed(inst.reports, inst.notices)
            for rendering in inst.renderings:
                assert parse_glossary(rendering.prompt) == bound.glossary()
                assert extract_digest(rendering.prompt) == rendering.digest and detect_genre(rendering.digest) is TABLE
                data = TABLE.parse(rendering.digest, bound.glossary(), bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices)
            original = inst.renderings[0].style
            for variant in inst.renderings[1:]:  # a variant in the same genre avoids the original's look
                assert variant.style["format"] != original["format"]
                assert all(variant.style[k] != original[k] for k in set(variant.style) & set(original)
                           if k not in ("format", "caption"))


def test_the_oracle_scores_100_percent_on_tables(table_runs):
    for name, run in table_runs.items():
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "table")}
        for inst in load_instances(run):
            for rendering in inst.renderings:  # by sniffing, as the generator's self-check reads it, and by name
                for genre in (None, "table"):
                    assert solve(rendering.prompt, question_specs(inst), domain=name, genre=genre) == inst.answer_key


def test_the_naive_reader_falls_for_the_traps_in_tables_and_only_for_them(table_runs):
    kinds = {"earthquakes": ["duplicate", "retraction", "conflicting_magnitude"]}
    for name, run in table_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        assert list(naive["by_kind"]) == kinds.get(name, ["duplicate", "retraction", "conflicting_value"])
        assert models["naive"]["accuracy"]["rate"] < 1.0 == models["oracle"]["accuracy"]["rate"]
        assert 0 < models["mock"]["accuracy"]["rate"] < 1
        instances = load_instances(run)
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], row


def test_a_table_run_regenerates_exactly_from_its_seed(table_runs, tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        domains.load_plugin(TOY_PLUGIN)
        for name, seed in (("earthquakes", 61), ("toy", 62)):
            again = pipeline.generate(GenerateConfig(seed=seed, instances=4, mix=f"{name}:table"), tmp_path / name,
                                      log=quiet)
            for path in sorted((table_runs[name] / "instances").iterdir()):
                assert (again / "instances" / path.name).read_bytes() == path.read_bytes(), path.name
