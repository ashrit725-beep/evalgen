"""The news genre (evalgen/genres/news.py): a digest as a news article, with a headline, a byline, a
dateline and lede, one paragraph per report, quotes, a background paragraph, chatter and a sign-off,
and notices as "Update", "Correction", "Retraction notice" or "Correction notice" paragraphs (never an
editor's note, which the rules count among the lines without data). It is written and read from the
prompt's glossary alone, for the earthquake domain's 0.2.0 reports and notices and for the records of
the test-only toy domain (toy_domain.py) with and without a position and with a surface drawn per
instance. Every style reads back to exactly its reports and notices, every notice wording is named
by the reading rules, lines without data change nothing, sniffing tells an article from every other
format, and whole runs in the news genre give the oracle 100% and the naive reader exactly the traps'
wrong answers."""
from __future__ import annotations

import itertools
import random
import re
import sys
from collections import Counter
from dataclasses import replace

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.domains import get_domain
from evalgen.domains.base import record_rules
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import all_genres, detect_genre, get_genre
from evalgen.genres.base import TIME_FORMATS, RenderContext, id_pattern, parse_glossary, parse_time
from evalgen.genres.classic import CLASSIC
from evalgen.genres.news import (AGENCIES, COORD_PRESETS, HEADLINES, NEWS, NOTICE_PLACEMENTS, ORDERS,
                                 PARAGRAPH_PRESETS, SUBHEADS, TIME_PRESETS, UNIT_PRESETS, VALUE_PRESETS)
from evalgen.models import Instance, Notice, Report
from evalgen.ops import DigestData
from evalgen.pipeline import GenerateConfig, build_instance, question_specs
from evalgen.questions import Context, compose_new_templates
from evalgen.report import load_instances
from evalgen.seeds import SeedLog
from evalgen.solver import extract_digest, solve
from evalgen.sources import build_domain_snapshot
from evalgen.util import read_json, write_json

from conftest import TOY_PLUGIN, quiet
from test_usgs_forecast import fake_feed

DOMAINS = ("earthquakes", "toy", "toy-flat", "toy-dressed")
COMPILED_AT = 1_780_000_000
PRESET_KEYS = ("paragraphs", "values", "units", "time", "coords", "order", "notices")
# The words that name each kind of notice: in the 0.2.0 rules (tests/test_reading_rules.py), and in the
# record rules of every other domain, besides the phrases of its trap vocabulary, which they quote.
RULE_WORDS_020 = {"retraction": ("retraction", "retracted", "withdrawn"), "correction": ("correction", "revised")}
RECORD_RULE_WORDS = {"retraction": ("retraction", "was not a real"), "correction": ("correction", "revised")}
NOTICE_FORMS = {  # how each wording opens, and a phrase only it has
    ("retraction", "Update ("): "has been retracted", ("retraction", "Update, "): "is now marked",
    ("retraction", "Retraction notice ("): "listed in this article", ("correction", "Correction ("): "was revised to",
    ("correction", "Update, "): "in a correction", ("correction", "Correction notice ("): "has been revised to",
}
FORM_MARKERS = {
    "attributed": re.compile(r"^The .+ reported .+ \(.+\), recorded at .+, with .+\."),
    "record_first": re.compile(r"^.+, filed as .+, was recorded at .+\. It gave .+\."),
    "time_first": re.compile(r"^At .+, the .+ recorded .+, listing .+; the report is marked "),
    "figures_first": re.compile(r"^.+: those were the figures for .+, filed as .+ and recorded at "),
    "place_first": re.compile(r"^(?:In|Near) .+ was recorded at .+ by the .+: .+\. Status: "),
}
COORDS = {"hemi": re.compile(r"\d+(?:\.\d+)?°[NS], \d+(?:\.\d+)?°[EW]"),
          "signed": re.compile(r"\(-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\)")}
DATELINE = re.compile(r"^[A-Z ]+, [A-Z][a-z]+ \d{1,2} \(.+\) — ")
SIGNOFF = re.compile(r"^\(?(?:.+ )?[Rr]eporting by .+; editing by .+[.)]$")


def instances_of(domain, seed: int, n: int = 2, genre=NEWS) -> list[Instance]:
    """The first instances of a run about `domain` off the legacy path, every rendering in `genre`."""
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


def styles(domain, seed: int):
    """Every notice placement under every paragraph order, with every preset of the other dimensions
    taken in turn, then six styles that choose_style draws; each with a stream of its own."""
    rng = random.Random(seed)
    for i, (order, notices) in enumerate(itertools.product(ORDERS, NOTICE_PLACEMENTS)):
        style = {"paragraphs": sorted(PARAGRAPH_PRESETS)[i % 3], "values": sorted(VALUE_PRESETS)[i % 3],
                 "units": sorted(UNIT_PRESETS)[i // 3], "time": sorted(TIME_PRESETS)[(i + i // 3) % 3]}
        if domain.has_location:
            style["coords"] = sorted(COORD_PRESETS)[i % 3]
        style.update(order=order, notices=notices, headline=i % len(HEADLINES), masthead=i % len(AGENCIES))
        yield style, random.Random(rng.random())
    for _ in range(6):
        yield NEWS.choose_style(rng, domain), random.Random(rng.random())


def _time_slots(paragraph: str, fmt: str) -> list[str]:
    """The pieces of a paragraph that a time in `fmt` could fill (enough to tell which formats it uses)."""
    pattern = {"iso": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "iso_space": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC",
               "rfc": r"\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} \+0000",
               "prose_utc": r"\d{2}:\d{2}:\d{2} UTC on \d{1,2} [A-Z][a-z]+ \d{4}",
               "iso_offset": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:00",
               "local12": r"\d{1,2}:\d{2}:\d{2} [AP]M local time \(UTC[+-]\d{2}:00\) on \w{3} \d{1,2}, \d{4}",
               "local24": r"\d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:00\) \d{1,2} \w{3} \d{4}"}[fmt]
    return re.findall(pattern, paragraph)


def anatomy(text: str, glossary, domain, reports) -> Counter:
    """What an article shows: its report forms and value phrasings, time and position formats, where
    its notices stand (top, interleaved, end) and whether subheads group its reports."""
    seen: Counter = Counter()
    paragraphs = text.split("\n\n")
    lede = next(i for i, p in enumerate(paragraphs) if DATELINE.match(p))
    signoff = next(i for i, p in enumerate(paragraphs) if SIGNOFF.match(p))
    spelled = [f"{'an' if f.label[:1].lower() in 'aeiou' else 'a'} {f.label} of ".lower() for f in glossary.fields]
    for i, paragraph in enumerate(paragraphs):
        data = NEWS.parse(paragraph, glossary, domain)
        if data.reports:
            seen.update(f"form:{form}" for form, marker in FORM_MARKERS.items() if marker.match(paragraph))
            seen["values:spelled" if any(s in paragraph.lower() for s in spelled) else "values:plain"] += 1
            seen.update(f"time:{fmt}" for fmt in TIME_FORMATS
                        if any(parse_time(slot) is not None for slot in _time_slots(paragraph, fmt)))
            seen.update(f"coords:{fmt}" for fmt, pattern in COORDS.items() if pattern.search(paragraph))
        for _ in data.notices:
            seen["notice:" + ("top" if i < lede else "end" if i > signoff else "interleaved")] += 1
        if any(paragraph == head.format(group=r.group) for heads in SUBHEADS.values() for head in heads
               for r in reports):
            seen["subhead"] += 1
    return seen


# ------------------------------------------------------------------------- reading back


@pytest.mark.parametrize("name", DOMAINS)
def test_news_reads_back_every_report_and_notice_exactly_in_every_style(toys, name):
    domain = get_domain(name)
    seen: Counter = Counter()
    for seed in (3, 27):
        for inst in instances_of(domain, seed):
            bound = domain.bind(inst.surface)
            glossary = bound.glossary()
            reports, notices = printed(inst.reports, inst.notices)
            for rendering in inst.renderings:  # as the pipeline rendered them
                assert rendering.genre == "news" and detect_genre(rendering.digest) is NEWS
            for style, rng in styles(bound, inst.seed):
                text = NEWS.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at,
                                                 inst.window_hours, style, rng, bound, glossary))
                data = NEWS.parse(text, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), style
                assert NEWS.sniff(text) and detect_genre(text) is NEWS and not CLASSIC.sniff(text)
                seen.update(anatomy(text, glossary, bound, inst.reports))
    wanted = {f"form:{form}" for form in FORM_MARKERS} | {"values:plain", "values:spelled", "subhead"}
    wanted |= {f"time:{fmt}" for fmt in TIME_FORMATS} | {f"notice:{place}" for place in ("top", "interleaved", "end")}
    wanted |= {"coords:hemi", "coords:signed"} if domain.has_location else set()
    assert wanted <= set(seen), sorted(wanted - set(seen))
    assert ("coords:hemi" in seen) == domain.has_location


def test_earthquake_articles_hold_0_2_0_reports_and_notices(instances):
    """The legacy run's reports and notices (0.2.0 Reports, magnitude corrections with old values and
    retraction reasons) read back exactly, through the earthquake glossary or none at all."""
    glossary = EARTHQUAKES.glossary()
    assert any(n.kind == "correction" for inst in instances for n in inst.notices)
    for inst in instances:
        reports, notices = printed(inst.reports, inst.notices)
        for seed in range(3):
            rng = random.Random(seed)
            text = NEWS.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at,
                                             inst.window_hours, NEWS.choose_style(rng, EARTHQUAKES), rng, EARTHQUAKES))
            for reader in (glossary, None):
                data = NEWS.parse(text, reader, EARTHQUAKES)
                assert same_items(data.reports, reports) and same_items(data.notices, notices)
            assert all(isinstance(r, Report) for r in data.reports) and "earthquake" in text and "magnitude" in text


def test_a_rendering_is_drawn_from_its_own_stream_and_leaves_the_instance_alone(toys):
    inst = instances_of(toys.toy, seed=12, n=1)[0]
    before = (list(inst.reports), list(inst.notices), list(inst.chatter))
    glossary = toys.toy.glossary()
    texts = set()
    for seed in (1, 1, 2):
        rng = random.Random(seed)
        style = NEWS.choose_style(rng, toys.toy)
        texts.add(NEWS.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at,
                                            inst.window_hours, style, rng, toys.toy, glossary)))
    assert len(texts) == 2  # the same seed, the same article; another seed, another article
    assert (inst.reports, inst.notices, inst.chatter) == before


def test_awkward_words_are_printed_and_read_back(toys):
    """Free words sit between typed slots, so delimiters inside them are read as part of them."""
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    glossary = toys.toy.glossary()
    groups = ("North | Sea (St. Kilda: west) @ 56°N", "Gulf, at 56.5°N, 3.1°E",
              "Zone at 06:12:19 UTC on 12 April 2026", "Reach: those were the figures", "Sea, reading")
    sources = ("by the sea", "The", "ops; team")
    names = {g: groups[i % len(groups)] for i, g in enumerate(sorted({r.group for r in inst.reports}))}
    owners = {s: sources[i % len(sources)] for i, s in enumerate(sorted({r.source for r in inst.reports}))}
    awkward = [replace(r, group=names[r.group], source=owners[r.source],
                       place="Pier 3. Location given: east" if i % 2 else None) for i, r in enumerate(inst.reports)]
    notices = [replace(n, reason="it was (not) real") if n.kind == "retraction" else n for n in inst.notices]
    for seed in range(12):
        rng = random.Random(seed)
        text = NEWS.render(RenderContext(awkward, notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                         NEWS.choose_style(rng, toys.toy), rng, toys.toy, glossary))
        data = NEWS.parse(text, glossary, toys.toy)
        assert all(map(same_items, (data.reports, data.notices), printed(awkward, notices)))


# ---------------------------------------------------------------------- notices and noise


def test_every_notice_wording_is_named_by_the_reading_rules_and_reads_back_alone(toys):
    """Whatever wording a notice is drawn with, the domain's rules name it, and so do both rule sets:
    the 0.2.0 words, and "was not a real <noun>" or "revised"."""
    dressed = toys.dressed.bind(toys.dressed.draw_surface(random.Random(1)))
    cases = (
        (EARTHQUAKES, RULE_WORDS_020,
         [Notice("retraction", "us7000abcd", COMPILED_AT - 99, reason="quarry blast"),
          Notice("correction", "us7000abcd", COMPILED_AT - 9, new_mag=4.2, old_mag=4.5, mag_type="mb")]),
        (toys.toy, None,
         [Notice("retraction", "by-12345", COMPILED_AT - 99, reason="probe fault"),
          Notice("correction", "by-12345", COMPILED_AT - 9, field="temp", new_value=1.5, old_value=-2.0,
                 qualifier="spar"),
          Notice("correction", "by-12345", COMPILED_AT - 5, field="wave", new_value=1.25, old_value=2.5)]),
        (dressed, None,
         [Notice("retraction", f"{dressed.id_prefix}12345", COMPILED_AT - 99, reason="buoy adrift"),
          Notice("correction", f"{dressed.id_prefix}12345", COMPILED_AT - 9, field="temp", new_value=-0.4,
                 old_value=3.0, qualifier="hull")]),
    )
    for domain, words, notices in cases:
        rules = " ".join(domain.rules_text().split()).lower()
        vocab = domain.trap_vocab
        if words is None:
            assert domain.rules_text() == record_rules(domain)
            words = {"retraction": (*RECORD_RULE_WORDS["retraction"], *vocab.retraction_notices),
                     "correction": (*RECORD_RULE_WORDS["correction"], *vocab.correction_notices)}
        assert all(word.lower() in rules for kind in words for word in words[kind])
        assert f"was not a real {domain.noun}" in rules
        glossary = domain.glossary()
        wordings = set()
        for seed in range(150):
            rng = random.Random(seed)
            text = NEWS.render(RenderContext([], notices, [], COMPILED_AT, 24, NEWS.choose_style(rng, domain), rng,
                                             domain, glossary))
            wordings.update(p for p in text.split("\n\n") if NEWS.parse(p, glossary, domain).notices)
        forms = set()
        for paragraph in wordings:
            (notice,) = NEWS.parse(paragraph, glossary, domain).notices  # each wording reads back on its own
            assert replace(notice, time=0) in [replace(n, time=0) for n in notices], paragraph
            lowered = paragraph.lower()
            assert any(word.lower() in lowered for word in words[notice.kind]), paragraph
            assert any(word in lowered for word in RULE_WORDS_020[notice.kind]), paragraph
            assert (f"was not a real {domain.noun}" if notice.kind == "retraction" else "revised") in lowered
            assert not lowered.startswith("editor's note"), paragraph  # the rules call those lines without data
            forms.update((kind, opening) for (kind, opening), phrase in NOTICE_FORMS.items()
                         if kind == notice.kind and paragraph.startswith(opening) and phrase in paragraph)
        assert forms == set(NOTICE_FORMS), sorted(set(NOTICE_FORMS) - forms)
        assert any(f'"{phrase}"' in w for w in wordings for phrase in vocab.retraction_notices)


@pytest.mark.parametrize("name", DOMAINS)
def test_lines_without_data_change_nothing(toys, name):
    """Only report and notice paragraphs carry data: every other paragraph reads as nothing, and an
    article without reports or notices reads as an empty digest."""
    domain = get_domain(name)
    inst = instances_of(domain, seed=8, n=1)[0]
    bound = domain.bind(inst.surface)
    glossary = bound.glossary()
    for rendering in inst.renderings:
        paragraphs = rendering.digest.split("\n\n")
        read = [NEWS.parse(p, glossary, bound) for p in paragraphs]
        assert all(len(d.reports) + len(d.notices) <= 1 for d in read)
        assert sum(len(d.reports) + len(d.notices) for d in read) == len(inst.reports) + len(inst.notices)
        quiet_lines = [p for p, d in zip(paragraphs, read) if d == DigestData([], [])]
        assert len(quiet_lines) >= 5 + len(inst.chatter)  # headline and byline, lede, quote, background, sign-off
        assert not any(re.search(id_pattern(glossary), p) for p in quiet_lines)  # no line without data names a record
    for chatter in inst.chatter:
        assert NEWS.parse(chatter, glossary, bound) == DigestData([], [])
    rng = random.Random(4)
    empty = NEWS.render(RenderContext([], [], inst.chatter, inst.compiled_at, inst.window_hours,
                                      NEWS.choose_style(rng, bound), rng, bound, glossary))
    assert NEWS.sniff(empty) and NEWS.parse(empty, glossary, bound) == DigestData([], [])


# ------------------------------------------------------------------------------- sniffing


def test_sniffing_accepts_every_article_and_rejects_every_other_format(toys, instances):
    for inst in instances:  # the legacy (0.2.0) digests of the session's run
        for rendering in inst.renderings:
            assert not NEWS.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for domain in (toys.toy, toys.dressed):  # classic digests of any other domain's records
        for inst in instances_of(domain, seed=6, n=1, genre=CLASSIC):
            for rendering in inst.renderings:
                assert not NEWS.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    article = instances_of(toys.flat, seed=6, n=1)[0].renderings[0].digest
    lines = article.split("\n")
    assert NEWS.sniff(article) and NEWS.sniff("\n\n" + article)
    assert [g.name for g in all_genres() if g.sniff(article)] == ["news"]
    others = [
        "", "news", "\n".join(lines[:1]),  # a headline alone
        "\n".join(lines[1:]),  # an article without its headline
        "\n".join([lines[0], *lines[2:]]),  # without its byline
        "\n".join([lines[1], lines[0], *lines[2:]]),  # byline first
        "\n".join("> " + line for line in lines),  # quoted in a reply
        "ref,source,status,time\nby-12345,ukmo,reviewed,2026-04-12T06:12:19Z\n",
        "| ref | source |\n|---|---|\n| by-12345 | ukmo |\n",
        "From: desk@example.org\nSubject: readings\n\nBy Maria Okafor, staff reporter | Published 2026-04-12T06:12:19Z",
        "[06:12] maria: readings are in\n[06:13] dan: copy",
        "Readings: every report, as each operator filed it\nBy Maria Okafor, staff reporter | Published yesterday",
    ]
    for text in others:
        assert not NEWS.sniff(text), text[:80]


def test_news_refuses_what_it_cannot_print_and_reads_records_through_the_glossary(toys):
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    glossary = toys.toy.glossary()
    style = dict(NEWS.choose_style(random.Random(0), toys.toy), headline=0)

    def render(reports, words=glossary):
        return NEWS.render(RenderContext(reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                         style, random.Random(0), toys.toy, words))

    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        render([replace(r, group="North\nSea") for r in inst.reports])  # no paragraph can hold a line break
    with pytest.raises(ValueError, match="does not recognise its own rendering"):
        render(inst.reports, replace(glossary, plural="readings\nand more"))  # nor can the headline
    with pytest.raises(ValueError, match="prints none"):  # a digest of records is read through its glossary ...
        NEWS.parse(inst.renderings[0].digest, None, toys.toy)
    quake = instances_of(EARTHQUAKES, seed=5, n=1)[0].renderings[0].digest  # ... earthquakes' words are fixed
    assert NEWS.parse(quake, None, EARTHQUAKES) == NEWS.parse(quake, EARTHQUAKES.glossary(), EARTHQUAKES)
    assert get_genre("news") is NEWS


def test_a_variant_in_the_news_genre_avoids_every_preset_of_the_original(toys):
    for domain in (EARTHQUAKES, toys.flat):
        for seed in range(25):
            original = NEWS.choose_style(random.Random(seed), domain)
            variant = NEWS.choose_style(random.Random(seed + 100), domain, avoid=original)
            assert ("coords" in original) == domain.has_location
            assert all(variant[key] != original[key] for key in PRESET_KEYS if key in original)
    for inst in instances_of(toys.dressed, seed=2):
        v0, v1, v2 = inst.renderings
        assert v1.style != v0.style and v2.style != v0.style and len({v0.digest, v1.digest, v2.digest}) == 3


# ----------------------------------------------------------------------------- whole runs


@pytest.fixture(scope="module")
def news_runs(tmp_path_factory):
    """generate (mix "<domain>:news") -> run (oracle, naive, mock) -> score -> report, for four instances of
    earthquakes, of the toy domain with a position, of toy-flat without one, and of toy-dressed with a
    surface drawn per instance."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        toy = domains.load_plugin(TOY_PLUGIN)
        module = sys.modules[type(toy).__module__]
        for variant in (module.FLAT, module.DRESSED):
            domains.REGISTRY.register(variant)
        runs = {}
        for name, seed in zip(DOMAINS, (41, 42, 43, 44)):
            run_dir = pipeline.generate(GenerateConfig(seed=seed, instances=4, mix=f"{name}:news"),
                                        tmp_path_factory.mktemp("runs") / name, log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[name] = run_dir
        yield runs


def test_news_runs_print_articles_that_read_back_exactly(news_runs, tmp_path):
    for name, run in news_runs.items():
        manifest = read_json(run / "manifest.json")
        assert manifest["config"]["mix"] == f"{name}:news" and manifest["snapshot"]["domain"] == name
        check = manifest["self_check"]
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        for inst in load_instances(run):
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["news"] * 3
            bound = get_domain(name).bind(inst.surface)
            for rendering in inst.renderings:
                prompt = rendering.prompt
                glossary = parse_glossary(prompt)
                assert glossary == bound.glossary() and prompt.count("<glossary>") == 1
                assert prompt.index(bound.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
                assert extract_digest(prompt) == rendering.digest and detect_genre(rendering.digest) is NEWS
                data = NEWS.parse(rendering.digest, glossary, bound)
                assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
            assert len({r.digest for r in inst.renderings}) == 3
            assert all(r.style != inst.renderings[0].style for r in inst.renderings[1:])
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_the_oracle_scores_100_percent_on_news_and_reproduces_every_key(news_runs):
    for name, run in news_runs.items():
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "news")}
        for inst in load_instances(run):
            for rendering in inst.renderings:
                for genre in ("news", None):  # named, or recognised by sniffing
                    assert solve(rendering.prompt, question_specs(inst), domain=name, genre=genre) == inst.answer_key


def test_the_naive_reader_falls_for_the_traps_and_only_for_them_in_news(news_runs):
    for name, run in news_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        conflict = "conflicting_magnitude" if name == "earthquakes" else "conflicting_value"
        planted = {t.kind for inst in load_instances(run) for t in inst.traps}  # (none to three retractions a digest)
        assert list(naive["by_kind"]) == [k for k in ("duplicate", "retraction", conflict) if k in planted]
        assert {"duplicate", conflict} <= planted
        assert models["naive"]["accuracy"]["rate"] < 1.0 == models["oracle"]["accuracy"]["rate"]
        assert models["mock"]["format"]["replies"] == 12 and models["mock"]["accuracy"]["n"] > 0
        instances = load_instances(run)
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], row
        for inst in instances:  # the naive reader reads the same lines in every rendering
            specs = question_specs(inst)
            naive_answers = [solve(r.prompt, specs, mode="naive", domain=name) for r in inst.renderings]
            assert naive_answers[0] == naive_answers[1] == naive_answers[2]


def test_earthquakes_from_a_frozen_feed_keep_their_place_notes_in_news(tmp_path):
    """USGS places ("23 km NE of Somewhere, Utah"), two-decimal magnitudes and network ID sequences, with
    the forecast question, from a frozen feed and without the network."""
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    run = pipeline.generate(GenerateConfig(seed=11, instances=2, source="usgs", snapshot_file=str(feed),
                                           mix="earthquakes:news"), tmp_path / "run", log=quiet)
    check = read_json(run / "manifest.json")["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    for inst in load_instances(run):
        assert inst.forecast is not None and all(r.place for r in inst.reports)
        for rendering in inst.renderings:
            assert "Location " in rendering.digest and inst.forecast["qid"] in rendering.prompt
            data = NEWS.parse(rendering.digest, parse_glossary(rendering.prompt), EARTHQUAKES)
            assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
            assert solve(rendering.prompt, question_specs(inst), domain="earthquakes") == inst.answer_key
