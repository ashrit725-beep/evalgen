"""The chat genre: a channel's log of "[HH:MM] user: message" lines, for earthquakes (0.2.0 Reports and
Notices) and for the Records of any domain, here the test-only toy domains (toy_domain.py, with and
without a location, and with a surface drawn per instance).

Every style reads back to exactly its reports and notices (duplicate notes, retractions, corrections
and second reports included); quotes, read-backs, acknowledgements and the rest change nothing; a
message's time stamp is when it was posted, never a record's time; every notice wording says what the
reading rules name; sniff claims every chat log and nothing else; and whole runs ("toy:chat",
"earthquakes:chat", and a frozen USGS feed) reach 100% for the oracle with the traps caught by it and
falling where they should for the naive reader."""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.digest import MONTHS
from evalgen.domains import get_domain
from evalgen.domains.base import record_rules
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import all_genres, detect_genre
from evalgen.genres.base import RenderContext, parse_glossary
from evalgen.genres.chat import (ACKS, CHAT, CLOCKS, COORD_PRESETS, CORRECTIONS, GREETINGS, HEADERS, NOISE_LEVELS,
                                 NOTICE_PLACES, OFF_TOPIC, PACES, QUOTE, QUOTE_COMMENTS, READBACKS, RETRACTIONS,
                                 TIME_PRESETS, UNIT_PRESETS, VOICE_PRESETS)
from evalgen.genres.classic import CLASSIC
from evalgen.models import Instance, Notice
from evalgen.ops import DigestData, execute
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

COMPILED_AT = 1_780_000_000
MESSAGE = re.compile(r"\[(?P<hh>\d{2}):(?P<mm>\d{2})\] (?P<nick>[A-Za-z][\w.-]*): (?P<body>.+)")
DAY = re.compile(r"--- \w{3} (?P<day>\d{1,2}) (?P<mon>\w{3}) (?P<year>\d{4}) ---")


def instances_of(domain, seed: int, n: int = 3, source: str = "synthetic", fetch=None) -> list[Instance]:
    """The first instances of a run about `domain` in the chat genre, built as the pipeline builds them."""
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, source, log, fetch=fetch)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx, used, usage = Context(snapshot.compiled_at, snapshot.window_hours), set(), {}
    return [build_instance(i, snapshot, new, log, GenerateConfig(seed=seed), ctx, used, usage, domain=domain,
                           genre_plan=(CHAT,) * 3) for i in range(n)]


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
    """Every combination of voice, notice and pace presets, with every preset of the other dimensions taken
    in turn, each with a rendering stream of its own."""
    rng = random.Random(seed)
    combos = [(v, n, p) for v in sorted(VOICE_PRESETS) for n in sorted(NOTICE_PLACES) for p in sorted(PACES)]
    for i, (voices, notices, pace) in enumerate(combos):
        style = {"units": sorted(UNIT_PRESETS)[i % 3], "time": sorted(TIME_PRESETS)[(i // 2) % 3]}
        if domain.has_location:
            style["coords"] = sorted(COORD_PRESETS)[(i // 3) % 3]
        style.update(voices=voices, pace=pace, clock=CLOCKS[(i // 3) % 2], notices=notices,
                     noise=sorted(NOISE_LEVELS)[(i // 6) % 2], header=(i + 1) % len(HEADERS))
        yield style, random.Random(rng.random())


def render(inst, domain, style, rng) -> str:
    return CHAT.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                     style, rng, domain, domain.glossary()))


def voice_of(body: str, noun: str) -> str | None:
    """The voice a report's message is in, as far as a test needs to tell them apart."""
    if "`" in body:
        return "paste"
    if f" · {noun} " in body:
        return "relay"
    if f" reports {noun} " in body:
        return "casual"
    return "terse" if " — " in body else None


def message_times(text: str) -> list[tuple[int, str, str]]:
    """(when it was posted, to the minute, in UTC; nick; body) of every message of a log, from its stamp, its
    day line and the clock the header names."""
    clock = re.search(r"UTC(?:(?P<sign>[+-])(?P<hours>\d{2}):00)?[.)]?$", text.split("\n", 1)[0])
    offset = 0 if clock["sign"] is None else int(clock["hours"]) * (1 if clock["sign"] == "+" else -1)
    out, day = [], None
    for line in text.split("\n"):
        found = DAY.fullmatch(line)
        if found is not None:
            day = datetime(int(found["year"]), MONTHS.index(found["mon"]) + 1, int(found["day"]), tzinfo=timezone.utc)
            continue
        found = MESSAGE.fullmatch(line)
        if found is not None:
            local = day + timedelta(hours=int(found["hh"]), minutes=int(found["mm"]))
            out.append((int((local - timedelta(hours=offset)).timestamp()), found["nick"], found["body"]))
    return out


# ------------------------------------------------------------------------- reading back


@pytest.mark.parametrize("name", ["earthquakes", "toy", "toy-flat", "toy-dressed"])
def test_chat_reads_back_every_report_and_notice_exactly_in_every_style(toys, name):
    domain = get_domain(name)
    seen_styles, seen_voices, forms, types = Counter(), Counter(), Counter(), Counter()
    for seed in (27, 28):
        for inst in instances_of(domain, seed=seed, n=3):
            bound = domain.bind(inst.surface)
            glossary = bound.glossary()
            reports, notices = printed(inst.reports, inst.notices)
            forms.update((t.kind, t.form) for t in inst.traps)
            for rendering in inst.renderings:  # the styles choose_style draws
                assert rendering.genre == "chat" and detect_genre(rendering.digest) is CHAT
                data = CHAT.parse(rendering.digest, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices)
            for style, rng in styles(bound, inst.seed):  # and every preset of every dimension
                text = render(inst, bound, style, rng)
                data = CHAT.parse(text, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), style
                assert CHAT.sniff(text) and detect_genre(text) is CHAT
                assert not CLASSIC.sniff(text) and [g.name for g in all_genres() if g.sniff(text)] == ["chat"]
                seen_styles.update(f"{key}={value}" for key, value in style.items())
                seen_voices.update(voice_of(body, glossary.noun) for _, _, body in message_times(text))
                types.update(type(r).__name__ for r in data.reports)
                types.update("0.2.0 correction" for n in data.notices if n.new_mag is not None)
                types.update("correction" for n in data.notices if n.field is not None)
    if name == "earthquakes":  # the 0.2.0 types, exactly
        assert set(types) == {"Report", "0.2.0 correction"}
    else:
        assert set(types) == {"RecordReport", "correction"}
    dimensions = {"units": UNIT_PRESETS, "time": TIME_PRESETS, "voices": VOICE_PRESETS, "pace": PACES, "clock": CLOCKS,
                  "notices": NOTICE_PLACES, "noise": NOISE_LEVELS, "header": range(len(HEADERS))}
    if domain.has_location:
        dimensions["coords"] = COORD_PRESETS
    for key, options in dimensions.items():
        assert {f"{key}={value}" for value in options} <= set(seen_styles), key
    assert {"relay", "casual", "terse", "paste"} <= set(seen_voices)
    assert {("duplicate", "same_id"), ("duplicate", "cross_ref"), ("retraction", "phantom")} <= set(forms)
    assert {form for kind, form in forms if kind.startswith("conflicting")} == {"correction", "second_report"}


def test_frozen_usgs_earthquakes_with_place_notes_read_back_exactly():
    raw, _ = fake_feed(n=90, seed=5)
    insts = instances_of(EARTHQUAKES, seed=12, n=3, source="usgs", fetch=lambda: (raw, "file:feed"))
    assert all(r.place for inst in insts for r in inst.reports)  # "12 km NE of Somewhere, Central Chile"
    for inst in insts:
        reports, notices = printed(inst.reports, inst.notices)
        for style, rng in styles(EARTHQUAKES, inst.seed):
            data = CHAT.parse(render(inst, EARTHQUAKES, style, rng), EARTHQUAKES.glossary(), EARTHQUAKES)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style


def test_repeats_and_chatter_change_nothing(toys):
    """Quotes repeat an earlier report or notice word for word and read-backs retype an earlier report;
    neither is read as a new one, nor is any acknowledgement, greeting, off-topic remark or chatter line."""
    kinds = Counter()
    for domain in (EARTHQUAKES, toys.toy):
        glossary = domain.glossary()
        for inst in instances_of(domain, seed=31, n=2):
            for style, rng in styles(domain, inst.seed):
                text = render(inst, domain, style, rng)
                messages = message_times(text)
                bodies = [body for _, _, body in messages]
                for at, _, body in messages:
                    alone = CHAT.parse(f"[00:00] someone: {body}", glossary, domain)
                    if body.startswith(QUOTE):
                        kinds["quote"] += 1
                        quoted = body[len(QUOTE):]
                        earlier = [b for t, _, b in messages if t <= at and (b == quoted or quoted.endswith(": " + b))]
                        assert earlier, body  # word for word, of a message posted before it
                        assert alone == DigestData([], [])
                    elif body.startswith(READBACKS):
                        kinds["read-back"] += 1
                        retyped = body.split(": ", 1)[1]
                        (report,) = CHAT.parse(f"[00:00] someone: {retyped}", glossary, domain).reports
                        assert report in printed(inst.reports, [])[0]  # the same ID, source, status and values
                        assert alone == DigestData([], [])
                    elif re.sub(r"^@[\w.-]+ ", "", body) in ACKS + GREETINGS + OFF_TOPIC + QUOTE_COMMENTS + tuple(
                            inst.chatter):
                        kinds["mention" if body.startswith("@") else "noise"] += 1
                        assert alone == DigestData([], [])
                    else:
                        assert len(alone.reports) + len(alone.notices) == 1, body  # every other message is data
                assert sum(1 for b in bodies if b.startswith(QUOTE)) >= 1
                assert len(CHAT.parse(text, glossary, domain).reports) == len(inst.reports)
                assert any(line.startswith("--- ") for line in text.split("\n")[1:])  # day lines
                assert re.search(r"\] \* [\w.-]+ joined #", text)
    assert kinds["quote"] and kinds["read-back"] and kinds["noise"] and kinds["mention"]


def test_a_message_is_stamped_when_it_was_posted_never_with_a_records_time(toys):
    differ = checked = 0
    for domain in (EARTHQUAKES, toys.toy, toys.flat):
        glossary = domain.glossary()
        for inst in instances_of(domain, seed=33, n=2):
            for style, rng in styles(domain, inst.seed):
                for at, _, body in message_times(render(inst, domain, style, rng)):
                    data = CHAT.parse(f"[00:00] someone: {body}", glossary, domain)
                    for item in data.reports + data.notices:  # posted when it was recorded or issued, or after
                        assert at >= item.time - item.time % 60, (body, at)
                        checked += 1
                        differ += at // 60 != item.time // 60
    assert checked > 1000 and differ > 0.6 * checked  # so the stamp is no stand-in for the time a report gives


def test_the_log_is_drawn_from_the_rendering_stream_alone(toys):
    inst = instances_of(toys.toy, seed=3, n=1)[0]
    style = inst.renderings[0].style
    assert render(inst, toys.toy, style, random.Random(9)) == render(inst, toys.toy, style, random.Random(9))
    assert render(inst, toys.toy, style, random.Random(9)) != render(inst, toys.toy, style, random.Random(10))
    for rendering in inst.renderings[1:]:  # a variant in the same genre avoids every preset of the original's
        assert all(rendering.style[key] != style[key] for key in style if key != "header")


# ------------------------------------------------------------------------------- notices


def test_every_notice_wording_says_what_the_reading_rules_name(toys):
    """Whatever wording and phrase a notice is drawn with, it says RETRACTION or CORRECTION, which the rules of
    every record domain and the 0.2.0 earthquake rules both name; it quotes the domain's own phrases where they
    fit, and each wording reads back on its own."""
    dressed = toys.dressed.bind(toys.dressed.draw_surface(random.Random(1)))
    earthquake_notices = [Notice("retraction", "us7000abcd", COMPILED_AT - 99, reason="quarry blast"),
                          Notice("correction", "nc75012345", COMPILED_AT - 9, new_mag=4.2, old_mag=4.5, mag_type="mb")]
    record_notices = [Notice("retraction", "by-12345", COMPILED_AT - 99, reason="probe fault"),
                      Notice("correction", "by-54321", COMPILED_AT - 9, field="temp", new_value=1.5, old_value=-2.0,
                             qualifier="spar"),
                      Notice("correction", "by-11111", COMPILED_AT - 5, field="wave", new_value=1.25, old_value=2.5)]
    for domain, rules, notices in ((EARTHQUAKES, READING_RULES, earthquake_notices),
                                   (toys.toy, record_rules(toys.toy), record_notices),
                                   (dressed, record_rules(dressed), record_notices)):
        rules = " ".join(rules.split())
        assert "a RETRACTION notice" in rules and "a CORRECTION notice" in rules
        glossary = domain.glossary()
        wordings = set()
        for seed in range(150):
            style = CHAT.choose_style(random.Random(seed), domain)
            text = CHAT.render(RenderContext([], notices, [], COMPILED_AT, 24, style, random.Random(seed), domain,
                                             glossary))
            data = CHAT.parse(text, glossary, domain)
            assert same_items(data.notices, notices)
            for line in text.split("\n"):
                body = re.sub(r"^(?:\[\d{2}:\d{2}\] [\w.-]+|\U0001f4cc Pinned by [\w.-]+): ", "", line)
                if body != line and CHAT.parse(f"[00:00] x: {body}", glossary, domain).notices:
                    wordings.add(body)
        forms = Counter()
        for body in wordings:
            (notice,) = CHAT.parse(f"[00:00] x: {body}", glossary, domain).notices
            assert replace(notice, time=0) in [replace(n, time=0) for n in notices], body
            keyword = "RETRACTION" if notice.kind == "retraction" else "CORRECTION"
            assert keyword in body, body
            forms[notice.kind] += 1
        retractions = [b for b in wordings if "RETRACTION" in b]
        assert len(retractions) >= len(RETRACTIONS) and len(wordings) - len(retractions) >= len(CORRECTIONS)
        vocab = domain.trap_vocab
        for phrase in vocab.retraction_notices:  # the domain's own phrases, quoted where they fit
            assert any(f" {phrase} (" in body for body in retractions), phrase
        assert any(f" {phrase} " in body for body in wordings for phrase in vocab.correction_notices
                   if phrase.endswith(" to"))


def test_every_notice_wording_is_the_notice_itself(toys):
    """A notice opens with its keyword and the time it was issued ("heads up: " at most before them), so that
    it reads as the notice itself: never as a pointer to a notice printed elsewhere ("see the RETRACTION
    issued ...", which no log holds), nor as a relay of one, which a reader could take for a repeat (rule
    7). In a rendered log, every notice names its keyword and its time before the record it is about."""
    opening = re.compile(r"(?:heads up: )?(?:RETRACTION|CORRECTION) (?:\[\{when\}\]|issued \{when\}): ")
    for form in (*RETRACTIONS, *CORRECTIONS):
        assert opening.match(form) and form.index("{when}") < form.index("{ref}") and "see " not in form, form
    notices = [Notice("retraction", "by-12345", COMPILED_AT - 99, reason="probe fault"),
               Notice("correction", "by-54321", COMPILED_AT - 9, field="temp", new_value=1.5, old_value=-2.0,
                      qualifier="spar")]
    glossary = toys.toy.glossary()
    openings = set()
    for seed in range(80):
        style = CHAT.choose_style(random.Random(seed), toys.toy)
        text = CHAT.render(RenderContext([], notices, [], COMPILED_AT, 24, style, random.Random(seed), toys.toy,
                                         glossary))
        for line in text.split("\n"):
            body = re.sub(r"^(?:\[\d{2}:\d{2}\] [\w.-]+|\U0001f4cc Pinned by [\w.-]+): ", "", line)
            for notice in notices:
                if body == line or notice.target not in body or body.startswith(QUOTE):
                    continue  # not a message about this notice's record, or a quote of one
                keyword = "RETRACTION" if notice.kind == "retraction" else "CORRECTION"
                head = body[:body.index(notice.target)]
                assert keyword in head and re.search(r"\d{2}:\d{2}:\d{2}", head), body
                openings.add(head.split(keyword)[0] + keyword)
    assert openings == {"RETRACTION", "CORRECTION", "heads up: RETRACTION", "heads up: CORRECTION"}


# ------------------------------------------------------------------------------- sniffing


def test_sniff_claims_chat_logs_and_nothing_else(toys, instances):
    for inst in instances[:4]:  # legacy classic digests (0.2.0)
        for rendering in inst.renderings:
            assert not CHAT.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    toy = instances_of(toys.toy, seed=8, n=1)[0]
    for style, rng in list(styles(toys.toy, 1))[:4]:  # generic classic digests
        classic = CLASSIC.render(RenderContext(toy.reports, toy.notices, toy.chatter, toy.compiled_at,
                                               toy.window_hours, CLASSIC.choose_style(rng, toys.toy), rng, toys.toy,
                                               toys.toy.glossary()))
        assert not CHAT.sniff(classic) and detect_genre(classic) is CLASSIC
    chat = toy.renderings[0].digest
    lines = chat.split("\n")
    others = [
        "", "hello", lines[0],  # nothing, a word, a header without a message
        "\n".join(lines[2:]),  # messages without a header
        "\n".join(lines[:3] + ["[UKMO] by-12345 | 2026-04-12T06:12:19Z | North Sea | 12.3 °C | reviewed"] + lines[3:]),
        "From: dana@example.org\nSubject: readings\n\nHi all,\nsee below.\n> [06:12] mira: ok",
        "| ref | source | value |\n|---|---|---|\n| by-12345 | ukmo | 12.3 |",
        "DISPATCH LOG\n[06:12] UNIT 7: copy, over\n[06:13] DISPATCH: roger",
        "Chat log of the desk\n[06:12] mira: ok",
        "\n".join("> " + line for line in lines),  # the whole log quoted, as an email reply quotes it
    ]
    for text in others:
        assert not CHAT.sniff(text), text[:120]
    assert CHAT.sniff(chat) and CHAT.sniff("\n\n" + chat + "\n") and CHAT.sniff(chat.replace("\n", "\n  "))


# ---------------------------------------------------------------------------- odd words


def test_odd_words_are_pasted_and_what_no_message_can_hold_is_refused(toys):
    inst = instances_of(toys.toy, seed=5, n=1)[0]
    style = dict(inst.renderings[0].style, voices="people", notices="posted")
    glossary = toys.toy.glossary()
    # Words that would make a voice ambiguous: a quote marker, a line break, every delimiter, a backtick.
    odd = [replace(r, source=f"> {r.source} · reading",
                   group="North\nSea — (St. Kilda: west), recorded `x`; place: y") for r in inst.reports]
    text = CHAT.render(RenderContext(odd, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                     random.Random(0), toys.toy, glossary))
    data = CHAT.parse(text, glossary, toys.toy)
    assert same_items(data.reports, printed(odd, [])[0]) and CHAT.sniff(text)
    assert all("`" in body for _, _, body in message_times(text) if "North" in body and "Sea" in body
               and not body.startswith(READBACKS + (QUOTE,)))  # every report pasted, its words as quoted tokens
    unlike = [replace(r, ref=r.ref + "x") if r.role == "primary" else r for r in inst.reports]  # not an ID
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        CHAT.render(RenderContext(unlike, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                  random.Random(0), toys.toy, glossary))
    broken = [replace(n, reason="probe\nfault") if n.kind == "retraction" else n for n in inst.notices]
    with pytest.raises(ValueError, match="cannot be printed so that it reads back exactly"):
        CHAT.render(RenderContext(inst.reports, broken, inst.chatter, inst.compiled_at, inst.window_hours, style,
                                  random.Random(0), toys.toy, glossary))
    with pytest.raises(ValueError, match="prints none"):  # a record digest is read through its prompt's glossary
        CHAT.parse(inst.renderings[0].digest, None, toys.toy)
    quakes = instances_of(EARTHQUAKES, seed=5, n=1)[0]  # an earthquake one, through the domain's own if need be
    assert CHAT.parse(quakes.renderings[0].digest, None, EARTHQUAKES) == CHAT.parse(
        quakes.renderings[0].digest, EARTHQUAKES.glossary(), EARTHQUAKES)


def test_the_genres_own_words_carry_no_data():
    for text in ACKS + GREETINGS + OFF_TOPIC + QUOTE_COMMENTS:
        assert not re.search(r"\d|RETRACT|CORRECT|revised|withdrawn|reports ", text, re.IGNORECASE), text


# ----------------------------------------------------------------------------- whole runs


@pytest.fixture(scope="module")
def chat_runs(tmp_path_factory):
    """generate -> run (oracle, naive, mock) -> score -> report, in the chat genre: four instances of the toy
    domain, four of synthetic earthquakes, and two of earthquakes replayed from a frozen USGS feed (place notes,
    forecasts)."""
    raw, _ = fake_feed()
    feed = tmp_path_factory.mktemp("feed") / "all_day.geojson"
    feed.write_bytes(raw)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        domains.load_plugin(TOY_PLUGIN)
        runs = {}
        for label, cfg in (("toy", GenerateConfig(seed=41, instances=4, mix="toy:chat")),
                           ("earthquakes", GenerateConfig(seed=42, instances=4, mix="earthquakes:chat")),
                           ("usgs", GenerateConfig(seed=43, instances=2, source="usgs", snapshot_file=str(feed),
                                                   mix="earthquakes:chat"))):
            run_dir = pipeline.generate(cfg, tmp_path_factory.mktemp("runs") / label, log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[label] = run_dir
        yield runs


def test_chat_runs_read_back_every_prompt_and_reach_100_percent(chat_runs, tmp_path):
    for label, run in chat_runs.items():
        manifest = read_json(run / "manifest.json")
        name = manifest["snapshot"]["domain"]
        check = manifest["self_check"]
        assert manifest["config"]["mix"] == f"{name}:chat"
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "chat")}
        for inst in load_instances(run):
            domain = get_domain(name).bind(inst.surface)
            glossary = domain.glossary()
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["chat"] * 3
            assert (inst.forecast is not None) == (label == "usgs")
            for rendering in inst.renderings:
                prompt = rendering.prompt
                assert detect_genre(extract_digest(prompt)) is CHAT and parse_glossary(prompt) == glossary
                assert prompt.index(domain.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
                data = CHAT.parse(rendering.digest, glossary, domain)
                assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
                assert solve(prompt, question_specs(inst), domain=name) == inst.answer_key  # read by sniffing
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_chat_runs_catch_the_traps_and_the_naive_reader_falls_only_for_them(chat_runs):
    mocked = []  # (right, questions) of the simulated model in each run: a run of two instances may see it err nowhere
    for label, run in chat_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        assert models["oracle"]["accuracy_all_renderings"]["rate"] == 1.0
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        kinds = set(naive["by_kind"])
        assert kinds <= {"duplicate", "retraction", "conflicting_value", "conflicting_magnitude"} and len(kinds) >= 2
        instances = load_instances(run)
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], (label, row)
        domain = get_domain(read_json(run / "manifest.json")["snapshot"]["domain"])
        for inst in instances:  # the naive reader sees exactly the digest's lines: no repeat counts as a report
            data = DigestData(inst.reports, inst.notices)
            want = {q.qid: execute(q.chain, data, skip=("dedupe",), units=domain.units, resolver=domain.resolve)
                    for q in inst.questions}
            for rendering in inst.renderings:
                assert solve(rendering.prompt, question_specs(inst), mode="naive", domain=inst.domain) == want
        mock = models["mock"]
        assert mock["format"]["replies"] == 3 * len(instances)
        mocked.append((mock["accuracy"]["k"], mock["accuracy"]["n"]))
    assert 0 < sum(k for k, _ in mocked) < sum(n for _, n in mocked)
