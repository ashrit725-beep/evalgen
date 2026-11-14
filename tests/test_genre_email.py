"""The email genre (evalgen/genres/email.py): a digest written as an email thread, for every domain.

Its renderings read back to exactly their reports and notices in every style, for the earthquake
domain (whose lines stay 0.2.0 Reports and Notices) and for the test-only toy domains (toy_domain.py:
with and without a location, and with a surface drawn per instance). Quoted text only ever repeats an
earlier message, every notice arrives as a forwarded message in a wording the reading rules name, and
message dates, greetings, signatures and remarks change nothing. Its sniff recognises every style and
nothing else, classic digests included, and whole runs in the genre ("toy:email", "earthquakes:email")
score the oracle at 100% and read the traps as the reading rules say.
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from evalgen import domains, pipeline, scoring
from evalgen.digest import MONTHS
from evalgen.domains import get_domain
from evalgen.domains.base import GlossaryField, RecordReport, record_rules
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import all_genres, detect_genre
from evalgen.genres.base import TIME_FORMATS, RenderContext, parse_glossary, parse_time
from evalgen.genres.classic import CLASSIC
from evalgen.genres import email
from evalgen.genres.email import COORD_PRESETS, EMAIL, STYLE, TIME_PRESETS, UNIT_PRESETS
from evalgen.models import Instance, Notice, Report
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

DOMAINS = ("toy", "toy-flat", "toy-dressed", "earthquakes")  # with and without a location, a drawn surface, 0.2.0 types
COMPILED_AT = 1_780_000_000
FORWARD_MARKERS = ("---------- Forwarded message ----------", "Begin forwarded message:")
HEADER = re.compile(r"(?:From|To|Cc|Date|Subject): .+")
NOTICE_FORMS = {"RETRACTION, issued": "retraction", "RETRACTION NOTICE (": "retraction",
                "Notice of RETRACTION,": "retraction", "CORRECTION, issued": "correction",
                "CORRECTION NOTICE (": "correction", "Notice of a revised value,": "correction"}


def instances_of(domain, seed: int, n: int = 2, genres=(EMAIL,) * 3) -> list[Instance]:
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


def styles(domain, seed: int):
    """Twelve styles in which every option of every dimension appears, each with a stream of its own."""
    rng = random.Random(seed)
    options = {"units": sorted(UNIT_PRESETS), "time": sorted(TIME_PRESETS)}
    if domain.has_location:
        options["coords"] = sorted(COORD_PRESETS)
    options.update((key, sorted(names)) for key, names in STYLE.items())
    for i in range(12):
        yield {key: names[i % len(names)] for key, names in options.items()}, random.Random(rng.random())


def render(inst: Instance, domain, style, rng, glossary="domain") -> str:
    """`inst` rendered in the email genre with `style`, through `domain`'s glossary (or none)."""
    glossary = domain.glossary() if glossary == "domain" else glossary
    return EMAIL.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at, inst.window_hours,
                                      style, rng, domain, glossary, inst.chatter_after or ()))


def unquote(line: str, quotes: str) -> str:
    """A quoted line as the message it quotes printed it ("compact" quotes put ">" before a line, the
    others "> ", and a blank line is quoted as ">")."""
    if line == ">":
        return ""
    return line[1:] if quotes == "compact" else line[2:]


# ------------------------------------------------------------------------------ round trips

TIME_SLOTS = {"iso": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "iso_space": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC",
              "rfc": r"\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} \+0000",
              "prose_utc": r"\d{2}:\d{2}:\d{2} UTC on \d{1,2} [A-Z][a-z]+ \d{4}",
              "iso_offset": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:00",
              "local12": r"\d{1,2}:\d{2}:\d{2} [AP]M local time \(UTC[+-]\d{2}:00\) on \w{3} \d{1,2}, \d{4}",
              "local24": r"\d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:00\) \d{1,2} \w{3} \d{4}"}
FEATURES = {"bullet": re.compile(r"(?:[-*•]|\d{1,3}\.) \S+ \(.+"), "numbered": re.compile(r"\d{1,3}\. \S+ \(.+"),
            "sentence": re.compile(r"At .+, the .+ recorded .+"), "block": re.compile(r"    Recorded: .+"),
            "gmail quote": re.compile(r"On .+ wrote:"), "outlook quote": re.compile(r"-----Original Message-----"),
            "compact quote": re.compile(r">[^ ].*"), "rule": re.compile(r"_{32}"), "cc": re.compile(r"Cc: .+")}


def features(text: str) -> set[str]:
    """What a thread shows of its style: item styles, quotes, forwards, notice forms, the formats of the
    times its items give, and the order of its header fields."""
    lines = text.split("\n")
    found = {"date first"} if lines[0].startswith("Date: ") else set()
    for line in lines:
        found.update(name for name, pattern in FEATURES.items() if pattern.fullmatch(line))
        found.update(marker for marker in FORWARD_MARKERS if line == marker)
        found.update(form for form in NOTICE_FORMS if line.startswith(form))
        if not line.startswith(">") and (line.startswith("    Recorded: ") or FEATURES["bullet"].match(line)
                                         or FEATURES["sentence"].match(line)):
            found.update(fmt for fmt, slot in TIME_SLOTS.items()
                         if any(parse_time(piece) is not None for piece in re.findall(slot, line)))
    return found


@pytest.mark.parametrize("name", DOMAINS)
def test_email_reads_back_every_report_and_notice_exactly_in_every_style(toys, name):
    domain = get_domain(name)
    seen: Counter[str] = Counter()
    kinds = set()  # of the notices planted (every wording of each kind: see the notice wording test)
    for seed in (3, 4, 5, 6):
        for inst in instances_of(domain, seed):
            bound = domain.bind(inst.surface)
            glossary = bound.glossary()
            reports, notices = printed(inst.reports, inst.notices)
            kinds.update(n.kind for n in notices)
            for rendering in inst.renderings:  # as the pipeline rendered them
                assert rendering.genre == "email"
                data = EMAIL.parse(rendering.digest, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices)
            for style, rng in styles(bound, inst.seed):
                text = render(inst, bound, style, rng)
                data = EMAIL.parse(text, glossary, bound)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), style
                assert EMAIL.sniff(text) and detect_genre(text) is EMAIL
                seen.update(features(text))
    forms = {form for form, kind in NOTICE_FORMS.items() if kind in kinds}
    wanted = {*FEATURES, "date first", *FORWARD_MARKERS, *forms, *TIME_FORMATS}
    assert "retraction" in kinds and wanted <= set(seen), wanted - set(seen)
    if domain.has_location:  # positions in both formats
        texts = [render(inst, domain, style, rng) for inst in instances_of(domain, 5, n=1)
                 for style, rng in styles(domain, 5)]
        assert any("°N" in t or "°S" in t for t in texts) and any(re.search(r"\(-?\d+\.\d+, -?\d+", t) for t in texts)


def test_earthquake_threads_read_back_to_020_reports_and_notices(instances):
    """Earthquake reports and notices keep their 0.2.0 types, with or without a glossary in the prompt: a
    legacy instance (the small run's) rendered as a thread reads back to its own Reports and Notices."""
    glossary = EARTHQUAKES.glossary()
    corrections = 0
    for inst in instances:
        reports, notices = printed(inst.reports, inst.notices)
        for style, rng in list(styles(EARTHQUAKES, inst.seed))[:4]:
            text = render(inst, EARTHQUAKES, style, rng, glossary=None)  # the domain's own words
            for read in (EMAIL.parse(text, glossary, EARTHQUAKES), EMAIL.parse(text, None, EARTHQUAKES)):
                assert same_items(read.reports, reports) and same_items(read.notices, notices)
                assert {type(r) for r in read.reports} == {Report}
                for n in read.notices:
                    assert (n.field, n.new_value, n.old_value, n.qualifier) == (None,) * 4
                    corrections += n.kind == "correction" and n.new_mag is not None and n.mag_type is not None
    assert corrections > 0


def test_any_number_of_fields_reads_back(toys):
    """Nothing in the genre assumes two fields: a glossary of one field, and one of three (the third
    without units), read back in every style."""
    base = toys.flat.glossary()
    records = toys.flat.synthetic(random.Random(3), 8, COMPILED_AT, 168)
    for fields in ((base.fields[0],), (*base.fields, GlossaryField("period", "wave period", None, (), 1))):
        glossary = replace(base, fields=fields)
        reports = [RecordReport(r.id, r.source, "reviewed", r.time, None, None, r.group,
                                {f.key: r.values.get(f.key, 7.5) for f in fields}, dict(r.qualifiers))
                   for r in records]
        notices = [Notice("retraction", records[0].id, COMPILED_AT - 99, reason="probe fault"),
                   Notice("correction", records[1].id, COMPILED_AT - 9, field=fields[-1].key, new_value=12.5,
                          old_value=9.0)]
        for style, rng in styles(toys.flat, 16):
            text = EMAIL.render(RenderContext(reports, notices, [], COMPILED_AT, 168, style, rng, toys.flat, glossary))
            data = EMAIL.parse(text, glossary, toys.flat)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style


def test_rendering_follows_the_rendering_stream_exactly(toys):
    """An instance's renderings are drawn from their own streams: drawing the style and the thread again
    from a rendering's seed gives the same text."""
    for domain in (toys.toy, EARTHQUAKES):
        inst = instances_of(domain, seed=10, n=1)[0]
        original = inst.renderings[0]
        for rendering in inst.renderings:
            rng = random.Random(rendering.seed)
            avoid = None if rendering is original else original.style
            style = EMAIL.choose_style(rng, domain, avoid=avoid)
            assert style == rendering.style
            assert render(inst, domain, style, rng) == rendering.digest


# --------------------------------------------------------------------- what a thread holds


@pytest.mark.parametrize("name", ["toy", "earthquakes"])
def test_quoted_text_only_repeats_earlier_messages(toys, name):
    """A quote repeats lines of the message before it, which the thread printed earlier. Read as if
    unquoted, the quotes would add reports and notices, but only ones the thread already carries."""
    domain = get_domain(name)
    glossary = domain.glossary()
    repeats = 0
    for inst in instances_of(domain, seed=6):
        for style, rng in styles(domain, inst.seed):
            text = render(inst, domain, style, rng)
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith(">") and unquote(line, style["quotes"]) not in ("", "[...]"):
                    assert unquote(line, style["quotes"]) in lines[:i], line
            data = EMAIL.parse(text, glossary, domain)
            loose = EMAIL.parse("\n".join(unquote(line, style["quotes"]) if line.startswith(">") else line
                                          for line in lines), glossary, domain)
            # (a quote cut short may stop inside a pasted block, before its note: compare what it names)
            assert {r.ref for r in loose.reports} == {r.ref for r in data.reports}
            assert {(n.kind, n.target) for n in loose.notices} == {(n.kind, n.target) for n in data.notices}
            assert len(loose.reports) >= len(data.reports) and len(loose.notices) >= len(data.notices)
            repeats += len(loose.reports) - len(data.reports) + len(loose.notices) - len(data.notices)
    assert repeats > 0


def _message_parts(message: str) -> tuple[str, str, str, bool]:
    """A message of a thread: its sender, its subject, the first line of its own text, and whether it quotes."""
    lines = message.split("\n")
    blank = lines.index("")
    headers = dict(line.split(": ", 1) for line in lines[:blank])
    first = next(line for line in lines[blank:] if line)
    return headers["From"], headers["Subject"], first, any(line.startswith(">") for line in lines)


def test_a_reply_quotes_what_it_answers_under_that_messages_subject(toys):
    """A reply quotes the message before it. One that only acknowledges always does, whatever it answers,
    and comes from someone else ("Thanks, received." once quoted its own sender's message); it is titled after
    the message it answers ("Re: " and its subject, or its subject as it is when that is a reply's already):
    after a forward, which is titled after its notice, "Re: Fwd: CORRECTION: ...", where it was titled after
    the thread above a quote headed by the forward's subject, and so after an acknowledgement of a forward,
    which was titled after the thread. A message of reports is titled after the thread and never quotes a
    forward."""
    after_forward = acks = 0
    for domain in (toys.toy, EARTHQUAKES):
        for inst in instances_of(domain, seed=9, n=3):
            for style, rng in styles(domain, inst.seed):
                parts = [_message_parts(message) for message in MESSAGE_BREAK.split(render(inst, domain, style, rng))]
                thread = parts[0][1]
                for (sender, subject, first, quotes), previous in zip(parts[1:], parts):
                    forward = previous[1].startswith("Fwd: ")
                    if first in email.ACKS:
                        assert quotes and sender != previous[0], (sender, previous[0])
                        answered = previous[1]
                        assert subject == (answered if answered.startswith("Re: ") else f"Re: {answered}"), (
                            subject, previous[1])
                        assert subject == f"Re: {thread}" or "Fwd: " in subject, (subject, thread)
                        after_forward += forward
                        acks += 1
                    elif not subject.startswith("Fwd: "):  # a message of reports
                        assert subject == f"Re: {thread}" and not (quotes and forward), (subject, previous[1])
    assert acks > 10 and after_forward > 0, (acks, after_forward)


@pytest.mark.parametrize("name", ["toy-flat", "earthquakes"])
def test_every_notice_arrives_as_a_forwarded_message(toys, name):
    domain = get_domain(name)
    glossary = domain.glossary()
    for inst in instances_of(domain, seed=7):
        for style, rng in styles(domain, inst.seed):
            lines = render(inst, domain, style, rng).split("\n")
            found = 0
            for i, line in enumerate(lines):
                if line.startswith(">") or not EMAIL.parse(line, glossary, domain).notices:
                    continue
                found += 1
                j = i - 1  # back over the forwarded message's headers to its marker
                while lines[j] == "" or HEADER.fullmatch(lines[j]):
                    j -= 1
                assert lines[j] == {"gmail": FORWARD_MARKERS[0], "apple": FORWARD_MARKERS[1]}[style["forwards"]]
                assert lines[j + 1:i].count("") <= 2 and "Date: " in "".join(lines[j:i])
            assert found == len(inst.notices) > 0


def test_message_dates_greetings_signatures_and_remarks_change_nothing(toys):
    """A message's Date header is when it was sent, never a record's time: dating every message, quote and
    forward otherwise reads back the same. Header lines and the chatter carry no data, and messages are
    sent in thread order, each after what it carries."""
    for domain in (toys.toy, toys.flat, EARTHQUAKES):
        glossary = domain.glossary()
        for inst in instances_of(domain, seed=8, n=1):
            for style, rng in styles(domain, inst.seed):
                text = render(inst, domain, style, rng)
                data = EMAIL.parse(text, glossary, domain)
                moved = re.sub(r"^(Date|Sent): .*$", r"\1: Thu, 1 Jan 2026 00:00:00 +0000", text, flags=re.M)
                moved = re.sub(r"^On .+ wrote:$", "On Thu, 1 Jan 2026 at 00:00, A. Reader <a@b.example> wrote:", moved,
                               flags=re.M)
                assert moved != text and EMAIL.parse(moved, glossary, domain) == data
                for line in text.split("\n"):
                    if HEADER.fullmatch(line):
                        assert EMAIL.parse(line, glossary, domain) == DigestData([], [])
                for chatter in inst.chatter:
                    assert chatter in text and EMAIL.parse(chatter, glossary, domain) == DigestData([], [])
                previous = None
                for message in MESSAGE_BREAK.split(text):  # sent in thread order, each after what it carries
                    sent = _sent(next(line for line in message.split("\n") if line.startswith("Date: ")))
                    carried = EMAIL.parse(message, glossary, domain)
                    assert all(item.time <= sent for item in (*carried.reports, *carried.notices))
                    assert previous is None or previous <= sent
                    previous = sent


MESSAGE_BREAK = re.compile(r"\n\n\n(?:\n|_{32}\n\n)")  # what separates the messages of a thread


def test_a_remark_about_a_window_is_sent_after_it(toys):
    """A chatter line that tells of a window ("... was offline for maintenance from 00:00 UTC on 4 March 2026
    until 00:45 UTC") is a remark of a message sent after the window, never of one sent before it, which read as
    though it foretold it; where no message of the thread is sent that late, a reply that only acknowledges
    carries it, after the window and the thread's last message."""
    timed = 0
    for domain in (get_domain("weather"), EARTHQUAKES, toys.toy):
        for inst in instances_of(domain, seed=36, n=6):
            for rendering in inst.renderings:
                for message in MESSAGE_BREAK.split(rendering.digest):
                    sent = _sent(next(line for line in message.split("\n") if line.startswith("Date: ")))
                    own = [line for line in message.split("\n") if not line.startswith(">")]
                    for line, after in zip(inst.chatter, inst.chatter_after):
                        if after is not None and any(text.endswith(line) for text in own):
                            assert sent > after, (line, sent, after)
                            timed += 1
    assert timed >= 10, timed


def _sent(header: str) -> int:
    """The time a Date header gives (RFC 2822, with a numeric zone), as Unix seconds."""
    match = re.fullmatch(r"Date: \w{3}, (\d{1,2}) (\w{3}) (\d{4}) (\d{2}):(\d{2}):(\d{2}) ([+-])(\d{2})00", header)
    assert match is not None, header
    day, month, year, hour, minute, second, sign, zone = match.groups()
    local = datetime(int(year), MONTHS.index(month) + 1, int(day), int(hour), int(minute), int(second),
                     tzinfo=timezone.utc)
    return int(local.timestamp()) - (1 if sign == "+" else -1) * int(zone) * 3600


def test_every_notice_wording_is_named_by_the_reading_rules(toys):
    """Whatever wording a notice is drawn with, it calls itself a RETRACTION or a CORRECTION, or says that a
    value was revised: words that the reading rules of every domain name, earthquakes (the 0.2.0 rules)
    included. Each wording reads back on its own."""
    record = toys.flat.synthetic(random.Random(2), 1, COMPILED_AT, 24)[0]
    cases = [
        (toys.toy, record_rules(toys.toy),
         [Notice("retraction", record.id, COMPILED_AT - 99, reason="probe fault"),
          Notice("correction", record.id, COMPILED_AT - 9, field="temp", new_value=1.5, old_value=-2.0,
                 qualifier="spar"),
          Notice("correction", record.id, COMPILED_AT - 8, field="wave", new_value=1.25, old_value=2.5)]),
        (EARTHQUAKES, READING_RULES,
         [Notice("retraction", "us7000abcd", COMPILED_AT - 99, reason="quarry blast"),
          Notice("correction", "us7000abcd", COMPILED_AT - 9, new_mag=4.2, old_mag=4.55, mag_type="mb")]),
    ]
    for domain, rules, notices in cases:
        rules = " ".join(rules.split()).lower()
        named = {"retraction": ("retraction",), "correction": ("correction", "revised")}
        assert all(word in rules for words in named.values() for word in words)
        glossary = domain.glossary()
        wordings = set()
        for seed in range(60):
            rng = random.Random(seed)
            style = EMAIL.choose_style(rng, domain)
            ctx = RenderContext([], notices, [], COMPILED_AT, 24, style, rng, domain, glossary)
            wordings.update(line for line in EMAIL.render(ctx).split("\n")
                            if not line.startswith(">") and any(line.startswith(form) for form in NOTICE_FORMS))
        forms = set()
        for line in wordings:
            (notice,) = EMAIL.parse(line, glossary, domain).notices  # each wording reads back on its own
            assert notice in notices, line
            assert any(word in line.lower() for word in named[notice.kind]), line
            forms.add(next(form for form in NOTICE_FORMS if line.startswith(form)))
        assert forms == set(NOTICE_FORMS)
        assert len(wordings) >= 15


# ------------------------------------------------------------------------------ sniffing

SENDER = "From: Dana Whitfield <dana@desk.example.org>\nTo: Desk <desk@desk.example.org>\n"
OTHER_FORMATS = (
    "",
    "[06:12] dana: morning all\n[06:13] lee: by-12345 is in",  # a chat log
    "ref,source,status,time\nby-12345,ukmo,reviewed,2026-04-12T06:12:19Z",  # a table
    "| ref | source |\n|---|---|\n| by-12345 | ukmo |",
    "NORTH SEA, April 12 (Wire) — A reading came in overnight.",  # a news story
    "To: Records desk\nFrom: Station 4\nDate: 12/04/2026\nSubject: Readings\n\nAll quiet.",  # a memo
    SENDER + "Subject: Readings\n\nHi",  # headers without a date
    SENDER + "Date: 12 April 2026\nSubject: Readings\n\nHi",  # a date that is no RFC 2822 date
    SENDER + "Date: Sun, 12 Apr 2026 06:12:19 +0000\nX-Mailer: Desk\nSubject: Readings\n\nHi",  # an unknown header
)


def test_sniff_recognises_every_thread_and_nothing_else(toys, instances):
    for inst in instances:  # legacy classic digests (0.2.0)
        for rendering in inst.renderings:
            assert not EMAIL.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for domain in (toys.toy, toys.flat, toys.dressed):  # classic digests of other domains
        for inst in instances_of(domain, seed=12, n=1, genres=(CLASSIC,) * 3):
            for rendering in inst.renderings:
                assert not EMAIL.sniff(rendering.digest) and detect_genre(rendering.digest) is CLASSIC
    for text in OTHER_FORMATS:
        assert not EMAIL.sniff(text), text
    for name in DOMAINS:
        domain = get_domain(name)
        for inst in instances_of(domain, seed=13, n=1):
            for style, rng in styles(domain.bind(inst.surface), inst.seed):
                text = render(inst, domain.bind(inst.surface), style, rng)
                assert [g.name for g in all_genres() if g.sniff(text)] == ["email"]
                assert EMAIL.sniff("\n\n" + text)  # leading blank lines are no matter
                assert not EMAIL.sniff("\n".join("> " + line for line in text.split("\n")))  # a quoted thread
                assert not EMAIL.sniff(text.replace("Subject: ", "Topic: ", 1))


def test_a_variant_in_the_genre_differs_from_the_original_in_every_dimension(toys):
    for domain in (toys.toy, toys.flat, EARTHQUAKES):
        for seed in range(25):
            original = EMAIL.choose_style(random.Random(seed), domain)
            assert ("coords" in original) == domain.has_location
            variant = EMAIL.choose_style(random.Random(seed + 1000), domain, avoid=original)
            assert set(variant) == set(original) and all(variant[key] != original[key] for key in original)
    inst = instances_of(toys.flat, seed=14, n=1)[0]
    v0, v1, v2 = inst.renderings
    assert v1.style != v0.style != v2.style and len({v0.digest, v1.digest, v2.digest}) == 3


def test_a_thread_is_read_through_the_glossary_its_prompt_prints(toys):
    inst = instances_of(toys.dressed, seed=9, n=1)[0]
    bound = toys.dressed.bind(inst.surface)
    for rendering in inst.renderings:
        assert parse_glossary(rendering.prompt) == bound.glossary()  # the instance's own words
        assert solve(rendering.prompt, question_specs(inst), domain=inst.domain) == inst.answer_key
        assert solve(rendering.prompt, question_specs(inst), domain=inst.domain, genre="email") == inst.answer_key
        unglossed = re.sub(r"<glossary>.*?</glossary>\n\n", "", rendering.prompt, flags=re.DOTALL)
        with pytest.raises(ValueError, match="prints none"):  # no other way to read a drawn surface
            solve(unglossed, question_specs(inst), domain=inst.domain)


def test_email_refuses_a_rendering_it_cannot_read_back(toys):
    inst = instances_of(toys.toy, seed=15, n=1)[0]
    style, rng = next(styles(toys.toy, 15))
    awkward = [replace(r, group="North Sea (St. Kilda: west), at 56°N") for r in inst.reports]  # still readable
    render(replace(inst, reports=awkward), toys.toy, style, random.Random(1))
    split = [replace(r, group="North\nSea") for r in inst.reports]  # no line can hold a line break
    with pytest.raises(ValueError, match="does not read its rendering back exactly"):
        render(replace(inst, reports=split), toys.toy, style, rng)


# ------------------------------------------------------------------------------ whole runs


@pytest.fixture(scope="module")
def email_runs(tmp_path_factory):
    """generate (mix "<domain>:email") -> run (oracle, naive, mock) -> score -> report, for four instances
    of the earthquake domain, the toy domain with a position, toy-flat without one, and toy-dressed with a
    surface drawn per instance."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        toy = domains.load_plugin(TOY_PLUGIN)
        module = sys.modules[type(toy).__module__]
        for variant in (module.FLAT, module.DRESSED):
            domains.REGISTRY.register(variant)
        runs = {}
        for name, seed in (("earthquakes", 61), ("toy", 62), ("toy-flat", 63), ("toy-dressed", 64)):
            run_dir = pipeline.generate(GenerateConfig(seed=seed, instances=4, mix=f"{name}:email"),
                                        tmp_path_factory.mktemp("runs") / name, log=quiet)
            pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
            pipeline.score_run(run_dir, log=quiet)
            pipeline.build_report(run_dir, log=quiet)
            runs[name] = run_dir
        yield runs


def test_email_runs_are_generated_and_read_back_from_their_prompts(email_runs, tmp_path):
    for name, run in email_runs.items():
        manifest = read_json(run / "manifest.json")
        assert manifest["config"]["mix"] == f"{name}:email" and manifest["snapshot"]["domain"] == name
        check = manifest["self_check"]
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        for inst in load_instances(run):
            assert inst.domain == name and [r.genre for r in inst.renderings] == ["email"] * 3
            bound = get_domain(name).bind(inst.surface)
            rules = bound.rules_text()  # earthquakes too: GENRE_READING_RULES, off the legacy path
            for rendering in inst.renderings:
                prompt = rendering.prompt
                assert parse_glossary(prompt) == bound.glossary() and extract_digest(prompt) == rendering.digest
                assert prompt.index(rules) < prompt.index("<glossary>") < prompt.index("<digest>")
                assert detect_genre(rendering.digest) is EMAIL
                data = EMAIL.parse(rendering.digest, bound.glossary(), bound)
                assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_the_oracle_scores_100_percent_on_every_email_rendering(email_runs):
    for name, run in email_runs.items():
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows)
        assert {(row["domain"], row["genre"]) for row in rows} == {(name, "email")}
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
        for inst in load_instances(run):
            for rendering in inst.renderings:  # read by sniffing, as the generator's self-check reads it
                assert solve(rendering.prompt, question_specs(inst), domain=name) == inst.answer_key


def test_a_replayed_usgs_feed_reads_back_with_its_place_notes(tmp_path):
    """Earthquakes from the live source (replayed offline): every report gives a place note, commas and
    all, the prompts ask a forecast too, and the oracle still reproduces every key."""
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    cfg = GenerateConfig(seed=11, instances=3, source="usgs", snapshot_file=str(feed), mix="earthquakes:email")
    run_dir = pipeline.generate(cfg, tmp_path / "usgs-email", log=quiet)
    check = read_json(run_dir / "manifest.json")["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    for inst in load_instances(run_dir):
        assert inst.forecast is not None and all("," in r.place for r in inst.reports)
        for rendering in inst.renderings:
            assert "F1. Forecast" in rendering.prompt
            data = EMAIL.parse(rendering.digest, EARTHQUAKES.glossary(), EARTHQUAKES)
            assert all(map(same_items, (data.reports, data.notices), printed(inst.reports, inst.notices)))


def test_the_naive_reader_falls_for_the_traps_and_only_for_them(email_runs):
    """Traps behave as in every other genre: the naive reader reads the same reports and notices, so it
    answers exactly as the content dictates, falls for the traps, and for nothing else."""
    for name, run in email_runs.items():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        naive = models["naive"]["traps"]
        assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
        assert models["naive"]["accuracy"]["rate"] < 1.0 == models["oracle"]["accuracy"]["rate"]
        # The simulated model's errors are seeded by the prompt text, and a run of four instances has a dozen
        # questions: over all their renderings it is right and wrong in every run.
        assert 0 < models["mock"]["accuracy_all_renderings"]["rate"] < 1 and models["mock"]["format"]["replies"] == 12
        instances = load_instances(run)
        trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:
            if (row["instance_id"], row["qid"]) not in trapped:
                assert row["correct"], row
        for inst in instances:
            bound = get_domain(name).bind(inst.surface)
            content = DigestData(inst.reports, inst.notices)
            for rendering in inst.renderings:
                answers = solve(rendering.prompt, question_specs(inst), mode="naive", domain=name)
                for q in inst.questions:
                    assert answers[q.qid] == execute(q.chain, content, skip=("dedupe",), units=bound.units,
                                                     resolver=bound.resolve)
