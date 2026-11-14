"""The reading rules must name every wording the digest can print, so no notice is a gotcha."""
from __future__ import annotations

import copy
import itertools
import random
import re
from collections import Counter

from evalgen import pipeline
from evalgen.digest import CHATTER, render_notice
from evalgen.domains import base, get_domain
from evalgen.domains.base import REPEAT_RULE, Surface, indefinite
from evalgen.domains.invented import DUPLICATE_NOTES, INVENTED
from evalgen.genres import all_genres, chat, email, get_genre, radio
from evalgen.genres.base import parse_glossary, render_glossary
from evalgen.models import Notice
from evalgen.pipeline import GenerateConfig, build_instance
from evalgen.questions import Context, compose_new_templates
from evalgen.rules import READING_RULES
from evalgen.seeds import SeedLog
from evalgen.solver import solve
from evalgen.sources import build_domain_snapshot

# the words the rules use to identify each kind of notice
RULE_WORDS = {"retraction": ("retraction", "retracted", "withdrawn"), "correction": ("correction", "revised")}
RULES = " ".join(READING_RULES.split())  # the rules text with its line wrapping undone


def _all_wordings(kind: str) -> list[str]:
    notice = Notice(kind=kind, target="us7000abcd", time=1770530400, reason="quarry blast",
                    new_mag=4.2, old_mag=4.5, mag_type="mb")
    wordings = []
    for form in itertools.count():
        try:
            wordings.append(render_notice(notice, form, "iso"))
        except IndexError:
            return wordings


def test_every_notice_wording_is_named_by_the_reading_rules():
    rules = RULES.lower()
    for kind, words in RULE_WORDS.items():
        assert all(w in rules for w in words), (kind, words)
        wordings = _all_wordings(kind)
        assert len(wordings) >= 3
        for text in wordings:
            assert any(w in text.lower() for w in words), f"the rules do not name this {kind} wording: {text}"


def test_rules_say_how_to_compare_magnitudes_on_different_scales():
    assert any("scale" in line for line in CHATTER)  # digests may remind readers that scales differ ...
    assert "whatever their scale" in RULES  # ... so the rules must say how to compare them


def test_rules_fix_time_window_ends_and_coordinate_signs():
    assert "includes its start and excludes its end" in RULES  # ops.time_between: start <= t < end
    assert "negative for south and west" in RULES


def test_rules_cover_own_network_preliminaries_signed_depths_regions_and_other_lines():
    assert "a PRELIMINARY report from its own network" in RULES  # traps can file it under its own network
    assert "A depth can be negative" in RULES  # USGS hypocentres above sea level; the key keeps the sign
    assert "matched exactly as written" in RULES  # ops.passes: e.region == step["region"]
    assert "carry no data and change nothing" in RULES  # chatter, headers, station status lines
    assert "as printed" not in RULES  # rule 7 must not compete with rules 2 and 3


def test_coordinates_are_stored_at_the_printed_precision(snapshot_events, instances):
    for e in snapshot_events.values():
        assert (round(e.lat, 3), round(e.lon, 3)) == (e.lat, e.lon), e.id
    for inst in instances:
        for r in inst.reports:
            assert (round(r.lat, 3), round(r.lon, 3)) == (r.lat, r.lon), r.ref


def test_the_oracle_reproduces_distance_keys_exactly_from_the_printed_text(instances):
    """Not just within tolerance: a reader working from the printed digits gets the key itself."""
    checked = 0
    for inst in instances:
        specs = pipeline.question_specs(inst)
        for rendering in inst.renderings:
            answers = solve(rendering.prompt, specs, mode="oracle")
            for q in inst.questions:
                if q.answer_type == "distance":
                    assert answers[q.qid] == inst.answer_key[q.qid], (inst.instance_id, rendering.variant, q.qid)
                    checked += 1
    assert checked > 0


# ------------------------------------------------------------- repeats and forwards, in every genre

CHAT_MESSAGE = re.compile(r"^(?:\[\d{2}:\d{2}\] [\w.-]+|\U0001f4cc Pinned by [\w.-]+): ")


def is_repeat(genre: str, line: str) -> bool:
    """Whether a digest line is a copy the rules call a repeat ("a quoted or read-back repeat of a report
    or notice printed elsewhere in the digest"): a quoted line of an email thread, a quote or read-back
    posted in a chat log, or a radio operator reading a report back. No other genre prints repeats."""
    if genre == "email":
        return line.startswith(email.QUOTE)
    if genre == "chat":
        body = CHAT_MESSAGE.sub("", line)
        return body != line and body.startswith(chat.REPEATS)
    if genre == "radio":
        return any(intro in line for intro in radio.READBACK_INTROS)
    return False


def genre_instances(name: str, genre, seed: int = 5) -> list:
    """The first instance of a run about the domain `name` with every rendering in `genre`, as the
    pipeline builds it off the legacy path."""
    domain, log = get_domain(name), SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    return [build_instance(0, snapshot, new, log, GenerateConfig(seed=seed), ctx, set(), {}, domain=domain,
                           genre_plan=(genre,) * 3)]


def test_every_notice_is_printed_once_where_the_rules_count_it():
    """The rules of every domain off the legacy path (rule 7 of a record domain's, rule 11 of the
    earthquake rules) call a quote or a read-back of a report or notice printed elsewhere a repeat, which
    is not a new one, and say that a forwarded or relayed report or notice is not a repeat and counts.
    Some genres deliver every notice that way: an email thread forwards each one, from a notices desk
    whose own message never appears. So in every genre, a reader who strikes out every repeat the rules
    name still reads every notice, exactly once: its one copy is never a quote or a read-back."""
    struck = Counter()
    for name in ("earthquakes", "weather", "invented"):
        domain = get_domain(name)
        for genre in all_genres():
            for inst in genre_instances(name, genre):
                bound = domain.bind(inst.surface)
                rules = " ".join(bound.rules_text().split())
                assert REPEAT_RULE in rules and "quoted, forwarded or read-back" not in rules
                assert ("A forwarded or relayed report or notice is not a repeat: it counts, once, as the report or "
                        "notice it carries, and rule 1 still says which record that report is.") in rules
                notices = sorted((n.kind, n.target) for n in inst.notices)
                assert notices
                for rendering in inst.renderings:
                    lines = rendering.digest.split("\n")
                    kept = [line for line in lines if not is_repeat(genre.name, line)]
                    struck[genre.name] += len(lines) - len(kept)
                    data = genre.parse("\n".join(kept), bound.glossary(), bound)
                    assert sorted((n.kind, n.target) for n in data.notices) == notices, (name, genre.name)
                    if genre.name == "email":  # each one in a forwarded message, which the rules count
                        forwarded = [i for i, line in enumerate(kept) if line.startswith(email.FORWARD_MARKERS[
                            rendering.style["forwards"]][0])]
                        assert len(forwarded) >= len(notices)
    assert struck["email"] and struck["chat"] and struck["radio"]  # each of them printed repeats to strike out


# --------------------------------------------------------------- what a notice is, and what a note says


def rules_texts(name: str, surfaces: int = 3) -> set[str]:
    """The rules texts the prompts of domain `name` print off the legacy path: its own, and those of a few
    drawn surfaces for a domain that draws them."""
    domain = get_domain(name)
    bound = ([domain.bind(domain.draw_surface(random.Random(seed))) for seed in range(surfaces)]
             if domain.has_surface else [])
    return {d.rules_text() for d in (domain, *bound)}


def test_every_domains_rules_say_what_a_notice_is_wherever_it_is_printed():
    """The plan's first genre-neutral statement: what counts as a notice. In an email thread, a radio net,
    a form or a table, a notice looks like any other message, so the rules of every domain off the legacy
    path define one, in the same words, before they say which lines carry no data (the 0.2.0 rules name
    notices by their keywords alone). Most genres print some notices before the report they name (a chat
    log's pinned board heads the log), so the rules never tie a notice to an "earlier" report, which a
    reader could take to mean a report printed above it."""
    for name in ("earthquakes", "weather", "neo", "invented"):
        for rules_text in rules_texts(name):
            rules = " ".join(rules_text.split())
            assert rules.count(base.NOTICE_RULE) == 1, name
            assert rules.index(base.NOTICE_RULE) < rules.index("Lines that are neither reports nor notices"), name
            assert "before or after that report" in rules and "earlier report" not in rules, name
    assert base.NOTICE_RULE not in RULES  # the legacy rules stay the 0.2.0 text


# The labels of the lines that rule 9 says carry no data; the words a notice line names itself with; and
# what opens a message, a quote, a bullet or a footnote before its text, in the genres that print them.
DATA_FREE_LABELS = re.compile(r"^(?:editor's note|reminder|status|station status|header|section|greeting|signature)\b",
                              re.IGNORECASE)
NOTICE_WORDS = re.compile(r"retract|correct|revis|update|withdrawn|amend", re.IGNORECASE)
NOTICE_OPENINGS = re.compile(r"^(?:notice|retraction|retracted|correction|corrected|update|revised|amendment)\b",
                             re.IGNORECASE)
LINE_PREFIX = re.compile(r"^(?:\[\d{2}:\d{2}\] [\w.-]+: |\U0001f4cc Pinned by [\w.-]+: |\d{2}:\d{2}:\d{2} [A-Z0-9-]+: |"
                         r"> ?|[-*] |\[\d+\],?|\|\s*\[?\d+\]?\s*\|\s*|Subject: )*")


def test_no_notice_is_headed_as_a_line_without_data_and_no_such_line_as_a_notice():
    """Rule 9 counts editor's notes, reminders, status lines, headers, greetings and signatures among the
    lines that carry no data. The news genre once printed a retraction and a correction wording as
    "Editor's note (<time>): this is a retraction notice ...", which the rules told a careful reader to
    ignore, beside chatter that opens "Editor's note:". In no domain and no genre does a line that carries
    a notice open with such a label, nor a line of chatter with a notice's keyword."""
    for name in ("earthquakes", "weather", "neo", "invented"):
        domain = get_domain(name)
        for genre in all_genres():
            for inst in genre_instances(name, genre):
                bound = domain.bind(inst.surface)
                assert [line for line in bound.chatter if NOTICE_OPENINGS.match(line)] == [], name
                for rendering in inst.renderings:
                    for line in rendering.digest.split("\n"):
                        if any(n.target in line for n in inst.notices) and NOTICE_WORDS.search(line):
                            body = LINE_PREFIX.sub("", line).strip()
                            assert not DATA_FREE_LABELS.match(body), (name, genre.name, line)


# Words that the reading rules or the genres give a meaning of their own, which a duplicate note must not
# borrow. The rules on messages (rules 6 and 7 of a record domain, 10 and 11 of the earthquake rules) say
# that a forwarded or relayed report "is not a repeat: it counts, once", so a re-report noted "relay of
# <ID>" reads as a report that counts on its own, where rule 1 makes it that record; that a quoted or
# read-back "repeat" is no new report, so a re-report noted "repeat filing of <ID>" reads as one that rule 7
# sets aside; a message's sent time is never a record's; and a radio net's procedure words carry no data.
MESSAGE_WORDS = ("send", "sent", "forward", "relay", "quote", "repeat", "read back", "readback", "read-back", "copy",
                 "roger", "wilco", "over", "echo")


def message_words(note: str) -> list[str]:
    return [word for word in MESSAGE_WORDS if re.search(rf"\b{re.escape(word)}", note.casefold())]


def test_no_duplicate_note_reads_as_a_message_the_rules_count(toys):
    """A duplicate note makes a re-report the record it names (rule 1). Weather's "relay of <ID>" also
    named what rule 7 says "is not a repeat: it counts, once", and so read as the careless split the
    duplicate trap records, and the invented domain's "repeat filing of <ID>" named what rule 7 calls a
    repeat, which is no new report. No note of any domain, surface or plugin borrows a word the rules or
    the genres give another meaning."""
    assert message_words("relay of") == ["relay"] and message_words("Forwarded from") == ["forward"]
    assert message_words("repeat filing of") == ["repeat"]
    notes = [(name, note) for name in ("earthquakes", "weather", "neo", "invented")
             for note in get_domain(name).trap_vocab.duplicate_notes]
    notes += [("toy", note) for note in toys.toy.trap_vocab.duplicate_notes]
    notes += [("invented pool", note) for note in DUPLICATE_NOTES]
    notes += [("invented surface", note) for seed in range(200)
              for note in Surface.from_dict(INVENTED.draw_surface(random.Random(seed))).trap_vocab.duplicate_notes]
    assert ("weather", "same observation as") in notes and len({name for name, _ in notes}) == 7
    assert [(name, note) for name, note in notes if message_words(note)] == []


def test_the_record_rules_quote_every_duplicate_note_a_digest_prints(toys):
    """Rules 3 and 4 quote the phrases of a domain's corrections and retractions; rule 1 quotes the notes
    of its re-reports too ("same observation as", "mirror of" ... followed by the ID), so that a note that
    alone decides an answer ("also logged as QBM/25342", which the invented domain draws per instance) is
    never a gotcha either. The earthquake rules off the legacy path keep the 0.2.0 rules 1-8, whose rule 1
    names the note by what it does, as 0.2.0 did."""
    bound = [INVENTED.bind(INVENTED.draw_surface(random.Random(seed))) for seed in range(40)]
    dressed = toys.dressed
    bound += [dressed.bind(dressed.draw_surface(random.Random(seed))) for seed in range(10)]
    for domain in (get_domain("weather"), get_domain("neo"), toys.toy, toys.flat, *bound):
        rules = " ".join(domain.rules_text().split())
        rule_1 = rules[rules.index("1. "):rules.index("2. ")]
        notes = domain.trap_vocab.duplicate_notes
        assert notes and all(f'"{note}"' in rule_1 for note in notes), (domain.name, notes)
        assert f"followed by that {domain.id_label}." in rule_1
    earthquakes = " ".join(get_domain("earthquakes").rules_text().split())
    assert RULES[RULES.index("1. "):RULES.index("9. ")] in earthquakes  # 0.2.0 rules 1-8, word for word


def test_a_settling_report_settles_a_records_time_and_position_as_well_as_its_values(toys):
    """A re-report or a second report stands where its slot stands, at a time and a place of its own (traps),
    so it can fall inside a question's time window or nearer to a point than its record. Rule 2 says that the
    settling report's time and position (for records that have one) supersede the unsettled report's, not only
    its values, and so does the glossary's line on statuses, which every prompt off the legacy path prints,
    earthquakes' included (their rules keep the 0.2.0 rule 2). The glossary of a run generated before the line
    named them is still read."""
    for domain in (get_domain("weather"), toys.toy):
        rules = " ".join(domain.rules_text().split())
        final, draft = domain.statuses
        assert (f'use the time, position and values from its "{final}" report: "{draft}" times, positions and '
                "values are superseded.") in rules, domain.name
    for domain in (get_domain("neo"), get_domain("invented"), toys.flat):  # records without a position
        rules = " ".join(domain.rules_text().split())
        final, draft = domain.statuses
        assert (f'use the time and values from its "{final}" report: "{draft}" times and values are '
                "superseded.") in rules, domain.name
    for name, settled in (("earthquakes", "time, position and values"), ("weather", "time, position and values"),
                          ("neo", "time and values"), ("invented", "time and values")):
        domain = get_domain(name)
        final, draft = domain.statuses
        text = render_glossary(domain.glossary())
        line = f'- Statuses: {indefinite(final)} "{final}" report settles a record\'s {settled}; '
        assert line in text, (name, text)
        before = text.replace(f"settles a record's {settled};", "settles a record's values;")
        assert parse_glossary(before) == parse_glossary(text) == domain.glossary()


# ------------------------------------------------------------------------- times, articles, definitions


def test_the_rules_say_whose_time_a_wire_lines_brackets_give():
    """The classic genre's wire lines give their record's time in brackets after the source ("KPHL WIRE
    [05:55:26 (UTC-05:00) 18 Feb 2026] Midwest @ ..."), where a chat log prints a message's own time stamp
    ("[06:35] relaybot: ..."). Beside the rule that the time a message was sent, forwarded or relayed is
    never a record's, a careful reader could take the bracket for the time the wire relayed the report,
    which leaves the report no time of its own, and answer time questions without it. So the rules of
    every domain off the legacy path say that a WIRE line is a report and its bracketed time the report's,
    and every wire line of a classic digest gives that time, which the oracle reads, and no other."""
    classic = get_genre("classic")
    for name in ("earthquakes", "weather", "neo", "invented"):
        for rules_text in rules_texts(name):
            rules = " ".join(rules_text.split())
            assert rules.count(base.WIRE_RULE) == 1, name
            assert rules.index("never the time of") < rules.index(base.WIRE_RULE), name
        domain, wires = get_domain(name), 0
        for inst in genre_instances(name, classic):
            bound = domain.bind(inst.surface)
            for rendering in inst.renderings:
                for line in rendering.digest.split("\n"):
                    if " WIRE [" not in line:
                        continue
                    (report,) = classic.parse(line, bound.glossary(), bound).reports
                    (printed,) = [r for r in inst.reports
                                  if (r.ref, r.source, r.status) == (report.ref, report.source, report.status)]
                    assert abs(report.time - printed.time) < 60, line  # 0.2.0 digests may print the minute only
                    assert len(re.findall(r"\[[^\]]*\]", line)) == 1, line  # no other bracketed time
                    wires += 1
        assert wires, name
    assert base.WIRE_RULE not in RULES  # the 0.2.0 rules, which have no rule on message times, stay as they were


VOWEL_AFTER_A = re.compile(r"\b[Aa] \"?(?!unit\b)[aeiouAEIOU]")  # "a unit" is right
CONSONANT_AFTER_AN = re.compile(r"\b[Aa]n \"?(?!hour)[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]")


def test_every_article_in_the_rules_fits_the_word_it_stands_before(toys):
    """The record rules word every domain in its own words, and invented surfaces draw theirs (a group may
    be an "outpost"), as plugins may ("observation", statuses such as "estimated"): no article is assumed,
    so no rule reads "A outpost in a question is the outpost printed on the report"."""
    articles = {"outpost": "an", "observation": "an", "event": "an", "Inlet": "an", "estimated": "an",
                "unverified": "an", "hour": "an", "unit": "a", "user": "a", "one": "a", "European": "a",
                "depot": "a", "quay": "a", "sea area": "a", "region": "a", "close approach": "a", "": "a"}
    assert {word: indefinite(word) for word in articles} == articles
    texts = {text for name in ("earthquakes", "weather", "neo", "invented") for text in rules_texts(name, 400)}
    dressed = toys.dressed
    texts |= {dressed.bind(dressed.draw_surface(random.Random(seed))).rules_text() for seed in range(40)}
    vowels = copy.copy(toys.toy)
    vowels.noun, vowels.group_label, vowels.statuses = "event", "inlet", ("authoritative", "estimated")
    texts.add(vowels.rules_text())
    glossary = render_glossary(vowels.glossary())  # the glossary words its statuses as the rules do
    assert ('- Statuses: an "authoritative" report settles a record\'s time, position and values; an "estimated" '
            'report does not.' in glossary) and parse_glossary(glossary) == vowels.glossary()
    texts.add(glossary)
    flat = {" ".join(text.split()) for text in texts}
    assert any("An outpost in a question" in text for text in flat)  # the surfaces drew one
    assert any("An observation may appear" in text for text in flat)
    assert any('an "estimated" report' in text and 'on an "authoritative" report' in text for text in flat)
    assert [match.group(0) for text in flat for match in VOWEL_AFTER_A.finditer(text)] == []
    assert [match.group(0) for text in flat for match in CONSONANT_AFTER_AN.finditer(text)] == []


def test_every_domains_rules_say_once_what_a_reports_time_is():
    """A report's time is defined in the one rule on times (rule 6 of the record rules, as each domain
    words it: Domain.report_time), never twice: the close approaches' rules said both that it is when the
    approach "was recorded", which reads as when a catalogue logged it, and, in a rule of their own, that
    it is the moment the object passed closest to the body."""
    for name in ("earthquakes", "weather", "neo", "invented"):
        for rules_text in rules_texts(name):
            rules = " ".join(rules_text.split())
            assert rules.count("A report's time is") == 1 and "The time of a report is" not in rules, name
    neo = " ".join(get_domain("neo").rules_text().split())
    assert ("A report's time is the time of its close approach, the moment the object passed closest to the body; "
            "a notice's time is when the notice was issued.") in neo
    assert "close approach was recorded" not in neo
