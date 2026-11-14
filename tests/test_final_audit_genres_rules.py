"""The final audit's findings about genres and reading rules, each as a test.

Off the legacy path the reading rules of every built-in domain quote, word for word, every phrase that
any genre prints to note a re-report or to announce a notice, and every correction gives its new value
after a phrase they quote, so that no note or notice is a gotcha: the two tests below render every
domain in every genre, many styles each, and read every note and notice of the text. Every genre
prints any place a source gives, an empty one included; rule 1 says nothing of which source files a
re-report; no rule calls a table's header row, which gives its columns' units, a line without data;
nothing a genre prints contradicts its own timestamps; and the wordings that genres printed before
are still read. The legacy (0.2.0) rules and prompts are as they were (test_regression_020.py).
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any

import pytest

from evalgen.domains import get_domain
from evalgen.domains.base import (KEYED_NOTE, REVISED_FROM, REVISED_TO, SAME_AS_KEY, STORED_VALUE_RULE,
                                  TABLE_HEADER_RULE, RecordReport, indefinite, record_rules)
from evalgen.domains.earthquakes import GENRE_READING_RULES, TRAP_VOCAB, Earthquakes
from evalgen.genres import chat, email, form, get_genre, news, radio
from evalgen.genres.base import NUMBER, RenderContext, format_coords, format_time, parse_time, printed_place
from evalgen.models import Notice, Report
from evalgen.rules import READING_RULES

from test_reading_rules import genre_instances, is_repeat

COMPILED_AT = 1_780_000_000
GENRES = ("classic", "news", "email", "chat", "table", "form", "radio")


# ----------------------------------------------------------------------------------- helpers


def built_in_domains() -> list:
    """Every built-in domain as prompts off the legacy path word it (the invented one bound to a few drawn
    surfaces, as each of its instances is)."""
    invented = get_domain("invented")
    return [get_domain("earthquakes"), get_domain("weather"), get_domain("neo"),
            *(invented.bind(invented.draw_surface(random.Random(seed))) for seed in range(3))]


def rules_by_number(domain) -> dict[int, str]:
    """A domain's reading rules off the legacy path, by number, each with its lines joined."""
    parts = re.split(r"^(\d+)\. ", domain.rules_text(), flags=re.MULTILINE)
    return {int(number): " ".join(body.split()) for number, body in zip(parts[1::2], parts[2::2])}


def quoted(text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', text)


def window(domain) -> int:
    return (domain.unbound or domain).synthetic_window_hours


def records_of(domain, seed: int, n: int = 8) -> list:
    """`n` records of the domain's synthetic world, as an instance shows them (dressed in its surface)."""
    unbound = domain.unbound or domain
    return [domain.dress(r) for r in unbound.synthetic(random.Random(seed), n, COMPILED_AT, window(domain))]


KEEP = object()  # a report keeps its record's place


def report_of(domain, record, status: str, *, ref: str | None = None, same_as: str | None = None,
              place: Any = KEEP) -> Report | RecordReport:
    """A report of `record` (its own place, unless `place` is given) in the domain's record type."""
    place = record.place if place is KEEP else place
    if isinstance(domain, Earthquakes):
        return Report(ref=ref or record.id, network=record.network, status=status, time=record.time, lat=record.lat,
                      lon=record.lon, depth_km=record.depth_km, mag=record.mag, mag_type=record.mag_type,
                      region=record.region, place=place, same_as=same_as)
    return RecordReport(ref=ref or record.id, source=record.source, status=status, time=record.time, lat=record.lat,
                        lon=record.lon, group=record.group, values=dict(record.values),
                        qualifiers=dict(record.qualifiers), place=place, same_as=same_as)


def digest_items(domain, seed: int) -> tuple[list, list[Notice]]:
    """A digest's reports and notices: each record's settling report, re-reports of three records under
    references of their own with a note naming them, a retraction of one record's report and a correction
    of another's value (a magnitude with a tenths digit, so that 0.2.0 and the codec write it alike)."""
    records = records_of(domain, seed)
    rng = random.Random(seed)
    final, draft = domain.statuses
    taken = {r.id for r in records}
    reports = [report_of(domain, r, final) for r in records]
    reports += [report_of(domain, r, draft, ref=domain.mimic_id(r.id, rng, taken), same_as=r.id) for r in records[:3]]
    retracted, corrected = records[-1], records[-2]
    notices = [Notice("retraction", retracted.id, COMPILED_AT - 600, reason=domain.trap_vocab.retraction_reasons[0])]
    if isinstance(domain, Earthquakes):
        notices.append(Notice("correction", corrected.id, COMPILED_AT - 300, new_mag=3.7 if corrected.mag != 3.7 else 3.3,
                              old_mag=corrected.mag, mag_type=corrected.mag_type))
    else:
        spec = next(s for s in domain.fields if s.correctable)
        new = next(r.values[spec.key] for r in records if r.values[spec.key] != corrected.values[spec.key])
        notices.append(Notice("correction", corrected.id, COMPILED_AT - 300, field=spec.key, new_value=new,
                              old_value=corrected.values[spec.key]))
    return reports, notices


def render(genre_name: str, domain, reports, digest_notices, seed: int, **style) -> str:
    """One rendering of `genre_name`, its style drawn from `seed` (with `style` in place of what it draws)."""
    genre = get_genre(genre_name)
    rng = random.Random(seed)
    drawn = {**genre.choose_style(rng, domain), **style}
    return genre.render(RenderContext(reports, digest_notices, [], COMPILED_AT, window(domain), drawn, rng, domain,
                                      domain.glossary()))


def units_of(genre_name: str, text: str) -> list[str]:
    """The pieces of a rendering that each hold one report or notice: a form's blocks, else its lines (a news
    paragraph is one), without the repeats the rules name (quotes and read-backs) and an email's subject lines,
    headers that name a notice's ID."""
    if genre_name == "form":
        return text.split("\n\n")
    return [line for line in text.split("\n")
            if line.strip() and not is_repeat(genre_name, line) and not line.startswith("Subject: ")]


def naming(units: list[str], record_id: str) -> list[str]:
    pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(record_id)}(?![0-9A-Za-z])")
    return [unit for unit in units if pattern.search(unit)]


def new_numbers(domain, notice: Notice) -> list[str]:
    """A correction's new value as a digest can write it: its number in each unit of its field."""
    key, value = ("mag", notice.new_mag) if isinstance(domain, Earthquakes) else (notice.field, notice.new_value)
    f = next(f for f in domain.glossary().fields if f.key == key)
    return sorted({f.encode(value, unit) for unit in (f.units or (None,))})


def gives_new_value(unit: str, phrases: list[str], numbers: list[str], revised_from: bool) -> bool:
    """Whether a correction gives its new value right after one of `phrases` ("magnitude now MB 3.7" included),
    or after "revised from <old> to"."""
    for number in numbers:
        value = rf"(?:[A-Z]+ )?{re.escape(number)}(?!\.?\d)"
        if any(re.search(rf"{re.escape(p)} {value}", unit) for p in phrases):
            return True
        if revised_from and re.search(rf"{REVISED_FROM} .+? to {value}", unit):
            return True
    return False


# ---------------------------------------------------- every note and notice is quoted by the rules


def test_rule_1_quotes_every_note_that_ties_a_re_report_to_its_record():
    """code-domain-base#0, sweep-guarantees#1, code-genres-2#0: a classic log line and a pasted chat line print
    a re-report's note as the field same_as=<ID>, its only link to its record, which rule 1 did not quote
    ("Such a note reads "same observation as", "duplicate of" or "mirror of", followed by that observation
    ID"): a reader who took that list as complete counted the re-report as a record of its own, the careless
    split its trap records. The earthquake rules off the legacy path quoted no note at all. Now rule 1 of
    every domain quotes the field beside its notes, and every note that any genre prints, in every domain,
    is one of the phrases rule 1 quotes, followed by the ID."""
    keyed = Counter()
    for domain in built_in_domains():
        rule_1 = rules_by_number(domain)[1]
        phrases = [p for p in quoted(rule_1) if p not in domain.statuses]
        assert f"{SAME_AS_KEY}=" in phrases and KEYED_NOTE in rule_1, (domain.name, rule_1)
        assert all(note in phrases for note in domain.trap_vocab.duplicate_notes), domain.name
        for seed in range(6):
            reports, notices = digest_items(domain, seed)
            noted = Counter(r.same_as for r in reports if r.same_as)
            for name in GENRES:
                text = "\n".join(units_of(name, render(name, domain, reports, notices, seed)))
                for record_id, count in noted.items():
                    found = sum(text.count(f"{p}{record_id}" if p.endswith("=") else f"{p} {record_id}")
                                for p in phrases)
                    assert found == count, (domain.name, name, seed, record_id)
                    keyed[name] += text.count(f"{SAME_AS_KEY}={record_id}")
    assert keyed["classic"] and keyed["chat"], keyed  # the key=value lines were printed, and read by rule 1


def test_every_correction_gives_its_new_value_after_a_phrase_the_rules_quote():
    """finish3#3, code-genres-2#4, code-earthquakes#3: a table's "Correction (...): <ID> <field> now X (was
    Y)", an email's "CORRECTION NOTICE (...): <ID> <field> now X, not Y" and a radio net's "now reads X" and
    "amended from Y to X" gave the new value after words no rule quotes (in weather-9209 such a notice decided
    Q4), and the earthquake rules off the legacy path quoted none of their notices' phrases. Now every
    retraction any genre prints names itself with words its domain's rule 4 states (RETRACTION, a quoted
    phrase, "was not a real <noun>"), and every correction with words of rule 3 (CORRECTION or a quoted
    phrase), its new value right after a phrase rule 3 quotes ("revised to", a domain's own, "magnitude now"),
    or after "revised from" and the old value, which rule 3 quotes too; a form's amendment is headed
    CORRECTION and says that the value "was revised"."""
    for domain in built_in_domains():
        rules = rules_by_number(domain)
        rule_3, rule_4 = rules[3], rules[4]
        retraction_words = [w for w in ("RETRACTION", "retracted", f"was not a real {domain.noun}", *quoted(rule_4))
                            if w in rule_4]
        phrases_3 = quoted(rule_3)
        assert REVISED_TO in phrases_3 and REVISED_FROM in phrases_3, (domain.name, rule_3)
        if isinstance(domain, Earthquakes):
            assert set(TRAP_VOCAB.correction_notices) <= set(phrases_3)
            assert set(TRAP_VOCAB.retraction_notices) <= set(quoted(rule_4))
        else:
            assert set(domain.trap_vocab.correction_notices) <= set(phrases_3)
            assert set(domain.trap_vocab.retraction_notices) <= set(quoted(rule_4))
        _, notices = digest_items(domain, 0)
        retraction, correction = notices
        numbers = new_numbers(domain, correction)
        for name in GENRES:
            corrections = set()
            for seed in range(14):
                units = units_of(name, render(name, domain, [], notices, seed))
                for unit in naming(units, retraction.target):
                    assert any(w in unit for w in retraction_words), (domain.name, name, unit)
                for unit in naming(units, correction.target):
                    assert "CORRECTION" in unit or any(p in unit for p in phrases_3), (domain.name, name, unit)
                    if name == "form":  # an amendment form: headed CORRECTION, its value under "Corrected ..."
                        assert "CORRECTION" in unit and "was revised" in unit and "a value was revised" in rule_3
                    else:
                        assert gives_new_value(unit, [p for p in phrases_3 if p != REVISED_FROM], numbers,
                                               REVISED_FROM in phrases_3), (domain.name, name, unit)
                    corrections.add(re.sub(r"\d", "#", unit))
            assert corrections, (domain.name, name)
    assert not any(phrase in READING_RULES for phrase in ('"revised to"', '"same_as="', '"withdrawn"'))  # 0.2.0 as it was


def test_the_earthquake_rules_quote_the_phrases_of_their_trap_vocabulary():
    """code-earthquakes#3: the comment on TRAP_VOCAB said that the reading rules name its phrases; neither
    the 0.2.0 rules nor those off the legacy path quoted one. GENRE_READING_RULES quotes every one now (and
    the key=value field), and the comment says that the 0.2.0 rules name them only by what they do."""
    flat = " ".join(GENRE_READING_RULES.split())
    for phrase in (*TRAP_VOCAB.duplicate_notes, *TRAP_VOCAB.retraction_notices, *TRAP_VOCAB.correction_notices,
                   f"{SAME_AS_KEY}="):
        assert f'"{phrase}"' in flat, phrase
        assert f'"{phrase}"' not in " ".join(READING_RULES.split()), phrase


# ------------------------------------------------------------------------------------ the rules


def test_rule_1_says_nothing_of_which_source_files_a_re_report():
    """finish3#7: rule 1 said that a record may appear in "a preliminary report from its own <source>, or a
    re-report from another <source>", but off the legacy path the trap builders file a re-report, or a second
    report, under the source of the record it stands in for (traps._filed_as), which may be the record's own
    source, or another one. Rule 1 now says nothing of where either comes from."""
    for domain in built_in_domains():
        rule_1 = rules_by_number(domain)[1]
        source = domain.source_label
        assert f"from its own {source}" not in rule_1 and f"from another {source}" not in rule_1, domain.name
        assert "may appear in several reports: " in rule_1 and "or a re-report, either under the same" in rule_1
    same = other = 0
    for seed in range(1, 9):  # what the old wording denied happens in real instances
        (inst,) = genre_instances("weather", get_genre("classic"), seed)
        final = get_domain("weather").statuses[0]
        settling = {r.ref: r.source for r in inst.reports if r.status == final}
        for r in inst.reports:
            record = r.same_as or r.ref
            if r.status != final and record in settling:
                same += r.source == settling[record]
                other += r.source != settling[record]
    assert same and other, (same, other)


def test_no_rule_calls_a_tables_header_row_a_line_without_data():
    """finish3#4: rule 9 said that headers carry no data, but a table's header row can be the only place that
    gives the unit of a column of bare numbers ("depth (km)") or the zone of its times ("time (UTC)")."""
    for domain in built_in_domains():
        rule_9 = rules_by_number(domain)[9]
        assert "Lines that are neither reports nor notices (headers," in rule_9 and rule_9.endswith(TABLE_HEADER_RULE)
    weather = get_domain("weather")
    reports, notices = digest_items(weather, 1)
    text = render("table", weather, reports, notices, 3, units="brackets", time="utc_column", case="plain")
    assert "time (UTC)" in text and re.search(r"temperature \[(?:°C|°F|K)\]", text)  # the header gives them alone
    assert not re.search(r"\d (?:°C|°F|K)\b", text.split("Notes")[0].split("Footnotes")[0])  # bare cells


def test_every_record_domain_says_what_a_value_in_another_unit_stands_for():
    """blind-chat#1: only the invented domain's rules said that a value written in another unit stands for its
    conversion rounded to the field's stored places, never exactly halfway, though every genre writes every
    domain's values so (evalgen.units). Rule 5 of every record domain says it now, once."""
    for domain in built_in_domains()[1:]:
        assert STORED_VALUE_RULE in rules_by_number(domain)[5], domain.name
        assert " ".join(domain.rules_text().split()).count(STORED_VALUE_RULE) == 1, domain.name


def test_no_weather_question_words_a_region_by_the_stations_in_it():
    """code-weather#6: "from stations in {group}" could be read to leave out the observations a relay feed
    filed (the glossary calls whoever files a report a feed; a radio net calls its call-in units stations)."""
    wordings = get_domain("weather").lexicon.filters["in_group"]
    assert len(wordings) == 3 and not any("station" in wording for wording in wordings)


def test_the_docstrings_promise_no_article_that_indefinite_cannot_give():
    """code-domain-base#2: record_rules said that every article fits "whatever words a domain or a drawn
    surface gives", but indefinite goes by the first letter and a few exceptions; the docstrings now say
    which words it fits and which it can miss."""
    assert {w: indefinite(w) for w in ("MRI scan", "usage record")} == {"MRI scan": "a", "usage record": "an"}
    assert "whatever words" not in record_rules.__doc__ and "MRI" in record_rules.__doc__
    assert "initialism" in indefinite.__doc__


# ------------------------------------------------------------------------------- places


@pytest.mark.parametrize("place", ["", "   ", "5 km N of Town, Alaska ", " 5 km N of Town, Alaska",
                                   "5 km N of\nTown,  Alaska"])
def test_every_genre_prints_any_place_a_source_gives(place):
    """code-genres-2#1 and #2: an empty place (the USGS feed gives one now and then) stopped generation in the
    news, email, table, form and radio genres (each printed "Location note: ." or an empty cell that its
    parser could not read back), and a place ending in a space did so in chat (its fallback to the paste
    checked the message before the log's lines are stripped), table and form. Every genre now prints a place
    on one line without the whitespace around it, and an empty one as none."""
    for domain in (get_domain("earthquakes"), get_domain("weather")):
        records = records_of(domain, 4, n=4)
        reports = [report_of(domain, r, domain.statuses[0], place=place) for r in records]
        for name in GENRES:
            if name == "classic" and isinstance(domain, Earthquakes):
                continue  # the 0.2.0 digest, which prints no empty place and keeps its own parser
            for seed in range(8):
                text = render(name, domain, reports, [], seed)
                read = get_genre(name).parse(text, domain.glossary(), domain).reports
                assert [r.place for r in read] == [printed_place(place)] * len(reports), (domain.name, name, seed)
    assert printed_place("") is printed_place("  \n ") is None and printed_place("see  log ") == "see  log"


def test_a_chat_message_that_ends_in_a_space_is_pasted():
    """code-genres-2#2: the check that sends a report a voice cannot print to the paste read the message as it
    is, but the parser reads the log's lines stripped, so a message ending in a space passed the check and
    then read back otherwise."""
    weather = get_domain("weather")
    glossary = weather.glossary()
    (record,) = records_of(weather, 2, n=1)
    report = report_of(weather, record, weather.statuses[0], place=record.place + " ")
    printed = chat._Printed("relay", "", format_time(record.time, "iso"),
                            format_coords(record.lat, record.lon, "hemi", glossary.coord_decimals),
                            tuple(f.units[0] if f.units else None for f in glossary.fields), None)
    body = chat._report_body(report, printed, glossary)
    assert body.endswith(" ") and chat._read_body(body, chat._grammar(glossary), glossary) == report
    assert not chat._reads_back(body, report, chat._grammar(glossary), glossary)


# ---------------------------------------------------------------- timestamps and messages


def test_no_news_notice_or_heading_speaks_of_the_article_it_was_issued_before():
    """finish3#5: every notice is issued before the article's Published time, so "Updates to this article"
    over them and a retraction of "report X, listed in this article" contradicted the article's timestamps."""
    for domain in (get_domain("earthquakes"), get_domain("weather")):
        reports, notices = digest_items(domain, 2)
        for seed in range(30):
            text = render("news", domain, reports, notices, seed)
            for notice in notices:
                assert not any("article" in unit for unit in naming(units_of("news", text), notice.target)), seed
            assert not any(heading in text for heading in ("Updates to this article", "Corrections to this article"))
            published = parse_time(text.split("\n")[1].split(" | Published ")[1])
            assert all(n.time < published for n in notices)
    assert not any("article" in heading for heading in news.END_HEADINGS)
    assert not any("article" in wording for wording in news.RETRACTION_FORMS + news.CORRECTION_FORMS)


SEPARATOR = re.compile(r"\n\n\n(?:_{32}\n\n|\n)")  # between two messages of a thread


def thread(text: str) -> list[tuple[dict[str, str], str, str]]:
    """Each message of an email thread: its headers by name, the first line of its own text, and its text."""
    messages = []
    for chunk in SEPARATOR.split(text):
        head, _, body = chunk.partition("\n\n")
        headers = dict(line.split(": ", 1) for line in head.split("\n"))
        messages.append((headers, body.split("\n", 1)[0], body))
    return messages


def test_every_email_acknowledgement_is_titled_after_the_message_it_answers():
    """code-genres-2#3: only an acknowledgement right after a forward took the forward's subject; one after
    that acknowledgement was titled after the thread ("Re: Latest weather observations (2 Jan 2026)") while
    it quoted a message whose subject was "Re: Fwd: CORRECTION: ...". Every acknowledgement is now titled "Re:
    " and the subject of the message it answers, which it keeps when that is a reply's already."""
    weather = get_domain("weather")
    nested = 0
    for seed in range(160):
        reports, notices = digest_items(weather, seed % 4)
        messages = thread(render("email", weather, reports, notices, seed, notices="separate"))
        for k in range(1, len(messages)):
            headers, first, _ = messages[k]
            if first not in email.ACKS:
                continue
            answered = messages[k - 1][0]["Subject"]
            assert headers["Subject"] == (answered if answered.startswith("Re: ") else f"Re: {answered}"), seed
            nested += messages[k - 1][1] in email.ACKS and "Fwd: " in answered
    assert nested  # an acknowledgement of an acknowledgement of a forward was printed


def test_email_greetings_and_subjects_agree_with_their_date_headers():
    """code-genres-2#5, blind-email#3: "Morning all," opened messages sent at 20:43 or 01:39 local time, and a
    thread's subject was dated by the UTC day of its first message, which could be the day after the one its
    Date header gives. A greeting that names a part of the day is now drawn only then, and the subject is
    dated by the first message's own day."""
    mornings = 0
    for domain in (get_domain("weather"), get_domain("earthquakes")):
        for seed in range(80):
            reports, notices = digest_items(domain, seed % 3)
            messages = thread(render("email", domain, reports, notices, seed))
            first_headers = messages[0][0]
            day = " ".join(first_headers["Date"].split()[1:4])  # "Fri, 29 May 2026 07:23:39 +1200"
            assert day in first_headers["Subject"], (first_headers["Date"], first_headers["Subject"])
            for headers, first, _ in messages:
                if first == "Morning all,":
                    assert 5 <= int(headers["Date"].split()[4][:2]) < 12, headers["Date"]
                    mornings += 1
    assert mornings


def test_a_message_that_forwards_several_notices_says_so():
    """blind-email#4: a message that forwarded two notices announced "a notice", so the second could look
    nested in the first."""
    weather = get_domain("weather")
    plural = 0
    for seed in range(40):
        reports, notices = digest_items(weather, seed % 4)
        notices = [*notices, Notice("retraction", reports[1].ref, COMPILED_AT - 900, reason="sensor fault")]
        for headers, first, body in thread(render("email", weather, reports, notices, seed, notices="attached")):
            forwards = sum(line in ("---------- Forwarded message ----------", "Begin forwarded message:")
                           for line in body.split("\n"))
            if first in email.GREETINGS and forwards:
                intros = email.ATTACHED_INTROS if forwards == 1 else email.ATTACHED_INTROS_PLURAL
                assert any(intro in body for intro in intros), body
                plural += forwards > 1
    assert plural


def test_no_radio_report_is_introduced_as_a_message():
    """blind-radio#0: "message follows" introduced readouts, though the rules call a notice "a message about a
    report" and say that "A WIRE line is a report, not a message"; transcripts printed before still read."""
    assert not any("message" in intro for intro in radio.REPORT_INTROS)
    weather = get_domain("weather")
    reports, notices = digest_items(weather, 3)
    noun = weather.glossary().noun
    for seed in range(20):
        text = render("radio", weather, reports, notices, seed)
        assert "message follows" not in text
        old = re.sub(rf", (?:{noun} report follows|{noun} details follow|report follows): ", ", message follows: ", text)
        assert old != text
        assert radio.RADIO.parse(old, weather.glossary(), weather) == radio.RADIO.parse(text, weather.glossary(),
                                                                                         weather)


# ------------------------------------------------------------------------------------- forms


def test_a_form_amendment_stays_in_the_section_of_the_source_that_filed_its_report():
    """finish3#6 (a): in the by-source layout an interleaved amendment could land under the heading of another
    source's forms."""
    checked = 0
    for domain in (get_domain("weather"), get_domain("earthquakes")):
        reports, notices = digest_items(domain, 5)
        glossary = domain.glossary()
        filed: dict[str, set[str]] = {}
        for r in map(domain.to_generic, reports):
            filed.setdefault(r.ref, set()).add(r.source)
        heads = {head.format(source=source, source_label=glossary.source_label,
                             Source_label=form._cap(glossary.source_label)): source
                 for sources in filed.values() for source in sources for head in form.SECTION_HEADS}
        for seed in range(30):
            section = None
            text = render("form", domain, reports, notices, seed, layout="by_source", notices="interleaved")
            for block in text.split("\n\n")[1:]:
                if block in heads:
                    section = heads[block]
                    continue
                for notice in form.FORM.parse(block, glossary, domain).notices:
                    assert section in filed[notice.target], (domain.name, seed, notice.target, section)
                    checked += 1
    assert checked >= 100


FIRST_FILED = "(?:Entry as first filed|As first filed|Entry as filed|Value as first filed)"


def _printed_unit(block: str, label: str) -> tuple[str, str | None] | None:
    """The number and unit symbol that a form block prints after `label` (a regex): in the label, "Wind speed
    (kt): 12", or after the number, "Wind speed: 12 kt"."""
    match = re.search(rf"{label}(?: \((?P<a>[^)]+)\))?(?::| \.{{3,}}) +(?P<n>{NUMBER})(?: (?P<b>[^\s│]+))?", block)
    return None if match is None else (match["n"], match["a"] or match["b"])


def test_a_form_amendment_gives_the_entry_as_first_filed_in_the_unit_of_the_form_it_amends():
    """finish3#6 (b): an amendment printed the entry "as first filed" in a unit of its own, not in the one the
    form it names printed it in."""
    weather = get_domain("weather")
    glossary = weather.glossary()
    reports, notices = digest_items(weather, 6)
    (correction,) = [n for n in notices if n.kind == "correction"]
    label = form._cap(next(f.label for f in glossary.fields if f.key == correction.field))
    units = Counter()
    for seed in range(30):
        blocks = render("form", weather, reports, notices, seed).split("\n\n")
        (original,) = [b for b in blocks if [r.ref for r in form.FORM.parse(b, glossary, weather).reports]
                       == [correction.target]]
        (amendment,) = [b for b in blocks if form.FORM.parse(b, glossary, weather).notices
                        and "CORRECTION" in b.split("\n")[0] + b.split("\n")[1]]
        assert _printed_unit(original, re.escape(label)) == _printed_unit(amendment, FIRST_FILED), seed
        units[_printed_unit(original, re.escape(label))[1]] += 1
    assert len(units) > 1  # in more than one unit


def test_a_form_labels_a_qualifier_in_the_glossarys_words(monkeypatch):
    """finish3#6 (c): a row was labelled "Magnitude, mag_type" (and "mag_type", "mag_type for magnitude",
    "Corrected mag_type"); a qualifier the glossary names in a reader's words is labelled with them, and a
    batch printed with the old labels still reads."""
    quakes = get_domain("earthquakes")
    reports, notices = digest_items(quakes, 1)
    glossary = quakes.glossary()
    new = [render("form", quakes, reports, notices, seed) for seed in range(6)]
    assert all("mag_type" not in text and "Magnitude type" in text for text in new)
    keyed = form._label
    monkeypatch.setattr(form, "_label", lambda frame, slot, g, keyed_=False, **_: keyed(frame, slot, g, keyed=True))
    form._grammar.cache_clear()
    try:
        old = [render("form", quakes, reports, notices, seed) for seed in range(6)]
    finally:
        monkeypatch.undo()
        form._grammar.cache_clear()
    assert all("mag_type" in text for text in old)
    for before, now in zip(old, new):
        assert form.FORM.parse(before, glossary, quakes) == form.FORM.parse(now, glossary, quakes)


# ------------------------------------------------------------------- wordings printed before


def test_the_wordings_genres_printed_before_still_read():
    """The wordings that the fixes replaced (an email's and a table's "now", a radio net's "now reads" and
    "amended from", a news retraction "listed in this article") are still read, so that a run generated
    before them reads as it did."""
    weather = get_domain("weather")
    glossary = weather.glossary()
    _, notices = digest_items(weather, 0)
    replaced = Counter()
    rewordings = {
        "email": [(f" {REVISED_TO} ", " now ", "CORRECTION NOTICE (")],
        "table": [(f" {REVISED_TO} ", " now ", "Correction (")],
        "radio": [(f" {REVISED_TO} ", " now reads ", "CORRECTION, time of issue"),
                  (f" {REVISED_FROM} ", " amended from ", "CORRECTION issued")],
        "news": [(" was not a real ", ", listed in this article, was not a real ", "Retraction notice (")],
    }
    for name, changes in rewordings.items():
        genre = get_genre(name)
        for seed in range(40):
            lines = render(name, weather, [], notices, seed).split("\n")
            read = genre.parse("\n".join(lines), glossary, weather)
            for i, line in enumerate(lines):
                for new, old, opening in changes:
                    if opening in line and new in line and not line.startswith(">"):
                        before = "\n".join([*lines[:i], line.replace(new, old, 1), *lines[i + 1:]])
                        assert genre.parse(before, glossary, weather) == read, line
                        replaced[name, old] += 1
    assert len(replaced) == 5, replaced
