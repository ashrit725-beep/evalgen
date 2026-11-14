"""news: a digest written as a news article, for the records of every domain.

An article opens with a headline and a byline that gives its publication time, then a dateline that
starts the lede. Every report is a paragraph of its own that states all of the report's fields in
prose, in one of five sentence forms. Quotes from officials, a background paragraph, the domain's
chatter lines, section subheads and a sign-off are mixed in. Notices are paragraphs too: an "Update",
a "Correction", a "Retraction notice" or a "Correction notice", placed at the top of the article, among
its paragraphs or at its end. No notice is headed as an editor's note: the reading rules count
editor's notes among the lines that carry no data, as the domains' chatter prints them.

    The ukmo operator reported reading by-12345 (reviewed), recorded at 06:12:19 UTC on 12 April 2026 in
    North Sea, at 56.512°N, 3.104°E, with water temperature 12.3 °C (spar) and wave height 1.85 m.

    At 7:12:19 AM local time (UTC+01:00) on Apr 12, 2026, the gts operator recorded reading by-54321
    in North Sea, at (56.522, 3.118), listing a water temperature of 54.3 °F (spar) and a wave height
    of 6.07 ft; the report is marked preliminary. Filing note: duplicate of by-12345.

    Update (2026-04-12 14:05:33 UTC): report by-67890 has been retracted (probe fault); it was not a
    real reading and should be disregarded.

    Correction (Sun, 12 Apr 2026 15:20:00 +0000): the water temperature of reading by-12345 was
    revised to 12.8 °C (spar); it was first given as 14.1 °C.

Only report and notice paragraphs carry data. The headline, byline, lede, quotes, background, subheads
and sign-off are worded from the records' words alone (their noun, the source label, the time window,
a source's name), never from a report: no ID, value or record time appears in them, and the only time
they give is the article's own publication date. So the reading rules settle every line: a notice
names itself with words that both the 0.2.0 rules and every domain's record rules name ("retracted"
or "retraction" with "was not a real <noun>"; "revised" with "correction"), and every other line
carries no data.

Every word of a paragraph comes from the prompt's glossary (the records' noun, the source label, the
field labels, units and qualifiers, the status words and the ID pattern) or from the domain's trap
vocabulary (duplicate notes, retraction phrases), which the record rules of every domain but earthquakes
quote, and the 0.2.0 earthquake rules name by what they say ("a note naming the event ID it duplicates",
"withdrawn"). Each value is written
in a unit the style draws with the exact codec of evalgen.units, times to the second with their UTC
offset stated, and positions as stored. The parser reads each paragraph back through the glossary
alone; every free word (a source, a group, a place) sits between typed slots (an ID, a status, a
time, a position, a value in a unit) or long literal text, so no free word can take a delimiter for
its own. Every rendering is checked to read back to exactly its reports and notices
(genres.base.check_round_trip), so the text-only oracle recovers them all.

Earthquakes keep their 0.2.0 record types: their reports are models.Report, which hold their values
and magnitude type as attributes, and their corrections revise the magnitude (new_mag, old_mag,
mag_type). This genre writes them in the generic form of every domain (Domain.to_generic) and reads
them back through Domain.from_generic (genres.base.generic_items and domain_items), in the words of the
earthquake glossary.
"""
from __future__ import annotations

import functools
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

from ..digest import MONTHS_LONG
from ..domains.base import Domain, Glossary, GlossaryField, RecordReport, TrapVocab
from ..domains.earthquakes import Earthquakes
from ..models import Notice
from ..ops import DigestData
from ..units import Unit
from .base import (COORDS_PATTERN, GROUP_PREPOSITION, GROUP_PREPOSITION_CAPITALISED, TIME_FORMATS, TIME_PATTERN,
                   UTC_TIME_FORMATS, Genre, RenderContext, alternation, check_round_trip, domain_items, format_coords,
                   format_time, format_value, generic_items, id_pattern, parse_coords, parse_time, read_value,
                   value_pattern)

# ---------------------------------------------------------------------------------- templates
# A template is literal text with {slots}. A part in [brackets] is about the records' position: it is
# written only when they have one. The same template writes a paragraph (_fill) and reads one back
# (_pattern), where every slot is a regex: typed (an ID, a status, a time, a position, values in units)
# or free words (a source, a group) bounded on both sides by typed slots or long literal text.
#
# A report's time is always the time its record was recorded ("recorded at"), never the time a source
# filed or sent the report, as the reading rules define it. A record is placed in its group with its
# domain's preposition, {in} ("in North Sea", "near Moon": Domain.group_preposition).

REPORT_FORMS = {
    "attributed": ("The {source} {source_label} reported {noun} {ref} ({status}), recorded at {when} {in} {group}"
                   "[, at {where}], with {values}."),
    "record_first": ("{Noun} {ref}, filed as {status} by the {source} {source_label}, was recorded at {when} {in} "
                     "{group}[, at {where}]. It gave {values}."),
    "time_first": ("At {when}, the {source} {source_label} recorded {noun} {ref} {in} {group}[, at {where}], listing "
                   "{values}; the report is marked {status}."),
    "figures_first": ("{Values}: those were the figures for {noun} {ref}, filed as {status} by the {source} "
                      "{source_label} and recorded at {when} {in} {group}[, at {where}]."),
    "place_first": ("{In} {group}[, at {where}], {noun} {ref} was recorded at {when} by the {source} {source_label}: "
                    "{values}. Status: {status}."),
}
# A re-report's note ("{frame} {note} {ID}.") and a place note ("{frame} {place}.") end its paragraph.
NOTE_FRAMES = ("Filing note:", "The {source_label} adds:", "Note from the {source_label}:")
PLACE_FRAMES = ("Location note:", "Location given:")

# Every notice names itself with words that both the 0.2.0 rules and the record rules name, and none is
# headed as an editor's note, which the rules count among the lines without data.
RETRACTION_FORMS = (
    "Update ({when}): report {ref} has been retracted{reason}; it was not a real {noun} and should be disregarded.",
    "Update, {when}: report {ref} is now marked {phrase}{reason}. This is a retraction: it was not a real {noun}.",
    "Retraction notice ({when}): report {ref}, listed in this article, was not a real {noun}{reason}; disregard it.",
)
CORRECTION_FORMS = (
    "Correction ({when}): the {label} of {noun} {ref} was revised to {new}{qualifier}; it was first given as {old}.",
    "Update, {when}: in a correction, the {label} of report {ref} was revised from {old} to {new}{qualifier}.",
    "Correction notice ({when}): the {label} of report {ref} has been revised to {new}{qualifier}; it was first "
    "given as {old}.",
)
END_HEADINGS = ("Corrections and updates", "Updates to this article")  # over the notices at the end

# ------------------------------------------------------------------------- article furniture
# Lines without data, worded from the records' words: {noun}, {plural}, {Plural}, {source_label} and
# {span} (the time window), and the article's own people and masthead.

HEADLINES = (
    "{Plural}: every report, as each {source_label} filed it",
    "What each {source_label} reported: the latest {plural}",
    "A busy {span} for {plural}, report by report",
    "{Plural} in review: the past {span} as filed",
    "New {noun} reports pour in from every {source_label}",
)
BYLINE = "By {reporter}, {desk} | Published {published}"  # the article's own time: never a record's
DESKS = ("staff reporter", "data reporter", "news desk", "wire desk", "night editor")
PEOPLE = ("Maria Okafor", "Daniel Reyes", "Priya Natarajan", "Tomas Lindqvist", "Grace Mwangi", "Henrik Aalto",
          "Leila Haddad", "Samuel Osei", "Ana Ferreira", "Kenji Watanabe", "Fiona Gallagher", "Omar Siddiqui")
AGENCIES = ("Meridian Wire", "Northgate News", "The Harbour Ledger", "Coastline Post", "Fieldstone Press")
DATELINES = ("PORT HALLAM", "KESTON", "ALDERMOOR", "WEST LINTON", "BRAYFORD", "EASTMERE")  # fictional desks
LEDES = (
    "Every {source_label} that tracks {plural} has filed new reports over the past {span}; each one is set out "
    "below as it was filed.",
    "The past {span} brought a steady stream of {noun} reports. Here is what each {source_label} said, as filed.",
    "Readers asked for the full list, so here it is: every {noun} report of the past {span}, as each "
    "{source_label} filed it.",
)
QUOTES = (
    "\"We are still going through everything that came in,\" said {person}, {role}.",
    "\"It has been a long {span} for everyone who files {noun} reports,\" said {person}, {role}.",
    "\"We publish what we have as soon as we have it,\" said {person}, {role}.",
    "{person}, {role}, said the team would \"keep filing for as long as there is something to file.\"",
    "\"Nothing about the past {span} has been routine,\" {person}, {role}, told {agency}.",
)
ROLES = ("a spokesperson for the {source} {source_label}", "a duty officer with the {source} {source_label}",
         "who coordinates reporting for the {source} {source_label}")
BACKGROUND = (
    "Each {source_label} files its own reports, units and time conventions vary from one report to the next, and "
    "newsrooms such as this one compile them as they arrive.",
    "Reports like these are shared through common feeds; this article reproduces each one as it was filed.",
    "Round-ups of {plural} appear on this page every {span}; earlier editions remain in the archive.",
)
# Over each group's paragraphs, by the domain's group preposition: a body is no place reports come from.
SUBHEADS = {"in": ("Reports from {group}", "Across {group}", "{group}: the latest"),
            "near": ("Reports near {group}", "Near {group}", "{group}: the latest")}
SIGNOFFS = ("Reporting by {reporter}; editing by {editor}.", "({agency} reporting by {reporter}; editing by {editor})")

# ------------------------------------------------------------------------------ the look
# A rendering style picks one preset per dimension, as classic does; each preset is a weighted mix.

PARAGRAPH_PRESETS = {
    "straight": {"attributed": 0.35, "record_first": 0.35, "time_first": 0.1, "figures_first": 0.1, "place_first": 0.1},
    "feature": {"attributed": 0.1, "record_first": 0.1, "time_first": 0.25, "figures_first": 0.25, "place_first": 0.3},
    "mixed": {form: 1.0 for form in REPORT_FORMS},
}
# "plain": water temperature 12.3 °C; "spelled": a water temperature of 12.3 °C.
VALUE_PRESETS = {
    "terse": {"plain": 0.8, "spelled": 0.2},
    "spelled": {"plain": 0.2, "spelled": 0.8},
    "mixed": {"plain": 0.5, "spelled": 0.5},
}
TIME_PRESETS = {
    "utc": {"prose_utc": 0.35, "iso_space": 0.15, "iso": 0.1, "rfc": 0.1, "iso_offset": 0.1, "local12": 0.1,
            "local24": 0.1},
    "local": {"local12": 0.35, "local24": 0.2, "iso_offset": 0.15, "prose_utc": 0.15, "iso": 0.05, "iso_space": 0.05,
              "rfc": 0.05},
    "mixed": {fmt: 1.0 for fmt in TIME_FORMATS},
}
COORD_PRESETS = {
    "hemi": {"hemi": 0.7, "signed": 0.3},
    "signed": {"hemi": 0.3, "signed": 0.7},
    "mixed": {"hemi": 0.5, "signed": 0.5},
}
# The weight of a field's stored unit in each preset; the others share the rest (None: all alike).
UNIT_PRESETS: dict[str, float | None] = {"mostly_stored": 0.7, "mostly_other": 0.15, "mixed": None}
ORDERS = ("by_group", "by_time", "shuffled")  # by_group: a subhead over each group's paragraphs
NOTICE_PLACEMENTS = ("end", "interleaved", "top")
ZONES = (-12, 14)  # the UTC offsets, in whole hours, of sources whose records have no position

# ------------------------------------------------------------------------------ the genre


class News(Genre):
    name = "news"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One preset per dimension (a variant avoids each of the original's), the headline and the
        masthead."""

        def pick(key: str, options: Sequence[str] | Mapping[str, Any]) -> str:
            names = sorted(options)
            if avoid is not None and key in avoid:
                names = [n for n in names if n != avoid[key]] or names
            return rng.choice(names)

        style = {"paragraphs": pick("paragraphs", PARAGRAPH_PRESETS), "values": pick("values", VALUE_PRESETS),
                 "units": pick("units", UNIT_PRESETS), "time": pick("time", TIME_PRESETS)}
        if domain.has_location:
            style["coords"] = pick("coords", COORD_PRESETS)
        style.update(order=pick("order", ORDERS), notices=pick("notices", NOTICE_PLACEMENTS),
                     headline=rng.randrange(len(HEADLINES)), masthead=rng.randrange(len(AGENCIES)))
        return style

    def render(self, ctx: RenderContext) -> str:
        """The article, in the words of the context's glossary (the domain's own when it has none),
        checked to be recognised as news and to read back to exactly its reports and notices."""
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _write(ctx, glossary)
        if not self.sniff(text):
            raise ValueError("the news genre does not recognise its own rendering (does a word of the domain or its "
                             "surface hold a line break?)")
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """An article opens with one of its headlines and, on the next line, its byline. Neither is
        anything another genre writes; a quoted article ("> ") is not an article."""
        lines = digest_text.lstrip().split("\n", 2)
        return (len(lines) >= 2 and any(p.fullmatch(lines[0]) for p in _HEADLINE_LINES)
                and _BYLINE_LINE.fullmatch(lines[1]) is not None)

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """Every report and notice of the article, read through the prompt's glossary (earthquakes,
        whose words are fixed, also without one), as the domain's digests hold them."""
        if glossary is None:
            if not isinstance(domain, Earthquakes):
                raise ValueError(f"a digest of {domain.name} records is read through the glossary its prompt prints, "
                                 "and this prompt prints none")
            glossary = domain.glossary()
        data = _read(digest_text, glossary)
        return DigestData(domain_items(domain, data.reports), domain_items(domain, data.notices))


NEWS = News()
GENRE = NEWS  # what the registry looks up (see evalgen.genres)

# ----------------------------------------------------------------------------- templates

_TOKEN = re.compile(r"(\{\w+\}|\[|\])")


def _pieces(template: str, located: bool) -> Iterator[tuple[str | None, str]]:
    """A template's pieces in order: (slot name, "") for a slot, (None, text) for literal text. A part
    in brackets is kept only when the records have a position (`located`)."""
    keep = True
    for piece in _TOKEN.split(template):
        if piece in ("[", "]"):
            keep = piece == "]" or located
        elif piece and keep:
            yield (piece[1:-1], "") if piece.startswith("{") else (None, piece)


def _fill(template: str, slots: Mapping[str, str], located: bool = True) -> str:
    """The template written with the text of each slot."""
    return "".join(text if name is None else slots[name] for name, text in _pieces(template, located))


def _pattern(template: str, slots: Mapping[str, str], located: bool = True) -> str:
    """A regex for the template, its literal text escaped and each slot replaced by its pattern."""
    return "".join(re.escape(text) if name is None else slots[name] for name, text in _pieces(template, located))


def _wildcard(template: str) -> re.Pattern[str]:
    """A line written from `template` with any words in its slots (and a time in {published}). A slot
    that opens the line opens with a letter or digit, so that a quoted line ("> ...") is none."""
    parts = []
    for i, (name, text) in enumerate(_pieces(template, True)):
        if name is None:
            parts.append(re.escape(text))
        else:
            parts.append(f"(?:{TIME_PATTERN})" if name == "published" else r"[^\W_].*?" if i == 0 else ".+?")
    return re.compile("".join(parts))


_HEADLINE_LINES = tuple(_wildcard(headline) for headline in HEADLINES)
_BYLINE_LINE = _wildcard(BYLINE)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in ("a", "e", "i", "o", "u") else "a"


def _span(hours: int) -> str:
    """The time window an article covers, as "the past {span}" says it: "day", "week", "3 days", "6 hours"."""
    named = {1: "hour", 24: "day", 168: "week"}
    if hours in named:
        return named[hours]
    return f"{hours // 24} days" if hours % 24 == 0 else f"{hours} hours"


def _listed(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _value_head(field: GlossaryField, spelled: bool) -> str:
    """What precedes a field's value: its label, or "a <label> of"."""
    return f"{_article(field.label)} {field.label} of" if spelled else field.label


# -------------------------------------------------------------------------- writing articles


def _draw(rng: random.Random, table: Mapping[str, float]) -> str:
    return rng.choices(list(table), weights=list(table.values()), k=1)[0]


def _draw_unit(rng: random.Random, field: GlossaryField, preset: str) -> Unit | None:
    """The unit one value of `field` is printed in (None for a field without units)."""
    if not field.units:
        return None
    stored = UNIT_PRESETS[preset]
    if stored is None or len(field.units) == 1:
        return rng.choice(field.units)
    others = (1.0 - stored) / (len(field.units) - 1)
    weights = [stored if unit.symbol == field.canonical_unit else others for unit in field.units]
    return rng.choices(field.units, weights=weights, k=1)[0]


def _write(ctx: RenderContext, glossary: Glossary) -> str:
    """The article of one rendering, drawn in this order: each report's paragraph, the order of the
    paragraphs, each notice's paragraph, the people, the quotes and the background, where they and the
    chatter go, where the notices go, and last the byline, the dateline and lede, and the sign-off.
    Paragraphs are separated by blank lines; the headline (the style's) and the byline open the article.
    Reports and notices are written in their generic form (genres.base.generic_items)."""
    rng, style = ctx.rng, ctx.style
    vocab = ctx.domain.trap_vocab
    reports = generic_items(ctx.domain, ctx.reports)
    sources = sorted({r.source for r in reports})
    zones = {} if glossary.located else {source: rng.randint(*ZONES) for source in sources}
    paragraphs = [(r, _report_paragraph(r, rng, style, glossary, zones, vocab, ctx.domain)) for r in reports]
    body = _arranged(paragraphs, rng, style["order"], SUBHEADS[ctx.domain.group_preposition])
    notices = [_notice_paragraph(n, rng, glossary, vocab, style["units"]) for n in generic_items(ctx.domain, ctx.notices)]

    words = {"noun": glossary.noun, "plural": glossary.plural, "Plural": _cap(glossary.plural),
             "source_label": glossary.source_label, "span": _span(ctx.window_hours)}
    agency = AGENCIES[style["masthead"]]
    reporter, editor, *officials = rng.sample(PEOPLE, 4)
    extras = []
    for quote in rng.sample(QUOTES, rng.randint(1, 2)):
        role = rng.choice(ROLES).format(source=rng.choice(sources), **words) if sources else "a spokesperson"
        extras.append(quote.format(person=officials.pop(), role=role, agency=agency, **words))
    extras.append(rng.choice(BACKGROUND).format(**words))
    for text in [*extras, *ctx.chatter]:
        body.insert(rng.randrange(len(body) + 1), text)

    top: list[str] = []
    end: list[str] = []
    if style["notices"] == "interleaved":
        for text in notices:
            body.insert(rng.randrange(len(body) + 1), text)
    elif notices:
        rng.shuffle(notices)
        if style["notices"] == "top":
            top = notices
        else:
            end = [rng.choice(END_HEADINGS), *notices]

    byline = BYLINE.format(reporter=reporter, desk=rng.choice(DESKS),
                           published=format_time(ctx.compiled_at, rng.choice(UTC_TIME_FORMATS)))
    day = datetime.fromtimestamp(ctx.compiled_at, tz=timezone.utc)
    lede = (f"{rng.choice(DATELINES)}, {MONTHS_LONG[day.month - 1]} {day.day} ({agency}) — "
            + rng.choice(LEDES).format(**words))
    signoff = rng.choice(SIGNOFFS).format(reporter=reporter, editor=editor, agency=agency)
    head = HEADLINES[style["headline"]].format(**words) + "\n" + byline
    return "\n\n".join([head, *top, lede, *body, signoff, *end])


def _report_paragraph(r: RecordReport, rng: random.Random, style: Mapping[str, Any], glossary: Glossary,
                      zones: Mapping[str, int], vocab: TrapVocab, domain: Domain) -> str:
    """One report as a paragraph: a sentence form, a time (in the local time of its position, as its
    domain gives it, or of its source when records have none), a position, the values with their units
    and qualifiers, then its note naming the record it repeats and its place note, when it has them. The
    record is placed in its group with the domain's preposition (Domain.group_preposition)."""
    form = REPORT_FORMS[_draw(rng, PARAGRAPH_PRESETS[style["paragraphs"]])]
    offset = domain.utc_offset(r) if glossary.located else zones[r.source]
    preposition = domain.group_preposition
    when = format_time(r.time, _draw(rng, TIME_PRESETS[style["time"]]), offset)
    where = (format_coords(r.lat, r.lon, _draw(rng, COORD_PRESETS[style["coords"]]), glossary.coord_decimals)
             if glossary.located else "")
    spelled = _draw(rng, VALUE_PRESETS[style["values"]]) == "spelled"
    values = []
    for f in glossary.fields:
        number = format_value(f, r.values[f.key], _draw_unit(rng, f, style["units"]))
        qualifier = None if f.qualifier is None else r.qualifiers.get(f.qualifier)
        values.append(f"{_value_head(f, spelled)} {number}" + ("" if qualifier is None else f" ({qualifier})"))
    listed = _listed(values)
    text = _fill(form, {"source": r.source, "source_label": glossary.source_label, "noun": glossary.noun,
                        "Noun": _cap(glossary.noun), "ref": r.ref, "status": r.status, "when": when, "group": r.group,
                        "where": where, "values": listed, "Values": _cap(listed), "in": preposition,
                        "In": _cap(preposition)}, glossary.located)
    if r.same_as is not None:
        frame = _fill(rng.choice(NOTE_FRAMES), {"source_label": glossary.source_label})
        text += f" {frame} {rng.choice(vocab.duplicate_notes)} {r.same_as}."
    if r.place is not None:
        text += f" {rng.choice(PLACE_FRAMES)} {r.place}."
    return text


def _arranged(paragraphs: Sequence[tuple[Any, str]], rng: random.Random, order: str,
              subheads: Sequence[str] = SUBHEADS["in"]) -> list[str]:
    """The report paragraphs in the article's order: by time, by group under a subhead each (one of
    `subheads`; groups in a drawn order), or shuffled."""
    if order == "by_time":
        return [text for _, text in sorted(paragraphs, key=lambda pair: (pair[0].time, pair[0].ref, pair[0].source))]
    if order == "by_group":
        groups = sorted({r.group for r, _ in paragraphs})
        rng.shuffle(groups)
        body = []
        for group in groups:
            body.append(rng.choice(subheads).format(group=group))
            texts = [text for r, text in paragraphs if r.group == group]
            rng.shuffle(texts)
            body.extend(texts)
        return body
    if order != "shuffled":
        raise ValueError(f"unknown paragraph order {order!r}")
    shuffled = list(paragraphs)
    rng.shuffle(shuffled)
    return [text for _, text in shuffled]


def _notice_paragraph(n: Notice, rng: random.Random, glossary: Glossary, vocab: TrapVocab, units: str) -> str:
    """A notice in one of three wordings of its kind, issued at a UTC time. A retraction gives its
    reason, and one wording quotes a phrase of the domain's trap vocabulary; a correction gives both
    values in one drawn unit, and the new value's qualifier."""
    when = format_time(n.time, rng.choice(UTC_TIME_FORMATS))
    if n.kind == "retraction":
        template = rng.choice(RETRACTION_FORMS)
        phrase = rng.choice(vocab.retraction_notices) if "{phrase}" in template else ""
        return _fill(template, {"when": when, "ref": n.target, "noun": glossary.noun,
                                "reason": "" if n.reason is None else f" ({n.reason})",
                                "phrase": json.dumps(phrase, ensure_ascii=False)})
    key, new, old, qualifier = n.field, n.new_value, n.old_value, n.qualifier
    field = next((f for f in glossary.fields if f.key == key), None)
    if n.kind != "correction" or field is None or new is None or old is None:
        raise ValueError(f"a {n.kind} notice of {n.target} that names no field of the glossary, or no new and old "
                         "value, cannot be printed")
    unit = _draw_unit(rng, field, units)
    return _fill(rng.choice(CORRECTION_FORMS), {
        "when": when, "ref": n.target, "noun": glossary.noun, "label": field.label,
        "new": format_value(field, new, unit), "old": format_value(field, old, unit),
        "qualifier": "" if qualifier is None else f" ({qualifier})"})


# -------------------------------------------------------------------------- reading articles

_QUOTED = r'"(?:[^"\\]|\\.)*"'  # a phrase, written as a JSON string


@dataclass(frozen=True)
class _Grammar:
    """The patterns of one glossary's articles: a whole report paragraph in each sentence form and value
    phrasing, a whole retraction in each wording, and a whole correction of each field in each wording."""

    reports: tuple[re.Pattern[str], ...]
    retractions: tuple[re.Pattern[str], ...]
    corrections: tuple[tuple[re.Pattern[str], GlossaryField], ...]


def _values_pattern(fields: Sequence[GlossaryField], spelled: bool, capital: bool = False) -> str:
    """A regex for the values of a report, as _report_paragraph lists them (groups v{i}, v{i}_unit and q{i};
    `capital`: the list opens a sentence)."""
    parts = []
    for i, f in enumerate(fields):
        head = _value_head(f, spelled)
        qualifier = "" if f.qualifier is None else rf"(?: \((?P<q{i}>[^()]*)\))?"
        parts.append(re.escape(_cap(head) if capital and i == 0 else head) + " " + value_pattern(f, f"v{i}")
                     + qualifier)
    return _listed(parts)


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    """The patterns of the articles written in a glossary's words (cached: a run reads many of them)."""
    record_id = id_pattern(glossary)
    ref, when = f"(?P<ref>{record_id})", f"(?P<time>{TIME_PATTERN})"
    words = {"source_label": re.escape(glossary.source_label), "noun": re.escape(glossary.noun),
             "Noun": re.escape(_cap(glossary.noun))}
    notes = "|".join(_pattern(frame, words) for frame in NOTE_FRAMES)
    places = "|".join(re.escape(frame) for frame in PLACE_FRAMES)
    tail = rf"(?: (?:{notes}) (?P<note>.+?) (?P<same>{record_id})\.)?(?: (?:{places}) (?P<place>.+)\.)?"
    reports = []
    for spelled in (False, True):
        slots = {**words, "source": "(?P<source>.+?)", "ref": ref, "when": when, "group": "(?P<group>.+?)",
                 "in": GROUP_PREPOSITION, "In": GROUP_PREPOSITION_CAPITALISED,
                 "status": f"(?P<status>{alternation(glossary.statuses)})", "where": f"(?P<coords>{COORDS_PATTERN})",
                 "values": _values_pattern(glossary.fields, spelled),
                 "Values": _values_pattern(glossary.fields, spelled, capital=True)}
        reports += [re.compile(_pattern(form, slots, glossary.located) + tail) for form in REPORT_FORMS.values()]
    retracted = {"when": when, "ref": ref, "noun": words["noun"], "reason": r"(?: \((?P<reason>.+)\))?",
                 "phrase": _QUOTED}
    corrections = []
    for f in glossary.fields:
        revised = {"when": when, "ref": ref, "noun": words["noun"], "label": re.escape(f.label),
                   "new": value_pattern(f, "new"), "old": value_pattern(f, "old"),
                   "qualifier": "" if f.qualifier is None else r"(?: \((?P<qualifier>[^()]*)\))?"}
        corrections += [(re.compile(_pattern(form, revised)), f) for form in CORRECTION_FORMS]
    return _Grammar(reports=tuple(reports),
                    retractions=tuple(re.compile(_pattern(form, retracted)) for form in RETRACTION_FORMS),
                    corrections=tuple(corrections))


def _read(text: str, glossary: Glossary) -> DigestData:
    """Every report and notice of an article, in their generic form; paragraphs that are neither change
    nothing."""
    grammar = _grammar(glossary)
    reports: list[RecordReport] = []
    notices: list[Notice] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        notice = _read_notice(line, grammar)
        if notice is not None:
            notices.append(notice)
            continue
        report = _read_report(line, grammar, glossary)
        if report is not None:
            reports.append(report)
    return DigestData(reports, notices)


def _read_notice(line: str, grammar: _Grammar) -> Notice | None:
    """The notice a paragraph states, or None when it states none (or a time that is none)."""
    for pattern in grammar.retractions:
        match = pattern.fullmatch(line)
        if match is not None:
            when = parse_time(match["time"])
            return None if when is None else Notice(kind="retraction", target=match["ref"], time=when,
                                                    reason=match["reason"])
    for pattern, field in grammar.corrections:
        match = pattern.fullmatch(line)
        if match is not None:
            when = parse_time(match["time"])
            if when is None:
                return None
            return Notice(kind="correction", target=match["ref"], time=when, field=field.key,
                          new_value=read_value(field, match, "new"), old_value=read_value(field, match, "old"),
                          qualifier=match.groupdict().get("qualifier"))
    return None


def _read_report(line: str, grammar: _Grammar, glossary: Glossary) -> RecordReport | None:
    """The report a paragraph states, or None when it states none (or a time or position that is none)."""
    for pattern in grammar.reports:
        match = pattern.fullmatch(line)
        if match is None:
            continue
        when = parse_time(match["time"])
        position = parse_coords(match["coords"], glossary.coord_decimals) if glossary.located else (None, None)
        if when is None or position is None:
            return None
        values = {f.key: read_value(f, match, f"v{i}") for i, f in enumerate(glossary.fields)}
        qualifiers = {f.qualifier: match[f"q{i}"] for i, f in enumerate(glossary.fields)
                      if f.qualifier is not None and match[f"q{i}"] is not None}
        return RecordReport(ref=match["ref"], source=match["source"], status=match["status"], time=when,
                            lat=position[0], lon=position[1], group=match["group"], values=values,
                            qualifiers=qualifiers, place=match["place"], same_as=match["same"])
    return None
