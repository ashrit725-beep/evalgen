"""table: the digest as a table of reports, in CSV or markdown, for the records of any domain.

One row per report, re-reports and preliminary reports included, under a header row that names each
column in the words of the prompt's glossary: the records' ID label, source label and group label,
"status", "time", the position ("position", or "latitude" and "longitude"), each field's label, the
key of each field's qualifier, a "note" column for the notes that name the record a re-report repeats,
and a "place" column when some report names a place. A field's unit is named in its header or given
with every value; a qualifier has a column of its own or follows its value in parentheses. A caption
opens the table, comment lines (the instance's chatter among them) sit around and between its rows,
and the notices follow it as numbered footnotes. The examples below show two of the column orders a
style draws (COLUMN_ORDERS: id_first in the CSV, source_first in the markdown).

    csv       Table 2. Earthquakes reported in the 24 hours to 2026-02-25 06:00:00 UTC (edition 860-H).
              # One row per report, as received: preliminary rows and re-reports are kept.
              event ID,time (UTC),network,status,region,latitude,longitude,magnitude,mag_type,depth (km),note
              us644635vy,2026-02-24 13:03:48,us,reviewed,Tonga,-19.968,-174.191,5,mb,35.6,
              pt44035055,2026-02-24 20:36:34,pt,preliminary,Nevada,38.27,-118.314,1.2,ml,7.7,Dup of nn93493261
              # Station CI.BSN was offline from 13:00 to 14:39 UTC for maintenance; its readings are not included.

              Notes:
              [1],"RETRACTION (2026-02-24T21:00:00Z): us7000xyz1 was not a real earthquake (quarry blast); ignore..."
    markdown  | operator | reading ID | status   | time                      | sea area  | position          | ...
              | -------- | ---------- | -------- | ------------------------- | --------- | ----------------- | ...
              | ukmo     | by-12345   | reviewed | 2026-04-12T07:12:19+01:00 | North Sea | 56.512°N, 3.104°E | ...

              [^1]: Correction notice, issued 2026-04-12 09:00:00 UTC: water temperature for by-12345 revised
              from 11 °C to 12.3 °C (spar).

Every word comes from the glossary and every value is written with the exact codec of evalgen.units,
so the parser reads each report and notice back exactly through the glossary alone, and every
rendering is checked to do so (genres.base.check_round_trip). Notices are worded with the keywords
that every domain's reading rules name (RETRACTION, CORRECTION, revised) and with the domain's own
phrases. Earthquake digests carry the 0.2.0 record types (models.Report, and corrections that give a
new magnitude), which are written in their generic form (Domain.to_generic) and read back as 0.2.0
objects through Domain.from_generic (genres.base.generic_items and domain_items), as in every genre.
"""
from __future__ import annotations

import csv
import functools
import io
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from ..domains.base import REVISED_TO, Domain, Glossary, GlossaryField, RecordReport, TrapVocab
from ..models import Notice
from ..ops import DigestData
from ..units import Unit, decode, encode
from .base import (NUMBER, TIME_FORMATS, TIME_PATTERN, UTC_TIME_FORMATS, Genre, RenderContext, alternation,
                   check_round_trip, domain_items, format_coords, format_time, format_value, generic_items,
                   id_pattern, parse_coords, parse_time, read_value, value_pattern)
from .classic import UNIT_PRESETS, ZONES

# ------------------------------------------------------------------------------ the genre


class Table(Genre):
    name = "table"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Draw the look of one table: CSV (with its delimiter, quoting and fence) or markdown (compact
        or padded cells), how header labels are cased, where units and qualifiers go, how times and
        positions are written, the order of the columns and of the rows, how footnotes are numbered,
        and the caption. A variant avoids each choice of the original's that has an alternative."""

        def pick(key: str, options: Sequence[str] | Mapping[str, Any]) -> str:
            names = sorted(options)
            if avoid is not None and key in avoid:
                names = [n for n in names if n != avoid[key]] or names
            return rng.choice(names)

        style: dict[str, Any] = {"format": pick("format", FORMATS)}
        if style["format"] == "csv":
            style.update(delimiter=pick("delimiter", DELIMITERS), quoting=pick("quoting", QUOTING),
                         fence=pick("fence", FENCES))
        else:
            style["cells"] = pick("cells", CELL_LAYOUTS)
        style.update(case=pick("case", CASES), units=pick("units", UNIT_PLACES),
                     unit_mix=pick("unit_mix", UNIT_PRESETS), time=pick("time", TIME_STYLES))
        if domain.has_location:
            style["position"] = pick("position", POSITIONS)
        style.update(qualifiers=pick("qualifiers", QUALIFIER_PLACES), columns=pick("columns", COLUMN_ORDERS),
                     rows=pick("rows", ROW_ORDERS), notes=pick("notes", NOTE_MARKS),
                     caption=rng.randrange(len(CAPTIONS)))
        return style

    def render(self, ctx: RenderContext) -> str:
        """The table of the context's reports and notices, in the words of its glossary (the domain's
        own when it has none), checked to read back exactly."""
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _render(ctx, glossary)
        if "<digest>" in text or "</digest>" in text:
            raise ValueError("a table cannot print a digest tag, which readers would take for the digest's end")
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """A table digest has a header row of at least MIN_COLUMNS different labels, one of them
        "status" in some letter case: in markdown, a row that its separator row follows; in CSV, with
        any of the delimiters, a line that is no comment. No classic line splits into such cells: none
        has a cell that is just "status"."""
        lines = [line for line in digest_text.split("\n") if not line.strip().startswith("```")]
        for line, following in zip(lines, lines[1:] + [""]):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("|"):
                if _is_separator(following) and _header_like(_markdown_cells(text)):
                    return True
            elif any(_header_like(_csv_cells(line, delimiter)) for delimiter in DELIMITERS.values()):
                return True
        return False

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """Every report and notice of a table, read through the prompt's glossary (a prompt that prints
        none is read through the domain's own words, unless they are drawn per instance), as the
        domain's digests hold them: an earthquake table is read back to 0.2.0 Reports and Notices
        (Domain.from_generic, genres.base.domain_items)."""
        if glossary is None:
            if domain.has_surface and domain.surface is None:
                raise ValueError(f"the {domain.name} records of a table are read through the glossary its prompt "
                                 "prints, and this prompt prints none")
            glossary = domain.glossary()
        reports, notices = _read(digest_text, glossary)
        return DigestData(domain_items(domain, reports), domain_items(domain, notices))


TABLE = Table()
GENRE = TABLE  # what the registry looks up (see evalgen.genres)

# ------------------------------------------------------------------------ the look of a table
# choose_style picks one option of each; the keys of a mapping are the option names.

FORMATS = ("csv", "markdown")
DELIMITERS = {"comma": ",", "semicolon": ";", "tab": "\t"}  # CSV: the delimiters sniff and parse try
QUOTING = {"minimal": csv.QUOTE_MINIMAL, "all": csv.QUOTE_ALL}  # CSV
FENCES = ("bare", "fenced")  # CSV: as it is, or in a ```csv code block
CELL_LAYOUTS = ("compact", "padded")  # markdown: padded columns line up, numbers to the right
CASES: Mapping[str, Callable[[str], str]] = {  # how a header label is cased (a unit symbol never is)
    "plain": lambda label: label,
    "capitalized": lambda label: label[:1].upper() + label[1:],
    "upper": str.upper,
}
# Where a field's unit is given: after its label in the header, in parentheses or brackets ("depth (km)"
# over bare numbers), or with every value ("6.2 km"), each value in a unit of its own. Never in a row of
# its own, which a reader could take for a line that changes nothing.
UNIT_PLACES: Mapping[str, tuple[str, str] | None] = {"parentheses": ("(", ")"), "brackets": ("[", "]"), "cell": None}
BRACKETS = tuple(brackets for brackets in UNIT_PLACES.values() if brackets is not None)
TIME_STYLES: Mapping[str, tuple[str, ...]] = {
    "utc_column": (),  # every time in UTC, named once in the header: "time (UTC)" over "2026-04-12 06:12:19"
    "utc": ("iso", "iso_space", "rfc"),  # one UTC format for the whole column
    "local": ("iso_offset", "local24", "local12"),  # one format of local time, each row's offset stated
    "mixed": TIME_FORMATS,  # any format, row by row: as received
}
POSITIONS = ("columns", "hemi", "signed")  # "latitude" and "longitude" columns, or one "position" column
QUALIFIER_PLACES = ("column", "inline")  # a column headed by the qualifier's key, or "4.3 (mb)"
COLUMN_ORDERS: Mapping[str, tuple[str, ...]] = {  # "values": each field's value (and qualifier) column
    "id_first": ("id", "time", "source", "status", "group", "position", "values", "note", "place"),
    "time_first": ("time", "id", "group", "position", "values", "source", "status", "note", "place"),
    "source_first": ("source", "id", "status", "time", "group", "position", "values", "note", "place"),
}
ROW_ORDERS = ("by_id", "by_source", "by_time", "shuffled")  # by_source: a section of rows per source
NOTE_MARKS: Mapping[str, tuple[str, str]] = {  # how footnote n opens: its CSV marker cell, its markdown line
    "bracket": ("[{n}]", "[^{n}]:"),
    "number": ("{n})", "{n}."),
    "paren": ("({n})", "({n})"),
    "word": ("Note {n}", "Note {n}:"),
}
CAPTIONS = (
    "Table {number}. {Plural} reported in the {window} to {compiled} (edition {edition}).",
    "{Plural}: every report received in the {window} ending {compiled}. Version {edition}.",
    "Export of {noun} reports for the {window} up to {compiled}; batch {edition}.",
)
REMARKS = (  # comment lines that are true of every table
    "One row per report, as received: preliminary rows and re-reports are kept.",
    "Rows are reproduced as each {source_label} filed them, without merging.",
    "Exported automatically; not checked by an editor.",
    "An empty cell means that nothing was given.",
)
CSV_SECTION_HEADS = ("# -- {source} --", "# {source}", "# rows from {source}")
MARKDOWN_SECTION_HEADS = ("### {source}", "**{source}**", "Rows from {source}:")
NOTE_TITLES = ("Notes:", "Footnotes:", "Notes to the table:")
# Notices open with a keyword that every domain's reading rules name; each keyword has one wording.
RETRACTION_KEYWORDS = ("RETRACTION", "Retraction notice", "Retraction")
CORRECTION_KEYWORDS = ("CORRECTION", "Correction notice", "Correction")
NOTICE_HEADS = (" ({when}): ", ", issued {when}: ")  # between a notice's keyword and its body
MIN_COLUMNS = 6  # every table has an ID, source, status, time, group and at least one value column
NUMERIC = ("value", "lat", "lon")  # the kinds of column whose padded cells are aligned to the right

# --------------------------------------------------------------------------------- columns


@dataclass(frozen=True)
class _Column:
    """One column of a table: what its cells hold (`kind`: id, source, status, time, utc_time, group,
    position, lat, lon, value, qualifier, note or place), the field of a value or qualifier column, and
    the unit of a value column whose cells are bare numbers in the unit its header names."""

    kind: str
    field: GlossaryField | None = None
    unit: Unit | None = None


def _header(column: _Column, glossary: Glossary, case: str, brackets: tuple[str, str] | None = None) -> str:
    """A column's label in a header row of `case`, with its unit in `brackets` when they are given."""
    cased = CASES[case]
    if column.kind == "value":
        label = cased(column.field.label)
        if column.unit is None or brackets is None:
            return label
        return f"{label} {brackets[0]}{column.unit.symbol}{brackets[1]}"
    words = {"id": glossary.id_label, "source": glossary.source_label, "status": "status", "time": "time",
             "utc_time": "time", "group": glossary.group_label, "position": "position", "lat": "latitude",
             "lon": "longitude", "note": "note", "place": "place"}
    if column.kind == "qualifier":
        return cased(column.field.qualifier)
    return cased(words[column.kind]) + (" (UTC)" if column.kind == "utc_time" else "")


@functools.lru_cache(maxsize=64)
def _headers(glossary: Glossary) -> tuple[Mapping[str, _Column], ...]:
    """For each case style, every header label a table of these records can print and the column it
    names. A label that two columns would share is left out: no table prints it (see _check_headers)."""
    columns = [_Column(kind) for kind in ("id", "source", "status", "time", "utc_time", "group", "note", "place")]
    if glossary.located:
        columns += [_Column("position"), _Column("lat"), _Column("lon")]
    for f in glossary.fields:
        columns += [_Column("value", f)] + [_Column("value", f, unit) for unit in f.units]
        if f.qualifier is not None:
            columns.append(_Column("qualifier", f))
    lookups = []
    for case in CASES:
        found: dict[str, list[_Column]] = {}
        for column in columns:
            for brackets in BRACKETS:
                named = found.setdefault(_header(column, glossary, case, brackets), [])
                if column not in named:
                    named.append(column)
        lookups.append({label: named[0] for label, named in found.items() if len(named) == 1})
    return tuple(lookups)


def _header_columns(cells: Sequence[str], glossary: Glossary) -> list[_Column] | None:
    """The columns a header row names, read in the first case style that names them all; None when the
    cells are no header of these records (one cell is no label, a column is named twice, or a column
    every table has is missing)."""
    for lookup in _headers(glossary):
        columns = [lookup[cell] for cell in cells if cell in lookup]
        if len(columns) != len(cells) or len(set(columns)) != len(columns):
            continue
        kinds = [c.kind for c in columns]
        fields = [c.field.key for c in columns if c.kind == "value"]
        complete = (all(kinds.count(kind) == 1 for kind in ("id", "source", "status", "group"))
                    and kinds.count("time") + kinds.count("utc_time") == 1
                    and sorted(fields) == sorted(f.key for f in glossary.fields))
        if glossary.located:
            complete = complete and (kinds.count("position"), kinds.count("lat"), kinds.count("lon")) in (
                (1, 0, 0), (0, 1, 1))
        if complete:
            return columns
    return None


def _columns(glossary: Glossary, style: Mapping[str, Any], units: Mapping[str, Unit | None],
             with_place: bool) -> list[_Column]:
    """The columns of a table in `style`: the style's column order, each field's value column (with
    the unit its header names, `units`, when it names one) and its qualifier column (when qualifiers
    have one), and a place column when some report names a place."""
    columns: list[_Column] = []
    for slot in COLUMN_ORDERS[style["columns"]]:
        if slot == "time":
            columns.append(_Column("utc_time" if style["time"] == "utc_column" else "time"))
        elif slot == "position":
            if glossary.located:
                columns += ([_Column("lat"), _Column("lon")] if style["position"] == "columns"
                            else [_Column("position")])
        elif slot == "values":
            for f in glossary.fields:
                columns.append(_Column("value", f, units[f.key]))
                if f.qualifier is not None and style["qualifiers"] == "column":
                    columns.append(_Column("qualifier", f))
        elif slot != "place" or with_place:
            columns.append(_Column(slot))
    return columns


def _check_headers(columns: Sequence[_Column], labels: Sequence[str], glossary: Glossary, case: str) -> None:
    """Refuse a table whose header a reader could not tell apart: a label two columns of these records
    would share (a field labelled "status", a qualifier keyed like a field's label)."""
    lookup = _headers(glossary)[list(CASES).index(case)]
    for column, label in zip(columns, labels):
        if lookup.get(label) != column:
            raise ValueError(f"the table header {label!r} does not name one column of the {glossary.plural} alone")


# ---------------------------------------------------------------------------- writing a table

_Row = tuple[RecordReport, list[str]]  # a report and its cells
_Section = tuple[Optional[str], list[list[str]]]  # the source a section is for (None: every source), its rows


@dataclass(frozen=True)
class _Document:
    """What a table digest is made of, before it is laid out as CSV or markdown."""

    caption: str
    remarks: Sequence[str]  # comment lines that are true of every table (REMARKS)
    columns: Sequence[_Column]
    labels: Sequence[str]  # the header row
    sections: Sequence[_Section]
    chatter: Sequence[str]
    title: str  # the title of the footnotes
    notes: Sequence[str]  # the footnotes' texts, in their order


def _render(ctx: RenderContext, glossary: Glossary) -> str:
    """The table of `ctx.reports` and its footnotes, as `ctx.style` draws them, written from their
    generic form (genres.base.generic_items)."""
    rng, style, vocab = ctx.rng, ctx.style, ctx.domain.trap_vocab
    reports = generic_items(ctx.domain, ctx.reports)
    brackets = UNIT_PLACES[style["units"]]
    units = {f.key: None if brackets is None else _draw_unit(rng, f, style["unit_mix"]) for f in glossary.fields}
    columns = _columns(glossary, style, units, any(r.place is not None for r in reports))
    labels = [_header(column, glossary, style["case"], brackets) for column in columns]
    _check_headers(columns, labels, glossary, style["case"])
    when = _time_writer(rng, style, glossary, reports, ctx.domain)
    rows = [(r, [_cell(r, column, rng, style, glossary, vocab, when) for column in columns]) for r in reports]
    sections = _sections(rows, style["rows"], rng)

    notices = generic_items(ctx.domain, ctx.notices)
    rng.shuffle(notices)
    notes = [_notice_text(n, rng, glossary, vocab, style["unit_mix"]) for n in notices]
    caption = CAPTIONS[style["caption"]].format(
        number=rng.randint(1, 9), Plural=glossary.plural[:1].upper() + glossary.plural[1:], noun=glossary.noun,
        window=_window(ctx.window_hours), compiled=format_time(ctx.compiled_at, rng.choice(UTC_TIME_FORMATS)),
        edition=f"{rng.randrange(100, 1000)}-{rng.choice('ABCDEFGHJK')}")
    remarks = [REMARKS[i].format(source_label=glossary.source_label)
               for i in sorted(rng.sample(range(len(REMARKS)), rng.randint(0, 2)))]
    document = _Document(caption, remarks, columns, labels, sections, list(ctx.chatter), rng.choice(NOTE_TITLES),
                         notes)
    return (_csv_document if style["format"] == "csv" else _markdown_document)(document, style, rng)


def _cell(r: RecordReport, column: _Column, rng: random.Random, style: Mapping[str, Any], glossary: Glossary,
          vocab: TrapVocab, when: Callable[[RecordReport], str]) -> str:
    """What a report's row shows in one column (an empty string when it gives nothing there)."""
    kind = column.kind
    if kind in ("time", "utc_time"):
        return when(r)
    if kind == "position":
        return format_coords(r.lat, r.lon, style["position"], glossary.coord_decimals)
    if kind in ("lat", "lon"):
        return encode(r.lat if kind == "lat" else r.lon, glossary.coord_decimals)
    if kind == "value":
        return _value_cell(r, column, rng, style)
    if kind == "qualifier":
        return _qualifier(column.field, r) or ""
    if kind == "note":
        return "" if r.same_as is None else f"{rng.choice(vocab.duplicate_notes)} {r.same_as}"
    if kind == "place":
        return r.place or ""
    return {"id": r.ref, "source": r.source, "status": r.status, "group": r.group}[kind]


def _value_cell(r: RecordReport, column: _Column, rng: random.Random, style: Mapping[str, Any]) -> str:
    """A value as a bare number in the unit its column's header names, or with a unit of its own; with
    its qualifier in parentheses when qualifiers follow their values."""
    f = column.field
    value = r.values[f.key]
    if column.unit is not None:
        text = f.encode(value, column.unit)
    else:
        text = format_value(f, value, _draw_unit(rng, f, style["unit_mix"]))
    qualifier = _qualifier(f, r) if style["qualifiers"] == "inline" else None
    return text if qualifier is None else f"{text} ({qualifier})"


def _qualifier(f: GlossaryField, r: RecordReport) -> str | None:
    """The qualifier a report gives with a field's value (None: the field has none, or the report gives none)."""
    return None if f.qualifier is None else r.qualifiers.get(f.qualifier)


def _draw_unit(rng: random.Random, f: GlossaryField, preset: str) -> Unit | None:
    """A unit of `f` (None for a field without units), its stored unit weighted as `preset` says."""
    if not f.units:
        return None
    stored = UNIT_PRESETS[preset]
    if stored is None or len(f.units) == 1:
        return rng.choice(f.units)
    others = (1.0 - stored) / (len(f.units) - 1)
    return rng.choices(f.units, weights=[stored if u.symbol == f.canonical_unit else others for u in f.units])[0]


def _bare_utc(ts: int) -> str:
    """A UTC time under a "time (UTC)" header: "2026-04-12 06:12:19" (iso_space without its zone)."""
    return format_time(ts, "iso_space")[:-len(" UTC")]


_BARE_UTC = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _time_writer(rng: random.Random, style: Mapping[str, Any], glossary: Glossary,
                 reports: Sequence[RecordReport], domain: Domain) -> Callable[[RecordReport], str]:
    """How the time column writes a report's time, as the style's TIME_STYLES entry says, its offset
    always stated. Local times are those the domain gives a report's position (Domain.utc_offset: the
    zone in force at a weather station, else the nominal offset of the longitude, which is no real time
    zone), or those of its source (one zone each) for records without one."""
    if style["time"] == "utc_column":
        return lambda r: _bare_utc(r.time)
    formats = TIME_STYLES[style["time"]]
    if style["time"] == "utc":
        utc_format = rng.choice(formats)
        return lambda r: format_time(r.time, utc_format)
    if glossary.located:
        def offset(r: RecordReport) -> int:
            return domain.utc_offset(r)
    else:
        zones = {source: rng.randint(*ZONES) for source in sorted({r.source for r in reports})}

        def offset(r: RecordReport) -> int:
            return zones[r.source]
    if style["time"] == "mixed":
        return lambda r: format_time(r.time, rng.choice(formats), offset(r))
    column_format = rng.choice(formats)
    return lambda r: format_time(r.time, column_format, offset(r))


def _sections(rows: Sequence[_Row], order: str, rng: random.Random) -> list[_Section]:
    """The rows in `order`, as sections: one per source, in a shuffled order and each sorted by time,
    for by_source; else a single section for every source (as also for a table without rows, which
    still prints its header)."""
    if order == "by_source" and rows:
        sources = sorted({r.source for r, _ in rows})
        rng.shuffle(sources)
        return [(source, [cells for r, cells in sorted((row for row in rows if row[0].source == source),
                                                       key=lambda row: (row[0].time, row[0].ref))])
                for source in sources]
    ordered = list(rows)
    if order == "shuffled":
        rng.shuffle(ordered)
    elif order == "by_time":
        ordered.sort(key=lambda row: (row[0].time, row[0].ref, row[0].source))
    else:
        ordered.sort(key=lambda row: (row[0].ref, row[0].time, row[0].source))
    return [(None, [cells for _, cells in ordered])]


def _notice_text(n: Notice, rng: random.Random, glossary: Glossary, vocab: TrapVocab, unit_mix: str) -> str:
    """A notice as a footnote says it: its keyword, when it was issued, and the wording of its keyword,
    with a retraction phrase of the domain's trap vocabulary or the reading rules' own words; a
    correction gives both values in one drawn unit, its new value after "revised to", or after "revised
    from" and the old value, which the rules quote (the Correction wording gave it after "now", which they
    do not)."""
    when = format_time(n.time, rng.choice(UTC_TIME_FORMATS))
    head = rng.choice(NOTICE_HEADS).format(when=when)
    form = rng.randrange(3)
    if n.kind == "retraction":
        phrase = rng.choice(vocab.retraction_notices)
        reason = "" if n.reason is None else f" ({n.reason})"
        return RETRACTION_KEYWORDS[form] + head + (
            f"{n.target} was not a real {glossary.noun}{reason}; ignore that report.",
            f"report {n.target} {phrase}{reason}. Disregard it.",
            f"{n.target} {phrase}{reason}.")[form]
    key, new, old, qualifier = n.field, n.new_value, n.old_value, n.qualifier
    field = next((f for f in glossary.fields if f.key == key), None)
    if n.kind != "correction" or field is None or new is None or old is None:
        raise ValueError(f"a {n.kind} notice of {n.target} that names no field of the glossary, or no new and old "
                         "value, cannot be printed")
    unit = _draw_unit(rng, field, unit_mix)
    new_text, old_text = format_value(field, new, unit), format_value(field, old, unit)
    q = "" if qualifier is None else f" ({qualifier})"
    return CORRECTION_KEYWORDS[form] + head + (
        f"the {field.label} of {n.target} is revised to {new_text}{q}; it was first given as {old_text}.",
        f"{field.label} for {n.target} revised from {old_text} to {new_text}{q}.",
        f"{n.target} {field.label} {REVISED_TO} {new_text}{q} (was {old_text}).")[form]


def _window(hours: int) -> str:
    if hours > 24 and hours % 24 == 0:
        return f"{hours // 24} days"
    return f"{hours} hour{'' if hours == 1 else 's'}"


def _csv_document(document: _Document, style: Mapping[str, Any], rng: random.Random) -> str:
    """A CSV table: a title row, comment lines ("# "), the header, the rows (a comment opens each
    source's section), then a blank line and the footnote rows (marker, text). The chatter goes into
    comment lines anywhere from the header to the last row."""
    delimiter, quoting = DELIMITERS[style["delimiter"]], QUOTING[style["quoting"]]

    def row(cells: Sequence[str]) -> str:
        out = io.StringIO()
        csv.writer(out, delimiter=delimiter, quoting=quoting, lineterminator="\n").writerow(cells)
        return out.getvalue()[:-1]

    lines = [row([document.caption])] + [f"# {remark}" for remark in document.remarks]
    top = len(lines)
    lines.append(row(document.labels))
    for source, rows in document.sections:
        if source is not None:
            lines.append(rng.choice(CSV_SECTION_HEADS).format(source=source))
        lines += [row(cells) for cells in rows]
    for text in document.chatter:
        lines.insert(rng.randrange(top, len(lines) + 1), f"# {text}")
    if document.notes:
        marker = NOTE_MARKS[style["notes"]][0]
        lines += ["", row([document.title])] + [row([marker.format(n=n), text])
                                                for n, text in enumerate(document.notes, start=1)]
    if style["fence"] == "fenced":
        lines = ["```csv", *lines, "```"]
    return "\n".join(lines)


def _markdown_document(document: _Document, style: Mapping[str, Any], rng: random.Random) -> str:
    """A markdown table: the caption, comments (HTML comments), one table per section (each source's
    under a heading of its own) with the chatter as paragraphs around them, then the footnotes."""
    blocks = [[document.caption]] + [[f"<!-- {remark} -->"] for remark in document.remarks]
    top = len(blocks)
    numeric = [column.kind in NUMERIC for column in document.columns]
    for source, rows in document.sections:
        heading = [] if source is None else [rng.choice(MARKDOWN_SECTION_HEADS).format(source=source), ""]
        blocks.append(heading + _markdown_table(document.labels, rows, numeric, style["cells"] == "padded"))
    for text in document.chatter:
        blocks.insert(rng.randrange(top, len(blocks) + 1), [text])
    if document.notes:
        opening = NOTE_MARKS[style["notes"]][1]
        blocks.append([document.title] + [f"{opening.format(n=n)} {text}"
                                          for n, text in enumerate(document.notes, start=1)])
    return "\n\n".join("\n".join(block) for block in blocks)


def _markdown_table(labels: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool],
                    padded: bool) -> list[str]:
    """The lines of one markdown table: header, separator and rows, with every backslash and pipe in a
    cell escaped; padded cells line up, numbers to the right."""
    body = [[_markdown_escape(cell) for cell in cells] for cells in [labels, *rows]]
    if not padded:
        lines = ["| " + " | ".join(cells) + " |" for cells in body]
        return [lines[0], "|" + "|".join("---" for _ in labels) + "|", *lines[1:]]
    widths = [max(3, *(len(cells[i]) for cells in body)) for i in range(len(labels))]
    lines = ["| " + " | ".join(cell.rjust(width) if right else cell.ljust(width)
                               for cell, width, right in zip(cells, widths, numeric)) + " |" for cells in body]
    separator = "| " + " | ".join("-" * (width - 1) + ":" if right else "-" * width
                                  for width, right in zip(widths, numeric)) + " |"
    return [lines[0], separator, *lines[1:]]


def _markdown_escape(cell: str) -> str:
    return cell.replace("\\", "\\\\").replace("|", "\\|")


# ---------------------------------------------------------------------------- reading a table


def _markdown_cells(line: str) -> list[str]:
    """The cells of a markdown table row: split at every pipe no backslash escapes, without the outer
    pipes, unescaped and stripped."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            current.append(char if char in "\\|" else "\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current) + ("\\" if escaped else "")
    if tail.strip() or not cells:  # no closing pipe: the last cell runs to the end of the line
        cells.append(tail.strip())
    return cells


_SEPARATOR_CELL = re.compile(r":?-{3,}:?")


def _is_separator(line: str) -> bool:
    """A markdown table's separator row: "|---|---|", "| ---: | :--- |"."""
    text = line.strip()
    return text.startswith("|") and all(_SEPARATOR_CELL.fullmatch(cell) for cell in _markdown_cells(text))


def _csv_cells(line: str, delimiter: str) -> list[str]:
    """The cells of one CSV line ([] for a line the CSV reader refuses)."""
    try:
        return next(csv.reader([line], delimiter=delimiter), [])
    except csv.Error:
        return []


def _header_like(cells: Sequence[str]) -> bool:
    """Cells that could be a table's header row: at least MIN_COLUMNS different, non-empty labels, one
    of them "status"."""
    names = [cell.strip().casefold() for cell in cells]
    return len(names) >= MIN_COLUMNS and "status" in names and all(names) and len(set(names)) == len(names)


_CSV_MARK = re.compile(r"\[\d+\]|\d+\)|\(\d+\)|Note \d+")  # a footnote's marker cell
_MARKDOWN_NOTE = re.compile(r"(?:\[\^\d+\]:|\d+\.|\(\d+\)|Note \d+:) (?P<text>.+)")  # a footnote line


def _read(text: str, glossary: Glossary) -> tuple[list[RecordReport], list[Notice]]:
    """Every report of a table (as RecordReports) and every notice of its footnotes; lines that are
    neither change nothing."""
    lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
    grammar = _grammar(glossary)
    markdown = any(line.strip().startswith("|") and _is_separator(following)
                   for line, following in zip(lines, lines[1:]))
    return (_read_markdown if markdown else _read_csv)(lines, grammar)


def _read_csv(lines: Sequence[str], grammar: _Grammar) -> tuple[list[RecordReport], list[Notice]]:
    """A CSV table: the first line that is a header of these records (with any delimiter) fixes the
    delimiter and the columns; then each line with as many cells is a row, and each line of a marker
    and a text is a footnote. Comments and other lines are passed over."""
    reports: list[RecordReport] = []
    notices: list[Notice] = []
    delimiter, columns = "", None
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        if columns is None:
            for delimiter in DELIMITERS.values():
                columns = _header_columns(_csv_cells(line, delimiter), grammar.glossary)
                if columns is not None:
                    break
            continue
        cells = _csv_cells(line, delimiter)
        if len(cells) == len(columns):
            report = grammar.report(cells, columns)
            reports += [report] if report is not None else []
        elif len(cells) == 2 and _CSV_MARK.fullmatch(cells[0]):
            notice = grammar.notice(cells[1])
            notices += [notice] if notice is not None else []
    return reports, notices


def _read_markdown(lines: Sequence[str], grammar: _Grammar) -> tuple[list[RecordReport], list[Notice]]:
    """Markdown tables (one per section), each a header row of these records, its separator row and its
    rows, and footnote lines; other lines are passed over."""
    reports: list[RecordReport] = []
    notices: list[Notice] = []
    stripped = [line.strip() for line in lines]
    i = 0
    while i < len(stripped):
        line = stripped[i]
        if line.startswith("|") and i + 1 < len(stripped) and _is_separator(stripped[i + 1]):
            columns = _header_columns(_markdown_cells(line), grammar.glossary)
            i += 2
            while i < len(stripped) and stripped[i].startswith("|"):
                cells = _markdown_cells(stripped[i])
                if columns is not None and len(cells) == len(columns):
                    report = grammar.report(cells, columns)
                    reports += [report] if report is not None else []
                i += 1
            continue
        note = _MARKDOWN_NOTE.fullmatch(line)
        if note is not None:
            notice = grammar.notice(note["text"])
            notices += [notice] if notice is not None else []
        i += 1
    return reports, notices


@dataclass(frozen=True)
class _Grammar:
    """How the cells and footnotes of one glossary's tables read: its ID pattern, the note of a
    re-report, the cell patterns of each field's values (a bare number, or a number and its unit,
    either followed by a qualifier in parentheses), and the wordings of each notice keyword."""

    glossary: Glossary
    record_id: re.Pattern[str]
    note: re.Pattern[str]
    bare: Mapping[str, re.Pattern[str]]  # field key -> a bare number (in its column's unit, or of no unit)
    with_unit: Mapping[str, re.Pattern[str]]  # field key -> a number and its unit
    head: re.Pattern[str]  # a notice's keyword and time, then its body
    bodies: Mapping[str, tuple[tuple[re.Pattern[str], GlossaryField | None], ...]]  # keyword -> (wording, field)

    def report(self, cells: Sequence[str], columns: Sequence[_Column]) -> RecordReport | None:
        """The report a row gives; None when a cell does not read as its column's (no ID in the ID
        column, a status the glossary does not name, a value that is no number, ...)."""
        glossary = self.glossary
        parts: dict[str, Any] = {"lat": None, "lon": None, "place": None, "same_as": None}
        values: dict[str, float] = {}
        qualifiers: dict[str, str] = {}
        inline = {c.field.key for c in columns if c.kind == "value" and c.field.qualifier is not None} - {
            c.field.key for c in columns if c.kind == "qualifier"}
        for column, cell in zip(columns, cells):
            kind = column.kind
            if kind == "id":
                if not self.record_id.fullmatch(cell):
                    return None
                parts["ref"] = cell
            elif kind in ("source", "group"):
                if not cell:
                    return None
                parts[kind] = cell
            elif kind == "status":
                if cell not in glossary.statuses:
                    return None
                parts["status"] = cell
            elif kind in ("time", "utc_time"):
                when = parse_time(cell) if kind == "time" else (
                    parse_time(f"{cell} UTC") if _BARE_UTC.fullmatch(cell) else None)
                if when is None:
                    return None
                parts["time"] = when
            elif kind == "position":
                position = parse_coords(cell, glossary.coord_decimals)
                if position is None:
                    return None
                parts["lat"], parts["lon"] = position
            elif kind in ("lat", "lon"):
                if not re.fullmatch(NUMBER, cell):
                    return None
                parts[kind] = decode(cell, glossary.coord_decimals)
            elif kind == "value":
                read = self.value(cell, column, column.field.key in inline)
                if read is None:
                    return None
                value, qualifier = read
                values[column.field.key] = value
                if qualifier is not None:
                    qualifiers[column.field.qualifier] = qualifier
            elif kind == "qualifier":
                if cell:
                    qualifiers[column.field.qualifier] = cell
            elif kind == "note":
                if cell:
                    note = self.note.fullmatch(cell)
                    if note is None:
                        return None
                    parts["same_as"] = note["same"]
            elif kind == "place":
                parts["place"] = cell or None
        return RecordReport(values=values, qualifiers=qualifiers, **parts)

    def value(self, cell: str, column: _Column, inline: bool) -> tuple[float, str | None] | None:
        """A value cell's stored value, and the qualifier that follows it in parentheses (only where
        qualifiers follow their values): a bare number in the column's unit (or of a field without
        units), or a number with its own unit."""
        f = column.field
        bare = column.unit is not None or not f.units
        match = (self.bare if bare else self.with_unit)[f.key].fullmatch(cell)
        qualifier = None if match is None else match.groupdict().get("q")
        if match is None or (qualifier is not None and not inline):
            return None
        return (f.decode(match["v"], column.unit) if bare else read_value(f, match, "v")), qualifier

    def notice(self, text: str) -> Notice | None:
        """The notice a footnote's text gives (None when it is no notice of these records)."""
        head = self.head.fullmatch(text)
        when = None if head is None else parse_time(head["time"] or head["issued"])
        if when is None:
            return None
        for pattern, field in self.bodies[head["keyword"]]:
            body = pattern.fullmatch(head["body"])
            if body is None:
                continue
            if field is None:
                return Notice(kind="retraction", target=body["ref"], time=when, reason=body["reason"])
            return Notice(kind="correction", target=body["ref"], time=when, field=field.key,
                          new_value=read_value(field, body, "new"), old_value=read_value(field, body, "old"),
                          qualifier=body.groupdict().get("qualifier"))
        return None


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    """The patterns of one glossary's tables (see _Grammar), each notice keyword with one wording."""
    record_id = id_pattern(glossary)
    bare: dict[str, re.Pattern[str]] = {}
    with_unit: dict[str, re.Pattern[str]] = {}
    for f in glossary.fields:
        qualifier = "" if f.qualifier is None else r"(?: \((?P<q>[^()]*)\))?"
        bare[f.key] = re.compile(rf"(?P<v>{NUMBER}){qualifier}")
        with_unit[f.key] = re.compile(value_pattern(f, "v") + qualifier)
    reason = r"(?: \((?P<reason>.+)\))?"
    noun = re.escape(glossary.noun)
    retractions = (rf"(?P<ref>{record_id}) was not a real {noun}{reason}; ignore that report\.",
                   rf"report (?P<ref>{record_id}) (?P<phrase>.+?){reason}\. Disregard it\.",
                   rf"(?P<ref>{record_id}) (?P<phrase>.+?){reason}\.")
    bodies: dict[str, list[tuple[re.Pattern[str], GlossaryField | None]]] = {
        keyword: [(re.compile(pattern), None)] for keyword, pattern in zip(RETRACTION_KEYWORDS, retractions)}
    for f in glossary.fields:
        label, new, old = re.escape(f.label), value_pattern(f, "new"), value_pattern(f, "old")
        q = "" if f.qualifier is None else r"(?: \((?P<qualifier>[^()]*)\))?"
        corrections = (rf"the {label} of (?P<ref>{record_id}) is revised to {new}{q}; it was first given as {old}\.",
                       rf"{label} for (?P<ref>{record_id}) revised from {old} to {new}{q}\.",
                       # "now": as tables printed it before, so that a run generated then is still read
                       rf"(?P<ref>{record_id}) {label} (?:{REVISED_TO}|now) {new}{q} \(was {old}\)\.")
        for keyword, pattern in zip(CORRECTION_KEYWORDS, corrections):
            bodies.setdefault(keyword, []).append((re.compile(pattern), f))
    keywords = alternation(RETRACTION_KEYWORDS + CORRECTION_KEYWORDS)
    head = re.compile(rf"(?P<keyword>{keywords})(?: \((?P<time>{TIME_PATTERN})\)|, issued (?P<issued>{TIME_PATTERN})): "
                      r"(?P<body>.+)")
    return _Grammar(glossary=glossary, record_id=re.compile(record_id),
                    note=re.compile(rf"(?P<phrase>.+) (?P<same>{record_id})"), bare=bare, with_unit=with_unit,
                    head=head, bodies={keyword: tuple(patterns) for keyword, patterns in bodies.items()})
