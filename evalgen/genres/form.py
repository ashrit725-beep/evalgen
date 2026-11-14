"""form: field and incident report forms, one form per report, for the records of every domain.

A report is one filled-in form: a heading with the form's code and serial number, then labelled
fields: the record's ID, its source, its status as a pair of checkboxes, its time, group, position
and values (each value's qualifier in a field of its own), a cross-reference (one of the domain's
duplicate notes and the ID it names) and a location note, either left blank when the report has none,
and the form's own noise: when it was sent, forwarded or relayed (a message time, never the record's,
and never after a notice about it), who prepared it, and blank fields for a checker or a signature. A
notice is an amendment form that names the record it amends and the serial number of the form it
amends (when one form alone carries that ID): a CORRECTION gives the field, the entry as first filed
and the corrected entry; a RETRACTION gives the reason and withdraws the report in a phrase of the
domain's trap vocabulary. Headings, section titles and chatter lines stand between the forms, never
inside one.

A form is drawn in one of four frames, per form, from the rendering's preset:

    boxed     ┌─────────────────────────────────────────────────────────────────────────┐
              │ FIELD REPORT FORM FR-1                                       No. 004417 │
              ├─────────────────────────────────────────────────────────────────────────┤
              │ Reading ID: by-33617                       Operator: emsn               │
              │ Status: [x] preliminary  [ ] reviewed                                   │
              │ Date and time of reading: 2026-03-19 12:08:49 UTC                       │
              │ Sea area: Baltic Sea                       Position: 59.049°N, 20.183°E │
              │ Water temperature: 2.2 °C                  Probe: hull                  │
              │ Wave height: 0.57 m                                                     │
              │ Cross-reference: same reading as by-60852  Location note: ________      │
              │ Form sent: 2026-03-19T13:36:15Z                                         │
              └─────────────────────────────────────────────────────────────────────────┘
              (a field left blank in a row of two shows a line, as here, even where the rendering
              leaves blanks empty: an empty one would put the next label where its value belongs)
    ruled     "INCIDENT REPORT — FORM IR-3          Serial no. 26-04417" between rules of "=" and "-",
              one field a line, values aligned, each unit in its label ("Wave height (m):     0.57")
    numbered  "REPORT FORM OB-11 · no. 4417", numbered fields ("2. Reading ID: by-33617") and
              "--- end of form 4417 ---"
    dotted    "OBSERVATION REPORT FORM 40-C   Ref. no. 04440/26   Page 1 of 1", dot leaders
              ("Duplicate note .............. duplicate of by-50562"), units in the labels, latitude and
              longitude on lines of their own

Every word of a form is the glossary's (record noun, ID label, source and group labels, field labels,
units and qualifier keys, status words) or the genre's own. Values are written with the exact codec of
evalgen.units, times to the second (in UTC or with their UTC offset stated) and positions exactly, so
the parser reads every form back through the glossary alone, and every rendering is checked to read
back to exactly its reports and notices (genres.base.check_round_trip). Earthquake domains keep their
0.2.0 record types (models.Report, and the magnitude correction of models.Notice): their forms are
written from their generic form (Domain.to_generic), as any other record's are, and read back into
those types through Domain.from_generic (genres.base.generic_items and domain_items).
"""
from __future__ import annotations

import functools
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..domains.base import Domain, Glossary, GlossaryField, RecordReport
from ..domains.earthquakes import Earthquakes
from ..models import Notice
from ..ops import DigestData
from ..units import Unit, decode, encode, find_unit
from .base import (NUMBER, TIME_FORMATS, UTC_TIME_FORMATS, Genre, RenderContext, alternation, check_round_trip,
                   domain_items, format_coords, format_time, format_value, generic_items, id_pattern, parse_coords,
                   parse_time, read_value, value_pattern)

# ------------------------------------------------------------------------------ the genre


class Form(Genre):
    name = "form"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Draw the look of one rendering: the preset its forms' frames are drawn from, the presets of
        units, times and (for records with a position) coordinates, the order of the forms, the checkbox
        glyphs, how a blank field is left, which status box comes first, where the amendments go and
        the header. A variant in this genre avoids the original's value of every preset but the last two."""

        def pick(key: str, options: Sequence[str] | Mapping[str, Any]) -> str:
            names = sorted(options)
            if avoid is not None and key in avoid:
                names = [n for n in names if n != avoid[key]] or names
            return rng.choice(names)

        style = {"forms": pick("forms", FORM_PRESETS), "units": pick("units", UNIT_PRESETS),
                 "time": pick("time", TIME_PRESETS)}
        if domain.has_location:
            style["coords"] = pick("coords", COORD_PRESETS)
        style.update(layout=pick("layout", LAYOUTS), boxes=pick("boxes", CHECKBOXES), blank=pick("blank", BLANKS),
                     statuses=pick("statuses", STATUS_ORDERS), notices=rng.choice(NOTICE_PLACEMENTS),
                     header=rng.randrange(len(HEADERS)))
        return style

    def render(self, ctx: RenderContext) -> str:
        """The forms of the context's reports and notices, in the words of its glossary (the domain's own
        when it has none), checked to read back exactly."""
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _render(ctx, glossary)
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """A form digest opens with one of its two-line headers, whose title is the records' plural."""
        lines = digest_text.lstrip().split("\n", 2)
        return len(lines) >= 2 and any(first.fullmatch(lines[0]) is not None and second.fullmatch(lines[1]) is not None
                                       for first, second in _HEADER_LINES)

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """Every report and notice, read through the prompt's glossary. Earthquakes, whose words never
        change, may be read through the domain's own glossary when a prompt prints none; every other
        domain's records are read through the one its prompt prints."""
        if glossary is None:
            if not isinstance(domain, Earthquakes):
                raise ValueError(f"a digest of {domain.name} records is read through the glossary its prompt prints, "
                                 "and this prompt prints none")
            glossary = domain.glossary()
        data = _parse(digest_text, glossary)
        return DigestData(domain_items(domain, data.reports), domain_items(domain, data.notices))


FORM = Form()
GENRE = FORM  # what the registry looks up (see evalgen.genres)

# ---------------------------------------------------------------------- the look of a rendering
# A style picks one preset per dimension; a preset is a weighted mix that each form draws from.

FRAME_NAMES = ("boxed", "ruled", "numbered", "dotted")
FORM_PRESETS = {
    "boxed": {"boxed": 0.55, "ruled": 0.15, "numbered": 0.15, "dotted": 0.15},
    "ruled": {"boxed": 0.15, "ruled": 0.55, "numbered": 0.15, "dotted": 0.15},
    "numbered": {"boxed": 0.15, "ruled": 0.15, "numbered": 0.55, "dotted": 0.15},
    "dotted": {"boxed": 0.15, "ruled": 0.15, "numbered": 0.15, "dotted": 0.55},
    "mixed": {"boxed": 0.25, "ruled": 0.25, "numbered": 0.25, "dotted": 0.25},
}
TIME_PRESETS = {
    "utc_heavy": {"iso": 0.3, "iso_space": 0.2, "rfc": 0.2, "prose_utc": 0.2, "iso_offset": 0.04, "local12": 0.03,
                  "local24": 0.03},
    "local_heavy": {"iso": 0.1, "iso_space": 0.1, "rfc": 0.1, "prose_utc": 0.1, "iso_offset": 0.2, "local12": 0.2,
                    "local24": 0.2},
    "mixed": {fmt: 1.0 for fmt in TIME_FORMATS},
}
COORD_PRESETS = {"hemi": {"hemi": 0.7, "signed": 0.3}, "signed": {"hemi": 0.3, "signed": 0.7},
                 "mixed": {"hemi": 0.5, "signed": 0.5}}
# The weight of a field's stored unit in each preset; the others share the rest (None: all alike).
UNIT_PRESETS: dict[str, float | None] = {"mostly_stored": 0.7, "mostly_other": 0.15, "mixed": None}
LAYOUTS = ("by_source", "by_time", "shuffled")
NOTICE_PLACEMENTS = ("interleaved", "end")
CHECKBOXES = {"brackets": ("[x]", "[ ]"), "capitals": ("[X]", "[ ]"), "ballot": ("☒", "☐"), "ticks": ("☑", "☐")}
BLANKS = {"empty": "", "line": "________", "dash": "—", "word": "(blank)"}  # how a field left blank looks
# How a blank looks in a row of two fields (the boxed frame's) when the rendering leaves blanks empty:
# an empty field there would put the row's second label where the first label's value belongs
# ("Cross-reference:   Location note: Dogger Bank"), so a row of two never shows an empty field.
PAIRED_BLANK = BLANKS["line"]
STATUS_ORDERS = ("settling_first", "settling_last")  # which status box comes first
ZONES = (-12, 14)  # the UTC offsets, in whole hours, of sources whose records have no position

# Two-line headers; the title is the records' plural.
HEADERS = (
    ("FIELD REPORT FORMS — {TITLE}",
     "Batch {batch}: the forms filed in the {window} up to {compiled}, reproduced as filed. Blank fields were "
     "left blank by the sender."),
    ("Incident and field report forms: {title}",
     "Form batch {batch}, compiled {compiled}; covers the {window} before compilation. Each form is one report, "
     "or one amendment of a report filed before it."),
    ("REPORT FORM BATCH {batch} — {TITLE}",
     "Compiled {compiled} from the forms each {source_label} filed in the {window} before compilation."),
)
SECTION_HEADS = ("Forms filed by {source_label} {source}:", "{Source_label} {source}, forms received:",
                 "== Forms from {source} ==")
AMENDMENTS_HEAD = "Amendments to earlier forms:"
PREPARERS = ("duty officer", "field technician", "shift supervisor", "data clerk", "on-call analyst")
INITIALS = "ABCDEFGHJKLMNPRSTW"


def _header_line_pattern(template: str) -> re.Pattern[str]:
    """A header line: its title (the records' plural) as text that opens with a letter or digit, and its
    other placeholders as wildcards."""
    pieces = re.split(r"(\{\w+\})", template)
    return re.compile("".join(r"[^\W_].*?" if piece in ("{TITLE}", "{title}") else ".+?" if piece.startswith("{")
                              else re.escape(piece) for piece in pieces))


_HEADER_LINES = tuple((_header_line_pattern(first), _header_line_pattern(second)) for first, second in HEADERS)


def _draw(rng: random.Random, table: Mapping[str, float]) -> str:
    return rng.choices(list(table), weights=list(table.values()), k=1)[0]


def _draw_unit(rng: random.Random, field: GlossaryField, preset: str) -> Unit | None:
    """The unit one value of `field` is written in (None for a field without units)."""
    if not field.units:
        return None
    stored = UNIT_PRESETS[preset]
    if stored is None or len(field.units) == 1:
        return rng.choice(field.units)
    others = (1.0 - stored) / (len(field.units) - 1)
    weights = [stored if unit.symbol == field.canonical_unit else others for unit in field.units]
    return rng.choices(field.units, weights=weights, k=1)[0]


def _window(hours: int) -> str:
    if hours > 24 and hours % 24 == 0:
        return f"{hours // 24} days"
    return f"{hours} hour{'' if hours == 1 else 's'}"


def _cap(text: str) -> str:
    """`text` with its first letter upper-cased, as the labels of a form begin; a first word that is not
    all lower-case letters ("pH", "km/s") is left as it is."""
    first = text.split(" ", 1)[0]
    return text[:1].upper() + text[1:] if first.isalpha() and first.islower() else text


def _clear(text: str, glossary: Glossary) -> bool:
    """Whether a word of the genre's own (a form code, a serial or batch number) holds nothing a reader
    could take for a record ID."""
    return re.search(id_pattern(glossary), text) is None


# ------------------------------------------------------------------------------- the frames


@dataclass(frozen=True)
class _Slot:
    """One labelled field of a form: what it holds (`key`) and, for a value, a qualifier and the fields
    of a correction, the glossary field it is about (an index into the glossary's fields)."""

    key: str
    field: int | None = None


Rows = tuple[tuple[_Slot, ...], ...]  # a form's fields, row by row (one or two to a row)

FIELD_SLOTS = ("amended", "old", "new", "new_qualifier", "action")  # amendment slots about the field corrected
POSITION_SLOTS = ("position", "latitude", "longitude")
NUMBER_SLOTS = ("value", "old", "new")  # slots that hold a field's value


class _Frame:
    """One kind of form: the headings of its report and amendment forms (with "{code}", its code drawn
    from `codes`, or nothing, and "{kind}", CORRECTION or RETRACTION), how it numbers them, the labels
    and the order of its fields, and how its lines are laid out and read back."""

    name = ""
    title = ""
    amendment = ""
    codes: tuple[str, ...] = ()
    amendment_codes: tuple[str, ...] = ()
    labels: Mapping[str, str] = {}  # slot key -> label, worded with the glossary's words (_label)
    report: tuple[tuple[str, ...], ...] = ()  # rows of slot keys; ("fields",) stands for every field's rows
    retraction: tuple[tuple[str, ...], ...] = ()
    correction: tuple[tuple[str, ...], ...] = ()
    pairs = False  # a value and its qualifier share a row
    units_in_labels = False  # "Wave height (ft): 6.36" rather than "Wave height: 6.36 ft"
    heading_tail = ""  # a regex for what follows the title in a heading line

    def number(self, serial: int, year: int) -> str:
        """A form's serial number as the frame prints it."""
        raise NotImplementedError

    def lay_out(self, title: str, number: str, cells: Sequence[Sequence[tuple[str, str]]]) -> list[str]:
        """The lines of one form: its heading (the title and the serial number) and its cells, (label,
        value) pairs row by row, a blank value already written as the rendering leaves blanks."""
        raise NotImplementedError

    def unwrap(self, block: Sequence[str]) -> tuple[str, list[str]] | None:
        """A block of lines as this frame lays a form out: its heading and the text of each row (without
        borders or numbering); None for a block that is not one of its forms."""
        raise NotImplementedError

    def cell_pattern(self, label: str, j: int, last: bool) -> str:
        """A regex for a cell of a row: `label` (a regex), and its value in group v{j}."""
        return f"{label}:(?: (?P<v{j}>.*{'' if last else '?'}))?"

    @functools.cached_property
    def _headings(self) -> tuple[re.Pattern[str], re.Pattern[str]]:
        def pattern(template: str) -> re.Pattern[str]:
            pieces = re.split(r"(\{\w+\})", template)
            title = "".join(r"(?: \S+)?" if piece == "{code}" else r"(?P<kind>CORRECTION|RETRACTION)"
                            if piece == "{kind}" else re.escape(piece) for piece in pieces)
            return re.compile(title + self.heading_tail)

        return pattern(self.title), pattern(self.amendment)

    def kind(self, heading: str) -> str | None:
        """What a form with this heading is: "report", "correction" or "retraction" (None: no heading)."""
        report, amendment = self._headings
        if report.fullmatch(heading):
            return "report"
        match = amendment.fullmatch(heading)
        return None if match is None else match["kind"].lower()


def _cell(label: str, value: str) -> str:
    return f"{label}: {value}" if value else f"{label}:"


class _Boxed(_Frame):
    name = "boxed"
    title = "FIELD REPORT FORM{code}"
    amendment = "AMENDMENT FORM{code} — {kind}"
    codes = ("FR-1", "FR-2", "FR-7")
    amendment_codes = ("AF-1", "AF-3")
    labels = {
        "id": "{Id_label}", "source": "{Source_label}", "status": "Status", "time": "Date and time of {noun}",
        "group": "{Group_label}", "position": "Position", "value": "{Label}", "qualifier": "{qualifier}",
        "xref": "Cross-reference", "place": "Location note", "sent": "Form sent", "prepared": "Prepared by",
        # The amendment's own issue time shares a row with the number of the form it amends, so its label says
        # whose time it is: a bare "Issued" there could be read as when the original form was issued.
        "target": "Refers to {id_label}", "original": "Original form no.", "issued": "Amendment issued",
        "amended": "Field amended", "old": "Entry as first filed", "new": "Corrected entry",
        "new_qualifier": "Corrected {qualifier}", "reason": "Reason", "action": "Action taken",
    }
    report = (("id", "source"), ("status",), ("time",), ("group", "position"), ("fields",), ("xref", "place"),
              ("sent",))
    retraction = (("target",), ("original", "issued"), ("reason",), ("action",), ("prepared",))
    correction = (("target",), ("original", "issued"), ("amended",), ("old", "new"), ("new_qualifier",),
                  ("reason",), ("action",), ("prepared",))
    pairs = True
    heading_tail = r" {2,}No\. \S+"

    def number(self, serial: int, year: int) -> str:
        return f"{serial:06d}"

    def lay_out(self, title: str, number: str, cells: Sequence[Sequence[tuple[str, str]]]) -> list[str]:
        """The box: a row of two fields in two columns, each field of it shown even when left blank
        (PAIRED_BLANK), so that no label stands where the field before it is written."""
        texts = [[_cell(label, value or (PAIRED_BLANK if len(row) == 2 else "")) for label, value in row]
                 for row in cells]
        left = max((len(row[0]) for row in texts if len(row) == 2), default=0)
        rows = [row[0] if len(row) == 1 else f"{row[0].ljust(left)}  {row[1]}" for row in texts]
        tag = f"No. {number}"
        inner = max([len(title) + 2 + len(tag)] + [len(row) for row in rows])
        heading = title + " " * (inner - len(title) - len(tag)) + tag
        return (["┌" + "─" * (inner + 2) + "┐", f"│ {heading} │", "├" + "─" * (inner + 2) + "┤"]
                + [f"│ {row.ljust(inner)} │" for row in rows] + ["└" + "─" * (inner + 2) + "┘"])

    def unwrap(self, block: Sequence[str]) -> tuple[str, list[str]] | None:
        if len(block) < 4 or not (_BOX_TOP.fullmatch(block[0]) and _BOX_RULE.fullmatch(block[2])
                                  and _BOX_BOTTOM.fullmatch(block[-1])):
            return None
        width = len(block[0])
        inner = []
        for line in (block[1], *block[3:-1]):
            if len(line) != width or not (line.startswith("│ ") and line.endswith(" │")):
                return None
            inner.append(line[2:-2].rstrip(" "))
        if len(block[2]) != width or len(block[-1]) != width:
            return None
        return inner[0], inner[1:]


_BOX_TOP, _BOX_RULE, _BOX_BOTTOM = (re.compile(f"{a}─+{b}") for a, b in (("┌", "┐"), ("├", "┤"), ("└", "┘")))


class _Ruled(_Frame):
    name = "ruled"
    title = "INCIDENT REPORT — FORM{code}"
    amendment = "AMENDMENT — FORM{code} — {kind}"
    codes = ("IR-3", "IR-7", "IR-12")
    amendment_codes = ("IR-3A", "IR-7A")
    labels = {
        "id": "{Id_label}", "source": "Reporting {source_label}", "status": "Report status", "time": "Time of {noun}",
        "group": "{Group_label}", "position": "Position (lat, lon)", "value": "{Label}",
        "qualifier": "{Label}, {qualifier}", "xref": "Note", "place": "Location note", "sent": "Sent to desk",
        "checked": "Checked by", "target": "Report amended ({id_label})", "original": "Original form no.",
        "issued": "Date and time issued", "amended": "Field corrected", "old": "As first filed",
        "new": "Corrected value", "new_qualifier": "Corrected {qualifier}", "reason": "Reason given",
        "action": "Action",
    }
    report = (("source",), ("id",), ("time",), ("status",), ("group",), ("position",), ("fields",), ("xref",),
              ("place",), ("sent",), ("checked",))
    retraction = (("target",), ("original",), ("issued",), ("reason",), ("action",))
    correction = (("target",), ("original",), ("issued",), ("amended",), ("old",), ("new",), ("new_qualifier",),
                  ("reason",), ("action",))
    units_in_labels = True
    heading_tail = r" {2,}Serial no\. \S+"

    def number(self, serial: int, year: int) -> str:
        return f"{year:02d}-{serial:05d}"

    def lay_out(self, title: str, number: str, cells: Sequence[Sequence[tuple[str, str]]]) -> list[str]:
        column = max(len(label) for (label, _), in cells) + 3
        rows = [f"{label}:".ljust(column) + value if value else f"{label}:" for (label, value), in cells]
        tag = f"Serial no. {number}"
        width = max([len(title) + 2 + len(tag)] + [len(row) for row in rows])
        return (["=" * width, title + " " * (width - len(title) - len(tag)) + tag, "-" * width] + rows
                + ["=" * width])

    def unwrap(self, block: Sequence[str]) -> tuple[str, list[str]] | None:
        if len(block) < 4 or not (_RULE_EQUALS.fullmatch(block[0]) and _RULE_DASHES.fullmatch(block[2])
                                  and _RULE_EQUALS.fullmatch(block[-1])):
            return None
        return block[1], list(block[3:-1])

    def cell_pattern(self, label: str, j: int, last: bool) -> str:
        return f"{label}:(?: +(?P<v{j}>.*))?"


_RULE_EQUALS, _RULE_DASHES = re.compile(r"={10,}"), re.compile(r"-{10,}")


class _Numbered(_Frame):
    name = "numbered"
    title = "REPORT FORM{code}"
    amendment = "AMENDMENT FORM{code} · {kind}"
    codes = ("OB-3", "OB-11", "R-20")
    amendment_codes = ("A-20", "A-21")
    labels = {
        "id": "{Id_label}", "source": "{Source_label}", "status": "Status (tick one)", "time": "{Noun} date and time",
        "group": "{Group_label}", "latitude": "Latitude", "longitude": "Longitude", "value": "{Label}",
        "qualifier": "{qualifier} for {label}", "place": "Site or location note", "xref": "Related report",
        "sent": "Relayed to desk", "prepared": "Prepared by", "target": "{Id_label} of the report amended",
        "issued": "Issued", "original": "Original form", "amended": "Field", "old": "Entry as filed",
        "new": "Corrected entry", "new_qualifier": "Corrected {qualifier}", "reason": "Reason",
        "action": "Action taken",
    }
    report = (("time",), ("id",), ("source",), ("status",), ("group",), ("latitude",), ("longitude",), ("fields",),
              ("place",), ("xref",), ("sent",), ("prepared",))
    retraction = (("target",), ("issued",), ("original",), ("reason",), ("action",), ("prepared",))
    correction = (("target",), ("issued",), ("original",), ("amended",), ("old",), ("new",), ("new_qualifier",),
                  ("reason",), ("action",), ("prepared",))
    heading_tail = r" · no\. \S+"

    def number(self, serial: int, year: int) -> str:
        return f"{serial}"

    def lay_out(self, title: str, number: str, cells: Sequence[Sequence[tuple[str, str]]]) -> list[str]:
        rows = [f"{i}. {_cell(label, value)}" for i, ((label, value),) in enumerate(cells, start=1)]
        return [f"{title} · no. {number}"] + rows + [f"--- end of form {number} ---"]

    def unwrap(self, block: Sequence[str]) -> tuple[str, list[str]] | None:
        if len(block) < 3 or not _END_OF_FORM.fullmatch(block[-1]):
            return None
        rows = []
        for i, line in enumerate(block[1:-1], start=1):
            if not line.startswith(f"{i}. "):
                return None
            rows.append(line[len(f"{i}. "):])
        return block[0], rows


_END_OF_FORM = re.compile(r"--- end of form \S+ ---")


class _Dotted(_Frame):
    name = "dotted"
    title = "OBSERVATION REPORT FORM{code}"
    amendment = "AMENDMENT FORM{code} ({kind})"
    codes = ("12-B", "40-C", "7-F")
    amendment_codes = ("12-X", "40-X")
    labels = {
        "id": "{Id_label}", "status": "Status", "source": "Filed by ({source_label})", "group": "{Group_label}",
        "time": "Date/time of {noun}", "latitude": "Latitude", "longitude": "Longitude", "value": "{Label}",
        "qualifier": "{Label}, {qualifier}", "xref": "Duplicate note", "place": "Location note", "sent": "Forwarded",
        "signature": "Signature", "target": "Amends {id_label}", "original": "Original form no.", "issued": "Issued",
        "amended": "Field amended", "old": "Value as first filed", "new": "Corrected value",
        "new_qualifier": "Corrected {qualifier}", "reason": "Reason", "action": "Action",
    }
    report = (("id",), ("status",), ("source",), ("group",), ("time",), ("latitude",), ("longitude",), ("fields",),
              ("xref",), ("place",), ("sent",), ("signature",))
    retraction = (("target",), ("original",), ("issued",), ("reason",), ("action",), ("signature",))
    correction = (("target",), ("original",), ("issued",), ("amended",), ("old",), ("new",), ("new_qualifier",),
                  ("reason",), ("action",), ("signature",))
    units_in_labels = True
    heading_tail = r"   Ref\. no\. \S+   Page 1 of 1"

    def number(self, serial: int, year: int) -> str:
        return f"{serial:05d}/{year:02d}"

    def lay_out(self, title: str, number: str, cells: Sequence[Sequence[tuple[str, str]]]) -> list[str]:
        column = max(len(label) for (label, _), in cells) + 4
        rows = [f"{label} {'.' * (column - len(label))}" + (f" {value}" if value else "") for (label, value), in cells]
        return [f"{title}   Ref. no. {number}   Page 1 of 1"] + rows

    def unwrap(self, block: Sequence[str]) -> tuple[str, list[str]] | None:
        return (block[0], list(block[1:])) if len(block) >= 2 else None

    def cell_pattern(self, label: str, j: int, last: bool) -> str:
        return rf"{label} \.{{3,}}(?: (?P<v{j}>.*))?"


FRAMES = (_Boxed(), _Ruled(), _Numbered(), _Dotted())  # in the order of FRAME_NAMES
FRAMES_BY_NAME = {frame.name: frame for frame in FRAMES}


def _slots(frame: _Frame, rows: Sequence[Sequence[str]], glossary: Glossary, field: int | None = None) -> Rows:
    """A form's slots, row by row: `rows` of the frame with every field's value and qualifier rows in
    place of ("fields",), positions only for records that have one, and the slots of a correction about
    its `field` (no corrected qualifier for a field without one)."""
    out: list[tuple[_Slot, ...]] = []
    for row in rows:
        if tuple(row) == ("fields",):
            for i, f in enumerate(glossary.fields):
                value, qualifier = _Slot("value", i), _Slot("qualifier", i)
                if f.qualifier is None:
                    out.append((value,))
                elif frame.pairs:
                    out.append((value, qualifier))
                else:
                    out += [(value,), (qualifier,)]
            continue
        slots = tuple(_Slot(key, field if key in FIELD_SLOTS else None) for key in row
                      if (glossary.located or key not in POSITION_SLOTS)
                      and not (key == "new_qualifier" and (field is None or glossary.fields[field].qualifier is None)))
        if slots:
            out.append(slots)
    return tuple(out)


def _words(glossary: Glossary) -> dict[str, str]:
    """The glossary's words that labels are worded with, as they stand in a sentence and capitalised."""
    words = {"noun": glossary.noun, "id_label": glossary.id_label, "source_label": glossary.source_label,
             "group_label": glossary.group_label}
    return {**words, **{name[:1].upper() + name[1:]: _cap(word) for name, word in words.items()}}


def _label(frame: _Frame, slot: _Slot, glossary: Glossary) -> str:
    """The label of a slot in a frame, in the glossary's words (a value's unit is not part of it)."""
    words = _words(glossary)
    if slot.field is not None:
        f = glossary.fields[slot.field]
        words.update(label=f.label, Label=_cap(f.label), qualifier=f.qualifier or "")
    return _cap(frame.labels[slot.key].format(**words))


# ----------------------------------------------------------------------------- writing forms


@dataclass(frozen=True)
class _Look:
    """What one form draws: its frame and, for a report, its time and how its position and values are
    written, its cross-reference note, when it was sent and by whom it was prepared; for an amendment,
    when it was issued, the unit of its values and the phrase that withdraws a report."""

    frame: _Frame
    when: str = ""
    coords: str = "hemi"
    units: tuple[Unit | None, ...] = ()
    note: str = ""
    sent: str = ""
    preparer: str = ""
    unit: Unit | None = None
    phrase: str = ""


@dataclass
class _Draft:
    """A form before its serial number is known: a report (a RecordReport) or an amendment (a Notice
    under generic names), and what it drew."""

    kind: str  # "report", "correction" or "retraction"
    item: RecordReport | Notice
    look: _Look
    number: str = ""
    filed: int = 0  # when the form was filed: a report's sent time, an amendment's issue time (_number)

    @property
    def ref(self) -> str:
        """The record ID the form carries, or the one it amends."""
        return self.item.ref if isinstance(self.item, RecordReport) else self.item.target


def _render(ctx: RenderContext, glossary: Glossary) -> str:
    """The forms of a context's reports, notices and chatter, as `ctx.style` draws them: drafted in
    planting order, arranged, numbered in filing order and written out under the header. Reports and
    notices are written in their generic form (genres.base.generic_items)."""
    rng, style = ctx.rng, ctx.style
    reports = _report_drafts(ctx, glossary)
    amendments = [_amendment_draft(notice, ctx, glossary) for notice in generic_items(ctx.domain, ctx.notices)]
    body = _arrange(reports, amendments, ctx.chatter, rng, style, glossary)
    codes = {frame.name: (_code(rng, frame.codes, glossary), _code(rng, frame.amendment_codes, glossary))
             for frame in FRAMES}
    _number(body, rng, ctx.compiled_at, glossary)
    forms: dict[str, list[str]] = {}  # the serial numbers of the report forms that carry each ID
    for draft in reports:
        forms.setdefault(draft.ref, []).append(draft.number)
    blank = BLANKS[style["blank"]]
    blocks = []
    for item in body:
        if isinstance(item, str):
            blocks.append(item)
            continue
        frame = item.look.frame
        report_code, amendment_code = codes[frame.name]
        if item.kind == "report":
            statuses = glossary.statuses if style["statuses"] == "settling_first" else glossary.statuses[::-1]
            title = frame.title.format(code=report_code)
            cells = _report_cells(item, glossary, blank, CHECKBOXES[style["boxes"]], statuses)
        else:  # an amendment names the form it amends when one form alone carries the ID
            original = forms.get(item.ref, [])
            title = frame.amendment.format(code=amendment_code, kind=item.kind.upper())
            cells = _amendment_cells(item, glossary, blank, original[0] if len(original) == 1 else blank)
        blocks.append("\n".join(frame.lay_out(title, item.number, cells)))
    return "\n\n".join([_header(ctx, glossary), *blocks])


def _report_drafts(ctx: RenderContext, glossary: Glossary) -> list[_Draft]:
    """A draft of each report's form, in planting order: its frame, its time and message time (in the
    local time of its position, or of its source when records have none), how its position and values
    are written, its cross-reference note and who prepared it."""
    rng, style, vocab = ctx.rng, ctx.style, ctx.domain.trap_vocab
    entries: list[RecordReport] = generic_items(ctx.domain, ctx.reports)
    zones = {} if glossary.located else {s: rng.randint(*ZONES) for s in sorted({e.source for e in entries})}
    # A form is sent before the compilation and before any notice about its record. Every form of the
    # rendering shares one longest delay, shorter than the shortest time from any record to a notice about
    # it, so that no form's delay tells whether a notice names its record: capping the forms that notices
    # name alone gave the planted reports (every retracted and every corrected one) shorter delays.
    notified: dict[str, int] = {}
    for notice in ctx.notices:
        notified[notice.target] = min(notified.get(notice.target, ctx.compiled_at), notice.time)
    gaps = [notified[ref] - e.time for e in entries for ref in {e.ref, e.same_as} if ref in notified]
    longest = min([MAX_SENT_DELAY_S, *(gap - 1 for gap in gaps if gap > 1)])
    drafts = []
    for e in entries:
        frame = FRAMES_BY_NAME[_draw(rng, FORM_PRESETS[style["forms"]])]
        offset = ctx.domain.utc_offset(e) if glossary.located else zones[e.source]
        own = min(notified.get(ref, ctx.compiled_at) for ref in (e.ref, e.same_as or e.ref))
        sent = _sent_time(rng, e.time, min(own, e.time + longest + 1))
        look = _Look(frame, when=format_time(e.time, _draw(rng, TIME_PRESETS[style["time"]]), offset),
                     coords=_draw(rng, COORD_PRESETS[style["coords"]]) if glossary.located else "hemi",
                     units=tuple(_draw_unit(rng, f, style["units"]) for f in glossary.fields),
                     note=rng.choice(vocab.duplicate_notes) if e.same_as else "",
                     sent=format_time(sent, _draw(rng, TIME_PRESETS[style["time"]]), offset), preparer=_preparer(rng))
        drafts.append(_Draft("report", e, look, filed=sent))
    return drafts


def _amendment_draft(notice: Notice, ctx: RenderContext, glossary: Glossary) -> _Draft:
    """A draft of a notice's amendment form: its frame, when it was issued (in UTC), the unit of a
    correction's entries, the phrase of a retraction, and who prepared it."""
    rng, style = ctx.rng, ctx.style
    frame = FRAMES_BY_NAME[_draw(rng, FORM_PRESETS[style["forms"]])]
    field = _corrected_field(notice, glossary)
    look = _Look(frame, when=format_time(notice.time, rng.choice(UTC_TIME_FORMATS)),
                 unit=None if field is None else _draw_unit(rng, glossary.fields[field], style["units"]),
                 phrase=rng.choice(ctx.domain.trap_vocab.retraction_notices) if notice.kind == "retraction" else "",
                 preparer=_preparer(rng))
    return _Draft(notice.kind, notice, look, filed=notice.time)


def _preparer(rng: random.Random) -> str:
    return f"{rng.choice(PREPARERS)} ({rng.choice(INITIALS)}.{rng.choice(INITIALS)}.)"


def _number(body: Sequence[_Draft | str], rng: random.Random, compiled_at: int, glossary: Glossary) -> None:
    """Give every form its serial number in each frame's style ("—" for one that could be taken for a
    record ID), rising in the order the forms were filed (_Draft.filed; forms filed at one time in the
    order they are printed), not the order they are printed in: an amendment, issued after the report
    it amends was sent, always has a higher number than the form it names as the original, wherever
    the batch prints it."""
    year = datetime.fromtimestamp(compiled_at, tz=timezone.utc).year % 100
    serial = rng.randrange(1000, 9000)
    for item in sorted((item for item in body if isinstance(item, _Draft)), key=lambda draft: draft.filed):
        serial += rng.randint(1, 3)
        number = item.look.frame.number(serial, year)
        item.number = number if _clear(number, glossary) else "—"


def _header(ctx: RenderContext, glossary: Glossary) -> str:
    """The two header lines of the style's header, with a batch number and the compilation time."""
    rng = ctx.rng
    first, second = HEADERS[ctx.style["header"]]
    batch = f"{rng.randrange(100, 1000)}-{rng.choice('ABCDEFGHJK')}"
    words = dict(TITLE=glossary.plural.upper(), title=glossary.plural, batch=batch if _clear(batch, glossary) else "—",
                 window=_window(ctx.window_hours), compiled=format_time(ctx.compiled_at, rng.choice(UTC_TIME_FORMATS)),
                 source_label=glossary.source_label)
    return f"{first.format(**words)}\n{second.format(**words)}"


MAX_SENT_DELAY_S = 90 * 60 + 59  # a form is sent at most an hour and a half after its record's time


def _sent_time(rng: random.Random, when: int, deadline: int) -> int:
    """When a report's form was sent: drawn alike over one minute to MAX_SENT_DELAY_S after its record's
    time, and before `deadline` (the compilation, or the rendering's shared limit, see _report_drafts):
    from one second when the deadline leaves less than a minute, at the record's time when it leaves no
    second."""
    room = min(MAX_SENT_DELAY_S, deadline - when - 1)
    if room < 1:
        return when
    return when + rng.randint(60 if room >= 60 else 1, room)


def _code(rng: random.Random, codes: Sequence[str], glossary: Glossary) -> str:
    """A form code drawn for the rendering, with the space before it; nothing when every code of the
    frame could be taken for a record ID."""
    clear = [code for code in codes if _clear(code, glossary)]
    return f" {rng.choice(clear)}" if clear else ""


def _arrange(reports: list[_Draft], amendments: list[_Draft], chatter: Sequence[str], rng: random.Random,
             style: Mapping[str, Any], glossary: Glossary) -> list[_Draft | str]:
    """The forms and the lines between them, in the order the style draws: report forms by time, by
    source (under a heading each) or shuffled; amendments among them or after them; chatter anywhere
    between two forms."""
    body: list[_Draft | str] = []
    if style["layout"] == "by_time":
        body = sorted(reports, key=lambda d: (d.item.time, d.item.ref, d.item.source))
    elif style["layout"] == "by_source":
        sources = sorted({d.item.source for d in reports})
        rng.shuffle(sources)
        for source in sources:
            body.append(rng.choice(SECTION_HEADS).format(source=source, source_label=glossary.source_label,
                                                         Source_label=_cap(glossary.source_label)))
            group = [d for d in reports if d.item.source == source]
            rng.shuffle(group)
            body += group
    else:
        body = list(reports)
        rng.shuffle(body)
    if style["notices"] == "interleaved":
        for draft in amendments:
            body.insert(rng.randrange(len(body) + 1), draft)
    elif amendments:
        rest = list(amendments)
        rng.shuffle(rest)
        body += [AMENDMENTS_HEAD] + rest
    for text in chatter:
        body.insert(rng.randrange(len(body) + 1), text)
    return body


def _report_cells(draft: _Draft, glossary: Glossary, blank: str, marks: tuple[str, str],
                  statuses: Sequence[str]) -> list[list[tuple[str, str]]]:
    """The (label, value) cells of a report's form, row by row."""
    e, look = draft.item, draft.look
    frame, decimals = look.frame, glossary.coord_decimals
    texts = {"id": e.ref, "source": e.source, "group": e.group, "time": look.when, "sent": look.sent,
             "prepared": look.preparer,
             "status": "  ".join(f"{marks[0] if status == e.status else marks[1]} {status}" for status in statuses),
             "xref": blank if e.same_as is None else f"{look.note} {e.same_as}",
             "place": blank if e.place is None else e.place}
    if glossary.located:
        texts.update(position=format_coords(e.lat, e.lon, look.coords, decimals),
                     latitude=_angle(e.lat, "NS", look.coords, decimals),
                     longitude=_angle(e.lon, "EW", look.coords, decimals))

    def cell(slot: _Slot) -> tuple[str, str]:
        label = _label(frame, slot, glossary)
        if slot.key not in ("value", "qualifier"):
            return label, texts.get(slot.key, blank)  # a checker, a signature: left blank
        f = glossary.fields[slot.field]
        if slot.key == "qualifier":
            qualifier = e.qualifiers.get(f.qualifier)
            return label, blank if qualifier is None else qualifier
        unit = look.units[slot.field]
        if frame.units_in_labels and unit is not None:
            return f"{label} ({unit.symbol})", f.encode(e.values[f.key], unit)
        return label, format_value(f, e.values[f.key], unit)

    return [[cell(slot) for slot in row] for row in _slots(frame, frame.report, glossary)]


def _amendment_cells(draft: _Draft, glossary: Glossary, blank: str, original: str) -> list[list[tuple[str, str]]]:
    """The (label, value) cells of an amendment's form, row by row: a retraction, or a correction of
    one field, whose entries as first filed and as corrected are in one unit."""
    n, look = draft.item, draft.look
    frame = look.frame
    if n.kind == "retraction":
        rows, field, action = frame.retraction, None, f"report {n.target} {look.phrase}. Disregard it."
    else:
        rows, field = frame.correction, _corrected_field(n, glossary)
        if n.kind != "correction" or field is None or n.new_value is None or n.old_value is None:
            raise ValueError(f"a {n.kind} notice of {n.target} that names no field of the glossary, or no new and old "
                             "value, cannot be printed")
        action = f"the {glossary.fields[field].label} was revised; use the corrected entry."

    texts = {"target": n.target, "original": original, "issued": look.when, "action": action,
             "amended": "" if field is None else glossary.fields[field].label,
             "new_qualifier": blank if n.qualifier is None else n.qualifier,
             "reason": blank if n.reason is None else n.reason, "prepared": look.preparer}

    def cell(slot: _Slot) -> tuple[str, str]:
        label = _label(frame, slot, glossary)
        if slot.key not in ("old", "new"):
            return label, texts.get(slot.key, blank)  # a signature: left blank
        f = glossary.fields[field]
        value = n.new_value if slot.key == "new" else n.old_value
        if frame.units_in_labels and look.unit is not None:
            return f"{label} ({look.unit.symbol})", f.encode(value, look.unit)
        return label, format_value(f, value, look.unit)

    return [[cell(slot) for slot in row] for row in _slots(frame, rows, glossary, field)]


def _angle(value: float, hemispheres: str, fmt: str, decimals: int) -> str:
    """A latitude ("NS") or a longitude ("EW") on a line of its own: "56.512°N", or signed, "-3.104"."""
    if fmt == "signed":
        return encode(value, decimals)
    return f"{encode(abs(value), decimals)}°{hemispheres[0] if value >= 0 else hemispheres[1]}"


def _corrected_field(notice: Notice, glossary: Glossary) -> int | None:
    """The index of the glossary field a correction revises (None: a retraction, or a field it lacks)."""
    return next((i for i, f in enumerate(glossary.fields) if f.key == notice.field), None)


# ----------------------------------------------------------------------------- reading forms


@dataclass(frozen=True)
class _FormGrammar:
    """The rows of one kind of form in one frame, each a regex and its slots, and the field a
    correction revises."""

    rows: tuple[tuple[re.Pattern[str], tuple[_Slot, ...]], ...]
    field: int | None = None


@dataclass(frozen=True)
class _Grammar:
    """What reads the forms of one glossary: each frame's report form, retraction form and correction
    form of every field, and the patterns of an ID, a cross-reference, a status and each field's value."""

    frames: tuple[tuple[_Frame, _FormGrammar, _FormGrammar, tuple[_FormGrammar, ...]], ...]
    record_id: re.Pattern[str]
    xref: re.Pattern[str]
    withdrawn: re.Pattern[str]
    status: re.Pattern[str]
    values: tuple[re.Pattern[str], ...]


CHECKED = ("[x]", "[X]", "☒", "☑")
UNCHECKED = ("[ ]", "☐")
_BLANK_TEXTS = frozenset(BLANKS.values())


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    record_id = id_pattern(glossary)

    def form(frame: _Frame, rows: Rows, field: int | None = None) -> _FormGrammar:
        compiled = []
        for row in rows:
            cells = []
            for j, slot in enumerate(row):
                label = re.escape(_label(frame, slot, glossary))
                if slot.key in NUMBER_SLOTS:
                    f = glossary.fields[slot.field if slot.key == "value" else field]
                    if frame.units_in_labels and f.units:
                        label += rf" \((?P<u{j}>{alternation(u.symbol for u in f.units)})\)"
                cells.append(frame.cell_pattern(label, j, j == len(row) - 1))
            compiled.append((re.compile(" {2,}".join(cells)), row))
        return _FormGrammar(tuple(compiled), field)

    frames = tuple(
        (frame, form(frame, _slots(frame, frame.report, glossary)),
         form(frame, _slots(frame, frame.retraction, glossary)),
         tuple(form(frame, _slots(frame, frame.correction, glossary, i), i) for i in range(len(glossary.fields))))
        for frame in FRAMES)
    marks = alternation(CHECKED + UNCHECKED)
    status = alternation(glossary.statuses)
    return _Grammar(
        frames=frames, record_id=re.compile(record_id),
        xref=re.compile(rf"(?P<note>.+) (?P<same>{record_id})"),
        withdrawn=re.compile(rf"report (?P<ref>{record_id}) (?P<phrase>.+)\. Disregard it\."),
        status=re.compile(rf"(?P<m1>{marks}) (?P<s1>{status})  (?P<m2>{marks}) (?P<s2>{status})"),
        values=tuple(re.compile(value_pattern(f, "v")) for f in glossary.fields))


def _parse(text: str, glossary: Glossary) -> DigestData:
    """Every report and notice of a form digest, in their generic form; blocks that are not forms change
    nothing."""
    grammar = _grammar(glossary)
    reports: list[RecordReport] = []
    notices: list[Notice] = []
    for block in _blocks(text):
        item = _read_form(block, grammar, glossary)
        if isinstance(item, RecordReport):
            reports.append(item)
        elif isinstance(item, Notice):
            notices.append(item)
    return DigestData(reports, notices)


def _blocks(text: str) -> list[list[str]]:
    """The runs of non-blank lines of a digest: its header, forms, headings and chatter lines."""
    blocks: list[list[str]] = [[]]
    for raw in text.split("\n"):
        line = raw.rstrip()
        if line:
            blocks[-1].append(line)
        elif blocks[-1]:
            blocks.append([])
    return [block for block in blocks if block]


def _read_form(block: Sequence[str], grammar: _Grammar, glossary: Glossary) -> RecordReport | Notice | None:
    """The report or notice a block of lines is the form of; None for a block that is no form."""
    for frame, report, retraction, corrections in grammar.frames:
        unwrapped = frame.unwrap(block)
        if unwrapped is None:
            continue
        heading, rows = unwrapped
        kind = frame.kind(heading)
        candidates = ((report,) if kind == "report" else (retraction,) if kind == "retraction"
                      else corrections if kind == "correction" else ())
        for form in candidates:
            values = _read_rows(form, rows, grammar, glossary)
            if values is not None:
                item = _report(values, glossary) if kind == "report" else _notice(kind, values, form, glossary)
                if item is not None:
                    return item
    return None


_UNREAD = object()  # a cell whose text is not what its slot holds


def _read_rows(form: _FormGrammar, rows: Sequence[str], grammar: _Grammar,
               glossary: Glossary) -> dict[_Slot, Any] | None:
    """What every slot of a form holds, read from the text of its rows; None when a row is not the one
    the form has there."""
    if len(rows) != len(form.rows):
        return None
    values: dict[_Slot, Any] = {}
    for text, (pattern, slots) in zip(rows, form.rows):
        match = pattern.fullmatch(text)
        if match is None:
            return None
        for j, slot in enumerate(slots):
            value = _read_cell(slot, match[f"v{j}"] or "", match.groupdict().get(f"u{j}"), form, grammar, glossary)
            if value is _UNREAD:
                return None
            values[slot] = value
    return values


def _read_cell(slot: _Slot, text: str, symbol: str | None, form: _FormGrammar, grammar: _Grammar,
               glossary: Glossary) -> Any:
    """What one cell holds (None: a field left blank, or noise), or _UNREAD."""
    key, blank = slot.key, text in _BLANK_TEXTS
    if key in ("id", "target"):
        return text if grammar.record_id.fullmatch(text) else _UNREAD
    if key in ("source", "group"):
        return _UNREAD if blank else text
    if key in ("time", "issued", "sent"):
        when = parse_time(text)
        return _UNREAD if when is None else when
    if key == "status":
        return _status(text, grammar, glossary)
    if key == "position":
        position = parse_coords(text, glossary.coord_decimals)
        return _UNREAD if position is None else position
    if key in ("latitude", "longitude"):
        return _read_angle(text, "NS" if key == "latitude" else "EW", glossary.coord_decimals)
    if key in NUMBER_SLOTS:
        index = slot.field if key == "value" else form.field
        return _read_number(glossary.fields[index], text, symbol, grammar.values[index])
    if key in ("qualifier", "new_qualifier", "place", "reason"):
        return None if blank else text
    if key == "xref":
        if blank:
            return None
        match = grammar.xref.fullmatch(text)
        return _UNREAD if match is None else match["same"]
    if key == "amended":
        return None if text == glossary.fields[form.field].label else _UNREAD
    if key == "action" and form.field is not None:  # a correction says which value it revised
        revised = f"the {glossary.fields[form.field].label} was revised; use the corrected entry."
        return None if text == revised else _UNREAD
    if key == "action":  # a retraction names the report it withdraws
        match = grammar.withdrawn.fullmatch(text)
        return _UNREAD if match is None else match["ref"]
    return None  # the form's own noise: its serial references, who prepared, checked or signed it


def _status(text: str, grammar: _Grammar, glossary: Glossary) -> Any:
    """The status whose box is ticked, of the two the glossary names."""
    match = grammar.status.fullmatch(text)
    if match is None or {match["s1"], match["s2"]} != set(glossary.statuses):
        return _UNREAD
    ticked = [match[f"s{i}"] for i in (1, 2) if match[f"m{i}"] in CHECKED]
    return ticked[0] if len(ticked) == 1 else _UNREAD


def _read_angle(text: str, hemispheres: str, decimals: int) -> Any:
    """A latitude or longitude as _angle writes it."""
    match = re.fullmatch(rf"(?P<n>\d+(?:\.\d+)?)°(?P<h>[{hemispheres}])", text)
    if match is not None:
        value = decode(match["n"], decimals)
        return value if match["h"] == hemispheres[0] else -value
    return decode(text, decimals) if re.fullmatch(NUMBER, text) else _UNREAD


def _read_number(f: GlossaryField, text: str, symbol: str | None, pattern: re.Pattern[str]) -> Any:
    """A field's stored value, written with its unit after it, or in the unit its label names."""
    if symbol is not None:
        return f.decode(text, find_unit(f.units, symbol)) if re.fullmatch(NUMBER, text) else _UNREAD
    match = pattern.fullmatch(text)
    return _UNREAD if match is None else read_value(f, match, "v")


def _report(values: Mapping[_Slot, Any], glossary: Glossary) -> RecordReport:
    by_key: dict[str, Any] = {"latitude": None, "longitude": None}
    numbers: dict[str, float] = {}
    qualifiers: dict[str, str] = {}
    for slot, value in values.items():
        if slot.key == "value":
            numbers[glossary.fields[slot.field].key] = value
        elif slot.key == "qualifier":
            if value is not None:
                qualifiers[glossary.fields[slot.field].qualifier] = value
        elif slot.key == "position":
            by_key["latitude"], by_key["longitude"] = value
        else:
            by_key[slot.key] = value
    return RecordReport(ref=by_key["id"], source=by_key["source"], status=by_key["status"], time=by_key["time"],
                        lat=by_key["latitude"], lon=by_key["longitude"], group=by_key["group"], values=numbers,
                        qualifiers=qualifiers, place=by_key["place"], same_as=by_key["xref"])


def _notice(kind: str, values: Mapping[_Slot, Any], form: _FormGrammar, glossary: Glossary) -> Notice | None:
    """The notice an amendment form gives; None for a retraction whose action names another report."""
    by_key = {slot.key: value for slot, value in values.items()}
    if kind == "retraction":
        if by_key["action"] != by_key["target"]:
            return None
        return Notice(kind="retraction", target=by_key["target"], time=by_key["issued"], reason=by_key["reason"])
    return Notice(kind="correction", target=by_key["target"], time=by_key["issued"], reason=by_key["reason"],
                  field=glossary.fields[form.field].key, new_value=by_key["new"], old_value=by_key["old"],
                  qualifier=by_key.get("new_qualifier"))
