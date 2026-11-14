"""classic: the 0.2.0 digest as a genre, for earthquakes and for the records of every other domain.

Bulletin, prose, wire and log report lines, NOTICE, RETRACTION, UPDATE, CORRECTION and Revised
notice lines, a header, section heads and chatter.

On the earthquake domain this genre is the 0.2.0 code, called unchanged with the same arguments and
the same random stream, so legacy runs stay byte-identical: ``digest.choose_style`` draws the look,
``digest.render_digest`` writes the text and ``solver.parse_digest`` reads it back.

The records of any other domain are written in the same four line styles, from the words of the
prompt's glossary: its record noun, field labels (bulletin, prose, wire) or keys (log), units (each
value in a unit the style draws, with the exact codec of evalgen.units), qualifiers, status words and
ID pattern. Times are given to the second, in UTC or with their UTC offset stated; positions in
hemispheres or signed; duplicate notes and notices are worded through the domain's trap vocabulary,
which its reading rules quote. A prose line gives its time as the time the record was recorded, and
places the record in its group with the domain's preposition ("in North Sea", "near Moon":
Domain.group_preposition). The header and section heads are worded from the domain's words too.
The parser reads every line back through the glossary alone, and every rendering is checked to read
back to exactly its reports and notices (genres.base.check_round_trip).

    bulletin  [ukmo] by-12345 | 2026-04-12T06:12:19Z | North Sea | 56.512°N, 3.104°E | water temperature
              12.3 °C (spar) | wave height 6.07 ft | reviewed | same reading as by-54321 | near: Dogger Bank
    prose     06:12:19 UTC on 12 April 2026 — reading by-12345 (reviewed) was recorded in North Sea, at
              (56.512, 3.104): water temperature 54.1 °F (spar), wave height 1.85 m. Listed by ukmo. Note:
              duplicate of by-54321. Location note: Dogger Bank.
    wire      ukmo WIRE [2026-04-12 06:12:19 UTC] North Sea @ 56.512°N, 3.104°E: water temperature 12.3 °C
              (spar); wave height 1.85 m. Ref by-12345, reviewed.
    log       2026-04-12T07:12:19+01:00 ref=by-12345 src=ukmo status=reviewed group="North Sea"
              loc=56.512,3.104 temp=12.3°C probe=spar wave=1.85m same_as=by-54321 place="Dogger Bank"
"""
from __future__ import annotations

import functools
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..digest import LINE_PRESETS, NOTICE_PLACEMENTS, choose_style, render_digest
from ..domains.base import Domain, Glossary, GlossaryField, RecordReport, TrapVocab
from ..domains.earthquakes import Earthquakes
from ..models import Notice
from ..ops import DigestData
from ..solver import parse_digest
from ..units import Unit, decode, encode
from .base import (COORDS_PATTERN, GROUP_PREPOSITION, NUMBER, TIME_FORMATS, TIME_PATTERN, UTC_TIME_FORMATS, Genre,
                   RenderContext, alternation, check_round_trip, format_coords, format_time, format_value, id_pattern,
                   parse_coords, parse_time, read_value, value_pattern)

# ------------------------------------------------------------------------------ the genre


class Classic(Genre):
    name = "classic"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if _legacy(domain):
            return choose_style(rng, avoid=avoid)
        return _record_style(rng, domain, avoid)

    def render(self, ctx: RenderContext) -> str:
        """The 0.2.0 digest of earthquakes; the digest of any other domain's records, in the words of
        the context's glossary (the domain's own when it has none), checked to read back exactly."""
        if _legacy(ctx.domain):
            return render_digest(ctx.reports, ctx.notices, ctx.chatter, compiled_at=ctx.compiled_at,
                                 window_hours=ctx.window_hours, style=ctx.style, rng=ctx.rng)
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _render_records(ctx, glossary)
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """A classic digest opens with one of its headers, whose title is the records' plural (the
        0.2.0 headers are these with an earthquake title). Its report lines are no sign: the prose and
        log patterns accept any text before the time, a quoting prefix included."""
        first = digest_text.lstrip().split("\n", 1)[0]
        return any(pattern.match(first) for pattern in _HEADER_LINES)

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """The 0.2.0 parser for earthquakes; for any other domain, every report and notice exactly, read
        through the prompt's glossary, which such a prompt always prints."""
        if _legacy(domain):
            return parse_digest(digest_text)
        if glossary is None:
            raise ValueError(f"a digest of {domain.name} records is read through the glossary its prompt prints, and "
                             "this prompt prints none")
        return _parse_records(digest_text, glossary)


def _legacy(domain: Domain) -> bool:
    """Earthquake domains keep the 0.2.0 renderer and parser: their records are 0.2.0 Events and Reports."""
    return isinstance(domain, Earthquakes)


CLASSIC = Classic()
GENRE = CLASSIC  # what the registry looks up (see evalgen.genres)

# ------------------------------------------------------------------ the look of record digests
# A rendering style picks one preset per dimension, as for earthquakes; each preset is a weighted mix.

RECORD_TIME_PRESETS = {
    "utc_heavy": {"iso": 0.3, "iso_space": 0.2, "rfc": 0.2, "prose_utc": 0.2, "iso_offset": 0.04, "local12": 0.03,
                  "local24": 0.03},
    "local_heavy": {"iso": 0.1, "iso_space": 0.1, "rfc": 0.1, "prose_utc": 0.1, "iso_offset": 0.2, "local12": 0.2,
                    "local24": 0.2},
    "mixed": {fmt: 1.0 for fmt in TIME_FORMATS},
}
RECORD_COORD_PRESETS = {
    "hemi": {"hemi": 0.7, "signed": 0.3},
    "signed": {"hemi": 0.3, "signed": 0.7},
    "mixed": {"hemi": 0.5, "signed": 0.5},
}
# The weight of a field's stored unit in each preset; the others share the rest (None: all alike).
UNIT_PRESETS: dict[str, float | None] = {"mostly_stored": 0.7, "mostly_other": 0.15, "mixed": None}
RECORD_LAYOUTS = ("by_source", "by_time", "shuffled")
# The first lines are the 0.2.0 ones with the records' plural as the title (sniff recognises both).
RECORD_HEADERS = (
    "{TITLE} DIGEST (edition {edition})\n"
    "Compiled {compiled}; covers the {window} before compilation. Reports are reproduced as received.",
    "{Title} wire roundup: the {window} ending {compiled}\n"
    "Sources: every {source_label} that filed a report. Edition {edition}.",
    "== {TITLE} DIGEST {edition} ==\n"
    "Window: the {window} up to {compiled}. Entries are listed as filed by each {source_label}.",
)
RECORD_SECTION_HEADS = ("-- {source} --", "From {source}:", "== {source} feed ==")
ZONES = (-12, 14)  # the UTC offsets, in whole hours, of sources whose records have no position


def _first_line_pattern(header: str) -> re.Pattern[str]:
    """A header's first line: its title, the records' plural, as text that opens with a letter or digit
    (so a quoted digest's "> " is no title), and its other placeholders as wildcards."""
    pieces = re.split(r"(\{\w+\})", header.split("\n")[0])
    return re.compile("^" + "".join(r"[^\W_].*?" if piece in ("{TITLE}", "{Title}") else ".+" if piece.startswith("{")
                                    else re.escape(piece) for piece in pieces) + "$")


_HEADER_LINES = tuple(_first_line_pattern(header) for header in RECORD_HEADERS)


def _record_style(rng: random.Random, domain: Domain, avoid: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pick the presets of one rendering of a record digest; a variant avoids the original's presets."""

    def pick(key: str, options: Sequence[str] | Mapping[str, Any]) -> str:
        names = sorted(options)
        if avoid is not None:
            names = [n for n in names if n != avoid[key]] or names
        return rng.choice(names)

    style = {"units": pick("units", UNIT_PRESETS), "time": pick("time", RECORD_TIME_PRESETS)}
    if domain.has_location:
        style["coords"] = pick("coords", RECORD_COORD_PRESETS)
    style.update(lines=pick("lines", LINE_PRESETS), layout=pick("layout", RECORD_LAYOUTS),
                 notices=rng.choice(NOTICE_PLACEMENTS), header=rng.randrange(len(RECORD_HEADERS)))
    return style


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


def _window(hours: int) -> str:
    if hours > 24 and hours % 24 == 0:
        return f"{hours // 24} days"
    return f"{hours} hour{'' if hours == 1 else 's'}"


# ---------------------------------------------------------------------------- writing records


def _render_records(ctx: RenderContext, glossary: Glossary) -> str:
    """The digest of a non-earthquake domain's reports, notices and chatter, as `ctx.style` draws it."""
    rng, style, vocab = ctx.rng, ctx.style, ctx.domain.trap_vocab
    sources = sorted({r.source for r in ctx.reports})
    zones = {} if glossary.located else {source: rng.randint(*ZONES) for source in sources}
    lines: list[tuple[RecordReport, str]] = []
    for r in ctx.reports:
        line_style = _draw(rng, LINE_PRESETS[style["lines"]])
        offset = ctx.domain.utc_offset(r) if glossary.located else zones[r.source]
        when = format_time(r.time, _draw(rng, RECORD_TIME_PRESETS[style["time"]]), offset)
        where = (format_coords(r.lat, r.lon, _draw(rng, RECORD_COORD_PRESETS[style["coords"]]), glossary.coord_decimals)
                 if glossary.located else None)
        units = [_draw_unit(rng, f, style["units"]) for f in glossary.fields]
        note = rng.choice(vocab.duplicate_notes) if r.same_as else None
        lines.append((r, _record_line(r, line_style, glossary, when, where, units, note, ctx.domain.group_preposition)))

    body: list[str] = []
    if style["layout"] == "by_time":
        lines.sort(key=lambda pair: (pair[0].time, pair[0].ref, pair[0].source))
        body = [text for _, text in lines]
    elif style["layout"] == "by_source":
        rng.shuffle(sources)
        for source in sources:
            body.append(rng.choice(RECORD_SECTION_HEADS).format(source=source))
            group = [text for r, text in lines if r.source == source]
            rng.shuffle(group)
            body.extend(group)
    else:
        rng.shuffle(lines)
        body = [text for _, text in lines]

    notice_lines = [_notice_line(n, rng, glossary, vocab, style["units"]) for n in ctx.notices]
    if style["notices"] == "interleaved":
        for text in notice_lines:
            body.insert(rng.randrange(len(body) + 1), text)
    elif notice_lines:
        rng.shuffle(notice_lines)
        body += ["", "Corrections and retractions:"] + notice_lines
    for text in ctx.chatter:
        body.insert(rng.randrange(len(body) + 1), text)

    header = RECORD_HEADERS[style["header"]].format(
        TITLE=glossary.plural.upper(), Title=glossary.plural[:1].upper() + glossary.plural[1:],
        edition=f"{rng.randrange(100, 1000)}-{rng.choice('ABCDEFGHJK')}",
        compiled=format_time(ctx.compiled_at, rng.choice(UTC_TIME_FORMATS)), window=_window(ctx.window_hours),
        source_label=glossary.source_label)
    return "\n".join([header, ""] + body)


def _record_line(r: RecordReport, style: str, glossary: Glossary, when: str, where: str | None,
                 units: Sequence[Unit | None], note: str | None, preposition: str = "in") -> str:
    """One report as a line of `style`: bulletin, prose, wire or log (see the module docstring); a prose
    line places the record in its group with `preposition` (Domain.group_preposition)."""
    if style == "log":
        text = f"{when} ref={r.ref} src={_token(r.source)} status={r.status} group={_token(r.group)}"
        if glossary.located:
            text += f" loc={encode(r.lat, glossary.coord_decimals)},{encode(r.lon, glossary.coord_decimals)}"
        for f, unit in zip(glossary.fields, units):
            text += f" {f.key}={format_value(f, r.values[f.key], unit, separator='')}"
            if _qualifier(f, r) is not None:
                text += f" {f.qualifier}={_token(_qualifier(f, r))}"
        if r.same_as:
            text += f" same_as={r.same_as}"
        if r.place is not None:
            text += f" place={_token(r.place)}"
        return text
    values = [f"{f.label} {format_value(f, r.values[f.key], unit)}"
              + ("" if _qualifier(f, r) is None else f" ({_qualifier(f, r)})")
              for f, unit in zip(glossary.fields, units)]
    if style == "bulletin":
        parts = [f"[{r.source}] {r.ref}", when, r.group, *([where] if where is not None else []), *values, r.status]
        if r.same_as:
            parts.append(f"{note} {r.same_as}")
        if r.place is not None:
            parts.append(f"near: {r.place}")
        return " | ".join(parts)
    if style == "prose":  # the time is when the record was recorded, never when its source listed it
        text = (f"{when} — {glossary.noun} {r.ref} ({r.status}) was recorded {preposition} {r.group}"
                + ("" if where is None else f", at {where}") + ": " + ", ".join(values) + f". Listed by {r.source}.")
    elif style == "wire":
        text = (f"{r.source} WIRE [{when}] {r.group}" + ("" if where is None else f" @ {where}") + ": "
                + "; ".join(values) + f". Ref {r.ref}, {r.status}.")
    else:
        raise ValueError(f"unknown line style {style!r}")
    if r.same_as:
        text += f" Note: {note} {r.same_as}."
    if r.place is not None:
        text += f" Location note: {r.place}."
    return text


def _qualifier(f: GlossaryField, r: RecordReport) -> str | None:
    """The qualifier a report gives with a field's value (None: the field has none, or the report gives none)."""
    return None if f.qualifier is None else r.qualifiers.get(f.qualifier)


def _notice_line(n: Notice, rng: random.Random, glossary: Glossary, vocab: TrapVocab, units: str) -> str:
    """A notice in one of three wordings of its kind, each using a phrase of the domain's trap vocabulary
    or a keyword that its reading rules name; a correction gives both values in one drawn unit."""
    form = rng.randrange(3)
    when = format_time(n.time, rng.choice(UTC_TIME_FORMATS))
    if n.kind == "retraction":
        phrase = rng.choice(vocab.retraction_notices)
        reason = "" if n.reason is None else f" ({n.reason})"
        return (f"NOTICE [{when}]: report {n.target} has been {phrase}{reason}. Disregard it.",
                f"RETRACTION [{when}]: {n.target} was not a real {glossary.noun}{reason}; ignore that report.",
                f"UPDATE [{when}]: {n.target} {phrase}{reason}.")[form]
    field = next((f for f in glossary.fields if f.key == n.field), None)
    if n.kind != "correction" or field is None or n.new_value is None or n.old_value is None:
        raise ValueError(f"a {n.kind} notice of {n.target} that names no field of the glossary, or no new and old "
                         "value, cannot be printed")
    unit = _draw_unit(rng, field, units)
    phrase = rng.choice(vocab.correction_notices)
    new, old = format_value(field, n.new_value, unit), format_value(field, n.old_value, unit)
    qualifier = "" if n.qualifier is None else f" ({n.qualifier})"
    return (f"CORRECTION [{when}]: the {field.label} of {n.target} is {phrase} {new}{qualifier}; it was first given "
            f"as {old}.",
            f"UPDATE [{when}]: {field.label} for {n.target} {phrase} {new}{qualifier} (it was {old}).",
            f"Revised [{when}]: {n.target} {field.label} now {new}{qualifier} (was {old}).")[form]


_PLAIN_TOKEN = re.compile(r'[^\s"=]+')
TOKEN = r'"(?:[^"\\]|\\.)*"|[^\s"=]+'  # a log value: as it is, or a JSON string when it has spaces, quotes or "="


def _token(text: str) -> str:
    return text if _PLAIN_TOKEN.fullmatch(text) else json.dumps(text, ensure_ascii=False)


def _untoken(text: str) -> str:
    return json.loads(text) if text.startswith('"') else text


# ---------------------------------------------------------------------------- reading records


@dataclass(frozen=True)
class _Grammar:
    """The patterns of one glossary's record digests: a whole line of each style, and the bodies of the
    notices that open with each keyword (with the field a correction revises; None: a retraction)."""

    lines: tuple[tuple[re.Pattern[str], bool], ...]  # bulletin, prose, wire and log; True: the log's
    notices: Mapping[str, tuple[tuple[re.Pattern[str], GlossaryField | None], ...]]


NOTICE_KEYWORDS = ("NOTICE", "RETRACTION", "UPDATE", "CORRECTION", "Revised")  # how notice lines open
_NOTICE = re.compile(rf"^(?P<keyword>{'|'.join(NOTICE_KEYWORDS)}) \[(?P<time>[^\]]+)\]: (?P<body>.*)$")


def _labelled(f: GlossaryField, i: int) -> str:
    """A field's value as bulletin, prose and wire lines print it: its label, its value and unit, and its
    qualifier in parentheses (groups v{i}, v{i}_unit and q{i})."""
    qualifier = "" if f.qualifier is None else rf"(?: \((?P<q{i}>[^()]*)\))?"
    return re.escape(f.label) + " " + value_pattern(f, f"v{i}") + qualifier


def _keyed(f: GlossaryField, i: int) -> str:
    """A field's value as log lines print it: key=value, then qualifier=value."""
    qualifier = "" if f.qualifier is None else rf"(?: {re.escape(f.qualifier)}=(?P<q{i}>{TOKEN}))?"
    return re.escape(f.key) + "=" + value_pattern(f, f"v{i}", separator="") + qualifier


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    record_id, status = id_pattern(glossary), alternation(glossary.statuses)
    located = glossary.located
    tail = rf"(?: Note: (?P<note>.+?) (?P<same>{record_id})\.)?(?: Location note: (?P<place>.+)\.)?"
    labelled = [_labelled(f, i) for i, f in enumerate(glossary.fields)]
    # Times and positions are matched by their own patterns, so that free words around them (a group,
    # a source, a place) cannot take a delimiter for theirs: a match backtracks until a real time,
    # position or value fits.
    when, where = f"(?P<time>{TIME_PATTERN})", f"(?P<coords>{COORDS_PATTERN})"
    bulletin = (rf"^\[(?P<source>.+?)\] (?P<ref>{record_id}) \| {when} \| (?P<group>.+?)"
                + (rf" \| {where}" if located else "") + "".join(rf" \| {part}" for part in labelled)
                + rf" \| (?P<status>{status})(?: \| (?P<note>.+?) (?P<same>{record_id}))?(?: \| near: (?P<place>.+))?$")
    prose = (rf"^{when} — {re.escape(glossary.noun)} (?P<ref>{record_id}) \((?P<status>{status})\) was recorded "
             rf"{GROUP_PREPOSITION} (?P<group>.+?)" + (f", at {where}" if located else "") + ": "
             + ", ".join(labelled) + r"\. Listed by (?P<source>.+?)\." + tail + "$")
    wire = (rf"^(?P<source>.+?) WIRE \[{when}\] (?P<group>.+?)" + (f" @ {where}" if located else "")
            + ": " + "; ".join(labelled) + rf"\. Ref (?P<ref>{record_id}), (?P<status>{status})\." + tail + "$")
    log = (rf"^{when} ref=(?P<ref>{record_id}) src=(?P<source>{TOKEN}) status=(?P<status>{status}) "
           rf"group=(?P<group>{TOKEN})" + (rf" loc=(?P<lat>{NUMBER}),(?P<lon>{NUMBER})" if located else "")
           + "".join(" " + _keyed(f, i) for i, f in enumerate(glossary.fields))
           + rf"(?: same_as=(?P<same>{record_id}))?(?: place=(?P<place>{TOKEN}))?$")

    reason = r"(?: \((?P<reason>.+)\))?"
    retractions = {
        "NOTICE": rf"report (?P<ref>{record_id}) has been (?P<phrase>.+?){reason}\. Disregard it\.",
        "RETRACTION": rf"(?P<ref>{record_id}) was not a real {re.escape(glossary.noun)}{reason}; ignore that report\.",
        "UPDATE": rf"(?P<ref>{record_id}) (?P<phrase>.+?){reason}\.",
    }
    notices: dict[str, list[tuple[re.Pattern[str], GlossaryField | None]]] = {k: [] for k in NOTICE_KEYWORDS}
    for f in glossary.fields:  # corrections first: an UPDATE that revises a value is no retraction
        label, new, old = re.escape(f.label), value_pattern(f, "new"), value_pattern(f, "old")
        qualifier = "" if f.qualifier is None else r"(?: \((?P<qualifier>[^()]*)\))?"
        for keyword, body in (
                ("CORRECTION", rf"the {label} of (?P<ref>{record_id}) is (?P<phrase>.+?) {new}{qualifier}; it was "
                               rf"first given as {old}\."),
                ("UPDATE", rf"{label} for (?P<ref>{record_id}) (?P<phrase>.+?) {new}{qualifier} \(it was {old}\)\."),
                ("Revised", rf"(?P<ref>{record_id}) {label} now {new}{qualifier} \(was {old}\)\.")):
            notices[keyword].append((re.compile(body), f))
    for keyword, body in retractions.items():
        notices[keyword].append((re.compile(body), None))
    return _Grammar(lines=tuple((re.compile(pattern), pattern is log) for pattern in (bulletin, prose, wire, log)),
                    notices={k: tuple(v) for k, v in notices.items()})


def _parse_records(text: str, glossary: Glossary) -> DigestData:
    """Every report and notice of a record digest; lines that are neither change nothing."""
    grammar = _grammar(glossary)
    reports: list[Any] = []
    notices: list[Notice] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        notice = _parse_notice(line, grammar)
        if notice is not None:
            notices.append(notice)
            continue
        report = _parse_record(line, grammar, glossary)
        if report is not None:
            reports.append(report)
    return DigestData(reports, notices)


def _parse_notice(line: str, grammar: _Grammar) -> Notice | None:
    match = _NOTICE.match(line)
    if match is None:
        return None
    when = parse_time(match["time"])
    if when is None:
        return None
    for pattern, field in grammar.notices[match["keyword"]]:
        body = pattern.fullmatch(match["body"])
        if body is None:
            continue
        if field is None:
            return Notice(kind="retraction", target=body["ref"], time=when, reason=body["reason"])
        return Notice(kind="correction", target=body["ref"], time=when, field=field.key,
                      new_value=read_value(field, body, "new"), old_value=read_value(field, body, "old"),
                      qualifier=body.groupdict().get("qualifier"))
    return None


def _parse_record(line: str, grammar: _Grammar, glossary: Glossary) -> RecordReport | None:
    for pattern, log in grammar.lines:
        match = pattern.match(line)
        if match is not None:
            return _report(match, glossary, log)
    return None


def _report(match: re.Match[str], glossary: Glossary, log: bool) -> RecordReport | None:
    """The report a line's match gives (`log`: a log line, whose words are tokens, as they are or JSON
    strings, and whose position is "lat,lon"); None when its time or position is none."""
    when = parse_time(match["time"])
    if not glossary.located:
        position: tuple[float | None, float | None] | None = (None, None)
    elif log:
        position = decode(match["lat"], glossary.coord_decimals), decode(match["lon"], glossary.coord_decimals)
    else:
        position = parse_coords(match["coords"], glossary.coord_decimals)
    if when is None or position is None:
        return None
    word = _untoken if log else str
    values = {f.key: read_value(f, match, f"v{i}") for i, f in enumerate(glossary.fields)}
    qualifiers = {f.qualifier: word(match[f"q{i}"]) for i, f in enumerate(glossary.fields)
                  if f.qualifier is not None and match[f"q{i}"] is not None}
    return RecordReport(ref=match["ref"], source=word(match["source"]), status=match["status"], time=when,
                        lat=position[0], lon=position[1], group=word(match["group"]), values=values,
                        qualifiers=qualifiers, place=None if match["place"] is None else word(match["place"]),
                        same_as=match["same"])
