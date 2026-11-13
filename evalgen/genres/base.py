"""The genre interface: how one kind of document presents a digest's content.

An instance's content is fixed before it is rendered: its reports, notices and chatter lines. A
*genre* only decides how that content looks as text (the 0.2.0 bulletin, wire, log and prose lines,
a news article, an email thread, a chat log, a table, a form, a radio transcript), and must read its
own text back to exactly the same reports and notices.

    RenderContext  everything one rendering is drawn from
    Genre          the base class; each genre module exports an instance as ``GENRE``

A non-legacy prompt prints its records' glossary (``Domain.glossary``, see evalgen.domains.base):
``render_glossary`` prints it and ``parse_glossary`` reads it back from the prompt text alone, so a
genre's parser learns every word, unit, precision and ID pattern from the prompt itself.

Every genre writes the fields of a record in these shared formats, each read back exactly by its
inverse, so the text-only oracle recovers every report:

    format_time / parse_time        a time to the second, in UTC or with its UTC offset stated
      / TIME_PATTERN                (any of them, as a regex to place in a line's pattern)
    format_coords / parse_coords    a position, in hemispheres or signed
      / COORDS_PATTERN
    format_value / value_pattern    a field's value in one of its units, through the exact codec of
      / read_value                  evalgen.units
    id_pattern                      a record ID, as the glossary's pattern describes it

``check_round_trip`` refuses a rendering that its genre does not read back to exactly its reports
and notices.
"""
from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from ..digest import MONTHS, MONTHS_LONG, WEEKDAYS
from ..domains.base import Domain, Glossary, GlossaryField, RecordReport
from ..models import Notice, Report
from ..ops import DigestData
from ..units import Unit, decode, encode, find_unit
from ..util import plain_number

# ------------------------------------------------------------------------------- the interface


@dataclass(frozen=True)
class RenderContext:
    """What one rendering is drawn from. The lists are shared by every rendering of an instance, so a
    genre must never modify them."""

    reports: Sequence[Report | RecordReport]  # in planting order: what every per-line draw follows
    notices: Sequence[Notice]
    chatter: Sequence[str]  # lines that carry no data
    compiled_at: int  # when the digest was compiled: the end of its time window (Unix seconds, UTC)
    window_hours: int
    style: Mapping[str, Any]  # what the genre's choose_style drew for this rendering
    rng: random.Random  # the rendering's own stream, continued from choose_style
    domain: Domain
    glossary: Glossary | None = None  # what the prompt prints about the records; None on the legacy path


class Genre(ABC):
    """One kind of document a digest can be written as. A genre module subclasses this, implements
    the four methods and exports an instance as ``GENRE`` (see evalgen.genres)."""

    name: str  # registry key: "classic"

    @abstractmethod
    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Draw how one rendering looks. A metamorphic variant in the same genre passes the original's
        style as `avoid`, and must then differ from it."""

    @abstractmethod
    def render(self, ctx: RenderContext) -> str:
        """The digest text of one rendering."""

    @abstractmethod
    def sniff(self, digest_text: str) -> bool:
        """Whether `digest_text` is this genre's output. No genre may recognise another genre's text:
        the oracle picks its parser by sniffing (see evalgen.genres.detect_genre)."""

    @abstractmethod
    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """The exact inverse of render: every report and notice, read from the text alone, with the
        labels, units and precision of the prompt's glossary (None: a legacy prompt, which prints
        none). The caller names the records' domain, which the prompt never does."""


def check_round_trip(genre: Genre, text: str, ctx: RenderContext, glossary: Glossary) -> None:
    """Raise ValueError unless `genre` reads `text`, its rendering of `ctx`, back to exactly the
    reports and notices of `ctx`, in any order (apart from the generator's bookkeeping, which is never
    printed: a report's role and trap, a notice's trap). A genre checks every rendering it writes this
    way, so that the oracle recovers every one of them exactly."""
    data = genre.parse(text, glossary, ctx.domain)
    reports = [replace(r, role="primary", trap_id=None) for r in ctx.reports]
    notices = [replace(n, trap_id=None) for n in ctx.notices]
    missing = _unmatched(reports, data.reports) + _unmatched(notices, data.notices)
    extra = _unmatched(data.reports, reports) + _unmatched(data.notices, notices)
    if missing or extra:
        raise ValueError(f"the {genre.name} genre does not read its rendering back exactly: it misses {missing[:3]} "
                         f"and reads {extra[:3]} instead (are some words of the domain or its surface unprintable "
                         "in this genre?)")


def _unmatched(items: Sequence[Any], pool: Sequence[Any]) -> list[Any]:
    """The items that are not in `pool`, counting repeats (a multiset difference by ==)."""
    remaining = list(pool)
    out = []
    for item in items:
        if item in remaining:
            remaining.remove(item)
        else:
            out.append(item)
    return out


# ----------------------------------------------------------------------------------- times

UTC_TIME_FORMATS = ("iso", "iso_space", "rfc", "prose_utc")  # also the times of notices and headers
OFFSET_TIME_FORMATS = ("iso_offset", "local12", "local24")  # local time, with its UTC offset stated
TIME_FORMATS = UTC_TIME_FORMATS + OFFSET_TIME_FORMATS


def format_time(ts: int, fmt: str, offset_hours: int = 0) -> str:
    """A time to the second in one of TIME_FORMATS: in UTC, or in the local time `offset_hours` from
    UTC with the offset stated ("19:39:19 (UTC-05:00) 12 Apr 2026"). Every format reads back exactly."""
    utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    clock = f"{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"
    date = f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}"
    if fmt == "iso":
        return f"{date}T{clock}Z"
    if fmt == "iso_space":
        return f"{date} {clock} UTC"
    if fmt == "rfc":
        return f"{WEEKDAYS[utc.weekday()]}, {utc.day:02d} {MONTHS[utc.month - 1]} {utc.year} {clock} +0000"
    if fmt == "prose_utc":
        return f"{clock} UTC on {utc.day} {MONTHS_LONG[utc.month - 1]} {utc.year}"
    local = utc + timedelta(hours=offset_hours)
    zone = f"{'+' if offset_hours >= 0 else '-'}{abs(offset_hours):02d}:00"
    if fmt == "iso_offset":
        return (f"{local.year:04d}-{local.month:02d}-{local.day:02d}T{local.hour:02d}:{local.minute:02d}:"
                f"{local.second:02d}{zone}")
    if fmt == "local12":
        return (f"{local.hour % 12 or 12}:{local.minute:02d}:{local.second:02d} {'AM' if local.hour < 12 else 'PM'} "
                f"local time (UTC{zone}) on {MONTHS[local.month - 1]} {local.day}, {local.year}")
    if fmt == "local24":
        return (f"{local.hour:02d}:{local.minute:02d}:{local.second:02d} (UTC{zone}) {local.day} "
                f"{MONTHS[local.month - 1]} {local.year}")
    raise ValueError(f"unknown time format {fmt!r}")


_CLOCK = r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
_DATE = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
_ZONE = r"(?P<sign>[+-])(?P<offset>\d{2}):00"
_MON = "(?P<mon>" + "|".join(MONTHS) + ")"
_TIME_PATTERNS = tuple(re.compile(pattern) for pattern in (
    _DATE + "T" + _CLOCK + "Z",
    _DATE + " " + _CLOCK + " UTC",
    "(?:" + "|".join(WEEKDAYS) + r"), (?P<day>\d{2}) " + _MON + r" (?P<year>\d{4}) " + _CLOCK + r" \+0000",
    _CLOCK + r" UTC on (?P<day>\d{1,2}) (?P<month_name>" + "|".join(MONTHS_LONG) + r") (?P<year>\d{4})",
    _DATE + "T" + _CLOCK + _ZONE,
    r"(?P<hour12>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2}) (?P<half>AM|PM) local time \(UTC" + _ZONE + r"\) on "
    + _MON + r" (?P<day>\d{1,2}), (?P<year>\d{4})",
    _CLOCK + r" \(UTC" + _ZONE + r"\) (?P<day>\d{1,2}) " + _MON + r" (?P<year>\d{4})",
))


TIME_PATTERN = "|".join(re.sub(r"\(\?P<\w+>", "(?:", pattern.pattern) for pattern in _TIME_PATTERNS)  # any, no groups


def parse_time(text: str) -> int | None:
    """The Unix time that `text`, the whole of it, gives in one of TIME_FORMATS; None if it is none."""
    for pattern in _TIME_PATTERNS:
        match = pattern.fullmatch(text)
        if match is not None:
            return _timestamp(match.groupdict())
    return None


def _timestamp(g: Mapping[str, str | None]) -> int | None:
    if g.get("month") is not None:
        month = int(g["month"])
    elif g.get("mon") is not None:
        month = MONTHS.index(g["mon"]) + 1
    else:
        month = MONTHS_LONG.index(g["month_name"]) + 1
    hour = int(g["hour"]) if g.get("hour") is not None else int(g["hour12"]) % 12 + (12 if g["half"] == "PM" else 0)
    offset = 0 if g.get("sign") is None else int(g["offset"]) * (1 if g["sign"] == "+" else -1)
    try:
        local = datetime(int(g["year"]), month, int(g["day"]), hour, int(g["minute"]), int(g["second"]),
                         tzinfo=timezone.utc)
    except ValueError:  # 31 April, 25:00
        return None
    return int(local.timestamp()) - offset * 3600  # local time = UTC + offset


# ------------------------------------------------------------------------------- positions

COORD_FORMATS = ("hemi", "signed")  # "56.512°N, 3.104°W" or "(56.512, -3.104)"
NUMBER = r"-?\d+(?:\.\d+)?"
_HEMI = re.compile(r"(?P<lat>\d+(?:\.\d+)?)°(?P<ns>[NS]), (?P<lon>\d+(?:\.\d+)?)°(?P<ew>[EW])")
_SIGNED = re.compile(rf"\((?P<lat>{NUMBER}), (?P<lon>{NUMBER})\)")
COORDS_PATTERN = "|".join(re.sub(r"\(\?P<\w+>", "(?:", pattern.pattern) for pattern in (_HEMI, _SIGNED))  # no groups


def format_coords(lat: float, lon: float, fmt: str, decimals: int) -> str:
    """A position stored to `decimals` places, in one of COORD_FORMATS, exactly (evalgen.units)."""
    if fmt == "hemi":
        return (f"{encode(abs(lat), decimals)}°{'N' if lat >= 0 else 'S'}, "
                f"{encode(abs(lon), decimals)}°{'E' if lon >= 0 else 'W'}")
    if fmt == "signed":
        return f"({encode(lat, decimals)}, {encode(lon, decimals)})"
    raise ValueError(f"unknown coordinate format {fmt!r}")


def parse_coords(text: str, decimals: int) -> tuple[float, float] | None:
    """The (lat, lon) that `text`, the whole of it, gives in one of COORD_FORMATS; None if it is none."""
    match = _HEMI.fullmatch(text)
    if match is not None:
        lat, lon = decode(match["lat"], decimals), decode(match["lon"], decimals)
        return (lat if match["ns"] == "N" else -lat), (lon if match["ew"] == "E" else -lon)
    match = _SIGNED.fullmatch(text)
    if match is not None:
        return decode(match["lat"], decimals), decode(match["lon"], decimals)
    return None


# ---------------------------------------------------------------------------- values and IDs


def format_value(field: GlossaryField, value: float, unit: Unit | None, separator: str = " ") -> str:
    """A stored value of `field` written in `unit` (None for a field without units) with the exact
    codec, then the unit's symbol: "12.3 °C", or "12.3°C" with no separator."""
    text = field.encode(value, unit)
    return text if unit is None else f"{text}{separator}{unit.symbol}"


def value_pattern(field: GlossaryField, name: str, separator: str = " ") -> str:
    """A regex for a value of `field` as format_value writes it, with its number in the group `name`
    and, for a field with units, the unit's symbol in the group `name`_unit (read_value reads both)."""
    number = f"(?P<{name}>{NUMBER})"
    if not field.units:
        return number
    return f"{number}{re.escape(separator)}(?P<{name}_unit>{alternation(u.symbol for u in field.units)})"


def read_value(field: GlossaryField, match: re.Match[str], name: str) -> float:
    """The stored value that a match of value_pattern(field, name) stands for."""
    symbol = match[f"{name}_unit"] if field.units else None
    return field.decode(match[name], None if symbol is None else find_unit(field.units, symbol))


def id_pattern(glossary: Glossary) -> str:
    """A regex for a record ID in digest text: the glossary's pattern, ignoring letter case (IDs are
    printed as they are, "WX-482137"; the pattern describes them lower-cased, and the glossary says
    that it ignores letter case), never inside a longer word."""
    return rf"(?<![0-9A-Za-z])(?i:{glossary.id_pattern})(?![0-9A-Za-z])"


def alternation(words: Iterable[str]) -> str:
    """A regex matching any of `words` literally, the longest first ("m/s" before "m")."""
    return "|".join(re.escape(word) for word in sorted(set(words), key=lambda w: (-len(w), w)))


# ---------------------------------------------------------------------------------- glossary

GLOSSARY_TITLE = "Glossary: what the records in the digest are, and how their values are written."
# How the glossary qualifies its ID pattern. A pattern describes IDs lower-cased, and digests print
# IDs as their sources file them ("WX-482137", "KX/40512"), so the pattern is true of every printed
# ID only once letter case is ignored, as every reader of a digest (id_pattern, the scorer) matches it.
ID_CASE = "ignoring letter case"
_GLOSSARY = re.compile(r"<glossary>\n(?P<body>.*?)\n</glossary>", re.DOTALL)
_QUOTED = r'"(?:[^"\\]|\\.)*"'  # a word, written as a JSON string


def _quote(word: str) -> str:
    return json.dumps(word, ensure_ascii=False)


def _places(n: int) -> str:
    return f"{n} decimal place{'' if n == 1 else 's'}"


def render_glossary(glossary: Glossary) -> str:
    """The glossary as a non-legacy prompt prints it, between <glossary> tags and ending in a newline:
    one line each for the records and their IDs, sources, groups, statuses and positions, and one per
    field with a line per unit and its conversion. Words are quoted as JSON strings and conversions
    are written exactly, so parse_glossary reads it back to exactly `glossary`. A glossary that would
    not read back, or that holds a digest tag (which readers would take for the digest), raises
    ValueError."""
    final, draft = glossary.statuses
    lines = [GLOSSARY_TITLE,
             f"- Records: each is one {_quote(glossary.noun)} (plural {_quote(glossary.plural)}), identified by its "
             f"{_quote(glossary.id_label)}; IDs match the pattern {glossary.id_pattern}, {ID_CASE}.",
             f"- Sources: each report is filed by one {_quote(glossary.source_label)}.",
             f"- Groups: each record belongs to one {_quote(glossary.group_label)}.",
             f"- Statuses: a {_quote(final)} report settles a record's values; a {_quote(draft)} report does not."]
    if glossary.coord_decimals is not None:
        lines.append(f"- Positions: latitude and longitude, stored to {_places(glossary.coord_decimals)} of a degree.")
    for f in glossary.fields:
        stored = "a plain number, stored" if f.canonical_unit is None else f"stored in {_quote(f.canonical_unit)}"
        qualifier = "" if f.qualifier is None else f"; its {_quote(f.qualifier)} is given with each value"
        lines.append(f"- Field {_quote(f.key)}: {_quote(f.label)}, {stored} to {_places(f.decimals)}{qualifier}.")
        for unit in f.units:
            lines.append(f"  - {_quote(unit.symbol)} ({_quote(unit.words)}): " + _conversion(unit, f.canonical_unit))
    text = "<glossary>\n" + "\n".join(lines) + "\n</glossary>\n"
    if parse_glossary(text) != glossary or "<digest>" in text or "</digest>" in text:
        raise ValueError(f"the glossary of the {glossary.plural} cannot be printed so that it reads back exactly")
    return text


def _conversion(unit: Unit, canonical: str | None) -> str:
    if unit.symbol == canonical:
        return "the stored unit."
    offset = "" if not unit.offset else f" {'+' if unit.offset > 0 else '-'} {plain_number(abs(unit.offset))}"
    return (f"a value in {_quote(canonical)} is the value in {_quote(unit.symbol)} × {plain_number(unit.scale)}"
            f"{offset}.")


_RECORDS = re.compile(rf"- Records: each is one (?P<noun>{_QUOTED}) \(plural (?P<plural>{_QUOTED})\), identified by "
                      rf"its (?P<id_label>{_QUOTED}); IDs match the pattern (?P<pattern>.+), {ID_CASE}\.")
_SOURCES = re.compile(rf"- Sources: each report is filed by one (?P<label>{_QUOTED})\.")
_GROUPS = re.compile(rf"- Groups: each record belongs to one (?P<label>{_QUOTED})\.")
_STATUSES = re.compile(rf"- Statuses: a (?P<final>{_QUOTED}) report settles a record's values; a "
                       rf"(?P<draft>{_QUOTED}) report does not\.")
_POSITIONS = re.compile(r"- Positions: latitude and longitude, stored to (?P<decimals>\d+) decimal places? of a "
                        r"degree\.")
_FIELD = re.compile(rf"- Field (?P<key>{_QUOTED}): (?P<label>{_QUOTED}), (?:a plain number, stored|stored in "
                    rf"(?P<canonical>{_QUOTED})) to (?P<decimals>\d+) decimal places?(?:; its (?P<qualifier>{_QUOTED}) "
                    r"is given with each value)?\.")
_UNIT = re.compile(rf"  - (?P<symbol>{_QUOTED}) \((?P<words>{_QUOTED})\): (?:(?P<stored>the stored unit)|a value in "
                   rf"{_QUOTED} is the value in {_QUOTED} × (?P<scale>{NUMBER})(?: (?P<sign>[+-]) "
                   r"(?P<offset>\d+(?:\.\d+)?))?)\.")


def printed_glossary(prompt: str) -> str | None:
    """The glossary block of a prompt exactly as it prints it, its <glossary> tags included (None for a
    prompt that prints none): what a report shows beside a digest, so that its reader has the words
    and units the model had."""
    found = _GLOSSARY.search(prompt)
    return None if found is None else found.group(0)


def parse_glossary(prompt: str) -> Glossary | None:
    """The glossary a prompt prints (render_glossary), read back from its text alone; None for a
    prompt that prints none, as every legacy (0.2.0) prompt does. A glossary with a line it cannot
    read raises ValueError."""
    found = _GLOSSARY.search(prompt)
    if found is None:
        return None
    lines = found["body"].split("\n")
    if lines[0] != GLOSSARY_TITLE:
        raise ValueError(f"the glossary does not open with its title: {lines[0]!r}")
    words: dict[str, Any] = {"coord_decimals": None}
    fields: list[dict[str, Any]] = []
    for line in lines[1:]:
        if _read_glossary_line(line, words, fields):
            continue
        raise ValueError(f"unreadable glossary line: {line!r}")
    try:
        return Glossary(**words, fields=tuple(GlossaryField(**{**f, "units": tuple(f["units"])}) for f in fields))
    except TypeError as exc:  # a line that every glossary prints is missing
        raise ValueError(f"incomplete glossary ({exc})") from None


def _read_glossary_line(line: str, words: dict[str, Any], fields: list[dict[str, Any]]) -> bool:
    """Read one glossary line into `words` (the Glossary's arguments) or `fields` (each field's
    arguments, its units included); False for a line that is none of the glossary's."""
    match = _UNIT.fullmatch(line)
    if match is not None and fields:
        sign = -1.0 if match["sign"] == "-" else 1.0
        unit = (Unit(json.loads(match["symbol"]), json.loads(match["words"])) if match["stored"] else
                Unit(json.loads(match["symbol"]), json.loads(match["words"]), float(match["scale"]),
                     sign * float(match["offset"]) if match["offset"] else 0.0))
        fields[-1]["units"].append(unit)
        return True
    match = _FIELD.fullmatch(line)
    if match is not None:
        fields.append({"key": json.loads(match["key"]), "label": json.loads(match["label"]),
                       "canonical_unit": None if match["canonical"] is None else json.loads(match["canonical"]),
                       "decimals": int(match["decimals"]), "units": [],
                       "qualifier": None if match["qualifier"] is None else json.loads(match["qualifier"])})
        return True
    match = _RECORDS.fullmatch(line)
    if match is not None:
        words.update(noun=json.loads(match["noun"]), plural=json.loads(match["plural"]),
                     id_label=json.loads(match["id_label"]), id_pattern=match["pattern"])
        return True
    for pattern, name in ((_SOURCES, "source_label"), (_GROUPS, "group_label")):
        match = pattern.fullmatch(line)
        if match is not None:
            words[name] = json.loads(match["label"])
            return True
    match = _STATUSES.fullmatch(line)
    if match is not None:
        words["statuses"] = (json.loads(match["final"]), json.loads(match["draft"]))
        return True
    match = _POSITIONS.fullmatch(line)
    if match is not None:
        words["coord_decimals"] = int(match["decimals"])
        return True
    return False
