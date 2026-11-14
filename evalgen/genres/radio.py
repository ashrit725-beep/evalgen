"""radio: a digest as the transcript of a radio reporting net, for the records of any domain.

A net control station (Control, Dispatch, Net Control or Base) opens the net, and each source's
station calls in under a call sign of its own and reads its reports out, one transmission each, with
the procedure words of the trade ("over", "out", "copy", "roger", "wilco"). A report is read out with
every field, numbers as digits: its ID, status and source, the time of the record, its group and
position, each value with its unit (by symbol, or spoken in the unit's words) and qualifier, a remark
naming the record it repeats, and a location note. Control reads reports back to their station word
for word: a readback repeats a report and is never a new one. Notices are broadcasts by control to all
stations, each saying when it was issued. Chatter lines are announcements, and call-ups, radio checks
and acknowledgements carry no data. The times in the margin, when a transcript has them, are when each
message was sent, never the time of a record.

    RADIO TRANSCRIPT: READINGS NET ON CHANNEL 7
    Session of 13 Apr 2026: reports for the 24 hours up to 2026-04-13T00:00:00Z.
    Times at the left are UTC transmission times.

    00:04:12 KILO 3: Control, Kilo 3, report follows: reading ID by-12345; status reviewed; operator
             ukmo; time of reading 2026-04-12T06:12:19Z; sea area North Sea; position 56.512°N,
             3.104°E; water temperature 12.3 °C (spar); wave height 6.07 feet; remarks same reading
             as by-54321; location Dogger Bank. Over.
    00:04:31 CONTROL: Kilo 3, Control, I read back: reading ID by-12345; ... Over.
    00:04:40 KILO 3: Control, Kilo 3, readback correct, out.
    00:09:02 CONTROL: All stations, all stations, this is Control. Net broadcast: RETRACTION
             issued 2026-04-12T08:30:00Z: reading report by-12399 withdrawn (probe fault); disregard
             it. Control out.

Each transmission is one line (they are wrapped here). Everything is worded from the prompt's glossary
(the record noun, ID label and pattern, source and group labels, statuses, field labels, units and
their words) in the shared formats and with the exact codec of genres.base, so the parser reads any
domain's transcript back through the glossary alone, and every rendering is checked to read back to
exactly its reports and notices (genres.base.check_round_trip). Every broadcast names its kind,
RETRACTION or CORRECTION, which the reading rules of every domain name. A domain whose digests hold
record types of their own (earthquakes) maps them to and from the generic form that this genre writes
(Domain.to_generic and Domain.from_generic).
"""
from __future__ import annotations

import functools
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..digest import MONTHS
from ..domains.base import REVISED_FROM, REVISED_TO, Domain, Glossary, GlossaryField, RecordReport, TrapVocab
from ..models import Notice
from ..ops import DigestData
from ..units import Unit
from .base import (COORDS_PATTERN, NUMBER, TIME_PATTERN, UTC_TIME_FORMATS, Genre, RenderContext, alternation,
                   check_round_trip, domain_items, format_coords, format_time, generic_items, id_pattern, parse_coords,
                   parse_time)
from .classic import RECORD_COORD_PRESETS, RECORD_TIME_PRESETS, UNIT_PRESETS, ZONES

# ------------------------------------------------------------------------------ the genre


class Radio(Genre):
    name = "radio"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One preset per dimension; a variant in this genre avoids every preset of the original's."""

        def pick(key: str, options: Iterable[Any]) -> Any:
            names = sorted(options)
            if avoid is not None and key in avoid:
                names = [n for n in names if n != avoid[key]] or names
            return rng.choice(names)

        style = {"units": pick("units", UNIT_PRESETS), "unit_names": pick("unit_names", UNIT_NAMES),
                 "time": pick("time", RECORD_TIME_PRESETS)}
        if domain.has_location:
            style["coords"] = pick("coords", RECORD_COORD_PRESETS)
        style.update(separator=pick("separator", SEPARATORS), stamps=pick("stamps", STAMPS),
                     calls=pick("calls", CALL_SIGNS), control=pick("control", CONTROL_SIGNS),
                     readbacks=pick("readbacks", READBACKS), layout=pick("layout", LAYOUTS),
                     notices=pick("notices", NOTICE_PLACEMENTS), header=pick("header", range(len(HEADERS))))
        return style

    def render(self, ctx: RenderContext) -> str:
        """The transcript of the context's reports, notices and chatter, in the words of its glossary
        (the domain's own when it has none), checked to read back exactly."""
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _render(ctx, glossary)
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """A transcript opens with one of this genre's titles and holds transmissions that end in a
        procedure word ("over", "out"). No classic digest opens so, legacy or not."""
        lines = digest_text.lstrip().split("\n")
        if not any(pattern.fullmatch(lines[0]) for pattern in _HEADER_LINES):
            return False
        return any(_signed_off(line) for line in lines[1:])

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """Every report and notice, read through the prompt's glossary: report transmissions, and the
        bodies of broadcasts. Readbacks, call-ups, acknowledgements, announcements, radio checks, the
        header and the transmission times change nothing. Without a glossary (a prompt never prints
        none in this genre) the domain's own is used, unless its words are drawn per instance."""
        if glossary is None:
            if domain.has_surface:
                raise ValueError(f"a transcript of {domain.name} records is read through the glossary its prompt "
                                 "prints, and this prompt prints none")
            glossary = domain.glossary()
        return _parse(digest_text, glossary, domain)


RADIO = Radio()
GENRE = RADIO  # what the registry looks up (see evalgen.genres)

# ------------------------------------------------------------------------------- the look
# Units, unit choice, record times and positions are drawn with the presets of the classic genre's
# record digests; the dimensions below are this genre's own. Each style picks one name per dimension.

UNIT_NAMES = {"symbols": 0.0, "mixed": 0.5, "words": 1.0}  # how often a unit is spoken in its words, not its symbol
SEPARATORS = {"semicolon": "; ", "slash": " / ", "comma": ", "}  # between the fields of a readout
STAMPS = ("clock", "zulu", "none")  # transmission times in the margin: 21:04:12, 2104Z, or none
CALL_SIGNS = {  # a station's call sign is one of these words and a number of its own
    "phonetic": ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet", "Kilo",
                 "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform",
                 "Victor", "Whiskey", "Yankee", "Zulu"),
    "unit": ("Unit",),
    "mobile": ("Mobile",),
}
CONTROL_SIGNS = ("Base", "Control", "Dispatch", "Net Control")  # the net control station's call sign
READBACKS = {"always": 1.0, "often": 0.7, "sometimes": 0.4}  # how often control reads a report back
LAYOUTS = ("by_station", "by_time", "shuffled")  # the order reports are passed in
NOTICE_PLACEMENTS = ("end", "interleaved")  # broadcasts after the reports, or among them

HEADERS = (  # the first line of a transcript, which sniff recognises
    "RADIO TRANSCRIPT: {TITLE} NET ON CHANNEL {channel}",
    "Log of the {title} net on channel {channel}",
    "Transcript of radio traffic on the {title} reporting net (channel {channel})",
)
# The session follows the window it reports on (_render), so its line says what the reports are about,
# never that the traffic itself lies in the window: every transmission is sent after it.
SESSION_LINE = "Session of {date}: reports for the {window} up to {compiled}."
STAMP_LINE = "Times at the left are UTC transmission times."

# The wording of procedure. A report's intro and a readback's intro tell the two apart. No intro calls a
# report a "message", which the reading rules say of notices and of what sends, forwards or relays them ("A
# notice is a message about a report", "A WIRE line is a report, not a message"): "message follows" did.
REPORT_INTROS = ("report follows", "{noun} report follows", "{noun} details follow")
EARLIER_REPORT_INTROS = ("message follows",)  # read, never written: so that a run generated before is still read
REPORT_ENDS = (". Over.", ", over.", ". How copy? Over.")
READBACK_INTROS = ("I read back", "reading back", "read back follows")
READBACK_ENDS = (". Over.", ". Is that correct? Over.")
CONFIRMATIONS = ("readback correct", "that is correct", "correct")
ACKNOWLEDGEMENTS = ("copy", "roger", "received")
CALL_UPS = ("traffic for you", "{noun} traffic to pass", "reports to pass")
GO_AHEADS = ("go ahead", "send your traffic", "ready to copy")
# Words that name no station: a net run by Base or Control has no "Dispatch" to broadcast.
BROADCAST_INTROS = ("Net broadcast", "Broadcast message", "General broadcast")
BROADCAST_ACKS = ("copy", "wilco", "roger")
ANNOUNCEMENT_INTROS = ("Information only", "For information", "Net announcement")
SIGNALS = ("loud and clear", "weak but readable", "readable with static")

# A transmission: its time in the margin (if any), the speaker's call sign in capitals, and its message.
STAMP_PATTERN = r"\d{2}:\d{2}:\d{2}|\d{4}Z"
SIGN = ("(?:" + alternation(CONTROL_SIGNS) + "|(?:" + alternation(w for words in CALL_SIGNS.values() for w in words)
        + r") \d+)")
_TRANSMISSION = re.compile(r"(?:(?:" + STAMP_PATTERN + r") )?(?P<speaker>[A-Z][A-Z0-9 ]*): (?P<message>.+)")
_SIGNED_OFF = re.compile(r"(?:[Oo]ver|out)[.?]$")
_BROADCAST = re.compile(rf"All stations, all stations, this is (?P<caller>{SIGN})\. "
                        rf"(?:{alternation(BROADCAST_INTROS)}): (?P<body>.+)\. (?P=caller) out\.")


def _header_pattern(header: str) -> re.Pattern[str]:
    """A header's first line, its title (the records' plural) and channel as wildcards."""
    pieces = re.split(r"(\{\w+\})", header)
    return re.compile("".join(r"\d+" if piece == "{channel}" else ".+?" if piece.startswith("{") else re.escape(piece)
                              for piece in pieces))


_HEADER_LINES = tuple(_header_pattern(header) for header in HEADERS)


def _signed_off(line: str) -> bool:
    """Whether a line is a transmission that ends in a procedure word."""
    match = _TRANSMISSION.fullmatch(line.strip())
    return match is not None and _SIGNED_OFF.search(match["message"]) is not None


def _draw(rng: random.Random, table: Mapping[str, float]) -> str:
    return rng.choices(list(table), weights=list(table.values()), k=1)[0]


def _draw_unit(rng: random.Random, field: GlossaryField, preset: str) -> Unit | None:
    """The unit one value of `field` is spoken in (None for a field without units), as UNIT_PRESETS weighs
    the field's stored unit against the others."""
    if not field.units:
        return None
    stored = UNIT_PRESETS[preset]
    if stored is None or len(field.units) == 1:
        return rng.choice(field.units)
    others = (1.0 - stored) / (len(field.units) - 1)
    weights = [stored if unit.symbol == field.canonical_unit else others for unit in field.units]
    return rng.choices(field.units, weights=weights, k=1)[0]


@functools.lru_cache(maxsize=256)
def _unit_names(field: GlossaryField) -> Mapping[str, Unit]:
    """Every way a transcript may name a unit of `field`: by its symbol, or by its words wherever no
    other unit of the field is named alike (by symbol or by words), so that each name means one unit."""
    names = {unit.symbol: unit for unit in field.units}
    for unit in field.units:
        alike = [u for u in field.units if u != unit and unit.words in (u.symbol, u.words)]
        if not alike:
            names[unit.words] = unit
    return MappingProxyType(names)


def _spoken(field: GlossaryField, value: float, unit: Unit | None, words: bool) -> str:
    """A stored value written in `unit` with the exact codec, then the unit by its words (`words`, where
    they name it alone) or by its symbol: "54.1 degrees Fahrenheit", "12.3 °C", or "4.27" without units. A
    unit's words are its plural ("knots"), and the glossary a transcript is read through names no singular,
    so a value of exactly 1 (or -1) takes its symbol instead: "1 kt", never "1 knots"."""
    text = field.encode(value, unit)
    if unit is None:
        return text
    plural = text.lstrip("-") != "1"
    name = unit.words if words and plural and _unit_names(field).get(unit.words) == unit else unit.symbol
    return f"{text} {name}"


def _separator(preferred: str, words: Iterable[str]) -> str:
    """The separator between the fields of one readout: the style's, unless a word of the readout holds
    it (a region "Honshu, Japan" between commas), then the first other separator that none holds."""
    words = list(words)
    for separator in (preferred, *SEPARATORS.values()):
        if not any(separator in word for word in words):
            return separator
    return preferred


def _window(hours: int) -> str:
    if hours > 24 and hours % 24 == 0:
        return f"{hours // 24} days"
    return f"{hours} hour{'' if hours == 1 else 's'}"


def _stamp(ts: int, fmt: str) -> str:
    utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{utc.hour:02d}{utc.minute:02d}Z" if fmt == "zulu" else f"{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"


# ------------------------------------------------------------------------------- writing


@dataclass
class _Net:
    """One session of the net as it is written: the rendering's stream and style, the words of the
    glossary and the trap vocabulary, and the call signs of control and of each source's station."""

    rng: random.Random
    style: Mapping[str, Any]
    glossary: Glossary
    vocab: TrapVocab
    control: str
    stations: dict[str, str]  # source -> its station's call sign, in source order

    def readout(self, r: RecordReport, offset: int) -> str:
        """Every field of a report, as its station reads it out (and control reads it back)."""
        g, rng, style = self.glossary, self.rng, self.style
        when = format_time(r.time, _draw(rng, RECORD_TIME_PRESETS[style["time"]]), offset)
        parts = [f"{g.id_label} {r.ref}", f"status {r.status}", f"{g.source_label} {r.source}",
                 f"time of {g.noun} {when}", f"{g.group_label} {r.group}"]
        if g.located:
            coords = _draw(rng, RECORD_COORD_PRESETS[style["coords"]])
            parts.append(f"position {format_coords(r.lat, r.lon, coords, g.coord_decimals)}")
        words = [g.id_label, g.source_label, g.noun, g.group_label, r.source, r.group]
        for f in g.fields:
            unit = _draw_unit(rng, f, style["units"])
            value = f"{f.label} {_spoken(f, r.values[f.key], unit, rng.random() < UNIT_NAMES[style['unit_names']])}"
            qualifier = None if f.qualifier is None else r.qualifiers.get(f.qualifier)
            parts.append(value if qualifier is None else f"{value} ({qualifier})")
            words += [f.label, *([] if qualifier is None else [qualifier]), *(u.words for u in f.units)]
        if r.same_as:
            note = rng.choice(self.vocab.duplicate_notes)
            parts.append(f"remarks {note} {r.same_as}")
            words.append(note)
        if r.place is not None:
            parts.append(f"location {r.place}")
            words.append(r.place)
        return _separator(SEPARATORS[style["separator"]], words).join(parts)

    def report(self, source: str, body: str, *, call_up: bool, last: bool) -> list[tuple[str, str]]:
        """A station passing one report: its call-up (when `call_up`), the readout, then control's
        readback and the station's confirmation, or control's acknowledgement. The exchange ends with
        "out" when it is the station's `last` report, else with "over"."""
        rng, control, noun = self.rng, self.control, self.glossary.noun
        station = self.stations[source]
        out = []
        if call_up:
            out += [(station, f"{control}, {station}, {rng.choice(CALL_UPS).format(noun=noun)}, over."),
                    (control, f"{station}, {control}, {rng.choice(GO_AHEADS)}, over.")]
        intro = rng.choice(REPORT_INTROS).format(noun=noun)
        out.append((station, f"{control}, {station}, {intro}: {body}{rng.choice(REPORT_ENDS)}"))
        end = "out" if last else "over"
        if rng.random() < READBACKS[self.style["readbacks"]]:
            readback = f"{rng.choice(READBACK_INTROS)}: {body}{rng.choice(READBACK_ENDS)}"
            out += [(control, f"{station}, {control}, {readback}"),
                    (station, f"{control}, {station}, {rng.choice(CONFIRMATIONS)}, {end}.")]
        else:
            out.append((control, f"{station}, {control}, {rng.choice(ACKNOWLEDGEMENTS)}, {end}."))
        return out

    def broadcast(self, n: Notice) -> list[tuple[str, str]]:
        """A notice as control broadcasts it to all stations, then up to two stations acknowledging it."""
        rng, control = self.rng, self.control
        out = [(control, f"All stations, all stations, this is {control}. {rng.choice(BROADCAST_INTROS)}: "
                         f"{self.notice(n)}. {control} out.")]
        signs = list(self.stations.values())
        for station in rng.sample(signs, min(len(signs), rng.randint(0, 2))):
            out.append((station, f"{control}, {station}, {rng.choice(BROADCAST_ACKS)}, out."))
        return out

    def notice(self, n: Notice) -> str:
        """A broadcast's body: its kind in capitals, when it was issued, and the notice in one of three
        wordings; a retraction may use a phrase of the domain's trap vocabulary, and a correction gives
        both values in one drawn unit, its new value after "revised to", or after "revised from" and the old
        value, which the rules quote (two wordings gave it after "now reads" and "amended from", which they
        do not)."""
        rng, g = self.rng, self.glossary
        form = rng.randrange(3)
        when = format_time(n.time, rng.choice(UTC_TIME_FORMATS))
        if n.kind == "retraction":
            phrase = rng.choice(self.vocab.retraction_notices)
            reason = "" if n.reason is None else f" ({n.reason})"
            return (f"RETRACTION issued {when}: {g.noun} report {n.target} {phrase}{reason}; disregard it",
                    f"RETRACTION issued {when}: {n.target} was not a real {g.noun}{reason}; strike it from the log",
                    f"RETRACTION, time of issue {when}: disregard {g.noun} report {n.target}, {phrase}{reason}")[form]
        field = next((f for f in g.fields if f.key == n.field), None)
        if n.kind != "correction" or field is None or n.new_value is None or n.old_value is None:
            raise ValueError(f"a {n.kind} notice of {n.target} that names no field of the glossary, or no new and old "
                             "value, cannot be broadcast")
        unit = _draw_unit(rng, field, self.style["units"])
        words = rng.random() < UNIT_NAMES[self.style["unit_names"]]
        new, old = _spoken(field, n.new_value, unit, words), _spoken(field, n.old_value, unit, words)
        qualifier = "" if n.qualifier is None else f" ({n.qualifier})"
        return (f"CORRECTION issued {when}: the {field.label} of {n.target} is revised to {new}{qualifier}; it was "
                f"first given as {old}",
                f"CORRECTION, time of issue {when}: for {n.target}, {field.label} {REVISED_TO} {new}{qualifier}, "
                f"not {old}",
                f"CORRECTION issued {when}: {n.target} {field.label} {REVISED_FROM} {old} to {new}{qualifier}")[form]

    def opening(self) -> tuple[str, str]:
        control = self.control
        return control, (f"All stations, all stations, this is {control}. The {self.glossary.plural} net is open; "
                         "stations with traffic, call in. Over.")

    def closing(self) -> tuple[str, str]:
        control = self.control
        return control, f"All stations, this is {control}. No further traffic; the net is closed. {control} out."

    def announcement(self, line: str) -> list[tuple[str, str]]:
        """A chatter line, which carries no data, announced by control."""
        control = self.control
        return [(control, f"All stations, this is {control}. {self.rng.choice(ANNOUNCEMENT_INTROS)}: {line} "
                          f"{control} out.")]

    def radio_check(self) -> list[tuple[str, str]]:
        rng, control = self.rng, self.control
        station = rng.choice(list(self.stations.values()))
        return [(station, f"{control}, {station}, radio check, over."),
                (control, f"{station}, {control}, {rng.choice(SIGNALS)}, over."),
                (station, f"{control}, {station}, roger, out.")]


def _render(ctx: RenderContext, glossary: Glossary) -> str:
    """The transcript of a context's reports, notices and chatter, as `ctx.style` draws it."""
    rng, style, domain = ctx.rng, ctx.style, ctx.domain
    reports: list[RecordReport] = generic_items(domain, ctx.reports)  # each place as genres print it
    notices: list[Notice] = generic_items(domain, ctx.notices)
    sources = sorted({r.source for r in reports})
    numbers = rng.sample(range(1, max(100, 2 * len(sources))), len(sources))  # one call sign per source
    words = CALL_SIGNS[style["calls"]]
    stations = {source: f"{rng.choice(words)} {number}" for source, number in zip(sources, numbers)}
    net = _Net(rng, style, glossary, domain.trap_vocab, style["control"], stations)
    zones = {} if glossary.located else {source: rng.randint(*ZONES) for source in sources}
    readouts = [(r, net.readout(r, domain.utc_offset(r) if glossary.located else zones[r.source])) for r in reports]

    exchanges: list[list[tuple[str, str]]] = []
    if style["layout"] == "by_station":
        order = list(sources)
        rng.shuffle(order)
        for source in order:
            block = [body for r, body in readouts if r.source == source]
            rng.shuffle(block)
            exchanges.append([line for i, body in enumerate(block)
                              for line in net.report(source, body, call_up=i == 0, last=i == len(block) - 1)])
    else:
        if style["layout"] == "by_time":
            readouts.sort(key=lambda pair: (pair[0].time, pair[0].ref, pair[0].source))
        else:
            rng.shuffle(readouts)
        for r, body in readouts:
            exchanges.append(net.report(r.source, body, call_up=rng.random() < 0.5, last=True))

    broadcasts = [net.broadcast(n) for n in notices]
    if style["notices"] == "interleaved":
        for exchange in broadcasts:
            exchanges.insert(rng.randrange(len(exchanges) + 1), exchange)
    elif broadcasts:
        rng.shuffle(broadcasts)
        exchanges.append([(net.control, f"All stations, this is {net.control}. Stand by for broadcasts, over.")])
        exchanges += broadcasts
    for line in ctx.chatter:
        exchanges.insert(rng.randrange(len(exchanges) + 1), net.announcement(line))
    for _ in range(rng.randint(0, 2) if stations else 0):
        exchanges.insert(rng.randrange(len(exchanges) + 1), net.radio_check())

    start = ctx.compiled_at + rng.randint(60, 1200)  # the session follows the window it reports on
    clock, lines = start, []
    for speaker, message in [net.opening(), *(t for exchange in exchanges for t in exchange), net.closing()]:
        margin = "" if style["stamps"] == "none" else f"{_stamp(clock, style['stamps'])} "
        lines.append(f"{margin}{speaker.upper()}: {message}")
        clock += rng.randint(4, 30) + len(message) // 15  # a long message takes longer to say

    day = datetime.fromtimestamp(start, tz=timezone.utc)
    header = [HEADERS[style["header"]].format(TITLE=glossary.plural.upper(), title=glossary.plural,
                                              channel=rng.randint(1, 16)),
              SESSION_LINE.format(date=f"{day.day} {MONTHS[day.month - 1]} {day.year}",
                                  window=_window(ctx.window_hours),
                                  compiled=format_time(ctx.compiled_at, rng.choice(UTC_TIME_FORMATS)))]
    if style["stamps"] != "none":
        header.append(STAMP_LINE)
    return "\n".join(header + [""] + lines)


# ------------------------------------------------------------------------------- reading


@dataclass(frozen=True)
class _Grammar:
    """The patterns of one glossary's transcripts: the message of a report transmission and of a
    readback, and the bodies of broadcasts (with the field a correction revises; None: a retraction)."""

    report: re.Pattern[str]
    readback: re.Pattern[str]
    notices: tuple[tuple[re.Pattern[str], GlossaryField | None], ...]


def _value_pattern(f: GlossaryField, name: str) -> str:
    """A value of `f` as _spoken writes it: its number in the group `name` and, for a field with units,
    the unit's name (a symbol, or words that name one unit alone) in the group `name`_unit."""
    number = f"(?P<{name}>{NUMBER})"
    return number if not f.units else f"{number} (?P<{name}_unit>{alternation(_unit_names(f))})"


def _read(f: GlossaryField, match: re.Match[str], name: str) -> float:
    """The stored value that a match of _value_pattern(f, name) stands for."""
    return f.decode(match[name], _unit_names(f)[match[f"{name}_unit"]] if f.units else None)


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    record_id, noun = id_pattern(glossary), re.escape(glossary.noun)
    # A readout's fields come in a fixed order, each opening with its label, and the separator after
    # the ID is the one between every other pair of fields. Times, positions and values are matched by
    # their own patterns, so the free words between them (a source, a group, a place) backtrack until
    # the real slots fit.
    fields = "".join(rf"(?P=sep){re.escape(f.label)} {_value_pattern(f, f'v{i}')}"
                     + ("" if f.qualifier is None else rf"(?: \((?P<q{i}>[^()]*)\))?")
                     for i, f in enumerate(glossary.fields))
    body = (rf"{re.escape(glossary.id_label)} (?P<ref>{record_id})(?P<sep>{alternation(SEPARATORS.values())})"
            rf"status (?P<status>{alternation(glossary.statuses)})"
            rf"(?P=sep){re.escape(glossary.source_label)} (?P<source>.+?)"
            rf"(?P=sep)time of {noun} (?P<time>{TIME_PATTERN})"
            rf"(?P=sep){re.escape(glossary.group_label)} (?P<group>.+?)"
            + (rf"(?P=sep)position (?P<coords>{COORDS_PATTERN})" if glossary.located else "") + fields
            + rf"(?:(?P=sep)remarks (?P<note>.+?) (?P<same>{record_id}))?(?:(?P=sep)location (?P<place>.+?))?")
    intros = alternation(intro.format(noun=glossary.noun) for intro in REPORT_INTROS + EARLIER_REPORT_INTROS)
    report = re.compile(rf"(?P<callee>{SIGN}), (?P<caller>{SIGN}), (?:{intros}): {body}(?:{alternation(REPORT_ENDS)})")
    readback = re.compile(rf"(?P<callee>{SIGN}), (?P<caller>{SIGN}), (?:{alternation(READBACK_INTROS)}): .+")

    when, reason = f"(?P<time>{TIME_PATTERN})", r"(?: \((?P<reason>.+)\))?"
    notices: list[tuple[re.Pattern[str], GlossaryField | None]] = [(re.compile(pattern), None) for pattern in (
        rf"RETRACTION issued {when}: {noun} report (?P<ref>{record_id}) (?P<phrase>.+?){reason}; disregard it",
        rf"RETRACTION issued {when}: (?P<ref>{record_id}) was not a real {noun}{reason}; strike it from the log",
        rf"RETRACTION, time of issue {when}: disregard {noun} report (?P<ref>{record_id}), (?P<phrase>.+?){reason}")]
    for f in glossary.fields:
        label, new, old = re.escape(f.label), _value_pattern(f, "new"), _value_pattern(f, "old")
        qualifier = "" if f.qualifier is None else r"(?: \((?P<qualifier>[^()]*)\))?"
        ref = f"(?P<ref>{record_id})"
        # "now reads" and "amended from": as transcripts printed them before, so that a run generated then is read
        for pattern in (rf"CORRECTION issued {when}: the {label} of {ref} is revised to {new}{qualifier}; it was first "
                        rf"given as {old}",
                        rf"CORRECTION, time of issue {when}: for {ref}, {label} (?:{REVISED_TO}|now reads) "
                        rf"{new}{qualifier}, not {old}",
                        rf"CORRECTION issued {when}: {ref} {label} (?:{REVISED_FROM}|amended from) {old} to "
                        rf"{new}{qualifier}"):
            notices.append((re.compile(pattern), f))
    return _Grammar(report, readback, tuple(notices))


def _parse(text: str, glossary: Glossary, domain: Domain) -> DigestData:
    """Every report and notice of a transcript, in the form the domain's digests hold them."""
    grammar = _grammar(glossary)
    reports: list[RecordReport] = []
    notices: list[Notice] = []
    for raw in text.splitlines():
        transmission = _TRANSMISSION.fullmatch(raw.strip())
        if transmission is None:  # the header, or a blank line
            continue
        message = transmission["message"]
        if grammar.readback.fullmatch(message):  # control reading a report back: a repeat, never a new report
            continue
        match = grammar.report.fullmatch(message)
        if match is not None:
            report = _report(match, glossary)
            if report is not None:
                reports.append(report)
            continue
        match = _BROADCAST.fullmatch(message)
        if match is not None:
            notice = _notice(match["body"], grammar)
            if notice is not None:
                notices.append(notice)
    return DigestData(domain_items(domain, reports), domain_items(domain, notices))


def _report(match: re.Match[str], glossary: Glossary) -> RecordReport | None:
    """The report a readout gives; None when its time or position is none."""
    when = parse_time(match["time"])
    position = parse_coords(match["coords"], glossary.coord_decimals) if glossary.located else (None, None)
    if when is None or position is None:
        return None
    values = {f.key: _read(f, match, f"v{i}") for i, f in enumerate(glossary.fields)}
    qualifiers = {f.qualifier: match[f"q{i}"] for i, f in enumerate(glossary.fields)
                  if f.qualifier is not None and match[f"q{i}"] is not None}
    return RecordReport(ref=match["ref"], source=match["source"], status=match["status"], time=when, lat=position[0],
                        lon=position[1], group=match["group"], values=values, qualifiers=qualifiers,
                        place=match["place"], same_as=match["same"])


def _notice(body: str, grammar: _Grammar) -> Notice | None:
    """The notice a broadcast's body gives; None for a body that is no notice."""
    for pattern, field in grammar.notices:
        match = pattern.fullmatch(body)
        if match is None:
            continue
        when = parse_time(match["time"])
        if when is None:
            return None
        if field is None:
            return Notice(kind="retraction", target=match["ref"], time=when, reason=match["reason"])
        return Notice(kind="correction", target=match["ref"], time=when, field=field.key,
                      new_value=_read(field, match, "new"), old_value=_read(field, match, "old"),
                      qualifier=match.groupdict().get("qualifier"))
    return None
