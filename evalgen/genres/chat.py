"""chat: the export of a channel's chat log, for earthquakes and for the records of every other domain.

Several people and a feed bot relay, in one channel, what each source files. The digest is the
channel's log: a header that names the channel and the clock its message times are in, then
"[HH:MM] user: message" lines in posting order, with a line between days. Among the messages:

    reports   each in one of four voices (below), with every field of its report;
    notices   retractions and corrections, posted as messages, posted and then pinned, or listed on
              the channel's pinned board above the log ("📌 Pinned by user: ...");
    noise     joins, greetings, acknowledgements, off-topic remarks and the instance's chatter lines,
              and repeats of earlier messages: quotes ("> " and an earlier report or notice, word for
              word) and read-backs ("to confirm: " and an earlier report, retyped in another voice).

A message's time stamp is when it was posted, in the header's clock (UTC, or a stated offset), and
never the time of a record: every report states its record's time in full, to the second, after a
delay or in a later batch. The reading rules of every record domain (domains.base.record_rules) say
that the time a message was sent is never a record's time, that a quoted or read-back repeat is not a
new report or notice and that lines which are neither change nothing; a repeat carries the ID and the
values of the message it repeats, so it changes nothing under the 0.2.0 earthquake rules either.
Every notice wording says RETRACTION or CORRECTION, which both sets of rules name, and quotes the
domain's own phrases (its trap vocabulary) where they fit; a correction gives its new value after
"revised to", "revised from <old> to" or a phrase of the domain's, which the rules quote. A re-report's
note is one of the domain's duplicate notes, or, in a paste, the field same_as=, which rule 1 quotes.

The four voices of a report, here a re-report in the words of the test-only toy domain:

    relay   relaybot: emsn · reading by-54321 · preliminary · North Sea · 56.512°N, 3.104°E · water
            temperature 12.3 °C (spar) · wave height 1.85 m · recorded 2026-04-12T06:12:19Z · same
            reading as by-12345 · place: Dogger Bank
    casual  mira: fyi: emsn reports reading by-54321 (preliminary) in North Sea at (56.512, 3.104),
            recorded 06:12:19 UTC on 12 April 2026: water temperature 54.1 °F (spar), wave height
            6.07 ft; same reading as by-12345. Place: Dogger Bank.
    terse   tomh: by-54321 (emsn, preliminary) — North Sea — 56.512°N, 3.104°E — 2026-04-12 06:12:19
            UTC — water temperature 12.3 °C (spar); wave height 1.85 m — duplicate of by-12345 — place:
            Dogger Bank
    paste   dana_k: `2026-04-12T06:12:19+00:00 ref=by-54321 src=emsn status=preliminary group="North
            Sea" loc=56.512,3.104 temp=12.3°C probe=spar wave=1.85m same_as=by-12345 place="Dogger
            Bank"`

(each on one line; wrapped here). Notices read, for instance, "RETRACTION issued 2026-04-12T08:02:11Z:
by-77777 withdrawn (probe fault). Please disregard it." or "CORRECTION [2026-04-12 07:40:00 UTC]: the
water temperature of by-12345 is revised to 12.3 °C (spar); it was first given as 9.8 °C."

Everything a voice prints comes from the prompt's glossary (record noun, field keys and labels, units,
qualifiers, statuses, ID pattern, position precision) and the domain's trap vocabulary, written with
the shared formats and the exact codec of genres.base, so that the parser reads every report and
notice back exactly from the text and the glossary alone. Each message is checked to read back as
it is written (a report whose words a voice cannot print unambiguously is pasted instead, its words
as quoted tokens), and so is every rendering (genres.base.check_round_trip). Earthquakes keep their
0.2.0 record types, a Report and a correction that revises the magnitude: they are printed in their
generic form (Domain.to_generic) and read back through Domain.from_generic, as in every other genre.
"""
from __future__ import annotations

import functools
import json
import random
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from ..digest import MONTHS, WEEKDAYS
from ..domains.base import SAME_AS_KEY, Domain, Glossary, GlossaryField, RecordReport, TrapVocab
from ..domains.earthquakes import Earthquakes
from ..models import Notice
from ..ops import DigestData
from ..units import Unit, decode, encode
from .base import (COORDS_PATTERN, GROUP_PREPOSITION, NUMBER, TIME_FORMATS, TIME_PATTERN, UTC_TIME_FORMATS, Genre,
                   RenderContext, alternation, check_round_trip, domain_items, format_coords, format_time,
                   format_value, generic_items, id_pattern, parse_coords, parse_time, read_value, value_pattern)

# ------------------------------------------------------------------------------ the genre


class Chat(Genre):
    name = "chat"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Pick one preset per dimension of the look (a variant in this genre avoids the original's
        presets), and one of the headers."""

        def pick(key: str, options: Sequence[str] | Mapping[str, Any]) -> str:
            names = sorted(options)
            if avoid is not None:
                names = [n for n in names if n != avoid.get(key)] or names
            return rng.choice(names)

        style = {"units": pick("units", UNIT_PRESETS), "time": pick("time", TIME_PRESETS)}
        if domain.has_location:
            style["coords"] = pick("coords", COORD_PRESETS)
        style.update(voices=pick("voices", VOICE_PRESETS), pace=pick("pace", PACES), clock=pick("clock", CLOCKS),
                     notices=pick("notices", NOTICE_PLACES), noise=pick("noise", NOISE_LEVELS),
                     header=rng.randrange(len(HEADERS)))
        return style

    def render(self, ctx: RenderContext) -> str:
        """The channel's log, in the words of the context's glossary (the domain's own when it has
        none), checked to read back exactly."""
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _render_log(ctx, glossary)
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """A chat digest opens with one of its headers, and every other line is a message, a system
        line, a day line or a pinned notice; at least one is a message."""
        lines = [line.strip() for line in digest_text.strip().splitlines()]
        if not lines or not any(pattern.fullmatch(lines[0]) for pattern in _HEADER_LINES):
            return False
        rest = [line for line in lines[1:] if line]
        return (any(_MESSAGE.fullmatch(line) for line in rest)
                and all(any(kind.fullmatch(line) for kind in _LINE_KINDS) for line in rest))

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """Every report and notice exactly, read through the prompt's glossary (an earthquake digest in
        a prompt without one through the domain's own), as the domain's digests hold them."""
        if glossary is None:
            if not isinstance(domain, Earthquakes):
                raise ValueError(f"a digest of {domain.name} records is read through the glossary its prompt prints, "
                                 "and this prompt prints none")
            glossary = domain.glossary()
        data = _parse_log(digest_text, glossary)
        return DigestData(domain_items(domain, data.reports), domain_items(domain, data.notices))


CHAT = Chat()
GENRE = CHAT  # what the registry looks up (see evalgen.genres)

# ----------------------------------------------------------------------- the look of a chat log
# A rendering's style picks one preset per dimension; each preset of the first ones is a weighted mix.

VOICES = ("relay", "casual", "terse", "paste")
VOICE_PRESETS = {
    "bots": {"relay": 0.55, "casual": 0.15, "terse": 0.15, "paste": 0.15},
    "people": {"relay": 0.1, "casual": 0.4, "terse": 0.3, "paste": 0.2},
    "mixed": {voice: 1.0 for voice in VOICES},
}
TIME_PRESETS = {  # the time each report gives its record (notices: a UTC format)
    "utc_heavy": {"iso": 0.3, "iso_space": 0.25, "rfc": 0.1, "prose_utc": 0.2, "iso_offset": 0.05, "local12": 0.05,
                  "local24": 0.05},
    "local_heavy": {"iso": 0.1, "iso_space": 0.1, "rfc": 0.05, "prose_utc": 0.05, "iso_offset": 0.25, "local12": 0.25,
                    "local24": 0.2},
    "mixed": {fmt: 1.0 for fmt in TIME_FORMATS},
}
COORD_PRESETS = {"hemi": {"hemi": 0.75, "signed": 0.25}, "signed": {"hemi": 0.25, "signed": 0.75},
                 "mixed": {"hemi": 0.5, "signed": 0.5}}
# The weight of a field's stored unit in each preset; the others share the rest (None: all alike).
UNIT_PRESETS: dict[str, float | None] = {"mostly_stored": 0.75, "mostly_other": 0.15, "mixed": None}
PACES = ("live", "batched")  # reports posted soon after their records, or in a few catch-up batches
CLOCKS = ("utc", "local")  # the clock of the message times: UTC, or a stated offset
NOTICE_PLACES = ("posted", "pinned", "board")  # a notice is a message, a message then pinned, or on the board
NOISE_LEVELS = {  # acknowledgements per report or notice, off-topic remarks per report, quotes, read-backs
    "quiet": {"ack": 0.12, "off_topic": 0.08, "quotes": (1, 2), "readbacks": (0, 1)},
    "chatty": {"ack": 0.35, "off_topic": 0.25, "quotes": (2, 4), "readbacks": (1, 2)},
}
HEADERS = (
    "Chat log of #{channel}: the {window} up to {compiled}. Message times are {clock}.",
    "#{channel} — exported {compiled} ({window} of history; message times in {clock})",
    "--- Log opened for #{channel}: the {window} up to {compiled} --- message times: {clock}",
)
CHANNEL_SUFFIXES = ("desk", "duty", "feed", "ops", "relay", "watch")
PEOPLE = ("ade", "amaru", "bea", "dana_k", "felix", "ines", "jo.s", "kwame", "lin_w", "marta", "mira", "nora",
          "oksana", "pablo_r", "ravi", "sasha", "tomh", "yuki")
BOTS = ("feedbot", "ingest-bot", "relaybot", "watchbot")
LOCAL_OFFSETS = tuple(hours for hours in range(-12, 15) if hours)  # a local message clock: never UTC itself
ZONES = (-12, 14)  # the UTC offsets, in whole hours, of sources whose records have no position
LEAD_RATE = 0.3  # the share of people's reports that open with a lead-in
MENTION_RATE = 0.3  # the share of acknowledgements of a person's message that name them ("@mira thx")

# Words without data. None of them holds an ID, a number, a time or a notice keyword.
GREETINGS = ("hi all", "hello", "hey team", "o/", "back online")
ACKS = ("thx", "thanks", "ty", "noted", "ok", "got it", "seen", "on it", "nice", "cheers", "👍", "ack")
OFF_TOPIC = ("coffee run, anyone?", "brb", "back", "lunch?", "anyone else's dashboard frozen?",
             "the wiki is down again", "who has the meeting room booked?", "my VPN keeps dropping",
             "printer's jammed again", "is the standup still on?", "remember to log your hours",
             "new coffee beans in the kitchen", "can someone water the plant?", "fire drill later, apparently")
QUOTE_COMMENTS = ("noted", "thanks for flagging", "good catch", "adding it to the sheet", "seen, thanks", "ok, got it")
LEADS = ("fyi: ", "new: ", "just in: ", "another one: ", "for the log: ")  # how people may open a report
QUOTE = "> "  # a quote: "> " and an earlier message, word for word
READBACKS = ("to confirm: ", "reading back: ", "confirming: ")  # a read-back: an earlier report, retyped
REPEATS = (QUOTE, *READBACKS)  # how every repeat of an earlier message opens
PIN = "\U0001f4cc"  # 📌, before each notice on the pinned board

# Notice wordings. Each says RETRACTION or CORRECTION; a retraction may quote one of the domain's
# retraction phrases, and the last correction wording a correction phrase that leads to the new
# value ("revised to", "corrected to"). Each opens with its keyword and the time it was issued (after
# "heads up: " at most), so that it reads as the notice itself: never as a pointer to a notice
# printed elsewhere ("see the RETRACTION issued ..."), which the log never holds, nor as a relay of
# one, which a reader might take for a repeat (rule 7).
RETRACTIONS = (
    "RETRACTION [{when}]: {ref} was not a real {noun}{reason}; ignore that report.",
    "RETRACTION issued {when}: {ref} {phrase}{reason}. Please disregard it.",
    "heads up: RETRACTION issued {when}: {ref} {phrase}{reason}",
)
CORRECTIONS = (
    "CORRECTION [{when}]: the {label} of {ref} is revised to {new}{qualifier}; it was first given as {old}.",
    "CORRECTION issued {when}: {ref} {label} revised from {old} to {new}{qualifier}.",
    "heads up: CORRECTION issued {when}: {label} for {ref} revised to {new}{qualifier} (was {old})",
    "CORRECTION issued {when}: the {label} of {ref} {phrase} {new}{qualifier}, first given as {old}.",
)
PHRASED_CORRECTION = CORRECTIONS[-1]

# ------------------------------------------------------------------------- the lines of a log

NICK = r"[A-Za-z][A-Za-z0-9._-]*"
_STAMP = r"\[\d{2}:\d{2}\] "
_MESSAGE = re.compile(rf"{_STAMP}(?P<nick>{NICK}): (?P<body>.+)")
_SYSTEM = re.compile(rf"{_STAMP}\* {NICK} .+")  # joins, pins
_DAY = re.compile(rf"--- (?:{'|'.join(WEEKDAYS)}) \d{{1,2}} (?:{'|'.join(MONTHS)}) \d{{4}} ---")
_PINNED = re.compile(rf"{PIN} Pinned by (?P<nick>{NICK}): (?P<body>.+)")
_LINE_KINDS = (_MESSAGE, _SYSTEM, _DAY, _PINNED)
CHANNEL = r"[^\W_]+(?:-[^\W_]+)*"


def _header_pattern(header: str) -> re.Pattern[str]:
    """A header's line: its channel as a channel name, its other placeholders as wildcards."""
    pieces = re.split(r"(\{\w+\})", header)
    return re.compile("".join(CHANNEL if piece == "{channel}" else ".+" if piece.startswith("{")
                              else re.escape(piece) for piece in pieces))


_HEADER_LINES = tuple(_header_pattern(header) for header in HEADERS)


def _channel(plural: str, suffix: str) -> str:
    """A channel named after the records: "#readings-desk", "#log-entries-watch"."""
    slug = "-".join(re.findall(r"[^\W_]+", plural.lower())) or "records"
    return f"{slug}-{suffix}"


def _window(hours: int) -> str:
    if hours > 24 and hours % 24 == 0:
        return f"{hours // 24} days"
    return f"{hours} hour{'' if hours == 1 else 's'}"


def _clock(offset: int) -> str:
    """The message clock as the header names it: "UTC", "UTC+02:00"."""
    return "UTC" if offset == 0 else f"UTC{'+' if offset > 0 else '-'}{abs(offset):02d}:00"


def _local(ts: int, offset: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=offset)


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


# --------------------------------------------------------------------- 0.2.0 earthquake records
# Voices print, and the parser reads, the generic RecordReport and Notice. An earthquake domain keeps
# the 0.2.0 Report and the 0.2.0 Notice, whose correction revises the magnitude: they are printed in their
# generic form (Domain.to_generic) and read back through Domain.from_generic (genres.base.domain_items),
# as in every other genre.


def _generic_report(domain: Domain, r: Any) -> RecordReport:
    """A report as every voice prints it, in its generic form (its place as genres print it, printed_place)
    and without the generator's bookkeeping (role, trap), so that it compares equal to the report its
    message reads back to."""
    return replace(generic_items(domain, [r])[0], role="primary", trap_id=None)


def _generic_notice(domain: Domain, n: Notice) -> Notice:
    """A notice as every wording prints it, in its generic form (a 0.2.0 correction as one of the
    magnitude field) and without its trap."""
    return replace(domain.to_generic(n), trap_id=None)


# ------------------------------------------------------------------------------ writing a log


@dataclass(frozen=True)
class _Printed:
    """How one report is printed: its voice and lead-in, and the words drawn for it."""

    voice: str
    lead: str  # "" or one of LEADS (never in the relay voice: the bot has none)
    when: str  # the record's time, in a drawn format
    where: str | None  # its position, in a drawn format (None: records have none)
    units: tuple[Unit | None, ...]  # the unit of each field's value, in glossary order
    note: str | None  # the duplicate note before same_as
    preposition: str = "in"  # how the casual voice places a record in its group (Domain.group_preposition)


@dataclass
class _Message:
    """One line of the log, before its time stamp: "nick: body", or a system line ("* nick joined ...")."""

    at: int  # when it was posted (Unix seconds, UTC)
    line: str
    nick: str | None = None
    body: str | None = None  # a report's or a notice's message, which a quote may repeat
    printed: tuple[RecordReport, _Printed] | None = None  # a report's message, which a read-back may retype


class _Schedule:
    """When reports and notices are posted: soon after what they tell (live), or at the first of a few
    catch-up batches that follows it (batched), never after the log ends."""

    def __init__(self, pace: str, rng: random.Random, start: int, end: int) -> None:
        self.rng, self.end = rng, end
        self.batches = sorted(rng.randrange(start, end) for _ in range(rng.randint(3, 6))) if pace == "batched" else []

    def after(self, t: int) -> int:
        later = [batch for batch in self.batches if batch >= t]
        at = later[0] if later else t + self.rng.randint(20, 1200)
        return min(self.end, at + self.rng.randint(0, 90))


def _render_log(ctx: RenderContext, glossary: Glossary) -> str:
    """The log of a digest's reports, notices and chatter, as `ctx.style` draws it."""
    rng, style, vocab = ctx.rng, ctx.style, ctx.domain.trap_vocab
    grammar = _grammar(glossary)
    end = ctx.compiled_at
    start = end - ctx.window_hours * 3600
    channel = _channel(glossary.plural, rng.choice(CHANNEL_SUFFIXES))
    people = rng.sample(PEOPLE, rng.randint(3, 5))
    bot = rng.choice(BOTS)
    offset = 0 if style["clock"] == "utc" else rng.choice(LOCAL_OFFSETS)
    sources = sorted({r.source for r in ctx.reports})
    zones = {} if glossary.located else {source: rng.randint(*ZONES) for source in sources}
    noise = NOISE_LEVELS[style["noise"]]
    schedule = _Schedule(style["pace"], rng, start, end)
    log: list[_Message] = []

    def acknowledge(message: _Message) -> None:  # now and then, someone acknowledges a message, by name or not
        if rng.random() < noise["ack"]:
            nick = _someone_else(rng, people, message.nick)
            mention = f"@{message.nick} " if message.nick in people and rng.random() < MENTION_RATE else ""
            log.append(_Message(min(end, message.at + rng.randint(5, 300)), f"{nick}: {mention}{rng.choice(ACKS)}",
                                nick))

    for r in ctx.reports:
        report = _generic_report(ctx.domain, r)
        printed = _print_report(report, rng, style, glossary, vocab, zones, ctx.domain)
        body = _report_body(report, printed, glossary)
        if not _reads_back(body, report, grammar, glossary):  # words this voice cannot print unambiguously
            printed = replace(printed, voice="paste", lead="")
            body = _report_body(report, printed, glossary)
        nick = bot if printed.voice == "relay" else rng.choice(people)
        message = _Message(schedule.after(report.time), f"{nick}: {body}", nick, body, (report, printed))
        log.append(message)
        acknowledge(message)

    board = []
    for n in ctx.notices:
        notice = _generic_notice(ctx.domain, n)
        body = _notice_body(notice, rng, glossary, vocab, style["units"], grammar)
        nick = rng.choice([*people, bot])
        if style["notices"] == "board":
            board.append(f"{PIN} Pinned by {nick}: {body}")
            continue
        message = _Message(schedule.after(notice.time), f"{nick}: {body}", nick, body)
        log.append(message)
        if style["notices"] == "pinned":
            pinner, pinned = rng.choice(people), min(end, message.at + rng.randint(5, 300))
            log.append(_Message(pinned, f"* {pinner} pinned a message to #{channel}", pinner))
        acknowledge(message)

    _add_repeats(log, rng, people, noise, glossary, end)
    for _ in range(max(1, round(noise["off_topic"] * len(ctx.reports)))):
        nick = rng.choice(people)
        log.append(_Message(rng.randrange(start, end), f"{nick}: {rng.choice(OFF_TOPIC)}", nick))
    for i, text in enumerate(ctx.chatter):  # a line that tells of a window is posted after it (timed_chatter)
        nick = rng.choice([*people, bot])
        after = ctx.chatter_after[i] if i < len(ctx.chatter_after) else None
        earliest = start if after is None else min(max(start, after + 60), end - 1)
        log.append(_Message(rng.randrange(earliest, end), f"{nick}: {text}", nick))
    for nick in rng.sample(people, rng.randint(1, 2)):  # joins, each before the newcomer's first line
        first = min((message.at for message in log if message.nick == nick), default=end)
        joined = min(max(start, first - rng.randint(60, 1800)), first - 1)
        log.append(_Message(joined, f"* {nick} joined #{channel}", nick))
        if rng.random() < 0.7:
            log.append(_Message(joined, f"{nick}: {rng.choice(GREETINGS)}", nick))

    log.sort(key=lambda message: message.at)  # stable: a reply never comes before what it answers
    rng.shuffle(board)
    header = HEADERS[style["header"]].format(channel=channel, window=_window(ctx.window_hours),
                                             compiled=format_time(end, rng.choice(UTC_TIME_FORMATS)),
                                             clock=_clock(offset))
    lines = [header, ""] + (board + [""] if board else [])
    day = None
    for message in log:
        local = _local(message.at, offset)
        if local.date() != day:
            day = local.date()
            lines.append(f"--- {WEEKDAYS[local.weekday()]} {local.day} {MONTHS[local.month - 1]} {local.year} ---")
        lines.append(f"[{local.hour:02d}:{local.minute:02d}] {message.line}")
    return "\n".join(lines)


def _someone_else(rng: random.Random, people: Sequence[str], nick: str | None) -> str:
    return rng.choice([person for person in people if person != nick] or list(people))


def _print_report(r: RecordReport, rng: random.Random, style: Mapping[str, Any], glossary: Glossary, vocab: TrapVocab,
                  zones: Mapping[str, int], domain: Domain) -> _Printed:
    """Draw how one report is printed: its voice, lead-in, time and position formats, units and note.
    Its time is in the local time of its position, as its domain gives it (Domain.utc_offset), or of its
    source when records have none; the casual voice places it in its group with the domain's preposition."""
    voice = _draw(rng, VOICE_PRESETS[style["voices"]])
    lead = rng.choice(LEADS) if voice != "relay" and rng.random() < LEAD_RATE else ""
    offset = domain.utc_offset(r) if glossary.located else zones[r.source]
    preposition = domain.group_preposition
    when = format_time(r.time, _draw(rng, TIME_PRESETS[style["time"]]), offset)
    where = (format_coords(r.lat, r.lon, _draw(rng, COORD_PRESETS[style["coords"]]), glossary.coord_decimals)
             if glossary.located else None)
    units = tuple(_draw_unit(rng, f, style["units"]) for f in glossary.fields)
    note = rng.choice(vocab.duplicate_notes) if r.same_as else None
    return _Printed(voice, lead, when, where, units, note, preposition)


def _report_body(r: RecordReport, p: _Printed, glossary: Glossary) -> str:
    """One report as a message in its voice (see the module docstring)."""
    if p.voice == "paste":
        text = f"{p.when} ref={r.ref} src={_token(r.source)} status={r.status} group={_token(r.group)}"
        if glossary.located:
            text += f" loc={encode(r.lat, glossary.coord_decimals)},{encode(r.lon, glossary.coord_decimals)}"
        for f, unit in zip(glossary.fields, p.units):
            text += f" {f.key}={format_value(f, r.values[f.key], unit, separator='')}"
            if _qualifier(f, r) is not None:
                text += f" {f.qualifier}={_token(_qualifier(f, r))}"
        if r.same_as:  # the note of a re-report, as rule 1 of the reading rules quotes it
            text += f" {SAME_AS_KEY}={r.same_as}"
        if r.place is not None:
            text += f" place={_token(r.place)}"
        return f"{p.lead}`{text}`"
    values = [f"{f.label} {format_value(f, r.values[f.key], unit)}"
              + ("" if _qualifier(f, r) is None else f" ({_qualifier(f, r)})")
              for f, unit in zip(glossary.fields, p.units)]
    where = [] if p.where is None else [p.where]
    if p.voice == "relay":
        parts = [r.source, f"{glossary.noun} {r.ref}", r.status, r.group, *where, *values, f"recorded {p.when}"]
        parts += [f"{p.note} {r.same_as}"] if r.same_as else []
        parts += [] if r.place is None else [f"place: {r.place}"]
        return " · ".join(parts)
    if p.voice == "casual":
        text = (f"{p.lead}{r.source} reports {glossary.noun} {r.ref} ({r.status}) {p.preposition} {r.group}"
                + "".join(f" at {w}" for w in where) + f", recorded {p.when}: " + ", ".join(values)
                + (f"; {p.note} {r.same_as}" if r.same_as else "") + ".")
        return text + ("" if r.place is None else f" Place: {r.place}.")
    if p.voice == "terse":
        parts = [f"{p.lead}{r.ref} ({r.source}, {r.status})", r.group, *where, p.when, "; ".join(values)]
        parts += [f"{p.note} {r.same_as}"] if r.same_as else []
        parts += [] if r.place is None else [f"place: {r.place}"]
        return " — ".join(parts)
    raise ValueError(f"unknown voice {p.voice!r}")


def _qualifier(f: GlossaryField, r: RecordReport) -> str | None:
    """The qualifier a report gives with a field's value (None: the field has none, or the report gives none)."""
    return None if f.qualifier is None else r.qualifiers.get(f.qualifier)


def _notice_body(n: Notice, rng: random.Random, glossary: Glossary, vocab: TrapVocab, units: str,
                 grammar: _Grammar) -> str:
    """A notice as a message: a wording of its kind, drawn, or the next one that reads back exactly when
    the domain's words make the drawn one ambiguous. A correction gives both values in one drawn unit."""
    when = format_time(n.time, rng.choice(UTC_TIME_FORMATS))
    words: dict[str, str] = {"when": when, "ref": n.target}
    if n.kind == "retraction":
        forms = RETRACTIONS
        words.update(noun=glossary.noun, phrase=rng.choice(vocab.retraction_notices),
                     reason="" if n.reason is None else f" ({n.reason})")
    else:
        field = next((f for f in glossary.fields if f.key == n.field), None)
        if n.kind != "correction" or field is None or n.new_value is None or n.old_value is None:
            raise ValueError(f"a {n.kind} notice of {n.target} that names no field of the glossary, or no new and old "
                             "value, cannot be printed")
        phrases = [phrase for phrase in vocab.correction_notices if phrase.endswith(" to")]
        forms = CORRECTIONS if phrases else tuple(form for form in CORRECTIONS if form != PHRASED_CORRECTION)
        unit = _draw_unit(rng, field, units)
        words.update(label=field.label, new=format_value(field, n.new_value, unit),
                     old=format_value(field, n.old_value, unit), phrase=rng.choice(phrases) if phrases else "",
                     qualifier="" if n.qualifier is None else f" ({n.qualifier})")
    first = rng.randrange(len(forms))
    for form in forms[first:] + forms[:first]:
        body = form.format(**words)
        if _reads_back(body, n, grammar, glossary):
            return body
    raise ValueError(f"the {n.kind} notice of {n.target} cannot be printed so that it reads back exactly")


def _add_repeats(log: list[_Message], rng: random.Random, people: Sequence[str], noise: Mapping[str, Any],
                 glossary: Glossary, end: int) -> None:
    """Read-backs of earlier reports, each retyped in another voice that keeps it on its line, and quotes of
    earlier reports and notices, word for word, each posted after what it repeats by someone other than its
    poster."""
    reports = [message for message in log if message.printed is not None]
    for _ in range(rng.randint(*noise["readbacks"]) if reports else 0):
        original = rng.choice(reports)
        report, printed = original.printed
        nick = _someone_else(rng, people, original.nick)
        lead, at = rng.choice(READBACKS), min(end, original.at + rng.randint(30, 1800))
        others = [voice for voice in VOICES if voice != printed.voice]
        rng.shuffle(others)
        retyped = (lead + _report_body(report, replace(printed, voice=voice, lead=""), glossary) for voice in others)
        body = next((text for text in retyped if text.splitlines() == [text]), None)
        if body is not None:
            log.append(_Message(at, f"{nick}: {body}", nick))
    quotable = [message for message in log if message.body is not None]
    for _ in range(rng.randint(*noise["quotes"]) if quotable else 0):
        original = rng.choice(quotable)
        nick = _someone_else(rng, people, original.nick)
        at = min(end, original.at + rng.randint(30, 1800))
        quoted = QUOTE + (original.body if rng.random() < 0.5 else f"{original.nick}: {original.body}")
        log.append(_Message(at, f"{nick}: {quoted}", nick))
        if rng.random() < 0.6:
            log.append(_Message(at, f"{nick}: {rng.choice(QUOTE_COMMENTS)}", nick))


_PLAIN_TOKEN = re.compile(r'[^\s"=`]+')
TOKEN = r'"(?:[^"\\]|\\.)*"|[^\s"=`]+'  # a pasted word: as it is, or a JSON string if it has spaces, '"', "=" or "`"


def _token(text: str) -> str:
    if _PLAIN_TOKEN.fullmatch(text):
        return text
    quoted = json.dumps(text, ensure_ascii=False)
    return quoted if quoted.splitlines() == [quoted] else json.dumps(text)  # every kind of line break escaped


def _untoken(text: str) -> str:
    return json.loads(text) if text.startswith('"') else text


# ------------------------------------------------------------------------------ reading a log


@dataclass(frozen=True)
class _Grammar:
    """The patterns of one glossary's messages: the body of a report in each voice (True: the paste,
    whose words are tokens), and the body of each notice wording, corrections first, with the field a
    correction revises (None: a retraction)."""

    reports: tuple[tuple[re.Pattern[str], bool], ...]
    notices: tuple[tuple[re.Pattern[str], GlossaryField | None], ...]


def _labelled(f: GlossaryField, i: int) -> str:
    """A field's value as the relay, casual and terse voices print it: its label, value and unit, and its
    qualifier in parentheses (groups v{i}, v{i}_unit and q{i})."""
    qualifier = "" if f.qualifier is None else rf"(?: \((?P<q{i}>[^()]*)\))?"
    return re.escape(f.label) + " " + value_pattern(f, f"v{i}") + qualifier


def _keyed(f: GlossaryField, i: int) -> str:
    """A field's value as the paste prints it: key=value, then qualifier=value."""
    qualifier = "" if f.qualifier is None else rf"(?: {re.escape(f.qualifier)}=(?P<q{i}>{TOKEN}))?"
    return re.escape(f.key) + "=" + value_pattern(f, f"v{i}", separator="") + qualifier


def _wording(template: str, slots: Mapping[str, str]) -> re.Pattern[str]:
    """The pattern of a notice wording: its text as it is, each {slot} as the slot's pattern."""
    pieces = re.split(r"(\{\w+\})", template)
    return re.compile("".join(slots[piece[1:-1]] if re.fullmatch(r"\{\w+\}", piece) else re.escape(piece)
                              for piece in pieces))


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    """The patterns of a glossary's messages, mirroring _report_body and the notice wordings."""
    record_id, status = id_pattern(glossary), alternation(glossary.statuses)
    noun, located = re.escape(glossary.noun), glossary.located
    # Times, positions and values are matched by their own patterns, so that free words around them (a
    # source, a group, a place) cannot take a delimiter for theirs: a match backtracks until each fits.
    when, where = f"(?P<time>{TIME_PATTERN})", f"(?P<coords>{COORDS_PATTERN})"
    values = [_labelled(f, i) for i, f in enumerate(glossary.fields)]
    lead = f"(?:{alternation(LEADS)})?"
    relay = (rf"(?P<source>.+?) · {noun} (?P<ref>{record_id}) · (?P<status>{status}) · (?P<group>.+?)"
             + (f" · {where}" if located else "") + "".join(f" · {value}" for value in values)
             + rf" · recorded {when}(?: · (?P<note>.+?) (?P<same>{record_id}))?(?: · place: (?P<place>.+))?")
    casual = (rf"{lead}(?P<source>.+?) reports {noun} (?P<ref>{record_id}) \((?P<status>{status})\) {GROUP_PREPOSITION} "
              rf"(?P<group>.+?)"
              + (f" at {where}" if located else "") + f", recorded {when}: " + ", ".join(values)
              + rf"(?:; (?P<note>.+?) (?P<same>{record_id}))?\.(?: Place: (?P<place>.+)\.)?")
    terse = (rf"{lead}(?P<ref>{record_id}) \((?P<source>.+?), (?P<status>{status})\) — (?P<group>.+?)"
             + (f" — {where}" if located else "") + f" — {when} — " + "; ".join(values)
             + rf"(?: — (?P<note>.+?) (?P<same>{record_id}))?(?: — place: (?P<place>.+))?")
    paste = (rf"{lead}`{when} ref=(?P<ref>{record_id}) src=(?P<source>{TOKEN}) status=(?P<status>{status}) "
             rf"group=(?P<group>{TOKEN})" + (rf" loc=(?P<lat>{NUMBER}),(?P<lon>{NUMBER})" if located else "")
             + "".join(" " + _keyed(f, i) for i, f in enumerate(glossary.fields))
             + rf"(?: {SAME_AS_KEY}=(?P<same>{record_id}))?(?: place=(?P<place>{TOKEN}))?`")

    slots = {"when": when, "ref": f"(?P<ref>{record_id})", "noun": noun, "phrase": "(?P<phrase>.+?)",
             "reason": r"(?: \((?P<reason>.+)\))?"}
    notices: list[tuple[re.Pattern[str], GlossaryField | None]] = []
    for f in glossary.fields:  # corrections first, as in classic
        field_slots = {**slots, "label": re.escape(f.label), "new": value_pattern(f, "new"),
                       "old": value_pattern(f, "old"),
                       "qualifier": "" if f.qualifier is None else r"(?: \((?P<qualifier>[^()]*)\))?"}
        notices += [(_wording(form, field_slots), f) for form in CORRECTIONS]
    notices += [(_wording(form, slots), None) for form in RETRACTIONS]
    return _Grammar(reports=tuple((re.compile(pattern), pattern is paste) for pattern in (relay, casual, terse, paste)),
                    notices=tuple(notices))


def _parse_log(text: str, glossary: Glossary) -> DigestData:
    """Every report and notice of a chat log, in their generic form. Lines that are not messages (the
    header, days, joins, pins) and messages that are neither (repeats among them) change nothing."""
    grammar = _grammar(glossary)
    reports: list[RecordReport] = []
    notices: list[Notice] = []
    for raw in text.splitlines():
        line = raw.strip()
        found = _MESSAGE.fullmatch(line) or _PINNED.fullmatch(line)
        item = None if found is None else _read_body(found["body"], grammar, glossary)
        if isinstance(item, Notice):
            notices.append(item)
        elif isinstance(item, RecordReport):
            reports.append(item)
    return DigestData(reports, notices)


def _reads_back(body: str, item: RecordReport | Notice, grammar: _Grammar, glossary: Glossary) -> bool:
    """Whether a message stays on its line and reads back as exactly `item` (generic types), as _parse_log
    reads it: from its line with the whitespace around it stripped, so a body that ends in a space (a word
    that does) is no body a voice can print, and is pasted instead."""
    return (body == body.strip() and body.splitlines() == [body]
            and _read_body(body, grammar, glossary) == item)


def _read_body(body: str, grammar: _Grammar, glossary: Glossary) -> RecordReport | Notice | None:
    """What one message says: a report or a notice (generic types), or None for anything else. Repeats
    are recognised first: what a quote or a read-back repeats is never a new report or notice."""
    if body.startswith(REPEATS):
        return None
    for pattern, field in grammar.notices:
        match = pattern.fullmatch(body)
        if match is not None:
            notice = _notice(match, field)
            if notice is not None:
                return notice
    for pattern, tokens in grammar.reports:
        match = pattern.fullmatch(body)
        if match is not None:
            report = _report(match, glossary, tokens)
            if report is not None:
                return report
    return None


def _notice(match: re.Match[str], field: GlossaryField | None) -> Notice | None:
    when = parse_time(match["time"])
    if when is None:
        return None
    if field is None:
        return Notice(kind="retraction", target=match["ref"], time=when, reason=match["reason"])
    return Notice(kind="correction", target=match["ref"], time=when, field=field.key,
                  new_value=read_value(field, match, "new"), old_value=read_value(field, match, "old"),
                  qualifier=match.groupdict().get("qualifier"))


def _report(match: re.Match[str], glossary: Glossary, tokens: bool) -> RecordReport | None:
    """The report a message's match gives (`tokens`: a paste, whose words are tokens, as they are or JSON
    strings, and whose position is "lat,lon"); None when its time or position is none."""
    when = parse_time(match["time"])
    if not glossary.located:
        position: tuple[float | None, float | None] | None = (None, None)
    elif tokens:
        position = decode(match["lat"], glossary.coord_decimals), decode(match["lon"], glossary.coord_decimals)
    else:
        position = parse_coords(match["coords"], glossary.coord_decimals)
    if when is None or position is None:
        return None
    word = _untoken if tokens else str
    values = {f.key: read_value(f, match, f"v{i}") for i, f in enumerate(glossary.fields)}
    try:
        qualifiers = {f.qualifier: word(match[f"q{i}"]) for i, f in enumerate(glossary.fields)
                      if f.qualifier is not None and match[f"q{i}"] is not None}
        return RecordReport(ref=match["ref"], source=word(match["source"]), status=match["status"], time=when,
                            lat=position[0], lon=position[1], group=word(match["group"]), values=values,
                            qualifiers=qualifiers, place=None if match["place"] is None else word(match["place"]),
                            same_as=match["same"])
    except ValueError:  # a quoted word that is no JSON string: not a line this genre wrote
        return None
