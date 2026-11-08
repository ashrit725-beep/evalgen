"""email: a digest written as an email thread, for the records of any domain.

A thread of several messages, each opening with From, To, Date and Subject headers (sometimes a Cc),
then a greeting, the sender's own text, a sign-off and a signature. The reports are listed in the
messages' own text: as bullets, as sentences, or as pasted blocks with one field per line. Replies
quote an earlier message with ">", so its reports reappear as quoted repeats, which are never new
reports. Every notice arrives as a forwarded message ("---------- Forwarded message ----------" or
"Begin forwarded message:"), sent on its own or below a message's text. The instance's chatter lines
are remarks in the messages. A message's Date header says when it was sent, never when a record was.

    bullet    - ew41257723 (preliminary, network ew): Northern California; 38.815°N, 122.297°W;
              2026-04-12T06:12:21Z; magnitude 4.1 (md), depth 6.4 km. Note: same event as nc75012345.
    sentence  At 06:12:19 UTC on 12 April 2026, the operator ukmo recorded reading by-12345 (reviewed) in
              North Sea at (56.512, 3.104), with water temperature 54.1 °F (spar) and wave height 1.85 m.
    block     Reading by-12345 (reviewed)
                  Operator: ukmo
                  Recorded: 2026-04-12T07:12:19+01:00
                  Sea area: North Sea
                  Position: 56.512°N, 3.104°E
                  Water temperature: 12.3 °C (spar)
                  Wave height: 6.07 ft
                  Location note: Dogger Bank
    notice    ---------- Forwarded message ----------
              From: Notices desk <notices@records-exchange.example.org>
              Date: Sun, 12 Apr 2026 07:42:19 +0000
              Subject: RETRACTION: by-54321
              To: Reading exchange <exchange@records-exchange.example.org>

              RETRACTION, issued 2026-04-12T07:42:19Z: by-54321 withdrawn. Reason: probe fault. Please
              disregard that report.

Everything a report or notice says is written from the words of the prompt's glossary (the record
noun, ID pattern, source and group labels, statuses, field labels, units and qualifiers) in the shared
formats and exact codec of genres.base, so the parser reads every report and notice back exactly
through the glossary alone. Duplicate notes and retraction phrases come from the domain's trap
vocabulary, and every notice calls itself a RETRACTION or a CORRECTION, or says that a value was
revised, which the reading rules of every domain name. The parser skips quoted lines and reads the
rest; everything else (headers, greetings, sign-offs, signatures, attribution lines, remarks) carries
no data. Every rendering is checked to read back exactly (genres.base.check_round_trip).

Earthquake reports and notices keep their 0.2.0 types (models.Report and Notice), whose generic
names (values, group, source) the parser maps back onto their 0.2.0 attributes; every other domain's
are RecordReports and generic Notices.
"""
from __future__ import annotations

import functools
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from ..digest import MONTHS, WEEKDAYS
from ..domains.base import Domain, Glossary, GlossaryField, RecordReport, TrapVocab
from ..domains.earthquakes import Earthquakes
from ..geo import utc_offset_hours
from ..models import Notice, Report
from ..ops import DigestData
from ..units import Unit
from .base import (COORDS_PATTERN, TIME_FORMATS, TIME_PATTERN, UTC_TIME_FORMATS, Genre, RenderContext, alternation,
                   check_round_trip, format_coords, format_time, format_value, id_pattern, parse_coords, parse_time,
                   read_value, value_pattern)

# ------------------------------------------------------------------------------ the genre


class Email(Genre):
    name = "email"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One preset per dimension of the look (see STYLE); a variant avoids each of the original's."""

        def pick(key: str, options: Sequence[str] | Mapping[str, Any]) -> str:
            names = sorted(options)
            if avoid is not None and key in avoid:
                names = [n for n in names if n != avoid[key]] or names
            return rng.choice(names)

        style = {"units": pick("units", UNIT_PRESETS), "time": pick("time", TIME_PRESETS)}
        if domain.has_location:
            style["coords"] = pick("coords", COORD_PRESETS)
        for key, options in STYLE.items():
            style[key] = pick(key, options)
        return style

    def render(self, ctx: RenderContext) -> str:
        """The thread, in the words of the context's glossary (the domain's own when it has none),
        checked to read back to exactly its reports and notices."""
        glossary = ctx.glossary or ctx.domain.glossary()
        text = _render(ctx, glossary)
        check_round_trip(self, text, ctx, glossary)
        return text

    def sniff(self, digest_text: str) -> bool:
        """A thread opens with its first message's header block: header lines only, up to the first
        blank line, with a From that gives an address, an RFC 2822 Date, a To and a Subject, each
        once. No other genre opens that way: a classic digest opens with its title, and a quoted
        message opens with ">"."""
        fields: dict[str, str] = {}
        for line in digest_text.lstrip().split("\n"):
            if not line.strip():
                break
            match = _HEADER.fullmatch(line)
            if match is None or match["name"] in fields:
                return False
            fields[match["name"]] = match["value"]
        return (REQUIRED_HEADERS <= set(fields) and _MAILBOX.fullmatch(fields["From"]) is not None
                and _RFC_DATE.fullmatch(fields["Date"]) is not None)

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """Every report and notice, read through the prompt's glossary (or, when it prints none, the
        words of `domain`, as render writes them without one). Quoted lines are repeats and are
        skipped; lines that are neither reports nor notices change nothing."""
        if glossary is None:
            if domain.has_surface and domain.surface is None:
                raise ValueError(f"a digest of {domain.name} records is read through the glossary its prompt "
                                 "prints, and this prompt prints none")
            glossary = domain.glossary()
        return _parse(digest_text, glossary, _legacy(domain))


def _legacy(domain: Domain) -> bool:
    """Earthquake domains keep the 0.2.0 record types: their lines are models.Report, and their
    corrections give new_mag, old_mag and mag_type."""
    return isinstance(domain, Earthquakes)


EMAIL = Email()
GENRE = EMAIL  # what the registry looks up (see evalgen.genres)

# ------------------------------------------------------------------------------ the look
# A rendering style picks one preset per dimension. Values, times and positions are drawn per item
# from weighted mixes, as in the classic genre; the other dimensions shape the thread.

TIME_PRESETS = {
    "utc_heavy": {"iso": 0.3, "iso_space": 0.2, "rfc": 0.2, "prose_utc": 0.15, "iso_offset": 0.05, "local12": 0.05,
                  "local24": 0.05},
    "local_heavy": {"iso": 0.1, "iso_space": 0.05, "rfc": 0.1, "prose_utc": 0.05, "iso_offset": 0.2, "local12": 0.3,
                    "local24": 0.2},
    "mixed": {fmt: 1.0 for fmt in TIME_FORMATS},
}
COORD_PRESETS = {
    "hemi": {"hemi": 0.75, "signed": 0.25},
    "signed": {"hemi": 0.25, "signed": 0.75},
    "mixed": {"hemi": 0.5, "signed": 0.5},
}
# The weight of a field's stored unit in each preset; the others share the rest (None: all alike).
UNIT_PRESETS: dict[str, float | None] = {"mostly_stored": 0.7, "mostly_other": 0.15, "mixed": None}
# How the senders list their reports: each message draws one item style from the preset's mix.
ITEM_PRESETS = {
    "bullets": {"bullet": 0.6, "sentence": 0.2, "block": 0.2},
    "sentences": {"bullet": 0.2, "sentence": 0.6, "block": 0.2},
    "blocks": {"bullet": 0.2, "sentence": 0.2, "block": 0.6},
    "mixed": {"bullet": 1.0, "sentence": 1.0, "block": 1.0},
}
STYLE: dict[str, tuple[str, ...]] = {
    "items": tuple(ITEM_PRESETS),
    # by_source: one message per source; batches: a coordinator's batches in time order; roundup: one
    # message with every report, then replies.
    "thread": ("by_source", "batches", "roundup"),
    # How a reply quotes: "On <date>, <sender> wrote:" and "> "; an "-----Original Message-----" header
    # and "> "; or "<name> wrote:" and ">".
    "quotes": ("gmail", "outlook", "compact"),
    "forwards": ("gmail", "apple"),  # "---------- Forwarded message ----------" or "Begin forwarded message:"
    "notices": ("separate", "attached"),  # a forward of its own, or below a message's text
    "headers": ("from_first", "date_first"),  # the order of the header fields
    "separators": ("blank", "rule"),  # between messages: blank lines, or a rule as well
    "signatures": ("dashes", "block", "oneline"),
}

ZONES = (-12, 14)  # the UTC offsets, in whole hours, of sources whose records have no position
COORDINATOR_ZONES = (-8, -7, -5, -4, -3, 0, 1, 2, 3, 5, 8, 9, 10, 12)  # the list's coordinators write from these
BULLETS = ("- ", "* ", "• ", "numbered")
INDENT = "    "  # the indent of the lines of a pasted block
QUOTE = ">"  # every quoted line starts with it, and no other line does
RULE = "_" * 32
FORWARD_MARKERS = {"gmail": ("---------- Forwarded message ----------",), "apple": ("Begin forwarded message:", "")}
ORIGINAL_MESSAGE = "-----Original Message-----"
EXCHANGE = "records-exchange.example.org"  # the mail domain of the list, its coordinators and its notices desk

FIRST_NAMES = ("Dana", "Lee", "Priya", "Marco", "Ingrid", "Tomás", "Aisha", "Kenji", "Ruth", "Oskar", "Nadia",
               "Felix", "Grace", "Hamid", "Sofia", "Owen", "Lena", "Yusuf", "Maeve", "Arjun")
LAST_NAMES = ("Whitfield", "Park", "Raman", "Bianchi", "Halvorsen", "Ortega", "Bello", "Sato", "Abernathy",
              "Lindqvist", "Haddad", "Brandt", "Okafor", "Karimi", "Moreau", "Price", "Novak", "Demir", "Quinn",
              "Ferreira")
ROLES = ("Duty officer", "Data manager", "Operations analyst", "Watch supervisor", "Records coordinator",
         "Liaison officer", "Shift lead")
# The wording of messages. None of it names a record, a value or a notice: it changes nothing.
GREETINGS = ("Hi all,", "Hello all,", "Hi everyone,", "Morning all,", "Dear colleagues,", "Hi team,", "Hello,")
INTROS = ("Here are the {plural} received since the last update:", "Please find the latest {plural} below.",
          "Sharing the {plural} we have so far:", "{Plural} from our side:", "The next set of {plural} is below.")
EMPTY_INTROS = ("Nothing to add from our side yet.", "No {plural} from us in this window.")
ACKS = ("Thanks, received.", "Got these, thank you.", "Thanks for sending these.", "Received, thanks.",
        "Thanks, all received on our side.")
FORWARD_INTROS = ("Forwarding the notice below for the record.", "FYI, see the notice below.",
                  "Passing on this notice in case it has not reached everyone.",
                  "Forwarding this notice; please see below.")
ATTACHED_INTROS = ("I am also forwarding a notice that reached us; see below.",
                   "A notice also came in; it is forwarded below.")
REMARKS = ("", "FYI: ", "For information: ")  # what a chatter line without a label of its own opens with
LABELLED = re.compile(r"[A-Z][\w' ]{0,24}: ")  # "Reminder: ...", "Editor's note: ..."
SIGN_OFFS = ("Thanks,", "Best,", "Regards,", "Cheers,", "Kind regards,", "Many thanks,", "Best wishes,")
SUBJECTS = ("{Plural}: exchange for {date}", "Latest {plural} ({date})", "{Noun} reports from {date}",
            "{Plural} update, {date}")
DISCLAIMER = "This message and any attachments are intended only for the named recipients."


def _draw(rng: random.Random, table: Mapping[str, float]) -> str:
    return rng.choices(list(table), weights=list(table.values()), k=1)[0]


def _draw_unit(rng: random.Random, f: GlossaryField, preset: str) -> Unit | None:
    """The unit one value of `f` is printed in (None for a field without units)."""
    if not f.units:
        return None
    stored = UNIT_PRESETS[preset]
    if stored is None or len(f.units) == 1:
        return rng.choice(f.units)
    others = (1.0 - stored) / (len(f.units) - 1)
    return rng.choices(f.units, weights=[stored if u.symbol == f.canonical_unit else others for u in f.units],
                       k=1)[0]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _slug(text: str) -> str:
    """`text` as the plain part of a mail address: "Tomás" -> "tomas", "Met Éireann" -> "met-eireann"."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", plain).strip("-") or "desk"


def _rfc2822(ts: int, zone: int) -> str:
    """A Date header: the time `ts` in the sender's zone, `zone` hours from UTC."""
    local = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=zone)
    return (f"{WEEKDAYS[local.weekday()]}, {local.day} {MONTHS[local.month - 1]} {local.year} "
            f"{local.hour:02d}:{local.minute:02d}:{local.second:02d} {'+' if zone >= 0 else '-'}{abs(zone):02d}00")


def _day(ts: int) -> str:
    """The UTC day of `ts`, as a subject line names it: "14 Apr 2026"."""
    utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{utc.day} {MONTHS[utc.month - 1]} {utc.year}"


def _short_date(ts: int, zone: int) -> str:
    """How an attribution line dates a message: "Tue, 14 Apr 2026 at 07:02"."""
    local = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=zone)
    return (f"{WEEKDAYS[local.weekday()]}, {local.day} {MONTHS[local.month - 1]} {local.year} at "
            f"{local.hour:02d}:{local.minute:02d}")


# ------------------------------------------------------------------------------ records
# Every domain's lines answer to the same generic names (ref, source, status, time, lat, lon, group,
# values, place, same_as); only an earthquake's 0.2.0 Report keeps its one qualifier, the magnitude
# type, as an attribute. These functions are the genre's only knowledge of the two record types.


def _qualifier(f: GlossaryField, r: Any) -> str | None:
    """The qualifier a report gives with a field's value (None: the field has none, or the report gives
    none). A RecordReport stores its qualifiers by key; a 0.2.0 Report stores its one as an attribute
    of that name ("mag_type")."""
    if f.qualifier is None:
        return None
    if isinstance(r, RecordReport):
        return r.qualifiers.get(f.qualifier)
    return getattr(r, f.qualifier, None)


def _new_report(legacy: bool, *, ref: str, source: str, status: str, time: int, lat: float | None,
                lon: float | None, group: str, values: dict[str, float], qualifiers: dict[str, str],
                place: str | None, same_as: str | None) -> Any:
    """A report as its domain stores it: a RecordReport, or (`legacy`) an earthquake's 0.2.0 Report, whose
    generic names map back onto its attributes (the inverse of models._GenericNames). None when a
    legacy line lacks what a Report needs."""
    if not legacy:
        return RecordReport(ref=ref, source=source, status=status, time=time, lat=lat, lon=lon, group=group,
                            values=values, qualifiers=qualifiers, place=place, same_as=same_as)
    if lat is None or lon is None or set(values) != {"mag", "depth"} or "mag_type" not in qualifiers:
        return None
    return Report(ref=ref, network=source, status=status, time=time, lat=lat, lon=lon, depth_km=values["depth"],
                  mag=values["mag"], mag_type=qualifiers["mag_type"], region=group, place=place, same_as=same_as)


def _revision(n: Notice) -> tuple[str, float, float, str | None]:
    """What a correction revises: (field key, new value, old value, new qualifier). A generic notice names
    its field; a 0.2.0 earthquake correction revises the magnitude ("mag") and gives its type."""
    if n.field is not None:
        if n.new_value is None or n.old_value is None:
            raise ValueError(f"the correction of {n.target} gives no new and old value")
        return n.field, n.new_value, n.old_value, n.qualifier
    if n.new_mag is None or n.old_mag is None:
        raise ValueError(f"the correction of {n.target} names no field and gives no magnitude")
    return "mag", n.new_mag, n.old_mag, n.mag_type


def _new_correction(legacy: bool, target: str, time: int, key: str, new: float, old: float,
                    qualifier: str | None) -> Notice | None:
    """A correction as its domain stores it (the inverse of _revision); None for a legacy correction of
    another field than the magnitude, which a 0.2.0 Notice cannot hold."""
    if not legacy:
        return Notice(kind="correction", target=target, time=time, field=key, new_value=new, old_value=old,
                      qualifier=qualifier)
    if key != "mag":
        return None
    return Notice(kind="correction", target=target, time=time, new_mag=new, old_mag=old, mag_type=qualifier)


# ---------------------------------------------------------------------------- writing items


@dataclass(frozen=True)
class _Item:
    """One report as a message lists it, with what was drawn for it."""

    report: Any  # a Report or a RecordReport
    when: str  # its time, formatted
    where: str | None  # its position, formatted (None: records have none)
    units: tuple[Unit | None, ...]  # the unit of each field's value, in glossary order
    note: str | None  # the duplicate note before its same_as


def _labelled(glossary: Glossary, item: _Item) -> list[str]:
    """Each field's value as bullets and sentences print it: "water temperature 12.3 °C (spar)"."""
    out = []
    for f, unit in zip(glossary.fields, item.units):
        qualifier = _qualifier(f, item.report)
        out.append(f"{f.label} {format_value(f, item.report.values[f.key], unit)}"
                   + ("" if qualifier is None else f" ({qualifier})"))
    return out


def _tail(item: _Item) -> str:
    r = item.report
    return ((f" Note: {item.note} {r.same_as}." if r.same_as else "")
            + (f" Location note: {r.place}." if r.place is not None else ""))


def _bullet(item: _Item, glossary: Glossary, marker: str) -> str:
    """A list entry whose slots are set off by semicolons, so that a group or a time with commas reads
    as one slot: "- ref (status, network nc): group; position; time; values."."""
    r = item.report
    where = "" if item.where is None else f"; {item.where}"
    return (f"{marker}{r.ref} ({r.status}, {glossary.source_label} {r.source}): {r.group}{where}; {item.when}; "
            + ", ".join(_labelled(glossary, item)) + "." + _tail(item))


def _listing(parts: Sequence[str]) -> str:
    """Parts as a sentence lists them: "a", "a and b", "a, b and c"."""
    return " and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _sentence(item: _Item, glossary: Glossary) -> str:
    r = item.report
    where = "" if item.where is None else f" at {item.where}"
    return (f"At {item.when}, the {glossary.source_label} {r.source} recorded {glossary.noun} {r.ref} ({r.status}) "
            f"in {r.group}{where}, with {_listing(_labelled(glossary, item))}." + _tail(item))


def _block(item: _Item, glossary: Glossary) -> list[str]:
    r = item.report
    lines = [f"{_cap(glossary.noun)} {r.ref} ({r.status})", f"{INDENT}{_cap(glossary.source_label)}: {r.source}",
             f"{INDENT}Recorded: {item.when}", f"{INDENT}{_cap(glossary.group_label)}: {r.group}"]
    if item.where is not None:
        lines.append(f"{INDENT}Position: {item.where}")
    for f, unit in zip(glossary.fields, item.units):
        qualifier = _qualifier(f, r)
        lines.append(f"{INDENT}{_cap(f.label)}: {format_value(f, r.values[f.key], unit)}"
                     + ("" if qualifier is None else f" ({qualifier})"))
    if r.same_as:
        lines.append(f"{INDENT}Note: {item.note} {r.same_as}")
    if r.place is not None:
        lines.append(f"{INDENT}Location note: {r.place}")
    return lines


# ------------------------------------------------------------------------- writing notices


def _notice_text(n: Notice, rng: random.Random, glossary: Glossary, vocab: TrapVocab, units: str) -> str:
    """A notice in one of three wordings of its kind. Each calls itself a RETRACTION or a CORRECTION,
    or says that a value was revised, as the reading rules of every domain name notices; a retraction
    may use a phrase of the domain's trap vocabulary, and a correction gives both values in one drawn
    unit (and the new qualifier, when it has one)."""
    form = rng.randrange(3)
    when = format_time(n.time, rng.choice(UTC_TIME_FORMATS))
    if n.kind == "retraction":
        phrase = rng.choice(vocab.retraction_notices)
        if form == 0:
            return (f"RETRACTION, issued {when}: {n.target} {phrase}."
                    + ("" if n.reason is None else f" Reason: {n.reason}.") + " Please disregard that report.")
        reason = "" if n.reason is None else f" ({n.reason})"
        if form == 1:
            return f"RETRACTION NOTICE ({when}): report {n.target} was not a real {glossary.noun}{reason}; ignore it."
        return f"Notice of RETRACTION, {when}. Report {n.target}: {phrase}{reason}."
    if n.kind != "correction":
        raise ValueError(f"a {n.kind} notice cannot be printed")
    key, new_value, old_value, qualifier = _revision(n)
    f = next((f for f in glossary.fields if f.key == key), None)
    if f is None:
        raise ValueError(f"the correction of {n.target} revises {key!r}, which the glossary does not define")
    unit = _draw_unit(rng, f, units)
    new, old = format_value(f, new_value, unit), format_value(f, old_value, unit)
    q = "" if qualifier is None else f" ({qualifier})"
    if form == 0:  # the domain's own phrase, when it reads "... to <value>"; else "revised to"
        phrase = rng.choice([p for p in vocab.correction_notices if p.endswith(" to")] or ["revised to"])
        return (f"CORRECTION, issued {when}: the {f.label} of {n.target} is {phrase} {new}{q}; it was first given as "
                f"{old}.")
    if form == 1:
        return f"CORRECTION NOTICE ({when}): {n.target} {f.label} now {new}{q}, not {old}."
    return f"Notice of a revised value, {when}. Report {n.target}: {f.label} revised from {old} to {new}{q}."


# ------------------------------------------------------------------------------- the thread


@dataclass(frozen=True)
class _Person:
    """Someone who writes to the list: a source's desk, or one of the list's coordinators."""

    name: str
    address: str
    role: str
    org: str
    zone: int  # the UTC offset of the Date headers this person sends

    @property
    def first(self) -> str:
        return self.name.split(" ")[0]

    @property
    def mailbox(self) -> str:
        return f"{self.name} <{self.address}>"


@dataclass
class _Forward:
    """A notice as a forwarded message: the lines of its forward and when it was issued."""

    lines: list[str]
    time: int
    subject: str


@dataclass
class _Message:
    """One message of the thread, filled in step by step (see _render)."""

    sender: _Person
    kind: str  # "reports", "forward" or "ack"
    items: list[_Item] = field(default_factory=list)
    item_style: str = "bullet"
    marker: str = "- "
    forwards: list[_Forward] = field(default_factory=list)
    chatter: list[str] = field(default_factory=list)
    cc: _Person | None = None
    subject: str = ""
    date: int = 0
    own: list[str] = field(default_factory=list)  # the lines of its own text, once written

    def content_time(self) -> int | None:
        """When the latest record or notice it carries was recorded or issued (None: it carries none)."""
        times = [item.report.time for item in self.items] + [forward.time for forward in self.forwards]
        return max(times) if times else None


@dataclass(frozen=True)
class _Words:
    """The words a thread is written in: the glossary's, and the list's own."""

    glossary: Glossary
    to: str  # the list's mailbox
    desk: _Person  # who sends notices

    def fill(self, template: str, **extra: str) -> str:
        g = self.glossary
        return template.format(noun=g.noun, Noun=_cap(g.noun), plural=g.plural, Plural=_cap(g.plural), **extra)


def _render(ctx: RenderContext, glossary: Glossary) -> str:
    """The thread of `ctx`'s reports, notices and chatter, as `ctx.style` draws it. Every step draws from
    the rendering's own stream, in this order: the items, the people, the messages and their item
    styles, the notices, the chatter, the replies that only acknowledge, the order, the dates and
    subjects, and the text."""
    rng, style, vocab = ctx.rng, ctx.style, ctx.domain.trap_vocab
    sources = sorted({r.source for r in ctx.reports})
    if glossary.located:
        first: dict[str, Any] = {}
        for r in ctx.reports:
            first.setdefault(r.source, r)
        zones = {source: utc_offset_hours(first[source].lon) for source in sources}
    else:
        zones = {source: rng.randint(*ZONES) for source in sources}
    items = []
    for r in ctx.reports:
        offset = utc_offset_hours(r.lon) if glossary.located else zones[r.source]
        when = format_time(r.time, _draw(rng, TIME_PRESETS[style["time"]]), offset)
        where = (format_coords(r.lat, r.lon, _draw(rng, COORD_PRESETS[style["coords"]]), glossary.coord_decimals)
                 if glossary.located else None)
        units = tuple(_draw_unit(rng, f, style["units"]) for f in glossary.fields)
        note = rng.choice(vocab.duplicate_notes) if r.same_as else None
        items.append(_Item(r, when, where, units, note))

    desks, coordinators = _people(rng, sources, zones, glossary)
    words = _Words(glossary, f"{_cap(glossary.noun)} exchange <exchange@{EXCHANGE}>",
                   _Person("Notices desk", f"notices@{EXCHANGE}", "Notices", "Records exchange", 0))
    messages = _report_messages(items, style, rng, desks, coordinators)
    forwards = [_forward(n, rng, style, words, vocab) for n in ctx.notices]
    for forward in forwards:
        if style["notices"] == "attached":
            rng.choice([m for m in messages if m.kind == "reports"]).forwards.append(forward)
        else:
            messages.append(_Message(rng.choice(coordinators + list(desks.values())), "forward", forwards=[forward]))
    talkers = [m for m in messages if m.kind != "forward"]
    for line in ctx.chatter:
        rng.choice(talkers).chatter.append(line if LABELLED.match(line) else rng.choice(REMARKS) + line)
    acks = [_Message(rng.choice(coordinators + list(desks.values())), "ack")
            for _ in range(rng.randint(1, 2) if style["thread"] == "roundup" else rng.randint(0, 2))]
    messages = _ordered(messages, acks, style["thread"], rng)
    _date(messages, ctx.compiled_at, rng)
    subject = words.fill(rng.choice(SUBJECTS), date=_day(messages[0].date))
    for k, m in enumerate(messages):
        m.subject = (f"Fwd: {m.forwards[0].subject}" if m.kind == "forward" else subject if k == 0
                     else f"Re: {subject}")
        if rng.random() < 0.3:
            m.cc = rng.choice([p for p in coordinators + list(desks.values()) if p != m.sender])
    return _write(messages, style, rng, words)


def _people(rng: random.Random, sources: Sequence[str], zones: Mapping[str, int],
            glossary: Glossary) -> tuple[dict[str, _Person], list[_Person]]:
    """A person at each source's desk, and two coordinators of the list, all with names of their own."""
    count = len(sources) + 2
    firsts = rng.sample(FIRST_NAMES, min(count, len(FIRST_NAMES)))
    lasts = rng.sample(LAST_NAMES, min(count, len(LAST_NAMES)))
    names = [(firsts[i % len(firsts)], lasts[i % len(lasts)]) for i in range(count)]

    def person(i: int, mail_domain: str, org: str, zone: int) -> _Person:
        first, last = names[i]
        return _Person(f"{first} {last}", f"{_slug(first)}.{_slug(last)}@{mail_domain}", rng.choice(ROLES), org, zone)

    desks = {source: person(i, f"{_slug(source)}.example.net", f"{source} ({glossary.source_label})", zones[source])
             for i, source in enumerate(sources)}
    coordinators = [person(len(sources) + k, EXCHANGE, "Records exchange", rng.choice(COORDINATOR_ZONES))
                    for k in range(2)]
    return desks, coordinators


def _report_messages(items: Sequence[_Item], style: Mapping[str, Any], rng: random.Random,
                     desks: Mapping[str, _Person], coordinators: Sequence[_Person]) -> list[_Message]:
    """The messages that list the reports, as the style's thread layout groups them, each with an item
    style drawn from the style's mix and its items in time order or as received."""
    layout = style["thread"]
    if layout == "by_source":
        groups: dict[str, list[_Item]] = {}
        for item in items:
            groups.setdefault(item.report.source, []).append(item)
        messages = [_Message(desks[source], "reports", group) for source, group in groups.items()]
    elif layout == "batches":
        ordered = sorted(items, key=lambda it: (it.report.time, it.report.ref, it.report.source))
        k = min(len(ordered), rng.randint(2, 4)) or 1
        cuts = sorted(rng.sample(range(1, len(ordered)), k - 1)) if k > 1 else []
        bounds = [0, *cuts, len(ordered)]
        messages = [_Message(coordinators[i % len(coordinators)], "reports", ordered[a:b])
                    for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    elif layout == "roundup":
        messages = [_Message(coordinators[0], "reports", list(items))]
    else:
        raise ValueError(f"unknown thread layout {layout!r}")
    if not messages:  # nothing to report: the thread still opens with a message
        messages = [_Message(coordinators[0], "reports")]
    for m in messages:
        m.item_style = _draw(rng, ITEM_PRESETS[style["items"]])
        m.marker = rng.choice(BULLETS)
        if rng.random() < 0.5:
            m.items.sort(key=lambda it: (it.report.time, it.report.ref, it.report.source))
        else:
            rng.shuffle(m.items)
    return messages


def _forward(n: Notice, rng: random.Random, style: Mapping[str, Any], words: _Words, vocab: TrapVocab) -> _Forward:
    """A notice as a forwarded message: its marker, its headers (sent by the notices desk when it was
    issued) and its text, signed by the desk."""
    text = _notice_text(n, rng, words.glossary, vocab, style["units"])
    keyword = "RETRACTION" if n.kind == "retraction" else "CORRECTION"
    subject = rng.choice((f"{keyword}: {n.target}", f"{_cap(keyword.lower())} for {n.target}"))
    desk, date = words.desk, _rfc2822(n.time, 0)
    if style["forwards"] == "gmail":
        headers = [f"From: {desk.mailbox}", f"Date: {date}", f"Subject: {subject}", f"To: {words.to}"]
    else:
        headers = [f"From: {desk.mailbox}", f"Subject: {subject}", f"Date: {date}", f"To: {words.to}"]
    lines = [*FORWARD_MARKERS[style["forwards"]], *headers, "", text, "", desk.name]
    return _Forward(lines, n.time, subject)


def _ordered(messages: list[_Message], acks: Sequence[_Message], layout: str, rng: random.Random) -> list[_Message]:
    """The thread's order. A roundup opens the thread and everything else follows in any order; otherwise
    messages follow the time of what they carry, and the thread opens with a message that lists
    reports. Replies that only acknowledge go anywhere after the first message."""
    if layout == "roundup":
        rest = messages[1:]
        rng.shuffle(rest)
        ordered = [messages[0], *rest]
    else:
        ordered = sorted(messages, key=lambda m: (m.content_time() is None, m.content_time() or 0))
        opener = next(i for i, m in enumerate(ordered) if m.kind == "reports")
        ordered.insert(0, ordered.pop(opener))
    for ack in acks:
        ordered.insert(rng.randint(1, len(ordered)), ack)
    return ordered


def _date(messages: Sequence[_Message], compiled_at: int, rng: random.Random) -> None:
    """Send the messages in thread order: each a little after the message before it, and after what it
    carries was recorded or issued (a thread that opens with nothing to carry opens after the window)."""
    previous: int | None = None
    for m in messages:
        content = m.content_time()
        after = rng.randint(4, 75) * 60 + rng.randint(0, 59)
        if previous is None:
            m.date = (compiled_at if content is None else content) + after
        else:
            m.date = max(previous + rng.randint(2, 50) * 60 + rng.randint(0, 59),
                         previous if content is None else content + after)
        previous = m.date


# ------------------------------------------------------------------------------ the text


def _write(messages: Sequence[_Message], style: Mapping[str, Any], rng: random.Random, words: _Words) -> str:
    """The thread as text: each message's headers, its own text and, for most replies, a quote of the
    message before it."""
    out: list[str] = []
    previous: _Message | None = None
    for m in messages:
        if previous is not None:
            out += ["", "", RULE, ""] if style["separators"] == "rule" else ["", "", ""]
        out += _headers(m, style, words)
        out.append("")
        m.own = _own_text(m, style, rng, words)
        out += m.own
        if previous is not None and m.kind != "forward" and (m.kind == "ack" or rng.random() < 0.5):
            out += ["", *_quote(previous, style, rng, words)]
        previous = m
    return "\n".join(out)


def _headers(m: _Message, style: Mapping[str, Any], words: _Words) -> list[str]:
    sender, to, date = f"From: {m.sender.mailbox}", f"To: {words.to}", f"Date: {_rfc2822(m.date, m.sender.zone)}"
    cc = [] if m.cc is None else [f"Cc: {m.cc.mailbox}"]
    if style["headers"] == "date_first":
        return [date, sender, to, *cc, f"Subject: {m.subject}"]
    return [sender, to, *cc, date, f"Subject: {m.subject}"]


def _own_text(m: _Message, style: Mapping[str, Any], rng: random.Random, words: _Words) -> list[str]:
    """What a message says itself: a greeting and an introduction, its items, remarks, a sign-off and a
    signature (an acknowledgement or a forward says less), then any forwarded notices."""
    sender, glossary = m.sender, words.glossary
    lines: list[str] = []
    if m.kind == "reports":
        lines += [rng.choice(GREETINGS), ""]
        intro = rng.choice(INTROS if m.items else EMPTY_INTROS)
        lines += [words.fill(intro), ""]
        lines += _items(m, glossary)
        if m.forwards:
            lines += [rng.choice(ATTACHED_INTROS), ""]
    elif m.kind == "ack":
        lines += [rng.choice(ACKS), ""]
    else:
        lines += [rng.choice(FORWARD_INTROS), ""]
    for remark in m.chatter:
        lines += [remark, ""]
    if m.kind == "reports":
        lines += [rng.choice(SIGN_OFFS), sender.first, ""]
        lines += _signature(sender, style["signatures"], rng)
    else:
        lines.append(sender.first)
    for forward in m.forwards:
        lines += ["", *forward.lines]
    return lines


def _items(m: _Message, glossary: Glossary) -> list[str]:
    """A message's items in its item style, followed by a blank line: bullets as consecutive lines,
    sentences and blocks as paragraphs."""
    lines: list[str] = []
    for number, item in enumerate(m.items, start=1):
        if m.item_style == "bullet":
            marker = f"{number}. " if m.marker == "numbered" else m.marker
            lines.append(_bullet(item, glossary, marker))
        elif m.item_style == "sentence":
            lines += [_sentence(item, glossary), ""]
        else:
            lines += [*_block(item, glossary), ""]
    if lines and lines[-1] != "":
        lines.append("")
    return lines


def _signature(person: _Person, style: str, rng: random.Random) -> list[str]:
    if style == "dashes":
        lines = ["-- ", person.name, f"{person.role}, {person.org}"]
    elif style == "block":
        lines = [person.name, person.role, person.org, person.address]
    else:
        lines = [f"{person.name} · {person.role} · {person.org}"]
    if rng.random() < 0.25:
        lines += ["", DISCLAIMER]
    return lines


def _quote(m: _Message, style: Mapping[str, Any], rng: random.Random, words: _Words) -> list[str]:
    """A quote of message `m`'s own text: an attribution, then its lines behind ">", cut short after a
    few lines more often than not. Quoted lines repeat an earlier message: they are never new reports
    or notices."""
    own = list(m.own)
    while own and own[-1] == "":
        own.pop()
    limit = rng.randint(4, 14)
    if len(own) > limit and rng.random() < 0.6:
        own = own[:limit] + ["[...]"]
    kind = style["quotes"]
    prefix = ">" if kind == "compact" else "> "
    quoted = [f"{prefix}{line}" if line else QUOTE for line in own]
    if kind == "gmail":
        return [f"On {_short_date(m.date, m.sender.zone)}, {m.sender.mailbox} wrote:", *quoted]
    if kind == "outlook":
        return [ORIGINAL_MESSAGE, f"From: {m.sender.mailbox}", f"Sent: {_rfc2822(m.date, m.sender.zone)}",
                f"To: {words.to}", f"Subject: {m.subject}", "", *quoted]
    return [f"{m.sender.name} wrote:", *quoted]


# ------------------------------------------------------------------------------ reading


@dataclass(frozen=True)
class _Grammar:
    """The patterns of one glossary's threads: the one-line items (bullets and sentences), the first
    line of a pasted block and the whole block, and the notices, each with the field a correction
    revises (None: a retraction)."""

    lines: tuple[re.Pattern[str], ...]
    block_head: re.Pattern[str]
    block: re.Pattern[str]
    notices: tuple[tuple[re.Pattern[str], GlossaryField | None], ...]


def _valued(f: GlossaryField, i: int, label: str) -> str:
    """A field's value after its `label`: the value and unit (groups v{i}, v{i}_unit) and its qualifier in
    parentheses (group q{i})."""
    qualifier = "" if f.qualifier is None else rf"(?: \((?P<q{i}>[^()\n]*)\))?"
    return re.escape(label) + value_pattern(f, f"v{i}") + qualifier


@functools.lru_cache(maxsize=64)
def _grammar(glossary: Glossary) -> _Grammar:
    record_id, status = id_pattern(glossary), alternation(glossary.statuses)
    when, where = f"(?P<time>{TIME_PATTERN})", f"(?P<coords>{COORDS_PATTERN})"
    located = glossary.located
    labelled = [_valued(f, i, f"{f.label} ") for i, f in enumerate(glossary.fields)]
    tail =rf"(?: Note: (?P<note>.+?) (?P<same>{record_id})\.)?(?: Location note: (?P<place>.+)\.)?"
    source = re.escape(glossary.source_label) + " (?P<source>.+?)"
    bullet = (rf"(?:[-*•]|\d{{1,3}}\.) (?P<ref>{record_id}) \((?P<status>{status}), {source}\): (?P<group>.+?)"
              + (f"; {where}" if located else "") + f"; {when}; " + ", ".join(labelled) + r"\." + tail)
    sentence = (rf"At {when}, the {source} recorded {re.escape(glossary.noun)} (?P<ref>{record_id}) "
                rf"\((?P<status>{status})\) in (?P<group>.+?)" + (f" at {where}" if located else "")
                + ", with " + _listing(labelled) + r"\." + tail)
    head = rf"{re.escape(_cap(glossary.noun))} (?P<ref>{record_id}) \((?P<status>{status})\)"
    block = (head + rf"\n{INDENT}{re.escape(_cap(glossary.source_label))}: (?P<source>.+)"
             + rf"\n{INDENT}Recorded: {when}\n{INDENT}{re.escape(_cap(glossary.group_label))}: (?P<group>.+)"
             + (rf"\n{INDENT}Position: {where}" if located else "")
             + "".join(rf"\n{INDENT}" + _valued(f, i, f"{_cap(f.label)}: ") for i, f in enumerate(glossary.fields))
             + rf"(?:\n{INDENT}Note: (?P<note>.+?) (?P<same>{record_id}))?(?:\n{INDENT}Location note: (?P<place>.+))?")

    noun = re.escape(glossary.noun)
    notices: list[tuple[re.Pattern[str], GlossaryField | None]] = []
    for f in glossary.fields:  # corrections, by the field they revise
        label, new, old = re.escape(f.label), value_pattern(f, "new"), value_pattern(f, "old")
        qualifier = r"(?: \((?P<qualifier>[^()]*)\))?"
        for body in (rf"CORRECTION, issued {when}: the {label} of (?P<ref>{record_id}) is (?P<phrase>.+?) {new}"
                     rf"{qualifier}; it was first given as {old}\.",
                     rf"CORRECTION NOTICE \({when}\): (?P<ref>{record_id}) {label} now {new}{qualifier}, not {old}\.",
                     rf"Notice of a revised value, {when}\. Report (?P<ref>{record_id}): {label} revised from {old} to "
                     rf"{new}{qualifier}\."):
            notices.append((re.compile(body), f))
    for body in (rf"RETRACTION, issued {when}: (?P<ref>{record_id}) (?P<phrase>.+?)\.(?: Reason: (?P<reason>.+?)\.)? "
                 r"Please disregard that report\.",
                 rf"RETRACTION NOTICE \({when}\): report (?P<ref>{record_id}) was not a real {noun}"
                 r"(?: \((?P<reason>.+)\))?; ignore it\.",
                 rf"Notice of RETRACTION, {when}\. Report (?P<ref>{record_id}): (?P<phrase>.+?)"
                 r"(?: \((?P<reason>.+)\))?\."):
        notices.append((re.compile(body), None))
    return _Grammar(lines=(re.compile(bullet), re.compile(sentence)), block_head=re.compile(head),
                    block=re.compile(block), notices=tuple(notices))


def _parse(text: str, glossary: Glossary, legacy: bool) -> DigestData:
    """Every report and notice of a thread, from the lines that are not quoted."""
    grammar = _grammar(glossary)
    lines = text.split("\n")
    reports: list[Any] = []
    notices: list[Notice] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith(QUOTE):  # a quoted repeat of an earlier message
            continue
        if grammar.block_head.fullmatch(line):
            end = i
            while end < len(lines) and lines[end].startswith(INDENT):
                end += 1
            match = grammar.block.fullmatch("\n".join(lines[i - 1:end]))
            if match is not None:
                report = _read_report(match, glossary, legacy)
                if report is not None:
                    reports.append(report)
                i = end
                continue
        notice = _read_notice(line, grammar, legacy)
        if notice is not None:
            notices.append(notice)
            continue
        for pattern in grammar.lines:
            match = pattern.fullmatch(line)
            if match is not None:
                report = _read_report(match, glossary, legacy)
                if report is not None:
                    reports.append(report)
                break
    return DigestData(reports, notices)


def _read_report(match: re.Match[str], glossary: Glossary, legacy: bool) -> Any:
    """The report an item's match gives; None when its time or position is none."""
    when = parse_time(match["time"])
    position = parse_coords(match["coords"], glossary.coord_decimals) if glossary.located else (None, None)
    if when is None or position is None:
        return None
    values = {f.key: read_value(f, match, f"v{i}") for i, f in enumerate(glossary.fields)}
    qualifiers = {f.qualifier: match[f"q{i}"] for i, f in enumerate(glossary.fields)
                  if f.qualifier is not None and match[f"q{i}"] is not None}
    return _new_report(legacy, ref=match["ref"], source=match["source"], status=match["status"], time=when,
                       lat=position[0], lon=position[1], group=match["group"], values=values, qualifiers=qualifiers,
                       place=match["place"], same_as=match["same"])


def _read_notice(line: str, grammar: _Grammar, legacy: bool) -> Notice | None:
    for pattern, f in grammar.notices:
        match = pattern.fullmatch(line)
        if match is None:
            continue
        when = parse_time(match["time"])
        if when is None:
            return None
        if f is None:
            return Notice(kind="retraction", target=match["ref"], time=when, reason=match["reason"])
        return _new_correction(legacy, match["ref"], when, f.key, read_value(f, match, "new"),
                               read_value(f, match, "old"), match["qualifier"])
    return None


# ------------------------------------------------------------------------------ sniffing

REQUIRED_HEADERS = frozenset({"From", "To", "Date", "Subject"})
_HEADER = re.compile(r"(?P<name>From|To|Cc|Date|Subject): (?P<value>\S.*)")
_MAILBOX = re.compile(r"[^<>\n]+ <[^<>@\s]+@[^<>@\s]+\.[^<>@\s]+>")
_RFC_DATE = re.compile("(?:" + "|".join(WEEKDAYS) + r"), \d{1,2} (?:" + "|".join(MONTHS)
                       + r") \d{4} \d{2}:\d{2}:\d{2} [+-]\d{4}")
