"""The public reading rules: in English for the prompt, and in code for the answer key.

The rules are the contract between benchmark and model. ``READING_RULES`` are the 0.2.0 rules, identical
in every legacy digest, as their heading says, and published with the code; only the instances (which
quakes, which traps, which thresholds) change. Prompts off the legacy path print their domain's rules
(``Domain.rules_text``): for earthquakes, these rules reheaded and extended for every genre.
``resolve`` is the code version of the rules. ``Mishandle`` describes, for one trap, how a reader
who falls for it would resolve the digest instead; running a question under a mishandling gives
the answer a model would produce if it missed that trap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .domains.base import Record, RecordReport
from .models import Event, Notice, Report

READING_RULES = """\
Reading rules (identical for every digest):
1. Every earthquake has exactly one event ID: the ID printed on its REVIEWED report. An
   earthquake may appear in several reports: a PRELIMINARY report from its own network, or a
   re-report from another network, either under the same event ID or under its own reference with
   a note naming the event ID it duplicates. Reports that share an event ID, or name it in such a
   note, are the same earthquake. Count each earthquake once and refer to it by its event ID.
2. If reports of the same earthquake disagree, use the values from its REVIEWED report.
   PRELIMINARY values are superseded.
3. A correction (a CORRECTION notice, or any notice saying that a value was revised) replaces the
   stated value for the named event, even on a reviewed report.
4. A retraction (a RETRACTION notice, or any notice saying that a report was retracted or
   withdrawn) means the named report was not a real earthquake (for example a quarry blast or a
   false trigger): ignore that report completely.
5. Depths and distances may be given in km or in miles (1 mile = 1.609344 km). A depth can be
   negative (a hypocentre above sea level): depths are signed numbers, so -1.1 km is shallower
   than 0.5 km. Coordinates are latitude then longitude; signed decimal values are negative for
   south and west. Distances are great-circle distances on a sphere of radius 6371 km.
6. Times may be in UTC or in local time with the UTC offset stated. A report's time is when the
   earthquake occurred; a notice's time is when the notice was issued. A time window includes its
   start and excludes its end.
7. Magnitudes are compared as plain numbers, whatever their scale (md, ml, mb, mw, mww, ...),
   once rules 2 and 3 have settled each earthquake's value: a magnitude threshold or ranking in a
   question applies to every scale alike.
8. A region in a question is the region printed on the report, matched exactly as written
   ("Alaska" does not include "Alaska Peninsula").
9. Lines that are neither reports nor notices (headers, station status, editor's notes,
   reminders) carry no data and change nothing.
"""


@dataclass(frozen=True)
class Mishandle:
    """How a careless reader resolves the digest: which traps they misread, and how."""

    keep_retracted: tuple[str, ...] = ()  # retraction ignored: the retracted phantom counts as a record
    ignore_correction_for: tuple[str, ...] = ()  # correction ignored: the corrected field's wrong value kept
    split: tuple[tuple[str, str, str], ...] = ()  # (ref, source, status): a re-report counted as its own record
    # the unsettled report's values used instead of the settling one's (PRELIMINARY instead of REVIEWED)
    prefer_preliminary_for: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list]:
        return {
            "keep_retracted": list(self.keep_retracted),
            "ignore_correction_for": list(self.ignore_correction_for),
            "split": [list(key) for key in self.split],
            "prefer_preliminary_for": list(self.prefer_preliminary_for),
        }

    @classmethod
    def from_dict(cls, d: dict[str, list]) -> "Mishandle":
        return cls(
            keep_retracted=tuple(d.get("keep_retracted", ())),
            ignore_correction_for=tuple(d.get("ignore_correction_for", ())),
            split=tuple(tuple(key) for key in d.get("split", ())),
            prefer_preliminary_for=tuple(d.get("prefer_preliminary_for", ())),
        )

    @classmethod
    def merged(cls, readings: Iterable[dict[str, list]]) -> "Mishandle":
        """Several traps misread at once (a reader rarely misses just one)."""
        parts = [cls.from_dict(d) for d in readings]
        return cls(
            keep_retracted=tuple(x for p in parts for x in p.keep_retracted),
            ignore_correction_for=tuple(x for p in parts for x in p.ignore_correction_for),
            split=tuple(x for p in parts for x in p.split),
            prefer_preliminary_for=tuple(x for p in parts for x in p.prefer_preliminary_for),
        )


# The reading rules of one domain in code (Domain.resolve): (reports, notices, mishandle) -> records.
# Optional, not "X | None": this alias is evaluated at import, and Python 3.9 has no such unions.
Resolver = Callable[[Sequence[Any], Sequence[Notice], Optional[Mishandle]], list[Any]]


def report_to_event(report: Report, event_id: str) -> Event:
    return Event(
        id=event_id,
        time=report.time,
        lat=report.lat,
        lon=report.lon,
        depth_km=report.depth_km,
        mag=report.mag,
        mag_type=report.mag_type,
        region=report.region,
        network=report.network,
        place=report.place,
    )


def resolve(reports: Iterable[Report | RecordReport], notices: Iterable[Notice], mishandle: Mishandle | None = None,
            *, authoritative: str = "reviewed", qualifiers: Mapping[str, str] | None = None) -> list[Event | Record]:
    """Apply the reading rules: drop retracted reports, merge re-reports, prefer the values of the
    `authoritative` (REVIEWED) report, apply corrections. Returns one record per real thing: an
    Event per earthquake, or a Record for the RecordReports of a generic domain.

    A correction revises one value. An earthquake (0.2.0) correction gives the new magnitude and its
    type. A generic correction names its field and gives the new value, and may give a new
    qualifier, which is stored under that field's qualifier key in `qualifiers` (by field key: what a
    domain's field specs name). A generic domain passes its own `authoritative` status and
    `qualifiers` (Domain.resolve); the defaults are the earthquake rules."""
    m = mishandle or Mishandle()
    notices = list(notices)
    retracted = {n.target for n in notices if n.kind == "retraction"} - set(m.keep_retracted)
    corrections = {
        n.target: n for n in notices if n.kind == "correction" and n.target not in m.ignore_correction_for
    }
    split = {tuple(pair) for pair in m.split}

    groups: dict[str, list[Report | RecordReport]] = {}
    separate: list[Event | Record] = []
    for r in reports:
        record_id = r.same_as or r.ref
        if r.ref in retracted or record_id in retracted:  # rule 4
            continue
        if (r.ref, r.source, r.status) in split:  # mishandled re-report: counted as its own record
            separate.append(_as_record(r, r.ref))
            continue
        groups.setdefault(record_id, []).append(r)  # rule 1

    records: list[Event | Record] = []
    for record_id, group in groups.items():
        settled = [r for r in group if r.status == authoritative]
        unsettled = [r for r in group if r.status != authoritative]
        if record_id in m.prefer_preliminary_for and unsettled:
            base = unsettled[0]
        else:
            base = settled[0] if settled else group[0]  # rule 2
        record = _as_record(base, record_id)
        fix = corrections.get(record_id)
        if fix is not None:  # rule 3
            record = _corrected(record, fix, qualifiers or {})
        records.append(record)
    return records + separate


def naive_events(reports: Iterable[Report | RecordReport]) -> list[Event | Record]:
    """What a careless reader sees: every report line is a separate record; notices are ignored."""
    return [_as_record(r, r.ref) for r in reports]


def _as_record(report: Report | RecordReport, record_id: str) -> Event | Record:
    if isinstance(report, RecordReport):
        return _generic_record(report, record_id)
    return report_to_event(report, record_id)


def _corrected(record: Event | Record, fix: Notice, qualifiers: Mapping[str, str]) -> Event | Record:
    """A record as a correction revises it. An earthquake (0.2.0) correction gives its new magnitude;
    one that gives none (a generic correction of another field, such as a depth, which no 0.2.0
    correction holds) is an error, never silently ignored, as a generic correction that names no field
    is (_generic_correction)."""
    if isinstance(record, Record):
        return _generic_correction(record, fix, qualifiers)
    if fix.new_mag is None:
        revised = f": it revises {fix.field!r}, which an earthquake correction cannot" if fix.field else ""
        raise ValueError(f"the correction of {fix.target} gives no new magnitude{revised}")
    return replace(record, mag=fix.new_mag, mag_type=fix.mag_type or record.mag_type)


# --------------------------------------------------------------------------- generic records
# The RecordReports of a generic domain resolve into Records here. A correction revises the value of one
# field, and that field's qualifier when it gives one.


def _generic_record(report: RecordReport, record_id: str) -> Record:
    return Record(id=record_id, time=report.time, lat=report.lat, lon=report.lon, group=report.group,
                  source=report.source, values=dict(report.values), qualifiers=dict(report.qualifiers),
                  place=report.place)


def _generic_correction(record: Record, fix: Notice, qualifiers: Mapping[str, str]) -> Record:
    """A generic correction replaces the value of the field it names, and that field's qualifier
    (under its key in `qualifiers`) when it gives one. A correction that names no field, a field the
    record lacks, or a qualifier for a field that has none, is an error, never a guess."""
    if fix.field is None:
        raise ValueError(f"the correction of {fix.target} names no field")
    if fix.field not in record.values:
        raise ValueError(f"the correction of {fix.target} names {fix.field!r}, which its record does not have")
    if fix.new_value is None:
        return record
    revised = replace(record, values={**record.values, fix.field: fix.new_value})
    if fix.qualifier is None:
        return revised
    if fix.field not in qualifiers:
        raise ValueError(f"the correction of {fix.target} gives a qualifier, but {fix.field!r} has none")
    return replace(revised, qualifiers={**record.qualifiers, qualifiers[fix.field]: fix.qualifier})
