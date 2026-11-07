"""The public reading rules: in English for the prompt, and in code for the answer key.

The rules are the contract between benchmark and model. They are identical in every digest and
published with the code; only the instances (which quakes, which traps, which thresholds) change.
``resolve`` is the code version of the rules. ``Mishandle`` describes, for one trap, how a reader
who falls for it would resolve the digest instead; running a question under a mishandling gives
the answer a model would produce if it missed that trap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

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

    keep_retracted: tuple[str, ...] = ()  # retraction ignored: the phantom counts as a quake
    ignore_correction_for: tuple[str, ...] = ()  # correction ignored: wrong magnitude kept
    split: tuple[tuple[str, str, str], ...] = ()  # (ref, network, status) re-report counted as its own quake
    prefer_preliminary_for: tuple[str, ...] = ()  # preliminary values used instead of reviewed

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


def resolve(reports: Iterable[Report], notices: Iterable[Notice], mishandle: Mishandle | None = None) -> list[Event]:
    """Apply the reading rules: drop retracted reports, merge re-reports, prefer REVIEWED values,
    apply corrections. Returns one Event per real earthquake."""
    m = mishandle or Mishandle()
    notices = list(notices)
    retracted = {n.target for n in notices if n.kind == "retraction"} - set(m.keep_retracted)
    corrections = {
        n.target: n for n in notices if n.kind == "correction" and n.target not in m.ignore_correction_for
    }
    split = {tuple(pair) for pair in m.split}

    groups: dict[str, list[Report]] = {}
    separate: list[Event] = []
    for r in reports:
        event_id = r.same_as or r.ref
        if r.ref in retracted or event_id in retracted:  # rule 4
            continue
        if (r.ref, r.network, r.status) in split:  # mishandled re-report: counted as its own quake
            separate.append(report_to_event(r, r.ref))
            continue
        groups.setdefault(event_id, []).append(r)  # rule 1

    events: list[Event] = []
    for event_id, group in groups.items():
        reviewed = [r for r in group if r.status == "reviewed"]
        preliminary = [r for r in group if r.status != "reviewed"]
        if event_id in m.prefer_preliminary_for and preliminary:
            base = preliminary[0]
        else:
            base = reviewed[0] if reviewed else group[0]  # rule 2
        event = report_to_event(base, event_id)
        fix = corrections.get(event_id)
        if fix is not None and fix.new_mag is not None:  # rule 3
            event = replace(event, mag=fix.new_mag, mag_type=fix.mag_type or event.mag_type)
        events.append(event)
    return events + separate


def naive_events(reports: Iterable[Report]) -> list[Event]:
    """What a careless reader sees: every report line is a separate quake; notices are ignored."""
    return [report_to_event(r, r.ref) for r in reports]
