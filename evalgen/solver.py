"""Reference solvers that read the digest TEXT (they never see the snapshot or the answer key).

oracle  parses every line, applies the reading rules (rules.resolve) and runs each question's
        operation chain. It must score 100%: that proves every instance is answerable from the
        text alone, with nothing hidden.
naive   parses the same lines but skips deduplication and ignores notices, i.e. it treats every
        report as a separate record. It should lose points exactly where traps matter. In the classic
        genre every report is one line; the other genres' parsers have already set aside what their
        rules say is no report (quoted and read-back repeats, message times), so naive never sees a
        genre's noise: its answers depend only on the reports that a digest holds.

Both receive each question's operation chain (its machine-readable meaning) instead of parsing
the English wording: the point is to test the digest, not to build a language-understanding
system. Language models, of course, only ever get the English prompt.

The line parsers below read the classic genre's 0.2.0 digest of earthquakes. A prompt of a non-legacy
run is read from its text alone: the glossary it prints (genres.base.parse_glossary), and the parser of
the genre that recognises its digest (genres.detect_genre), unless the caller names the genre.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Sequence

from .digest import MONTHS, MONTHS_LONG, WEEKDAYS
from .domains import get_domain
from .domains.base import Domain
from .genres import detect_genre, get_genre
from .genres.base import Genre, parse_glossary
from .geo import KM_PER_MILE, haversine_km
from .models import EVENT_ID_PATTERN, Notice, Report
from .ops import DigestData, Units, execute
from .rules import resolve

ID = r"(?<![0-9a-z])" + EVENT_ID_PATTERN + r"(?![0-9a-z])"
NUM = r"-?\d+(?:\.\d+)?"
MAG_TYPE = r"[a-z0-9_]+"
MON = "|".join(MONTHS)

_DIGEST_RE = re.compile(r"<digest>\n?(.*?)\n?</digest>", re.DOTALL)

_TIME_RES = (
    ("iso", re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z")),
    ("iso_space", re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) UTC")),
    ("rfc", re.compile(r"(?:" + "|".join(WEEKDAYS) + r"), (\d{2}) (" + MON + r") (\d{4}) (\d{2}):(\d{2}):(\d{2}) \+0000")),
    ("prose_utc", re.compile(r"(\d{2}):(\d{2}) UTC on (\d{1,2}) (" + "|".join(MONTHS_LONG) + r") (\d{4})")),
    ("local12", re.compile(r"(\d{1,2}):(\d{2}) (AM|PM) local time \(UTC([+-])(\d{2}):00\) on (" + MON + r") (\d{1,2}), (\d{4})")),
    ("local24", re.compile(r"(\d{2}):(\d{2}) \(UTC([+-])(\d{2}):00\) (\d{1,2}) (" + MON + r") (\d{4})")),
)
_DDM = re.compile(r"(\d+)°(\d+(?:\.\d+)?)'([NS]),?\s+(\d+)°(\d+(?:\.\d+)?)'([EW])")
_HEMI = re.compile(r"(\d+(?:\.\d+)?)°([NS]),\s*(\d+(?:\.\d+)?)°([EW])")
_SIGNED = re.compile(r"\((" + NUM + r"), (" + NUM + r")\)")
_DEPTH = re.compile(r"(" + NUM + r") (km|mi)\b")
_XREF = re.compile(r"(?:same event as|same earthquake as|Dup of) (" + ID + r")")

_BULLETIN_HEAD = re.compile(r"^\[([A-Z]{2,4})\] (" + ID + r")$")
_BULLETIN_MAG = (re.compile(r"^M(" + NUM + r") \((" + MAG_TYPE + r")\)$"), re.compile(r"^([A-Z0-9_]+) (" + NUM + r")$"))
_PROSE = re.compile(
    r"^(?P<time>.+?) — A magnitude (?P<mag>" + NUM + r") \((?P<mt>" + MAG_TYPE + r")\) earthquake struck "
    r"(?P<region>.+?); epicentre (?P<coords>.+?), depth (?P<depth>.+?)\. The (?P<net>[A-Z]{2,4}) network lists it as "
    r"(?P<ref>" + ID + r") \((?P<status>reviewed|preliminary)\)\.(?P<rest>.*)$")
_WIRE = re.compile(
    r"^(?P<net>[A-Z]{2,4}) WIRE \[(?P<time>.+?)\] (?P<mt>[A-Z0-9_]+) (?P<mag>" + NUM + r") quake in (?P<region>.+?) @ "
    r"(?P<coords>.+?); (?P<depth>.+?) deep\. Ref (?P<ref>" + ID + r"), (?P<status>reviewed|preliminary)\.(?P<rest>.*)$")
_LOG = re.compile(
    r"^(?P<time>.+?) ref=(?P<ref>" + ID + r") net=(?P<net>[a-z]{2,4}) mag=(?P<mag>" + NUM + r") type=(?P<mt>"
    + MAG_TYPE + r") region=\"(?P<region>[^\"]+)\" loc=(?P<lat>" + NUM + r"),(?P<lon>" + NUM + r") depth=(?P<depth>.+?) "
    r"status=(?P<status>reviewed|preliminary)(?: same_as=(?P<same>" + ID + r"))?$")

_NOTICE = re.compile(r"^(?:NOTICE|RETRACTION|UPDATE|CORRECTION|Revised) \[(?P<time>[^\]]+)\]: (?P<body>.*)$")
_RETRACT_WORDS = re.compile(r"RETRACTED|was not an earthquake|withdrawn")
_CORRECTIONS = (
    re.compile(r"revised to (?P<new>" + NUM + r") \((?P<mt>" + MAG_TYPE + r")\)"),
    re.compile(r"revised from " + NUM + r" to (?P<new>" + NUM + r") (?P<mt>" + MAG_TYPE + r")"),
    re.compile(r"magnitude now (?P<mt>[A-Z0-9_]+) (?P<new>" + NUM + r")"),
)


# ---------------------------------------------------------------------------- field parsers


def extract_digest(prompt: str) -> str:
    match = _DIGEST_RE.search(prompt)
    return match.group(1) if match else prompt


def parse_time(text: str) -> int | None:
    for kind, rx in _TIME_RES:
        m = rx.search(text)
        if not m:
            continue
        g = m.groups()
        offset = 0
        if kind in ("iso", "iso_space"):
            y, mo, d, hh, mi, ss = (int(x) for x in g)
        elif kind == "rfc":
            d, mo, y, hh, mi, ss = int(g[0]), MONTHS.index(g[1]) + 1, int(g[2]), int(g[3]), int(g[4]), int(g[5])
        elif kind == "prose_utc":
            hh, mi, d, mo, y, ss = int(g[0]), int(g[1]), int(g[2]), MONTHS_LONG.index(g[3]) + 1, int(g[4]), 0
        elif kind == "local12":
            hh = int(g[0]) % 12 + (12 if g[2] == "PM" else 0)
            mi, ss = int(g[1]), 0
            offset = int(g[4]) * (1 if g[3] == "+" else -1)
            mo, d, y = MONTHS.index(g[5]) + 1, int(g[6]), int(g[7])
        else:  # local24
            hh, mi, ss = int(g[0]), int(g[1]), 0
            offset = int(g[3]) * (1 if g[2] == "+" else -1)
            d, mo, y = int(g[4]), MONTHS.index(g[5]) + 1, int(g[6])
        local = datetime(y, mo, d, hh, mi, ss, tzinfo=timezone.utc)
        return int(local.timestamp()) - offset * 3600  # local = UTC + offset
    return None


def parse_coords(text: str) -> tuple[float, float] | None:
    m = _DDM.search(text)
    if m:
        lat = int(m.group(1)) + float(m.group(2)) / 60
        lon = int(m.group(4)) + float(m.group(5)) / 60
        return (lat if m.group(3) == "N" else -lat), (lon if m.group(6) == "E" else -lon)
    m = _HEMI.search(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(3))
        return (lat if m.group(2) == "N" else -lat), (lon if m.group(4) == "E" else -lon)
    m = _SIGNED.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_depth_km(text: str) -> float | None:
    m = _DEPTH.search(text)  # with "6.2 km (3.9 mi)" the first, more precise value wins
    if not m:
        return None
    value = float(m.group(1))
    return value if m.group(2) == "km" else value * KM_PER_MILE


def _report(ref, network, status, when, coords, depth, mag, mag_type, region, same_as) -> Report | None:
    if when is None or coords is None or depth is None:
        return None
    return Report(ref=ref, network=network.lower(), status=status, time=when, lat=coords[0], lon=coords[1],
                  depth_km=depth, mag=float(mag), mag_type=mag_type.lower(), region=region, same_as=same_as)


def parse_report_line(line: str) -> Report | None:
    if line.startswith("[") and " | " in line:
        parts = line.split(" | ")
        head = _BULLETIN_HEAD.match(parts[0])
        if not head or len(parts) < 7:
            return None
        mag = mag_type = None
        m = _BULLETIN_MAG[0].match(parts[5])
        if m:
            mag, mag_type = m.group(1), m.group(2)
        else:
            m = _BULLETIN_MAG[1].match(parts[5])
            if m:
                mag_type, mag = m.group(1), m.group(2)
        if mag is None:
            return None
        same = next((x.group(1) for x in map(_XREF.search, parts[7:]) if x), None)
        return _report(head.group(2), head.group(1), parts[6].strip().lower(), parse_time(parts[1]),
                       parse_coords(parts[3]), parse_depth_km(parts[4]), mag, mag_type, parts[2], same)
    for rx in (_PROSE, _WIRE):
        m = rx.match(line)
        if m:
            xref = _XREF.search(m.group("rest"))
            return _report(m.group("ref"), m.group("net"), m.group("status"), parse_time(m.group("time")),
                           parse_coords(m.group("coords")), parse_depth_km(m.group("depth")), m.group("mag"),
                           m.group("mt"), m.group("region"), xref.group(1) if xref else None)
    m = _LOG.match(line)
    if m:
        return _report(m.group("ref"), m.group("net"), m.group("status"), parse_time(m.group("time")),
                       (float(m.group("lat")), float(m.group("lon"))), parse_depth_km(m.group("depth")),
                       m.group("mag"), m.group("mt"), m.group("region"), m.group("same"))
    return None


def parse_notice_line(line: str) -> Notice | None:
    m = _NOTICE.match(line)
    if not m:
        return None
    body = m.group("body")
    target = re.search(ID, body)
    when = parse_time(m.group("time"))
    if not target or when is None:
        return None
    if _RETRACT_WORDS.search(body):
        return Notice(kind="retraction", target=target.group(0), time=when)
    for rx in _CORRECTIONS:
        c = rx.search(body)
        if c:
            return Notice(kind="correction", target=target.group(0), time=when,
                          new_mag=float(c.group("new")), mag_type=c.group("mt").lower())
    return None


def parse_digest(text: str) -> DigestData:
    reports, notices = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        notice = parse_notice_line(line)
        if notice is not None:
            notices.append(notice)
            continue
        report = parse_report_line(line)
        if report is not None:
            reports.append(report)
    return DigestData(reports, notices)


# ---------------------------------------------------------------------------------- solving


def read_digest(prompt: str, genre: str | None = None, domain: Domain | None = None, *,
                readers: Sequence[Genre] | None = None) -> DigestData:
    """The reports and notices of a prompt's digest about `domain`'s records (earthquakes when not
    given), read through the glossary the prompt prints (if any) by the parser of the genre that wrote
    it: `genre` when it is known, or else the genre that recognises the digest (genres.detect_genre,
    which falls back to classic), among `readers` when they are given."""
    text = extract_digest(prompt)
    reader = get_genre(genre) if genre is not None else detect_genre(text, readers)
    return reader.parse(text, parse_glossary(prompt), domain if domain is not None else get_domain())


def _reading(prompt: str, domain: str | None, genre: str | None,
             readers: Sequence[Genre] | None = None) -> tuple[DigestData, Domain | None, Units | None]:
    """A prompt's digest data, its domain and the units its chains use. Without a domain or a genre
    the prompt is a legacy one (earthquakes, classic), read exactly as 0.2.0 read it, and the domain
    and units are None. Otherwise the prompt's glossary is read from its text, and the units are those
    it prints (a domain whose unit words are drawn per instance is read through it), or the domain's
    when it prints none."""
    if domain is None and genre is None:
        return parse_digest(extract_digest(prompt)), None, None
    records_domain = get_domain(domain)
    glossary = parse_glossary(prompt)
    units = records_domain.units if glossary is None else glossary.units
    return read_digest(prompt, genre, records_domain, readers=readers), records_domain, units


def solve(prompt: str, questions: Sequence[dict], mode: str = "oracle", *, domain: str | None = None,
          genre: str | None = None, readers: Sequence[Genre] | None = None) -> dict[str, Any]:
    """Answer every question from the prompt text. mode: "oracle" or "naive".

    `domain` and `genre` name the instance's domain and the rendering's genre, as a non-legacy run
    records them; without either, the prompt is read as 0.2.0 read it. With the domain alone, the
    prompt is read from its text: its glossary, and the genre that recognises its digest (falling back
    to classic), sniffed among `readers` when they are given (genres.detect_genre). The oracle applies
    the reading rules of the instance's domain (Domain.resolve)."""
    data, records_domain, units = _reading(prompt, domain, genre, readers)
    resolver = resolve if records_domain is None else records_domain.resolve
    skip = ("dedupe",) if mode == "naive" else ()
    return {q["qid"]: execute(q["chain"], data, skip=skip, units=units, resolver=resolver) for q in questions}


def forecast_estimate(prompt: str, rule: dict, *, domain: str | None = None, genre: str | None = None) -> float:
    """Oracle's forecast: a naive base rate from M>=min_mag events near the point in the digest.
    `domain` and `genre` are read as in ``solve``."""
    data, records_domain, _ = _reading(prompt, domain, genre)
    events = (resolve(data.reports, data.notices) if records_domain is None
              else records_domain.resolve(data.reports, data.notices))
    nearby = [
        e for e in events
        if e.mag >= rule["min_mag"] and haversine_km(rule["lat"], rule["lon"], e.lat, e.lon) <= rule["radius_km"]
    ]
    rate_per_day = len(nearby) * 24.0 / max(1, rule.get("digest_window_hours", 24))
    p = 1.0 - math.exp(-(rate_per_day + 0.05))
    return round(min(0.95, max(0.02, p)), 3)
