"""Base data: a seeded synthetic world (default, fully offline) or a frozen USGS feed.

Either way the result is a ``Snapshot``: a list of canonical events plus its SHA-256 hash and the
seed. Everything downstream (digests, questions, answer key) is computed from the snapshot, so
re-using a saved snapshot with the same seed reproduces a run exactly, even for live USGS data.

A legacy run builds its earthquake snapshot with ``build_snapshot``, as 0.2.0 did. A run off the
legacy path builds the snapshot of its domain with ``build_domain_snapshot``: the domain's synthetic
records, drawn on seed paths of their own, or its live source, fetched once, frozen, hashed and
replayable, with the synthetic records as the fallback. Its records are the domain's own (Records,
or Events for earthquakes), and ``save_snapshot`` and ``load_snapshot`` handle either. A run of
several domains freezes one snapshot per domain; ``frozen_feed`` finds, in such a run's snapshot
folder, what replays each domain: its frozen bytes, or nothing when the run drew synthetic records.

Every live request goes through ``http_get``: a session of its own that reads nothing from the environment
but the proxies that its variables name (no netrc credentials, no CA bundle), bounded as a whole
(FETCH_DEADLINE_S), not only read by read. Bytes that a source's parser cannot read make it fall back to
synthetic data; a value in them that cannot be real (not a finite number, out of range, outside the
feed's window) is left out. A replay checks what a frozen snapshot's snapshot.json says against the bytes
beside it (``check_sidecar``) and refuses what was changed instead of using it.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

from . import __version__
from .domains import DEFAULT_DOMAIN
from .domains.base import Domain, Record
from .geo import destination, haversine_km
from .models import EVENT_ID_RE, Event
from .seeds import SeedLog
from .util import canonical_json, read_json, sha256_hex, write_in, write_json

USGS_FEEDS = {
    "hour": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
    "day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
}
USER_AGENT = f"evalgen/{__version__} (research demo: live LLM evaluation)"
MIN_POOL_EVENTS = 12  # fewer usable events than this and we widen the feed or fall back
# Coordinates are stored at the precision the digest prints (3 decimals; degrees-and-minutes with
# 2-decimal minutes is exact for these), so every distance key can be reproduced from the text.
COORD_DECIMALS = 3
# Two solutions closer than this in time and space are treated as one earthquake reported twice.
DUPLICATE_WINDOW_S, DUPLICATE_RADIUS_KM = 3, 15.0

ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"

# What a live record or snapshot can be. A time outside these two dates, a magnitude above 10, a position
# off the globe or a depth beyond the mantle's is no reading of the world: it is left out of a feed. A
# snapshot off the legacy path covers at least MIN_WINDOW_HOURS (a planted phantom's notice comes 92 minutes
# after it, within the window: traps.NOTICE_DELAY), as the USGS hour feed does not, and at most a year.
EARLIEST_TIME = int(datetime(1800, 1, 1, tzinfo=timezone.utc).timestamp())
LATEST_TIME = int(datetime(2200, 1, 1, tzinfo=timezone.utc).timestamp())
MAX_MAGNITUDE = 10.0
DEPTH_RANGE_KM = (-100.0, 1000.0)
MIN_WINDOW_HOURS, MAX_WINDOW_HOURS = 2, 366 * 24


@dataclass
class Snapshot:
    source: str  # "synthetic", or a live source: "usgs"
    seed: int
    compiled_at: int  # end of the covered window (Unix seconds); digests are "compiled" here
    window_hours: int
    events: list[Any]  # the records: Events in a legacy run, else the domain's own (Records)
    events_sha256: str
    raw_sha256: str | None = None  # hash of the frozen feed bytes (live sources only)
    url: str | None = None
    # When the live source was fetched, by the wall clock: like the manifest's generated_at_utc it differs
    # between two fetches of the same bytes. A replay off the legacy path keeps the time its frozen snapshot
    # records; a legacy replay is stamped with the time it ran, as 0.2.0 stamped it.
    fetched_at: str | None = None
    note: str | None = None  # e.g. why we fell back to synthetic data
    domain: str | None = None  # the domain of a snapshot off the legacy path; unset (and unrecorded) in legacy runs

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("events")
        if self.domain is None:
            d.pop("domain")
        d["n_events"] = len(self.events)
        return d


def events_sha256(events: Sequence[Any]) -> str:
    """SHA-256 of the records (Events or Records) as canonical JSON, in ID order."""
    return sha256_hex(canonical_json([asdict(e) for e in sorted(events, key=lambda e: e.id)]))


def collapse_duplicate_solutions(events: list[Event], rank: Callable[[Event], tuple]) -> tuple[list[Event], list[str]]:
    """Keep one event per cluster of solutions within 3 s and 15 km of each other.

    Real feeds sometimes list one earthquake twice (a regional and a global solution, or an
    automatic and a reviewed one). Left in, both would be printed as REVIEWED reports of two quakes,
    which is exactly what a planted duplicate looks like and would make "distinct earthquakes"
    ambiguous. `rank` orders the candidates; the lowest rank is kept. Returns (kept, dropped IDs).
    """
    kept: list[Event] = []
    dropped: list[str] = []
    for e in sorted(events, key=lambda x: (x.time, x.id)):
        clash = None
        for i in range(len(kept) - 1, -1, -1):
            if e.time - kept[i].time > 2 * DUPLICATE_WINDOW_S:  # replacements may reorder within a window
                break
            if (abs(e.time - kept[i].time) <= DUPLICATE_WINDOW_S
                    and haversine_km(kept[i].lat, kept[i].lon, e.lat, e.lon) <= DUPLICATE_RADIUS_KM):
                clash = i
                break
        if clash is None:
            kept.append(e)
        elif rank(e) < rank(kept[clash]):
            dropped.append(kept[clash].id)
            kept[clash] = e
        else:
            dropped.append(e.id)
    kept.sort(key=lambda x: (x.time, x.id))
    return kept, dropped


def new_event_id(network: str, rng: random.Random, taken: set[str]) -> str:
    """A fresh USGS-style ID that is not already in use."""
    while True:
        if network == "us":
            ref = f"us{rng.randrange(6000, 8000)}{''.join(rng.choice(ALNUM) for _ in range(4))}"
        else:
            ref = f"{network}{rng.randrange(10_000_000, 100_000_000)}"
        if ref not in taken:
            taken.add(ref)
            return ref


def mimic_event_id(example: str, rng: random.Random, taken: set[str]) -> str:
    """A fresh ID in the same style as a real one, so planted reports carry no tell-tale format:
    'aka2026vupcsv' -> 'aka2026qhxtrb', 'ci41572136' -> 'ci41579904'."""
    m = re.match(r"([a-z]+)(\d{4})?(.*)$", example)
    prefix, rest = (m.group(1) + (m.group(2) or "")), m.group(3)
    for _ in range(1000):
        body = "".join(rng.choice("0123456789") if c.isdigit() else rng.choice(ALNUM[:26]) if c.isalpha() else c
                       for c in rest)
        ref = prefix + body
        if ref not in taken and EVENT_ID_RE.fullmatch(ref):
            taken.add(ref)
            return ref
    return new_event_id(re.match(r"[a-z]+", example).group(0)[:2], rng, taken)


# --------------------------------------------------------------------------- synthetic world


@dataclass(frozen=True)
class SynthRegion:
    name: str
    lat: float
    lon: float
    spread_km: float
    depth_km: tuple[float, float]
    network: str
    weight: float


SYNTH_REGIONS = (
    SynthRegion("Northern California", 38.80, -122.80, 45, (1.0, 14.0), "nc", 1.4),
    SynthRegion("Southern California", 34.05, -117.20, 60, (2.0, 18.0), "ci", 1.5),
    SynthRegion("Southern Alaska", 60.60, -151.80, 140, (5.0, 120.0), "ak", 1.6),
    SynthRegion("Hawaii", 19.40, -155.30, 35, (0.5, 40.0), "hv", 1.0),
    SynthRegion("Puerto Rico", 18.05, -66.85, 40, (4.0, 30.0), "pr", 0.8),
    SynthRegion("Nevada", 38.60, -118.00, 70, (1.0, 15.0), "nn", 0.8),
    SynthRegion("Utah", 39.30, -111.70, 60, (1.0, 12.0), "uu", 0.8),
    SynthRegion("Central Chile", -33.20, -71.70, 110, (10.0, 120.0), "us", 0.6),
    SynthRegion("Honshu, Japan", 37.40, 141.60, 120, (10.0, 90.0), "us", 0.6),
    SynthRegion("Tonga", -20.20, -174.60, 140, (10.0, 300.0), "us", 0.5),
)
LOCAL_MAG_TYPE = {"nc": "md", "hv": "md", "pr": "md", "ci": "ml", "ak": "ml", "nn": "ml", "uu": "ml"}


def _synthetic_magnitude(rng: random.Random, network: str) -> float:
    """Gutenberg-Richter style: exponential above a completeness magnitude."""
    if network == "us":  # global network: only larger events are catalogued abroad
        mag = 4.0 + rng.expovariate(1.0 * math.log(10))
    else:  # local networks; a gentle b-value keeps digests interesting
        mag = 0.8 + rng.expovariate(0.6 * math.log(10))
    return round(min(mag, 7.8), 1)


def _synthetic_mag_type(rng: random.Random, network: str, mag: float) -> str:
    if network == "us":
        return "mww" if mag >= 5.5 else "mb"
    if mag >= 4.0 and rng.random() < 0.5:
        return "mw"
    return LOCAL_MAG_TYPE.get(network, "ml")


synthetic_mag_type = _synthetic_mag_type  # the magnitude type a network gives a synthetic event (also off the legacy path)


def generate_synthetic(rng: random.Random, n_events: int, compiled_at: int, window_hours: int = 24) -> list[Event]:
    taken: set[str] = set()
    weights = [r.weight for r in SYNTH_REGIONS]
    events = []
    for _ in range(n_events):
        region = rng.choices(SYNTH_REGIONS, weights=weights, k=1)[0]
        lat, lon = destination(region.lat, region.lon, abs(rng.gauss(0.0, region.spread_km)), rng.uniform(0.0, 360.0))
        lo, hi = region.depth_km
        depth = lo + (hi - lo) * rng.random() ** 1.6  # most events shallow within the range
        mag = _synthetic_magnitude(rng, region.network)
        events.append(Event(
            id=new_event_id(region.network, rng, taken),
            time=compiled_at - rng.randrange(60, window_hours * 3600),
            lat=round(lat, COORD_DECIMALS),
            lon=round(lon, COORD_DECIMALS),
            depth_km=round(depth, 1),
            mag=mag,
            mag_type=_synthetic_mag_type(rng, region.network, mag),
            region=region.name,
            network=region.network,
        ))
    events, _ = collapse_duplicate_solutions(events, rank=lambda e: (e.time, e.id))
    return events


def synthetic_compiled_at(rng: random.Random) -> int:
    """A reproducible 'as of' time in 2026 (never the wall clock, so runs are reproducible)."""
    base = int(datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc).timestamp())
    return base + rng.randrange(0, 300) * 86400


# --------------------------------------------------------------------------------- USGS feed

PLACE_ABBREVIATIONS = {"CA": "California", "MX": "Mexico", "NV": "Nevada", "B.C.": "Mexico"}
# "off the coast of" and similar location phrases are stripped once; then adjectival qualifiers
# ("Southern Alaska" -> "Alaska"). Bare compass words are kept, because they belong to proper names
# (West Virginia, South Sandwich Islands, North Korea), and so are names like "Central America".
_LOCATION = re.compile(r"^(?:(?:off|near) the (?:(?:north|south|east|west)(?:ern)? )?coast of|near the|far)\s+",
                       re.IGNORECASE)
_QUALIFIER = re.compile(r"^(?:(?:central|northern|southern|eastern|western|northeastern|northwestern|"
                        r"southeastern|southwestern)\s+)+", re.IGNORECASE)
_RELATIVE = re.compile(r"^(?:north|south|east|west|northeast|northwest|southeast|southwest)\s+of\s+", re.IGNORECASE)
_KEEP_QUALIFIED = {"northern mariana islands", "western australia", "central african republic", "northern territory",
                   "western sahara", "southern ocean", "eastern cape", "western cape", "northern cape",
                   "central america", "central asia", "east timor"}


def region_from_place(place: str | None) -> str:
    """Coarse region from a USGS place string: '6 km NE of Delta, B.C., MX' -> 'Mexico',
    'Southern Alaska' -> 'Alaska', 'Japan region' -> 'Japan', 'south of Panama' -> 'South of Panama'."""
    if not place:
        return "Unknown region"
    tail = place.rsplit(", ", 1)[-1].strip()
    tail = PLACE_ABBREVIATIONS.get(tail, tail)
    tail = re.sub(r"\s+region$", "", tail, flags=re.IGNORECASE).strip()
    if not _RELATIVE.match(tail):
        tail = _LOCATION.sub("", tail, count=1).strip()
        if tail.lower() not in _KEEP_QUALIFIED:
            tail = _QUALIFIER.sub("", tail).strip()
    return (tail[:1].upper() + tail[1:]) if tail else "Unknown region"


def read_feed_json(raw: bytes, what: str) -> Any:
    """`raw` read as JSON text in UTF-8. ValueError for bytes that are not, and for JSON nested too deeply
    to read (which the json module reports as a RecursionError), so that a live source answering with
    either falls back to synthetic data, and a frozen copy of either is refused, as for any bytes that
    are not its feed."""
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except RecursionError:
        raise ValueError(f"{what} is JSON nested too deeply to read") from None


def parse_usgs_geojson(raw: bytes) -> tuple[list[Event], int, list[str]]:
    """Parse a USGS GeoJSON summary feed: Events (earthquakes only, one per quake), the feed's
    generation time, and the IDs of duplicate solutions that were collapsed. Bytes that are not such a
    feed, whether not JSON or JSON of another shape, raise ValueError, so that a live source that answers
    with them falls back to synthetic data as one that cannot be reached does. An earthquake that no real
    feed lists is left out, as one without a magnitude is: a magnitude, a position or a depth that is not
    a finite number or not in range (a magnitude above MAX_MAGNITUDE, a latitude beyond a pole, a depth
    outside DEPTH_RANGE_KM), or a time outside EARLIEST_TIME..LATEST_TIME; a feed generated outside them
    is no feed (ValueError)."""
    return parse_usgs_document(usgs_document(raw))


def usgs_document(raw: bytes) -> dict:
    """A USGS GeoJSON feed read as JSON: an object with a list of features (ValueError otherwise)."""
    data = read_feed_json(raw, "the USGS feed")
    if not isinstance(data, dict) or not isinstance(data.get("features", []), list):
        raise ValueError("not a USGS GeoJSON feed: it is not an object with a list of features")
    return data


def parse_usgs_document(data: dict) -> tuple[list[Event], int, list[str]]:
    """The events of a USGS feed read as JSON (usgs_document), as parse_usgs_geojson gives them."""
    try:
        return _usgs_events(data)
    except (AttributeError, TypeError, IndexError, OverflowError) as exc:  # a feature, its properties or metadata
        raise ValueError(f"not a USGS GeoJSON feed ({type(exc).__name__}: {exc})") from None  # of another shape


def _real_quake(mag: float, lat: float, lon: float, depth: float, when: Any) -> bool:
    """Whether a feed's earthquake could be a real one: a finite magnitude of at most MAX_MAGNITUDE, a place
    on the globe, a depth in DEPTH_RANGE_KM, and a time (in milliseconds) between EARLIEST_TIME and
    LATEST_TIME. The time is checked before it is read as a whole number, which an infinite one is not."""
    if not all(math.isfinite(x) for x in (mag, lat, lon, depth)):
        return False
    if mag > MAX_MAGNITUDE or not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return False
    if not DEPTH_RANGE_KM[0] <= depth <= DEPTH_RANGE_KM[1]:
        return False
    if isinstance(when, float) and not math.isfinite(when):
        return False
    return EARLIEST_TIME <= int(when) // 1000 <= LATEST_TIME


def _usgs_events(data: dict) -> tuple[list[Event], int, list[str]]:
    """The events of a USGS GeoJSON feed read as JSON (parse_usgs_geojson)."""
    events = []
    preference: dict[str, tuple] = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        event_id = str(feature.get("id", "")).lower()
        mag = props.get("mag")
        if (props.get("type") or "earthquake") != "earthquake":  # quarry blasts, explosions, ...
            continue
        if mag is None or float(mag) < 0 or len(coords) < 3 or props.get("time") is None:
            continue
        if not EVENT_ID_RE.fullmatch(event_id):
            continue
        lon, lat, depth = (float(c) for c in coords[:3])
        if not _real_quake(float(mag), lat, lon, depth, props["time"]):
            continue
        place = props.get("place")
        events.append(Event(
            id=event_id,
            time=int(props["time"]) // 1000,
            lat=round(lat, COORD_DECIMALS),
            lon=round(lon, COORD_DECIMALS),
            depth_km=round(depth, 1),
            mag=round(float(mag), 2),
            mag_type=str(props.get("magType") or "ml").lower(),
            region=region_from_place(place),
            network=str(props.get("net") or event_id[:2]).lower(),
            place=place,
        ))
        # keep the reviewed solution of a duplicated quake, then the more significant one
        preference[event_id] = (0 if props.get("status") == "reviewed" else 1, -(props.get("sig") or 0), event_id)
    generated_ms = int((data.get("metadata") or {}).get("generated") or 0)
    if not generated_ms:
        generated_ms = max((e.time for e in events), default=0) * 1000 + 1000
    if not EARLIEST_TIME <= generated_ms // 1000 <= LATEST_TIME:
        raise ValueError(f"not a USGS GeoJSON feed: it says it was generated at {generated_ms} ms, which no feed was")
    events, dropped = collapse_duplicate_solutions(events, rank=lambda e: preference[e.id])
    return events, generated_ms // 1000, dropped


# The summary feeds of the USGS, by the period their name ends in ("all_day.geojson"): the hours each covers.
USGS_PERIODS = {"hour": 1, "day": 24, "week": 7 * 24, "month": 30 * 24}
_PERIOD = re.compile(r"_(hour|day|week|month)\b")


def usgs_window(data: dict) -> int:
    """The hours that a USGS summary feed read as JSON covers: the period its metadata.url names (the hour,
    day, week or month feed), and a day when it names none."""
    metadata = data.get("metadata")
    url = metadata.get("url") if isinstance(metadata, dict) else None
    found = _PERIOD.search(url) if isinstance(url, str) else None
    return USGS_PERIODS[found.group(1)] if found else 24


# ------------------------------------------------------------------------------- live requests

# Each live request waits at most its timeout to connect and for each read, and at most FETCH_DEADLINE_S
# for the whole answer, from the first connection to the last byte of a body of at most MAX_BODY_BYTES.
FETCH_DEADLINE_S = 60.0
MAX_BODY_BYTES = 64 * 2 ** 20
_CHUNK_BYTES = 64 * 1024


class FetchTimeout(requests.exceptions.Timeout):
    """A live request that took longer than its deadline as a whole, however quick each read of it was."""


class BodyTooLarge(requests.exceptions.RequestException):
    """A live answer whose body is longer than MAX_BODY_BYTES."""


def proxies_for(url: str) -> dict[str, str]:
    """The proxies that a live request to `url` goes through: those the environment's proxy variables name
    explicitly (http_proxy, https_proxy and all_proxy, in either case), unless no_proxy exempts the URL's
    host. Nothing else of the environment is read: no netrc file, no proxy settings of the system."""
    named = urllib.request.getproxies_environment()
    host = urllib.parse.urlsplit(url).hostname or ""
    if not named or urllib.request.proxy_bypass_environment(host, named):
        return {}
    return {scheme: proxy for scheme, proxy in sorted(named.items()) if scheme in ("http", "https", "all")}


def http_get(url: str, *, params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None,
             timeout: float = 20.0, deadline: float | None = None) -> requests.Response:
    """GET `url` for a live source: the response, with its body read in full whatever its status (the caller
    decides what an error status means).

    The request goes through a session of its own with trust_env off, so it sends no credentials from a
    netrc file and reads no CA bundle or proxy settings from the environment; it goes through a proxy only
    when the environment's proxy variables name one (proxies_for). It waits at most `timeout` seconds to
    connect and for each read, and at most `deadline` seconds (FETCH_DEADLINE_S unless given) for the whole
    answer: a server that sends its answer a byte at a time, or that holds the connection open and never
    answers, is given up once the deadline passes (FetchTimeout), and so is a body longer than
    MAX_BODY_BYTES (BodyTooLarge). The request runs on a thread of its own, which the caller stops waiting
    for at the deadline, wherever it is (connecting, waiting for the headers, reading the body)."""
    limit = FETCH_DEADLINE_S if deadline is None else deadline
    until = time.monotonic() + limit
    wait = max(0.001, min(timeout, limit))
    late = f"{urllib.parse.urlsplit(url).hostname or url} did not answer in full within {limit:g} s"
    outcome: dict[str, Any] = {}

    def fetch() -> None:
        try:
            with requests.Session() as session:
                session.trust_env = False
                answer = session.get(url, params=params, headers=headers, timeout=(wait, wait), stream=True,
                                     proxies=proxies_for(url))
                outcome["streaming"] = answer
                try:
                    body = bytearray()
                    for chunk in answer.iter_content(_CHUNK_BYTES):
                        body += chunk
                        if len(body) > MAX_BODY_BYTES:
                            raise BodyTooLarge(f"{urllib.parse.urlsplit(url).hostname} answered with more than "
                                               f"{MAX_BODY_BYTES} bytes")
                        if time.monotonic() > until:
                            raise FetchTimeout(late)
                    answer._content = bytes(body)  # what Response.content gives from now on
                finally:
                    answer.close()
            outcome["response"] = answer
        except BaseException as exc:  # handed to the caller as it is
            outcome["error"] = exc

    worker = threading.Thread(target=fetch, name="evalgen-live-request", daemon=True)
    worker.start()
    worker.join(max(0.0, until - time.monotonic()))
    if worker.is_alive():  # still connecting, waiting or reading: given up, and its connection closed
        _abandon(outcome.get("streaming"))
        raise FetchTimeout(late)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]


def _abandon(answer: requests.Response | None) -> None:
    """Shut the connection of an answer given up on, so that the thread reading it stops at once rather
    than at the server's next byte. (Shutting down, unlike closing, wakes a read blocked on the socket.)
    The socket is the connection's, or, for an answer after which the server closes the connection, the
    one its body is read from, which http.client keeps in the response's file alone."""
    raw = getattr(answer, "raw", None)
    sock = getattr(getattr(raw, "connection", None), "sock", None)
    if sock is None:
        body = getattr(getattr(raw, "_fp", None), "fp", None)
        sock = getattr(getattr(body, "raw", None), "_sock", None)
    if isinstance(sock, socket.socket):
        with contextlib.suppress(OSError):
            socket.socket.shutdown(sock, socket.SHUT_RDWR)


def fetch_usgs(feed: str, timeout: float = 20.0) -> tuple[bytes, str]:
    url = USGS_FEEDS[feed]
    response = http_get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.content, url


def _usgs_snapshot(raw: bytes, seed: int, url: str, window_hours: int, note: str | None = None) -> Snapshot:
    events, generated, dropped = parse_usgs_geojson(raw)
    if dropped:
        collapsed = (f"collapsed {len(dropped)} duplicate feed solution(s) of the same quake "
                     f"({', '.join(dropped[:6])}{', ...' if len(dropped) > 6 else ''})")
        note = f"{note}; {collapsed}" if note else collapsed
    return Snapshot(
        source="usgs",
        seed=seed,
        compiled_at=generated,
        window_hours=window_hours,
        events=events,
        events_sha256=events_sha256(events),
        raw_sha256=sha256_hex(raw),
        url=url,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        note=note,
    )


def _synthetic_snapshot(seedlog: SeedLog, pool_size: int, note: str | None = None) -> Snapshot:
    compiled_at = synthetic_compiled_at(seedlog.rng("snapshot", "clock"))
    events = generate_synthetic(seedlog.rng("snapshot", "synthetic"), pool_size, compiled_at)
    return Snapshot(
        source="synthetic",
        seed=seedlog.master,
        compiled_at=compiled_at,
        window_hours=24,
        events=events,
        events_sha256=events_sha256(events),
        note=note,
    )


SIDECAR_TYPES = {"source": (str,), "domain": (str,), "url": (str, type(None)), "window_hours": (int,),
                 "fetched_at": (str, type(None)), "raw_sha256": (str, type(None))}
LEGACY_WINDOWS = (1, 24)  # the windows a legacy run records: its hour feed's or its day feed's
_SHA256 = re.compile(r"[0-9a-f]{64}")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:[^\x00-\x1f\x7f]+")  # a scheme, a colon, then no control character
_FETCHED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def read_sidecar(path: Path) -> dict[str, Any]:
    """The snapshot.json at `path`, as a replay reads it beside frozen bytes: a JSON object whose keys that
    a replay reads have the types that save_snapshot writes (SIDECAR_TYPES; a window is a whole number of
    hours). ValueError, naming the file, otherwise: frozen snapshots come from someone else's run as often
    as from one's own."""
    try:
        meta = read_json(path)
    except (OSError, ValueError, RecursionError) as exc:  # not readable, not UTF-8, not JSON
        raise ValueError(f"{path} is not the snapshot.json of a frozen snapshot ({type(exc).__name__}: {exc})") from None
    if not isinstance(meta, dict):
        raise ValueError(f"{path} is not the snapshot.json of a frozen snapshot: it holds no JSON object")
    wrong = [key for key, types in SIDECAR_TYPES.items() if key in meta and (
        not isinstance(meta[key], types) or isinstance(meta[key], bool))]
    if wrong:
        raise ValueError(f"{path} is not the snapshot.json of a frozen snapshot: {', '.join(wrong)} "
                         f"{'is' if len(wrong) == 1 else 'are'} not as evalgen writes {'it' if len(wrong) == 1 else 'them'}")
    return meta


def check_sidecar(meta: Mapping[str, Any], raw: bytes, path: Path, *, source: str, domain: str, window: int,
                  legacy: bool = False) -> None:
    """Refuse (ValueError, naming the file) the snapshot.json `meta`, read from `path`, if it does not describe
    the frozen bytes `raw` beside it as the run that froze them wrote it.

    Its hash must be theirs: it records their SHA-256 (raw_sha256, which a live snapshot always records),
    and they match it. What no hash covers is checked against the bytes and the replay: they came from
    `source`, the live source of `domain` (a legacy run's snapshot names no domain: it is of earthquakes),
    and they cover the window they give (`window` hours; a legacy run records its hour or its day feed's,
    LEGACY_WINDOWS). Their origin (url) and the time they were fetched (fetched_at) have the form evalgen
    writes; the source run's manifest, when a replay finds it, vouches for their values
    (pipeline._replay_check). A replay refuses what fails, rather than replay bytes that were changed."""
    def refuse(why: str) -> ValueError:
        return ValueError(f"{path} does not describe the frozen bytes beside it: {why}, so they are not replayed")

    if meta.get("source") != source:
        said = meta.get("source")
        raise refuse("it says its snapshot is synthetic" if said == "synthetic"
                     else f"it says they came from {said!r}, not from {domain}'s live source ({source})")
    if meta.get("domain", DEFAULT_DOMAIN) != domain:
        raise refuse(f"it says they are of {meta.get('domain', DEFAULT_DOMAIN)}, not of {domain}")
    recorded = meta.get("raw_sha256")
    if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
        raise refuse("it records no raw_sha256 of them, which the snapshot.json of a live snapshot always does")
    if recorded != sha256_hex(raw):
        raise refuse("they do not match the raw_sha256 that it records (one or the other changed since they were "
                     "frozen)")
    if "window_hours" in meta and (meta["window_hours"] not in LEGACY_WINDOWS if legacy
                                   else meta["window_hours"] != window):
        raise refuse(f"it gives a window of {meta['window_hours']} hours, which "
                     + ("is neither a legacy run's hour nor its day" if legacy else f"they do not: they cover {window}"))
    url = meta.get("url")
    if url is not None and (len(url) > 2048 or not _URL.fullmatch(url)):
        raise refuse(f"its url {url[:80]!r} is not a URL")
    fetched_at = meta.get("fetched_at")
    if fetched_at is not None and not _utc_time(fetched_at):
        raise refuse(f"its fetched_at {fetched_at[:40]!r} is not a time in UTC as evalgen writes one")


def _utc_time(text: str) -> bool:
    """Whether `text` is a time in UTC as a snapshot records one ("2026-11-08T02:40:00Z")."""
    if not _FETCHED_AT.fullmatch(text):
        return False
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _replayed(path: Path, raw: bytes, window_hours: int, *, source: str, domain: str,
              legacy: bool = False) -> tuple[int, str, str, str | None]:
    """(window_hours, url, note, fetched_at) for frozen feed bytes read from `path`. Replaying a run's own
    frozen snapshot keeps its window, its origin and when it was fetched, from the snapshot.json beside it,
    which must describe the bytes (read_sidecar, check_sidecar): only then is the replay "SHA-256 verified".
    A bare frozen file says nothing of when it was fetched (None), and its replay claims no verification."""
    url, note, fetched_at = f"file:{path.name}", f"frozen feed file {path.name}", None
    sidecar = path.with_name("snapshot.json")
    if sidecar.exists():
        meta = read_sidecar(sidecar)
        check_sidecar(meta, raw, sidecar, source=source, domain=domain, window=window_hours, legacy=legacy)
        window_hours, url = meta.get("window_hours", window_hours), meta.get("url") or url
        fetched_at = meta.get("fetched_at")
        note = f"replay of frozen feed {path.name} (SHA-256 verified)"
    return window_hours, url, note, fetched_at


def build_snapshot(source: str, seedlog: SeedLog, *, feed: str = "day", pool_size: int = 360,
                   snapshot_file: str | None = None,
                   fetch: Callable[[str], tuple[bytes, str]] = fetch_usgs) -> tuple[Snapshot, bytes | None]:
    """Return (snapshot, raw feed bytes or None). USGS failures fall back to synthetic data. A frozen feed's
    snapshot.json, when it lies beside the file, must describe it (check_sidecar), or ValueError is raised."""
    if snapshot_file:
        path = Path(snapshot_file)
        raw = path.read_bytes()
        # A legacy replay is stamped with the time it ran, as 0.2.0 stamped it.
        window, url, note, _ = _replayed(path, raw, 1 if "hour" in path.name else 24, source="usgs",
                                         domain=DEFAULT_DOMAIN, legacy=True)
        return _usgs_snapshot(raw, seedlog.master, url, window, note=note), raw
    if source == "usgs":
        try:
            raw, url = fetch(feed)
            snapshot = _usgs_snapshot(raw, seedlog.master, url, 1 if feed == "hour" else 24)
            if len(snapshot.events) < MIN_POOL_EVENTS and feed == "hour":
                raw, url = fetch("day")
                snapshot = _usgs_snapshot(raw, seedlog.master, url, 24,
                                          note="hour feed had too few events; widened to the day feed")
            if len(snapshot.events) < MIN_POOL_EVENTS:
                raise ValueError(f"only {len(snapshot.events)} usable events in the feed")
            return snapshot, raw
        except (requests.RequestException, ValueError, KeyError, RecursionError, ArithmeticError) as exc:
            note = f"USGS feed unavailable ({failure_text(exc)}); fell back to synthetic data"  # no object address
            return _synthetic_snapshot(seedlog, pool_size, note=note), None
    if source != "synthetic":
        raise ValueError(f"unknown source {source!r}")
    return _synthetic_snapshot(seedlog, pool_size), None


# ------------------------------------------------------------------ any domain, off the legacy path

LIVE = "live"  # the source that names a domain's own live source, whichever it is


def build_domain_snapshot(domain: Domain, source: str, seedlog: SeedLog, *, pool_size: int = 360,
                          snapshot_file: str | None = None,
                          fetch: Callable[[], tuple[bytes, str]] | None = None) -> tuple[Snapshot, bytes | None]:
    """The snapshot of a run about `domain` off the legacy path, and its frozen raw bytes (None for
    synthetic records).

    "synthetic" draws the domain's synthetic records over its synthetic window, from seed paths that
    name the domain ("snapshot", <domain>, "clock" and "synthetic"), so no legacy path ever moves.
    "live", or the name of the domain's live source ("usgs"), fetches that source once (`fetch` stands
    in for the adapter's own) and falls back to synthetic records, with a note, when it cannot be
    reached, cannot be read, gives fewer records than its minimum or a snapshot that cannot be right
    (window_problems); a domain without a live source gets synthetic records. A snapshot file holds
    frozen bytes of the live source, which are replayed (read_frozen_feed), and must be what the
    snapshot.json beside it describes (check_sidecar). The records must suit the domain
    (record_problems), or ValueError is raised: the parsers leave out what no real feed holds, so a
    record that does not suit the domain is an adapter's error."""
    live = domain.live
    if snapshot_file:
        raw, records, compiled_at, window = read_frozen_feed(domain, snapshot_file)
        window, url, note, fetched_at = _replayed(Path(snapshot_file), raw, window, source=live.source,
                                                  domain=domain.name)
        return _live_domain_snapshot(domain, seedlog, raw, records, compiled_at, window, url, note,
                                     fetched_at=fetched_at), raw
    if source == "synthetic":
        return _synthetic_domain_snapshot(domain, seedlog, pool_size), None
    if source != LIVE and (live is None or source != live.source):
        raise ValueError(f"unknown source {source!r} for {domain.name}: use synthetic, {LIVE}"
                         + ("" if live is None else f" or {live.source}"))
    if live is None:
        note = f"{domain.name} has no live source; synthetic data"
        return _synthetic_domain_snapshot(domain, seedlog, pool_size, note=note), None
    try:
        raw, url = (fetch or live.fetch)()
        records, compiled_at, window = live.parse(raw)
        problems = window_problems(records, compiled_at, window)
        if problems:
            raise ValueError("; ".join(problems[:3]))
        if len(records) < live.min_records:
            raise ValueError(f"only {len(records)} usable records in the feed")
    except (requests.RequestException, *PARSE_ERRORS) as exc:  # out of reach, or bytes its parser cannot read
        note = f"{live.source} unavailable ({failure_text(exc)}); fell back to synthetic data"
        return _synthetic_domain_snapshot(domain, seedlog, pool_size, note=note), None
    return _live_domain_snapshot(domain, seedlog, raw, records, compiled_at, window, url, None, fetched_at=_now()), raw


# Bytes a live parser cannot read: of another shape, of values out of any range, or JSON nested too deeply.
PARSE_ERRORS = (ValueError, KeyError, TypeError, AttributeError, IndexError, RecursionError, ArithmeticError)
# What a failure's message may hold that differs between two runs that fail alike, and so stays out of the
# note a snapshot records: the query of a URL (a request for the last day of NWS observations names the time
# it was sent) and the address of an object.
VOLATILE = re.compile(r"(?<=[\w/.-])\?[\w%.~=&+:-]*|\s+at 0x[0-9a-fA-F]+")


def failure_text(exc: BaseException) -> str:
    """Why a live source failed, as a snapshot's note records it: the exception's type and message without
    what changes from one moment to the next (VOLATILE), so that two offline runs of one seed, which both
    fall back to the same synthetic records, write the same files, and without the proxy that the
    environment names (proxies_for), whose host and port, and any user name in its URL, are written
    "<proxy>": a note reaches the manifest and the report, which are published."""
    return f"{type(exc).__name__}: {_without_proxies(VOLATILE.sub('', str(exc)))}"


def _without_proxies(text: str) -> str:
    """`text` with every proxy that the environment names written "<proxy>"."""
    for proxy in sorted(set(urllib.request.getproxies_environment().values())):
        parts = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        try:
            host = parts.hostname
        except ValueError:
            continue
        if not host:
            continue
        text = text.replace(proxy, "<proxy>")
        name = re.escape(host)
        text = re.sub(rf"host='{name}', port=\d+", "host='<proxy>', port=<proxy>", text, flags=re.IGNORECASE)
        text = re.sub(rf"(?:[^\s'\"/@]+@)?(?<![\w.-]){name}(?::\d+)?(?![\w-])", "<proxy>", text, flags=re.IGNORECASE)
    return text


def window_problems(records: Sequence[Any], compiled_at: int, window_hours: int) -> list[str]:
    """What cannot be right about a live snapshot off the legacy path: a compiled_at outside
    EARLIEST_TIME..LATEST_TIME, a window that is not a whole number of hours from MIN_WINDOW_HOURS (too
    short for the traps: the USGS hour feed is read in legacy runs only) to MAX_WINDOW_HOURS, or a record
    timed outside the window, which its digests say they cover."""
    if isinstance(compiled_at, bool) or not isinstance(compiled_at, int) or not (
            EARLIEST_TIME <= compiled_at <= LATEST_TIME):
        return [f"it was compiled at {compiled_at!r}, which no feed was"]
    if isinstance(window_hours, bool) or not isinstance(window_hours, int) or not (
            MIN_WINDOW_HOURS <= window_hours <= MAX_WINDOW_HOURS):
        return [f"it covers {window_hours!r} hour(s): a snapshot off the legacy path covers {MIN_WINDOW_HOURS} to "
                f"{MAX_WINDOW_HOURS} hours (the USGS hour feed is read in legacy runs only)"]
    start = compiled_at - window_hours * 3600
    return [f"{r.id} is timed {r.time}, outside the {window_hours} hours before {compiled_at} that it covers"
            for r in records if not start <= r.time <= compiled_at]


def read_frozen_feed(domain: Domain, snapshot_file: str | Path, *,
                     legacy: bool = False) -> tuple[bytes, list[Any], int, int]:
    """The frozen bytes of `domain`'s live source in `snapshot_file`, read and checked as a replay
    reads them: (the bytes, their records, the snapshot's compiled_at, its window in hours). A run
    calls it before it writes anything, so that a file it cannot replay is refused up front. A legacy
    run's frozen USGS feed (`legacy`) is read as build_snapshot reads it: parse_usgs_geojson, over the
    window its file name gives (1 hour for an hour feed's name, else 24).

    FileNotFoundError when the file is not there. ValueError when the domain has no live source, or the
    bytes are not its live source's: its parser cannot read them, they give no record (another
    source's feed can parse to none), records that do not suit the domain (record_problems), or (off the
    legacy path) a snapshot that cannot be right (window_problems); and when the snapshot.json beside them
    does not describe them (check_sidecar)."""
    live = domain.live
    if live is None:
        raise ValueError(f"{domain.name} has no live source whose frozen bytes could be replayed")
    path = Path(snapshot_file)
    if not path.is_file():
        raise FileNotFoundError(f"the frozen feed {path} is not a file")
    sidecar = path.with_name("snapshot.json")
    meta = read_sidecar(sidecar) if sidecar.exists() else None  # what the replay will read beside the bytes
    raw = path.read_bytes()
    source = f"the live source of {domain.name} ({live.source})"
    try:
        if legacy:
            records, compiled_at, _ = parse_usgs_geojson(raw)
            window = 1 if "hour" in path.name else 24
        else:
            records, compiled_at, window = live.parse(raw)
    except PARSE_ERRORS as exc:
        raise ValueError(f"{path} is not a frozen feed of {source}: {type(exc).__name__}: {exc}") from None
    if not records:
        raise ValueError(f"{path} holds no usable record of {source}: is it that source's frozen feed?")
    problems = record_problems(domain, records)
    if problems:
        raise ValueError(f"{path} is not a frozen feed of {source}: its records do not suit the domain: "
                         + "; ".join(problems[:3]))
    problems = [] if legacy else window_problems(records, compiled_at, window)
    if problems:
        raise ValueError(f"{path} cannot be replayed off the legacy path: " + "; ".join(problems[:3]))
    if meta is not None:
        check_sidecar(meta, raw, sidecar, source=live.source, domain=domain.name, window=window, legacy=legacy)
    return raw, list(records), compiled_at, window


def frozen_events_sha256(domain: Domain, raw: bytes, *, legacy: bool = False) -> str:
    """The events_sha256 of the snapshot that frozen bytes of `domain`'s live source give: read as a legacy run
    reads a USGS feed (parse_usgs_geojson), or as the domain's live source does."""
    records = parse_usgs_geojson(raw)[0] if legacy else domain.live.parse(raw)[0]
    return events_sha256(records)


def frozen_folder(directory: str | Path, domain: Domain) -> tuple[Path, dict[str, Any]]:
    """The folder of `domain`'s frozen snapshot among the frozen snapshots of an earlier run, and its
    snapshot.json (read_sidecar): snapshot/<domain>/ in a run of several domains, snapshot/ itself in a
    run of one, whose snapshot.json names its domain (a legacy run's names none: its snapshot is of
    earthquakes). `directory` is the run's snapshot folder or the run's own. ValueError when it holds no
    snapshot of the domain."""
    directory = Path(directory)
    for folder in (directory / domain.name, directory, directory / "snapshot" / domain.name, directory / "snapshot"):
        meta = read_sidecar(folder / "snapshot.json") if (folder / "snapshot.json").is_file() else None
        if meta is not None and meta.get("domain", DEFAULT_DOMAIN) == domain.name:
            return folder, meta
    raise ValueError(f"{directory} holds no frozen snapshot of {domain.name}")


def frozen_feed(directory: str | Path, domain: Domain) -> Path | None:
    """What replays `domain` from the frozen snapshots of an earlier run (frozen_folder): the frozen bytes
    of its live source, or None when that run drew synthetic records for it (synthetic data was asked for,
    or the source was out of reach and the run fell back), which the same seed draws again.

    Raises ValueError when the folder holds no snapshot of the domain, or no frozen bytes of the live source
    its snapshot came from; when its snapshot is synthetic while frozen bytes of the domain's live source lie
    beside it (one of the two was changed); and when it is a legacy run's synthetic snapshot, which a run off
    the legacy path cannot draw again: it draws its synthetic records with a generator and on seed paths of
    its own, so that "replay" would be another world under the same seed."""
    folder, meta = frozen_folder(directory, domain)
    if "source" not in meta:
        raise ValueError(f"{folder / 'snapshot.json'} is not the snapshot.json of a frozen snapshot: it names no source")
    source = meta["source"]
    if source == "synthetic":
        if "domain" not in meta:
            raise ValueError(f"{folder / 'snapshot.json'} is a legacy run's synthetic snapshot, which a run off the "
                             "legacy path cannot draw again (it draws synthetic records on seed paths of its own): "
                             "regenerate that legacy run from its seed instead")
        if domain.live is not None and (folder / domain.live.raw_filename).exists():
            raise ValueError(f"{folder / 'snapshot.json'} says its snapshot of {domain.name} is synthetic, yet frozen "
                             f"bytes of its live source ({domain.live.raw_filename}) lie beside it: one of the two was "
                             "changed, so neither is replayed")
        return None
    live = domain.live
    if live is None or live.source != source:
        raise ValueError(f"the frozen snapshot of {domain.name} in {folder} came from {source!r}, which is not its "
                         "live source")
    raw = folder / live.raw_filename
    if not raw.is_file():
        raise ValueError(f"{folder} holds no frozen {live.raw_filename} of {domain.name}'s live source")
    return raw


def _synthetic_domain_snapshot(domain: Domain, seedlog: SeedLog, pool_size: int, note: str | None = None) -> Snapshot:
    compiled_at = synthetic_compiled_at(seedlog.rng("snapshot", domain.name, "clock"))
    window = domain.synthetic_window_hours
    records = _checked(domain, domain.synthetic(seedlog.rng("snapshot", domain.name, "synthetic"), pool_size,
                                                compiled_at, window))
    return Snapshot(source="synthetic", seed=seedlog.master, compiled_at=compiled_at, window_hours=window,
                    events=records, events_sha256=events_sha256(records), note=note, domain=domain.name)


def _now() -> str:
    """The current time, as a snapshot records when its live source was fetched: by the wall clock, so it is
    one of a run's volatile values (Snapshot.fetched_at), which a replay off the legacy path keeps as frozen."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _live_domain_snapshot(domain: Domain, seedlog: SeedLog, raw: bytes, records: Sequence[Any], compiled_at: int,
                          window_hours: int, url: str, note: str | None, *, fetched_at: str | None) -> Snapshot:
    """A snapshot of a domain's live source. `fetched_at` is when its bytes were fetched: now for a fetch,
    and for a replay what the frozen snapshot recorded (None for a bare frozen file), never the replay's
    own time, so that a replay writes the files of the run it replays."""
    records = _checked(domain, records)
    return Snapshot(source=domain.live.source, seed=seedlog.master, compiled_at=compiled_at,
                    window_hours=window_hours, events=records, events_sha256=events_sha256(records),
                    raw_sha256=sha256_hex(raw), url=url, fetched_at=fetched_at, note=note, domain=domain.name)


def _checked(domain: Domain, records: Sequence[Any]) -> list[Any]:
    """The records in time order (then ID order), once record_problems finds nothing wrong."""
    problems = record_problems(domain, records)
    if problems:
        more = f"; and {len(problems) - 5} more" if len(problems) > 5 else ""
        raise ValueError(f"{domain.name} records do not suit the domain: " + "; ".join(problems[:5]) + more)
    return sorted(records, key=lambda r: (r.time, r.id))


def _finite_number(value: Any) -> bool:
    """Whether `value` is a number (not a bool) that is finite: an int always is."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (
        isinstance(value, int) or math.isfinite(value))


def record_problems(domain: Domain, records: Sequence[Any]) -> list[str]:
    """What is wrong with snapshot records of `domain`. Each record needs an ID of its own that the
    scorer can pick out of an answer (the domain's id_pattern), a time in whole seconds between
    EARLIEST_TIME and LATEST_TIME, a group, a value for exactly the domain's fields, each a finite number
    stored to its field's places (so that every printed form of it reads back exactly), a location exactly
    when the domain's records have one, on the globe and stored to the places of its coordinates, and
    qualifiers only under the keys of its fields' qualifiers (a digest prints each qualifier with its
    field's value)."""
    problems: list[str] = []
    keys = sorted(spec.key for spec in domain.fields)
    qualifier_keys = set(domain.qualifier_keys.values())
    coord_decimals = None if domain.spatial is None else domain.spatial.coord_decimals
    seen: set[str] = set()
    for r in records:
        if r.id in seen:
            problems.append(f"two records are {r.id}")
        seen.add(r.id)
        if not re.fullmatch(domain.id_pattern, r.id.casefold()):
            problems.append(f"{r.id} does not match the ID pattern {domain.id_pattern}")
        if not isinstance(r.time, int) or isinstance(r.time, bool):
            problems.append(f"{r.id} is timed {r.time!r}, not in whole seconds")
        elif not EARLIEST_TIME <= r.time <= LATEST_TIME:
            problems.append(f"{r.id} is timed {r.time}, not between the years 1800 and 2200")
        if not r.group:
            problems.append(f"{r.id} has no {domain.group_label}")
        if sorted(r.values) != keys:
            problems.append(f"{r.id} has values for {', '.join(sorted(r.values))}, not for {', '.join(keys)}")
        for spec in domain.fields:
            value = r.values.get(spec.key)
            if value is not None and (not _finite_number(value) or round(value, spec.decimals) != value):
                problems.append(f"{r.id} has {spec.key} {value!r}, not a number to {spec.decimals} decimal places")
        if (r.lat is None) != (r.lon is None) or (r.lat is not None) != domain.has_location:
            problems.append(f"{r.id} is at ({r.lat}, {r.lon}), but {domain.name} records "
                            + ("have a location" if domain.has_location else "have none"))
        elif r.lat is not None and not (_finite_number(r.lat) and _finite_number(r.lon)
                                        and -90 <= r.lat <= 90 and -180 <= r.lon <= 180):
            problems.append(f"{r.id} is at ({r.lat}, {r.lon}), which is no place on the globe")
        elif coord_decimals is not None and (round(r.lat, coord_decimals) != r.lat
                                             or round(r.lon, coord_decimals) != r.lon):
            problems.append(f"{r.id} is at ({r.lat}, {r.lon}), not to {coord_decimals} decimal places")
        if isinstance(r, Record) and set(r.qualifiers) - qualifier_keys:
            problems.append(f"{r.id} has qualifiers {', '.join(sorted(set(r.qualifiers) - qualifier_keys))}, "
                            "which no field has")
    return problems


def save_snapshot(snapshot: Snapshot, raw: bytes | None, directory: Path, *,
                  raw_filename: str = "raw_feed.geojson", root: Path | None = None) -> None:
    """events.json (the records), snapshot.json (the metadata) and, for a live source, its frozen raw
    bytes under `raw_filename` (the live adapter's), written into `directory` in the run directory `root`
    (by default `directory` itself) through its one writer (util.write_in)."""
    directory = Path(directory)
    root = directory if root is None else Path(root)
    write_json(directory / "events.json", [asdict(e) for e in snapshot.events], root=root)
    if raw is not None:
        write_in(root, directory / raw_filename, raw)
    write_json(directory / "snapshot.json", snapshot.meta(), root=root)


def load_snapshot(directory: Path) -> Snapshot:
    """A saved snapshot, its records as they were: Events, or the Records of a generic domain."""
    directory = Path(directory)
    meta = read_json(directory / "snapshot.json")
    meta.pop("n_events", None)
    events = [Record(**e) if "values" in e else Event(**e) for e in read_json(directory / "events.json")]
    return Snapshot(events=events, **meta)
