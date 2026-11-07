"""Base data: a seeded synthetic world (default, fully offline) or a frozen USGS feed.

Either way the result is a ``Snapshot``: a list of canonical events plus its SHA-256 hash and the
seed. Everything downstream (digests, questions, answer key) is computed from the snapshot, so
re-using a saved snapshot with the same seed reproduces a run exactly, even for live USGS data.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from . import __version__
from .geo import destination, haversine_km
from .models import EVENT_ID_RE, Event
from .seeds import SeedLog
from .util import canonical_json, read_json, sha256_hex, write_json

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


@dataclass
class Snapshot:
    source: str  # "synthetic" or "usgs"
    seed: int
    compiled_at: int  # end of the covered window (Unix seconds); digests are "compiled" here
    window_hours: int
    events: list[Event]
    events_sha256: str
    raw_sha256: str | None = None  # hash of the frozen feed bytes (usgs only)
    url: str | None = None
    fetched_at: str | None = None
    note: str | None = None  # e.g. why we fell back to synthetic data

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("events")
        d["n_events"] = len(self.events)
        return d


def events_sha256(events: list[Event]) -> str:
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


def parse_usgs_geojson(raw: bytes) -> tuple[list[Event], int, list[str]]:
    """Parse a USGS GeoJSON summary feed: Events (earthquakes only, one per quake), the feed's
    generation time, and the IDs of duplicate solutions that were collapsed."""
    data = json.loads(raw.decode("utf-8"))
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
    events, dropped = collapse_duplicate_solutions(events, rank=lambda e: preference[e.id])
    return events, generated_ms // 1000, dropped


def fetch_usgs(feed: str, timeout: float = 20.0) -> tuple[bytes, str]:
    url = USGS_FEEDS[feed]
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
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


def build_snapshot(source: str, seedlog: SeedLog, *, feed: str = "day", pool_size: int = 360,
                   snapshot_file: str | None = None,
                   fetch: Callable[[str], tuple[bytes, str]] = fetch_usgs) -> tuple[Snapshot, bytes | None]:
    """Return (snapshot, raw feed bytes or None). USGS failures fall back to synthetic data."""
    if snapshot_file:
        path = Path(snapshot_file)
        raw = path.read_bytes()
        window, url, note = (1 if "hour" in path.name else 24), f"file:{path.name}", f"frozen feed file {path.name}"
        sidecar = path.with_name("snapshot.json")
        if sidecar.exists():  # replaying a run's own frozen snapshot: keep its window and origin
            meta = read_json(sidecar)
            window, url = meta.get("window_hours", window), meta.get("url") or url
            if meta.get("raw_sha256") and meta["raw_sha256"] != sha256_hex(raw):
                note += " -- WARNING: bytes do not match the raw_sha256 recorded in snapshot.json"
            else:
                note = f"replay of frozen feed {path.name} (SHA-256 verified)"
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
        except (requests.RequestException, ValueError, KeyError) as exc:
            note = f"USGS feed unavailable ({type(exc).__name__}: {exc}); fell back to synthetic data"
            return _synthetic_snapshot(seedlog, pool_size, note=note), None
    if source != "synthetic":
        raise ValueError(f"unknown source {source!r}")
    return _synthetic_snapshot(seedlog, pool_size), None


def save_snapshot(snapshot: Snapshot, raw: bytes | None, directory: Path) -> None:
    directory = Path(directory)
    write_json(directory / "events.json", [asdict(e) for e in snapshot.events])
    if raw is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "raw_feed.geojson").write_bytes(raw)
    write_json(directory / "snapshot.json", snapshot.meta())


def load_snapshot(directory: Path) -> Snapshot:
    directory = Path(directory)
    meta = read_json(directory / "snapshot.json")
    meta.pop("n_events", None)
    events = [Event(**e) for e in read_json(directory / "events.json")]
    return Snapshot(events=events, **meta)
