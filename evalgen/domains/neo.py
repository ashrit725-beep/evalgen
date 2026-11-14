"""Near-Earth objects: close approaches of asteroids and comets to the planets and the Moon.

One record is one close approach of one object to one body, as JPL's Close Approach Data (CAD) lists
it: when the object passed closest, how far from the body it passed (the miss distance), how fast it
moved relative to the body (the relative velocity) and the object's absolute magnitude H. The body
approached (Earth, the Moon, Venus, Mars, Mercury) is the record's group, and the catalogue that
lists the approach is its source. Records have no position, so the spatial templates and their twins
drop out as a pair. Relative velocity plays the *primary* role of the role-based templates
(thresholds, the fastest approach and its time) and miss distance the *secondary* one (thresholds,
the largest and the smallest miss distance); H takes part in thresholds only.

    miss distance       stored in km, to the whole km; written and asked in km, astronomical units
                        (1 au = 149,597,870.7 km, the IAU definition) or lunar distances (1 LD =
                        384,400 km, as this domain defines it)
    relative velocity   stored in km/s, to 2 decimals; written and asked in km/s, km/h (1/3600 km/s)
                        or mph (exactly 0.00044704 km/s)
    absolute magnitude  H, a plain number stored to 3 decimals (a smaller H is a larger object)

Close approaches span weeks, not hours. Synthetic snapshots cover 30 days, and a live one the whole
UTC days its approaches fall in (about 60 for the 60 days the live query asks for). Time windows scale
with that length (questions.Context): their ends lie on marks of five minutes per day of the window,
rounded up to the next mark of questions.TIME_STEPS_S (3 hours for 25 to 36 days, so for 30; 6 hours for
37 to 72, so for 60), and every approach lies at least two minutes per day of the window from their ends
(an hour for 30 days, two hours for 60, 56 minutes for the 28 days of the frozen CAD fixture the tests
replay). A question about the time of an approach asks for its UTC time of day.

An object's designation repeats across its approaches (2026 VK2 passes Earth, then the Moon ten hours
later), so an approach ID is the designation without spaces, "-ca" and the number of the object's
approach in the snapshot, in time order, all in lower case (other domains may print their IDs in
capitals; every ID is matched without regard to letter case): 2026vk2-ca1, 2026vk2-ca2. The ID names
nothing else about the approach (not its body, not its date): a re-report may give an approach under
another designation, with a note naming the approach it duplicates, and an ID must never contradict
the line it is printed on. Off the legacy path a planted report takes the designation of an approach
drawn with the digest but not kept (traps.py), or, when none is left, a fresh one shaped like those in
the digest (``mimic_id``): a re-report under it is a second designation of the same object, linked by
its note, and a phantom is an approach that was never made. A re-report or a second report under an
approach's own ID is filed only for an approach whose object makes no other approach in the digest or its
slots (``id_place``), as every approach of an object prints one H, which a report beside a sibling could
neither repeat (two reports of a body with one H are a pair as plainly as a shared ID) nor change (a
sibling's H tells it): it gives the object's H as another catalogue fits it (``rereport_values``).

The live source is the CAD API (``CadFeed``): fetched once, frozen as ``raw_cad.json``, SHA-256 hashed
and replayable with a snapshot file, with synthetic approaches as the fallback when it cannot be
reached or read (sources.build_domain_snapshot). CAD times are TDB; records keep UTC.
"""
from __future__ import annotations

import math
import random
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .. import sources
from ..sources import EARLIEST_TIME, LATEST_TIME, USER_AGENT, read_feed_json
from ..units import Unit
from .base import RULES_WIDTH, Domain, FieldSpec, Lexicon, Record, TrapVocab, folded, record_rules, stored_places

AU_KM = 149_597_870.7  # the astronomical unit, in km (IAU 2012, exact)
LD_KM = 384_400.0  # the lunar distance, in km, as this domain defines it
MPH_KM_PER_S = 0.00044704  # 1 mph = 1.609344 km per 3600 s, exactly
DIST_MAX_AU = 0.05  # the live query's limit, which the synthetic approaches keep too
DIST_MIN_AU = 0.0005  # the closest synthetic approach (about 75,000 km)
DAY_S = 86_400
SYNTHETIC_WINDOW_DAYS = 30
# CAD times are TDB (Barycentric Dynamical Time). TDB - UTC = TT - UTC = 32.184 s + 37 leap seconds,
# unchanged since the start of 2017; TDB stays within 2 ms of TT.
TDB_MINUS_UTC_S = 69.184
JD_UNIX_EPOCH = 2_440_587.5  # the Julian date of 1970-01-01 00:00 UTC

KILOMETRE = Unit("km", "kilometres", one="kilometre")
ASTRONOMICAL_UNIT = Unit("au", "astronomical units", AU_KM, decimals=6, step=0.001, one="astronomical unit")
LUNAR_DISTANCE = Unit("LD", "lunar distances", LD_KM, decimals=3, step=0.5, one="lunar distance")
KM_PER_S = Unit("km/s", "kilometres per second", one="kilometre per second")
KM_PER_H = Unit("km/h", "kilometres per hour", 1 / 3600, decimals=0, step=2000.0, one="kilometre per hour")
MILES_PER_HOUR = Unit("mph", "miles per hour", MPH_KM_PER_S, decimals=0, step=1000.0, one="mile per hour")

# Margins and steps follow each field's spread: miss distances run from about 100,000 km to 7.5 million
# km (0.05 au), relative velocities from under 1 to about 40 km/s (median near 10), and H from about 16
# to 32. A threshold restated in another unit moves by at most 192 km (LD to 3 places) or 0.005 km/s
# (km/s to 2 places), well inside the absolute margins; the relative margins keep every approach 2% (miss
# distance) or 1% (velocity) clear of a threshold, so that no rounded conversion can move one across it.
MISS_DISTANCE = FieldSpec(
    key="dist", label="miss distance", canonical_unit="km", units=(KILOMETRE, ASTRONOMICAL_UNIT, LUNAR_DISTANCE),
    decimals=0, answer_type="neo_miss_distance", tolerance=(500.0, 0.005), margin=(2000.0, 0.02),
    rank_gap=0.0,  # no ranked terminal reads it
    quantiles=(0.2, 0.8), threshold_step=50_000.0, positive_threshold=True,
    filters=("below", "at_least"),  # "closer than" first, as depth's "shallower than"
    terminals=("max", "min"), correctable=True,
)
RELATIVE_VELOCITY = FieldSpec(
    key="v_rel", label="relative velocity", canonical_unit="km/s", units=(KM_PER_S, KM_PER_H, MILES_PER_HOUR),
    decimals=2, answer_type="neo_velocity", tolerance=(0.01, 0.002), margin=(0.02, 0.01),
    rank_gap=0.25,  # the fastest approach beats the next by a quarter km/s, whatever units they are given in
    quantiles=(0.2, 0.8), threshold_step=0.5, positive_threshold=True,
    filters=("at_least", "below"), correctable=True,
    terminals=("argmax", "max", "time_of_max"),  # no rank: the one rank template, P09, is spatial
)
# Printed exactly as stored, like an earthquake's magnitude, so it needs no margin: a threshold is an
# approach's own H rounded to half a magnitude, with no jitter, and lands on an H of the digest whenever that H
# is a multiple of 0.5 (most H have one to three decimals, so it seldom does). Such a question turns on "at
# least" against "below", which the wording states exactly.
ABSOLUTE_MAGNITUDE = FieldSpec(
    key="h", label="absolute magnitude", canonical_unit=None, units=(), decimals=3,
    answer_type="neo_absolute_magnitude", tolerance=(0.05, 0.0),
    margin=(0.0, 0.0),
    rank_gap=0.0, quantiles=(0.2, 0.8), threshold_step=0.5, filters=("below", "at_least"), terminals=(),
)

# A record ID: a designation without its spaces (2026vk2, 18109, p2019ld2, 2p), "-ca" and the number of
# the object's approach, as a regex over lower-cased text (IDs are lower case, as the pattern is).
ID_PATTERN = r"[0-9a-z]{2,12}-ca[1-9]"
ID_RE = re.compile(ID_PATTERN)

LEXICON = Lexicon(
    records="close approaches",
    all_records="close approaches in the digest",
    filters={
        "at_least(v_rel)": ("with a relative velocity of at least {value}",
                            "whose relative velocity is {value} or more"),
        "below(v_rel)": ("with a relative velocity below {value}",
                         "that moved slower than {value} relative to the body"),
        "below(dist)": ("with a miss distance below {value}", "that passed closer than {value}"),
        "at_least(dist)": ("with a miss distance of at least {value}", "that passed at least {value} away"),
        "below(h)": ("with an absolute magnitude (H) below {value}", "whose absolute magnitude is less than {value}"),
        "at_least(h)": ("with an absolute magnitude (H) of at least {value}",
                        "whose absolute magnitude is {value} or more"),
        "time_between": ("that occurred between {start} and {end}", "with closest approach from {start} up to {end}"),
        "in_group": ("with {group} as the body approached", "whose body is {group}"),
    },
    questions={
        "count": ("How many distinct {records} are there?", "Count the distinct {records}.",
                  "How many separate {records} does the digest list?"),
        "compare_groups": ("Counting only {records}, which body had more of them: {a} or {b}? Answer \"tie\" if the "
                           "counts are equal.",
                           "Which body had more {records}: {a} or {b}? Say \"tie\" if they are equal."),
        "argmax(v_rel)": ("Among the {records}, which one had the highest relative velocity? Give its approach ID.",
                          "Which of the {records} was the fastest relative to the body it passed? Give its approach "
                          "ID."),
        "max(v_rel)": ("What was the highest relative velocity, in {unit}, among the {records}? {rounding}",
                       "Among the {records}, how fast was the fastest one, in {unit}? {rounding}"),
        "time_of_max(v_rel)": ("At what UTC time of day (HH:MM) did the fastest of the {records} make its closest "
                               "approach?",
                               "Among the {records}, at what time of day (UTC, HH:MM) was the fastest one at its "
                               "closest approach?"),
        "max(dist)": ("What was the largest miss distance, in {unit}, among the {records}? {rounding}",
                      "Among the {records}, how far away did the most distant one pass, in {unit}? {rounding}"),
        "min(dist)": ("What was the smallest miss distance, in {unit}, among the {records}? {rounding}",
                      "Among the {records}, how close did the closest one pass, in {unit}? {rounding}"),
    },
)

# Catalogues of close approaches. JPL's lists every live record; synthetic records come from all four, the
# Minor Planet Center's included, so that every catalogue that re-reports approaches also lists its own.
SOURCES = ("jpl", "neocc", "neodys", "mpc")
SOURCE_WEIGHTS = (0.5, 0.25, 0.13, 0.12)
TRAP_VOCAB = TrapVocab(
    rereporting_sources=("mpc", "neocc", "neodys"),
    duplicate_notes=("same approach as", "duplicate of", "mirror of"),
    retraction_reasons=("spurious detection", "image artefact", "false linkage of observations", "detector noise"),
    retraction_notices=("withdrawn", "RETRACTED", "deleted from the catalogue"),
    correction_notices=("revised to", "recomputed as", "corrected to"),
)

# Lines that carry no data ({station}, {h1}, {date}, {until} and the domain's words are filled in by
# digest.timed_chatter); none reads like a report or a notice, none holds a number, and none uses a word that the
# rules give a meaning ("mirror", which a re-report's note "mirror of" uses, once named the catalogue's server).
CHATTER = (
    "The {source_label} server {station} was offline from {h1} UTC on {date} until {until}.",
    "Miss distances are nominal values; the uncertainty of each approach is not reproduced here.",
    "No impact risk was flagged for any object in this digest.",
    "Editor's note: entries are reproduced exactly as each {source_label} filed them, units and time conventions "
    "included.",
    "Objects discovered after compilation will be listed in the next edition.",
)

# What a report's time is, in the rule on times that every record domain shares (record_rules, rule 6), in
# place of "when its close approach was recorded", which could be read as when a catalogue logged it.
REPORT_TIME = "the time of its close approach, the moment the object passed closest to the body"
# What these records need beyond the rules every record domain shares (record_rules).
RULES = (
    "An approach ID is an object's designation in lower case without its spaces, then \"-ca\" and a number that "
    "tells that object's close approaches apart: 2026vk2-ca1 and 2026vk2-ca2 are two close approaches of the object "
    "2026 VK2 (one to Earth and one to the Moon, say), so they are two records, each counted on its own. A re-report "
    "may give a close approach under another designation of the same object, with a note naming the approach ID it "
    "duplicates: it is then the same close approach.",
    "A miss distance is how far from the body the object passed, a relative velocity is the object's speed relative "
    "to that body, and an absolute magnitude (H) is a plain number.",
)


# ----------------------------------------------------------------------------- designations

HALF_MONTHS = "ABCDEFGHJKLMNOPQRSTUVWXY"  # a provisional designation's first letter: when the object was found
ORDER_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # its second letter: its order within that half-month
FIRST_YEAR = 1998  # the oldest provisional designation drawn
# Packed designations, as approach IDs begin (in upper case here): provisional (2026VK2), a comet's
# provisional one (P2019LD2) and a periodic comet's number (2P); any other is an asteroid's number (18109).
_PROVISIONAL = re.compile(r"(?P<year>\d{4})(?P<half>[A-Y])(?P<order>[A-Z])(?P<cycle>\d*)")
_COMET = re.compile(r"(?P<kind>[ACDIPX])(?P<rest>\d{4}[A-Y][A-Z]\d*)")
_PERIODIC = re.compile(r"(?P<number>\d+)(?P<kind>[PDI])")


def pack(designation: str) -> str:
    """A designation without its spaces and punctuation, in upper case: "2026 VK2" is 2026VK2, "P/2019 LD2"
    is P2019LD2 (approach IDs write it in lower case)."""
    return re.sub(r"[^0-9A-Za-z]", "", designation).upper()


def half_month(ts: int) -> int:
    """The index in HALF_MONTHS of the half-month that `ts` falls in."""
    day = datetime.fromtimestamp(ts, tz=timezone.utc)
    return (day.month - 1) * 2 + (day.day > 15)


def _cycle(rng: random.Random) -> str:
    """A provisional designation's cycle count: none, or how often its letters went round in the half-month."""
    roll = rng.random()
    if roll < 0.1:
        return ""
    if roll < 0.55:
        return str(rng.randint(1, 9))
    if roll < 0.95:
        return str(rng.randint(10, 99))
    return str(rng.randint(100, 250))


def _provisional(rng: random.Random, year: int, last_half: int) -> str:
    """A packed provisional designation of `year`, found in one of its first `last_half` + 1 half-months."""
    return f"{year}{rng.choice(HALF_MONTHS[:last_half + 1])}{rng.choice(ORDER_LETTERS)}{_cycle(rng)}"


NUMBERED = (3000, 650_000)  # the asteroid numbers synthetic objects carry (_draw_designation)


def _number(rng: random.Random, digits: int, within: tuple[int, int] = (1, 1 << 62)) -> str:
    """A number of `digits` digits, inside `within` (from, below) where the two ranges meet."""
    low, high = max(10 ** (digits - 1), within[0]), min(10 ** digits, within[1])
    if low >= high:
        low, high = 10 ** (digits - 1), 10 ** digits
    return str(rng.randrange(low, high))


def _like(example: str, rng: random.Random) -> str:
    """A packed designation of the same kind as `example` (upper case), drawn as such designations are:
    a provisional one of its year and half-month, with its order letter and cycle drawn afresh (a comet's
    with its type letter), a periodic comet's number of as many digits, an asteroid's number of as many
    digits inside the range numbered objects come from, or else its digits redrawn."""
    comet = _COMET.fullmatch(example)
    if comet is not None:
        return comet["kind"] + _like(comet["rest"], rng)
    found = _PROVISIONAL.fullmatch(example)
    if found is not None:
        return f"{found['year']}{found['half']}{rng.choice(ORDER_LETTERS)}{_cycle(rng)}"
    periodic = _PERIODIC.fullmatch(example)
    if periodic is not None:
        return _number(rng, len(periodic["number"])) + periodic["kind"]
    if example.isdigit():
        return _number(rng, len(example), NUMBERED)
    return "".join(rng.choice("0123456789") if c.isdigit() else c for c in example)


def approach_id(designation: str, number: int) -> str:
    """The ID of an object's `number`-th approach: ("2026VK2", 2) is 2026vk2-ca2."""
    return f"{designation.lower()}-ca{number}"


def designation_of(record_id: str) -> str:
    """The packed designation (upper case) an approach ID begins with: 2026vk2-ca1 is 2026VK2."""
    return record_id.rsplit("-", 1)[0].upper()


# ------------------------------------------------------------------------------- approaches


@dataclass(frozen=True)
class Approach:
    """One close approach before it is numbered: what a record holds, with its object's packed designation."""

    designation: str
    time: int
    body: str
    source: str
    values: Mapping[str, float]


def numbered(approaches: Sequence[Approach]) -> list[Record]:
    """Records of the approaches, each object's approaches numbered in time order (2026vk2-ca1, -ca2, ...),
    in time order and then ID order. An approach ID has one digit, so an object makes at most nine
    approaches in a snapshot: its tenth and later are left out. (No real object makes ten close approaches
    within 0.05 au in the 60 days that the live query covers, and a synthetic object makes one or two.)"""
    seen: Counter[str] = Counter()
    records = []
    for a in sorted(approaches, key=lambda a: (a.time, a.designation, a.body)):
        seen[a.designation] += 1
        if seen[a.designation] > 9:
            continue
        records.append(Record(id=approach_id(a.designation, seen[a.designation]), time=a.time, lat=None, lon=None,
                              group=a.body, source=a.source, values=dict(a.values)))
    return sorted(records, key=lambda r: (r.time, r.id))


@dataclass(frozen=True)
class BodyModel:
    """How synthetic approaches to one body are drawn. The Moon's come with close approaches to Earth."""

    name: str
    weight: float  # the share of synthetic objects that approach this body
    distance_power: float  # miss distance = the limit × u ** power: 1 spreads them evenly, 0.5 favours far ones
    velocity: tuple[float, float, float, float]  # km/s: median, log-normal spread, least, most
    magnitude: tuple[float, float, float, float]  # H: mean, spread, least, most


# Shaped on the CAD lists: approaches to Earth are as likely at any distance up to the limit (small
# objects are found when they pass close) and are mostly slow and faint; the other planets are passed by
# larger, already known objects, as often at twice the distance (the area of a ring), Mercury's fast.
BODIES = (
    BodyModel("Earth", 0.70, 1.0, (8.5, 0.45, 1.0, 40.0), (26.8, 1.5, 19.0, 32.0)),
    BodyModel("Venus", 0.16, 0.5, (12.0, 0.40, 3.0, 45.0), (23.2, 2.4, 16.5, 29.0)),
    BodyModel("Mars", 0.08, 0.5, (7.5, 0.35, 2.0, 25.0), (21.8, 2.2, 15.5, 27.5)),
    BodyModel("Mercury", 0.06, 0.5, (28.0, 0.25, 12.0, 55.0), (20.5, 2.4, 15.0, 27.0)),
)
MOON = "Moon"
MOON_PAIR_MAX_AU = 0.012  # an object passing Earth closer than this often passes the Moon too ...
MOON_PAIR_CHANCE = 0.8
MOON_ORBIT_AU = LD_KM / AU_KM  # ... at about its distance from Earth, give or take the Moon's distance
MOON_PAIR_HOURS = 14  # ... and within this many hours of its approach to Earth
H_PER_AU = 40.0  # objects that pass Earth closer are fainter: H rises by this much per au closer than 0.025 au


def _clamped(value: float, least: float, most: float) -> float:
    return min(max(value, least), most)


MOON_MIN_AU = 0.0003  # the closest synthetic approach to the Moon (about 45,000 km)
FOUND_BEFORE_DAYS, FOUND_AFTER_DAYS = 21, 5  # when an object is found, around its approach (_draw_object)


def _draw_year(rng: random.Random, h: float, current: int) -> int:
    """The year of a provisional designation: small objects are mostly found in the year they pass close."""
    if h >= 24.0 and rng.random() < 0.65:
        return current
    return current - min(current - FIRST_YEAR, int(rng.expovariate(0.25)))


def _year(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


def _draw_designation(rng: random.Random, h: float, when: int, taken: set[str]) -> str:
    """A packed designation not in `taken` (and added to it): a number for most large objects, else a
    provisional designation found by the half-month of `when` in its year (an older one, in any of its
    half-months). `when` is when the object was found, which a caller keeps before the compilation."""
    numbered_chance = 0.85 if h < 21.0 else 0.35 if h < 23.5 else 0.03
    current = _year(when)
    while True:
        if rng.random() < numbered_chance:
            designation = str(rng.randrange(*NUMBERED))
        else:
            year = _draw_year(rng, h, current)
            designation = _provisional(rng, year, half_month(when) if year == current else len(HALF_MONTHS) - 1)
        if designation not in taken:
            taken.add(designation)
            return designation


def _draw_object(rng: random.Random, body: BodyModel, compiled_at: int, window_hours: int,
                 taken: set[str]) -> list[Approach]:
    """One object's approach to `body`, and to the Moon too when it passes Earth close enough. A recently
    found object's H has 3 decimals, an older one's 1 or 2, as in the CAD lists."""
    time = compiled_at - rng.randrange(60, window_hours * 3600)
    dist_au = DIST_MIN_AU + (DIST_MAX_AU - DIST_MIN_AU) * rng.random() ** body.distance_power
    v_median, v_spread, v_least, v_most = body.velocity
    velocity = _clamped(rng.lognormvariate(math.log(v_median), v_spread), v_least, v_most)
    h_mean, h_spread, h_least, h_most = body.magnitude
    h = rng.gauss(h_mean, h_spread)
    if body.name == "Earth":
        h += (0.025 - dist_au) * H_PER_AU
    h = _clamped(h, h_least, h_most)
    # Small objects are found around their close approach (up to three weeks before it, or days after),
    # never after the compilation: a designation never postdates the digest that lists it.
    found = min(compiled_at - 60, time + rng.randint(-FOUND_BEFORE_DAYS * DAY_S, FOUND_AFTER_DAYS * DAY_S))
    designation = _draw_designation(rng, h, found, taken)
    provisional = _PROVISIONAL.fullmatch(designation)
    recent = provisional is not None and int(provisional["year"]) >= _year(compiled_at) - 2
    h = round(h, 3 if recent else rng.choice((1, 2)))
    source = rng.choices(SOURCES, weights=SOURCE_WEIGHTS, k=1)[0]
    approaches = [Approach(designation, time, body.name, source, _values(dist_au, velocity, h))]
    if body.name == "Earth" and dist_au < MOON_PAIR_MAX_AU and rng.random() < MOON_PAIR_CHANCE:
        moon_au = folded(dist_au + rng.uniform(-MOON_ORBIT_AU, MOON_ORBIT_AU), MOON_MIN_AU, MOON_PAIR_MAX_AU - 0.0001)
        moon_velocity = max(0.15, velocity + rng.uniform(-1.0, 1.0))
        moon_time = time + rng.randint(-MOON_PAIR_HOURS * 3600, MOON_PAIR_HOURS * 3600)
        if compiled_at - window_hours * 3600 < moon_time <= compiled_at - 60:
            approaches.append(Approach(designation, moon_time, MOON, source, _values(moon_au, moon_velocity, h)))
    return approaches


def _values(dist_au: float, velocity: float, h: float) -> dict[str, float]:
    """A record's values, stored as its fields store them."""
    return {MISS_DISTANCE.key: float(round(dist_au * AU_KM)), RELATIVE_VELOCITY.key: round(velocity, 2),
            ABSOLUTE_MAGNITUDE.key: h}


# ----------------------------------------------------------------------------- the live source

CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api?date-min=now&date-max=%2B60&dist-max=0.05&body=ALL&fullname=true"
CAD_COLUMNS = ("des", "jd", "dist", "v_rel", "body", "h")  # the columns a record is read from
# Bodies under the short names CAD's queries use, should a response give them: by their full names.
BODY_NAMES = {"Merc": "Mercury", "Juptr": "Jupiter", "Satrn": "Saturn", "Urnus": "Uranus", "Nptun": "Neptune"}


def parse_cad(raw: bytes) -> tuple[list[Record], int, int]:
    """A frozen CAD response as (records, compiled_at, window_hours), without the clock or the network.

    Every row with a designation, a time, a miss distance, a relative velocity, a body and an H becomes
    a record of the catalogue "jpl" (_cad_approach), but an object's tenth approach and later (numbered).
    A row without one of these values, such as a comet's, which has no H, is left out (no value is ever
    invented), and so is a row with a value that cannot be right: a designation or a body that is not
    text, a time outside the years 1800 to 2200, a miss distance outside (0, 1) au, a velocity outside
    (0, 1000) km/s, or an H outside H_RANGE. Times are converted from TDB to UTC and kept to the second,
    miss distances from au to the whole km, and velocities to 2 decimals. The window is the whole UTC days
    the approaches fall in: it starts at midnight before the first and is compiled at midnight after the
    last. A response that is not a list of close approaches, or that has none usable, raises ValueError."""
    doc = read_feed_json(raw, "the close-approach response")
    if not isinstance(doc, dict) or not isinstance(doc.get("fields"), list):
        raise ValueError("not a close-approach response: it has no list of fields")
    fields = doc["fields"]
    missing = [name for name in CAD_COLUMNS if name not in fields]
    if missing:
        raise ValueError(f"the close-approach response has no {', '.join(missing)} column")
    rows = doc.get("data") or []  # a response without approaches has no data at all
    if not isinstance(rows, list):
        raise ValueError("the close-approach response's data is not a list of rows")
    approaches = [a for a in (_cad_approach(dict(zip(fields, row))) for row in rows if isinstance(row, list))
                  if a is not None]
    records = numbered(approaches)
    if not records:
        raise ValueError("the close-approach response lists no usable approach")
    first, last = records[0].time, records[-1].time
    start, end = first - first % DAY_S, last - last % DAY_S + DAY_S
    return records, end, (end - start) // 3600


H_RANGE = (-5.0, 40.0)  # absolute magnitudes of real bodies: Ceres's is 3.3, the faintest known NEOs' about 34


def _cad_approach(row: Mapping[str, Any]) -> Approach | None:
    """One CAD row as an approach, or None when it lacks a value or holds one that cannot be right (see
    parse_cad). The designation and the body must be text: a null names nothing (not an object "NONE")."""
    if not isinstance(row.get("des"), str) or not isinstance(row.get("body"), str):
        return None
    try:
        designation = pack(row["des"])
        jd, dist, velocity, h = (float(row[name]) for name in ("jd", "dist", "v_rel", "h"))
        body = row["body"].strip()
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (2 <= len(designation) <= 12 and body and math.isfinite(jd) and H_RANGE[0] <= h <= H_RANGE[1]
            and 0 < dist < 1 and 0 < velocity < 1000):
        return None
    time = round((jd - JD_UNIX_EPOCH) * DAY_S - TDB_MINUS_UTC_S)
    if not EARLIEST_TIME <= time <= LATEST_TIME:
        return None
    return Approach(designation, time, BODY_NAMES.get(body, body), "jpl", _values(dist, velocity, round(h, 3)))


class CadFeed:
    """JPL's Close Approach Data API as the neo domain's live source (a LiveAdapter): every approach
    within 0.05 au of any body over the next 60 days, fetched once (no key needed)."""

    source = "jpl"
    raw_filename = "raw_cad.json"
    user_agent = USER_AGENT
    min_records = 30  # enough for digests of 20 to 28 approaches without repeating one in every digest
    url = CAD_URL

    def fetch(self, timeout: float = 30.0) -> tuple[bytes, str]:
        """The CAD response, requested as every live source is (sources.http_get: nothing from the
        environment but its proxies, and a deadline for the whole answer)."""
        response = sources.http_get(self.url, timeout=timeout,
                                    headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        response.raise_for_status()
        return response.content, self.url

    def parse(self, raw: bytes) -> tuple[list[Record], int, int]:
        return parse_cad(raw)


# ---------------------------------------------------------------------------------- the domain


class Neo(Domain):
    """Close approaches of near-Earth objects (see the module docstring)."""

    name = "neo"
    noun = "close approach"
    plural = "close approaches"
    id_label = "approach ID"
    source_label = "catalogue"
    group_label = "body"
    group_preposition = "near"  # an approach is recorded near the body it passes, never in it

    fields = (MISS_DISTANCE, RELATIVE_VELOCITY, ABSOLUTE_MAGNITUDE)
    roles = {"primary": RELATIVE_VELOCITY.key, "secondary": MISS_DISTANCE.key}
    lexicon = LEXICON
    live = CadFeed()
    synthetic_window_hours = SYNTHETIC_WINDOW_DAYS * 24

    id_pattern = ID_PATTERN
    id_answer_type = "neo_approach_id"
    group_answer_type = "neo_body"
    statuses = ("confirmed", "preliminary")
    trap_vocab = TRAP_VOCAB
    chatter = CHATTER
    # An object's close approaches are each other's near twins: a digest that lists an object passing
    # Earth often lists it passing the Moon hours away (pipeline.sample_with_slots).
    twin_seconds = MOON_PAIR_HOURS * 3600
    twin_share = 0.5
    # An object passes the Moon within a km/s of its speed past Earth and about a lunar distance from its distance,
    # with the same H: a corrected velocity or distance agrees with its object's other approach as a real one does.
    twin_fields = (MISS_DISTANCE.key, RELATIVE_VELOCITY.key, ABSOLUTE_MAGNITUDE.key)

    def twin_key(self, record: Record) -> str:
        """An approach's twins are the other approaches of its object: its designation."""
        return designation_of(record.id)

    def planted_value(self, spec: FieldSpec, value: float, like: float, peers: Sequence[float] = ()) -> float:
        """An object's H is written to the places its own listing gives it, which go with its designation (a
        recently found object's to 3, an older one's to 1 or 2: _draw_object), whatever the catalogue's other
        objects have; the other fields as every domain's (Domain.planted_value)."""
        if spec.key == ABSOLUTE_MAGNITUDE.key:
            return super().planted_value(spec, value, like)
        return super().planted_value(spec, value, like, peers)

    def rereport_values(self, record: Record, slot: Record, rng: random.Random, *, same_id: bool) -> dict[str, float]:
        """What a report filed in a slot's place gives for an approach (Domain.rereport_values): the slot's miss
        distance and velocity, and, under the approach's own designation, the H of that object as another
        catalogue lists it: the record's, off by 0.05 to 0.3 (catalogues fit H to their own observations), at
        the places the record's designation gives it (_draw_object), so that a designation and its H never
        disagree. Under the slot's own designation, a report keeps the slot's H with the rest, its own object's.

        The trap builders file a report under an approach's own designation only where the object makes no
        other approach in the digest or its slots (traps._with_siblings, by id_place): every approach of an
        object prints one H, so beside another approach the report would have to give that H too, and so the
        very H of its record, a copy that no two real reports of a body share (an AUC of 0.95), or give
        another, which no real approach does (an AUC of 0 for its sibling's H, before the second round)."""
        values = dict(slot.values)
        if same_id:
            h = record.values[ABSOLUTE_MAGNITUDE.key]
            places = stored_places(h, ABSOLUTE_MAGNITUDE.decimals)
            step = 10.0 ** -places
            off = max(step, round(rng.uniform(0.05, 0.3) / step) * step)
            values[ABSOLUTE_MAGNITUDE.key] = round(h + rng.choice((-1, 1)) * off, places) + 0.0
        return values

    def id_place(self, record_id: str) -> tuple[str, int] | None:
        """An approach ID numbers the approaches of its object in time order (2026vk2-ca1 before 2026vk2-ca2):
        (its designation, its number). The approaches of one designation are siblings: they share the
        object's H."""
        designation, _, number = record_id.casefold().rpartition("-ca")
        return (designation, int(number)) if designation and number.isdigit() else None

    def sample_weight(self, record: Record) -> float:
        """Closer approaches make the digest more often: one that grazes the body is up to four times as
        likely as one at the 0.05 au limit."""
        return 1.0 + 3.0 * max(0.0, 1.0 - record.values[MISS_DISTANCE.key] / (DIST_MAX_AU * AU_KM))

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Record]:
        """`n` approaches in the window: objects drawn body by body (BODIES), each passing its body once,
        and an object that passes Earth close often passing the Moon within hours too."""
        weights = [body.weight for body in BODIES]
        taken: set[str] = set()
        approaches: list[Approach] = []
        while len(approaches) < n:
            body = rng.choices(BODIES, weights=weights, k=1)[0]
            approaches += _draw_object(rng, body, compiled_at, window_hours, taken)[:n - len(approaches)]
        return numbered(approaches)

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh approach ID: the first approach of a new designation shaped like one of those in use
        (every catalogue writes designations alike, so `source` changes nothing)."""
        designations = sorted({designation_of(ref) for ref in taken if ID_RE.fullmatch(ref.casefold())})
        example = rng.choice(designations) if designations else "2026VA1"
        return self._fresh(example, rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        return self._fresh(designation_of(example), rng, taken)

    def _fresh(self, example: str, rng: random.Random, taken: set[str]) -> str:
        """The first approach of a designation shaped like `example` that no ID in `taken` uses, added to
        `taken`: a new object, never another approach of one already in the digest."""
        used = {designation_of(ref) for ref in taken}
        for _ in range(1000):
            designation = _like(example, rng)
            ref = approach_id(designation, 1)
            if designation not in used and ID_RE.fullmatch(ref):
                taken.add(ref)
                return ref
        raise ValueError(f"no fresh designation like {example!r} is left")

    def report_time(self) -> str:
        """A report's time, as the shared rule on times gives it: the moment of its close approach."""
        return REPORT_TIME

    def rules_text(self) -> str:
        """The rules every record domain shares (record_rules), whose rule on times says what the time of a
        close approach is (report_time), then what close approaches need besides: how approach IDs tell an
        object's approaches apart, and what the fields mean."""
        shared = record_rules(self)
        last = max(int(number) for number in re.findall(r"^(\d+)\. ", shared, flags=re.MULTILINE))
        extra = [textwrap.fill(f"{number}. {rule}", width=RULES_WIDTH, subsequent_indent=" " * len(f"{number}. "),
                               break_long_words=False, break_on_hyphens=False)
                 for number, rule in enumerate(RULES, start=last + 1)]
        return shared + "\n".join(extra) + "\n"


NEO = Neo()
DOMAIN = NEO  # what the registry looks up (see evalgen.domains)
