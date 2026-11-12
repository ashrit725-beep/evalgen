"""Weather: hourly observations from US airport weather stations, as the NWS API serves them.

A record is one observation at one station: its time, the station's position, its US region (the
group, from the static station table below), the feed that filed it (the station itself, or a relay
feed that repeats it) and three numeric fields:

    temperature  °C (canonical), °F and K: affine units; signed     primary role: the warmest, its rank
                                                                    and time, thresholds
    wind speed   km/h (canonical), mph, kt and m/s                  secondary role: thresholds, the
                                                                    strongest and the lightest wind
    pressure     hPa (canonical) and inHg                           thresholds in the new pool's twins

Every conversion is exact as defined (1 mph = 1.609344 km/h, 1 kt = 1.852 km/h, 1 m/s = 3.6 km/h,
0 °C = 32 °F = 273.15 K, and 1 inHg = 33.86389 hPa, the conventional inch of mercury the NWS uses),
and every value is stored to a tenth of its canonical unit, which every printed form reads back to
(evalgen.units). Margins, threshold steps and rank gaps are scaled to each field's spread over a day
of US observations: a few tenths of a degree, half a km/h, a few tenths of a hectopascal.

IDs are "<feed>-<5 digits>": a station's observations carry its ICAO identifier ("KJFK-48213") and a
relay's own references carry the relay's name ("MADIS-30571"). The digits are an opaque serial, not
the observation's time: the trap builders draw a planted report's ID apart from its time, so an ID
that encoded the time would give every planted observation away.

Synthetic observations follow each station's climate (January and July means, diurnal range, mean
wind) at the season and local solar hour of their time, plus a weather system per region that moves
temperature, pressure and wind together over the window (``Weather.synthetic``).

The live source (``NwsObservations``) asks api.weather.gov for the last 24 hours of observations of
every station in the table, freezes every response as it was served (its HTTP status and the exact
text of its body) in one JSON bundle (raw_feed.json), which the snapshot hashes, and reads the bodies
back without the network: observations that lack a value, or whose value the NWS quality control
rejected or questioned, are dropped (no value is ever invented), and one observation per station and
UTC hour is kept (the hourly METAR when there is one). A station that fails on its own (an error
status, a timeout) is recorded as such and the others are kept. Replaying the frozen bundle
(GenerateConfig.snapshot_file) rebuilds the same snapshot; when the API cannot be reached, no station
answers, or the stations give too few usable observations, the synthetic generator is the fallback
(sources.build_domain_snapshot).

Synthetic observations are timed to the second and their values are continuous, as the reports the
trap builders plant are. Real NWS reports are not: they are timed to the minute, with winds in whole
knots and pressures in hundredths of an inch, and the generic trap builders take no hint from the
domain to plant reports that way. So in a digest of live observations a re-report or a phantom can
differ in form from the real reports, and a reader could spot it by its form rather than by the note
or notice that marks it. No answer key depends on this.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from ..sources import USER_AGENT
from ..units import Unit
from ..util import parse_utc_iso
from .base import DISTANCE_WORDINGS, RULES_WIDTH, Domain, FieldSpec, Lexicon, Record, Spatial, TrapVocab, record_rules

# ------------------------------------------------------------------------------ units and fields

CELSIUS = Unit("°C", "degrees Celsius")
FAHRENHEIT = Unit("°F", "degrees Fahrenheit", 5 / 9, -160 / 9, step=1.0)  # °C = (°F - 32) × 5/9
# °C = K - 273.15. Answers and restated thresholds in kelvins keep two places: a tenth of a degree
# Celsius always ends in 5 hundredths of a kelvin (0.7 °C is 273.85 K), which one place would round
# exactly halfway, either way.
KELVIN = Unit("K", "kelvins", 1.0, -273.15, decimals=2, step=1.0)
KMH = Unit("km/h", "kilometres per hour")
MPH = Unit("mph", "miles per hour", 1.609344, step=1.0)
KNOT = Unit("kt", "knots", 1.852, step=1.0)
MPS = Unit("m/s", "metres per second", 3.6, step=0.5)
HPA = Unit("hPa", "hectopascals")
# Altimeter settings are read to hundredths of an inch, so answers and restated thresholds keep two places.
INHG = Unit("inHg", "inches of mercury", 33.86389, decimals=2, step=0.05)

# A day of US observations spans tens of degrees, so thresholds sit on half degrees (or whole degrees,
# or whole °F and K), moved up to 1.5 °C from a quantile so that they rarely sit on an observed value,
# and kept 0.3 °C clear of every one; a ranked answer leads the next by half a degree.
TEMPERATURE = FieldSpec(
    key="temp", label="temperature", canonical_unit="°C", units=(CELSIUS, FAHRENHEIT, KELVIN), decimals=1,
    answer_type="weather_temperature", tolerance=(0.3, 0.0), margin=(0.3, 0.0), rank_gap=0.5,
    quantiles=(0.2, 0.8), threshold_step=0.5, filters=("at_least", "below"),
    terminals=("argmax", "rank", "max", "time_of_max"), signed=True, correctable=True, jitter=1.5,
    coarse_step=1.0,
)
# Sustained winds run from calm to about 50 km/h: whole-unit thresholds (half m/s), moved up to 3 km/h,
# never at zero, clear of every observation by half a km/h or 2 %.
WIND = FieldSpec(
    key="wind", label="wind speed", canonical_unit="km/h", units=(KMH, MPH, KNOT, MPS), decimals=1,
    answer_type="weather_wind_speed", tolerance=(0.5, 0.02), margin=(0.5, 0.02), rank_gap=1.8,
    quantiles=(0.25, 0.8), threshold_step=1.0, filters=("below", "at_least"), terminals=("max", "min"),
    jitter=3.0, positive_threshold=True,
)
# Sea-level pressures lie within a few tens of hectopascals of 1016: whole hPa (or 0.05 inHg), moved up
# to 1.5 hPa, clear of every observation by 0.3 hPa, which covers a threshold restated to 0.01 inHg.
# Pressure plays no role in P01-P12, so it is only ever a filter of the twins: no question ends in it.
PRESSURE = FieldSpec(
    key="pressure", label="pressure", canonical_unit="hPa", units=(HPA, INHG), decimals=1,
    answer_type="weather_pressure", tolerance=(0.5, 0.0), margin=(0.3, 0.0), rank_gap=1.0,
    quantiles=(0.2, 0.8), threshold_step=1.0, filters=("at_least", "below"), terminals=(),
    jitter=1.5,
)

# Stations are tens to hundreds of km apart, and a question's point lies 5 to 60 km from one of them.
SPATIAL = Spatial(
    radii={"km": (25, 40, 50, 75, 100, 150, 200, 300), "mi": (15, 25, 30, 50, 75, 100, 150, 200)},
    margin=(3.0, 0.06), nearest_gap=(3.0, 0.08), nearest_max_km=500.0, point_offset_km=(5.0, 60.0),
    point_label="COOP site {number} near {source}",
)

# ---------------------------------------------------------------------------------- stations

REGIONS = ("Northeast", "Southeast", "Midwest", "South Central", "Mountain West", "Pacific")


@dataclass(frozen=True)
class Station:
    """One observing station: where it is, its region, and the climate its synthetic observations
    follow (normals rounded from the 1991-2020 US climate normals)."""

    id: str  # ICAO identifier, the NWS stationId: "KJFK"
    name: str  # the station's name, printed as an observation's place
    lat: float
    lon: float
    region: str  # one of REGIONS
    january: float  # mean temperature in January, °C
    july: float  # mean temperature in July, °C
    diurnal: float  # mean daily range of temperature, °C
    wind_kt: float  # mean wind speed, knots


STATIONS = (
    Station("KBOS", "Boston, Logan International Airport", 42.366, -71.010, "Northeast", -1.4, 23.6, 8.0, 11.0),
    Station("KJFK", "New York, Kennedy International Airport", 40.640, -73.779, "Northeast", 0.9, 25.0, 8.0, 11.0),
    Station("KPHL", "Philadelphia, Philadelphia International Airport", 39.874, -75.242, "Northeast", 0.9, 26.4,
            10.0, 9.0),
    Station("KPIT", "Pittsburgh, Pittsburgh International Airport", 40.491, -80.233, "Northeast", -2.3, 22.9, 11.0,
            8.0),
    Station("KATL", "Atlanta, Hartsfield-Jackson International Airport", 33.641, -84.428, "Southeast", 7.0, 27.3,
            10.0, 8.0),
    Station("KCLT", "Charlotte, Charlotte/Douglas International Airport", 35.214, -80.943, "Southeast", 5.1, 26.8,
            12.0, 7.0),
    Station("KMCO", "Orlando, Orlando International Airport", 28.431, -81.308, "Southeast", 16.1, 28.3, 11.0, 7.0),
    Station("KMIA", "Miami, Miami International Airport", 25.796, -80.287, "Southeast", 20.3, 29.4, 7.0, 8.0),
    Station("KORD", "Chicago, Chicago-O'Hare International Airport", 41.974, -87.907, "Midwest", -4.4, 23.6, 9.0,
            10.0),
    Station("KMSP", "Minneapolis, Minneapolis-St Paul International Airport", 44.885, -93.222, "Midwest", -9.1,
            23.6, 10.0, 10.0),
    Station("KDTW", "Detroit, Detroit Metropolitan Wayne County Airport", 42.216, -83.355, "Midwest", -3.6, 23.4,
            10.0, 9.0),
    Station("KSTL", "St. Louis, Lambert-St. Louis International Airport", 38.749, -90.370, "Midwest", 0.6, 27.3,
            10.0, 8.0),
    Station("KDFW", "Dallas/Fort Worth, Dallas/Fort Worth International Airport", 32.900, -97.040, "South Central",
            8.3, 30.3, 11.0, 10.0),
    Station("KIAH", "Houston, George Bush Intercontinental Airport", 29.990, -95.337, "South Central", 11.7, 29.4,
            10.0, 7.0),
    Station("KOKC", "Oklahoma City, Will Rogers World Airport", 35.393, -97.601, "South Central", 3.4, 28.3, 12.0,
            11.0),
    Station("KMSY", "New Orleans, Louis Armstrong New Orleans International Airport", 29.993, -90.258,
            "South Central", 11.9, 28.6, 9.0, 7.0),
    Station("KDEN", "Denver, Denver International Airport", 39.856, -104.674, "Mountain West", -0.6, 23.6, 15.0,
            9.0),
    Station("KSLC", "Salt Lake City, Salt Lake City International Airport", 40.790, -111.979, "Mountain West", -1.1,
            27.3, 13.0, 8.0),
    Station("KPHX", "Phoenix, Phoenix Sky Harbor International Airport", 33.437, -112.008, "Mountain West", 13.7,
            35.4, 13.0, 6.0),
    Station("KABQ", "Albuquerque, Albuquerque International Sunport", 35.040, -106.609, "Mountain West", 2.4, 26.6,
            14.0, 8.0),
    Station("KSEA", "Seattle, Seattle-Tacoma International Airport", 47.450, -122.309, "Pacific", 5.6, 19.6, 9.0,
            8.0),
    Station("KPDX", "Portland, Portland International Airport", 45.590, -122.595, "Pacific", 5.2, 21.5, 10.0, 7.0),
    Station("KSFO", "San Francisco, San Francisco International Airport", 37.621, -122.379, "Pacific", 10.6, 17.9,
            8.0, 11.0),
    Station("KLAX", "Los Angeles, Los Angeles International Airport", 33.942, -118.408, "Pacific", 14.3, 21.4, 7.0,
            7.0),
)
STATION_BY_ID = {station.id: station for station in STATIONS}

# ----------------------------------------------------------------------------------- IDs

ID_PATTERN = r"[a-z][a-z0-9]{3,7}-\d{5}"  # "kjfk-48213", "madis-30571" (IDs are matched lower-cased)
_PREFIX = re.compile(r"[A-Z][A-Z0-9]{3,7}")
_ID = re.compile(r"(?P<prefix>[A-Za-z][A-Za-z0-9]{3,7})-\d{5}")
SERIALS = (10000, 100000)  # an ID's five digits


def id_prefix(source: str) -> str:
    """How the IDs a feed files begin: its name in capitals, letters and digits only ("KJFK",
    "NOAAPORT"). A name that gives no prefix of four to eight characters, starting with a letter, is
    an error."""
    prefix = re.sub(r"[^A-Za-z0-9]", "", source).upper()
    if not _PREFIX.fullmatch(prefix):
        raise ValueError(f"the feed {source!r} gives no ID prefix of 4 to 8 letters and digits")
    return prefix


def _fresh(prefix: str, rng: random.Random, taken: set[str]) -> str:
    while True:
        ref = f"{prefix}-{rng.randrange(*SERIALS)}"
        if ref not in taken:
            taken.add(ref)
            return ref


# ------------------------------------------------------------------------- wording and traps

LEXICON = Lexicon(
    records="observations",
    all_records="observations in the digest",
    filters={
        "at_least(temp)": ("with a temperature of at least {value}", "whose temperature was {value} or higher",
                           "at {value} or warmer"),
        "below(temp)": ("with a temperature below {value}", "colder than {value}",
                        "whose temperature was under {value}"),
        "below(wind)": ("with a wind speed below {value}", "with winds under {value}",
                        "whose wind speed was less than {value}"),
        "at_least(wind)": ("with a wind speed of at least {value}", "with winds of {value} or more",
                           "whose wind speed was {value} or higher"),
        "at_least(pressure)": ("with a pressure of at least {value}", "whose pressure was {value} or higher"),
        "below(pressure)": ("with a pressure below {value}", "whose pressure was under {value}"),
        "within_distance": DISTANCE_WORDINGS,  # never read as a range of wind speeds or temperatures
        "time_between": ("recorded between {start} and {end}", "taken from {start} up to {end}"),
        "in_group": ("in the {group} region", "from stations in {group}", "whose region is {group}"),
    },
    questions={
        "count": ("How many distinct {records} are there?", "Count the distinct {records}.",
                  "What is the number of separate {records}?"),
        "argmax(temp)": ("Among the {records}, which one was the warmest? Give its observation ID.",
                         "What is the observation ID of the warmest of the {records}?"),
        "rank(temp)": ("Among the {records}, which one has the {nth}-highest temperature? Give its observation ID.",
                       "Rank the {records} by temperature, warmest first: which one is {nth}? Give its observation "
                       "ID."),
        "max(temp)": ("What was the highest temperature, in {unit}, among the {records}? {rounding}",
                      "Among the {records}, how warm was the warmest one, in {unit}? {rounding}"),
        "time_of_max(temp)": ("At what time (UTC, HH:MM) was the warmest of the {records} recorded?",
                              "When, in UTC (HH:MM), was the highest temperature among the {records} observed?"),
        "max(wind)": ("What was the highest wind speed, in {unit}, among the {records}? {rounding}",
                      "Among the {records}, how strong was the strongest wind, in {unit}? {rounding}"),
        "min(wind)": ("What was the lowest wind speed, in {unit}, among the {records}? {rounding}",
                      "Among the {records}, how light was the lightest wind, in {unit}? {rounding}"),
        "nearest_to": ("Which of the {records} was closest to {place}? Give its observation ID.",
                       "Among the {records}, give the observation ID of the one nearest to {place}."),
        "nearest_distance": ("How far from {place}, in {unit}, is the closest of the {records}? {rounding}",
                             "Among the {records}, what is the distance in {unit} from {place} to the nearest one? "
                             "{rounding}"),
        "compare_groups": ("Counting only {records}, which region has more of them: {a} or {b}? Answer \"tie\" if "
                           "the counts are equal.",
                           "Which region had more {records}: {a} or {b}? Say \"tie\" if they are equal."),
    },
)

# Relay feeds repeat a station's observation: under its ID, or under a reference of their own with a
# note naming it. No note says "relay", which the rules give a meaning of their own ("a forwarded or
# relayed report ... counts, once"): a re-report noted "relay of" an ID would read as a report that counts
# on its own, where rule 1 makes it that observation. A phantom is withdrawn for a reason a station
# really fails for, and a temperature is corrected as a corrected METAR would be.
TRAP_VOCAB = TrapVocab(
    rereporting_sources=("MADIS", "NOAAPORT", "AWOSNET"),
    duplicate_notes=("same observation as", "duplicate of", "mirror of"),
    retraction_reasons=("sensor fault", "test transmission", "frozen sensor", "maintenance in progress",
                        "garbled transmission"),
    retraction_notices=("RETRACTED", "withdrawn", "cancelled"),
    correction_notices=("corrected to", "revised to", "amended to"),
)

# Lines without data ({station} is a sensor of one of the digest's feeds, see digest.make_chatter).
CHATTER = (
    "Sensor {station} was offline from {h1} to {h2} UTC for maintenance; nothing it sent then is included.",
    "Wind speeds are sustained two-minute averages; gusts are not listed in this digest.",
    "Editor's note: entries are reproduced exactly as each {source_label} filed them, units and time conventions "
    "included.",
    "Reminder: {plural} are listed in the units their {source_label} reports in; read the unit printed with every "
    "value.",
    "Pressures are altimeter settings, reduced to sea level.",
    "This digest is compiled automatically and has not been checked by a forecaster.",
)

# The reading rule a weather digest adds to every record domain's rules (record_rules).
WEATHER_RULE = ("Temperatures are signed numbers, so -2.5 °C is colder than 0.5 °C. The warmest weather observation "
                "is the one with the highest temperature. A wind speed of 0 means calm air.")

# ----------------------------------------------------------------------- the synthetic weather

GAMMA_1_5 = math.gamma(1.5)  # the mean of a Weibull distribution of shape 2 is its scale times this
MAX_WIND_KMH = 100.0  # sustained winds above this are left to hurricanes
MEAN_PRESSURE_HPA = 1016.0


@dataclass(frozen=True)
class _System:
    """The weather over one region during a synthetic window: anomalies of temperature and pressure
    that drift as a front passes, and how much windier than usual it is (lower pressure, more wind)."""

    temp_anomaly: float  # °C at the middle of the window
    temp_trend: float  # °C of change over the window
    pressure_anomaly: float  # hPa at the middle of the window
    pressure_trend: float  # hPa of change over the window
    wind_factor: float

    @classmethod
    def draw(cls, rng: random.Random) -> _System:
        temp_anomaly, temp_trend = rng.gauss(0.0, 3.0), rng.gauss(0.0, 2.0)
        pressure_anomaly, pressure_trend = rng.gauss(0.0, 7.0), rng.gauss(0.0, 4.0)
        wind_factor = math.exp(rng.gauss(0.0, 0.2) - pressure_anomaly / 50.0)
        return cls(temp_anomaly, temp_trend, pressure_anomaly, pressure_trend, wind_factor)


def _solar_hour(when: int, lon: float) -> float:
    """Local mean solar time, in hours, at longitude `lon`."""
    return ((when % 86400) / 3600.0 + lon / 15.0) % 24.0


def _seasonal_mean(station: Station, when: int) -> float:
    """The station's normal daily mean temperature on that day: coldest around 20 January."""
    day = datetime.fromtimestamp(when, tz=timezone.utc).timetuple().tm_yday + (when % 86400) / 86400.0
    middle, amplitude = (station.january + station.july) / 2, (station.july - station.january) / 2
    return middle - amplitude * math.cos(2 * math.pi * (day - 20.0) / 365.25)


def _stored(value: float) -> float:
    """A value as the fields store it: to a tenth, and never -0.0."""
    return round(value, 1) + 0.0


# ------------------------------------------------------------------------------ the domain


class Weather(Domain):
    """Observations from US airport weather stations (see the module docstring)."""

    name = "weather"
    # "weather observation", not "observation": the generic rules write "A {noun} may appear ...".
    noun = "weather observation"
    plural = "weather observations"
    id_label = "observation ID"
    source_label = "feed"
    group_label = "region"

    fields = (TEMPERATURE, WIND, PRESSURE)
    roles = {"primary": TEMPERATURE.key, "secondary": WIND.key}
    spatial = SPATIAL
    lexicon = LEXICON
    synthetic_window_hours = 24

    id_pattern = ID_PATTERN
    id_answer_type = "weather_observation_id"
    group_answer_type = "weather_region"
    statuses = ("verified", "preliminary")  # the NWS quality-control codes V and Z
    trap_vocab = TRAP_VOCAB
    chatter = CHATTER

    def __init__(self, live: NwsObservations | None = None) -> None:
        self.live = NwsObservations() if live is None else live

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Record]:
        """`n` observations at stations drawn alike, at any second of the window: each region's weather
        system is drawn first, then every station's own offsets of temperature and pressure, then the
        observations one by one."""
        span = window_hours * 3600
        systems = {region: _System.draw(rng) for region in REGIONS}
        offsets = {station.id: (rng.gauss(0.0, 1.0), rng.gauss(0.0, 0.8)) for station in STATIONS}
        taken: set[str] = set()
        records = []
        for _ in range(n):
            station = rng.choice(STATIONS)
            when = compiled_at - rng.randrange(60, span)
            progress = (when - (compiled_at - span)) / span - 0.5  # -0.5 at the window's start, 0.5 at its end
            system, (temp_offset, pressure_offset) = systems[station.region], offsets[station.id]
            hour = _solar_hour(when, station.lon)
            afternoon = math.cos(2 * math.pi * (hour - 15.0) / 24.0)  # 1 at 15:00 local time, -1 at 03:00
            temp = (_seasonal_mean(station, when) + station.diurnal / 2 * afternoon + system.temp_anomaly
                    + system.temp_trend * progress + temp_offset + rng.gauss(0.0, 0.5))
            if rng.random() < 0.25:  # a five-minute observation: whole degrees
                temp = float(round(temp))
            mean_wind = station.wind_kt * KNOT.scale * system.wind_factor * (1.0 + 0.25 * afternoon)
            wind = min(rng.weibullvariate(mean_wind / GAMMA_1_5, 2.0), MAX_WIND_KMH)
            pressure = (MEAN_PRESSURE_HPA + system.pressure_anomaly + system.pressure_trend * progress
                        + pressure_offset + rng.gauss(0.0, 0.4))
            records.append(Record(id=self.new_id(station.id, rng, taken), time=when, lat=station.lat,
                                  lon=station.lon, group=station.region, source=station.id,
                                  values={"temp": _stored(temp), "wind": _stored(wind),
                                          "pressure": _stored(pressure)},
                                  place=station.name))
        return sorted(records, key=lambda r: (r.time, r.id))

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh "<feed>-<5 digits>" ID for an observation `source` files, added to `taken`."""
        return _fresh(id_prefix(source), rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID with the prefix of `example`, added to `taken`."""
        match = _ID.fullmatch(example)
        if match is None:
            raise ValueError(f"{example!r} is not a weather observation ID")
        return _fresh(match["prefix"], rng, taken)

    def rules_text(self) -> str:
        """The reading rules of every record domain (record_rules) in this domain's words, and one more
        on temperatures and calm air."""
        rules = record_rules(self)
        number = len(re.findall(r"^\d+\. ", rules, flags=re.MULTILINE)) + 1
        extra = textwrap.fill(f"{number}. {WEATHER_RULE}", width=RULES_WIDTH, subsequent_indent=" " * 4,
                              break_long_words=False, break_on_hyphens=False)
        return rules + extra + "\n"


# --------------------------------------------------------------------------- the live source

API_ROOT = "https://api.weather.gov"
OBSERVATIONS_URL = API_ROOT + "/stations/{station}/observations"
# A bundle holds each station's response as served: {"status": <HTTP status>, "body": <the body's exact
# text>}, or {"error": ...} for a request that failed on its own (a timeout, a dropped connection).
BUNDLE_FORMAT = "evalgen-nws-bundle/2"
FAILURES_NAMED = 4  # how many failed stations the error of a fetch that no station answered names
# NWS quality-control codes of values that are not used: questioned, subjectively bad, rejected.
REJECTED_QC = frozenset({"Q", "B", "X"})
# The NWS value of each field, its accepted units (WMO unit codes) as maps to the canonical unit, and
# the range a real reading lies in; anything else is dropped.
Convert = Callable[[float], float]
NWS_VALUES: Mapping[str, tuple[str, Mapping[str, Convert], tuple[float, float]]] = {
    "temp": ("temperature", {"degC": lambda v: v, "degF": lambda v: (v - 32.0) * 5.0 / 9.0,
                             "K": lambda v: v - 273.15}, (-70.0, 60.0)),
    "wind": ("windSpeed", {"km_h-1": lambda v: v, "m_s-1": lambda v: v * 3.6, "kt": lambda v: v * 1.852},
             (0.0, 400.0)),
    "pressure": ("barometricPressure", {"Pa": lambda v: v / 100.0, "hPa": lambda v: v}, (850.0, 1100.0)),
}


def _utc_now() -> datetime:
    """The clock, read only when the live source is fetched (never when a bundle is parsed)."""
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class _Reading:
    """One usable observation of the bundle, before it is given an ID."""

    station: Station
    time: int
    lat: float
    lon: float
    values: dict[str, float]
    metar: bool  # it carries a raw METAR (the hourly report), not only the five-minute data


@dataclass(frozen=True)
class NwsObservations:
    """The NWS API as the weather domain's live source (a LiveAdapter).

    ``fetch`` asks for the last `window_hours` of observations of every station in `stations` (one
    request each, with the User-Agent the API requires) and returns them as one JSON bundle that keeps
    every response as the API served it, its HTTP status and the exact text of its body, so that the
    snapshot's hash commits to what the NWS sent, not to evalgen's reading of it. ``parse`` reads the
    bodies of a bundle without the clock or the network, so replaying it rebuilds the snapshot
    exactly."""

    stations: tuple[str, ...] = tuple(station.id for station in STATIONS)
    min_records: int = 40  # fewer usable observations than this and the snapshot falls back to synthetic
    window_hours: int = 24
    limit: int = 500  # observations asked per station (the API's maximum; a day of five-minute data is 288)
    timeout: float = 20.0
    source: str = "nws"
    raw_filename: str = "raw_feed.json"
    user_agent: str = USER_AGENT

    def fetch(self) -> tuple[bytes, str]:
        """Every station's observations as one bundle: (its bytes, the URL asked for each station).

        Each response is kept as served, its HTTP status and the exact text of its body (UTF-8, as JSON
        is), whatever the status; parse decides what can be read. A request that fails on its own (a
        timeout, a connection dropped after the first station answered, a body that is not UTF-8) is
        recorded as an error for that station, and the other stations are still asked. Only a first
        request that cannot connect (no network: the others would not either) raises at once, and a
        fetch in which no station answered with observations raises ValueError."""
        fetched_at = _utc_now().replace(microsecond=0)
        params = {"start": _iso(fetched_at - timedelta(hours=self.window_hours)), "limit": self.limit}
        headers = {"User-Agent": self.user_agent, "Accept": "application/geo+json"}
        responses: dict[str, Any] = {}
        for index, station in enumerate(self.stations):
            try:
                response = requests.get(OBSERVATIONS_URL.format(station=station), params=params, headers=headers,
                                        timeout=self.timeout)
            except requests.ConnectionError as exc:  # a timeout while connecting too
                if index == 0:
                    raise
                responses[station] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            except requests.RequestException as exc:  # a timeout reading this station's answer, say
                responses[station] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            try:
                responses[station] = {"status": response.status_code, "body": response.content.decode("utf-8")}
            except UnicodeDecodeError as exc:
                responses[station] = {"status": response.status_code, "error": f"UnicodeDecodeError: {exc}"}
        if not any(_payload(r) is not None for r in responses.values()):
            failed = [f"{station}: {_failure(r)}" for station, r in responses.items()]
            more = len(failed) - FAILURES_NAMED
            raise ValueError("no station answered with observations (" + "; ".join(failed[:FAILURES_NAMED])
                             + (f"; and {more} more" if more > 0 else "") + ")")
        url = f"{OBSERVATIONS_URL}?start={params['start']}&limit={self.limit}"
        bundle = {"format": BUNDLE_FORMAT, "fetched_at": _iso(fetched_at), "window_hours": self.window_hours,
                  "url": url, "user_agent": self.user_agent, "stations": responses}
        return json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), url

    def parse(self, raw: bytes) -> tuple[list[Record], int, int]:
        """The bundle's observations as Records, its fetch time (the snapshot's compiled_at) and its
        window. Each station's body is read as JSON when the API answered it with success; only stations
        in the table count; an observation outside the window, without a value of every field, or with
        a value the NWS rejected or questioned, is dropped; of each station's observations in one UTC
        hour, the latest hourly METAR is kept, or else the latest observation. Raises ValueError (or
        KeyError) on bytes that are not a bundle."""
        bundle = json.loads(raw.decode("utf-8"))
        if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"not a bundle of NWS observations ({BUNDLE_FORMAT})")
        compiled_at = parse_utc_iso(str(bundle["fetched_at"]))
        window, stations = bundle.get("window_hours", self.window_hours), bundle["stations"]
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 24 * 7:
            raise ValueError(f"the bundle covers {window!r} hours, not a whole number of hours up to a week")
        if not isinstance(stations, dict):
            raise ValueError("the bundle gives no responses by station")
        start = compiled_at - window * 3600
        readings = []
        for station_id, response in stations.items():
            station = STATION_BY_ID.get(station_id)
            if station is None:
                continue
            for feature in _features(_payload(response)):
                reading = _reading(feature, station)
                if reading is not None and start <= reading.time < compiled_at:
                    readings.append(reading)
        return _identified(_hourly(readings)), compiled_at, window


def _payload(response: Any) -> Any:
    """What one station's frozen response says: its body read as JSON, when the API answered with
    success (a 2xx status); None for an error status, a failed request or a body that is not JSON."""
    if not isinstance(response, dict) or not isinstance(response.get("body"), str):
        return None
    status = response.get("status")
    if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
        return None
    try:
        return json.loads(response["body"])
    except ValueError:
        return None


def _failure(response: Any) -> str:
    """Why a station's response holds no observations, in a few words."""
    if isinstance(response, dict) and "error" in response:
        return str(response["error"])
    status = response.get("status") if isinstance(response, dict) else None
    return f"HTTP {status}" if isinstance(status, int) and not 200 <= status < 300 else "a body that is not JSON"


def _features(payload: Any) -> list[Mapping[str, Any]]:
    """The observations of one station's response: a feature collection (the observations endpoint)
    or a single feature (the latest-observation endpoint); none for anything else."""
    if not isinstance(payload, dict):
        return []
    if payload.get("type") == "FeatureCollection":
        return [f for f in payload.get("features") or [] if isinstance(f, dict)]
    return [payload] if payload.get("type") == "Feature" else []


def _reading(feature: Mapping[str, Any], station: Station) -> _Reading | None:
    """One observation of `station` as the API gives it, or None when it cannot be used as it is."""
    props = feature.get("properties")
    if not isinstance(props, dict) or props.get("stationId", station.id) != station.id:
        return None
    try:
        when = parse_utc_iso(str(props["timestamp"]))
    except (KeyError, ValueError):
        return None
    values = {}
    for key, (name, units, (low, high)) in NWS_VALUES.items():
        value = _value(props.get(name), units)
        if value is None or not low <= value <= high:
            return None
        values[key] = _stored(value)
    lat, lon = _position(feature.get("geometry"), station)
    return _Reading(station, when, lat, lon, values, metar=bool(props.get("rawMessage")))


def _value(quantity: Any, units: Mapping[str, Convert]) -> float | None:
    """A quantity of the API ({"unitCode": "wmoUnit:degC", "value": 11, "qualityControl": "V"}) in the
    canonical unit, or None when it has no value, a unit not in `units`, or a rejected value."""
    if not isinstance(quantity, dict):
        return None
    value, code = quantity.get("value"), str(quantity.get("unitCode") or "")
    convert = units.get(code.rsplit(":", 1)[-1])
    if (convert is None or quantity.get("qualityControl") in REJECTED_QC or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)):
        return None
    return float(convert(float(value)))


def _position(geometry: Any, station: Station) -> tuple[float, float]:
    """Where the API puts the observation, to the domain's places (its point is [lon, lat]), or the
    station table's position when it gives none."""
    coords = geometry.get("coordinates") if isinstance(geometry, dict) and geometry.get("type") == "Point" else None
    if (isinstance(coords, list) and len(coords) >= 2 and all(isinstance(c, (int, float)) for c in coords[:2])
            and -90.0 <= coords[1] <= 90.0 and -180.0 <= coords[0] <= 180.0):
        places = SPATIAL.coord_decimals
        return round(float(coords[1]), places) + 0.0, round(float(coords[0]), places) + 0.0
    return station.lat, station.lon


def _hourly(readings: Iterable[_Reading]) -> list[_Reading]:
    """One observation per station and UTC hour: the latest METAR of the hour, or else the latest one."""
    kept: dict[tuple[str, int], _Reading] = {}
    for reading in readings:
        slot = (reading.station.id, reading.time // 3600)
        best = kept.get(slot)
        if best is None or (reading.metar, reading.time) > (best.metar, best.time):
            kept[slot] = reading
    return list(kept.values())


def _identified(readings: Sequence[_Reading]) -> list[Record]:
    """The readings as Records in time order, each with an ID that depends only on its station and time
    (a hash, so replaying a bundle gives the same IDs, and a collision takes the next free serial)."""
    taken: set[str] = set()
    records = []
    for reading in sorted(readings, key=lambda r: (r.time, r.station.id)):
        station = reading.station
        digest = hashlib.sha256(f"{station.id}|{reading.time}".encode("utf-8")).digest()
        low, high = SERIALS
        serial = int.from_bytes(digest[:8], "big") % (high - low)
        while f"{station.id}-{low + serial}" in taken:
            serial = (serial + 1) % (high - low)
        ref = f"{station.id}-{low + serial}"
        taken.add(ref)
        records.append(Record(id=ref, time=reading.time, lat=reading.lat, lon=reading.lon, group=station.region,
                              source=station.id, values=dict(reading.values), place=station.name))
    return records


WEATHER = Weather()
DOMAIN = WEATHER  # what the registry looks up (see evalgen.domains)
