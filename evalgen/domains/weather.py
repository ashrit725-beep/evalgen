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
(evalgen.units). Margins and threshold steps are scaled to each field's spread over a day of US
observations: a few tenths of a degree, half a km/h, a few tenths of a hectopascal; the warmest
observation, the one ranked field, leads the next by half a degree.

IDs are "WX-" and six digits ("WX-482137"): an opaque serial, which names neither the station nor the
feed nor the time. A re-report filed under its observation's ID gives a time, a station and values of its
own (traps.py: it stands in for an observation drawn with the digest but not kept), so an ID that named
the time or the station would contradict the line it is printed on. An ID that named the station would
also pin the re-report to its observation's station, so that it could never be the only report at its
station, as a real observation often is (an AUC of about 0.37 for the number of other observations at its
station, and 0.40 for the time to the nearest one, when IDs began with the station's identifier).

Synthetic observations follow each station's climate (January and July means, diurnal range, mean
wind) at the season and local solar hour of their time, plus a weather system per region that moves
temperature, pressure and wind together over the window (``Weather.synthetic``). They are timed and
written as NWS observations are: at the station's own minute of the hour or, a quarter of them, on a
five-minute mark (a five-minute observation, in whole degrees), with winds in whole knots and
pressures in hundredths of an inch of mercury; a fifth of them reach the digest through a relay feed.

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

Off the legacy path a planted report is an observation drawn with the digest but not kept (traps.py),
so in a live digest as in a synthetic one a re-report or a phantom has the form of a real observation:
only its note, its status or a notice tells it apart.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests
from urllib3.exceptions import ProtocolError

from .. import sources
from ..geo import haversine_km
from ..sources import EARLIEST_TIME, LATEST_TIME, USER_AGENT, failure_text, read_feed_json
from ..units import Unit
from ..util import parse_utc_iso
from .base import DISTANCE_WORDINGS, RULES_WIDTH, Domain, FieldSpec, Lexicon, Record, Spatial, TrapVocab, record_rules

# ------------------------------------------------------------------------------ units and fields

CELSIUS = Unit("°C", "degrees Celsius", one="degree Celsius")
FAHRENHEIT = Unit("°F", "degrees Fahrenheit", 5 / 9, -160 / 9, step=1.0, one="degree Fahrenheit")  # °C = (°F - 32) × 5/9
# °C = K - 273.15. Answers and restated thresholds in kelvins keep two places: a tenth of a degree
# Celsius always ends in 5 hundredths of a kelvin (0.7 °C is 273.85 K), which one place would round
# exactly halfway, either way.
KELVIN = Unit("K", "kelvins", 1.0, -273.15, decimals=2, step=1.0, one="kelvin")
KMH = Unit("km/h", "kilometres per hour", one="kilometre per hour")
MPH = Unit("mph", "miles per hour", 1.609344, step=1.0, one="mile per hour")
KNOT = Unit("kt", "knots", 1.852, step=1.0, one="knot")
MPS = Unit("m/s", "metres per second", 3.6, step=0.5, one="metre per second")
HPA = Unit("hPa", "hectopascals", one="hectopascal")
# Altimeter settings are read to hundredths of an inch, so thresholds asked or restated in inches keep two
# places (no question ends in a pressure, so none is answered in inches).
INHG = Unit("inHg", "inches of mercury", 33.86389, decimals=2, step=0.05, one="inch of mercury")

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
    answer_type="weather_wind_speed", tolerance=(0.5, 0.02), margin=(0.5, 0.02),
    rank_gap=0.0,  # no ranked terminal reads it
    quantiles=(0.25, 0.8), threshold_step=1.0, filters=("below", "at_least"), terminals=("max", "min"),
    jitter=3.0, positive_threshold=True,
)
# Sea-level pressures lie within a few tens of hectopascals of 1016: whole hPa (or 0.05 inHg), moved up
# to 1.5 hPa, clear of every observation by 0.3 hPa, which covers a threshold restated to 0.01 inHg.
# Pressure plays no role in P01-P12, so it is only ever a filter of the twins: no question ends in it.
PRESSURE = FieldSpec(
    key="pressure", label="pressure", canonical_unit="hPa", units=(HPA, INHG), decimals=1,
    answer_type="weather_pressure", tolerance=(0.5, 0.0), margin=(0.3, 0.0),
    rank_gap=0.0,  # no ranked terminal reads it
    quantiles=(0.2, 0.8), threshold_step=1.0, filters=("at_least", "below"), terminals=(),
    jitter=1.5,
)

# Stations are tens to hundreds of km apart, and a question's point lies 5 to 60 km from one of them. The
# point is named by its own number alone: the station it lies near is almost always the answer's.
SPATIAL = Spatial(
    radii={"km": (25, 40, 50, 75, 100, 150, 200, 300), "mi": (15, 25, 30, 50, 75, 100, 150, 200)},
    margin=(3.0, 0.06), nearest_gap=(3.0, 0.08), nearest_max_km=500.0, point_offset_km=(5.0, 60.0),
    point_label="COOP site {number}",
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
STATION_BY_NAME = {station.name: station for station in STATIONS}

# Local times. Each station's time zone, as its standard offset from UTC in hours; every one keeps US
# daylight saving time but Phoenix (Arizona does not). Digests write an observation's local time in the
# offset in force at its station at the observation's time (Weather.utc_offset), stated with the time,
# so a station is never given a zone it does not have.
STATION_ZONES = {"KBOS": -5, "KJFK": -5, "KPHL": -5, "KPIT": -5, "KATL": -5, "KCLT": -5, "KMCO": -5, "KMIA": -5,
                 "KORD": -6, "KMSP": -6, "KDTW": -5, "KSTL": -6, "KDFW": -6, "KIAH": -6, "KOKC": -6, "KMSY": -6,
                 "KDEN": -7, "KSLC": -7, "KPHX": -7, "KABQ": -7, "KSEA": -8, "KPDX": -8, "KSFO": -8, "KLAX": -8}
NO_DAYLIGHT_SAVING = frozenset({"KPHX"})
STATION_REACH_KM = 100.0  # an observation without its station's name belongs to the nearest station this close


def _sunday(year: int, month: int, n: int) -> datetime:
    """Midnight, as a UTC time, of the date of the n-th Sunday of a month."""
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def us_daylight_saving(ts: int, standard: int) -> bool:
    """Whether US daylight saving time is in force at the Unix time `ts` in a zone `standard` hours from
    UTC: from 2:00 local standard time on the second Sunday of March to 2:00 local daylight time (1:00
    standard time) on the first Sunday of November, the rule since 2007. Computed from the calendar,
    never from the host's time zone database, so that a run regenerates alike on every machine."""
    year = datetime.fromtimestamp(ts, tz=timezone.utc).year
    start = _sunday(year, 3, 2) + timedelta(hours=2 - standard)
    end = _sunday(year, 11, 1) + timedelta(hours=1 - standard)
    return start.timestamp() <= ts < end.timestamp()


def station_of(record: Any) -> Station | None:
    """The station an observation was made at: the one its place names (every observation's place is its
    station's name), else the nearest one within STATION_REACH_KM of its position; None for neither."""
    station = STATION_BY_NAME.get(getattr(record, "place", None) or "")
    if station is not None or record.lat is None or record.lon is None:
        return station
    nearest = min(STATIONS, key=lambda s: haversine_km(record.lat, record.lon, s.lat, s.lon))
    return nearest if haversine_km(record.lat, record.lon, nearest.lat, nearest.lon) <= STATION_REACH_KM else None

# ----------------------------------------------------------------------------------- IDs

ID_PATTERN = r"wx-\d{6}"  # "wx-482137" (IDs are matched lower-cased; digests print "WX-482137")
ID_PREFIX = "WX-"
_ID = re.compile(r"wx-\d{6}", re.IGNORECASE)
SERIALS = (100000, 1000000)  # an ID's six digits


def _fresh(rng: random.Random, taken: set[str]) -> str:
    """A new ID, not in `taken` (and added to it)."""
    while True:
        ref = f"{ID_PREFIX}{rng.randrange(*SERIALS)}"
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
        # A region is the one printed on the report (rule 8), whoever filed it: never "from stations in", which a
        # reader could take to leave out the observations a relay feed filed (the glossary calls whoever files a
        # report a feed, and a radio net calls its call-in units stations).
        "in_group": ("in the {group} region", "taken in the {group} region", "whose region is {group}"),
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
        "time_of_max(temp)": ("Among the {records}, at what time (UTC, HH:MM) was the warmest one recorded?",
                              "Among the {records}, when (UTC, HH:MM) was the highest temperature observed?"),
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

# Lines without data. {station} is a sensor at one of the digest's stations ("KIAH.PGH"), never at a relay
# feed, which has none, and its window holds no observation made at that station, whichever feed filed it
# (Weather.status_station, digest.timed_chatter).
CHATTER = (
    "Sensor {station} was offline for maintenance from {h1} UTC on {date} until {until}.",
    "Wind speeds are sustained two-minute averages; gusts are not listed in this digest.",
    "Editor's note: entries are reproduced exactly as each {source_label} filed them, units and time conventions "
    "included.",
    "Reminder: units vary from entry to entry; read the unit printed with every value.",
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
# When observations are taken, as at NWS stations: each station's routine (hourly) observation at a minute
# of its own near the end of every hour, and five-minute observations, which give whole degrees.
ROUTINE_MINUTES = (51, 52, 53, 54, 56)
FIVE_MINUTE_SHARE = 0.25
# Feeds that relay stations' observations. They file a share of the observations themselves (one that
# reached the digest through them), as the stations' own feeds file the rest.
RELAY_SHARE = 0.2
RELAY_WEIGHTS = (0.5, 0.3, 0.2)  # of TRAP_VOCAB.rereporting_sources, in its order


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


def whole_knots(kmh: float) -> float:
    """A wind speed (km/h) as an observation reports it: in whole knots, stored in km/h to a tenth."""
    return _stored(max(0, round(kmh / KNOT.scale)) * KNOT.scale)


def altimeter(hpa: float) -> float:
    """A pressure (hPa) as an observation reports it: in hundredths of an inch of mercury, stored in hPa to
    a tenth."""
    return _stored(round(hpa / INHG.scale, 2) * INHG.scale)


# ------------------------------------------------------------------------------ the domain


class Weather(Domain):
    """Observations from US airport weather stations (see the module docstring)."""

    name = "weather"
    # The records' noun in every rule, glossary and question ("a weather observation"); the generic rules give
    # whatever noun a domain names its records by the article that fits it (domains.base.indefinite).
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
    # A station's observations are each other's near twins (Domain.twin_key: its region and its place): a
    # digest often lists a station's next observation, five minutes or an hour later.
    twin_seconds = 2 * 3600
    twin_share = 0.6
    # A station's temperature and pressure move little from one observation to the next (its wind does not:
    # gusts come and go); a corrected temperature agrees with the station's other observations as a real one does.
    twin_fields = (TEMPERATURE.key, PRESSURE.key)

    def __init__(self, live: NwsObservations | None = None) -> None:
        self.live = NwsObservations() if live is None else live

    def utc_offset(self, record: Any) -> int:
        """The offset of the local time in force at an observation's station at its time: the station's
        standard offset, an hour more under US daylight saving time (station_of, STATION_ZONES). An
        observation at no known station keeps the nominal offset of its longitude (Domain.utc_offset)."""
        station = station_of(record)
        if station is None:
            return super().utc_offset(record)
        standard = STATION_ZONES[station.id]
        summer = station.id not in NO_DAYLIGHT_SAVING and us_daylight_saving(record.time, standard)
        return standard + (1 if summer else 0)

    def status_station(self, record: Any) -> str:
        """The station an observation was made at (station_of), whichever feed filed it: a status line about
        one of its sensors ("Sensor KIAH.PGH was offline ...") is about every observation made there, the
        relayed ones too, and no line names a relay feed, which has no sensors. An observation at no known
        station keeps its feed (Domain.status_station)."""
        station = station_of(record)
        return super().status_station(record) if station is None else station.id

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Record]:
        """`n` observations at stations drawn alike, timed as NWS stations take them: each region's weather
        system is drawn first, then every station's own offsets of temperature and pressure and the minute
        of its routine observations, then the observations one by one, each at its station's routine minute
        of an hour of the window or, a share of the time, on a five-minute mark (a five-minute observation),
        to the minute. Values are reported as observations report them: temperatures to a tenth of a degree
        (whole degrees in a five-minute observation), winds in whole knots and pressures in hundredths of an
        inch of mercury, stored in the canonical units. A share of the observations are filed by a relay
        feed. IDs are opaque serials (see the module docstring)."""
        span = window_hours * 3600
        systems = {region: _System.draw(rng) for region in REGIONS}
        offsets = {station.id: (rng.gauss(0.0, 1.0), rng.gauss(0.0, 0.8)) for station in STATIONS}
        minutes = {station.id: rng.choice(ROUTINE_MINUTES) for station in STATIONS}
        taken: set[str] = set()
        seen: set[tuple[str, int]] = set()
        records = []
        while len(records) < n:
            station = rng.choice(STATIONS)
            drawn = compiled_at - rng.randrange(60, span)
            five_minute = rng.random() < FIVE_MINUTE_SHARE
            when = drawn - drawn % 300 if five_minute else drawn - drawn % 3600 + minutes[station.id] * 60
            if not compiled_at - span < when <= compiled_at - 60 or (station.id, when) in seen:
                continue  # inside the window, a minute before the compilation at the latest, as ever
            seen.add((station.id, when))
            progress = (when - (compiled_at - span)) / span - 0.5  # -0.5 at the window's start, 0.5 at its end
            system, (temp_offset, pressure_offset) = systems[station.region], offsets[station.id]
            hour = _solar_hour(when, station.lon)
            afternoon = math.cos(2 * math.pi * (hour - 15.0) / 24.0)  # 1 at 15:00 local time, -1 at 03:00
            temp = (_seasonal_mean(station, when) + station.diurnal / 2 * afternoon + system.temp_anomaly
                    + system.temp_trend * progress + temp_offset + rng.gauss(0.0, 0.5))
            if five_minute:  # whole degrees
                temp = float(round(temp))
            mean_wind = station.wind_kt * KNOT.scale * system.wind_factor * (1.0 + 0.25 * afternoon)
            wind = min(rng.weibullvariate(mean_wind / GAMMA_1_5, 2.0), MAX_WIND_KMH)
            pressure = (MEAN_PRESSURE_HPA + system.pressure_anomaly + system.pressure_trend * progress
                        + pressure_offset + rng.gauss(0.0, 0.4))
            relayed = rng.random() < RELAY_SHARE
            source = (rng.choices(TRAP_VOCAB.rereporting_sources, weights=RELAY_WEIGHTS, k=1)[0] if relayed
                      else station.id)
            records.append(Record(id=_fresh(rng, taken), time=when, lat=station.lat, lon=station.lon,
                                  group=station.region, source=source,
                                  values={"temp": _stored(temp), "wind": whole_knots(wind),
                                          "pressure": altimeter(pressure)},
                                  place=station.name))
        return sorted(records, key=lambda r: (r.time, r.id))

    def planted_value(self, spec: FieldSpec, value: float, like: float, peers: Sequence[float] = ()) -> float:
        """A planted value as an observation reports it (Domain.planted_value): a wind in whole knots and a
        pressure in hundredths of an inch of mercury when the observation it stands for gives them so (every
        synthetic one does), and a temperature to the places of that observation (a five-minute observation
        gives whole degrees, a routine one tenths), whatever the station's other observations give."""
        if spec.key == WIND.key and like == whole_knots(like):
            return whole_knots(value)
        if spec.key == PRESSURE.key and like == altimeter(like):
            return altimeter(value)
        return super().planted_value(spec, value, like)

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID, added to `taken`: every feed's IDs look alike, so `source` changes nothing."""
        return _fresh(rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        """A fresh ID, shaped as every observation ID is (`example` must be one), added to `taken`."""
        if _ID.fullmatch(example) is None:
            raise ValueError(f"{example!r} is not a weather observation ID")
        return _fresh(rng, taken)

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
    """The clock, read only when the live source is fetched (never when a bundle is parsed). The fetch time
    is the bundle's fetched_at, which is the snapshot's compiled_at: a live weather snapshot's window ends
    when it was fetched (and its replay keeps that time, which the frozen bundle holds)."""
    return datetime.now(timezone.utc)


_clock = time.monotonic  # what a fetch measures its time budget by


def _never_connected(exc: requests.RequestException) -> bool:
    """Whether a request failed before it reached the server: it timed out connecting, or no connection could
    be made at all (no network, a name that does not resolve, a proxy that refuses it), as against a
    connection that was made and then dropped (a ConnectionError of a ProtocolError: "Connection aborted.")
    or an answer that was too slow."""
    if isinstance(exc, requests.ConnectTimeout):
        return True
    if not isinstance(exc, requests.ConnectionError):
        return False
    reason = exc.args[0] if exc.args else None
    return not isinstance(reason, (ProtocolError, OSError))


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
    timeout: float = 20.0  # seconds to connect to a station's endpoint, and for each read of its answer
    deadline: float = 30.0  # seconds for one station's whole answer
    budget: float = 120.0  # seconds for the whole fetch: once they are spent, no other station is asked
    source: str = "nws"
    raw_filename: str = "raw_feed.json"
    user_agent: str = USER_AGENT

    def fetch(self) -> tuple[bytes, str]:
        """Every station's observations as one bundle: (its bytes, the URL asked for each station).

        Each station is asked once, as every live source is (sources.http_get), within `deadline` seconds,
        and the whole fetch within `budget` seconds: a station still to be asked once they are spent is
        recorded as not asked, so that an API that holds every connection open costs the budget, not every
        station's deadline. Each response is kept as served, its HTTP status and the exact text of its body
        (UTF-8, as JSON is), whatever the status; parse decides what can be read. A request that fails on
        its own (a timeout, a connection dropped, a body that is not UTF-8) is recorded as an error for that
        station, and the other stations are still asked. Only a first request that cannot connect at all (no
        network, or no route to the API: the others would not connect either; _never_connected) raises at
        once. A fetch in which no station answered with a body that parse can read as JSON (a 2xx status)
        raises ValueError; answers that hold too few usable observations fall back later, through
        min_records (sources.build_domain_snapshot)."""
        fetched_at = _utc_now().replace(microsecond=0)
        params = {"start": _iso(fetched_at - timedelta(hours=self.window_hours)), "limit": self.limit}
        headers = {"User-Agent": self.user_agent, "Accept": "application/geo+json"}
        responses: dict[str, Any] = {}
        started = _clock()
        for index, station in enumerate(self.stations):
            left = self.budget - (_clock() - started)
            if left <= 0:
                responses[station] = {"error": f"not asked: the fetch had spent its {self.budget:g} s"}
                continue
            try:
                response = sources.http_get(OBSERVATIONS_URL.format(station=station), params=params, headers=headers,
                                            timeout=self.timeout, deadline=min(self.deadline, left))
            except requests.RequestException as exc:  # a timeout, or a connection dropped or never made
                if index == 0 and _never_connected(exc):
                    raise
                responses[station] = {"error": failure_text(exc)}
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
        a value the NWS rejected or questioned (or a quality code that is not a code), is dropped; a
        station whose body holds something of another shape than the API's is left out, and the other
        stations are kept; of each station's observations in one UTC hour, the latest hourly METAR is
        kept, or else the latest observation. Raises ValueError (or KeyError) on bytes that are not a
        bundle, JSON nested too deeply to read included, and on a bundle fetched outside the years 1800
        to 2200."""
        bundle = read_feed_json(raw, "the bundle of NWS observations")
        if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"not a bundle of NWS observations ({BUNDLE_FORMAT})")
        compiled_at = parse_utc_iso(str(bundle["fetched_at"]))
        if not EARLIEST_TIME <= compiled_at <= LATEST_TIME:
            raise ValueError(f"the bundle says it was fetched at {str(bundle['fetched_at'])[:40]!r}, which no bundle was")
        window, stations = bundle.get("window_hours", self.window_hours), bundle["stations"]
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 24 * 7:
            raise ValueError(f"the bundle covers {window!r} hours, not a whole number of hours up to a week")
        if not isinstance(stations, dict):
            raise ValueError("the bundle gives no responses by station")
        start = compiled_at - window * 3600
        readings = []
        for station_id, response in stations.items():
            station = STATION_BY_ID.get(station_id)
            if station is not None:
                readings += _station_readings(response, station, start, compiled_at)
        return _identified(_hourly(readings)), compiled_at, window


def _station_readings(response: Any, station: Station, start: int, end: int) -> list[_Reading]:
    """The usable observations of one station's frozen response, timed in [start, end): none at all when
    its body holds something of another shape than the API's, which leaves the other stations as read."""
    try:
        readings = [_reading(feature, station) for feature in _features(_payload(response))]
    except (TypeError, ValueError, KeyError, AttributeError, ArithmeticError):
        return []
    return [r for r in readings if r is not None and start <= r.time < end]


def _payload(response: Any) -> Any:
    """What one station's frozen response says: its body read as JSON, when the API answered with
    success (a 2xx status); None for an error status, a failed request or a body that is not JSON
    (or JSON nested too deeply to read)."""
    if not isinstance(response, dict) or not isinstance(response.get("body"), str):
        return None
    status = response.get("status")
    if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
        return None
    try:
        return json.loads(response["body"])
    except (ValueError, RecursionError):
        return None


def _failure(response: Any) -> str:
    """Why a station's response holds no observations, in a few words."""
    if isinstance(response, dict) and "error" in response:
        return str(response["error"])
    status = response.get("status") if isinstance(response, dict) else None
    return f"HTTP {status}" if isinstance(status, int) and not 200 <= status < 300 else "a body that is not JSON"


def _features(payload: Any) -> list[Mapping[str, Any]]:
    """The observations of one station's response: a feature collection (the observations endpoint)
    or a single feature (the latest-observation endpoint); none for anything else, a collection whose
    features are not a list included."""
    if not isinstance(payload, dict):
        return []
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        return [f for f in features if isinstance(f, dict)] if isinstance(features, list) else []
    return [payload] if payload.get("type") == "Feature" else []


def _reading(feature: Mapping[str, Any], station: Station) -> _Reading | None:
    """One observation of `station` as the API gives it, or None when it cannot be used as it is."""
    props = feature.get("properties")
    if not isinstance(props, dict) or props.get("stationId", station.id) != station.id:
        return None
    try:
        when = parse_utc_iso(str(props["timestamp"]))
    except (KeyError, ValueError, OverflowError):
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
    canonical unit, or None when it has no value, a unit not in `units`, a rejected value, or a quality
    code that is not a code (text), which cannot say the value was not rejected."""
    if not isinstance(quantity, dict):
        return None
    value, code = quantity.get("value"), str(quantity.get("unitCode") or "")
    convert = units.get(code.rsplit(":", 1)[-1])
    quality = quantity.get("qualityControl")
    if (convert is None or (quality is not None and (not isinstance(quality, str) or quality in REJECTED_QC))
            or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
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
    """The readings as Records in time order, each with an opaque ID that depends only on its station and
    time (a hash, so replaying a bundle gives the same IDs, and a collision takes the next free serial)."""
    taken: set[str] = set()
    records = []
    for reading in sorted(readings, key=lambda r: (r.time, r.station.id)):
        station = reading.station
        digest = hashlib.sha256(f"{station.id}|{reading.time}".encode("utf-8")).digest()
        low, high = SERIALS
        serial = int.from_bytes(digest[:8], "big") % (high - low)
        while f"{ID_PREFIX}{low + serial}" in taken:
            serial = (serial + 1) % (high - low)
        ref = f"{ID_PREFIX}{low + serial}"
        taken.add(ref)
        records.append(Record(id=ref, time=reading.time, lat=reading.lat, lon=reading.lon, group=station.region,
                              source=station.id, values=dict(reading.values), place=station.name))
    return records


WEATHER = Weather()
DOMAIN = WEATHER  # what the registry looks up (see evalgen.domains)
