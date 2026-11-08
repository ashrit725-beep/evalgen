"""A test-only domain. It lives in tests/, never in evalgen/domains, so it is never public: tests load
it through the registry's plugin loader (evalgen.domains.load_plugin), as a held-out domain is loaded.

Readings from moored buoys. The water temperature (°C or °F, an affine unit; it can be below zero)
plays the primary role and the wave height (m or ft, units of different sizes) the secondary one.
Readings are grouped by sea area and reported by operators, and the probe that took a temperature is
its qualifier. The module defines three domains of one class, which share their answer types:

    DOMAIN   "toy": readings with a position, over a day
    FLAT     "toy-flat": readings without a position, over a week, so the spatial templates and their
             twins drop out and time windows lie on the coarser marks of a longer window
    DRESSED  "toy-dressed": FLAT with a surface drawn for every instance: the record noun, labels, unit
             words, ID prefix, and the names of sea areas and operators
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from evalgen.domains.base import Domain, FieldSpec, Record, Spatial, Surface, TrapVocab, Unit, standard_lexicon
from evalgen.geo import destination

CELSIUS = Unit("°C", "degrees Celsius")
FAHRENHEIT = Unit("°F", "degrees Fahrenheit", 5 / 9, -160 / 9, step=1.0)
METRE = Unit("m", "metres")
FOOT = Unit("ft", "feet", 0.3048, step=0.5)

TEMPERATURE = FieldSpec(
    key="temp", label="water temperature", canonical_unit="°C", units=(CELSIUS, FAHRENHEIT), decimals=1,
    answer_type="toy_temperature", tolerance=(0.3, 0.0), margin=(0.3, 0.0), rank_gap=0.4, quantiles=(0.2, 0.8),
    threshold_step=0.5, filters=("at_least", "below"), terminals=("argmax", "rank", "max", "time_of_max"),
    qualifier="probe", signed=True, correctable=True,
)
WAVE = FieldSpec(
    key="wave", label="wave height", canonical_unit="m", units=(METRE, FOOT), decimals=2,
    answer_type="toy_wave_height", tolerance=(0.05, 0.02), margin=(0.08, 0.02), rank_gap=0.1, quantiles=(0.25, 0.8),
    threshold_step=0.1, jitter=0.15, positive_threshold=True, filters=("below", "at_least"),
    terminals=("max", "min"),
)

SPATIAL = Spatial(
    radii={"km": (30, 50, 75, 100, 150, 200, 250, 300), "mi": (20, 30, 50, 75, 100, 150, 200)},
    margin=(3.0, 0.06), nearest_gap=(3.0, 0.08), nearest_max_km=500.0, point_offset_km=(5.0, 60.0),
    point_label="buoy {source}-{number}",
)


@dataclass(frozen=True)
class Area:
    name: str
    lat: float
    lon: float
    temperature: tuple[float, float]  # mean and spread, °C
    wave: float  # median wave height, m
    source: str
    weight: float


AREAS = (
    Area("North Sea", 56.5, 3.0, (10.0, 2.5), 1.8, "ukmo", 1.5),
    Area("Irish Sea", 53.7, -5.0, (11.0, 2.0), 1.2, "ukmo", 1.0),
    Area("Bay of Biscay", 45.5, -4.5, (15.0, 2.5), 2.2, "ifrm", 1.2),
    Area("Baltic Sea", 58.0, 20.0, (3.0, 3.5), 0.9, "smhi", 1.3),
    Area("Norwegian Sea", 66.0, 4.0, (5.0, 3.0), 2.6, "metn", 1.0),
    Area("Celtic Sea", 50.5, -8.0, (12.5, 2.0), 2.0, "ifrm", 0.8),
)
SOURCES = tuple(sorted({area.source for area in AREAS}))
PROBES = ("hull", "spar", "drifter")

RULES = """\
Reading rules (identical for every digest):
1. Every {noun} has exactly one {id_label}: the ID printed on its REVIEWED report. A {noun} may be
   reported again: PRELIMINARY by its own {source_label}, or relayed by another {source_label}, either
   under the same {id_label} or under its own reference with a note naming the {id_label} it repeats.
   Such reports are the same {noun}: count it once and refer to it by its {id_label}.
2. If reports of the same {noun} disagree, use the values of its REVIEWED report.
3. A correction (any notice saying that a value was revised or corrected) replaces the stated value of
   the named {noun}, even on a reviewed report.
4. A retraction (any notice saying that a report was retracted or withdrawn) means the named report
   was not a real {noun}: ignore that report completely.
5. A value may be given in any unit the glossary lists for it; convert it as the glossary says.
6. Times may be in UTC or in local time with the UTC offset stated. A report's time is when its {noun}
   was taken; a notice's time is when it was issued. A time window includes its start and excludes its
   end.
7. Lines that are neither reports nor notices carry no data and change nothing.
"""

# Surface words of the dressed variant.
NOUNS = (("reading", "readings"), ("observation", "observations"), ("log entry", "log entries"))
SOURCE_LABELS = ("operator", "agency", "station network")
GROUP_LABELS = ("sea area", "basin", "zone")
TEMPERATURE_LABELS = ("water temperature", "sea temperature", "surface temperature")
WAVE_LABELS = ("wave height", "swell height", "sea height")
PREFIXES = ("bq-", "sx-", "tz-", "wm-", "ko-")
AREA_NAMES = ("Varnholt Deep", "Kessel Reach", "Ostra Sound", "Mirren Banks", "Tolva Basin", "Drask Shelf",
              "Hemmar Sound", "Lusk Deep", "Fenwy Reach", "Ardal Banks")
SOURCE_NAMES = ("ostr", "kvel", "maro", "tusk", "bren", "dalv")


class Toy(Domain):
    noun = "reading"
    plural = "readings"
    id_label = "reading ID"
    source_label = "operator"
    group_label = "sea area"

    fields = (TEMPERATURE, WAVE)
    roles = {"primary": TEMPERATURE.key, "secondary": WAVE.key}

    id_pattern = r"[a-z]{2}-\d{5}"
    id_prefix = "by-"
    id_answer_type = "toy_reading_id"
    group_answer_type = "toy_area"
    trap_vocab = TrapVocab(
        rereporting_sources=("gts", "emsn"),
        duplicate_notes=("same reading as", "relay of"),
        retraction_reasons=("probe fault", "test transmission", "buoy adrift"),
        retraction_notices=("RETRACTED", "withdrawn"),
        correction_notices=("revised to", "corrected to"),
    )

    def __init__(self, name: str, *, located: bool, window_hours: int = 24, dressed: bool = False) -> None:
        self.name = name
        self.spatial = SPATIAL if located else None
        self.synthetic_window_hours = window_hours
        self.has_surface = dressed
        self.lexicon = standard_lexicon(self)

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Record]:
        weights = [area.weight for area in AREAS]
        taken: set[str] = set()
        records = []
        for _ in range(n):
            area = rng.choices(AREAS, weights=weights, k=1)[0]
            temp = round(rng.gauss(*area.temperature), 1) + 0.0
            wave = max(0.05, round(rng.lognormvariate(math.log(area.wave), 0.45), 2))
            lat = lon = None
            if self.has_location:
                lat, lon = destination(area.lat, area.lon, abs(rng.gauss(0.0, 120.0)), rng.uniform(0.0, 360.0))
                lat, lon = round(lat, 3), round(lon, 3)
            records.append(Record(id=self.new_id(area.source, rng, taken),
                                  time=compiled_at - rng.randrange(60, window_hours * 3600), lat=lat, lon=lon,
                                  group=area.name, source=area.source, values={"temp": temp, "wave": wave},
                                  qualifiers={"probe": rng.choice(PROBES)}))
        return sorted(records, key=lambda r: (r.time, r.id))

    def sample_weight(self, record: Record) -> float:
        """Rough seas make the digest: a reading is likelier the higher its waves."""
        return 1.0 + record.values["wave"]

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        while True:
            ref = f"{self.id_prefix}{rng.randrange(10000, 100000)}"
            if ref not in taken:
                taken.add(ref)
                return ref

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        while True:
            ref = f"{example[:-5]}{rng.randrange(10000, 100000)}"
            if ref not in taken:
                taken.add(ref)
                return ref

    def rules_text(self) -> str:
        return RULES.format(**self.words)

    def draw_surface(self, rng: random.Random) -> dict:
        noun, plural = rng.choice(NOUNS)
        return Surface(
            noun=noun, plural=plural, id_label=f"{noun} ID", source_label=rng.choice(SOURCE_LABELS),
            group_label=rng.choice(GROUP_LABELS),
            labels={"temp": rng.choice(TEMPERATURE_LABELS), "wave": rng.choice(WAVE_LABELS)},
            units={"temp": (replace(CELSIUS, words=rng.choice(("degrees Celsius", "°C"))),
                            replace(FAHRENHEIT, words=rng.choice(("degrees Fahrenheit", "°F")))),
                   "wave": (replace(METRE, words=rng.choice(("metres", "meters"))), FOOT)},
            id_prefix=rng.choice(PREFIXES),
            groups=dict(zip((area.name for area in AREAS), rng.sample(AREA_NAMES, len(AREAS)))),
            sources=dict(zip(SOURCES, rng.sample(SOURCE_NAMES, len(SOURCES)))),
        ).to_dict()


DOMAIN = Toy("toy", located=True)
FLAT = Toy("toy-flat", located=False, window_hours=168)
DRESSED = Toy("toy-dressed", located=False, window_hours=168, dressed=True)
