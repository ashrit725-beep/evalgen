"""Earthquakes: the 0.2.0 domain, described through the domain interface.

Everything here reproduces 0.2.0 exactly. The field specs carry the 0.2.0 threshold draws and guard
margins, the op names are the 0.2.0 operation names (``ops.ALIASES``), the lexicon holds every 0.2.0
question wording, the trap vocabulary is the one the 0.2.0 digests print, and the reading rules are
``rules.READING_RULES``. Records stay the 0.2.0 types: ``models.Event`` (the snapshot) and
``models.Report`` (a digest line).

Magnitude plays the *primary* role of the role-based templates (thresholds, rankings, the largest
value and its time) and depth the *secondary* one (thresholds, the deepest and the shallowest value).
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from ..geo import EARTH_RADIUS_KM
from ..models import EVENT_ID_PATTERN, Event, Notice, Report
from ..ops import ALIASES
from ..rules import READING_RULES, Mishandle, resolve
from ..sources import generate_synthetic, mimic_event_id, new_event_id
from .base import LENGTH, Domain, FieldSpec, Lexicon, Spatial, TrapVocab, Unit

# Margin guards: no event may sit this close to a threshold, so that display rounding (0.1 km or mi,
# 0.001 degrees, minute-precision times) can never change a correct answer. Magnitudes need no
# margin: digests print them exactly as stored.
DEPTH_MARGIN_KM, DEPTH_MARGIN_REL = 0.6, 0.03
DISTANCE_MARGIN_KM, DISTANCE_MARGIN_REL = 3.0, 0.06
MAG_GAP = 0.1  # ranked answers: every gap down to the asked rank must be at least 0.1
NEAREST_GAP_KM, NEAREST_GAP_REL = 3.0, 0.08  # the nearest quake must beat the runner-up clearly
NEAREST_MAX_KM = 500.0  # keeps "nearest" questions local, so the 3% distance tolerance stays tight
RADII = {
    "km": (25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300),
    "mi": (15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200),
}

MAGNITUDE = FieldSpec(
    key="mag", label="magnitude", canonical_unit=None, units=(),
    decimals=2,  # stored as the feed gives it (USGS: up to 2 decimals); answers are stored values
    answer_type="magnitude", tolerance=(0.05, 0.0), margin=(0.0, 0.0), rank_gap=MAG_GAP,
    quantiles=(0.25, 0.85), threshold_step=0.1,
    filters=("at_least", "below"), terminals=("argmax", "rank", "max", "time_of_max"),
    qualifier="mag_type", correctable=True,
)
DEPTH = FieldSpec(
    key="depth", label="depth", canonical_unit="km", units=LENGTH, decimals=1,
    answer_type="depth", tolerance=(0.3, 0.02), margin=(DEPTH_MARGIN_KM, DEPTH_MARGIN_REL),
    rank_gap=0.0,  # no ranked terminal reads depth
    quantiles=(0.25, 0.8), threshold_step=0.1, coarse_step=1.0, jitter=1.5, positive_threshold=True,
    filters=("below", "at_least"),  # shallower-than first: the 0.2.0 order of ops.FILTER_OPS
    terminals=("max", "min"), signed=True,
)

SPATIAL = Spatial(
    radii=RADII, margin=(DISTANCE_MARGIN_KM, DISTANCE_MARGIN_REL),
    nearest_gap=(NEAREST_GAP_KM, NEAREST_GAP_REL), nearest_max_km=NEAREST_MAX_KM,
    point_offset_km=(5.0, 60.0), point_label="station {source}-{number}",
)

LEXICON = Lexicon(
    records="earthquakes",
    all_records="earthquakes in the digest",
    filters={
        "mag_at_least": ("of magnitude {value} or greater", "with magnitude at least {value}",
                         "of magnitude {value} or more"),
        "mag_below": ("with magnitude below {value}", "smaller than magnitude {value}"),
        "depth_shallower_than": ("shallower than {value}", "with a depth of less than {value}"),
        "depth_at_least": ("at least {value} deep", "with a depth of {value} or more"),
        "within_distance": ("within {value} of {place}", "no more than {value} from {place}"),
        "time_between": ("that occurred between {start} and {end}", "that were recorded from {start} up to {end}"),
        "in_region": ("in {group}", "located in {group}", "whose region is listed as {group}"),
    },
    questions={
        "count": ("How many distinct {records} are there?", "Count the distinct {records}.",
                  "What is the number of separate {records}?"),
        "argmax_mag": ("Among the {records}, which one had the largest magnitude? Give its event ID.",
                       "What is the event ID of the strongest of the {records}?"),
        "rank_mag": ("Among the {records}, which one has the {nth}-largest magnitude? Give its event ID.",
                     "Rank the {records} by magnitude, largest first: which event is {nth}? Give its event ID."),
        "max_mag": ("What was the largest magnitude among the {records}?",
                    "Among the {records}, what was the highest magnitude recorded?"),
        "max_depth": ("What was the greatest depth, in {unit}, among the {records}? Round to 1 decimal place.",
                      "Among the {records}, how deep was the deepest one, in {unit}? Round to 1 decimal place."),
        "min_depth": ("What was the smallest depth, in {unit}, among the {records}? Round to 1 decimal place.",
                      "Among the {records}, how deep was the shallowest one, in {unit}? Round to 1 decimal place."),
        "nearest_to": ("Which of the {records} was closest to {place}? Give its event ID.",
                       "Among the {records}, give the event ID of the one nearest to {place}."),
        "nearest_distance": ("How far from {place}, in {unit}, was the closest of the {records}? "
                             "Round to 1 decimal place.",
                             "Among the {records}, what is the distance in {unit} from {place} to the nearest one? "
                             "Round to 1 decimal place."),
        "compare_regions": ("Counting only {records}, which region has more of them: {a} or {b}? "
                            "Answer \"tie\" if the counts are equal.",
                            "Which region had more {records}: {a} or {b}? Say \"tie\" if they are equal."),
        "time_of_largest": ("At what time (UTC, HH:MM) did the largest of the {records} occur?",
                            "When, in UTC (HH:MM), was the strongest of the {records} recorded?"),
    },
)

# The wording of the 0.2.0 traps. traps.py plants with the sources and reasons; digest.py prints the
# notes and notices (solver.py parses exactly these phrases), and rules.READING_RULES names them.
TRAP_VOCAB = TrapVocab(
    rereporting_sources=("us", "at", "pt", "ew"),
    duplicate_notes=("same event as", "same earthquake as", "Dup of"),
    retraction_reasons=("quarry blast", "mine blast", "false trigger from instrument noise", "sonic boom",
                        "automatic solution with no real event"),
    retraction_notices=("RETRACTED", "was not an earthquake", "withdrawn"),
    correction_notices=("revised to", "revised from", "magnitude now"),
)


def _units_text(units: Sequence[Unit], canonical: str) -> str:
    """'km or mi (1 mi = 1.609344 km)'."""
    names = " or ".join(u.symbol for u in units)
    conversions = "; ".join(f"1 {u.symbol} = {u.scale!r} {canonical}" for u in units if u.symbol != canonical)
    return f"{names} ({conversions})" if conversions else names


class Earthquakes(Domain):
    name = "earthquakes"
    noun = "earthquake"
    plural = "earthquakes"
    id_label = "event ID"
    source_label = "network"
    group_label = "region"

    fields = (MAGNITUDE, DEPTH)
    roles = {"primary": MAGNITUDE.key, "secondary": DEPTH.key}
    spatial = SPATIAL
    lexicon = LEXICON
    op_names = {(alias.kind, alias.field): name for name, alias in ALIASES.items()}
    # The USGS feed is still fetched, frozen and replayed by sources.build_snapshot; it becomes this
    # domain's LiveAdapter with the generic snapshot (plan phase 1b).
    live = None

    id_pattern = EVENT_ID_PATTERN
    id_answer_type = "event_id"
    group_answer_type = "region"
    trap_vocab = TRAP_VOCAB

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Event]:
        return generate_synthetic(rng, n, compiled_at, window_hours)

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        return new_event_id(source, rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        return mimic_event_id(example, rng, taken)

    def rules_text(self) -> str:
        return READING_RULES

    def glossary(self, surface: Mapping[str, Any] | None = None) -> str:
        """The data dictionary for non-legacy prompts. Legacy (0.2.0) prompts print only the rules."""
        return "\n".join([
            "Glossary (earthquake reports):",
            f"- {self.id_label}: identifies one earthquake; it is the ID printed on the earthquake's REVIEWED report.",
            f"- {self.source_label}: the seismic network that filed a report.",
            f"- {self.group_label}: where the earthquake struck, as printed on its report.",
            f"- {MAGNITUDE.label}: a plain number; the scale printed beside it (md, ml, mb, mw, mww, ...) "
            "does not change how magnitudes compare.",
            f"- {DEPTH.label}: in {_units_text(DEPTH.units, 'km')} below sea level; "
            "a negative depth is above sea level.",
            f"- distance: great-circle, in {_units_text(LENGTH, 'km')}, "
            f"on a sphere of radius {EARTH_RADIUS_KM:g} km.",
        ]) + "\n"

    def resolve(self, reports: Sequence[Report], notices: Sequence[Notice],
                mishandle: Mishandle | None = None) -> list[Event]:
        return resolve(reports, notices, mishandle)


EARTHQUAKES = Earthquakes()
DOMAIN = EARTHQUAKES  # what the registry looks up (see evalgen.domains)
