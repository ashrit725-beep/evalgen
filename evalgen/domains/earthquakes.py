"""Earthquakes: the 0.2.0 domain, described through the domain interface.

Everything here reproduces 0.2.0 exactly. The field specs carry the 0.2.0 threshold draws and guard
margins, the op names are the 0.2.0 operation names (``ops.ALIASES``), the lexicon holds every 0.2.0
question wording, and the trap vocabulary is the one the 0.2.0 digests print. Legacy prompts print the
0.2.0 reading rules (``rules.READING_RULES``) alone; a prompt off the legacy path, in any genre, prints
them followed by the two genre-neutral rules that every genre's quotes, read-backs, forwards and message
times need (``GENRE_READING_RULES``, ``rules_text``). Records stay the 0.2.0 types: ``models.Event``
(the snapshot) and ``models.Report`` (a digest line); genres other than classic write and read Reports
and magnitude corrections in the generic form of every domain (``to_generic``, ``from_generic``).

Magnitude plays the *primary* role of the role-based templates (thresholds, rankings, the largest
value and its time) and depth the *secondary* one (thresholds, the deepest and the shallowest value).

A legacy run fetches, freezes and replays the USGS feed through ``sources.build_snapshot``. Off the
legacy path the day feed is this domain's live source (``UsgsFeed``), used by the generic snapshot.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from typing import Sequence

from ..digest import CHATTER
from ..forecast import forecast_text, make_forecast, resolve_forecasts
from ..models import EVENT_ID_PATTERN, Event, Notice, Report
from ..ops import ALIASES
from ..rules import READING_RULES, Mishandle, resolve
from ..sources import (MIN_POOL_EVENTS, USER_AGENT, fetch_usgs, generate_synthetic, mimic_event_id, new_event_id,
                       parse_usgs_geojson)
from ..units import LENGTH
from .base import (REPEAT_RULE, Domain, FieldSpec, Forecaster, Lexicon, RecordReport, Spatial, TrapVocab,
                   message_time_rule, numbered_rules)

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
    tolerance_scales=False,  # 0.2.0 scores 0.3 km and 0.3 miles alike
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

# The reading rules of every earthquake prompt off the legacy path: the 0.2.0 rules, which say nothing of
# messages, then the genre-neutral rules on message times and on quoted, read-back, forwarded and relayed
# repeats that every record domain's rules give (domains.base.record_rules, rules 6 and 7). Chat logs,
# email threads and radio transcripts all carry repeats and message times; a legacy prompt is the 0.2.0
# classic digest, which carries neither, and prints READING_RULES alone.
GENRE_READING_RULES = READING_RULES + numbered_rules(
    (message_time_rule("an earthquake"), REPEAT_RULE),
    start=len(re.findall(r"^\d+\. ", READING_RULES, flags=re.MULTILINE)) + 1)

# USGS-mode instances ask whether an M4.5+ quake will strike near a point in the next 24 hours; the
# USGS catalogue answers it later (see forecast.py).
FORECASTER = Forecaster(source="usgs", make=make_forecast, text=forecast_text, resolve=resolve_forecasts)


class UsgsFeed:
    """The USGS day feed as this domain's live source (a LiveAdapter), for runs off the legacy path.
    A legacy run reaches the same feed through sources.build_snapshot, which can also widen the hour
    feed to the day feed."""

    source = "usgs"
    raw_filename = "raw_feed.geojson"
    user_agent = USER_AGENT
    min_records = MIN_POOL_EVENTS

    def fetch(self) -> tuple[bytes, str]:
        return fetch_usgs("day")

    def parse(self, raw: bytes) -> tuple[list[Event], int, int]:
        """One Event per earthquake (duplicate solutions collapsed), the feed's generation time, and its
        window: an hour for the hour feed (as its metadata names it), else a day."""
        events, generated, _ = parse_usgs_geojson(raw)
        url = str((json.loads(raw.decode("utf-8")).get("metadata") or {}).get("url") or "")
        return events, generated, 1 if "_hour" in url else 24


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
    live = UsgsFeed()
    forecaster = FORECASTER

    id_pattern = EVENT_ID_PATTERN
    id_answer_type = "event_id"
    group_answer_type = "region"
    trap_vocab = TRAP_VOCAB
    chatter = CHATTER

    def sample_weight(self, record: Event) -> float:
        """Larger quakes are likelier to be in a digest (the 0.2.0 weights: 1.5 ** magnitude)."""
        return 1.5 ** record.mag

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Event]:
        return generate_synthetic(rng, n, compiled_at, window_hours)

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        return new_event_id(source, rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        return mimic_event_id(example, rng, taken)

    def rules_text(self) -> str:
        """The reading rules of an earthquake prompt off the legacy path (GENRE_READING_RULES): the 0.2.0
        rules and the genre-neutral rules on message times and repeats. A legacy prompt prints the 0.2.0
        rules (rules.READING_RULES) alone."""
        return GENRE_READING_RULES

    def resolve(self, reports: Sequence[Report], notices: Sequence[Notice],
                mishandle: Mishandle | None = None) -> list[Event]:
        return resolve(reports, notices, mishandle)

    def to_generic(self, item: Report | Notice) -> RecordReport | Notice:
        """A 0.2.0 Report as a RecordReport (its magnitude type is the qualifier of its magnitude), and
        a 0.2.0 magnitude correction as a correction of the field "mag". Anything else is unchanged."""
        if isinstance(item, Report):
            return RecordReport(ref=item.ref, source=item.network, status=item.status, time=item.time, lat=item.lat,
                                lon=item.lon, group=item.region, values=item.values,
                                qualifiers={MAGNITUDE.qualifier: item.mag_type}, place=item.place,
                                same_as=item.same_as, role=item.role, trap_id=item.trap_id)
        if isinstance(item, Notice) and item.kind == "correction" and item.field is None:
            return replace(item, new_mag=None, old_mag=None, mag_type=None, field=MAGNITUDE.key,
                           new_value=item.new_mag, old_value=item.old_mag, qualifier=item.mag_type)
        return item

    def from_generic(self, item: RecordReport | Notice) -> Report | Notice:
        """A RecordReport or a correction of "mag", as read from a digest, as the 0.2.0 Report or
        magnitude correction it stands for (the inverse of to_generic). A RecordReport that is not
        exactly a magnitude with its type and a depth is no earthquake report (ValueError)."""
        if isinstance(item, RecordReport):
            if set(item.values) != {MAGNITUDE.key, DEPTH.key} or set(item.qualifiers) != {MAGNITUDE.qualifier}:
                raise ValueError(f"{item.ref} is not an earthquake report: it gives {sorted(item.values)} and "
                                 f"{sorted(item.qualifiers)}")
            return Report(ref=item.ref, network=item.source, status=item.status, time=item.time, lat=item.lat,
                          lon=item.lon, depth_km=item.values[DEPTH.key], mag=item.values[MAGNITUDE.key],
                          mag_type=item.qualifiers[MAGNITUDE.qualifier], region=item.group, place=item.place,
                          same_as=item.same_as, role=item.role, trap_id=item.trap_id)
        if isinstance(item, Notice) and item.kind == "correction" and item.field == MAGNITUDE.key:
            return replace(item, field=None, new_value=None, old_value=None, qualifier=None,
                           new_mag=item.new_value, old_mag=item.old_value, mag_type=item.qualifier)
        return item


EARTHQUAKES = Earthquakes()
DOMAIN = EARTHQUAKES  # what the registry looks up (see evalgen.domains)
