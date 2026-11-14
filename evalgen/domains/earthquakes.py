"""Earthquakes: the 0.2.0 domain, described through the domain interface.

Everything here reproduces 0.2.0 exactly. The field specs carry the 0.2.0 threshold draws and guard
margins, the op names are the 0.2.0 operation names (``ops.ALIASES``), the lexicon holds every 0.2.0
question wording, and the trap vocabulary is the one the 0.2.0 digests print. Legacy prompts print the
0.2.0 reading rules (``rules.READING_RULES``) alone; a prompt off the legacy path, in any genre, prints
them as every other domain's prompts head theirs ("Reading rules:", since the rules of a run's domains
differ), with rules 1, 3 and 4 quoting every note and notice phrase the digests print, rule 9 naming the
greetings and signatures that emails and chats print, followed by the two genre-neutral rules that every
genre's quotes, read-backs, forwards and message times need (``GENRE_READING_RULES``, ``rules_text``).
Its questions there drop the one 0.2.0 wording that reads as a range beside a depth ("with a depth of
16 km or more and no more than 40 km from ..."):
``GENRE_LEXICON``, ``prompt_lexicon``. Records stay the 0.2.0 types: ``models.Event`` (the snapshot)
and ``models.Report`` (a digest line); genres other than classic write and read Reports and magnitude
corrections in the generic form of every domain (``to_generic``, ``from_generic``).

Magnitude plays the *primary* role of the role-based templates (thresholds, rankings, the largest
value and its time) and depth the *secondary* one (thresholds, the deepest and the shallowest value).

A legacy run fetches, freezes and replays the USGS feed through ``sources.build_snapshot``. Off the
legacy path the day feed is this domain's live source (``UsgsFeed``), used by the generic snapshot.
"""
from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Sequence

from ..digest import CHATTER
from ..forecast import forecast_text, make_forecast, resolve_forecasts
from ..geo import destination, haversine_km
from ..models import EVENT_ID_PATTERN, Event, Notice, Report
from ..ops import ALIASES
from ..rules import READING_RULES, Mishandle, resolve
from ..sources import (COORD_DECIMALS, MIN_POOL_EVENTS, USER_AGENT, fetch_usgs, generate_synthetic, mimic_event_id,
                       new_event_id, parse_usgs_document, synthetic_mag_type, usgs_document, usgs_window)
from ..units import LENGTH
from .base import (DISTANCE_WORDINGS, KEYED_NOTE, NOTICE_RULE, REPEAT_RULE, REVISED_FROM, REVISED_TO, TABLE_HEADER_RULE,
                   Domain, FieldSpec, Forecaster, Lexicon, Record, RecordReport, Spatial, TrapVocab, _alternatives,
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
    # The glossary names the qualifier in words too: the 0.2.0 digest prints it as "(md)", "MD 3.2" or
    # "type=md", never under its key, which the other genres print ("mag_type=md").
    qualifier="mag_type", qualifier_label="magnitude type", correctable=True,
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

# A question's point is a seismic station. The 0.2.0 label names the network of the quake the point is
# drawn near, which is almost always the network of the answer (of the nearest quake); legacy runs keep
# it, and every other run names the station by its number alone.
SPATIAL = Spatial(
    radii=RADII, margin=(DISTANCE_MARGIN_KM, DISTANCE_MARGIN_REL),
    nearest_gap=(NEAREST_GAP_KM, NEAREST_GAP_REL), nearest_max_km=NEAREST_MAX_KM,
    point_offset_km=(5.0, 60.0), point_label="station {number}", legacy_point_label="station {source}-{number}",
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
    legacy=True,  # "1 miles", and the regions of a comparison unquoted, as 0.2.0 words them
)


# The 0.2.0 wordings that say an earthquake was "recorded" at its time, where the rules say when it
# "occurred" (rule 6), and that a reader could take for the time a message was sent; off the legacy path
# each is worded as the rules word it, as many wordings for as many, so the draws stay the same.
OCCURRED_FILTERS = {"time_between": ("that occurred between {start} and {end}", "that occurred from {start} up to {end}")}
OCCURRED_QUESTIONS = {"time_of_largest": ("At what time (UTC, HH:MM) did the largest of the {records} occur?",
                                          "When, in UTC (HH:MM), did the strongest of the {records} occur?")}


def genre_lexicon(lexicon: Lexicon) -> Lexicon:
    """The wording of earthquake questions off the legacy path: `lexicon`, except that a radius is never
    worded "no more than {value} from {place}", which beside a depth threshold reads as a range of depths
    ("with a depth of 16 km or more and no more than 40 km from station US-28"). A lexicon that still
    words radii as 0.2.0 did takes DISTANCE_WORDINGS instead, which name the distance before the number
    (as many wordings, so the draws stay the same), and one that still says an earthquake was "recorded"
    at a time takes the wordings of OCCURRED_FILTERS and OCCURRED_QUESTIONS; any other wording of its own
    is kept. A quantity of 1 takes the singular ("1 mile") and the regions of a comparison are quoted
    (Lexicon.legacy). Legacy prompts keep the 0.2.0 wording."""
    filters, questions = dict(lexicon.filters), dict(lexicon.questions)
    if filters.get("within_distance") == LEXICON.filters["within_distance"]:
        filters["within_distance"] = DISTANCE_WORDINGS
    for table, words, new in ((filters, LEXICON.filters, OCCURRED_FILTERS), (questions, LEXICON.questions,
                                                                          OCCURRED_QUESTIONS)):
        for op, wordings in new.items():
            if table.get(op) == words[op]:
                table[op] = wordings
    return replace(lexicon, filters=filters, questions=questions, legacy=False)


GENRE_LEXICON = genre_lexicon(LEXICON)

# The wording of the 0.2.0 traps. traps.py plants with the sources and reasons; digest.py prints the
# notes and notices (solver.py parses exactly these phrases), and every other genre prints them too. The
# 0.2.0 rules (rules.READING_RULES) quote none of them: they name a note by what it does ("a note naming the
# event ID it duplicates") and a notice by its label (CORRECTION, RETRACTION) or by what it says ("a value
# was revised", "retracted or withdrawn"). GENRE_READING_RULES, which every prompt off the legacy path prints,
# quotes every one of them, word for word, as the record rules of every other domain quote theirs.
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
#
# Some parts of the 0.2.0 text change off the legacy path. Its heading says the rules are "identical for
# every digest", which a run of several domains contradicts (each prints its own), so the rules are
# headed as every domain's are (record_rules). Rules 1, 3 and 4 quote, word for word, every note and
# notice phrase that the digests print (TRAP_VOCAB, the same_as= field of a line of key=value fields, and
# the "revised to" and "revised from" of every genre's corrections), as the record rules of every other
# domain do: the 0.2.0 rules name them by what they do or by their labels only, so a log line's
# "same_as=<ID>" was the one link from a re-report to its earthquake that no rule named. And rule 1 no
# longer says which network files a preliminary report or a re-report: off the legacy path the trap builders
# file each under the network of the event it stands in for, which may be the earthquake's own (REWORDED_RULES;
# the other 0.2.0 rules keep their lines). Rule 9 lists the lines without data that the 0.2.0 digest prints;
# the other genres also print section titles (subheads, captions), greetings and signatures, which the rule
# now names as the record rules do, and a table's header row, which gives the units of its columns, is not
# one of them (TABLE_HEADER_RULE). It opens with what a notice is (NOTICE_RULE, as the record rules' rule 3
# does), which the 0.2.0 rules leave to the notices' keywords: in an email thread, a radio net, a form or a
# table a notice looks like any other message, and it may be printed before the report it names. Rule 10
# also says that a WIRE line's bracketed time is its report's own (message_time_rule).
LEGACY_RULES_HEADING = "Reading rules (identical for every digest):\n"
RULES_HEADING = "Reading rules:\n"
LEGACY_RULE_9 = ("9. Lines that are neither reports nor notices (headers, station status, editor's notes,\n"
                 "   reminders) carry no data and change nothing.\n")
OTHER_LINES_RULE = (f"{NOTICE_RULE} Lines that are neither reports nor notices (headers, section titles, station "
                    f"status, editor's notes, reminders, greetings, signatures) carry no data and change nothing. "
                    f"{TABLE_HEADER_RULE}")
# The 0.2.0 rules that change off the legacy path, by number: each as the 0.2.0 text words it (its lines
# joined), then as every prompt off the legacy path prints it.
REWORDED_RULES = {
    1: ("Every earthquake has exactly one event ID: the ID printed on its REVIEWED report. An earthquake may appear "
        "in several reports: a PRELIMINARY report from its own network, or a re-report from another network, either "
        "under the same event ID or under its own reference with a note naming the event ID it duplicates. Reports "
        "that share an event ID, or name it in such a note, are the same earthquake. Count each earthquake once and "
        "refer to it by its event ID.",
        "Every earthquake has exactly one event ID: the ID printed on its REVIEWED report. An earthquake may appear "
        "in several reports: a PRELIMINARY report, or a re-report, either under the same event ID or under its own "
        "reference with a note naming the event ID it duplicates. Such a note reads "
        f"{_alternatives(TRAP_VOCAB.duplicate_notes)} {KEYED_NOTE}, followed by that event ID. Reports that share an "
        "event ID, or name it in such a note, are the same earthquake. Count each earthquake once and refer to it by "
        "its event ID."),
    3: ("A correction (a CORRECTION notice, or any notice saying that a value was revised) replaces the stated value "
        "for the named event, even on a reviewed report.",
        f'A correction (a CORRECTION notice, or any notice saying that a value was revised: "{REVISED_TO}" a new '
        f'value, "{REVISED_FROM}" an old value to a new one, or "magnitude now" and a new value) replaces the stated '
        "value for the named event, even on a reviewed report."),
    4: ("A retraction (a RETRACTION notice, or any notice saying that a report was retracted or withdrawn) means the "
        "named report was not a real earthquake (for example a quarry blast or a false trigger): ignore that report "
        "completely.",
        'A retraction (a RETRACTION notice, or any notice saying that a report was retracted, "RETRACTED" or '
        '"withdrawn", or that it "was not an earthquake") means the named report was not a real earthquake (for '
        "example a quarry blast or a false trigger): ignore that report completely."),
}


def _genre_reading_rules() -> str:
    """GENRE_READING_RULES, from the 0.2.0 text: refuses to build if that text ever changes."""
    if not READING_RULES.startswith(LEGACY_RULES_HEADING) or not READING_RULES.endswith(LEGACY_RULE_9):
        raise ValueError("the 0.2.0 reading rules no longer open and end as GENRE_READING_RULES expects")
    rules_1_to_8 = [rule for rule in re.split(r"^(?=\d+\. )", READING_RULES[len(LEGACY_RULES_HEADING):
                                                                           -len(LEGACY_RULE_9)], flags=re.MULTILINE)
                    if rule]
    for number, (legacy, text) in REWORDED_RULES.items():
        if " ".join(rules_1_to_8[number - 1].split()) != f"{number}. {legacy}":
            raise ValueError(f"the 0.2.0 rule {number} no longer reads as GENRE_READING_RULES expects")
        rules_1_to_8[number - 1] = numbered_rules((text,), start=number)
    last = len(re.findall(r"^\d+\. ", READING_RULES, flags=re.MULTILINE))
    return RULES_HEADING + "".join(rules_1_to_8) + numbered_rules(
        (OTHER_LINES_RULE, message_time_rule("an earthquake"), REPEAT_RULE), start=last)


GENRE_READING_RULES = _genre_reading_rules()

# USGS-mode instances ask whether an M4.5+ quake will strike near a point in the next 24 hours; the
# USGS catalogue answers it later (see forecast.py).
FORECASTER = Forecaster(source="usgs", make=make_forecast, text=forecast_text, resolve=resolve_forecasts)

# The synthetic world off the legacy path (synthetic_world). Aftershocks: a share of the events follow a
# larger one of their network within seconds to hours, a few km away. Relays: the networks that re-report
# other networks' events file events of their own, as the USGS feed shows them doing (the tsunami warning
# centres in Alaska and the Pacific, the NEIC for larger events anywhere): (network, regions, least
# magnitude, share of those events it files).
AFTERSHOCK_SHARE = 0.15
RELAY_FILINGS = (
    ("at", ("Southern Alaska",), 0.0, 0.2),
    ("pt", ("Hawaii", "Tonga"), 0.0, 0.2),
    ("us", ("Northern California", "Southern California", "Southern Alaska", "Hawaii", "Puerto Rico", "Nevada",
            "Utah"), 3.0, 0.3),
)
COLLAPSED_S, COLLAPSED_KM = 3, 15.0  # solutions this close are one quake (sources.collapse_duplicate_solutions)


def synthetic_world(rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Event]:
    """`n` synthetic events of the 0.2.0 world (sources.generate_synthetic) as runs off the legacy path
    draw them: a share of them filed by a relay network instead (RELAY_FILINGS, with an ID and a magnitude
    type of that network's), and AFTERSHOCK_SHARE of them aftershocks of the others: 8 seconds to hours
    after a mainshock drawn by the 0.2.0 weights (1.5 ** magnitude), a few km from it, at about its depth,
    0.5 to 2 magnitudes smaller, from its network. No two events are one quake's solutions (within 3 s
    and 15 km of each other), as no two of a real feed's are."""
    n_after = round(n * AFTERSHOCK_SHARE) if n >= 4 else 0
    events = []
    first = generate_synthetic(rng, n - n_after, compiled_at, window_hours)
    taken = {e.id for e in first}  # a new ID never takes one that any event had
    for e in first:
        for network, regions, least, share in RELAY_FILINGS:
            if e.region in regions and e.mag >= least and rng.random() < share:
                e = replace(e, id=new_event_id(network, rng, taken), network=network,
                            mag_type=synthetic_mag_type(rng, network, e.mag))
                break
        events.append(e)
    mains = [e for e in events if e.mag >= 2.0] or events
    weights = [1.5 ** e.mag for e in mains]
    for _ in range(50 * n_after):
        if len(events) >= n:
            break
        main = rng.choices(mains, weights=weights, k=1)[0]
        when = main.time + 8 + int(15 * rng.paretovariate(0.7))  # often within a minute, now and then hours later
        lat, lon = destination(main.lat, main.lon, abs(rng.gauss(0.0, 3.0)), rng.uniform(0.0, 360.0))
        lat, lon = round(lat, COORD_DECIMALS), round(lon, COORD_DECIMALS)
        mag = round(main.mag - rng.uniform(0.5, 2.0), 1)
        if (when >= compiled_at or mag < (4.0 if main.network == "us" else 0.8)
                or any(abs(e.time - when) <= COLLAPSED_S and haversine_km(e.lat, e.lon, lat, lon) <= COLLAPSED_KM
                       for e in events)):
            continue
        events.append(Event(id=new_event_id(main.network, rng, taken), time=when, lat=lat, lon=lon,
                            depth_km=round(max(0.0, main.depth_km + rng.gauss(0.0, 1.5)), 1), mag=mag,
                            mag_type=synthetic_mag_type(rng, main.network, mag), region=main.region,
                            network=main.network))
    return sorted(events, key=lambda e: (e.time, e.id))


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
        """One Event per earthquake (sources.parse_usgs_geojson: duplicate solutions collapsed, and no
        earthquake that no real feed lists), the feed's generation time, and its window: the period that
        its metadata.url names, an hour, a day, a week or a month of 30 days, and a day when it names none
        (sources.usgs_window). An earthquake timed outside that window (before it, or after the feed was
        generated), whose digests say they cover it, is left out, and so is a second earthquake under an
        event ID already read (the earlier is kept): a real feed lists neither."""
        data = usgs_document(raw)
        events, generated, _ = parse_usgs_document(data)
        window = usgs_window(data)
        start, kept, seen = generated - window * 3600, [], set()
        for e in events:  # in time order
            if start <= e.time <= generated and e.id not in seen:
                seen.add(e.id)
                kept.append(e)
        return kept, generated, window


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

    # An aftershock is its mainshock's near twin: off the legacy path a digest often lists the event of its
    # region nearest in time to one it lists, within a quarter of an hour (pipeline.sample_with_slots).
    twin_seconds = 15 * 60
    twin_share = 0.25

    def sample_weight(self, record: Event) -> float:
        """Larger quakes are likelier to be in a digest (the 0.2.0 weights: 1.5 ** magnitude)."""
        return 1.5 ** record.mag

    def twin_key(self, record: Event | Record) -> str:
        """An earthquake's twins are the events of its region (the nearest in time: its aftershock): its
        group, which an Event and the generic Record the trap builders read it as both give."""
        return record.group

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Event]:
        """`n` synthetic events off the legacy path: the 0.2.0 world (sources.generate_synthetic), with
        aftershocks, and with a share of the events filed by the networks that re-report others' events.
        A legacy run draws its snapshot from sources.generate_synthetic itself, through build_snapshot."""
        return synthetic_world(rng, n, compiled_at, window_hours)

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        return new_event_id(source, rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        return mimic_event_id(example, rng, taken)

    def rules_text(self) -> str:
        """The reading rules of an earthquake prompt off the legacy path (GENRE_READING_RULES): the 0.2.0
        rules, headed as every domain's are, with rules 1, 3 and 4 quoting every note and notice phrase and
        rule 9 saying what a notice is and naming every kind of line without data, and the genre-neutral rules
        on message times and repeats. A legacy prompt prints the 0.2.0 rules (rules.READING_RULES) alone."""
        return GENRE_READING_RULES

    def prompt_lexicon(self) -> Lexicon:
        """The wording of questions off the legacy path (genre_lexicon; GENRE_LEXICON for this domain's
        own words): its lexicon, as bound to an instance's words, less the 0.2.0 wording of a radius that
        reads as a range of depths. Legacy prompts word them with the 0.2.0 lexicon (``lexicon``)."""
        return genre_lexicon(self.lexicon)

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
