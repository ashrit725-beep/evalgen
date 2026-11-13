"""The earthquake domain object and role-based composition: questions, wording and traps composed
through the domain must be exactly those of 0.2.0, and the domain-generic machinery must also run on
a domain that is not earthquakes."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import Counter
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import pytest

from evalgen import adapters, scoring, solver, traps
from evalgen.adapters import Task, _perturb
from evalgen.digest import render_notice, render_report
from evalgen.domains import DEFAULT_DOMAIN, get_domain
from evalgen.domains.base import (REPEAT_RULE, Domain, FieldSpec, Glossary, Lexicon, Record, RecordReport, TrapVocab,
                                  numbered_rules, record_rules)
from evalgen.domains.base import DISTANCE_WORDINGS, standard_lexicon
from evalgen.domains.earthquakes import (AFTERSHOCK_SHARE, DOMAIN, EARTHQUAKES, GENRE_LEXICON, LEXICON,
                                         RELAY_FILINGS, genre_lexicon)
from evalgen.domains.weather import WEATHER
from evalgen.geo import KM_PER_MILE, from_km, haversine_km
from evalgen.models import EVENT_ID_PATTERN, Notice, Question, Trap
from evalgen.ops import FILTER_OPS, DigestData, execute, rows_after_filters
from evalgen.pipeline import sample_events
from evalgen.questions import (_EXCLUSIVE, _MONOTONE, FLOAT_SLACK, NEAREST_MAX_KM, PUBLIC_TEMPLATES, RANKS,
                               TIME_MARGIN_S, Context, _compatible, compose_new_templates, compose_questions,
                               draw_filter, exclusive_pairs, filter_ops, fingerprint, monotone_pairs, phrase_question,
                               public_templates, round_to_step)
from evalgen.rules import READING_RULES, Mishandle, naive_events, report_to_event, resolve
from evalgen.scoring import is_correct, register_answer_type
from evalgen.seeds import SeedLog
from evalgen.solver import solve
from evalgen.sources import build_snapshot, generate_synthetic, mimic_event_id, new_event_id, record_problems
from evalgen.traps import builders_for, inject_traps, primary_report, threshold_values
from evalgen.units import LENGTH, MI, Unit

from generic_quakes import GENERIC_OP_NAMES, GENERIC_QUAKES
from test_domain_interface import generic as generic_step


# --- composition digest (its source is also run against evalgen 0.2.0 to pin LEGACY_DIGESTS) ---
def composition_digest(seed: int, domain: Domain | None = None, instances: int = 12) -> str:
    """SHA-256 of what composition produces for the first instances of a synthetic run: the twin
    templates, every question (chain, answer type, unit, evidence), three wordings of each (the
    metamorphic ones may restate units) and the planted reports, notices and traps. Without a
    domain, every call is the legacy one; with one, the domain is passed to every call."""
    kw = {} if domain is None else {"domain": domain}
    public = PUBLIC_TEMPLATES if domain is None else public_templates(domain)
    log = SeedLog(seed)
    snapshot, _ = build_snapshot("synthetic", log, pool_size=360)
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    new = compose_new_templates(log.rng("templates", "new"), **kw)
    used: set[str] = set()
    usage: dict[str, int] = {}
    out: dict = {"twins": [t.to_dict() for t in new], "instances": []}
    for i in range(instances):
        events = sample_events(snapshot.events, log.rng("instance", i, "events"), 20, 28)
        rng = log.rng("instance", i, "questions")
        questions = compose_questions(events, public, new, rng, ctx, rng.randint(3, 5), used, usage, **kw)
        reports, notices, traps = inject_traps(events, questions, log.rng("instance", i, "traps"),
                                               compiled_at=snapshot.compiled_at,
                                               window_hours=snapshot.window_hours, **kw)
        wordings = []
        for v in range(3):
            words = random.Random(log.seed("instance", i, "render", v))
            wordings.append([phrase_question(q, words, alt_units=v > 0 and words.random() < 0.5, **kw)
                             for q in questions])
        out["instances"].append({"questions": [asdict(q) for q in questions], "wordings": wordings,
                                 "reports": [asdict(r) for r in reports], "notices": [asdict(n) for n in notices],
                                 "traps": [asdict(t) for t in traps]})
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
# --- end of composition digest ---


LEGACY_DIGESTS = {  # composition_digest(seed) run against evalgen 0.2.0, before the role-based refactor
    3: "33ebc8fc854689873d78334d42039c000a638bdcad581dd8b8a5324ecb5d9ee4",
    11: "fbb977888a8a4faae2eeeead0cd98996e3eb900e541f767f86e0ece33a52d458",
    42: "3493cac6113c4347e755e51761113dd8dabf800a2dc894d57824fce4c44bf2d5",
    99991: "796e171908d937e20c1cedbcc043b50a667457cf3bb985970d22fcc672aa992d",
}


@pytest.mark.parametrize("seed", sorted(LEGACY_DIGESTS))
def test_composition_through_the_domain_object_equals_the_020_result(seed):
    assert composition_digest(seed, get_domain("earthquakes")) == LEGACY_DIGESTS[seed]
    assert composition_digest(seed) == LEGACY_DIGESTS[seed]  # the defaults are the earthquake domain


def composed(seed: int, domain: Domain, instances: int = 8) -> dict:
    """What composition makes of the first instances of a synthetic run, in `domain`'s op names."""
    log = SeedLog(seed)
    snapshot, _ = build_snapshot("synthetic", log, pool_size=360)
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    new = compose_new_templates(log.rng("templates", "new"), domain)
    used: set[str] = set()
    usage: dict[str, int] = {}
    out: dict = {"twins": [(t.template_id, t.filters, t.terminal) for t in new], "instances": []}
    for i in range(instances):
        events = sample_events(snapshot.events, log.rng("instance", i, "events"), 20, 28)
        rng = log.rng("instance", i, "questions")
        questions = compose_questions(events, public_templates(domain), new, rng, ctx, rng.randint(3, 5), used, usage,
                                      domain=domain)
        reports, notices, planted = inject_traps(events, questions, log.rng("instance", i, "traps"),
                                                 compiled_at=snapshot.compiled_at,
                                                 window_hours=snapshot.window_hours, domain=domain)
        words = random.Random(log.seed("instance", i, "render", 1))
        out["instances"].append({
            "chains": [q.chain for q in questions],
            "questions": [(q.template_id, q.answer_type, q.unit, q.evidence) for q in questions],
            "wordings": [phrase_question(q, words, alt_units=True, domain=domain) for q in questions],
            "digest": (reports, notices), "traps": [(t.affected, t.fallen) for t in planted],
            "question_stream": rng.getstate(),
        })
    return out


def test_composition_does_not_depend_on_op_names():
    """Earthquakes under the generic op names compose, plant and word the same questions draw for
    draw: nothing in composition reads an op's name where it means its kind and field (an earthquake
    domain under any name is planted by the 0.2.0 builders)."""
    assert traps.builders_for(GENERIC_QUAKES) is traps.builders_for(EARTHQUAKES) is traps.EARTHQUAKE_BUILDERS
    for seed in (3, 42):
        named, plain = composed(seed, EARTHQUAKES), composed(seed, GENERIC_QUAKES)
        assert plain["twins"] == [(tid, tuple(GENERIC_OP_NAMES.get(f, f) for f in filters),
                                   GENERIC_OP_NAMES.get(end, end)) for tid, filters, end in named["twins"]]
        assert len(named["instances"]) == len(plain["instances"])
        for a, b in zip(named["instances"], plain["instances"]):
            assert [[generic_step(s) for s in chain] for chain in a.pop("chains")] == b.pop("chains")
            assert a == b  # templates, answer types, units, evidence, wordings, digest, traps, draws


# ------------------------------------------------------------ 0.2.0 literals and recipes
# questions.py as it was in 0.2.0, written out independently of the role-based derivations.

PUBLIC_020 = [
    ("P01", ("mag_at_least",), "count"),
    ("P02", ("within_distance",), "count"),
    ("P03", ("in_region",), "argmax_mag"),
    ("P04", ("mag_at_least",), "max_depth"),
    ("P05", ("mag_at_least",), "compare_regions"),
    ("P06", ("depth_shallower_than", "time_between"), "count"),
    ("P07", ("in_region",), "nearest_to"),
    ("P08", ("in_region",), "time_of_largest"),
    ("P09", ("within_distance",), "rank_mag"),
    ("P10", ("depth_at_least",), "max_mag"),
    ("P11", ("time_between",), "min_depth"),
    ("P12", ("mag_below",), "nearest_distance"),
]
FILTER_OPS_020 = ("mag_at_least", "mag_below", "depth_shallower_than", "depth_at_least", "within_distance",
                  "time_between", "in_region")
TERMINAL_OPS_020 = ("count", "argmax_mag", "rank_mag", "max_mag", "max_depth", "min_depth", "nearest_to",
                    "nearest_distance", "compare_regions", "time_of_largest")
EXCLUSIVE_020 = (frozenset({"mag_at_least", "mag_below"}), frozenset({"depth_shallower_than", "depth_at_least"}))
MONOTONE_020 = {
    ("mag_at_least", "max_mag"), ("mag_at_least", "argmax_mag"), ("mag_at_least", "time_of_largest"),
    ("mag_at_least", "rank_mag"), ("depth_at_least", "max_depth"), ("depth_shallower_than", "min_depth"),
}
RADII_020 = {"km": (25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300),
             "mi": (15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200)}


def compatible_020(filters, terminal) -> bool:
    chosen = set(filters)
    if any(pair <= chosen for pair in EXCLUSIVE_020):
        return False
    if any((f, terminal) in MONOTONE_020 for f in chosen):
        return False
    if terminal == "compare_regions" and "in_region" in chosen:
        return False
    if terminal in ("nearest_to", "nearest_distance") and "within_distance" in chosen:
        return False
    return True


def threshold_020(op, events, rng):
    """draw_filter for a magnitude or depth threshold, as it was in 0.2.0."""
    def pick(values, lo, hi):
        ordered = sorted(values)
        return ordered[int(rng.uniform(lo, hi) * (len(ordered) - 1))]

    if op in ("mag_at_least", "mag_below"):
        return {"op": op, "value": round(pick([e.mag for e in events], 0.25, 0.85), 1)}
    unit = rng.choice(("km", "mi"))
    depth_km = pick([e.depth_km for e in events], 0.25, 0.8) + rng.uniform(-1.5, 1.5)
    value = from_km(depth_km, unit)
    value = float(round(value)) if rng.random() < 0.5 else round(value, 1)
    return {"op": op, "value": value, "unit": unit} if value > 0 else None


def resolve_020(reports, notices, mishandle=None):
    """rules.resolve as it was in 0.2.0."""
    m = mishandle or Mishandle()
    notices = list(notices)
    retracted = {n.target for n in notices if n.kind == "retraction"} - set(m.keep_retracted)
    corrections = {n.target: n for n in notices if n.kind == "correction" and n.target not in m.ignore_correction_for}
    split = {tuple(pair) for pair in m.split}
    groups, separate = {}, []
    for r in reports:
        event_id = r.same_as or r.ref
        if r.ref in retracted or event_id in retracted:
            continue
        if (r.ref, r.network, r.status) in split:
            separate.append(report_to_event(r, r.ref))
            continue
        groups.setdefault(event_id, []).append(r)
    events = []
    for event_id, group in groups.items():
        reviewed = [r for r in group if r.status == "reviewed"]
        preliminary = [r for r in group if r.status != "reviewed"]
        base = preliminary[0] if event_id in m.prefer_preliminary_for and preliminary else (
            reviewed[0] if reviewed else group[0])
        event = report_to_event(base, event_id)
        fix = corrections.get(event_id)
        if fix is not None and fix.new_mag is not None:
            event = replace(event, mag=fix.new_mag, mag_type=fix.mag_type or event.mag_type)
        events.append(event)
    return events + separate


def test_derived_earthquake_guard_sets_equal_the_020_literals():
    assert len(_EXCLUSIVE) == len(EXCLUSIVE_020) and set(_EXCLUSIVE) == set(EXCLUSIVE_020)
    assert _MONOTONE == MONOTONE_020
    assert filter_ops(EARTHQUAKES) == FILTER_OPS_020 == FILTER_OPS  # the population twins are sampled from
    assert [(t.template_id, t.filters, t.terminal) for t in PUBLIC_TEMPLATES] == PUBLIC_020
    assert {t.pool for t in PUBLIC_TEMPLATES} == {"public"} and not any(t.twin_of for t in PUBLIC_TEMPLATES)
    for size in (1, 2, 3):
        for filters in itertools.permutations(FILTER_OPS_020, size):
            for terminal in TERMINAL_OPS_020:
                assert _compatible(filters, terminal) == compatible_020(filters, terminal), (filters, terminal)


def test_earthquake_field_specs_carry_the_020_constants():
    mag, depth = EARTHQUAKES.field("mag"), EARTHQUAKES.field("depth")
    assert dict(EARTHQUAKES.roles) == {"primary": "mag", "secondary": "depth"}
    # magnitude: round(quantile in [0.25, 0.85), 1); no unit draw, jitter, coin, positivity or margin
    assert (mag.units, mag.quantiles, mag.jitter) == ((), (0.25, 0.85), 0.0)
    assert (mag.threshold_step, mag.coarse_step) == (0.1, None)
    assert not mag.positive_threshold and mag.margin == (0.0, 0.0)  # magnitudes are printed exactly
    assert mag.rank_gap - FLOAT_SLACK == 0.1 - 1e-9  # 0.2.0 MAG_GAP
    # depth: a unit draw (km, mi), quantile in [0.25, 0.8) ± 1.5 km, whole or 1 decimal, above zero
    assert (depth.canonical_unit, [u.symbol for u in depth.units], depth.units) == ("km", ["km", "mi"], LENGTH)
    assert MI.scale == KM_PER_MILE == 1.609344 and [u.words for u in LENGTH] == ["km", "miles"]
    assert (depth.quantiles, depth.jitter, depth.threshold_step, depth.coarse_step) == ((0.25, 0.8), 1.5, 0.1, 1.0)
    assert depth.positive_threshold and depth.margin == (0.6, 0.03)  # DEPTH_MARGIN_KM, DEPTH_MARGIN_REL
    assert depth.decimals == 1  # answers: convert, then round(1)
    assert (mag.filters, depth.filters) == (("at_least", "below"), ("below", "at_least"))
    assert (mag.answer_type, depth.answer_type) == ("magnitude", "depth")
    assert [scoring.ANSWER_TYPES[s.answer_type].tolerance for s in (mag, depth)] == [mag.tolerance, depth.tolerance]
    assert all(u.decimals is None and u.tolerance is None for u in depth.units)  # the field's, in km and mi
    spatial = EARTHQUAKES.spatial
    assert EARTHQUAKES.has_location and dict(spatial.radii) == RADII_020
    assert (spatial.margin, spatial.nearest_gap) == ((3.0, 0.06), (3.0, 0.08))  # DISTANCE_*, NEAREST_GAP_*
    assert spatial.nearest_max_km == NEAREST_MAX_KM == 500.0
    assert (spatial.point_offset_km, spatial.decimals) == ((5.0, 60.0), 1)
    assert spatial.point_label.format(source="NC", number=42) == "station NC-42"
    assert (TIME_MARGIN_S, RANKS) == (120, (2, 3))


def test_threshold_draws_follow_the_020_recipe_draw_for_draw(snapshot_events):
    events = sorted(snapshot_events.values(), key=lambda e: (e.time, e.id))
    shallow = [replace(e, depth_km=round(e.depth_km / 20 - 0.4, 1)) for e in events]  # thresholds at or below 0
    refused = 0
    for seed in range(300):
        sample = random.Random(seed).sample(shallow if seed % 3 == 0 else events, 25)
        for op in ("mag_at_least", "mag_below", "depth_shallower_than", "depth_at_least"):
            new, old = random.Random(seed), random.Random(seed)
            drawn, expected = draw_filter(op, sample, new, Context(0, 24), set()), threshold_020(op, sample, old)
            assert json.dumps(drawn) == json.dumps(expected), (seed, op)  # 12.0 stays 12.0, never 12
            assert new.getstate() == old.getstate(), (seed, op)  # the same draws, refused or not
            refused += drawn is None
    assert refused > 0


def test_thresholds_round_to_their_step():
    for x in (0.05, 0.15, 2.675, 12.45, -0.35, 3.0, 1234.5678, 0.0):
        assert round_to_step(x, 0.1) == round(x, 1) and round_to_step(x, 0.01) == round(x, 2)
        assert repr(round_to_step(x, 1.0)) == repr(float(round(x))) and round_to_step(x, 10) == round(x, -1)
    assert (round_to_step(21.3, 0.5), round_to_step(21.2, 0.5), round_to_step(7.4, 5), round_to_step(0.3, 0.25)) == (
        21.5, 21.0, 5.0, 0.25)


# ------------------------------------------------------------------ the earthquake domain


RANGE = re.compile(r"\d (?:km|miles?) or more and no more than \d")  # "16 km or more and no more than 40 km"


def test_off_the_legacy_path_no_radius_reads_as_the_end_of_a_range():
    """Joined to a depth threshold, the 0.2.0 wording of a radius reads as a range of depths: "with a depth
    of 16 km or more and no more than 40 km from station US-28" (16 to 40 km deep?). Legacy prompts keep it,
    word for word; prompts off the legacy path word every radius by naming the distance before its number
    (DISTANCE_WORDINGS), in every domain with a location, and draw exactly as often."""
    assert LEXICON.filters["within_distance"] == ("within {value} of {place}", "no more than {value} from {place}")
    assert EARTHQUAKES.lexicon is LEXICON  # what legacy prompts are worded with
    assert EARTHQUAKES.prompt_lexicon() == GENRE_LEXICON == genre_lexicon(LEXICON)
    assert GENRE_LEXICON.filters == {**LEXICON.filters, "within_distance": DISTANCE_WORDINGS}
    assert (GENRE_LEXICON.records, GENRE_LEXICON.all_records, GENRE_LEXICON.questions) == (
        LEXICON.records, LEXICON.all_records, LEXICON.questions)
    assert len(DISTANCE_WORDINGS) == len(LEXICON.filters["within_distance"])
    assert all(re.match(r"(?:within|at a distance of no more than) \{value\} (?:of|from) \{place\}$", w)
               for w in DISTANCE_WORDINGS)
    assert WEATHER.prompt_lexicon().filters["within_distance"] == DISTANCE_WORDINGS
    assert standard_lexicon(WEATHER).filters["within_distance"] == DISTANCE_WORDINGS  # any domain worded from its words
    own = replace(LEXICON, filters={**LEXICON.filters, "within_distance": ("inside {value} of {place}",)})
    assert genre_lexicon(own) is own  # a wording of a domain's own is kept

    question = Question("Q1", "P02", "public", [
        {"op": "dedupe"}, {"op": "depth_at_least", "value": 16.0, "unit": "km"},
        {"op": "within_distance", "lat": -18.038, "lon": -173.353, "label": "station US-28", "radius": 40,
         "unit": "km"},
        {"op": "count"}], "count")
    legacy = [phrase_question(question, random.Random(seed)) for seed in range(60)]
    genre = [phrase_question(question, random.Random(seed), lexicon=EARTHQUAKES.prompt_lexicon()) for seed in range(60)]
    assert any(RANGE.search(text) for text in legacy)  # the 0.2.0 wording, kept on the legacy path
    assert not any(RANGE.search(text) for text in genre)
    # The same draws, word for word, but for the radius: nothing else about a question moves.
    assert [t.replace("at a distance of no more than", "no more than") for t in genre] == legacy
    assert any("within 40 km of" in t for t in genre) and any("at a distance of no more than 40 km" in t for t in genre)


def test_the_registry_serves_the_earthquake_domain():
    assert get_domain() is get_domain(DEFAULT_DOMAIN) is EARTHQUAKES is DOMAIN
    names = (EARTHQUAKES.name, EARTHQUAKES.noun, EARTHQUAKES.plural, EARTHQUAKES.id_label,
             EARTHQUAKES.source_label, EARTHQUAKES.group_label)
    assert names == ("earthquakes", "earthquake", "earthquakes", "event ID", "network", "region")
    assert EARTHQUAKES.id_pattern == EVENT_ID_PATTERN
    assert (EARTHQUAKES.id_answer_type, EARTHQUAKES.group_answer_type) == ("event_id", "region")
    # Its prompts off the legacy path print the 0.2.0 rules 1-8 word for word, headed as every domain's
    # rules are (a run of several domains prints several rules, so none is "identical for every digest"),
    # then rule 9 naming every kind of line without data that the genres print, and the genre-neutral
    # rules on message times and repeats. Legacy prompts print the 0.2.0 rules alone
    # (test_regression_020.py).
    rules = EARTHQUAKES.rules_text()
    heading, rule_9 = "Reading rules (identical for every digest):\n", "\n9. Lines that are neither reports nor notices"
    rules_1_to_8 = READING_RULES[len(heading):READING_RULES.index(rule_9) + 1]
    assert READING_RULES.startswith(heading) and rules_1_to_8.startswith("1. Every earthquake has exactly one")
    assert rules == "Reading rules:\n" + rules_1_to_8 + numbered_rules(
        ("Lines that are neither reports nor notices (headers, section titles, station status, editor's notes, "
         "reminders, greetings, signatures) carry no data and change nothing.",
         "The time at which a message was sent, forwarded or relayed is never the time of an earthquake.",
         REPEAT_RULE), start=9)
    assert rules.split("\n", 1)[0] == record_rules(EARTHQUAKES).split("\n", 1)[0] == "Reading rules:"
    assert "identical for every digest" not in rules


def test_the_earthquake_domain_delegates_to_the_020_functions(instances):
    compiled_at = 1_780_000_000
    for seed in range(4):
        a, b = random.Random(seed), random.Random(seed)
        # A legacy run draws its snapshot from the 0.2.0 world itself (build_snapshot); off the legacy path the
        # domain draws that world first, then re-files some of its events and adds aftershocks (synthetic_world).
        n_after = round(60 * AFTERSHOCK_SHARE)
        world = EARTHQUAKES.synthetic(a, 60, compiled_at, 24)
        first = generate_synthetic(b, 60 - n_after, compiled_at, 24)
        assert {(e.time, e.lat, e.lon, e.depth_km, e.mag, e.region) for e in first} <= {
            (e.time, e.lat, e.lon, e.depth_km, e.mag, e.region) for e in world}
        a, b = random.Random(seed), random.Random(seed)
        taken_a, taken_b = set(), set()
        assert ([EARTHQUAKES.new_id(n, a, taken_a) for n in ("us", "nc", "ak")]
                == [new_event_id(n, b, taken_b) for n in ("us", "nc", "ak")])
        assert EARTHQUAKES.mimic_id("aka2026vupcsv", a, taken_a) == mimic_event_id("aka2026vupcsv", b, taken_b)
        assert taken_a == taken_b
    event = generate_synthetic(random.Random(0), 1, compiled_at)[0]
    assert builders_for(EARTHQUAKES).primary(event) == primary_report(event)
    for inst in instances:
        assert EARTHQUAKES.resolve(inst.reports, inst.notices) == resolve(inst.reports, inst.notices)


def test_off_the_legacy_path_the_world_has_aftershocks_and_relays_that_file_events():
    """The synthetic world off the legacy path: the 0.2.0 events, some of them filed by the networks that
    re-report others' events (RELAY_FILINGS), and aftershocks seconds to hours after a larger event of their
    network, a few km away, so that two real events are sometimes as close as a re-report to its event. No
    two events are one quake's solutions (within 3 s and 15 km of each other), as no two of a real feed's."""
    compiled_at = 1_780_000_000
    world = EARTHQUAKES.synthetic(random.Random(11), 2000, compiled_at, 24)
    assert world == EARTHQUAKES.synthetic(random.Random(11), 2000, compiled_at, 24) and len(world) == 2000
    assert record_problems(EARTHQUAKES, world) == [] and len({e.id for e in world}) == len(world)
    assert all(re.fullmatch(EVENT_ID_PATTERN, e.id) and e.id.startswith(e.network) for e in world)
    networks = Counter(e.network for e in world)
    for network, regions, _, _ in RELAY_FILINGS:  # every network that re-reports files events of its own
        assert networks[network] > 0 and all(e.region in regions or network == "us" for e in world
                                             if e.network == network and e.region not in regions)
    close = 0
    for i, e in enumerate(world):
        for o in world[i + 1:i + 60]:
            if o.time - e.time > 900:
                break
            km = haversine_km(e.lat, e.lon, o.lat, o.lon)
            assert not (o.time - e.time <= 3 and km <= 15.0)  # never one quake's two solutions
            close += o.region == e.region and o.time - e.time <= 40 and km <= 8.5
    assert 0.03 * len(world) < close < 0.2 * len(world)  # aftershocks: real events as close as a re-report


def test_the_trap_vocabulary_is_what_the_020_digests_print(instances):
    vocab = EARTHQUAKES.trap_vocab
    assert vocab.rereporting_sources == ("us", "at", "pt", "ew")  # the order rng.choice indexes
    assert {"quarry blast", "false trigger from instrument noise"} <= set(vocab.retraction_reasons)
    printed = {n.reason for inst in instances for n in inst.notices if n.kind == "retraction"}
    assert printed and printed <= set(vocab.retraction_reasons)
    notice = Notice(kind="retraction", target="us7000abcd", time=1770530400, reason="quarry blast",
                    new_mag=4.2, old_mag=4.5, mag_type="mb")
    for kind, phrases in (("retraction", vocab.retraction_notices), ("correction", vocab.correction_notices)):
        wordings = [render_notice(replace(notice, kind=kind), form, "iso") for form in range(3)]
        assert all(sum(p in w for p in phrases) == 1 for w in wordings), wordings  # each wording uses one phrase
        assert all(any(p in w for w in wordings) for p in phrases)  # and every phrase is printed
    duplicate = replace(primary_report(generate_synthetic(random.Random(0), 1, 1_780_000_000)[0]),
                        ref="at12345678", network="at", status="preliminary", same_as="us7000abcd")
    lines = [render_report(duplicate, style, time_fmt="iso", coord_fmt="hemi", depth_fmt="km", mag_form=0)
             for style in ("bulletin", "prose", "wire")]  # the log style prints a same_as= field instead
    assert all(sum(f"{note} us7000abcd" in line for note in vocab.duplicate_notes) == 1 for line in lines), lines
    assert all(any(note in line for line in lines) for note in vocab.duplicate_notes)


def test_the_glossary_describes_every_field_and_stays_out_of_legacy_prompts(instances):
    glossary = EARTHQUAKES.glossary()
    assert (glossary.noun, glossary.plural, glossary.id_label, glossary.source_label, glossary.group_label) == (
        "earthquake", "earthquakes", "event ID", "network", "region")
    assert [(f.key, f.label, f.canonical_unit, f.units, f.decimals) for f in glossary.fields] == [
        ("mag", "magnitude", None, (), 2), ("depth", "depth", "km", LENGTH, 1)]
    assert glossary.units == EARTHQUAKES.units
    assert "domain" not in {f.name for f in fields(Glossary)}  # a prompt never names a (held-out) domain
    assert not any("<glossary>" in r.prompt for inst in instances for r in inst.renderings)


# --------------------------------------------------------------------------------- traps


def generic(step: dict) -> dict:
    """A 0.2.0 threshold step in its generic form."""
    names = {"mag_at_least": ("at_least", "mag"), "mag_below": ("below", "mag"),
             "depth_shallower_than": ("below", "depth"), "depth_at_least": ("at_least", "depth")}
    if step["op"] not in names:
        return step
    kind, field = names[step["op"]]
    return {"op": kind, "field": field, **{k: v for k, v in step.items() if k != "op"}}


def test_conflict_thresholds_are_read_from_aliases_and_generic_steps(instances):
    questions = [q for inst in instances for q in inst.questions]
    legacy = [s["value"] for q in questions for s in q.chain if s["op"] in ("mag_at_least", "mag_below")]  # 0.2.0
    assert legacy and threshold_values(questions, "mag") == legacy
    as_generic = [replace(q, chain=[generic(s) for s in q.chain]) for q in questions]
    assert threshold_values(as_generic, "mag") == legacy
    depths = [s["value"] for q in questions for s in q.chain if s["op"] in ("depth_shallower_than", "depth_at_least")]
    assert threshold_values(questions, "depth") == threshold_values(as_generic, "depth") == depths


# ------------------------------------------------------------------------- reading rules


def test_resolve_defaults_read_every_digest_as_020_did(instances):
    for inst in instances:
        readings = [None] + [Mishandle.from_dict(m) for t in inst.traps for m in t.mishandles]
        readings.append(Mishandle.merged(m for t in inst.traps for m in t.mishandles))
        for reading in readings:
            assert resolve(inst.reports, inst.notices, reading) == resolve_020(inst.reports, inst.notices, reading)
        assert naive_events(inst.reports) == [report_to_event(r, r.ref) for r in inst.reports]


def test_resolve_reads_record_reports_generically():
    t0 = 1_780_000_000

    def line(ref, status, temp, source="nws", same_as=None):
        return RecordReport(ref=ref, source=source, status=status, time=t0, lat=None, lon=None, group="West",
                            values={"temp": temp}, qualifiers={"sensor": "A"}, same_as=same_as)

    reports = [line("ksfo-01", "provisional", 23.0, "madis"),  # a re-report before the verified line
               line("ksfo-01", "verified", 21.0),
               line("mx-77", "provisional", 22.0, "madis", same_as="ksfo-01"),
               line("ksea-02", "verified", 12.5),
               line("kbos-03", "verified", 30.0)]
    fix = Notice(kind="correction", target="ksea-02", time=t0, field="temp", new_value=13.5, old_value=12.5,
                 qualifier="B")
    notices = [Notice(kind="retraction", target="kbos-03", time=t0, reason="sensor fault"), fix]
    rules = {"authoritative": "verified", "qualifiers": {"temp": "sensor"}}
    assert resolve(reports, notices, **rules) == [
        Record("ksfo-01", t0, None, None, "West", "nws", {"temp": 21.0}, {"sensor": "A"}),
        Record("ksea-02", t0, None, None, "West", "nws", {"temp": 13.5}, {"sensor": "B"})]
    unverified = resolve(reports, notices, qualifiers={"temp": "sensor"})  # nothing is "reviewed": first lines stand
    assert [r.values["temp"] for r in unverified] == [23.0, 13.5]
    careless = resolve(reports, notices, Mishandle(split=(("mx-77", "madis", "provisional"),),
                                                   keep_retracted=("kbos-03",)), **rules)
    assert [r.id for r in careless] == ["ksfo-01", "ksea-02", "kbos-03", "mx-77"]
    assert [r.values["temp"] for r in naive_events(reports)] == [23.0, 21.0, 22.0, 12.5, 30.0]
    # A correction is never guessed: it names a field the record has, and a qualifier needs the field's key.
    for notice, message in ((replace(fix, field=None), "names no field"),
                            (replace(fix, field="wind"), "names 'wind', which its record does not have")):
        with pytest.raises(ValueError, match=message):
            resolve(reports, [notice], **rules)
    with pytest.raises(ValueError, match="gives a qualifier, but 'temp' has none"):
        resolve(reports, notices, authoritative="verified")  # the earthquake rules cannot read this digest
    assert resolve(reports, [replace(fix, qualifier=None)], **rules)[1].qualifiers == {"sensor": "A"}


# ---------------------------------------------------------- a generic domain without locations

CELSIUS, FAHRENHEIT = Unit("°C", "degrees Celsius"), Unit("°F", "degrees Fahrenheit", 5 / 9, -160 / 9)
KMH, MPH = Unit("km/h", "km/h"), Unit("mph", "mph", KM_PER_MILE)


class Observations(Domain):
    """A test-only generic domain: temperature in the primary role, wind speed in the secondary one,
    and no locations, so no spatial templates. A "verified" line settles an observation's values, and
    a corrected temperature names its sensor."""

    name = "observations"
    noun = "observation"
    plural = "observations"
    id_label = "observation ID"
    source_label = "provider"
    group_label = "region"
    fields = (
        FieldSpec(key="temp", label="temperature", canonical_unit="°C", units=(CELSIUS, FAHRENHEIT), decimals=1,
                  answer_type="temperature", tolerance=(0.5, 0.0), margin=(0.3, 0.0), rank_gap=0.5,
                  quantiles=(0.2, 0.8), threshold_step=0.5, filters=("at_least", "below"),
                  terminals=("argmax", "rank", "max", "time_of_max"), signed=True),
        FieldSpec(key="wind", label="wind speed", canonical_unit="km/h", units=(KMH, MPH), decimals=1,
                  answer_type="wind_speed", tolerance=(1.0, 0.05), margin=(1.0, 0.05), rank_gap=2.0,
                  quantiles=(0.25, 0.75), threshold_step=1.0, jitter=2.0, positive_threshold=True,
                  filters=("below", "at_least"), terminals=("max", "min")),
    )
    roles = {"primary": "temp", "secondary": "wind"}
    lexicon = Lexicon(
        records="observations",
        all_records="all observations",
        filters={"at_least(temp)": ("with a temperature of at least {value}",),
                 "below(temp)": ("colder than {value}",),
                 "at_least(wind)": ("with wind of {value} or more",),
                 "below(wind)": ("with wind below {value}",),
                 "time_between": ("that were made from {start} up to {end}",),
                 "in_group": ("from the {group} region",)},
        questions={"count": ("How many {records} are there?",),
                   "argmax(temp)": ("Which of the {records} was the warmest?",),
                   "rank(temp)": ("Which of the {records} was the {nth} warmest?",),
                   "time_of_max(temp)": ("When was the warmest of the {records} made?",),
                   "max(temp)": ("What was the highest temperature, in {unit}, among the {records}?",),
                   "max(wind)": ("What was the strongest wind, in {unit}, among the {records}?",),
                   "min(wind)": ("What was the lightest wind, in {unit}, among the {records}?",),
                   "compare_groups": ("Which region had more {records}: {a} or {b}?",)},
    )
    id_pattern = r"obs-\d{3}"
    id_answer_type = "observation_id"
    group_answer_type = "us_region"
    trap_vocab = TrapVocab(rereporting_sources=("madis",), duplicate_notes=("same observation as",),
                           retraction_reasons=("sensor fault",), retraction_notices=("withdrawn",),
                           correction_notices=("corrected to",))

    def resolve(self, reports, notices, mishandle=None):
        return resolve(reports, notices, mishandle, authoritative="verified", qualifiers={"temp": "sensor"})


OBSERVATIONS = Observations()
T0 = 1_780_000_000


def observation_line(ref: str, status: str, temp: float, source: str = "nws") -> RecordReport:
    return RecordReport(ref=ref, source=source, status=status, time=T0, lat=None, lon=None, group="Pacific",
                        values={"temp": temp, "wind": 10.0}, qualifiers={"sensor": "A"})


def test_every_reader_of_a_digest_applies_its_domains_reading_rules(answer_registry, monkeypatch):
    """The executor, the oracle and trap grading read a generic digest by its domain's rules (here a
    "verified" line settles the values, and a correction's qualifier is a sensor), never by the
    earthquake rules, which cannot read it at all."""
    OBSERVATIONS.validate()
    for answer_type in OBSERVATIONS.answer_types():
        register_answer_type(answer_type)
    reports = [observation_line("obs-001", "provisional", 23.0, "madis"), observation_line("obs-001", "verified", 21.0),
               observation_line("obs-002", "verified", 12.5)]
    notices = [Notice(kind="correction", target="obs-002", time=T0, field="temp", new_value=13.5, old_value=12.5,
                      qualifier="B")]
    data = DigestData(reports, notices)
    warmest = [{"op": "dedupe"}, {"op": "max", "field": "temp"}]
    coolest = [{"op": "dedupe"}, {"op": "below", "field": "temp", "value": 20.0}, {"op": "argmax", "field": "temp"}]
    assert execute(warmest, data, units=OBSERVATIONS.units, resolver=OBSERVATIONS.resolve) == 21.0
    with pytest.raises(ValueError, match="gives a qualifier"):
        execute(warmest, data, units=OBSERVATIONS.units)
    monkeypatch.setattr(solver, "get_domain", lambda name=None: OBSERVATIONS)
    monkeypatch.setattr(solver, "read_digest", lambda prompt, genre=None, domain=None: data)
    specs = [{"qid": "Q1", "chain": warmest}, {"qid": "Q2", "chain": coolest}]
    assert solve("PROMPT", specs, domain="observations", genre="classic") == {"Q1": 21.0, "Q2": "obs-002"}
    assert solve("PROMPT", specs, mode="naive", domain="observations", genre="classic") == {"Q1": 23.0, "Q2": "obs-002"}
    question = Question("Q1", "P10", "public", warmest, "temperature")
    trap = Trap("T1", "conflicting_value", "second_report", "obs-001", "obs-001 also reported as 23.0",
                mishandles=[Mishandle(prefer_preliminary_for=("obs-001",)).to_dict()])
    traps.score_traps([trap], [question], {"Q1": 21.0}, reports, notices, domain=OBSERVATIONS)
    assert (trap.affected, trap.fallen) == ({"Q1": 23.0}, {"Q1": [23.0]})


def test_the_mock_gets_every_answer_type_of_a_domain_wrong(answer_registry, monkeypatch):
    for answer_type in OBSERVATIONS.answer_types():
        register_answer_type(answer_type)
    monkeypatch.setattr(adapters, "get_domain", lambda name=None: OBSERVATIONS)
    hidden = SimpleNamespace(reports=[observation_line(f"obs-00{i}", "verified", 20.0 + i) for i in range(1, 5)])
    task = Task("inst-000", "v1", [], hidden=hidden, domain="observations", genre="classic")
    in_fahrenheit = [{"op": "dedupe"}, {"op": "max", "field": "temp"}, {"op": "convert", "to": "°F"},
                     {"op": "round", "ndigits": 1}]
    in_mph = [{"op": "dedupe"}, {"op": "min", "field": "wind"}, {"op": "convert", "to": "mph"},
              {"op": "round", "ndigits": 1}]
    compare = [{"op": "dedupe"}, {"op": "compare_groups", "a": "Pacific", "b": "South"}]
    cases = [("temperature", "°F", in_fahrenheit, 70.3), ("temperature", None, in_fahrenheit[:2], 21.3),
             ("wind_speed", "mph", in_mph, 12.4), ("observation_id", None, in_fahrenheit[:1] + [
                 {"op": "argmax", "field": "temp"}], "obs-002"),
             ("us_region", None, compare, "Pacific"), ("region", None, compare, "tie"),  # a group type named "region"
             ("count", None, [{"op": "dedupe"}, {"op": "count"}], 3),
             ("utc_time", None, [{"op": "dedupe"}, {"op": "time_of_max", "field": "temp"}], "23:55")]
    for seed in range(20):
        for answer_type, unit, chain, value in cases:
            q = {"answer_type": answer_type, "unit": unit, "chain": chain}
            wrong = _perturb(q, value, random.Random(seed), task)
            assert not is_correct(answer_type, value, wrong, unit=unit), (answer_type, value, wrong)
            assert not isinstance(value, float) or round(wrong, 1) == wrong  # to the question's places
    with pytest.raises(ValueError, match="cannot get a probability answer"):
        _perturb({"answer_type": "probability", "unit": None, "chain": []}, 0.3, random.Random(0), task)


def observations(rng: random.Random, n: int) -> list[Record]:
    regions = ("Northeast", "Midwest", "Pacific", "South")
    records = [Record(id=f"obs-{i:03d}", time=T0 - rng.randrange(60, 86400), lat=None, lon=None,
                      group=rng.choice(regions), source="nws",
                      values={"temp": round(rng.uniform(-5.0, 35.0), 1), "wind": round(rng.uniform(0.0, 60.0), 1)})
               for i in range(n)]
    return sorted(records, key=lambda r: (r.time, r.id))


def test_role_templates_bind_to_a_domain_without_locations():
    public = public_templates(OBSERVATIONS)
    assert [t.template_id for t in public] == ["P01", "P03", "P04", "P05", "P06", "P08", "P10", "P11"]
    assert [(t.filters, t.terminal) for t in public[:4]] == [
        (("at_least(temp)",), "count"), (("in_group",), "argmax(temp)"), (("at_least(temp)",), "max(wind)"),
        (("at_least(temp)",), "compare_groups")]
    assert filter_ops(OBSERVATIONS) == ("at_least(temp)", "below(temp)", "below(wind)", "at_least(wind)",
                                        "time_between", "in_group")
    assert exclusive_pairs(OBSERVATIONS) == {frozenset({"at_least(temp)", "below(temp)"}),
                                             frozenset({"at_least(wind)", "below(wind)"})}
    assert monotone_pairs(OBSERVATIONS) == {
        ("at_least(temp)", "max(temp)"), ("at_least(temp)", "argmax(temp)"), ("at_least(temp)", "rank(temp)"),
        ("at_least(temp)", "time_of_max(temp)"), ("at_least(wind)", "max(wind)"), ("below(wind)", "min(wind)")}
    for seed in range(30):
        twins = compose_new_templates(random.Random(seed), domain=OBSERVATIONS)
        assert [t.twin_of for t in twins] == [t.template_id for t in public]  # the pools stay matched
        assert all(t.terminal == public[i].terminal and len(t.filters) == len(public[i].filters)
                   for i, t in enumerate(twins))
        assert {t.signature for t in twins}.isdisjoint(t.signature for t in public)


def test_questions_compose_guard_and_read_on_generic_records():
    records = observations(random.Random(5), 26)
    public = public_templates(OBSERVATIONS)
    twins = compose_new_templates(random.Random(5), domain=OBSERVATIONS)
    used: set[str] = set()
    usage: dict[str, int] = {}
    questions = []
    for i in range(10):
        q_rng = random.Random(100 + i)
        questions += compose_questions(records, public, twins, q_rng, Context(T0, 24), 4, used, usage,
                                       domain=OBSERVATIONS)
    assert len(questions) >= 30 and len({fingerprint(q.chain) for q in questions}) == len(questions)
    units = OBSERVATIONS.units
    for q in questions:
        for step in q.chain[1:]:
            if step["op"] in ("at_least", "below"):  # stored generically, at the field's step, clear of every record
                spec = OBSERVATIONS.field(step["field"])
                assert list(step) == ["op", "field", "value", "unit"]
                assert step["value"] == round_to_step(step["value"], spec.threshold_step)
                threshold = next(u for u in spec.units if u.symbol == step["unit"]).to_canonical(step["value"])
                assert all(abs(r.values[spec.key] - threshold) >= spec.margin[0] for r in records)
        answer = execute(q.chain, records, units=units)
        assert answer is not None and q.evidence == [r.id for r in rows_after_filters(q.chain, records, units=units)]
        kind = q.chain[-1]["op"] if q.chain[-1]["op"] != "round" else q.chain[-3]["op"]
        expected = {"count": "count", "argmax": "observation_id", "rank": "observation_id", "time_of_max": "utc_time",
                    "compare_groups": "us_region"}.get(kind) or OBSERVATIONS.field(q.chain[-3]["field"]).answer_type
        assert q.answer_type == expected
        for alt_units in (False, True):
            text = phrase_question(q, random.Random(q.qid), alt_units=alt_units, domain=OBSERVATIONS)
            assert "observations" in text and "{" not in text
    kinds = {q.chain[-1]["op"] if q.chain[-1]["op"] != "round" else q.chain[-3]["op"] for q in questions}
    assert kinds == {"count", "argmax", "max", "min", "compare_groups", "time_of_max"}  # rank came with P09 (spatial)
    assert not any(s["op"] in ("within_distance", "nearest_to", "nearest_distance") for q in questions for s in q.chain)
