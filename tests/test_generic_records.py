"""The generic record path, on the test-only toy domains (toy_domain.py, loaded as a plugin): mixes,
snapshots of Records, answer types whose tolerances hold in every unit, composition, guards and
phrasing for any schema (with and without a location, over a day and over a week), surfaces drawn
per instance, and the readers of a run (instance files, chatter, audit rows, report tallies)."""
from __future__ import annotations

import copy
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, replace

import pytest
import requests

from evalgen import pipeline, scoring
from evalgen.audit import write_audit_sample
from evalgen.digest import make_chatter
from evalgen.domains import all_domains, get_domain
from evalgen.domains.base import GENERIC_CHATTER, Lexicon, Record, RecordReport, Surface, TrapVocab, standard_lexicon
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres.classic import CLASSIC
from evalgen.mix import Entry, choices, parse_mix, single_choice
from evalgen.models import Instance, Notice, Rendering, Trap
from evalgen.ops import Step, execute, rows_after_filters, terminal_step
from evalgen.pipeline import GenerateConfig, compose_instance, sample_events
from evalgen.questions import (Context, _draw_threshold, _draw_window, compose_new_templates, fingerprint, fmt_num,
                               phrase_question, public_templates, round_to_step, rounding_text)
from evalgen.report import TRAP_KINDS, answer_type_order, load_instances, model_metrics, trap_kinds
from evalgen.scoring import get_answer_type, is_correct, normalize
from evalgen.seeds import SeedLog
from evalgen.sources import (build_domain_snapshot, events_sha256, load_snapshot, record_problems, save_snapshot)
from evalgen.units import Unit
from evalgen.util import read_json, sha256_hex, write_json

from conftest import quiet

COMPILED_AT = 1_780_000_000
SPATIAL_TEMPLATES = {"P02", "P07", "P09", "P12", "N02", "N07", "N09", "N12"}  # P02, P07, P09, P12 and their twins


# ------------------------------------------------------------------- domains and answer types


def test_the_toy_domains_load_as_plugins_and_register_their_answer_types(toys):
    assert get_domain("toy") is toys.toy and get_domain("toy-dressed") is toys.dressed
    names = [d.name for d in all_domains()]  # the built-in domains in module order, then the plugins as loaded
    assert names[0] == "earthquakes" and names[-3:] == ["toy", "toy-flat", "toy-dressed"]
    assert type(toys.toy).__module__.startswith("evalgen_plugin_")  # loaded by path, as a held-out plugin is
    temperature, wave = (scoring.ANSWER_TYPES[name] for name in ("toy_temperature", "toy_wave_height"))
    assert (temperature.tolerance, dict(temperature.unit_tolerance)) == ((0.3, 0.0), {"°F": (0.3 / (5 / 9), 0.0)})
    assert (wave.tolerance, dict(wave.unit_tolerance)) == ((0.05, 0.02), {"ft": (0.05 / 0.3048, 0.02)})
    # The same tolerance in °C and °F: 0.3 °C is 0.54 °F.
    assert is_correct("toy_temperature", 21.0, "21.3 °C", unit="°C") and not is_correct("toy_temperature", 21.0, 21.4)
    assert is_correct("toy_temperature", 70.0, 70.5, unit="°F")
    assert not is_correct("toy_temperature", 70.0, 70.6, unit="°F")
    assert is_correct("toy_wave_height", 10.0, 10.19, unit="ft") and not is_correct("toy_wave_height", 10.0, 10.25,
                                                                                   unit="ft")
    assert normalize("toy_reading_id", "Reading BY-01234.") == "by-01234"
    assert normalize("toy_area", "the North Sea sea area") == "north sea"
    for domain in (toys.toy, toys.flat, toys.dressed):
        domain.validate()
        assert domain.answer_types() == toys.toy.answer_types()


def test_a_tolerance_holds_in_every_unit_an_answer_can_be_asked_in(toys):
    module = toys.module
    miss = replace(module.WAVE, key="miss", canonical_unit="km", tolerance=(5000.0, 0.0),
                   units=(Unit("km", "km"), Unit("au", "au", 149597870.7, decimals=6), Unit("LD", "lunar distances",
                                                                                           384400.0, decimals=3)))
    assert miss.unit_tolerances() == {"au": (5000.0 / 149597870.7, 0.0), "LD": (5000.0 / 384400.0, 0.0)}

    def variant(*fields):
        return type("Variant", (module.Toy,), {"fields": fields})("variant", located=False)

    temperature, wave = module.TEMPERATURE, module.WAVE
    kilometres = Unit("km", "kilometres", 1000.0, decimals=5)
    cases = [
        (variant(temperature, replace(wave, tolerance_scales=False, units=(module.METRE, kilometres))),
         "the tolerance of wave is one number in m and in km, units of very different sizes"),
        (variant(replace(temperature, tolerance=(0.3, 0.01)), wave),
         "temp has a relative tolerance, which means nothing in °F"),
    ]
    for domain, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            domain.validate()
    own = replace(module.FAHRENHEIT, tolerance=(0.5, 0.0))  # a unit may set its own tolerance instead
    variant(replace(temperature, tolerance=(0.3, 0.01), answer_type="toy_temperature_2",
                    units=(module.CELSIUS, own)), wave).validate()
    depth = EARTHQUAKES.field("depth")  # 0.2.0 depths keep one tolerance in km and in miles
    assert not depth.tolerance_scales and depth.unit_tolerances() == {}


# ----------------------------------------------------------------------------------- mixes


def test_a_mix_names_one_domain_and_genre_until_phase_4(toys):
    assert parse_mix("legacy") == () and single_choice("legacy") is None
    assert parse_mix(" toy : classic ") == (Entry("toy", "classic"),) and parse_mix("toy") == (Entry("toy"),)
    assert single_choice("toy:classic") == single_choice("toy") == (toys.toy, CLASSIC)  # a bare domain: any genre
    assert choices(parse_mix("toy,toy-flat:classic")) == [(toys.toy, CLASSIC), (toys.flat, CLASSIC)]
    for text in ("toy,toy-flat:classic", "random"):
        with pytest.raises(NotImplementedError, match="phase 4"):
            single_choice(text)
    for text, error in (("nowhere", KeyError), ("toy:sonnet", KeyError), ("toy:", ValueError), (":classic", ValueError),
                        ("toy,toy", ValueError), ("", ValueError)):
        with pytest.raises(error):
            parse_mix(text)


def test_a_mix_puts_a_run_on_seed_paths_of_its_own(tmp_path):
    legacy = pipeline.generate(GenerateConfig(seed=4, instances=3), tmp_path / "legacy", log=quiet)
    mixed = pipeline.generate(GenerateConfig(seed=4, instances=3, mix="earthquakes:classic"), tmp_path / "mixed",
                              log=quiet)
    a, b = read_json(legacy / "manifest.json"), read_json(mixed / "manifest.json")
    assert "mix" not in a["config"] and b["config"]["mix"] == "earthquakes:classic"
    assert "domain" not in a["snapshot"] and b["snapshot"]["domain"] == "earthquakes"
    own = {"snapshot/clock", "snapshot/synthetic", "templates/new"}
    mine = {"snapshot/earthquakes/clock", "snapshot/earthquakes/synthetic", "templates/new/earthquakes"}
    assert own <= set(a["seeds"]) and not mine & set(a["seeds"])
    assert mine <= set(b["seeds"]) and not own & set(b["seeds"])
    assert set(a["seeds"]) - own == set(b["seeds"]) - mine  # the instances' own paths, and no surface path
    assert b["snapshot"]["events_sha256"] != a["snapshot"]["events_sha256"]
    for inst in load_instances(legacy):
        assert (inst.domain, inst.surface) == (None, None) and {r.genre for r in inst.renderings} == {None}
    for inst in load_instances(mixed):
        assert (inst.domain, inst.surface) == ("earthquakes", None)
        assert {r.genre for r in inst.renderings} == {"classic"}
    assert b["self_check"]["oracle_agreements"] == b["self_check"]["oracle_checks"] > 0
    snapshot = load_snapshot(mixed / "snapshot")
    assert snapshot.domain == "earthquakes" and record_problems(EARTHQUAKES, snapshot.events) == []


# ------------------------------------------------------------------------------- snapshots


def test_snapshots_of_records_are_built_saved_loaded_and_hashed(toys, tmp_path):
    for domain in (toys.toy, toys.flat):
        log = SeedLog(9)
        snapshot, raw = build_domain_snapshot(domain, "synthetic", log, pool_size=120)
        assert raw is None and list(log.seeds) == [f"snapshot/{domain.name}/clock", f"snapshot/{domain.name}/synthetic"]
        assert build_domain_snapshot(domain, "synthetic", SeedLog(9), pool_size=120)[0] == snapshot
        assert (snapshot.source, snapshot.domain, snapshot.window_hours) == ("synthetic", domain.name,
                                                                           domain.synthetic_window_hours)
        records = snapshot.events
        assert len(records) == 120 and all(isinstance(r, Record) for r in records)
        assert records == sorted(records, key=lambda r: (r.time, r.id)) and record_problems(domain, records) == []
        assert all((r.lat is not None) == domain.has_location for r in records)
        assert min(r.values["temp"] for r in records) < 0 < min(r.values["wave"] for r in records)  # signed and not
        assert all(snapshot.compiled_at - domain.synthetic_window_hours * 3600 <= r.time < snapshot.compiled_at
                   for r in records)
        directory = tmp_path / domain.name
        save_snapshot(snapshot, raw, directory)
        loaded = load_snapshot(directory)
        assert loaded == snapshot and events_sha256(loaded.events) == snapshot.events_sha256
        meta = read_json(directory / "snapshot.json")
        assert (meta["domain"], meta["n_events"]) == (domain.name, 120) and list(meta)[-2:] == ["domain", "n_events"]


class FakeFeed:
    """A live source that never touches the network: "fetching" builds a document of toy readings."""

    source = "buoyfeed"
    raw_filename = "raw_feed.json"
    user_agent = "evalgen-tests"
    min_records = 12

    def __init__(self, domain, n: int = 40, decimals: int = 1) -> None:
        self.domain, self.n, self.decimals = domain, n, decimals

    def fetch(self) -> tuple[bytes, str]:
        records = self.domain.synthetic(random.Random(5), self.n, COMPILED_AT, 6)
        rows = [dict(asdict(r), values={**r.values, "temp": round(r.values["temp"] + 0.04, self.decimals)})
                for r in records]
        return json.dumps({"compiled_at": COMPILED_AT, "window_hours": 6, "records": rows}).encode(), "fake://buoys"

    def parse(self, raw: bytes) -> tuple[list[Record], int, int]:
        doc = json.loads(raw.decode("utf-8"))
        return [Record(**row) for row in doc["records"]], doc["compiled_at"], doc["window_hours"]


def test_a_live_source_is_fetched_once_frozen_replayed_and_falls_back(toys, tmp_path):
    live = copy.copy(toys.flat)
    live.live = FakeFeed(toys.flat)
    log = SeedLog(1)
    snapshot, raw = build_domain_snapshot(live, "live", log)
    assert log.seeds == {}  # no synthetic draw
    assert (snapshot.source, snapshot.url, snapshot.window_hours, snapshot.raw_sha256) == (
        "buoyfeed", "fake://buoys", 6, sha256_hex(raw))
    assert build_domain_snapshot(live, "buoyfeed", SeedLog(1))[0].events == snapshot.events  # the source's own name
    run = tmp_path / "run" / "snapshot"
    save_snapshot(snapshot, raw, run, raw_filename=live.live.raw_filename)
    assert (run / "raw_feed.json").read_bytes() == raw
    replay, again = build_domain_snapshot(live, "live", SeedLog(2), snapshot_file=str(run / "raw_feed.json"))
    assert again == raw and replay.events_sha256 == snapshot.events_sha256
    assert (replay.url, replay.window_hours, replay.note) == ("fake://buoys", 6, "replay of frozen feed raw_feed.json "
                                                                                "(SHA-256 verified)")
    write_json(run / "snapshot.json", dict(read_json(run / "snapshot.json"), raw_sha256="0" * 64))
    tampered, _ = build_domain_snapshot(live, "live", SeedLog(2), snapshot_file=str(run / "raw_feed.json"))
    assert "WARNING" in tampered.note

    def offline():
        raise requests.ConnectionError("no network")

    for fallback, why in ((build_domain_snapshot(live, "live", SeedLog(3), fetch=offline), "ConnectionError"),
                          (build_domain_snapshot(replace_live(live, FakeFeed(toys.flat, n=5)), "live", SeedLog(3)),
                           "only 5 usable records")):
        snapshot, raw = fallback
        assert (snapshot.source, raw) == ("synthetic", None) and "fell back to synthetic" in snapshot.note
        assert why in snapshot.note and snapshot.window_hours == 168
    unread = build_domain_snapshot(toys.flat, "live", SeedLog(3))[0]  # a domain without a live source
    assert unread.source == "synthetic" and unread.note == "toy-flat has no live source; synthetic data"
    with pytest.raises(ValueError, match="no live source whose frozen bytes"):
        build_domain_snapshot(toys.flat, "live", SeedLog(3), snapshot_file=str(run / "raw_feed.json"))
    with pytest.raises(ValueError, match="unknown source 'usgs' for toy-flat"):
        build_domain_snapshot(live, "usgs", SeedLog(3))
    with pytest.raises(ValueError, match="not a number to 1 decimal places"):  # a broken adapter is an error
        build_domain_snapshot(replace_live(live, FakeFeed(toys.flat, decimals=2)), "live", SeedLog(3))


def replace_live(domain, feed):
    other = copy.copy(domain)
    other.live = feed
    return other


def test_record_problems_say_what_does_not_suit_the_domain(toys):
    good = toys.flat.synthetic(random.Random(1), 5, COMPILED_AT, 24)
    bad = [good[0], replace(good[1], id=good[0].id), replace(good[2], values={"temp": 1.25, "wave": 0.5}),
           replace(good[3], lat=50.0, lon=1.0, id="XY1"), replace(good[4], values={"temp": 1.0}, time=1.5)]
    problems = "; ".join(record_problems(toys.flat, bad))
    for message in (f"two records are {good[0].id}", "has temp 1.25, not a number to 1 decimal places",
                    "is at (50.0, 1.0), but toy-flat records have none", "XY1 does not match the ID pattern",
                    "has values for temp, not for temp, wave", "is timed 1.5, not in whole seconds"):
        assert message in problems
    assert record_problems(toys.flat, good) == []


# -------------------------------------------------------------------------------- sampling


def sample_020(pool, rng, n_min, n_max):
    """pipeline.sample_events as it was in 0.2.0."""
    def weighted(items, k):
        keyed = sorted(((rng.random() ** (1.0 / 1.5 ** e.mag), i) for i, e in enumerate(items)), reverse=True)
        return [items[i] for _, i in keyed[:k]]

    n = min(len(pool), rng.randint(n_min, n_max))
    counts = Counter(e.region for e in pool)
    eligible = sorted(r for r, c in counts.items() if c >= 4)
    focus = set(rng.sample(eligible, min(len(eligible), rng.randint(3, 4)))) if eligible else set()
    in_focus = [e for e in pool if e.region in focus]
    others = [e for e in pool if e.region not in focus]
    n_focus = min(len(in_focus), round(n * 0.8))
    chosen = weighted(in_focus, n_focus) + weighted(others, min(len(others), n - n_focus))
    if len(chosen) < n:
        chosen_ids = {e.id for e in chosen}
        chosen += weighted([e for e in pool if e.id not in chosen_ids], n - len(chosen))
    return sorted(chosen, key=lambda e: (e.time, e.id))


def test_sampling_follows_the_domains_groups_and_weights(toys, snapshot_events):
    quakes = sorted(snapshot_events.values(), key=lambda e: (e.time, e.id))
    for seed in range(12):  # earthquakes: the 0.2.0 draws and weights, float for float
        new, old = random.Random(seed), random.Random(seed)
        assert sample_events(quakes, new, 20, 28) == sample_020(quakes, old, 20, 28)
        assert new.getstate() == old.getstate()
    pool = toys.flat.synthetic(random.Random(2), 200, COMPILED_AT, 168)
    heavy = pool[17]
    seen: Counter = Counter()

    class Watched(toys.module.Toy):
        def sample_group(self, record):
            seen["group"] += 1
            return record.id[-1]  # ten groups by the last digit of the ID, not six sea areas

        def sample_weight(self, record):
            seen["weight"] += 1
            return 1e9 if record.id == heavy.id else 1.0

    watched = Watched("watched", located=False)
    for seed in range(10):
        sample = sample_events(pool, random.Random(seed), 20, 28, domain=watched)
        assert heavy in sample and len(sample) == len({r.id for r in sample}) >= 20
        digits = Counter(r.id[-1] for r in sample).most_common()
        assert sum(n for _, n in digits[:4]) >= round(0.8 * len(sample))  # focused on 3-4 of the hook's groups
    assert seen["group"] > 0 and seen["weight"] > 0


# --------------------------------------------------------------------- time and thresholds


def test_time_windows_and_their_guard_scale_with_the_window():
    hours = (1, 24, 25, 168, 720, 1440, 9600)
    steps = [(Context(0, h).time_step, Context(0, h).time_margin) for h in hours]
    assert steps == [(300, 120), (300, 120), (600, 240), (3600, 840), (10800, 3600), (21600, 7200), (172800, 48000)]
    for hours in (1, 24, 168, 720):
        ctx = Context(COMPILED_AT, hours)
        for seed in range(30):
            window = _draw_window(random.Random(seed), ctx)
            if window is None:
                continue
            assert window["start"] % ctx.time_step == 0 and window["end"] % ctx.time_step == 0
            assert COMPILED_AT - hours * 3600 <= window["start"] < window["end"] <= COMPILED_AT


def test_thresholds_round_to_their_units_step_and_are_never_minus_zero(toys):
    module = toys.module
    records = toys.flat.synthetic(random.Random(4), 60, COMPILED_AT, 168)
    steps = {"°C": 0.5, "°F": 1.0, "m": 0.1, "ft": 0.5}
    drawn = Counter()
    for seed in range(60):
        for spec in (module.TEMPERATURE, module.WAVE):
            params = _draw_threshold(spec, records, random.Random(seed))
            if params is None:
                continue
            drawn[params["unit"]] += 1
            assert round_to_step(params["value"], steps[params["unit"]]) == params["value"]
    assert set(drawn) == set(steps)
    near_zero = [replace(r, values={**r.values, "temp": -0.04}) for r in records]
    fine = replace(module.TEMPERATURE, units=(module.CELSIUS,), threshold_step=0.1)
    value = _draw_threshold(fine, near_zero, random.Random(0))["value"]
    assert value == 0.0 and math.copysign(1.0, value) == 1.0 and json.dumps(value) == "0.0"
    assert [rounding_text([{"op": "round", "ndigits": n}]) for n in (0, 1, 2)] == [
        "Round to a whole number.", "Round to 1 decimal place.", "Round to 2 decimal places."]
    assert rounding_text([{"op": "count"}]) == ""


# -------------------------------------------------------------------------- composition


def compose_run(domain, seed: int, instances: int = 12):
    """The first instances of a run about `domain` off the legacy path, composed as the pipeline does."""
    seedlog = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", seedlog)
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    new = compose_new_templates(seedlog.rng("templates", "new", domain.name), domain)
    used: set[str] = set()
    usage: dict[str, int] = {}
    composed = [compose_instance(i, snapshot, new, seedlog, GenerateConfig(seed=seed), ctx, used, usage,
                                 domain=domain, legacy=False) for i in range(instances)]
    return composed, new, ctx, seedlog


def check_guards(q, records, domain, ctx):
    """Every parameter of a question keeps its guard from every record of its instance."""
    units = domain.units
    for raw in q.chain:
        step = Step(raw, units)
        if step.kind in ("at_least", "below"):
            spec = domain.field(step.field)
            unit = step.unit(step["unit"])
            assert round_to_step(step["value"], unit.step or spec.threshold_step) == step["value"]
            assert not (step["value"] == 0 and math.copysign(1.0, step["value"]) < 0)
            threshold = unit.to_canonical(step["value"])
            margin = max(spec.margin[0], spec.margin[1] * abs(threshold))
            assert all(abs(r.values[spec.key] - threshold) >= margin for r in records), raw
            assert not spec.positive_threshold or step["value"] > 0
        elif step.kind == "time_between":
            assert step["start"] % ctx.time_step == 0 == step["end"] % ctx.time_step
            assert all(abs(r.time - edge) >= ctx.time_margin for r in records for edge in (step["start"], step["end"]))
    term = Step(terminal_step(q.chain), units)
    rows = rows_after_filters(q.chain, records, units=units)
    if term.kind in ("argmax", "rank", "time_of_max"):  # no tie down to the asked rank, nor just below it
        top = term["k"] + 1 if term.kind == "rank" else 2
        values = sorted((r.values[term.field] for r in rows), reverse=True)[:top]
        gap = domain.field(term.field).rank_gap - 1e-9
        assert all(a - b >= gap for a, b in zip(values, values[1:])), (q.chain, values)


@pytest.mark.parametrize("name", ["toy", "toy-flat", "toy-dressed"])
def test_questions_compose_guard_and_read_on_any_schema(toys, name):
    domain = get_domain(name)
    composed, new, ctx, seedlog = compose_run(domain, seed=21)
    public = public_templates(domain)
    assert [t.twin_of for t in new] == [t.template_id for t in public]  # every public template keeps its twin
    if not domain.has_location:
        assert {t.template_id for t in public} == {"P01", "P03", "P04", "P05", "P06", "P08", "P10", "P11"}
    assert ("instance/0/surface" in seedlog.seeds) == domain.has_surface
    fingerprints = []
    templates = set()
    for c in composed:
        bound, records = c.domain, c.events
        assert len(c.questions) >= 3 and record_problems(bound, records) == []
        for q in c.questions:
            fingerprints.append(fingerprint(q.chain))
            templates.add(q.template_id)
            assert execute(q.chain, records, units=bound.units) == c.answer_key[q.qid] is not None
            assert q.evidence == [r.id for r in rows_after_filters(q.chain, records, units=bound.units)]
            assert get_answer_type(q.answer_type).name == q.answer_type
            assert is_correct(q.answer_type, c.answer_key[q.qid], c.answer_key[q.qid], unit=q.unit)
            check_guards(q, records, bound, ctx)
            for alt_units in (False, True):
                text = phrase_question(q, random.Random(f"{q.qid}{alt_units}"), alt_units=alt_units, domain=bound)
                assert "{" not in text and "}" not in text and not re.search(r"\d[eE][-+]?\d", text), text
                assert bound.plural in text, text
                for raw in q.chain:
                    step = Step(raw, bound.units)
                    if step.kind in ("at_least", "below") and not alt_units:  # stated exactly as the chain has it
                        assert f"{fmt_num(step['value'])} {step.unit(step['unit']).words}" in text, text
                    if step.kind == "in_group":
                        assert step["group"] in text
    assert len(fingerprints) == len(set(fingerprints))  # no question is asked twice in a run
    assert bool(templates & SPATIAL_TEMPLATES) == domain.has_location
    assert len(templates) >= 8


def test_a_drawn_surface_words_and_names_each_instance(toys):
    composed, _, _, seedlog = compose_run(toys.dressed, seed=5)
    module = toys.module
    canonical_areas = {area.name for area in module.AREAS}
    surfaces = []
    for i, c in enumerate(composed):
        drawn = Surface.from_dict(c.surface)
        assert drawn.to_dict() == c.surface and c.surface == toys.dressed.draw_surface(
            random.Random(seedlog.seeds[f"instance/{i}/surface"]))
        surfaces.append(json.dumps(c.surface, sort_keys=True))
        bound = c.domain
        assert bound.unbound is toys.dressed and bound.surface == drawn and (bound.noun, bound.plural) == (
            drawn.noun, drawn.plural)
        assert [spec.label for spec in bound.fields] == [drawn.labels["temp"], drawn.labels["wave"]]
        assert all(r.id.startswith(drawn.id_prefix) for r in c.events)
        assert {r.group for r in c.events} <= set(drawn.groups.values())
        assert {r.source for r in c.events} <= set(drawn.sources.values())
        for q in c.questions:
            words = json.dumps(q.chain) + phrase_question(q, random.Random(0), domain=bound)
            assert not any(area in words for area in canonical_areas), words
        assert "{" not in bound.rules_text() and f"{drawn.noun} ID" in bound.rules_text()
    assert len(set(surfaces)) >= 3  # the surface is drawn for every instance
    assert toys.dressed.noun == "reading" and toys.dressed.surface is None  # binding never changes the domain
    legacy_log = SeedLog(5)  # the legacy path never draws one
    snapshot, _ = build_domain_snapshot(toys.dressed, "synthetic", legacy_log)
    c = compose_instance(0, snapshot, compose_new_templates(random.Random(0), toys.dressed), legacy_log,
                         GenerateConfig(seed=5), Context(snapshot.compiled_at, snapshot.window_hours), set(), {},
                         domain=toys.dressed)
    assert c.surface is None and c.domain is toys.dressed and "instance/0/surface" not in legacy_log.seeds


def test_binding_refuses_a_surface_that_does_not_hold_together(toys):
    module, dressed = toys.module, toys.dressed
    surface = dressed.draw_surface(random.Random(3))
    assert dressed.bind(None) is dressed
    yard = Unit("yd", "yards", 0.9144)
    wordy = Surface(lexicon=Lexicon("buoys", "all buoys", {}, {}))
    cases = [
        ({**surface, "labels": {"salinity": "salinity"}}, "a field 'salinity' that it does not have"),
        ({**surface, "groups": {"North Sea": "Kessel Reach", "Irish Sea": "Kessel Reach"}}, "two groups the same name"),
        ({**surface, "units": {"wave": [module.METRE.to_dict(), yard.to_dict()]}},
         "answers in the drawn unit yd would be scored within (0.05, 0.02)"),
        ({**surface, "units": {"wave": [module.METRE.to_dict(), replace(module.FOOT, decimals=0).to_dict()]}},
         "the wave margin 0.08 is smaller than the rounding of a threshold restated in ft"),
        ({**surface, "units": {"temp": []}}, "the surface gives temp no units, which it has"),
        (wordy.to_dict(), "its lexicon has no wording for"),
        ({**surface, "colour": "teal"}, "a surface has no colour"),
    ]
    for drawn, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            dressed.bind(drawn)
    prefixless = copy.copy(toys.flat)
    prefixless.id_prefix = None
    with pytest.raises(ValueError, match="its IDs have no prefix to replace"):
        prefixless.bind({"id_prefix": "zz-"})
    relative = copy.copy(dressed)  # a field whose tolerance is relative only is scored right in any drawn unit
    relative.fields = (module.TEMPERATURE, replace(module.WAVE, tolerance=(0.0, 0.02)))
    bound = relative.bind({**surface, "units": {"wave": [module.METRE.to_dict(), yard.to_dict()]}})
    assert bound.units["wave"][1] == yard
    renamed = dressed.bind(surface)
    record = toys.flat.synthetic(random.Random(1), 1, COMPILED_AT, 24)[0]
    assert renamed.dress(record) == replace(record, id=surface["id_prefix"] + record.id[3:],
                                            group=surface["groups"][record.group],
                                            source=surface["sources"][record.source])
    assert dressed.dress(record) is record and toys.flat.bind(None).dress(record) is record
    assert renamed.bind(surface).unbound is dressed  # a bound domain binds as the domain it was bound from
    with pytest.raises(TypeError, match="renames Records"):
        renamed.dress(EARTHQUAKES.synthetic(random.Random(0), 1, COMPILED_AT, 24)[0])
    partial = dressed.bind({**surface, "groups": {"North Sea": "Kessel Reach"}})  # renames every group, or fails
    with pytest.raises(ValueError, match="no name for the group 'Irish Sea'"):
        partial.dress(replace(record, group="Irish Sea"))


def test_surfaces_round_trip_through_json():
    surface = Surface(noun="consignment", plural="consignments", id_label="waybill", source_label="agency",
                      group_label="depot", labels={"mass": "heft"},
                      units={"mass": (Unit("vk", "varks"), Unit("pl", "pells", 1 / 12, decimals=2, step=5.0,
                                                                tolerance=(0.1, 0.0)))},
                      id_prefix="kx-", groups={"d1": "Varnholt"}, sources={"a1": "Ostrel Freight"},
                      trap_vocab=TrapVocab(("relay",), ("same as",), ("ledger echo",), ("void",), ("now",)),
                      chatter=("Quiet day at the {group_label}s.",),
                      lexicon=Lexicon("consignments", "all consignments", {"in_group": ("at {group}",)},
                                      {"count": ("How many {records}?",)}))
    d = surface.to_dict()
    assert json.loads(json.dumps(d)) == d and Surface.from_dict(json.loads(json.dumps(d))) == surface
    assert d["units"]["mass"] == [{"symbol": "vk", "words": "varks"}, {"symbol": "pl", "words": "pells",
                                                                       "scale": 1 / 12, "decimals": 2,
                                                                       "tolerance": [0.1, 0.0], "step": 5.0}]
    assert Surface().to_dict() == {} and Surface.from_dict({}) == Surface()


def test_the_standard_lexicon_words_every_operation_from_the_domains_words(toys):
    lexicon = standard_lexicon(toys.toy)
    assert lexicon == toys.toy.lexicon and (lexicon.records, lexicon.all_records) == ("readings",
                                                                                     "readings in the digest")
    assert set(lexicon.filters) == {"at_least(temp)", "below(temp)", "below(wave)", "at_least(wave)",
                                    "within_distance", "time_between", "in_group"}
    assert set(lexicon.questions) == {"count", "compare_groups", "argmax(temp)", "rank(temp)", "max(temp)",
                                      "time_of_max(temp)", "max(wave)", "min(wave)", "nearest_to", "nearest_distance"}
    assert "within_distance" not in standard_lexicon(toys.flat).filters
    assert lexicon.questions["max(wave)"][0] == ("What is the highest wave height, in {unit}, among the {records}? "
                                                 "{rounding}")
    braced = copy.copy(toys.flat)  # words are written as they are, braces included
    braced.fields = (replace(toys.module.TEMPERATURE, label="temp {raw}"), toys.module.WAVE)
    wording = standard_lexicon(braced).filters["at_least(temp)"][0]
    assert wording.format(value="5 °C") == "whose temp {raw} is at least 5 °C"


# -------------------------------------------------------------------- readers of a run


def toy_instance(c, index: int = 0) -> Instance:
    """An instance of a composition, assembled by hand: one reviewed line per record, a correction and two
    traps (instances that the pipeline plants and renders are tested in test_record_digests.py)."""
    bound, questions = c.domain, c.questions
    reports = [RecordReport(ref=r.id, source=r.source, status="reviewed", time=r.time, lat=r.lat, lon=r.lon,
                            group=r.group, values=dict(r.values), qualifiers=dict(r.qualifiers)) for r in c.events]
    notices = [Notice(kind="correction", target=c.events[0].id, time=c.events[0].time + 600, field="temp",
                      new_value=c.events[0].values["temp"], old_value=25.0, qualifier="spar", trap_id="T2")]
    traps = [Trap("T1", "duplicate", "same_id", c.events[1].id, "a relay", mishandles=[],
                  affected={questions[0].qid: 99}),
             Trap("T2", "conflicting_value", "correction", c.events[0].id, "a revised temperature", mishandles=[],
                  affected={questions[1].qid: "x"})]
    texts = {q.qid: phrase_question(q, random.Random(1), domain=bound) for q in questions}
    rendering = Rendering("v0", 1, {}, "DIGEST", texts, "PROMPT", sha256_hex("PROMPT"), genre="classic")
    return Instance(f"inst-{index:03d}", 1, COMPILED_AT, 168, [r.id for r in c.events], reports, notices, [], traps,
                    questions, dict(c.answer_key), [rendering], domain=bound.name, surface=c.surface)


def test_instances_of_record_domains_round_trip_through_json(toys):
    composed, *_ = compose_run(toys.dressed, seed=8, instances=2)
    inst = toy_instance(composed[0])
    d = json.loads(json.dumps(inst.to_dict()))
    assert d["domain"] == "toy-dressed" and d["surface"] == composed[0].surface and "split" not in d
    assert list(d["reports"][0]) == ["ref", "source", "status", "time", "lat", "lon", "group", "values", "qualifiers",
                                     "place", "same_as", "role", "trap_id"]
    again = Instance.from_dict(d)
    assert again == inst and all(isinstance(r, RecordReport) for r in again.reports)
    assert again.notices[0].field == "temp" and again.traps[1].kind == "conflicting_value"


def test_chatter_speaks_the_domains_words(toys):
    bound = toys.dressed.bind(toys.dressed.draw_surface(random.Random(2)))
    for domain in (toys.flat, bound):
        sources = sorted({r.source for r in map(domain.dress, toys.flat.synthetic(random.Random(1), 30, COMPILED_AT,
                                                                                   24))})
        for seed in range(8):
            rng, legacy = random.Random(seed), random.Random(seed)
            lines = make_chatter(rng, COMPILED_AT, sources, domain.chatter, domain.words)
            make_chatter(legacy, COMPILED_AT, sources)  # every line costs the draws a 0.2.0 line does
            assert rng.getstate() == legacy.getstate() and 1 <= len(lines) <= 3
            for line in lines:
                assert "{" not in line and not re.search(r"by-\d|\d\.\d", line), line
                assert line in {template.format(station=_station(line), h1=_clock(line, 0), h2=_clock(line, 1),
                                                **domain.words) for template in GENERIC_CHATTER}
    assert toys.flat.chatter is GENERIC_CHATTER and EARTHQUAKES.chatter != GENERIC_CHATTER


def _station(line: str) -> str:
    match = re.search(r"[A-Z]{2,4}\.[A-Z]{3}", line)
    return match.group(0) if match else ""


def _clock(line: str, n: int) -> str:
    clocks = re.findall(r"\d{2}:\d{2}", line)
    return clocks[n] if len(clocks) > n else ""


def test_audit_rows_and_report_tallies_read_records(toys, tmp_path):
    composed, *_ = compose_run(toys.dressed, seed=8, instances=2)
    instances = [toy_instance(c, i) for i, c in enumerate(composed)]
    records = {inst.instance_id: {r.id: r for r in c.events} for inst, c in zip(instances, composed)}
    audit = write_audit_sample(tmp_path, instances, records, random.Random(0), rate=1.0)
    assert audit["sampled"] == audit["population"] == sum(len(inst.questions) for inst in instances)
    entries = [json.loads(line) for line in (tmp_path / "audit" / "audit_sample.jsonl").read_text().splitlines()]
    rows = [row for entry in entries for row in entry["evidence"]]
    assert rows and all(list(row) == ["id", "time_utc", "lat", "lon", "values", "qualifiers", "group", "source"]
                        for row in rows)
    markdown = (tmp_path / "audit" / "audit_sample.md").read_text()
    for inst in instances:
        bound = toys.dressed.bind(inst.surface)
        assert f"**Evidence ({bound.plural} passing the filters" in markdown
        assert f"{bound.fields[1].label} " in markdown and " m, " in markdown
    assert trap_kinds(instances) == ("duplicate", "conflicting_value")
    assert trap_kinds([replace(instances[0], domain=None)]) == TRAP_KINDS  # a legacy report lists the 0.2.0 kinds
    assert answer_type_order(["toy_temperature", "count", "toy_area", "utc_time", "region"]) == [
        "count", "region", "utc_time", "toy_area", "toy_temperature"]
    rows = [{"instance_id": inst.instance_id, "variant": "v0", "qid": q.qid, "template_id": q.template_id,
             "pool": q.pool, "answer_type": q.answer_type, "unit": q.unit, "key": inst.answer_key[q.qid],
             "given": inst.answer_key[q.qid], "normalized": normalize(q.answer_type, inst.answer_key[q.qid]),
             "correct": True} for inst in instances for q in inst.questions]
    metrics = model_metrics({"model": "oracle", "kind": "reference", "rows": rows, "replies": []}, instances, [])
    assert list(metrics["by_answer_type"]) == answer_type_order(r["answer_type"] for r in rows)
    assert list(metrics["traps"]["by_kind"]) == ["duplicate", "conflicting_value"]
    assert metrics["traps"]["caught"]["rate"] == 1.0 and metrics["accuracy"]["rate"] == 1.0
