"""The invented domain (evalgen/domains/invented.py): consignments at depots of a world that does not
exist, whose every surface word is drawn per instance and defined only by the prompt's glossary.

Determinism from the seed (pinned in tests/fixtures/invented/), the drawn surfaces and their words,
units with exact conversions and the guards they keep, the synthetic world, snapshots of a domain
without a live source (which never touch the network), and whole runs with mix "invented:classic":
exact round trips of every report and notice, the guards, the traps, no repeated question, the oracle
at 100%, and metamorphic variants that restate thresholds in other units without changing the key."""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import requests

from evalgen import pipeline, scoring
from evalgen.adapters import Task, _perturb
from evalgen.digest import LINE_PRESETS, NOTICE_PLACEMENTS
from evalgen.domains import all_domains, get_domain
from evalgen.domains.base import RULES_WIDTH, Record, RecordReport, Surface
from evalgen.domains.invented import (BATCH_SECONDS, BATCH_SHARE, CHATTER, COMMON_WORDS, DEPOTS, FLOORS, INVENTED,
                                      LARGER_SCALES, RELAY_SHARE, RELAYS, RESERVED, SMALLER_SCALES, SOURCES, UNFIT_PARTS,
                                      WINDOW_HOURS, WORKING_HOURS, _Coiner, _fit)
from evalgen.genres import detect_genre
from evalgen.genres.base import RenderContext, parse_glossary, render_glossary
from evalgen.genres.classic import CLASSIC, RECORD_HEADERS, RECORD_LAYOUTS, RECORD_TIME_PRESETS, UNIT_PRESETS
from evalgen.models import Instance
from evalgen.ops import DigestData, Step, execute, rows_after_filters, terminal_step
from evalgen.pipeline import GenerateConfig, question_specs
from evalgen.questions import Context, fingerprint, fmt_num, public_templates, round_to_step
from evalgen.report import load_instances
from evalgen.rules import Mishandle
from evalgen.scoring import AnswerType, is_correct, normalize, register_answer_type, same_answer
from evalgen.seeds import SeedLog
from evalgen.solver import solve
from evalgen.sources import build_domain_snapshot, load_snapshot, record_problems
from evalgen.units import decode, encode, find_unit
from evalgen.util import plain_number, read_json, write_json

from conftest import quiet

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "invented"
COMPILED_AT = 1_780_000_000
SEED = 7101  # the run every run-level test reads
VOLATILE = {"run_id", "seed_commitment", "seed_nonce", "generated_at_utc", "code_sha256", "evalgen_version"}


@pytest.fixture(scope="module", autouse=True)
def registered():
    """The domain's answer types in the scorer, as discovering the domain registers them (whatever
    answer-type table a test before this module discovered the built-in domains with)."""
    for answer_type in INVENTED.answer_types():
        register_answer_type(answer_type)


def surfaces(n: int, start: int = 0) -> list[dict]:
    """The surfaces drawn from `n` consecutive seeds."""
    return [INVENTED.draw_surface(random.Random(seed)) for seed in range(start, start + n)]


def printed(reports, notices) -> tuple[list, list]:
    """Reports and notices as a digest prints them: without the generator's bookkeeping."""
    return ([replace(r, role="primary", trap_id=None) for r in reports], [replace(n, trap_id=None) for n in notices])


def same_items(got, want) -> bool:
    """Equal as multisets (by ==; the items are not hashable)."""
    remaining = list(want)
    for item in got:
        if item not in remaining:
            return False
        remaining.remove(item)
    return not remaining


# ------------------------------------------------------------------------- the domain itself


def test_the_domain_is_built_in_valid_and_scores_its_answers_in_any_unit():
    assert get_domain("invented") is INVENTED and INVENTED in all_domains()
    INVENTED.validate()
    assert (INVENTED.has_surface, INVENTED.has_location, INVENTED.live, INVENTED.forecaster) == (True, False, None, None)
    assert [t.template_id for t in public_templates(INVENTED)] == ["P01", "P03", "P04", "P05", "P06", "P08", "P10",
                                                                   "P11"]  # no location: no spatial templates
    f1, f2 = (scoring.ANSWER_TYPES[name] for name in ("invented_f1", "invented_f2"))
    assert f1 == AnswerType("invented_f1", "number", (0.0, 0.005), unit_tolerance={"tl": (0.0, 0.005)})
    assert f2 == AnswerType("invented_f2", "number", (0.0, 0.02), unit_tolerance={"sk": (0.0, 0.02)})
    for unit in ("vk", "tl", "zq", None):  # relative tolerances: the same in every unit, drawn or not
        assert is_correct("invented_f1", 183.7, 184.6, unit=unit)  # 0.5% of 183.7 is 0.92
        assert not is_correct("invented_f1", 183.7, 184.7, unit=unit)
        assert is_correct("invented_f2", 27.0, "27.5", unit=unit)  # 2% of 27 is 0.54
        assert not is_correct("invented_f2", 27.0, 27.6, unit=unit)
    assert normalize("invented_id", "Consignment KX/40512.") == "kx/40512" == normalize("invented_id", "kx/40512")
    assert normalize("invented_id", "kx-40512 or kx-40513") is None
    assert normalize("invented_group", "Kessel Reach") == "kessel reach" and normalize("invented_group", "same") == "tie"


def test_a_surface_is_drawn_from_its_seed_and_words_every_part_of_the_domain():
    seen = defaultdict(set)
    for seed, surface in enumerate(surfaces(60)):
        assert INVENTED.draw_surface(random.Random(seed)) == surface  # the seed decides every word
        assert json.loads(json.dumps(surface)) == surface and Surface.from_dict(surface).to_dict() == surface
        assert set(surface) == {"noun", "plural", "id_label", "source_label", "group_label", "id_prefix", "labels",
                                "groups", "sources", "units", "trap_vocab", "chatter"}  # all but the lexicon
        drawn = Surface.from_dict(surface)
        assert set(drawn.groups) == {depot.name for depot in DEPOTS} and set(drawn.sources) == set(SOURCES)
        assert re.fullmatch(r"(?:[A-Z]{2,3}|[a-z]{2,3})[-/.]", drawn.id_prefix)
        vocab = drawn.trap_vocab
        assert len(vocab.rereporting_sources) == 3 and vocab.rereporting_sources[2] in drawn.sources.values()
        assert len(vocab.duplicate_notes) == 2 and all(  # filled in with the instance's own noun
            note == f"same {drawn.noun} as" if note.startswith("same ") else "{" not in note
            for note in vocab.duplicate_notes)
        assert set(drawn.chatter) <= set(CHATTER) and len(drawn.chatter) == 5
        assert set(drawn.units) == {"f1", "f2"} and set(drawn.labels) == {"f1", "f2"}
        for units in drawn.units.values():  # a stored unit, then one or two others
            assert (units[0].scale, units[0].decimals) == (1.0, None) and 2 <= len(units) <= 3
        bound = INVENTED.bind(surface)  # binding checks units, margins, tolerances and wording
        assert (bound.noun, bound.unbound, bound.surface) == (drawn.noun, INVENTED, drawn)
        seen["noun"].add(drawn.noun)
        seen["labels"].add(tuple(drawn.labels.values()))
        seen["prefix"].add(drawn.id_prefix)
        seen["group_label"].add(drawn.group_label)
        seen["units"].add(tuple(u.words for units in drawn.units.values() for u in units))
        seen["groups"].add(tuple(drawn.groups.values()))
        seen["unit count"].add(sum(len(units) for units in drawn.units.values()))
    assert len(seen["labels"]) == len(seen["units"]) == len(seen["groups"]) == 60  # never the same twice
    assert len(seen["noun"]) >= 35 and len(seen["prefix"]) >= 55 and len(seen["group_label"]) >= 12
    assert seen["unit count"] == {4, 5, 6}
    assert INVENTED.noun == "consignment" and INVENTED.surface is None  # binding never changes the domain


def test_every_word_of_a_surface_is_its_own_and_none_is_reserved():
    for surface in surfaces(120, start=1000):
        drawn = Surface.from_dict(surface)
        units = [u for us in drawn.units.values() for u in us]
        words = ([drawn.noun, drawn.plural, drawn.id_label, drawn.source_label, drawn.group_label, *drawn.labels.values(),
                  *(u.symbol for u in units), *(u.words for u in units), drawn.id_prefix[:-1],
                  *drawn.groups.values(), *drawn.sources.values()])
        assert len({w.casefold() for w in words}) == len(words), words  # no word means two things
        # The agencies that re-report others' lots are the clearing agencies and one depot agency, and each of
        # them files lots of its own, so that no name marks a re-report.
        assert set(drawn.trap_vocab.rereporting_sources) <= set(drawn.sources.values())
        assert drawn.trap_vocab.rereporting_sources[:2] == tuple(drawn.sources[relay] for relay in RELAYS)
        coined = ([u.symbol for u in units] + [u.words[:-1] for u in units]
                  + [name.split()[0].lower() for name in (*drawn.groups.values(), *drawn.sources.values())])
        for word in coined:
            assert word not in RESERVED and not any(part in word for part in UNFIT_PARTS), word
        for name in (*drawn.groups.values(), *drawn.sources.values()):  # names a scorer reads as they are
            assert re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", name)
            assert normalize("invented_group", name) == name.casefold()


def test_a_group_answer_may_add_the_group_word_its_instance_prints():
    """Every domain reads "<group> <group word>" as the group: "Nevada region", "Earth body". An invented
    instance draws its own group word ("berth", "quay", ...), and the scorer reads answers by the type
    registered when the domain was loaded, so that type strips every group word a surface can draw, the
    drawn one included, never only the unbound domain's "depot". No group name ends in one of them."""
    labels: Counter = Counter()
    for surface in surfaces(150, start=300):
        bound = INVENTED.bind(surface)
        labels[bound.group_label] += 1
        for name in bound.surface.groups.values():
            assert is_correct("invented_group", name, f"{name} {bound.group_label}"), (name, bound.group_label)
            assert is_correct("invented_group", name, f"the {name} {bound.group_label.upper()}.")
            assert not is_correct("invented_group", name, f"{name} {bound.group_label}s")
    assert len(labels) >= 12 and sum(n for label, n in labels.items() if label != "depot") >= 130
    berth = INVENTED.bind(INVENTED.draw_surface(random.Random(5)))  # the audit's example
    assert berth.group_label == "berth" and is_correct("invented_group", "Nerm Halt", "Nerm Halt berth")


def test_no_unit_symbol_or_coined_word_is_a_common_english_word():
    """A unit symbol follows every value it writes, so a symbol that is a word ("a gross heft of 15.3 so",
    "dwell time 4 set") makes prose read two ways: about one surface in six drew such a symbol before the
    common English words of two and three letters were reserved. Coined names, labels and unit names
    avoid them too, and a symbol, drawn from the letters of a name that is fit to print, is checked as
    the name is: never a common word, never offensive."""
    assert {"so", "do", "me", "we", "he", "be", "set", "put"} <= COMMON_WORDS <= RESERVED
    assert all(re.fullmatch(r"[a-z]{2,3}", word) for word in COMMON_WORDS)
    symbols: Counter = Counter()
    for surface in surfaces(1500, start=5000):
        drawn = Surface.from_dict(surface)
        units = [u for us in drawn.units.values() for u in us]
        symbols.update(u.symbol for u in units)
        coined = [u.words[:-1] for u in units] + [name.split()[0].lower()
                                                  for name in (*drawn.groups.values(), *drawn.sources.values())]
        assert not set(coined) & COMMON_WORDS, coined
    assert len(symbols) > 250 and not set(symbols) & COMMON_WORDS and all(map(_fit, symbols))

    def scripted(*names: str) -> _Coiner:  # a coiner whose names are given, the first candidates of each taken
        coiner = _Coiner(random.Random(0))
        queue = iter(names)
        coiner.coin = lambda syllables: next(queue)
        return coiner

    coiner = scripted("sket")
    coiner.take("sk")  # "st" is a real unit, "sk" is taken, and "set" is a word: the symbol is "ske"
    assert coiner.unit() == ("ske", "skets")
    coiner = scripted("flag")
    coiner.take("fg", "fl")  # a fit name, whose next candidate symbol, "f" + "ag", is not
    assert coiner.unit() == ("fla", "flags")


def test_drawn_units_convert_exactly_and_keep_every_guard():
    records = INVENTED.synthetic(random.Random(5), 300, COMPILED_AT, WINDOW_HOURS)
    for surface in surfaces(40, start=200):
        bound = INVENTED.bind(surface)
        glossary = render_glossary(bound.glossary())
        for spec in bound.fields:
            stored, *others = spec.units
            assert (stored.scale, stored.offset, stored.decimals, stored.step) == (1.0, 0.0, None, None)
            assert spec.canonical_unit == stored.symbol and others
            for unit in others:
                assert unit.scale in LARGER_SCALES + SMALLER_SCALES and unit.offset == 0.0 and unit.tolerance is None
                assert (f'  - "{unit.symbol}" ("{unit.words}"): a value in "{stored.symbol}" is the value in '
                        f'"{unit.symbol}" × {plain_number(unit.scale)}.') in glossary  # stated exactly
                last = 10.0 ** -unit.decimals * unit.scale  # one last digit of an answer, in stored units
                assert last <= 10.0 ** -spec.decimals + 1e-12  # never coarser than a stored step ...
                assert unit.decimals == 0 or 10 * last > 10.0 ** -spec.decimals  # ... and no digit to spare
                assert 0.5 * last <= spec.margin[0]  # a restated threshold moves less than the guard margin
                assert 0.7 <= unit.step * unit.scale / spec.threshold_step <= 1.42  # a step near the field's own
                step_places = max(0, -Decimal(repr(unit.step)).normalize().as_tuple().exponent)
                assert unit.decimals >= step_places  # a threshold has no more places than an answer in its unit
            for r in records:  # every stored value prints in every unit and reads back exactly
                for unit in spec.units:
                    text = encode(r.values[spec.key], spec.decimals, unit)
                    assert decode(text, spec.decimals, unit) == r.values[spec.key], (r.values, unit, text)


def test_the_synthetic_world_is_seeded_and_shaped_as_described():
    records = INVENTED.synthetic(random.Random(3), 2400, COMPILED_AT, WINDOW_HOURS)
    assert records == INVENTED.synthetic(random.Random(3), 2400, COMPILED_AT, WINDOW_HOURS)
    assert records != INVENTED.synthetic(random.Random(4), 2400, COMPILED_AT, WINDOW_HOURS)
    assert record_problems(INVENTED, records) == [] and records == sorted(records, key=lambda r: (r.time, r.id))
    assert all(isinstance(r, Record) and r.id.startswith("iv-") and (r.lat, r.lon, r.place, r.qualifiers) ==
               (None, None, None, {}) for r in records)
    assert all(COMPILED_AT - WINDOW_HOURS * 3600 <= r.time < COMPILED_AT for r in records)
    by_depot = {depot.name: [r for r in records if r.group == depot.name] for depot in DEPOTS}
    total = sum(depot.weight for depot in DEPOTS)
    for depot in DEPOTS:
        mine = by_depot[depot.name]
        assert abs(len(mine) / len(records) - depot.weight / total) < 0.025  # drawn by weight
        assert {r.source for r in mine} == set(depot.sources) | set(RELAYS)  # its own agencies, and the clearing ones
        relayed = sum(r.source in RELAYS for r in mine) / len(mine)
        assert abs(relayed - RELAY_SHARE) < 0.06
        for key, (median, _) in (("f1", depot.f1), ("f2", depot.f2)):
            assert abs(statistics.median(r.values[key] for r in mine) / median - 1) < 0.15
    for key, floor in FLOORS.items():
        assert min(r.values[key] for r in records) >= floor
    logs = [(math.log(r.values["f1"]), math.log(r.values["f2"])) for r in records]
    assert 0.15 < _correlation(logs) < 0.45  # heavier lots dwell a little longer
    working = sum(WORKING_HOURS[0] <= r.time // 3600 % 24 < WORKING_HOURS[1] for r in records) / len(records)
    assert 0.75 < working < 0.86  # 14 busy hours, and the 10 others at 0.35 of their rate
    light, heavy = sorted(records, key=lambda r: r.values["f1"])[::len(records) - 1]
    assert INVENTED.sample_weight(heavy) > 1 > INVENTED.sample_weight(light) > 0  # heavy lots make the digest
    # A share of lots come in batches: the same agency files a second lot at the depot within a minute and a
    # half, so that two real records are sometimes as close as a re-report can be to its record.
    batched = sum(any(o is not r and (o.group, o.source) == (r.group, r.source)
                      and 0 < o.time - r.time <= BATCH_SECONDS[1] for o in records[i + 1:i + 40])
                  for i, r in enumerate(records))
    assert 0.6 * BATCH_SHARE < batched / len(records) < 1.6 * BATCH_SHARE


def _correlation(pairs) -> float:
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


def test_ids_take_each_instances_prefix_and_stay_unique():
    for surface in surfaces(20, start=50):
        bound = INVENTED.bind(surface)
        prefix = surface["id_prefix"]
        record = INVENTED.synthetic(random.Random(1), 1, COMPILED_AT, WINDOW_HOURS)[0]
        dressed = bound.dress(record)
        assert dressed.id == prefix + record.id[3:] and dressed.group == surface["groups"][record.group]
        assert dressed.source == surface["sources"][record.source] and dressed.values == record.values
        taken = {dressed.id}
        fresh = [bound.new_id("x", random.Random(i), taken) for i in range(50)]
        mimics = [bound.mimic_id(dressed.id, random.Random(i), taken) for i in range(50)]
        assert len(taken) == 101 and all(ref.startswith(prefix) for ref in fresh + mimics)
        assert all(re.fullmatch(INVENTED.id_pattern, ref.casefold()) for ref in taken)
        assert all(normalize("invented_id", f"It is {ref}.") == ref.casefold() for ref in fresh[:5])


# ------------------------------------------------------------------- snapshots and the seed


def test_the_seed_pins_the_world_and_every_surface():
    """What one seed draws in this domain alone, pinned when the domain was written: later code may
    change how instances are composed, but never the world or the words a seed gives. (The surfaces
    were pinned again, deliberately, when common English words joined the words no surface may coin,
    when the chatter's reminder about units was reworded, and with the world, when clearing agencies and
    batches of lots joined the world, so that the agencies that re-report lots file lots too and real lots
    are sometimes as close as a re-report.)"""
    pinned = read_json(FIXTURES / "pinned.json")
    log = SeedLog(pinned["seed"])
    snapshot, raw = build_domain_snapshot(INVENTED, "synthetic", log, pool_size=pinned["pool_size"])
    assert raw is None and (snapshot.compiled_at, snapshot.events_sha256) == (pinned["compiled_at"],
                                                                            pinned["events_sha256"])
    assert [{"id": r.id, "time": r.time, "group": r.group, "source": r.source, "values": r.values}
            for r in snapshot.events[:3]] == pinned["first_records"]
    drawn = [INVENTED.draw_surface(log.rng("instance", i, "surface")) for i in range(len(pinned["surfaces"]))]
    assert drawn == pinned["surfaces"]
    assert list(log.seeds) == ["snapshot/invented/clock", "snapshot/invented/synthetic", "instance/0/surface",
                               "instance/1/surface", "instance/2/surface"]


def test_without_a_live_source_every_snapshot_is_synthetic_and_never_touches_the_network(monkeypatch, tmp_path):
    def no_network(*_args, **_kwargs):
        raise AssertionError("the invented domain touched the network")

    monkeypatch.setattr(requests.Session, "request", no_network)
    assert INVENTED.live is None
    synthetic, raw = build_domain_snapshot(INVENTED, "synthetic", SeedLog(4))
    live, live_raw = build_domain_snapshot(INVENTED, "live", SeedLog(4))
    assert raw is live_raw is None and (live.source, live.domain, live.window_hours) == ("synthetic", "invented", 72)
    assert live.events == synthetic.events and live.note == "invented has no live source; synthetic data"
    with pytest.raises(ValueError, match="invented has no live source whose frozen bytes could be replayed"):
        build_domain_snapshot(INVENTED, "live", SeedLog(4), snapshot_file=str(FIXTURES / "pinned.json"))
    with pytest.raises(ValueError, match="unknown source 'usgs' for invented: use synthetic, live$"):
        build_domain_snapshot(INVENTED, "usgs", SeedLog(4))
    run = pipeline.generate(GenerateConfig(seed=4, instances=2, source="live", mix="invented:classic"),
                            tmp_path / "live", log=quiet)
    snapshot = read_json(run / "manifest.json")["snapshot"]
    assert (snapshot["source"], snapshot["note"], snapshot["raw_sha256"]) == ("synthetic", live.note, None)
    assert sorted(p.name for p in (run / "snapshot").iterdir()) == ["events.json", "snapshot.json"]  # nothing frozen
    assert load_snapshot(run / "snapshot").events == synthetic.events


# ------------------------------------------------------------------- rules, glossary, wording


def test_the_rules_name_every_notice_and_end_with_the_glossary_rule():
    for surface in surfaces(30, start=300):
        bound = INVENTED.bind(surface)
        rules = bound.rules_text()
        flat = " ".join(rules.split())
        vocab = bound.trap_vocab
        for phrase in (*vocab.retraction_notices, *vocab.correction_notices, "RETRACTION", "CORRECTION"):
            assert f'"{phrase}"' in flat or phrase in flat, phrase
        assert f'"{bound.statuses[0]}" report' in flat and f"was not a real {bound.noun}" in flat
        numbered = re.findall(r"^(\d+)\. ", rules, flags=re.MULTILINE)
        assert numbered == [str(n) for n in range(1, 11)] and rules.endswith("\n")
        assert f"10. The glossary is the only definition of the words in this digest: what a {bound.noun} is" in flat
        assert ("rounded to the decimal places the glossary gives for the field; no value is written so that it "
                "converts to exactly halfway between two of them.") in flat  # so no reader rounds a half
        assert all(len(line) <= RULES_WIDTH for line in rules.splitlines())
        glossary = bound.glossary()
        assert parse_glossary(render_glossary(glossary)) == glossary == INVENTED.glossary(surface)
        assert (glossary.noun, glossary.statuses, glossary.coord_decimals) == (bound.noun, ("verified", "provisional"),
                                                                              None)


def test_questions_are_worded_from_each_surfaces_words():
    wordings = Counter()
    for surface in surfaces(40, start=400):
        bound = INVENTED.bind(surface)
        lexicon = bound.lexicon
        assert lexicon.records == bound.plural and lexicon.all_records == f"{bound.plural} in the digest"
        for op, options in [*lexicon.filters.items(), *lexicon.questions.items()]:
            assert len(options) == len(set(options)) >= 2, op
            for text in options:
                filled = text.format(value="12 vk", group="Kessel Reach", start="s", end="e", records="R",
                                     unit="varks", rounding="Round to 1 decimal place.", nth="second", a="A", b="B")
                assert "{" not in filled and "}" not in filled
                wordings[op] += 1
                if re.search(r"\bwith an? ", filled):  # the article fits the label
                    label = next(spec.label for spec in bound.fields if spec.label in filled)
                    assert f"with {'an' if label[0] in 'aeiou' else 'a'} {label} of" in filled
    assert set(wordings) == {"at_least(f1)", "below(f1)", "below(f2)", "at_least(f2)", "time_between", "in_group",
                             "count", "compare_groups", "argmax(f1)", "max(f1)", "time_of_max(f1)", "max(f2)",
                             "min(f2)"}  # no rank: records have no location, and the one rank template is spatial


# ------------------------------------------------------------------------------- whole runs


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    """generate (mix "invented:classic") -> run (oracle, naive, mock) -> score -> report, 12 instances."""
    run_dir = pipeline.generate(GenerateConfig(seed=SEED, instances=12, mix="invented:classic"),
                                tmp_path_factory.mktemp("invented") / "run", log=quiet)
    pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)
    return run_dir


@pytest.fixture(scope="module")
def worlds(run) -> list[tuple[Instance, object, list[Record]]]:
    """Each instance, its domain bound to its surface, and its records (the snapshot's, as it names them)."""
    snapshot = load_snapshot(run / "snapshot")
    out = []
    for inst in load_instances(run):
        bound = INVENTED.bind(inst.surface)
        by_id = {r.id: r for r in map(bound.dress, snapshot.events)}
        out.append((inst, bound, [by_id[record_id] for record_id in inst.event_ids]))
    return out


def test_a_run_records_its_domain_its_surfaces_and_seed_paths_of_its_own(run, worlds, tmp_path):
    manifest = read_json(run / "manifest.json")
    assert manifest["config"]["mix"] == "invented:classic" and manifest["snapshot"]["domain"] == "invented"
    check = manifest["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    seeds = set(manifest["seeds"])
    assert {"snapshot/invented/clock", "snapshot/invented/synthetic", "templates/new/invented"} <= seeds
    assert {f"instance/{i}/surface" for i in range(12)} <= seeds and "templates/new" not in seeds
    assert len({json.dumps(inst.surface, sort_keys=True) for inst, _, _ in worlds}) == 12  # each its own words
    for inst, bound, records in worlds:
        assert inst.domain == "invented" and [r.genre for r in inst.renderings] == ["classic"] * 3
        assert {r.id[:len(inst.surface["id_prefix"])] for r in records} == {inst.surface["id_prefix"]}
        assert {r.group for r in records} <= set(inst.surface["groups"].values())
        for rendering in inst.renderings:
            prompt = rendering.prompt
            assert prompt.count("<glossary>") == 1 and parse_glossary(prompt) == bound.glossary()
            assert prompt.index(bound.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
            assert not re.search(r"\b(?:alder|birch|cedar|dogwood|elm|fir|gorse|hazel)\b|\biv-\d", prompt, re.I)
        path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
        write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def styles(seed: int):
    """Every combination of line and time presets, with the other presets taken in turn."""
    rng = random.Random(seed)
    for i, (lines, time) in enumerate((a, b) for a in sorted(LINE_PRESETS) for b in sorted(RECORD_TIME_PRESETS)):
        yield ({"units": sorted(UNIT_PRESETS)[i % 3], "time": time, "lines": lines, "layout": RECORD_LAYOUTS[i % 3],
                "notices": NOTICE_PLACEMENTS[i % 2], "header": i % len(RECORD_HEADERS)}, random.Random(rng.random()))


def test_classic_reads_back_every_report_and_notice_exactly(worlds):
    noticed = 0  # (a digest without a retraction and with a second report for its conflict has no notice)
    for n, (inst, bound, _) in enumerate(worlds):
        glossary = bound.glossary()
        reports, notices = printed(inst.reports, inst.notices)
        assert all(isinstance(r, RecordReport) for r in inst.reports)
        noticed += bool(notices)
        for rendering in inst.renderings:  # every rendering of the run, as the oracle reads it
            data = CLASSIC.parse(rendering.digest, glossary, bound)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), rendering.variant
            assert detect_genre(rendering.digest) is CLASSIC
        for style, rng in (styles(inst.seed) if n < 4 else ()):  # and every look the genre can draw
            text = CLASSIC.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at,
                                                inst.window_hours, style, rng, bound, glossary))
            data = CLASSIC.parse(text, glossary, bound)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style
        for line in inst.chatter:  # lines without data change nothing
            assert CLASSIC.parse(line, glossary, bound) == DigestData([], []) and "{" not in line
    assert noticed > len(worlds) // 2


def test_the_guards_hold_and_no_question_is_asked_twice(worlds):
    fingerprints, templates = [], Counter()
    for inst, bound, records in worlds:
        assert 3 <= len(inst.questions) <= 5
        ctx = Context(inst.compiled_at, inst.window_hours)
        assert (ctx.time_step, ctx.time_margin) == (900, 360)  # three days: quarter-hour marks, kept 6 minutes clear
        for q in inst.questions:
            fingerprints.append(fingerprint(q.chain))
            templates[q.template_id] += 1
            assert execute(q.chain, records, units=bound.units) == inst.answer_key[q.qid] is not None
            assert q.evidence == [r.id for r in rows_after_filters(q.chain, records, units=bound.units)]
            assert is_correct(q.answer_type, inst.answer_key[q.qid], inst.answer_key[q.qid], unit=q.unit)
            for raw in q.chain:
                step = Step(raw, bound.units)
                if step.kind in ("at_least", "below"):
                    spec, unit = bound.field(step.field), step.unit(step["unit"])
                    assert round_to_step(step["value"], unit.step or spec.threshold_step) == step["value"] > 0
                    threshold = unit.to_canonical(step["value"])
                    margin = max(spec.margin[0], spec.margin[1] * abs(threshold))
                    assert all(abs(r.values[spec.key] - threshold) >= margin for r in records), raw
                    text = inst.renderings[0].question_text[q.qid]
                    assert f"{fmt_num(step['value'])} {unit.words}" in text  # stated exactly as the chain has it
                elif step.kind == "time_between":
                    assert step["start"] % ctx.time_step == 0 == step["end"] % ctx.time_step
                    assert all(abs(r.time - edge) >= ctx.time_margin for r in records
                               for edge in (step["start"], step["end"]))
            term = Step(terminal_step(q.chain), bound.units)
            if term.kind in ("argmax", "rank", "time_of_max"):  # no near-tie down to the asked rank
                top = term["k"] + 1 if term.kind == "rank" else 2
                rows = rows_after_filters(q.chain, records, units=bound.units)
                values = sorted((r.values[term.field] for r in rows), reverse=True)[:top]
                gap = bound.field(term.field).rank_gap
                assert gap == 2.0 and all(a - b >= gap - 1e-9 for a, b in zip(values, values[1:]))
            for rendering in inst.renderings:
                text = rendering.question_text[q.qid]
                assert bound.plural in text and "{" not in text and not re.search(r"\d[eE][-+]?\d", text)
    assert len(fingerprints) == len(set(fingerprints))  # no question is asked twice in a run
    assert not set(templates) & {"P02", "P07", "P09", "P12", "N02", "N07", "N09", "N12"}
    assert any(t.startswith("P") for t in templates) and any(t.startswith("N") for t in templates)


def test_traps_are_planted_and_their_careless_readings_give_their_wrong_answers(worlds):
    forms = Counter()
    for inst, bound, records in worlds:
        data = DigestData(inst.reports, inst.notices)
        vocab = bound.trap_vocab
        # (a digest off the legacy path holds none to three retractions: traps.RETRACTION_WEIGHTS)
        assert {"duplicate", "conflicting_value"} <= {t.kind for t in inst.traps} <= {"duplicate", "retraction",
                                                                                   "conflicting_value"}
        for trap in inst.traps:
            forms[(trap.kind, trap.form)] += 1
            planted = [r for r in inst.reports if r.trap_id == trap.trap_id]
            readings = [Mishandle.from_dict(m) for m in trap.mishandles]
            if trap.kind == "duplicate":
                (dup,) = planted
                # Filed by the agency of a lot drawn with the digest but not kept: an agency that files lots.
                assert dup.status == "provisional" and dup.source in bound.surface.sources.values()
                assert (dup.same_as is None) == (trap.form == "same_id")
                assert readings == [Mishandle(split=((dup.ref, dup.source, "provisional"),))]
            elif trap.kind == "retraction":
                (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                assert notice.reason in vocab.retraction_reasons and planted[0].role == "phantom"
                assert readings == [Mishandle(keep_retracted=(notice.target,))] and notice.target == planted[0].ref
            elif trap.form == "correction":
                (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                assert notice.field in ("f1", "f2") and notice.qualifier is None
                assert planted[0].values[notice.field] == notice.old_value != notice.new_value
                assert readings == [Mishandle(ignore_correction_for=(trap.target,))]
            else:
                (second,) = planted
                assert (second.ref, second.status) == (trap.target, "provisional")
                assert readings == [Mishandle(prefer_preliminary_for=(trap.target,)),
                                    Mishandle(split=((trap.target, second.source, "provisional"),))]
        everything = Mishandle.merged(t.mishandles[-1] for t in inst.traps)  # every trap misread at once ...
        for q in inst.questions:  # ... is what the naive reader does
            assert (execute(q.chain, data, mishandle=everything, units=bound.units, resolver=bound.resolve)
                    == execute(q.chain, data, skip=("dedupe",), units=bound.units, resolver=bound.resolve))
            assert execute(q.chain, data, units=bound.units, resolver=bound.resolve) == inst.answer_key[q.qid]
        for trap in inst.traps:
            for qid, fallen in trap.affected.items():  # each careless reading reproduces its wrong answer
                q = inst.question(qid)
                misread = [execute(q.chain, data, mishandle=Mishandle.from_dict(m), units=bound.units,
                                   resolver=bound.resolve) for m in trap.mishandles]
                assert fallen in misread and not same_answer(q.answer_type, fallen, inst.answer_key[qid], unit=q.unit)
    assert {kind for kind, _ in forms} == {"duplicate", "retraction", "conflicting_value"}
    assert forms[("duplicate", "same_id")] and forms[("duplicate", "cross_ref")] and len(forms) == 5


def test_the_oracle_scores_100_percent_and_the_naive_reader_falls_only_for_traps(run, worlds):
    rows = read_json(run / "scores" / "oracle.json")["rows"]
    assert len(rows) == 3 * sum(len(inst.questions) for inst, _, _ in worlds) and all(r["correct"] for r in rows)
    assert {(row["domain"], row["genre"]) for row in rows} == {("invented", "classic")}
    models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
    assert models["oracle"]["traps"]["caught"]["rate"] == 1.0 == models["oracle"]["consistency"]["rate"]
    naive = models["naive"]["traps"]
    assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0
    assert models["naive"]["accuracy"]["rate"] < 0.8 and 0 < models["mock"]["accuracy"]["rate"] < 1
    trapped = {(inst.instance_id, qid) for inst, _, _ in worlds for t in inst.traps for qid in t.fallen}
    for row in read_json(run / "scores" / "naive.json")["rows"]:
        assert row["correct"] or (row["instance_id"], row["qid"]) in trapped, row
    for inst, _, _ in worlds:  # the text alone answers every rendering, read through its own glossary
        for rendering in inst.renderings:
            assert solve(rendering.prompt, question_specs(inst), domain="invented") == inst.answer_key
        unglossed = re.sub(r"<glossary>.*?</glossary>\n\n", "", inst.renderings[0].prompt, flags=re.DOTALL)
        with pytest.raises(ValueError, match="prints none"):  # and nothing but the glossary defines the words
            solve(unglossed, question_specs(inst), domain="invented")
    for inst, _, _ in worlds[:4]:  # the simulated model's mistakes are real mistakes in any drawn unit
        task = Task(inst.instance_id, "v1", question_specs(inst), hidden=inst, domain="invented", genre="classic")
        for q in question_specs(inst):
            for seed in range(10):
                wrong = _perturb(q, inst.answer_key[q["qid"]], random.Random(seed), task)
                assert not is_correct(q["answer_type"], inst.answer_key[q["qid"]], wrong, unit=q["unit"]), (q, wrong)


def test_metamorphic_variants_restate_thresholds_in_other_units_without_changing_the_key(worlds):
    restated = 0
    for inst, bound, records in worlds:
        original = inst.renderings[0]
        for variant in inst.renderings[1:]:
            assert variant.style != original.style and variant.digest != original.digest
            assert variant.prompt_sha256 != original.prompt_sha256 and variant.genre == "classic"
            for q in inst.questions:
                text = variant.question_text[q.qid]
                for i, raw in enumerate(q.chain):
                    step = Step(raw, bound.units)
                    if step.kind not in ("at_least", "below"):
                        continue
                    units = bound.field(step.field).units
                    unit = find_unit(units, step["unit"])
                    other = units[(units.index(unit) + 1) % len(units)]
                    value = other.from_canonical(unit.to_canonical(step["value"]))
                    shown = f"{value:.{other.places(bound.field(step.field).decimals)}f}"
                    if f"{shown} {other.words}" not in text:
                        assert f"{fmt_num(step['value'])} {unit.words}" in text  # stated as the original states it
                        continue
                    restated += 1  # the same threshold in the next unit: the same records pass, the same key
                    chain = [*q.chain[:i], {**raw, "value": float(shown), "unit": other.symbol}, *q.chain[i + 1:]]
                    assert rows_after_filters(chain, records, units=bound.units) == rows_after_filters(
                        q.chain, records, units=bound.units)
                    assert execute(chain, records, units=bound.units) == inst.answer_key[q.qid]
    assert restated >= 5


def test_two_runs_from_one_seed_are_identical_and_another_seed_words_them_anew(tmp_path):
    def generate(seed: int, name: str) -> Path:
        return pipeline.generate(GenerateConfig(seed=seed, instances=3, mix="invented:classic"), tmp_path / name,
                                 log=quiet)

    a, b, c = generate(11, "a"), generate(11, "b"), generate(12, "c")
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file() and p.name != "manifest.json")
    assert files and files == sorted(p.relative_to(b).as_posix() for p in b.rglob("*")
                                     if p.is_file() and p.name != "manifest.json")
    for name in files:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    manifests = [{k: v for k, v in read_json(run / "manifest.json").items() if k not in VOLATILE} for run in (a, b)]
    assert manifests[0] == manifests[1]
    surfaces_a = [inst.surface for inst in load_instances(a)]
    surfaces_c = [inst.surface for inst in load_instances(c)]
    assert not {json.dumps(s, sort_keys=True) for s in surfaces_a} & {json.dumps(s, sort_keys=True) for s in surfaces_c}
