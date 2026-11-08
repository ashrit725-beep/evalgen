"""The domain interface: generic records, field-generic operations with the 0.2.0 names as aliases,
the answer-type registry and the domain registry. The earthquake path must read, run, describe and
serialize exactly as in 0.2.0."""
from __future__ import annotations

import copy
import importlib
import json
import pkgutil
import sys
from dataclasses import asdict, fields

import pytest

from evalgen import domains, scoring
from evalgen.domains import DEFAULT_DOMAIN, Registry, all_domains
from evalgen.domains.base import KM, LENGTH, MI, Domain, FieldSpec, Record, RecordReport, Unit
from evalgen.geo import from_km, to_km
from evalgen.models import Instance, Notice
from evalgen.ops import (ALIASES, FILTER_OPS, KINDS, TERMINAL_OPS, answer_spec, describe, execute, passes,
                         rows_after_filters)
from evalgen.rules import Mishandle, report_to_event
from evalgen.scoring import AnswerType, is_correct, normalize, register_answer_type
from evalgen.solver import parse_digest
from evalgen.util import utc_iso, write_json

# The generic form of every 0.2.0 name, written out here independently of ops.ALIASES.
GENERIC = {
    "mag_at_least": ("at_least", "mag"),
    "mag_below": ("below", "mag"),
    "depth_shallower_than": ("below", "depth"),
    "depth_at_least": ("at_least", "depth"),
    "in_region": ("in_group", None),
    "argmax_mag": ("argmax", "mag"),
    "rank_mag": ("rank", "mag"),
    "max_mag": ("max", "mag"),
    "max_depth": ("max", "depth"),
    "min_depth": ("min", "depth"),
    "compare_regions": ("compare_groups", None),
    "time_of_largest": ("time_of_max", "mag"),
}
ANSWER_TYPES_020 = {  # ops.ANSWER_TYPES in 0.2.0
    "count": "count", "argmax_mag": "event_id", "rank_mag": "event_id", "nearest_to": "event_id",
    "max_mag": "magnitude", "max_depth": "depth", "min_depth": "depth", "nearest_distance": "distance",
    "compare_regions": "region", "time_of_largest": "utc_time",
}


def generic(step: dict) -> dict:
    """A step in its generic form: kind and field instead of a 0.2.0 name, 'group' for 'region'."""
    if step["op"] not in GENERIC:
        return dict(step)
    kind, field = GENERIC[step["op"]]
    out = {"op": kind, **({"field": field} if field else {})}
    out.update(("group" if key == "region" else key, value) for key, value in step.items() if key != "op")
    return out


class QuakeLike(Domain):
    """A stand-in domain with the 0.2.0 earthquake vocabulary, to type and convert generic chains."""

    name = "quake-like"
    fields = (
        FieldSpec(key="mag", label="magnitude", canonical_unit=None, units=(), decimals=1,
                  answer_type="magnitude", tolerance=(0.05, 0.0), margin=(0.0, 0.0), rank_gap=0.1,
                  quantiles=(0.25, 0.85), threshold_step=0.1, filters=("at_least", "below"),
                  terminals=("argmax", "rank", "max", "time_of_max"), qualifier="mag_type", correctable=True),
        FieldSpec(key="depth", label="depth", canonical_unit="km", units=LENGTH, decimals=1, answer_type="depth",
                  tolerance=(0.3, 0.02), margin=(0.6, 0.03), rank_gap=0.0, quantiles=(0.25, 0.8),
                  threshold_step=0.1, filters=("at_least", "below"), terminals=("max", "min"), signed=True),
    )
    id_answer_type = "event_id"
    group_answer_type = "region"


QUAKES = QuakeLike()


def coverage_chains(events) -> list[list[dict]]:
    """Every 0.2.0 filter (and none) before every 0.2.0 terminal, with parameters taken from the events."""
    mags = sorted(e.mag for e in events)
    depth = sorted(e.depth_km for e in events)[len(events) // 2]
    regions = sorted({e.region for e in events})
    anchor = events[len(events) // 2]
    point = {"lat": round(anchor.lat + 0.05, 3), "lon": round(anchor.lon - 0.05, 3), "label": "station T-1"}
    filters = [
        [],
        [{"op": "mag_at_least", "value": mags[len(mags) // 2]}],
        [{"op": "mag_below", "value": mags[len(mags) // 2]}],
        [{"op": "depth_shallower_than", "value": round(depth, 1), "unit": "km"}],
        [{"op": "depth_shallower_than", "value": round(depth / 1.609344, 1), "unit": "mi"}],
        [{"op": "depth_at_least", "value": float(round(depth)), "unit": "km"}],
        [{"op": "depth_at_least", "value": round(depth / 1.609344, 1), "unit": "mi"}],
        [{"op": "within_distance", **point, "radius": 150, "unit": "km"}],
        [{"op": "within_distance", **point, "radius": 100, "unit": "mi"}],
        [{"op": "time_between", "start": events[len(events) // 4].time, "end": events[3 * len(events) // 4].time}],
        [{"op": "in_region", "region": regions[0]}],
        [{"op": "mag_at_least", "value": mags[1]}, {"op": "in_region", "region": regions[-1]}],
    ]
    terminals = [
        [{"op": "count"}],
        [{"op": "argmax_mag"}],
        [{"op": "rank_mag", "k": 2}],
        [{"op": "rank_mag", "k": 3}],
        [{"op": "max_mag"}],
        [{"op": "max_depth"}],
        [{"op": "max_depth"}, {"op": "convert", "to": "mi"}, {"op": "round", "ndigits": 1}],
        [{"op": "min_depth"}, {"op": "convert", "to": "km"}, {"op": "round", "ndigits": 1}],
        [{"op": "nearest_to", **point}],
        [{"op": "nearest_distance", **point}, {"op": "convert", "to": "mi"}, {"op": "round", "ndigits": 1}],
        [{"op": "compare_regions", "a": regions[0], "b": regions[-1]}],
        [{"op": "time_of_largest"}],
    ]
    return [[{"op": "dedupe"}, *f, *t] for f in filters for t in terminals]


def chains_for(inst, snapshot_events) -> tuple[list, list[list[dict]]]:
    """The instance's events, and its own question chains plus the coverage chains."""
    events = [snapshot_events[eid] for eid in inst.event_ids]
    return events, [q.chain for q in inst.questions] + coverage_chains(events)


def describe_020(step: dict) -> str:
    """ops.describe_step exactly as it was in 0.2.0."""
    op = step["op"]
    if op == "mag_at_least":
        return f"mag ≥ {step['value']:g}"
    if op == "mag_below":
        return f"mag < {step['value']:g}"
    if op == "depth_shallower_than":
        return f"depth < {step['value']:g} {step['unit']}"
    if op == "depth_at_least":
        return f"depth ≥ {step['value']:g} {step['unit']}"
    if op == "within_distance":
        return f"within {step['radius']:g} {step['unit']} of ({step['lat']:.3f}, {step['lon']:.3f})"
    if op == "time_between":
        return f"time in [{utc_iso(step['start'])}, {utc_iso(step['end'])})"
    if op == "in_region":
        return f"region = {step['region']}"
    if op == "rank_mag":
        return f"rank_mag(k={step['k']})"
    if op in ("nearest_to", "nearest_distance"):
        return f"{op}({step['lat']:.3f}, {step['lon']:.3f})"
    if op == "compare_regions":
        return f"compare_regions({step['a']} vs {step['b']})"
    if op == "convert":
        return f"convert → {step['to']}"
    if op == "round":
        return f"round({step['ndigits']})"
    return op


# ------------------------------------------------------------------- aliases and generic kinds


def test_every_020_name_resolves_to_a_generic_kind():
    assert {op: (alias.kind, alias.field) for op, alias in ALIASES.items()} == GENERIC
    for op in ("dedupe", *FILTER_OPS, *TERMINAL_OPS, "convert", "round"):
        assert GENERIC.get(op, (op,))[0] in KINDS, op


def test_coverage_chains_use_every_020_name(instances, snapshot_events):
    events, chains = chains_for(instances[0], snapshot_events)
    assert {s["op"] for chain in chains for s in chain} == {"dedupe", *FILTER_OPS, *TERMINAL_OPS, "convert", "round"}


def test_alias_chains_equal_their_generic_form_on_snapshot_events(instances, snapshot_events):
    for inst in instances:
        events, chains = chains_for(inst, snapshot_events)
        for chain in chains:
            stored = json.dumps(chain)
            twin = [generic(s) for s in chain]
            assert execute(twin, events, units=QUAKES.units) == execute(chain, events), describe(chain)
            assert rows_after_filters(twin, events, units=QUAKES.units) == rows_after_filters(chain, events)
            assert answer_spec(twin, QUAKES) == answer_spec(chain)
            assert json.dumps(chain) == stored  # aliases are resolved on reading, never written back


def test_alias_chains_equal_their_generic_form_on_parsed_digests(instances, snapshot_events):
    for inst in instances:
        _, chains = chains_for(inst, snapshot_events)
        for rendering in inst.renderings:  # the oracle's and the naive reader's view of every rendering
            data = parse_digest(rendering.digest)
            for chain in chains:
                twin = [generic(s) for s in chain]
                for skip in ((), ("dedupe",)):
                    assert (execute(twin, data, skip=skip, units=QUAKES.units)
                            == execute(chain, data, skip=skip)), (rendering.variant, describe(chain), skip)
        data = parse_digest(inst.renderings[0].digest)
        readings = [Mishandle.from_dict(m) for trap in inst.traps for m in trap.mishandles]
        for chain in chains:  # every careless reading the traps record
            twin = [generic(s) for s in chain]
            for mishandle in readings:
                assert (execute(twin, data, mishandle=mishandle, units=QUAKES.units)
                        == execute(chain, data, mishandle=mishandle)), (describe(chain), mishandle)


def test_alias_terminals_keep_their_020_answer_types():
    for op, answer_type in ANSWER_TYPES_020.items():
        assert answer_spec([{"op": "dedupe"}, {"op": op}]) == (answer_type, None), op
    assert answer_spec([{"op": "dedupe"}, {"op": "min_depth"}, {"op": "convert", "to": "mi"},
                        {"op": "round", "ndigits": 1}]) == ("depth", "mi")


def test_filters_read_reports_like_the_events_they_describe(instances, snapshot_events):
    for inst in instances:
        _, chains = chains_for(inst, snapshot_events)
        steps = {json.dumps(s): s for chain in chains for s in chain if s["op"] in FILTER_OPS}.values()
        for report in inst.reports:
            event = report_to_event(report, report.ref)
            for step in steps:
                assert passes(step, report) == passes(step, event) == passes(generic(step), report, QUAKES.units)


def test_events_and_reports_answer_to_generic_names_outside_their_json(instances):
    report = instances[0].reports[0]
    event = report_to_event(report, report.ref)
    for record in (report, event):
        assert record.values == {"mag": record.mag, "depth": record.depth_km}
        assert (record.group, record.source) == (record.region, record.network)
        assert list(asdict(record)) == [f.name for f in fields(record)]
        assert not {"values", "group", "source"} & set(asdict(record))


# ------------------------------------------------------------------------------ descriptions


def test_alias_chains_describe_exactly_as_in_020(instances, snapshot_events):
    for inst in instances:
        _, chains = chains_for(inst, snapshot_events)
        for chain in chains:
            assert describe(chain) == " → ".join(describe_020(s) for s in chain)


def test_generic_steps_name_their_field():
    chain = [{"op": "dedupe"}, {"op": "mag_at_least", "value": 3.1},
             {"op": "depth_shallower_than", "value": 12.0, "unit": "mi"}, {"op": "in_region", "region": "Nevada"},
             {"op": "rank_mag", "k": 2}]
    assert describe(chain) == "dedupe → mag ≥ 3.1 → depth < 12 mi → region = Nevada → rank_mag(k=2)"
    assert (describe([generic(s) for s in chain])
            == "dedupe → mag ≥ 3.1 → depth < 12 mi → group = Nevada → rank(mag, k=2)")
    terminals = [{"op": "time_of_largest"}, {"op": "compare_regions", "a": "A", "b": "B"}]
    assert describe([generic(s) for s in terminals]) == "time_of_max(mag) → compare_groups(A vs B)"


# ---------------------------------------------------------------------- generic records


TEMPERATURE = (Unit("°C", "degrees Celsius"), Unit("°F", "degrees Fahrenheit", 5 / 9, -160 / 9))
T0 = 1_780_000_000


class Weatherish(Domain):
    """A small generic domain: one signed field with an affine unit, and its own ID and group types."""

    name = "weatherish"
    fields = (FieldSpec(key="temp", label="temperature", canonical_unit="°C", units=TEMPERATURE, decimals=1,
                        answer_type="temperature", tolerance=(0.5, 0.0), margin=(0.3, 0.0), rank_gap=0.5,
                        quantiles=(0.2, 0.8), threshold_step=0.5, filters=("at_least", "below"),
                        terminals=("argmax", "rank", "max", "min", "time_of_max"), signed=True),)
    id_answer_type = "observation_id"
    group_answer_type = "us_region"


WEATHER = Weatherish()
RECORDS = [
    Record("ksfo-0001", T0, 37.619, -122.375, "West", "nws", {"temp": 21.0}),
    Record("ksea-0002", T0 + 3600, 47.449, -122.309, "West", "nws", {"temp": 12.5}),
    Record("kbos-0003", T0 + 7200, 42.361, -71.010, "Northeast", "nws", {"temp": 30.0}),
]


def run(*steps, rows=RECORDS):
    return execute([{"op": "dedupe"}, *steps], rows, units=WEATHER.units)


def test_generic_kinds_run_on_generic_records_with_affine_units():
    warm = {"op": "at_least", "field": "temp", "value": 70, "unit": "°F"}  # 21.1 °C
    assert run(warm, {"op": "count"}) == 1
    assert run({"op": "below", "field": "temp", "value": 21.0}, {"op": "count"}) == 1  # no unit: canonical °C
    assert run({"op": "max", "field": "temp"}, {"op": "convert", "to": "°F"}, {"op": "round", "ndigits": 1}) == 86.0
    assert run({"op": "in_group", "group": "West"}, {"op": "argmax", "field": "temp"}) == "ksfo-0001"
    assert run({"op": "rank", "field": "temp", "k": 3}) == "ksea-0002"
    assert run({"op": "time_of_max", "field": "temp"}) == "22:26"  # T0 + 2 h is 22:26:40 UTC
    assert run({"op": "compare_groups", "a": "West", "b": "Northeast"}) == "West"
    assert run({"op": "nearest_to", "lat": 42.0, "lon": -71.0, "label": "x"}) == "kbos-0003"
    reports = [RecordReport(r.id, r.source, "reviewed", r.time, r.lat, r.lon, r.group, dict(r.values)) for r in RECORDS]
    assert [passes(warm, r, WEATHER.units) for r in reports] == [False, False, True]


def test_generic_answer_types_come_from_the_domain():
    temp_max = [{"op": "dedupe"}, {"op": "max", "field": "temp"}, {"op": "convert", "to": "°F"},
                {"op": "round", "ndigits": 1}]
    assert answer_spec(temp_max, WEATHER) == ("temperature", "°F")
    assert answer_spec([{"op": "dedupe"}, {"op": "argmax", "field": "temp"}], WEATHER) == ("observation_id", None)
    assert answer_spec([{"op": "dedupe"}, {"op": "compare_groups", "a": "A", "b": "B"}], WEATHER) == ("us_region", None)
    assert answer_spec([{"op": "dedupe"}, {"op": "count"}], WEATHER) == ("count", None)
    with pytest.raises(ValueError, match="comes from its domain"):
        answer_spec(temp_max)


def test_a_generic_unit_needs_the_domains_units():
    with pytest.raises(ValueError, match="units of field 'temp' are unknown"):
        execute([{"op": "dedupe"}, {"op": "at_least", "field": "temp", "value": 70, "unit": "°F"}, {"op": "count"}],
                RECORDS)


def test_units_map_affinely_and_the_length_units_match_geo_to_the_bit():
    fahrenheit = TEMPERATURE[1]
    assert fahrenheit.to_canonical(212) == pytest.approx(100) and fahrenheit.to_canonical(32) == pytest.approx(0)
    assert fahrenheit.from_canonical(fahrenheit.to_canonical(70.5)) == pytest.approx(70.5)
    for x in (0.0, -0.0, 0.4, -0.7, 6.2, 12.0, 35.0, 287.3, 241.4016, 1e-12):
        for unit in (KM, MI):
            assert unit.to_canonical(x) == to_km(x, unit.symbol)  # thresholds are only compared
            assert repr(unit.from_canonical(x)) == repr(from_km(x, unit.symbol))  # answers are written


# ----------------------------------------------------------------------- instance JSON


def test_legacy_instance_json_round_trips_byte_identically(small_run, tmp_path):
    for path in sorted((small_run / "instances").glob("inst-*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        assert not set(Instance.OPTIONAL_KEYS) & set(d)
        assert not any(set(Notice.OPTIONAL_KEYS) & set(n) for n in d["notices"])
        assert not any("genre" in r for r in d["renderings"])
        write_json(tmp_path / path.name, Instance.from_dict(d).to_dict())
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_optional_fields_are_written_only_when_set(instances):
    inst = copy.deepcopy(instances[0])
    inst.domain, inst.split, inst.surface = "weather", "held_out", {"noun": "observation"}
    inst.renderings[0].genre = "news"
    inst.notices.append(Notice(kind="correction", target="ksfo-0001", time=inst.compiled_at, field="temp",
                               new_value=21.5, old_value=25.0, qualifier="sensor B"))
    d = inst.to_dict()
    assert list(d)[-4:] == ["forecast", "domain", "split", "surface"]
    assert d["renderings"][0]["genre"] == "news" and "genre" not in d["renderings"][1]
    assert list(d["notices"][-1])[-4:] == list(Notice.OPTIONAL_KEYS)
    assert not set(Notice.OPTIONAL_KEYS) & set(d["notices"][0])
    assert Instance.from_dict(json.loads(json.dumps(d))) == inst


# ------------------------------------------------------------------------ answer types


@pytest.fixture
def answer_registry(monkeypatch):
    """Answer types registered in a test stay in that test."""
    monkeypatch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
    monkeypatch.setattr(scoring, "TOLERANCE", dict(scoring.TOLERANCE))


def test_020_answer_types_and_tolerances_are_unchanged():
    kinds = {"count": "count", "event_id": "id", "magnitude": "number", "depth": "number", "distance": "number",
             "region": "group", "utc_time": "clock", "probability": "probability"}
    assert {name: scoring.ANSWER_TYPES[name].kind for name in kinds} == kinds
    tolerances = {"magnitude": (0.05, 0.0), "depth": (0.3, 0.02), "distance": (1.5, 0.03)}
    assert {name: scoring.TOLERANCE[name] for name in tolerances} == tolerances
    assert scoring.ANSWER_TYPES["utc_time"].tolerance[0] == scoring.TIME_TOLERANCE_MIN == 1


def test_domains_can_register_their_own_answer_types(answer_registry):
    register_answer_type(AnswerType("temperature", "number", (0.5, 0.0)))
    assert scoring.TOLERANCE["temperature"] == (0.5, 0.0)
    assert normalize("temperature", "21.30 °C") == 21.3
    assert is_correct("temperature", 21.3, "21.7") and not is_correct("temperature", 21.3, "21.9")
    register_answer_type(AnswerType("observation_id", "id", id_pattern=r"k[a-z]{3}-\d{4}"))
    assert normalize("observation_id", "Observation KSFO-0001.") == "ksfo-0001"
    assert normalize("observation_id", "ksfo-0001 or kbos-0003") is None
    register_answer_type(AnswerType("us_region", "group", suffix="states"))
    assert normalize("us_region", "The Northeast states") == "northeast"
    assert normalize("us_region", "equal") == "tie"


def test_a_type_can_parse_its_own_numbers(answer_registry):
    def scientific(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    register_answer_type(AnswerType("miss_distance", "number", (0.0001, 0.01), parse=scientific))
    assert normalize("miss_distance", "1.2e-3") == 0.0012
    assert normalize("magnitude", "1.2e-3") is None  # the default parser sees two numbers there


def test_registering_is_idempotent_and_conflicts_are_errors(answer_registry):
    temperature = AnswerType("temperature", "number", (0.5, 0.0))
    assert register_answer_type(temperature) is register_answer_type(temperature)
    with pytest.raises(ValueError, match="already registered differently"):
        register_answer_type(AnswerType("temperature", "number", (1.0, 0.0)))
    with pytest.raises(ValueError, match="already registered differently"):
        register_answer_type(AnswerType("depth", "number", (0.1, 0.0)))
    with pytest.raises(ValueError, match="unknown answer kind"):
        register_answer_type(AnswerType("colour", "colour"))


# ---------------------------------------------------------------------------- registry

TOY_DOMAIN = """\
from evalgen.domains.base import Domain


class Toy(Domain):
    name = {name!r}


DOMAIN = Toy()
"""
PLUGIN = """\
from __future__ import annotations

from dataclasses import dataclass

from evalgen.domains.base import Domain


@dataclass(frozen=True)
class Depot:  # a dataclass needs its module registered while the plugin runs
    name: str


class Consignments(Domain):
    name = {name!r}
    depots = (Depot("north"), Depot("south"))


DOMAIN = Consignments()
"""


@pytest.fixture
def toy_package(tmp_path, monkeypatch):
    """A package laid out like evalgen.domains, whose module order differs from its domain names."""
    root = tmp_path / "toy_domains"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for module, name in (("zeta", "zulu"), ("alpha", "yankee"), ("quakes", "earthquakes")):
        (root / f"{module}.py").write_text(TOY_DOMAIN.format(name=name))
    (root / "base.py").write_text("")  # the interface module, not a domain
    (root / "_shared.py").write_text("")  # private helpers, not a domain
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "toy_domains"
    for module in [m for m in sys.modules if m == "toy_domains" or m.startswith("toy_domains.")]:
        del sys.modules[module]


def test_registry_discovers_domain_modules_in_sorted_order(toy_package):
    registry = Registry(toy_package)
    assert [d.name for d in registry.all()] == ["yankee", "earthquakes", "zulu"]  # alpha, quakes, zeta
    assert registry.get() is registry.get(DEFAULT_DOMAIN) is registry.get("earthquakes")
    with pytest.raises(KeyError, match="unknown domain 'weather'"):
        registry.get("weather")


def test_plugins_load_from_files_outside_the_package(tmp_path, toy_package):
    path = tmp_path / "private" / "consignments.py"
    path.parent.mkdir()
    path.write_text(PLUGIN.format(name="consignments"))
    registry = Registry(toy_package)
    domain = registry.load_plugin(path)
    assert domain.name == "consignments" and registry.get("consignments") is domain
    assert registry.load_plugin(path) is domain  # loading the same file again is harmless
    assert [d.name for d in registry.all()] == ["yankee", "earthquakes", "zulu", "consignments"]


def test_a_plugin_cannot_take_a_built_in_name(tmp_path, toy_package):
    path = tmp_path / "shadow.py"
    path.write_text(PLUGIN.format(name="earthquakes"))
    with pytest.raises(ValueError, match="two domains are named 'earthquakes'"):
        Registry(toy_package).load_plugin(path)


def test_a_plugin_must_define_a_domain(tmp_path, toy_package):
    path = tmp_path / "empty.py"
    path.write_text("ANSWER = 42\n")
    with pytest.raises(TypeError, match="must define DOMAIN"):
        Registry(toy_package).load_plugin(path)


def test_the_evalgen_registry_lists_its_modules_in_order():
    names = sorted(m.name for m in pkgutil.iter_modules(domains.__path__)
                   if m.name != "base" and not m.name.startswith("_"))
    assert all_domains() == [importlib.import_module(f"evalgen.domains.{name}").DOMAIN for name in names]
    assert DEFAULT_DOMAIN == "earthquakes"
