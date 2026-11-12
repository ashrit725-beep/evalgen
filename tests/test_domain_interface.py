"""The domain interface: generic records, field-generic operations with the 0.2.0 names as aliases,
the answer-type registry and the domain registry. The earthquake path must read, run, describe and
serialize exactly as in 0.2.0."""
from __future__ import annotations

import copy
import importlib
import json
import os
import pkgutil
import random
import re
import subprocess
import sys
from dataclasses import asdict, fields, replace
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from evalgen import domains, scoring
from evalgen.domains import DEFAULT_DOMAIN, Registry, all_domains
from evalgen.domains import invented, neo, weather
from evalgen.domains.base import Domain, FieldSpec, GlossaryField, Record, RecordReport
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.geo import from_km, to_km
from evalgen.models import Instance, Notice
from evalgen.ops import (ALIASES, FILTER_OPS, KINDS, TERMINAL_OPS, answer_spec, describe, describe_step, execute,
                         passes, rows_after_filters)
from evalgen.questions import _answer_steps, _quantity, fmt_num
from evalgen.rules import Mishandle, report_to_event
from evalgen.scoring import AnswerType, equivalent, is_correct, normalize, register_answer_type
from evalgen.solver import parse_digest
from evalgen.units import KM, LENGTH, MI, Unit, decode, encode, exact_canonical
from evalgen.util import plain_number, utc_iso, write_json

from generic_quakes import GenericQuakes

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


# ------------------------------------------------------------ printed values and precision


def test_printed_values_read_back_exactly_in_every_unit():
    fahrenheit = TEMPERATURE[1]
    # Converting back is not exact: -30.7 °C has no decimal form in °F that converts to exactly -30.7.
    assert all(fahrenheit.to_canonical(round(fahrenheit.from_canonical(-30.7), d)) != -30.7 for d in range(16))
    temp, depth, mag = WEATHER.field("temp"), EARTHQUAKES.field("depth"), EARTHQUAKES.field("mag")
    printed = 0
    for spec, values in ((temp, [t / 10 for t in range(-400, 451)]), (depth, [t / 10 for t in range(-30, 7000, 3)])):
        glossary_field = GlossaryField(spec.key, spec.label, spec.canonical_unit, spec.units, spec.decimals)
        for unit in spec.units:
            for value in values:
                text = spec.encode(value, unit)
                assert spec.decode(text, unit) == glossary_field.decode(text, unit) == value
                assert len(text.partition(".")[2]) <= 3, (value, unit.symbol, text)
                printed += 1
    assert printed == 2 * (851 + 2344)
    assert (temp.encode(-30.7, fahrenheit), temp.encode(21.0, fahrenheit), temp.encode(21.0)) == ("-23.3", "69.8", "21")
    assert (mag.encode(4.27), mag.encode(4.0), mag.decode(4.27)) == ("4.27", "4", 4.27)  # unitless: as stored
    with pytest.raises(ValueError, match="stored to 1 decimal places"):
        temp.encode(21.05, fahrenheit)  # more places than the field stores: no printed form reads back


def glossary_reading(text: str, unit: Unit) -> Decimal:
    """A printed number converted exactly as the glossary states the conversion, with pencil and paper."""
    offset = Decimal(plain_number(abs(unit.offset))) * (-1 if unit.offset < 0 else 1)
    return Decimal(text) * Decimal(plain_number(unit.scale)) + offset


def test_no_printed_value_converts_to_halfway_between_two_stored_ones():
    """A reader who converts a printed number exactly as the glossary says, in decimal arithmetic, gets a
    value strictly nearer to the stored one than to any other, so it rounds back to it whether halves go
    up or to even: the codec never writes 273.9 K (exactly 0.75 °C) for a stored 0.7 °C, nor 37.5 kt
    (exactly 69.45 km/h), 0.85625 LD (exactly 329,142.5 km) or 33 units of a quarter (8.25)."""
    temp, wind, kelvin, knot = weather.TEMPERATURE, weather.WIND, weather.KELVIN, weather.KNOT
    lunar = neo.LUNAR_DISTANCE
    assert (encode(0.7, 1, kelvin), decode("273.85", 1, kelvin), decode("273.9", 1, kelvin)) == ("273.85", 0.7, 0.8)
    assert encode(69.4, 1, knot) == "37.47" and encode(329_142.0, 0, lunar) == "0.856249"
    assert exact_canonical("273.9", kelvin) == Decimal("0.75")  # exactly halfway: never printed
    quarter, twenty_five = Unit("hk", "hecks", 0.25), Unit("gn", "gnats", 25.0)
    assert encode(8.2, 1, quarter) == "32.8" and encode(36.2, 1, twenty_five) == "1.448"
    cases = [(temp, unit, [t / 10 for t in range(-450, 500)]) for unit in temp.units[1:]]
    cases += [(wind, knot, [t / 10 for t in range(0, 1200)]),
              (neo.MISS_DISTANCE, lunar, [float(d) for d in range(329_000, 331_000)])]
    cases += [(invented.SECONDARY, unit, [t / 10 for t in range(25, 2500)]) for unit in (quarter, twenty_five)]
    checked = 0
    for spec, unit, values in cases:
        step = Decimal(1).scaleb(-spec.decimals)
        for value in values:
            text = encode(value, spec.decimals, unit)
            exact, stored = glossary_reading(text, unit), Decimal(repr(value))
            assert abs(exact - stored) < step / 2, (value, unit.symbol, text)  # never on a half, never past it
            for rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                assert exact.quantize(step, rounding=rounding) == stored.quantize(step), (value, text, rounding)
            assert decode(text, spec.decimals, unit) == value
            checked += 1
    assert checked == 2 * 950 + 1200 + 2000 + 2 * 2475
    # Answers asked in kelvins keep two places: a tenth of a degree Celsius ends in 5 hundredths of a kelvin.
    assert kelvin.places(temp.decimals) == 2
    # An invented quantity's relative tolerance covers half a stored step at its floor, the most by which a
    # printed value read without rounding can miss the stored one.
    for spec in invented.INVENTED.fields:
        assert spec.tolerance[1] * invented.FLOORS[spec.key] >= 0.5 * 10 ** -spec.decimals


def test_stored_earthquake_values_have_the_places_their_fields_print(instances, snapshot_events):
    records = [*snapshot_events.values(), *(r for inst in instances for r in inst.reports)]
    for spec in EARTHQUAKES.fields:
        assert all(round(r.values[spec.key], spec.decimals) == r.values[spec.key] for r in records), spec.key


PASCALS = Unit("hPa", "hectopascals")
INCHES = Unit("inHg", "inches of mercury", 33.8639, decimals=2, tolerance=(0.02, 0.0))


class Barometers(GenericQuakes):
    """Earthquake records with their depth read as a pressure: a unit 34 times the canonical one sets
    its own places and tolerance."""

    name = "barometers"
    fields = (EARTHQUAKES.field("mag"),
              replace(EARTHQUAKES.field("depth"), label="pressure", canonical_unit="hPa", units=(PASCALS, INCHES),
                      answer_type="pressure", tolerance=(0.5, 0.0), margin=(1.0, 0.0)))


def test_a_unit_on_another_scale_sets_its_own_places_and_tolerance(answer_registry):
    domain = Barometers()
    domain.validate()
    pressure = next(t for t in domain.answer_types() if t.name == "pressure")
    assert (pressure.tolerance, dict(pressure.unit_tolerance)) == ((0.5, 0.0), {"inHg": (0.02, 0.0)})
    register_answer_type(pressure)
    assert is_correct("pressure", 1013.2, 1012.8, unit="hPa") and is_correct("pressure", 29.92, 29.93, unit="inHg")
    assert not is_correct("pressure", 29.92, 29.50, unit="inHg")  # 0.42 inHg is 14 hPa
    assert not equivalent("pressure", 29.92, 29.95, unit="inHg") and equivalent("pressure", 29.92, 29.95)
    assert (PASCALS.places(1), INCHES.places(1), MI.places(1)) == (1, 2, 1)
    assert _quantity(1017.6, PASCALS, (PASCALS, INCHES), 1, True) == "30.05 inches of mercury"
    assert _quantity(30.05, INCHES, (PASCALS, INCHES), 1, True) == "1017.6 hectopascals"
    rounding = {}
    for seed in range(20):
        convert, round_step = _answer_steps((PASCALS, INCHES), 1, random.Random(seed))
        rounding[convert["to"]] = round_step["ndigits"]
    assert rounding == {"hPa": 1, "inHg": 2}


def test_numbers_are_stated_exactly_at_every_magnitude():
    values = [m * 10.0 ** e for e in range(-6, 10) for m in (1.0, 1.5, -2.25, 3.14159, 9.87654321)]
    values += [123456.7, 1234567.5, 101325.5, 7480000.0, 0.00005]
    for x in values:
        for text in (fmt_num(x), plain_number(x)):
            assert "e" not in text and float(text) == x, (x, text)
        step = {"op": "at_least", "field": "p", "value": x}
        assert describe_step(step) == f"p ≥ {plain_number(x)}"
    # Where "g" is exact (every 0.2.0 threshold and radius), the text is what "g" printed.
    legacy = [t / 10 for t in range(-50, 10000)] + [-0.0, 12.0, 150.0, 25, 300]
    assert all(plain_number(x) == format(x, "g") for x in legacy)
    assert (fmt_num(-0.0), fmt_num(12.0), fmt_num(3.1)) == ("0", "12", "3.1")  # as 0.2.0 phrased them


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


def test_020_answer_types_and_tolerances_are_unchanged():
    kinds = {"count": "count", "event_id": "id", "magnitude": "number", "depth": "number", "distance": "number",
             "region": "group", "utc_time": "clock", "probability": "probability"}
    assert {name: scoring.ANSWER_TYPES[name].kind for name in kinds} == kinds
    tolerances = {"magnitude": (0.05, 0.0), "depth": (0.3, 0.02), "distance": (1.5, 0.03)}
    assert {name: scoring.ANSWER_TYPES[name].tolerance for name in tolerances} == tolerances
    assert not any(scoring.ANSWER_TYPES[name].unit_tolerance for name in kinds)  # one tolerance in km and mi
    assert scoring.ANSWER_TYPES["utc_time"].tolerance[0] == scoring.TIME_TOLERANCE_MIN == 1


def test_the_earthquake_domain_declares_exactly_the_builtin_types():
    declared = EARTHQUAKES.answer_types()
    assert [t.name for t in declared] == ["magnitude", "depth", "event_id", "region"]
    assert all(scoring.ANSWER_TYPES[t.name] == t for t in declared)  # registering them changes nothing


def test_an_answer_type_no_domain_declares_is_an_error_not_a_raw_comparison():
    for attempt in (lambda: normalize("colour", "red"), lambda: is_correct("colour", "red", "red"),
                    lambda: equivalent("colour", "red", "red")):
        with pytest.raises(KeyError, match="unknown answer type 'colour'"):
            attempt()


def test_domains_can_register_their_own_answer_types(answer_registry):
    register_answer_type(AnswerType("temperature", "number", (0.5, 0.0)))
    assert scoring.ANSWER_TYPES["temperature"].tolerance == (0.5, 0.0)
    assert normalize("temperature", "21.30 °C") == 21.3
    assert is_correct("temperature", 21.3, "21.7") and not is_correct("temperature", 21.3, "21.9")
    register_answer_type(AnswerType("observation_id", "id", id_pattern=r"k[a-z]{3}-\d{4}"))
    assert normalize("observation_id", "Observation KSFO-0001.") == "ksfo-0001"
    assert normalize("observation_id", "ksfo-0001 or kbos-0003") is None
    register_answer_type(AnswerType("us_region", "group", suffixes=("states",)))
    assert normalize("us_region", "The Northeast states") == "northeast"
    assert normalize("us_region", "equal") == "tie"
    register_answer_type(AnswerType("basin", "group", suffixes=("area", "sea area", "basin")))
    assert normalize("basin", "North Sea sea area") == normalize("basin", "the north sea basin.") == "north sea"
    assert normalize("basin", "Kessel area basin") == "kessel area"  # one group word, never two


def test_every_domain_reads_a_group_answer_that_adds_the_group_word_its_prompt_prints():
    """"Nevada region" is the region Nevada, and so in every domain: a group answer may end in the group
    word that its instance prints, which an invented instance draws for itself ("Nerm Halt berth" where
    the groups are berths). The scorer reads answers by the type registered when the domain was loaded,
    so that type strips every word a surface can draw, never only the unbound domain's own: otherwise
    14 invented instances in 15 would score compare_groups answers more strictly than every other domain,
    and the source gap, which compares domains, would measure the scorer."""
    compiled_at = 1_780_000_000
    words = set()
    for name in ("earthquakes", "weather", "neo", "invented"):
        domain = domains.get_domain(name)
        records = domain.synthetic(random.Random(3), 300, compiled_at, domain.synthetic_window_hours)
        bindings = ([domain.bind(domain.draw_surface(random.Random(seed))) for seed in range(60)]
                    if domain.has_surface else [domain])
        for bound in bindings:
            words.add(bound.group_label)
            groups = {bound.dress(record).group for record in records}
            assert len(groups) >= 3, name
            for group in groups:
                for answer in (f"{group} {bound.group_label}", f"The {group} {bound.group_label.title()}."):
                    assert is_correct(domain.group_answer_type, group, answer), (name, answer)
                assert not is_correct(domain.group_answer_type, group, f"{group} {bound.group_label}s"), name
    assert {"region", "body", "depot", "berth", "outpost"} <= words and len(words) >= 12


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
from evalgen.domains.earthquakes import Earthquakes


class Toy(Earthquakes):  # a complete domain under another name
    name = {name!r}


DOMAIN = Toy()
"""
PLUGIN = """\
from __future__ import annotations

from dataclasses import dataclass

from evalgen.domains.earthquakes import Earthquakes


@dataclass(frozen=True)
class Depot:  # a dataclass needs its module registered while the plugin runs
    name: str


class Consignments(Earthquakes):
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


FLAKY_DOMAIN = """\
import flaky_switch
from evalgen.domains.base import Domain
from evalgen.domains.earthquakes import Earthquakes

if flaky_switch.FAULT == "import":
    raise RuntimeError("beta cannot be imported")


class Beta(Earthquakes):
    name = "beta"


class Bare(Domain):  # sets nothing, so validation refuses it
    name = "beta"


DOMAIN = Bare() if flaky_switch.FAULT == "invalid" else Beta()
"""


@pytest.mark.parametrize("fault, error", [("import", RuntimeError), ("invalid", ValueError)])
def test_a_broken_domain_module_leaves_the_registry_as_it_was(tmp_path, monkeypatch, fault, error):
    root = tmp_path / "flaky_domains"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (tmp_path / "flaky_switch.py").write_text(f"FAULT = {fault!r}\n")
    for module in ("alpha", "gamma"):
        (root / f"{module}.py").write_text(TOY_DOMAIN.format(name=module))
    (root / "beta.py").write_text(FLAKY_DOMAIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        registry = Registry("flaky_domains")
        for attempt in (lambda: registry.get("gamma"), lambda: registry.get("alpha"), registry.all):
            with pytest.raises(error):  # every lookup fails as the first did: none quietly misses gamma
                attempt()
        assert registry._entries == {} and not registry._discovered
        importlib.import_module("flaky_switch").FAULT = None  # the module is mended
        sys.modules.pop("flaky_domains.beta", None)
        assert [d.name for d in registry.all()] == ["alpha", "beta", "gamma"]
    finally:
        for module in [m for m in sys.modules if m.startswith(("flaky_domains", "flaky_switch"))]:
            del sys.modules[module]


def test_every_domain_validates():
    for domain in [*all_domains(), GenericQuakes()]:
        domain.validate()


def variant(base: type = GenericQuakes, **attributes) -> Domain:
    return type("Variant", (base,), {"name": "variant", **attributes})()


def test_validation_names_what_is_wrong_with_a_domain():
    mag, depth = EARTHQUAKES.field("mag"), EARTHQUAKES.field("depth")
    wordings = GenericQuakes.lexicon.filters
    lexicon = replace(GenericQuakes.lexicon, filters={k: v for k, v in wordings.items() if k != "time_between"})
    cases = [
        (type("Bare", (Domain,), {"name": "bare"})(), "Bare does not set noun, plural, id_label"),
        (variant(fields=(mag, replace(depth, canonical_unit="m"))),
         "depth is stored in m, which is not one of its units"),
        (variant(fields=(mag, replace(depth, units=(replace(KM, scale=1000.0), MI)))),
         "the canonical unit of depth must have scale 1 and offset 0"),
        (variant(fields=(mag, replace(depth, units=(KM, replace(MI, decimals=0))))),
         "the depth margin 0.6 is smaller than the rounding of a threshold restated in mi (0.804672)"),
        (variant(lexicon=lexicon), "its lexicon has no wording for time_between"),
        (variant(op_names={("at_least", "mag"): "temp_at_least"}),
         "op name 'temp_at_least' is not the 0.2.0 alias of at_least(mag)"),
        (variant(EARTHQUAKES.__class__, fields=(replace(mag, answer_type="intensity"), depth)),
         "op name 'max_mag' answers as 'magnitude', not as 'intensity'"),
        (variant(roles={"primary": "depth", "secondary": "mag"}), "P01-P12 do not bind to its fields"),
        (variant(fields=(mag, replace(depth, tolerance=(0.1, 0.0)))),
         "answer type 'depth' is already registered differently"),
    ]
    for domain, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            domain.validate()
    with pytest.raises(ValueError, match="answer type 'depth'"):  # the registry refuses it
        Registry("evalgen.domains").register(cases[-1][0])


TREMORS = """\
from dataclasses import replace

from evalgen.domains.earthquakes import DEPTH, MAGNITUDE

from generic_quakes import GenericQuakes


class Tremors(GenericQuakes):  # a domain whose answer types are not built into the scorer
    name = "tremors"
    fields = (replace(MAGNITUDE, answer_type="intensity", tolerance=(0.5, 0.0)), DEPTH)
    id_answer_type = "tremor_id"
    group_answer_type = "zone"


DOMAIN = Tremors()
"""
TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture
def tremor_package(tmp_path, monkeypatch):
    """A domain package with one domain, whose answer types only that domain declares."""
    (tmp_path / "tremor_domains").mkdir()
    (tmp_path / "tremor_domains" / "__init__.py").write_text("")
    (tmp_path / "tremor_domains" / "tremors.py").write_text(TREMORS)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for module in [m for m in sys.modules if m.startswith("tremor_domains")]:
        del sys.modules[module]


def test_a_domains_answer_types_are_registered_with_it(answer_registry, tremor_package, monkeypatch):
    monkeypatch.setattr(domains, "REGISTRY", Registry("tremor_domains"))
    assert "intensity" not in scoring.ANSWER_TYPES
    assert is_correct("intensity", 4.0, "4.4") and not is_correct("intensity", 4.0, "4.6")  # discovered on demand
    assert normalize("tremor_id", "Tremor NC75012345.") == "nc75012345" and normalize("zone", "Utah region") == "utah"
    assert {"intensity", "tremor_id", "zone"} <= set(scoring.ANSWER_TYPES)


def test_a_fresh_process_scores_a_domains_answer_types(tremor_package):
    script = (
        "import evalgen.domains as domains\n"
        "from evalgen.scoring import is_correct, normalize\n"
        "domains.REGISTRY = domains.Registry('tremor_domains')  # nothing has looked a domain up yet\n"
        "print(is_correct('intensity', 4.0, '4.4'), is_correct('intensity', 4.0, '4.6'), normalize('zone', 'Utah'))\n"
        "try:\n    normalize('colour', 'red')\nexcept KeyError as exc:\n    print(exc)\n"
    )
    path = os.pathsep.join([str(tremor_package), str(TESTS_DIR), str(TESTS_DIR.parent)])
    done = subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": path}, capture_output=True,
                          text=True, cwd=tremor_package, check=True)
    assert done.stdout.splitlines() == ["True False utah", "\"unknown answer type 'colour'\""]
