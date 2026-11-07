"""The answer key is correct on hand-built cases whose answers were worked out by hand.

Five quakes on (or near) the 118°W meridian. On a 6371 km sphere one degree of latitude is
111.195 km, so distances from (35.0, -118.0) are 0, 55.6, 111.2 and 222.4 km.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalgen.models import Event, Notice, Report
from evalgen.ops import DigestData, execute
from evalgen.rules import Mishandle, resolve
from evalgen.traps import primary_report

T0 = int(datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).timestamp())


def quake(event_id, hours, lat, lon, depth_km, mag, region, mag_type="ml"):
    return Event(id=event_id, time=T0 + int(hours * 3600), lat=lat, lon=lon, depth_km=depth_km, mag=mag,
                 mag_type=mag_type, region=region, network=event_id[:2])


EVENTS = [
    quake("aa1000001", 0.0, 35.0, -118.0, 5.0, 2.4, "Alpha"),
    quake("aa1000002", 1.0, 35.5, -118.0, 12.0, 3.1, "Alpha"),
    quake("bb1000003", 2.0, 36.0, -118.0, 30.0, 4.6, "Beta", "mb"),
    quake("bb1000004", 3.0, 37.0, -118.0, 8.0, 3.8, "Beta"),
    quake("cc1000005", 5.5, 40.0, -120.0, 2.0, 1.2, "Gamma", "md"),
]
DEDUPE = {"op": "dedupe"}
POINT = {"lat": 35.0, "lon": -118.0}


def chain(*steps):
    return [DEDUPE, *steps]


@pytest.mark.parametrize("steps, expected", [
    # magnitude thresholds: "at least" includes the boundary, "below" excludes it
    ((({"op": "mag_at_least", "value": 3.1}), {"op": "count"}), 3),
    ((({"op": "mag_below", "value": 3.1}), {"op": "count"}), 2),
    # distances: 111.2 km is outside 100 km but inside 70 miles (112.65 km)
    (({"op": "within_distance", **POINT, "radius": 100, "unit": "km"}, {"op": "count"}), 2),
    (({"op": "within_distance", **POINT, "radius": 70, "unit": "mi"}, {"op": "count"}), 3),
    (({"op": "within_distance", **POINT, "radius": 60, "unit": "mi"}, {"op": "count"}), 2),
    # depth thresholds with unit conversion: 5 miles = 8.047 km, so the 8.0 km quake is shallower
    (({"op": "depth_shallower_than", "value": 10, "unit": "km"}, {"op": "count"}), 3),
    (({"op": "depth_at_least", "value": 5, "unit": "mi"}, {"op": "count"}), 2),
    # time window [00:30, 02:30) UTC contains the 01:00 and 02:00 quakes
    (({"op": "time_between", "start": T0 + 1800, "end": T0 + 9000}, {"op": "count"}), 2),
    # argmax / rank / max
    (({"op": "in_region", "region": "Beta"}, {"op": "argmax_mag"}), "bb1000003"),
    (({"op": "rank_mag", "k": 2},), "bb1000004"),
    (({"op": "rank_mag", "k": 3},), "aa1000002"),
    (({"op": "in_region", "region": "Alpha"}, {"op": "max_mag"}), 3.1),
    # depth answers with conversion: 30 km = 18.64 miles
    (({"op": "mag_at_least", "value": 3.0}, {"op": "max_depth"}, {"op": "convert", "to": "mi"},
      {"op": "round", "ndigits": 1}), 18.6),
    (({"op": "in_region", "region": "Beta"}, {"op": "min_depth"}, {"op": "convert", "to": "km"},
      {"op": "round", "ndigits": 1}), 8.0),
    # nearest: (36.1, -118.0) is 11.1 km (6.9 mi) from the 36.0 quake
    (({"op": "nearest_to", "lat": 36.1, "lon": -118.0, "label": "x"},), "bb1000003"),
    (({"op": "nearest_distance", "lat": 36.1, "lon": -118.0, "label": "x"}, {"op": "convert", "to": "km"},
      {"op": "round", "ndigits": 1}), 11.1),
    (({"op": "nearest_distance", "lat": 36.1, "lon": -118.0, "label": "x"}, {"op": "convert", "to": "mi"},
      {"op": "round", "ndigits": 1}), 6.9),
    # region comparison, including a legitimate tie
    (({"op": "mag_at_least", "value": 3.0}, {"op": "compare_regions", "a": "Alpha", "b": "Beta"}), "Beta"),
    (({"op": "mag_at_least", "value": 2.0}, {"op": "compare_regions", "a": "Alpha", "b": "Beta"}), "tie"),
    (({"op": "mag_below", "value": 3.0}, {"op": "compare_regions", "a": "Alpha", "b": "Beta"}), "Alpha"),
    # time of the largest quake, UTC HH:MM
    (({"op": "time_of_largest"},), "02:00"),
    # chained filters
    (({"op": "mag_at_least", "value": 3.0}, {"op": "within_distance", **POINT, "radius": 120, "unit": "km"},
      {"op": "count"}), 2),
    (({"op": "count"},), 5),
])
def test_answer_key_on_hand_built_events(steps, expected):
    assert execute(chain(*steps), EVENTS) == expected


# ------------------------------------------------------------------ traps, worked by hand

DUP_CROSS = Report(ref="us7000dupx", network="us", status="preliminary", time=T0 + 7212, lat=36.02, lon=-118.01,
                   depth_km=28.0, mag=4.9, mag_type="mb", region="Beta", same_as="bb1000003")
DUP_SAME_ID = Report(ref="aa1000002", network="at", status="preliminary", time=T0 + 3595, lat=35.51, lon=-118.0,
                     depth_km=15.0, mag=3.0, mag_type="ml", region="Alpha")
PHANTOM = Report(ref="cc1000099", network="cc", status="reviewed", time=T0 + 4 * 3600, lat=35.1, lon=-118.0,
                 depth_km=3.0, mag=5.0, mag_type="md", region="Gamma")


def trap_digest() -> DigestData:
    reports = [primary_report(e) for e in EVENTS]
    reports[3] = Report(**{**reports[3].__dict__, "mag": 4.4})  # bb1000004 filed with a wrong magnitude...
    notices = [
        Notice(kind="correction", target="bb1000004", time=T0 + 4 * 3600, new_mag=3.8, old_mag=4.4, mag_type="ml"),
        Notice(kind="retraction", target="cc1000099", time=T0 + 5 * 3600, reason="quarry blast"),
    ]
    return DigestData(reports + [DUP_CROSS, DUP_SAME_ID, PHANTOM], notices)


def test_reading_rules_recover_the_snapshot_exactly():
    data = trap_digest()
    assert {e.id: e for e in resolve(data.reports, data.notices)} == {e.id: e for e in EVENTS}


def test_trap_digest_answers_match_the_snapshot_key():
    data = trap_digest()
    for steps in [({"op": "mag_at_least", "value": 3.0}, {"op": "count"}), ({"op": "argmax_mag"},),
                  ({"op": "mag_at_least", "value": 4.0}, {"op": "count"})]:
        assert execute(chain(*steps), data) == execute(chain(*steps), EVENTS)


def test_naive_reading_falls_for_every_trap():
    data = trap_digest()
    count = chain({"op": "mag_at_least", "value": 3.0}, {"op": "count"})
    assert execute(count, EVENTS) == 3
    assert execute(count, data, skip=("dedupe",)) == 6  # 2 re-reports + the phantom counted
    assert execute(chain({"op": "argmax_mag"}), data, skip=("dedupe",)) == "cc1000099"


@pytest.mark.parametrize("mishandle, steps, fallen", [
    (Mishandle(keep_retracted=("cc1000099",)), ({"op": "mag_at_least", "value": 3.0}, {"op": "count"}), 4),
    (Mishandle(keep_retracted=("cc1000099",)), ({"op": "argmax_mag"},), "cc1000099"),
    (Mishandle(split=(("us7000dupx", "us", "preliminary"),)), ({"op": "argmax_mag"},), "us7000dupx"),
    (Mishandle(split=(("aa1000002", "at", "preliminary"),)), ({"op": "mag_at_least", "value": 3.0}, {"op": "count"}), 4),
    (Mishandle(ignore_correction_for=("bb1000004",)), ({"op": "mag_at_least", "value": 4.0}, {"op": "count"}), 2),
    (Mishandle(prefer_preliminary_for=("aa1000002",)), ({"op": "mag_at_least", "value": 3.1}, {"op": "count"}), 2),
])
def test_each_mishandling_gives_its_fallen_answer(mishandle, steps, fallen):
    data = trap_digest()
    correct = execute(chain(*steps), EVENTS)
    assert execute(chain(*steps), data, mishandle=mishandle) == fallen != correct


def test_missing_every_trap_at_once_is_what_the_naive_reader_does():
    data = trap_digest()
    everything = Mishandle(keep_retracted=("cc1000099",), ignore_correction_for=("bb1000004",),
                           split=(("us7000dupx", "us", "preliminary"), ("aa1000002", "at", "preliminary")))
    for steps in [({"op": "mag_at_least", "value": 3.0}, {"op": "count"}), ({"op": "argmax_mag"},),
                  ({"op": "mag_at_least", "value": 4.0}, {"op": "count"})]:
        assert execute(chain(*steps), data, mishandle=everything) == execute(chain(*steps), data, skip=("dedupe",))
