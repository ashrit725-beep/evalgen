"""The weather domain (evalgen/domains/weather.py): its fields and exact units, its IDs, its seeded
synthetic generator, its live source (NWS station observations, fetched once, frozen, hashed and
replayed, with a synthetic fallback), and whole runs with the mix "weather:classic": exact round trips
of every report and notice, the guards, the traps, unique questions, metamorphic variants and the
oracle at 100%.

tests/fixtures/weather/raw_feed.json is a bundle as NwsObservations.fetch freezes one (each station's
HTTP status and the text of its body), made from the latest observations of five stations captured
from api.weather.gov on 2026-11-08: each response with its "@context" trimmed, KSEA and KLAX in the
FeatureCollection shape of the observations endpoint and the others as the latest-observation endpoint
serves them. KDEN reported no pressure. The capture kept the parsed responses, so each body is the
trimmed response written out again as the API writes its JSON (indented by four spaces), not the
bytes it sent; a fetch freezes those bytes exactly (test_a_fetch_freezes_each_body_exactly_as_served).

No test touches the network: every test here runs offline (``offline``), and a test that needs the
API serves it from the fixture or from synthetic observations.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from evalgen import pipeline, scoring
from evalgen.digest import LINE_PRESETS, NOTICE_PLACEMENTS, make_chatter
from evalgen.domains import get_domain
from evalgen.domains import weather as wx
from evalgen.domains.base import Record
from evalgen.domains.weather import (FAHRENHEIT, HPA, ID_PATTERN, INHG, KELVIN, KMH, KNOT, MPH, MPS, PRESSURE,
                                     REGIONS, STATIONS, TEMPERATURE, WEATHER, WIND, NwsObservations, Weather)
from evalgen.genres import detect_genre
from evalgen.genres.base import RenderContext, parse_glossary
from evalgen.genres.classic import (CLASSIC, RECORD_COORD_PRESETS, RECORD_HEADERS, RECORD_LAYOUTS,
                                    RECORD_TIME_PRESETS, UNIT_PRESETS)
from evalgen.models import Instance, Notice
from evalgen.ops import DigestData, Step, canonical_threshold, distance_km, execute, rows_after_filters, terminal_step
from evalgen.pipeline import GenerateConfig, question_specs
from evalgen.questions import Context, fingerprint, round_to_step
from evalgen.report import load_instances
from evalgen.rules import Mishandle
from evalgen.scoring import is_correct, normalize, same_answer
from evalgen.seeds import SeedLog
from evalgen.solver import extract_digest, solve
from evalgen.sources import USER_AGENT, build_domain_snapshot, load_snapshot, record_problems, save_snapshot
from evalgen.traps import check_resolution
from evalgen.units import decode
from evalgen.util import read_json, sha256_hex, write_json

from conftest import quiet

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "weather" / "raw_feed.json"
FETCHED = datetime(2026, 11, 8, 2, 40, tzinfo=timezone.utc)  # when the fixture's bundle was frozen
COMPILED_AT = int(FETCHED.timestamp())
FIXTURE_RECORDS = {  # station -> (time, (lat, lon), temperature °C, wind km/h, pressure hPa), from the raw responses
    "KJFK": ("2026-11-08T02:25:00Z", (40.65, -73.78), 11.0, 27.8, 1010.5),  # 27.792 km/h, 101049.84 Pa
    "KORD": ("2026-11-08T02:25:00Z", (41.98, -87.9), 6.0, 11.1, 1019.3),  # 11.124 km/h, 101930.3 Pa
    "KLAX": ("2026-11-08T02:30:00Z", (33.93, -118.4), 23.0, 11.1, 1015.6),  # 101557.8 Pa
    "KSEA": ("2026-11-08T02:30:00Z", (47.45, -122.3), 16.0, 27.8, 1006.4),  # 100643.47 Pa
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No request leaves the test: sending one fails as it does without a network. The list holds the
    URLs that were attempted."""
    attempted: list[str] = []

    def unreachable(session, request, **kwargs):
        attempted.append(request.url)
        raise requests.ConnectionError("offline: tests never touch the network")

    monkeypatch.setattr(requests.Session, "send", unreachable)
    return attempted


def fixture_bundle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def served(station: str) -> dict:
    """A station's recorded response, read from the body the bundle keeps."""
    return json.loads(fixture_bundle()["stations"][station]["body"])


def answer(document: object, status: int = 200) -> dict:
    """A station's response as a bundle keeps it: its status and the text of its body (a JSON document, or
    text as it is)."""
    return {"status": status, "body": document if isinstance(document, str) else json.dumps(document)}


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


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


# ------------------------------------------------------------------------ the domain and its units


def test_weather_is_a_built_in_domain_with_its_own_answer_types():
    assert get_domain("weather") is WEATHER and type(WEATHER).__module__ == "evalgen.domains.weather"
    WEATHER.validate()
    assert (WEATHER.noun, WEATHER.id_label, WEATHER.source_label, WEATHER.group_label) == (
        "weather observation", "observation ID", "feed", "region")
    assert WEATHER.roles == {"primary": "temp", "secondary": "wind"} and WEATHER.has_location
    assert isinstance(WEATHER.live, NwsObservations) and WEATHER.forecaster is None and not WEATHER.has_surface
    assert (WEATHER.live.source, WEATHER.live.raw_filename, WEATHER.live.user_agent) == ("nws", "raw_feed.json",
                                                                                         USER_AGENT)
    assert WEATHER.live.stations == tuple(s.id for s in STATIONS) and len(STATIONS) == 24  # four per region
    assert sorted(Counter(s.region for s in STATIONS).values()) == [4] * len(REGIONS)
    assert WEATHER.units == {"temp": (wx.CELSIUS, FAHRENHEIT, KELVIN), "wind": (KMH, MPH, KNOT, MPS),
                             "pressure": (HPA, INHG)}
    declared = WEATHER.answer_types()
    assert [t.name for t in declared] == ["weather_temperature", "weather_wind_speed", "weather_pressure",
                                          "weather_observation_id", "weather_region"]
    assert all(scoring.ANSWER_TYPES[t.name] == t for t in declared)  # registered with the domain
    # One tolerance, the same in every unit: 0.3 °C is 0.54 °F and 0.3 K; 0.5 km/h or 2 %; 0.5 hPa.
    assert is_correct("weather_temperature", 70.0, 70.5, unit="°F") and not is_correct("weather_temperature", 70.0,
                                                                                       70.6, unit="°F")
    assert is_correct("weather_temperature", 294.4, "294.7 K", unit="K")
    assert not is_correct("weather_temperature", 294.4, 294.8, unit="K")
    assert is_correct("weather_wind_speed", 15.0, 15.3, unit="kt") and not is_correct("weather_wind_speed", 15.0,
                                                                                      15.35, unit="kt")
    assert is_correct("weather_wind_speed", 60.0, 61.2, unit="km/h")  # 2 % of 60 km/h
    assert is_correct("weather_pressure", 29.92, 29.93, unit="inHg") and not is_correct("weather_pressure", 29.92,
                                                                                        29.94, unit="inHg")
    assert normalize("weather_observation_id", "Observation WX-482137.") == "wx-482137"
    assert normalize("weather_observation_id", "wx-305710") == "wx-305710"
    assert normalize("weather_observation_id", "WX-482137 or WX-111111") is None
    assert normalize("weather_region", "the Mountain West region") == "mountain west"
    assert normalize("weather_region", "equal") == "tie"


def test_units_are_exact_and_every_stored_value_reads_back():
    assert FAHRENHEIT.to_canonical(32) == 0.0 and FAHRENHEIT.to_canonical(212) == pytest.approx(100.0, abs=1e-12)
    assert KELVIN.to_canonical(273.15) == 0.0 and FAHRENHEIT.from_canonical(-40.0) == pytest.approx(-40.0)
    assert (MPH.scale, KNOT.scale, MPS.scale, INHG.scale) == (1.609344, 1.852, 3.6, 33.86389)
    assert round(INHG.to_canonical(29.92), 1) == 1013.2 and round(INHG.from_canonical(1013.25), 2) == 29.92
    printed_as = [(TEMPERATURE, 11.0, FAHRENHEIT, "51.8"), (TEMPERATURE, 0.0, FAHRENHEIT, "32"),
                  (TEMPERATURE, -17.8, FAHRENHEIT, "0"), (TEMPERATURE, -40.0, FAHRENHEIT, "-40"),
                  (WIND, 27.8, KNOT, "15"), (WIND, 36.0, MPS, "10"), (PRESSURE, 1010.5, INHG, "29.84"),
                  (PRESSURE, 1019.3, INHG, "30.1")]
    for spec, value, unit, text in printed_as:
        assert spec.encode(value, unit) == text and spec.decode(text, unit) == value
    checked = 0
    for spec, lo, hi in ((TEMPERATURE, -400, 450), (WIND, 0, 1200), (PRESSURE, 9500, 10600)):
        for tenths in range(lo, hi, 3):
            value = round(tenths / 10, 1) + 0.0
            for unit in spec.units:
                text = spec.encode(value, unit)
                assert decode(text, 1, unit) == value and "e" not in text, (value, unit)
                assert not (float(text) == 0 and text.startswith("-")), text  # zero is never written "-0"
                checked += 1
    assert checked == 284 * 3 + 400 * 4 + 367 * 2


def test_margins_and_steps_cover_a_threshold_restated_in_any_unit():
    for spec in (TEMPERATURE, WIND, PRESSURE):
        for unit in spec.units:  # a restated threshold moves by half the last place of its unit
            assert spec.margin[0] >= 0.5 * 10 ** -unit.places(spec.decimals) * unit.scale, (spec.key, unit.symbol)
        assert spec.jitter > spec.margin[0] + spec.threshold_step / 2  # thresholds can clear the quantile they start at
    assert [u.step for u in TEMPERATURE.units] == [None, 1.0, 1.0] and TEMPERATURE.threshold_step == 0.5
    assert [u.step for u in WIND.units] == [None, 1.0, 1.0, 0.5] and WIND.positive_threshold
    assert INHG.places(PRESSURE.decimals) == 2 and INHG.step == 0.05
    assert TEMPERATURE.correctable and TEMPERATURE.signed and not (WIND.correctable or PRESSURE.correctable)


def test_ids_are_opaque_serials_that_name_no_station_feed_or_time(monkeypatch):
    """An observation ID names nothing about its observation: a re-report filed under it gives a station and a
    time of its own (traps.py), which an ID that named them would contradict, or pin to its observation's."""
    rng, taken = random.Random(3), set()
    refs = [WEATHER.new_id(source, rng, taken) for source in ("KJFK", "MADIS", "NOAAPORT", "AWOSNET", "KJFK")]
    assert all(re.fullmatch(ID_PATTERN, ref.lower()) and ref.startswith("WX-") for ref in refs)
    assert set(refs) == taken and len(taken) == 5
    assert len({WEATHER.new_id(source, random.Random(9), set()) for source in ("KJFK", "MADIS")}) == 1  # any feed alike
    monkeypatch.setattr(wx, "SERIALS", (100000, 100020))  # a new ID never takes one in use
    crowded = {f"WX-{n}" for n in range(100000, 100019)}
    assert WEATHER.new_id("KSEA", random.Random(0), crowded) == "WX-100019"
    monkeypatch.undo()
    mimic = WEATHER.mimic_id("WX-251030", random.Random(1), taken)
    assert re.fullmatch(ID_PATTERN, mimic.lower()) and mimic != "WX-251030" and mimic in taken
    for bad in ("nc75462032", "KORD-25103"):
        with pytest.raises(ValueError, match="not a weather observation ID"):
            WEATHER.mimic_id(bad, random.Random(1), set())
    records = WEATHER.synthetic(random.Random(4), 600, COMPILED_AT, 24)
    assert all(re.fullmatch(ID_PATTERN, r.id.lower()) for r in records) and len({r.id for r in records}) == 600


# ------------------------------------------------------------------------- the synthetic weather


def test_the_synthetic_generator_is_seeded_and_never_reads_the_clock():
    records = WEATHER.synthetic(random.Random(5), 300, COMPILED_AT, 24)
    assert WEATHER.synthetic(random.Random(5), 300, COMPILED_AT, 24) == records
    assert WEATHER.synthetic(random.Random(6), 300, COMPILED_AT, 24) != records
    assert record_problems(WEATHER, records) == [] and records == sorted(records, key=lambda r: (r.time, r.id))
    assert all(COMPILED_AT - 24 * 3600 < r.time < COMPILED_AT for r in records)
    log = SeedLog(9)
    snapshot, raw = build_domain_snapshot(WEATHER, "synthetic", log)
    assert raw is None and list(log.seeds) == ["snapshot/weather/clock", "snapshot/weather/synthetic"]
    assert build_domain_snapshot(WEATHER, "synthetic", SeedLog(9))[0] == snapshot
    assert (snapshot.source, snapshot.domain, snapshot.window_hours, len(snapshot.events)) == ("synthetic", "weather",
                                                                                              24, 360)
    stations = {s.id: s for s in STATIONS}
    by_name = {s.name: s for s in STATIONS}
    relays = WEATHER.trap_vocab.rereporting_sources
    for r in snapshot.events:  # every observation is where its station is, in its region, filed by it or a relay
        station = by_name[r.place]
        assert (r.lat, r.lon, r.group, r.place) == (station.lat, station.lon, station.region, station.name)
        assert r.source in (station.id, *relays) and r.qualifiers == {} and re.fullmatch(ID_PATTERN, r.id.lower())
    assert {r.source for r in snapshot.events} == set(stations) | set(relays)
    assert {r.group for r in snapshot.events} == set(REGIONS)
    relayed = sum(r.source in relays for r in snapshot.events) / len(snapshot.events)
    assert abs(relayed - wx.RELAY_SHARE) < 0.06  # the relays file observations of their own
    # Observations are timed as NWS stations take them: a station's routine observation at its own minute of
    # the hour, or a five-minute observation, which gives whole degrees; winds are in whole knots and
    # pressures in hundredths of an inch of mercury, as the station reports them.
    routine: dict[str, set] = {}
    for r in records:
        station = r.place
        if r.time % 300 == 0:
            assert r.values["temp"] == round(r.values["temp"])
        else:
            routine.setdefault(station, set()).add(r.time % 3600 // 60)
            assert r.time % 60 == 0
        assert r.values["wind"] == wx.whole_knots(r.values["wind"]) and r.values["pressure"] == wx.altimeter(
            r.values["pressure"])
    assert all(len(minutes) == 1 and minutes <= set(wx.ROUTINE_MINUTES) for minutes in routine.values())
    assert 0.2 < sum(r.time % 300 == 0 for r in records) / len(records) < 0.45


def climate(day: int, seed: int, n: int = 3000) -> list[Record]:
    """`n` synthetic observations over the day that ends at 06:00 UTC on day `day` of 2026."""
    compiled_at = int(datetime(2026, 1, 1, 6, tzinfo=timezone.utc).timestamp()) + day * 86400
    return WEATHER.synthetic(random.Random(seed), n, compiled_at, 24)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def test_synthetic_observations_follow_season_hour_and_climate():
    winter, summer = climate(15, 1), climate(196, 2)
    by = {name: {s.id: [r.values["temp"] for r in records if r.source == s.id] for s in STATIONS}
          for name, records in (("winter", winter), ("summer", summer))}
    assert mean(by["winter"]["KMSP"]) < -3 < 18 < mean(by["winter"]["KMIA"])  # Minneapolis and Miami in January
    assert mean(by["summer"]["KPHX"]) > 30 > 22 > mean(by["summer"]["KSFO"])  # Phoenix and San Francisco in July
    assert all(mean(by["summer"][s]) > mean(by["winter"][s]) + 5 for s in ("KBOS", "KORD", "KDEN", "KSEA"))
    for records in (winter, summer):
        temps = [r.values["temp"] for r in records]
        afternoon = [r.values["temp"] for r in records if 13 <= wx._solar_hour(r.time, r.lon) < 17]
        night = [r.values["temp"] for r in records if 3 <= wx._solar_hour(r.time, r.lon) < 7]
        assert mean(afternoon) > mean(night) + 4  # the diurnal cycle
        routine = [r.values["temp"] for r in records if r.time % 300 != 0]  # routine observations give tenths
        assert sum(t == round(t) for t in routine) / len(routine) < 0.2  # (a tenth of them whole, by chance)
        assert all(r.values["temp"] == round(r.values["temp"]) for r in records if r.time % 300 == 0)
        winds = sorted(r.values["wind"] for r in records)
        assert winds[0] >= 0.0 and 8.0 < winds[len(winds) // 2] < 25.0 and winds[-1] <= 100.0
        pressures = sorted(r.values["pressure"] for r in records)
        assert 960.0 < pressures[0] and pressures[-1] < 1065.0 and 1003.0 < pressures[len(pressures) // 2] < 1029.0
    assert min(r.values["temp"] for r in winter) < 0 < max(r.values["temp"] for r in winter)  # signed values


# ------------------------------------------------------------------------------ the live source


def test_the_live_source_reads_the_recorded_observations():
    records, compiled_at, window = WEATHER.live.parse(FIXTURE.read_bytes())
    assert (compiled_at, window) == (COMPILED_AT, 24)
    assert [r.source for r in records] == ["KJFK", "KORD", "KLAX", "KSEA"]  # time order; KDEN gave no pressure
    stations = {s.id: s for s in STATIONS}
    for r in records:
        when, position, temp, wind, pressure = FIXTURE_RECORDS[r.source]
        assert (r.time, (r.lat, r.lon), r.values) == (ts(when), position, {"temp": temp, "wind": wind,
                                                                           "pressure": pressure})
        assert (r.group, r.place, r.qualifiers) == (stations[r.source].region, stations[r.source].name, {})
        assert re.fullmatch(ID_PATTERN, r.id.lower())
    assert record_problems(WEATHER, records) == []
    assert WEATHER.live.parse(FIXTURE.read_bytes())[0] == records  # the IDs depend on the bytes alone
    head = {"format": wx.BUNDLE_FORMAT, "fetched_at": "2026-11-08T02:40:00Z"}
    parsed_responses = {**head, "format": "evalgen-nws-bundle/1", "stations": {"KJFK": served("KJFK")}}
    for broken, error in ((b"<html>Service Unavailable</html>", ValueError), (b'{"format": "other"}', ValueError),
                          (json.dumps(parsed_responses).encode(), ValueError),  # responses kept as parsed: not frozen
                          (json.dumps(head).encode(), KeyError),
                          (json.dumps({**head, "stations": []}).encode(), ValueError),
                          (json.dumps({**head, "stations": {}, "window_hours": None}).encode(), ValueError),
                          (json.dumps({**head, "fetched_at": 1794105600, "stations": {}}).encode(), ValueError)):
        with pytest.raises(error):  # what the snapshot catches to fall back to synthetic records
            WEATHER.live.parse(broken)


def observation(when: str, *, temp=(11.0, "degC"), wind=(27.792, "km_h-1"), pressure=(101049.84, "Pa"),
                metar: bool = False, qc: str = "V", station: str = "KJFK", point=(-73.78, 40.65)) -> dict:
    """The fixture's KJFK feature with other values, units, flags or time (test-made, not recorded)."""
    feature = copy.deepcopy(served("KJFK"))
    props = feature["properties"]
    props.update(timestamp=when, stationId=station, rawMessage="KJFK 080251Z 28015KT" if metar else "")
    for name, (value, unit) in (("temperature", temp), ("windSpeed", wind), ("barometricPressure", pressure)):
        props[name] = {"unitCode": f"wmoUnit:{unit}", "value": value, "qualityControl": qc if name == "windSpeed" else "V"}
    feature["geometry"] = None if point is None else {"type": "Point", "coordinates": list(point)}
    return feature


def bundle_of(stations: dict, fetched_at: datetime = FETCHED) -> bytes:
    """A bundle of `stations`: each a response as a bundle keeps it (answer), or a failed request's error."""
    return json.dumps({"format": wx.BUNDLE_FORMAT, "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "window_hours": 24, "url": "test", "stations": stations}).encode("utf-8")


def test_the_live_source_drops_what_it_cannot_use_and_keeps_one_observation_an_hour():
    kept = {"KJFK": answer({"type": "FeatureCollection", "features": [
        observation("2026-11-08T01:51:00+00:00", metar=True),  # the hourly METAR: kept over ...
        observation("2026-11-08T01:55:00+00:00", temp=(12, "degC")),  # ... a later five-minute observation
        observation("2026-11-08T00:05:00+00:00", temp=(53.6, "degF"), wind=(7.72, "m_s-1"), point=None),
        observation("2026-11-07T23:30:00+00:00", temp=(285.15, "K"), wind=(15, "kt"), pressure=(1012.0, "hPa")),
        observation("2026-11-07T22:10:00+00:00", temp=(None, "degC")),  # no temperature
        observation("2026-11-07T21:10:00+00:00", qc="X"),  # a wind speed the quality control rejected
        observation("2026-11-07T20:10:00+00:00", wind=(27.792, "furlong_fortnight-1")),  # a unit it does not know
        observation("2026-11-07T19:10:00+00:00", pressure=(50000.0, "Pa")),  # no real sea-level pressure
        observation("2026-11-07T02:39:59+00:00"),  # before the window
        observation("2026-11-08T02:40:00+00:00"),  # at its end, which it excludes
        observation("2026-11-08T00:30:00+00:00", station="KLGA"),  # another station's observation
    ]}), "KXYZ": answer(served("KJFK")),  # a station the table does not know
        "KSEA": answer(served("KSEA"), status=503),  # an error status, whatever its body says
        "KLAX": {"error": "ReadTimeout: read timed out"},  # a request that failed
        "KDEN": answer("<html>Service Unavailable</html>"),  # a success whose body is not JSON
        "KORD": {"status": 200}}  # no body at all
    records, _, _ = WEATHER.live.parse(bundle_of(kept))
    jfk, table = (40.65, -73.78), next((s.lat, s.lon) for s in STATIONS if s.id == "KJFK")
    assert [(r.time, r.values, (r.lat, r.lon)) for r in records] == [
        (ts("2026-11-07T23:30:00Z"), {"temp": 12.0, "wind": 27.8, "pressure": 1012.0}, jfk),
        (ts("2026-11-08T00:05:00Z"), {"temp": 12.0, "wind": 27.8, "pressure": 1010.5}, table),  # no geometry
        (ts("2026-11-08T01:51:00Z"), {"temp": 11.0, "wind": 27.8, "pressure": 1010.5}, jfk)]
    assert record_problems(WEATHER, records) == [] and table == (40.640, -73.779)


class FakeResponse:
    """A response of the NWS API as the requests library gives it: its status and the bytes of its body
    (a JSON document written as the API writes it, or text, or bytes, as they are)."""

    def __init__(self, body: object, status: int = 200) -> None:
        self.status_code = status
        if isinstance(body, bytes):
            self.content = body
        else:
            self.content = (body if isinstance(body, str) else json.dumps(body, indent=4)).encode("utf-8")


def test_fetching_asks_every_station_once_and_freezes_one_bundle(monkeypatch):
    calls = []
    answers = {"KJFK": FakeResponse({"type": "FeatureCollection", "features": [served("KJFK")]}),
               "KORD": FakeResponse("<html>", 200), "KSEA": FakeResponse(served("KSEA")),
               "KDEN": FakeResponse({}, 503)}

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, headers, timeout))
        return answers[url.split("/")[-2]]

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(wx, "_utc_now", lambda: FETCHED + timedelta(microseconds=1234))
    adapter = replace(WEATHER.live, stations=("KJFK", "KORD", "KSEA", "KDEN"))
    raw, url = adapter.fetch()
    assert [c[0] for c in calls] == [f"https://api.weather.gov/stations/{s}/observations" for s in adapter.stations]
    assert all(c[1:] == ({"start": "2026-11-07T02:40:00Z", "limit": 500},
                         {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}, 20.0) for c in calls)
    assert url == "https://api.weather.gov/stations/{station}/observations?start=2026-11-07T02:40:00Z&limit=500"
    bundle = json.loads(raw)
    assert (bundle["format"], bundle["fetched_at"], bundle["url"], bundle["user_agent"]) == (
        wx.BUNDLE_FORMAT, "2026-11-08T02:40:00Z", url, USER_AGENT)
    assert {station: response["status"] for station, response in bundle["stations"].items()} == {
        "KJFK": 200, "KORD": 200, "KSEA": 200, "KDEN": 503}  # every response kept, whatever its status
    assert bundle["stations"]["KORD"]["body"] == "<html>" and bundle["stations"]["KDEN"]["body"] == "{}"
    records, compiled_at, _ = adapter.parse(raw)
    assert [r.source for r in records] == ["KJFK", "KSEA"] and compiled_at == COMPILED_AT
    def unreachable(url, **kwargs):
        calls.append(url)
        raise requests.ConnectionError("Name or service not known")

    calls.clear()  # no network: the first station's failure ends the fetch at once
    monkeypatch.setattr(requests, "get", unreachable)
    with pytest.raises(requests.ConnectionError):
        adapter.fetch()
    assert len(calls) == 1
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse({}, 500))  # every station fails
    with pytest.raises(ValueError, match=r"no station answered with observations \(KJFK: HTTP 500; KORD: HTTP 500; "
                                         r"KSEA: HTTP 500; KDEN: HTTP 500\)"):
        adapter.fetch()


def test_a_fetch_freezes_each_body_exactly_as_served(monkeypatch):
    """The bundle keeps the text of every body as the API sent it (its layout, its escapes, its non-ASCII
    characters), so the snapshot's SHA-256 commits to what the NWS served rather than to evalgen's reading
    of it; the replay reads the same observations from it, and so does any reader of the frozen file."""
    document = served("KSEA")  # the FeatureCollection of the observations endpoint
    document["features"][0]["properties"]["stationName"] = "Seattle–Tacoma Int’l, été ✈"
    bodies = {"KSEA": json.dumps(document, indent=4, ensure_ascii=False).encode("utf-8"),
              "KJFK": json.dumps(served("KJFK"), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
              .replace(b'"Feature"', b'"\\u0046eature"', 1)}
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: FakeResponse(bodies[url.split("/")[-2]]))
    monkeypatch.setattr(wx, "_utc_now", lambda: FETCHED)
    adapter = replace(WEATHER.live, stations=("KSEA", "KJFK"), min_records=2)
    raw, _ = adapter.fetch()
    frozen = json.loads(raw)["stations"]
    assert {station: frozen[station]["body"].encode("utf-8") for station in bodies} == bodies  # byte for byte
    snapshot, again = build_domain_snapshot(Weather(live=adapter), "live", SeedLog(1), fetch=lambda: (raw, "nws://x"))
    assert again == raw and snapshot.raw_sha256 == sha256_hex(raw) and [r.source for r in snapshot.events] == [
        "KJFK", "KSEA"]
    assert adapter.parse(raw) == adapter.parse(bytes(raw))  # the replay reads exactly these records


def test_a_station_that_fails_on_its_own_is_recorded_and_the_others_are_kept(monkeypatch, offline):
    """A timeout reading one station's answer, or a connection dropped after the first station answered, is
    that station's failure: it is recorded in the bundle and every other station is still asked, so the
    snapshot stays live. Only a first request that cannot connect ends the fetch at once (the network is
    down, and every other request would wait for it too), and only a fetch that no station answered falls
    back to synthetic observations."""
    served_by = {s.id: [nws_feature(r) for r in WEATHER.synthetic(random.Random(8), 600, COMPILED_AT, 24)
                        if r.source == s.id] for s in STATIONS}
    failures = {"KBOS": requests.ReadTimeout("read timed out"), "KORD": requests.Timeout("too slow"),
                "KDEN": requests.ConnectionError("connection reset by peer")}
    calls: list[str] = []

    def get(url, **kwargs):
        station = url.split("/")[-2]
        calls.append(station)
        if station in failures:
            raise failures[station]
        return FakeResponse({"type": "FeatureCollection", "features": served_by[station]})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(wx, "_utc_now", lambda: FETCHED)
    assert STATIONS[0].id == "KBOS"  # the first request times out: not a missing network
    snapshot, raw = build_domain_snapshot(WEATHER, "live", SeedLog(2))
    assert calls == [s.id for s in STATIONS] and snapshot.source == "nws" and snapshot.note is None
    frozen = json.loads(raw)["stations"]
    assert {station: frozen[station] for station in failures} == {
        "KBOS": {"error": "ReadTimeout: read timed out"}, "KORD": {"error": "Timeout: too slow"},
        "KDEN": {"error": "ConnectionError: connection reset by peer"}}
    assert {r.source for r in snapshot.events} == {s.id for s in STATIONS} - set(failures)

    for first in (requests.ConnectionError("Name or service not known"), requests.ConnectTimeout("connect timed out")):
        calls.clear()
        failures["KBOS"] = first  # the first request cannot connect: no network, no other request
        snapshot, raw = build_domain_snapshot(WEATHER, "live", SeedLog(2))
        assert calls == ["KBOS"] and (snapshot.source, raw) == ("synthetic", None)
        assert f"nws unavailable ({type(first).__name__}: " in snapshot.note
    calls.clear()
    failures.update({s.id: requests.ReadTimeout("read timed out") for s in STATIONS})  # every station times out
    snapshot, raw = build_domain_snapshot(WEATHER, "live", SeedLog(2))
    assert calls == [s.id for s in STATIONS] and (snapshot.source, raw) == ("synthetic", None)
    assert snapshot.note.startswith("nws unavailable (ValueError: no station answered with observations (KBOS: "
                                    "ReadTimeout: read timed out; ") and f"and {len(STATIONS) - 4} more" in snapshot.note
    assert offline == []


def test_a_live_snapshot_is_frozen_hashed_replayed_and_falls_back(tmp_path, offline):
    live = Weather(live=replace(WEATHER.live, min_records=3))
    log = SeedLog(1)
    snapshot, raw = build_domain_snapshot(live, "live", log, fetch=lambda: (FIXTURE.read_bytes(), "nws://latest"))
    assert log.seeds == {} and raw == FIXTURE.read_bytes()  # no synthetic draw
    assert (snapshot.source, snapshot.domain, snapshot.url, snapshot.compiled_at, snapshot.window_hours) == (
        "nws", "weather", "nws://latest", COMPILED_AT, 24)
    assert snapshot.raw_sha256 == sha256_hex(raw) and len(snapshot.events) == 4
    directory = tmp_path / "run" / "snapshot"
    save_snapshot(snapshot, raw, directory, raw_filename=live.live.raw_filename)
    assert (directory / "raw_feed.json").read_bytes() == raw and load_snapshot(directory).events == snapshot.events
    replay, again = build_domain_snapshot(live, "nws", SeedLog(2), snapshot_file=str(directory / "raw_feed.json"))
    assert again == raw and replay.events_sha256 == snapshot.events_sha256
    assert (replay.url, replay.note) == ("nws://latest", "replay of frozen feed raw_feed.json (SHA-256 verified)")
    direct, _ = build_domain_snapshot(live, "live", SeedLog(2), snapshot_file=str(FIXTURE))  # the fixture itself
    assert direct.events == snapshot.events and direct.note == "frozen feed file raw_feed.json"

    def unreachable():
        raise requests.ConnectionError("no network")

    fallbacks = [(build_domain_snapshot(live, "live", SeedLog(3), fetch=unreachable), "ConnectionError: no network"),
                 (build_domain_snapshot(WEATHER, "live", SeedLog(3), fetch=lambda: (FIXTURE.read_bytes(), "x")),
                  "only 4 usable records"),
                 (build_domain_snapshot(live, "live", SeedLog(3), fetch=lambda: (b"<html>", "x")), "JSONDecodeError")]
    for (fallback, raw), why in fallbacks:
        assert (fallback.source, raw, fallback.window_hours) == ("synthetic", None, 24)
        assert why in fallback.note and "fell back to synthetic data" in fallback.note
        assert record_problems(WEATHER, fallback.events) == [] and len(fallback.events) == 360
    assert offline == []
    real, raw = build_domain_snapshot(WEATHER, "live", SeedLog(3))  # the adapter's own fetch, offline
    assert raw is None and real.source == "synthetic" and "nws unavailable (ConnectionError" in real.note
    assert len(offline) == 1 and offline[0].startswith("https://api.weather.gov/stations/KBOS/observations?start=")


def nws_feature(record: Record) -> dict:
    """A synthetic observation as the NWS API would serve it (test-made, for whole live runs)."""
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [record.lon, record.lat]},
            "properties": {"stationId": record.source, "timestamp": datetime.fromtimestamp(
                record.time, tz=timezone.utc).isoformat(), "rawMessage": "",
                "temperature": {"unitCode": "wmoUnit:degC", "value": record.values["temp"], "qualityControl": "V"},
                "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": record.values["wind"], "qualityControl": "V"},
                "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": record.values["pressure"] * 100,
                                       "qualityControl": "V"}}}


def test_a_live_run_is_frozen_and_its_replay_reproduces_it(tmp_path, monkeypatch, offline):
    served = WEATHER.synthetic(random.Random(8), 600, COMPILED_AT, 24)
    by_station = {s.id: [nws_feature(r) for r in served if r.source == s.id] for s in STATIONS}
    monkeypatch.setattr(wx, "_utc_now", lambda: FETCHED)
    with monkeypatch.context() as api:
        api.setattr(requests, "get", lambda url, **kwargs: FakeResponse(
            {"type": "FeatureCollection", "features": by_station[url.split("/")[-2]]}))
        live = pipeline.generate(GenerateConfig(seed=5, instances=3, mix="weather:classic", source="live"),
                                 tmp_path / "live", log=quiet)
    frozen = live / "snapshot" / "raw_feed.json"
    meta = read_json(live / "snapshot" / "snapshot.json")
    assert (meta["source"], meta["domain"], meta["raw_sha256"]) == ("nws", "weather", sha256_hex(frozen.read_bytes()))
    assert 300 < meta["n_events"] <= 24 * 25  # one observation per station and hour
    replay = pipeline.generate(GenerateConfig(seed=5, instances=3, mix="weather:classic", source="live",
                                              snapshot_file=str(frozen)), tmp_path / "replay", log=quiet)
    assert offline == []  # the replay never asked the API
    a, b = read_json(live / "manifest.json"), read_json(replay / "manifest.json")
    assert a["self_check"]["oracle_agreements"] == a["self_check"]["oracle_checks"] > 0
    assert b["snapshot"]["events_sha256"] == a["snapshot"]["events_sha256"] and b["replay"]["same_code"]
    for name in sorted(p.relative_to(live).as_posix() for p in (live / "instances").glob("*.json")) + [
            f"prompts/inst-00{i}/v{v}.txt" for i in range(3) for v in range(3)]:
        assert (replay / name).read_bytes() == (live / name).read_bytes(), name


# --------------------------------------------------------------------- whole runs: weather:classic


@pytest.fixture(scope="module")
def weather_run(tmp_path_factory):
    """generate (mix "weather:classic") -> run (oracle, naive, mock) -> score -> report: eight instances."""
    run_dir = pipeline.generate(GenerateConfig(seed=2611, instances=8, mix="weather:classic"),
                                tmp_path_factory.mktemp("runs") / "weather", log=quiet)
    pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)
    return run_dir


@pytest.fixture(scope="module")
def run_instances(weather_run):
    return load_instances(weather_run)


@pytest.fixture(scope="module")
def run_records(weather_run):
    return {r.id: r for r in load_snapshot(weather_run / "snapshot").events}


def test_a_run_is_generated_with_the_weather_glossary_rules_and_seed_paths(weather_run, run_instances, tmp_path):
    manifest = read_json(weather_run / "manifest.json")
    assert manifest["config"]["mix"] == "weather:classic" and manifest["snapshot"]["domain"] == "weather"
    assert manifest["self_check"]["oracle_agreements"] == manifest["self_check"]["oracle_checks"] > 0
    assert manifest["self_check"]["problems"] == []
    seeds = set(manifest["seeds"])
    assert {"snapshot/weather/clock", "snapshot/weather/synthetic", "templates/new/weather"} <= seeds
    assert not {"snapshot/clock", "snapshot/synthetic", "templates/new"} & seeds and not any("surface" in s for s in seeds)
    glossary = WEATHER.glossary()
    for inst in run_instances:
        assert inst.domain == "weather" and inst.surface is None and [r.genre for r in inst.renderings] == ["classic"] * 3
        for rendering in inst.renderings:
            prompt = rendering.prompt
            assert parse_glossary(prompt) == glossary and prompt.count("<glossary>") == 1
            assert prompt.index(WEATHER.rules_text()) < prompt.index("<glossary>") < prompt.index("<digest>")
            assert "reports about weather observations" in prompt and "Observation IDs, region names" in prompt
        path = weather_run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
        write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_every_rendering_reads_back_to_exactly_its_reports_and_notices(run_instances):
    glossary = WEATHER.glossary()
    for inst in run_instances:
        reports, notices = printed(inst.reports, inst.notices)
        for rendering in inst.renderings:
            assert extract_digest(rendering.prompt) == rendering.digest and detect_genre(rendering.digest) is CLASSIC
            data = CLASSIC.parse(rendering.digest, glossary, WEATHER)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), rendering.variant
        for line in inst.chatter:  # lines without data (clock times at most) change nothing
            assert CLASSIC.parse(line, glossary, WEATHER) == DigestData([], [])
            assert not re.search(r"\d\.\d|-\d{5}|°", line), line


def check_guards(q, records, ctx: Context, reports=()) -> None:
    """Every parameter of a question keeps its guard from every record of its instance. A question about the
    record nearest to a place has a clear answer, and its nearest record, or else the report nearest to the
    place (`reports`, the digest's, a record's or a planted one), lies within the domain's bound: a place
    moved near a report that then decides it (questions.reanchor_nearest) sets no bound on how far its key's
    nearest record lies."""
    units = WEATHER.units
    for raw in q.chain:
        step = Step(raw, units)
        if step.kind in ("at_least", "below"):
            spec, unit = WEATHER.field(step.field), step.unit(step["unit"])
            assert round_to_step(step["value"], unit.step or spec.threshold_step) == step["value"] or (
                spec.coarse_step and round_to_step(step["value"], spec.coarse_step) == step["value"])
            assert not (step["value"] == 0 and math.copysign(1.0, step["value"]) < 0)
            assert not spec.positive_threshold or step["value"] > 0
            threshold = canonical_threshold(step)
            margin = max(spec.margin[0], spec.margin[1] * abs(threshold))
            assert all(abs(r.values[spec.key] - threshold) >= margin for r in records), raw
        elif step.kind == "within_distance":
            radius = step.unit(step["unit"]).to_canonical(step["radius"])
            margin = max(WEATHER.spatial.margin[0], WEATHER.spatial.margin[1] * radius)
            assert all(abs(distance_km(raw, r) - radius) >= margin for r in records), raw
        elif step.kind == "time_between":
            assert step["start"] % ctx.time_step == 0 == step["end"] % ctx.time_step
            assert all(abs(r.time - edge) >= ctx.time_margin for r in records for edge in (step["start"], step["end"]))
    term = Step(terminal_step(q.chain), units)
    rows = rows_after_filters(q.chain, records, units=units)
    if term.kind in ("argmax", "rank", "time_of_max"):  # no tie down to the asked rank
        top = term["k"] + 1 if term.kind == "rank" else 2
        values = sorted((r.values[term.field] for r in rows), reverse=True)[:top]
        assert all(a - b >= WEATHER.field(term.field).rank_gap - 1e-9 for a, b in zip(values, values[1:]))
    if term.kind in ("nearest_to", "nearest_distance"):
        d = sorted(distance_km(term.raw, r) for r in rows)
        near = min((distance_km(term.raw, r) for r in rows_after_filters(q.chain, reports, units=units)),
                   default=d[0])
        assert min(near, d[0]) <= WEATHER.spatial.nearest_max_km and (len(d) < 2 or d[1] - d[0] >= max(
            WEATHER.spatial.nearest_gap[0], WEATHER.spatial.nearest_gap[1] * d[0]))


def test_questions_keep_their_guards_and_never_repeat(run_instances, run_records):
    fingerprints, units, templates = [], Counter(), Counter()
    for inst in run_instances:
        records = [run_records[record_id] for record_id in inst.event_ids]
        ctx = Context(inst.compiled_at, inst.window_hours)
        assert 3 <= len(inst.questions) <= 5
        for q in inst.questions:
            check_guards(q, records, ctx, inst.reports)
            assert execute(q.chain, records, units=WEATHER.units) == inst.answer_key[q.qid] is not None
            assert q.evidence == [r.id for r in rows_after_filters(q.chain, records, units=WEATHER.units)]
            fingerprints.append(fingerprint(q.chain))
            templates[q.template_id] += 1
            units.update(f"{raw['field']} in {raw['unit']}" for raw in q.chain if "unit" in raw and "field" in raw)
    assert len(fingerprints) == len(set(fingerprints))  # no question is asked twice in a run
    assert len(templates) >= 12 and any(t in templates for t in ("P02", "P07", "P09", "P12", "N02", "N07", "N09",
                                                                  "N12"))
    assert any(key.startswith("temp in") for key in units) and any(key.startswith("wind in") for key in units)


def test_traps_are_planted_in_weather_words_and_read_as_designed(run_instances, run_records):
    """A re-report filed under its observation's ID stands where its slot stands, at a station of the
    observation's region, which an opaque ID never contradicts: it is not pinned to the observation's own
    station, where it could never be the only report, as a real observation often is (traps._slot_pair)."""
    vocab, kinds, same_station = WEATHER.trap_vocab, Counter(), Counter()
    for inst in run_instances:
        records = [run_records[record_id] for record_id in inst.event_ids]
        by_id = {r.id: r for r in records}
        assert check_resolution(inst.reports, inst.notices, records, domain=WEATHER) == []
        # (a digest off the legacy path holds the traps its plan draws, of each kind none or more: traps.draw_plan)
        assert {t.kind for t in inst.traps} <= {"duplicate", "retraction", "conflicting_value"}
        data = DigestData(inst.reports, inst.notices)
        for trap in inst.traps:
            kinds[(trap.kind, trap.form)] += 1
            planted = [r for r in inst.reports if r.trap_id == trap.trap_id]
            if trap.kind == "duplicate":  # a feed repeats an observation: in the place of one the digest does not keep
                (dup,) = planted
                anchor = by_id[trap.target]
                station = next(s for s in STATIONS if s.name == dup.place)
                assert dup.source in (station.id, *vocab.rereporting_sources) and dup.status == "preliminary"
                assert (dup.lat, dup.lon, dup.place) == (station.lat, station.lon, station.name)
                assert dup.group == anchor.group and (dup.ref == anchor.id or dup.ref not in by_id)
                if trap.form == "same_id":  # under its observation's ID, at a station of the observation's region
                    assert dup.ref == anchor.id
                    same_station[dup.place == anchor.place] += 1
            elif trap.kind == "retraction":  # an observation the digest does not keep, withdrawn
                (phantom,) = planted
                (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                station = next(s for s in STATIONS if s.name == phantom.place)
                assert (phantom.lat, phantom.lon, phantom.place) == (station.lat, station.lon, station.name)
                assert phantom.source in (station.id, *vocab.rereporting_sources)
                assert phantom.ref not in by_id and notice.reason in vocab.retraction_reasons
            else:  # only temperatures are corrected
                assert planted[0].values["temp"] != by_id[trap.target].values["temp"]
                if trap.form == "correction":
                    (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                    assert (notice.field, notice.new_value) == ("temp", by_id[trap.target].values["temp"])
            for qid, fallen in trap.affected.items():  # each careless reading reproduces its wrong answer
                q = inst.question(qid)
                misread = [execute(q.chain, data, mishandle=Mishandle.from_dict(m), units=WEATHER.units,
                                   resolver=WEATHER.resolve) for m in trap.mishandles]
                assert fallen in misread and not same_answer(q.answer_type, fallen, inst.answer_key[qid], unit=q.unit)
        for q in inst.questions:  # the reading rules of the domain recover the key from the digest
            assert execute(q.chain, data, units=WEATHER.units, resolver=WEATHER.resolve) == inst.answer_key[q.qid]
    assert kinds[("duplicate", "same_id")] and kinds[("duplicate", "cross_ref")] and kinds[("retraction", "phantom")]
    assert same_station[False] > 0  # not always at its observation's station


def test_the_oracle_scores_100_percent_on_every_rendering_and_the_naive_reader_falls_for_traps(weather_run,
                                                                                               run_instances):
    rows = read_json(weather_run / "scores" / "oracle.json")["rows"]
    assert len(rows) == 3 * sum(len(inst.questions) for inst in run_instances) and all(row["correct"] for row in rows)
    assert {(row["domain"], row["genre"]) for row in rows} == {("weather", "classic")}
    for inst in run_instances:
        for rendering in inst.renderings:
            assert solve(rendering.prompt, question_specs(inst), domain="weather") == inst.answer_key
    models = {m["model"]: m for m in read_json(weather_run / "report.json")["models"]}
    assert models["oracle"]["accuracy"]["rate"] == models["oracle"]["traps"]["caught"]["rate"] == 1.0
    assert models["naive"]["accuracy"]["rate"] < 1.0 and 0 < models["mock"]["accuracy"]["rate"] < 1.0
    trapped = {(inst.instance_id, qid) for inst in run_instances for t in inst.traps for qid in t.fallen}
    for row in read_json(weather_run / "scores" / "naive.json")["rows"]:  # the naive reader errs only on traps
        assert row["correct"] or (row["instance_id"], row["qid"]) in trapped, row
    markdown = (weather_run / "report.md").read_text(encoding="utf-8")
    assert "conflicting value" in markdown and "weather observations" in markdown


def test_metamorphic_variants_restyle_the_digest_and_keep_the_key(run_instances):
    reworded = 0
    for inst in run_instances:
        original = inst.renderings[0]
        for variant in inst.renderings[1:]:
            assert variant.style != original.style and variant.digest != original.digest
            assert variant.prompt_sha256 != original.prompt_sha256
            assert set(variant.question_text) == set(original.question_text)
            reworded += sum(variant.question_text[q.qid] != original.question_text[q.qid] for q in inst.questions)
            assert solve(variant.prompt, question_specs(inst), domain="weather") == inst.answer_key
    assert reworded > 0


def test_a_run_regenerates_exactly_from_its_seed(weather_run, tmp_path):
    again = pipeline.generate(GenerateConfig(seed=2611, instances=8, mix="weather:classic"), tmp_path / "again",
                              log=quiet)
    for directory in ("instances", "prompts", "snapshot"):
        files = sorted(p.relative_to(weather_run).as_posix() for p in (weather_run / directory).rglob("*")
                       if p.is_file())
        assert files and all((again / f).read_bytes() == (weather_run / f).read_bytes() for f in files)
    assert read_json(again / "manifest.json")["seeds"] == read_json(weather_run / "manifest.json")["seeds"]


# ------------------------------------------------------------------- the classic genre and the rules


def styles(seed: int):
    """Every combination of line and time presets, with the other presets taken in turn."""
    rng = random.Random(seed)
    combos = [(a, b) for a in sorted(LINE_PRESETS) for b in sorted(RECORD_TIME_PRESETS)]
    for i, (lines, time) in enumerate(combos):
        style = {"units": sorted(UNIT_PRESETS)[i % 3], "time": time, "coords": sorted(RECORD_COORD_PRESETS)[i % 3],
                 "lines": lines, "layout": RECORD_LAYOUTS[i % 3], "notices": NOTICE_PLACEMENTS[i % 2],
                 "header": i % len(RECORD_HEADERS)}
        yield style, random.Random(rng.random())


def test_classic_reads_weather_digests_back_in_every_style(run_instances):
    glossary, symbols = WEATHER.glossary(), Counter()
    for inst in run_instances[:3]:
        reports, notices = printed(inst.reports, inst.notices)
        for style, rng in styles(inst.seed):
            text = CLASSIC.render(RenderContext(inst.reports, inst.notices, inst.chatter, inst.compiled_at,
                                                inst.window_hours, style, rng, WEATHER, glossary))
            data = CLASSIC.parse(text, glossary, WEATHER)
            assert same_items(data.reports, reports) and same_items(data.notices, notices), style
            assert text.startswith(("WEATHER OBSERVATIONS DIGEST", "Weather observations wire roundup",
                                    "== WEATHER OBSERVATIONS DIGEST"))
            symbols.update(re.findall(r"\d (°C|°F|K|km/h|mph|kt|m/s|hPa|inHg)\b", text))
    assert set(symbols) == {"°C", "°F", "K", "km/h", "mph", "kt", "m/s", "hPa", "inHg"}  # every unit is printed


def test_the_rules_name_every_notice_wording_and_add_the_weather_rule():
    rules = WEATHER.rules_text()
    flat = " ".join(rules.split()).lower()
    vocab = WEATHER.trap_vocab
    for phrase in (*vocab.retraction_notices, *vocab.correction_notices, "was not a real weather observation"):
        assert phrase.lower() in flat, phrase
    assert rules.splitlines()[-2].startswith("10. Temperatures are signed numbers") and rules.endswith("calm air.\n")
    assert all(len(line) <= 100 for line in rules.splitlines())
    glossary, record = WEATHER.glossary(), WEATHER.synthetic(random.Random(2), 1, COMPILED_AT, 24)[0]
    notices = [Notice("retraction", record.id, COMPILED_AT - 99, reason="sensor fault"),
               Notice("correction", record.id, COMPILED_AT - 9, field="temp", new_value=-1.5, old_value=4.0)]
    wordings = set()
    for seed in range(120):
        ctx = RenderContext([], notices, [], COMPILED_AT, 24, CLASSIC.choose_style(random.Random(seed), WEATHER),
                            random.Random(seed), WEATHER, glossary)
        wordings.update(line for line in CLASSIC.render(ctx).split("\n") if line.startswith(
            ("NOTICE", "RETRACTION", "UPDATE", "CORRECTION", "Revised")))
    for line in wordings:  # each wording reads back on its own and uses a phrase the rules quote, or a keyword
        (notice,) = CLASSIC.parse(line, glossary, WEATHER).notices
        assert replace(notice, time=0) in [replace(n, time=0) for n in notices], line
        phrases = vocab.retraction_notices if notice.kind == "retraction" else vocab.correction_notices
        # Every correction gives its value after a phrase the rules quote ("Revised ...: WX-... temperature now"
        # did not); a retraction may say RETRACTION and "was not a real weather observation" instead.
        assert any(p in line for p in phrases) or (notice.kind == "retraction" and line.startswith("RETRACTION")), line
    assert len(wordings) >= 15


def test_chatter_lines_carry_no_data():
    words = WEATHER.words
    for seed in range(20):
        for line in make_chatter(random.Random(seed), COMPILED_AT, ["KJFK", "MADIS"], WEATHER.chatter, words):
            assert "{" not in line and not re.search(r"\d\.\d|-\d{5}|°", line), line
            assert CLASSIC.parse(line, WEATHER.glossary(), WEATHER) == DigestData([], [])
    assert len(WEATHER.chatter) >= 3 and all("{" not in line.format(station="KJFK.ABC", h1="01:00", h2="02:00",
                                                                     date="9 September 2026", **words)
                                             for line in WEATHER.chatter)
