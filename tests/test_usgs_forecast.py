"""USGS mode, offline: feed parsing, frozen snapshots, fallback, and delayed-resolution forecasts; and the
day feed as the earthquake domain's live source off the legacy path (UsgsFeed), on a fake transport."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import requests

from evalgen import pipeline
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.seeds import SeedLog
from evalgen.sources import (USER_AGENT, USGS_FEEDS, build_domain_snapshot, build_snapshot, generate_synthetic,
                             load_snapshot, parse_usgs_geojson, region_from_place, save_snapshot)
from evalgen.util import read_json, sha256_hex, write_json

from conftest import quiet


def fake_feed(n: int = 80, seed: int = 3) -> tuple[bytes, int]:
    """A GeoJSON document shaped exactly like the USGS summary feed (generated, not real data)."""
    generated = 1_793_736_000
    rng = random.Random(seed)
    features = []
    for e in generate_synthetic(rng, n, generated):
        features.append({
            "type": "Feature", "id": e.id,
            "properties": {"mag": e.mag, "place": f"{rng.randint(2, 60)} km NE of Somewhere, {e.region}",
                           "time": e.time * 1000 + 123, "magType": e.mag_type, "type": "earthquake", "net": e.network,
                           "status": "reviewed"},
            "geometry": {"type": "Point", "coordinates": [e.lon, e.lat, e.depth_km]},
        })
    features.append({"type": "Feature", "id": "uu80000001", "properties": {"mag": 1.2, "place": "3 km N of X, Utah",
                     "time": generated * 1000 - 5000, "magType": "md", "type": "quarry blast", "net": "uu"},
                     "geometry": {"coordinates": [-112.0, 39.0, 0.0]}})
    features.append({"type": "Feature", "id": "ak0000001x", "properties": {"mag": None, "place": "Alaska",
                     "time": generated * 1000 - 9000, "magType": "ml", "type": "earthquake", "net": "ak"},
                     "geometry": {"coordinates": [-150.0, 61.0, 10.0]}})
    doc = {"type": "FeatureCollection", "metadata": {"generated": generated * 1000, "count": len(features)},
           "features": features}
    return json.dumps(doc).encode(), generated


def test_parse_feed_keeps_only_usable_earthquakes():
    raw, generated = fake_feed()
    events, compiled_at, dropped = parse_usgs_geojson(raw)
    assert compiled_at == generated
    assert len(events) + len(dropped) == 80  # the quarry blast and the magnitude-less event are dropped
    assert all(e.time < generated for e in events)
    assert all(e.place and e.region for e in events)


def _feature(event_id, t, lat, lon, mag, status, net, sig=100):
    return {"type": "Feature", "id": event_id,
            "properties": {"mag": mag, "place": "10 km N of Town, Alaska", "time": t * 1000, "magType": "ml",
                           "type": "earthquake", "net": net, "status": status, "sig": sig},
            "geometry": {"coordinates": [lon, lat, 10.0]}}


def test_duplicate_feed_solutions_of_one_quake_are_collapsed():
    """Live feeds list some quakes twice (automatic + reviewed, or regional + global solution)."""
    t = 1_793_700_000
    doc = {"metadata": {"generated": (t + 3600) * 1000}, "features": [
        _feature("aka2026vudyyh", t, 59.540, -152.810, 2.6, "automatic", "ak"),
        _feature("aka2026vudsgu", t, 59.550, -152.820, 2.7, "reviewed", "ak"),         # same quake, 1.2 km away
        _feature("ci41571744", t + 500, 32.404, -115.165, 3.28, "reviewed", "ci", 170),
        _feature("us7000tv2m", t + 500, 32.442, -115.176, 3.1, "reviewed", "us", 150),  # NEIC solution, 4.4 km
        _feature("nc75000001", t + 900, 38.800, -122.800, 1.1, "reviewed", "nc"),
        _feature("nc75000002", t + 907, 38.802, -122.801, 1.3, "reviewed", "nc"),       # 7 s later: a real doublet
    ]}
    events, _, dropped = parse_usgs_geojson(json.dumps(doc).encode())
    assert sorted(e.id for e in events) == ["aka2026vudsgu", "ci41571744", "nc75000001", "nc75000002"]
    assert sorted(dropped) == ["aka2026vudyyh", "us7000tv2m"]


def test_region_from_place():
    assert region_from_place("6 km NE of Delta, B.C., MX") == "Mexico"
    assert region_from_place("12 km NE of Cabazon, CA") == "California"
    assert region_from_place("Southern Alaska") == "Alaska"
    assert region_from_place("Japan region") == "Japan"
    assert region_from_place("8 km SSE of Maria Antonia, Puerto Rico") == "Puerto Rico"
    assert region_from_place(None) == "Unknown region"
    # compass words that are part of a proper name are kept; relative positions stay distinct
    assert region_from_place("3 km W of Logan, West Virginia") == "West Virginia"
    assert region_from_place("South Sandwich Islands region") == "South Sandwich Islands"
    assert region_from_place("12 km E of Gyeongju, South Korea") == "South Korea"
    assert region_from_place("south of the Fiji Islands") == "South of the Fiji Islands"
    assert region_from_place("south of Panama") == "South of Panama"
    assert region_from_place("Northern Mariana Islands") == "Northern Mariana Islands"
    assert region_from_place("off the coast of Central America") == "Central America"
    assert region_from_place("near the coast of Ecuador") == "Ecuador"
    assert region_from_place("near the coast of central Chile") == "Chile"
    assert region_from_place("off the east coast of Kamchatka") == "Kamchatka"


def test_frozen_snapshot_file_is_hashed(tmp_path):
    raw, _ = fake_feed()
    path = tmp_path / "all_day.geojson"
    path.write_bytes(raw)
    snapshot, saved = build_snapshot("usgs", SeedLog(1), snapshot_file=str(path))
    assert snapshot.source == "usgs" and saved == raw and snapshot.raw_sha256 == sha256_hex(raw)
    again, _ = build_snapshot("usgs", SeedLog(1), snapshot_file=str(path))
    assert again.events_sha256 == snapshot.events_sha256


def test_replaying_a_frozen_usgs_snapshot_reproduces_the_run(tmp_path):
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    first = pipeline.generate(pipeline.GenerateConfig(seed=21, instances=2, source="usgs", snapshot_file=str(feed)),
                              tmp_path / "first", log=quiet)
    frozen = first / "snapshot" / "raw_feed.geojson"
    replay = pipeline.generate(pipeline.GenerateConfig(seed=21, instances=2, source="usgs", snapshot_file=str(frozen)),
                               tmp_path / "replay", log=quiet)
    for sub in ("instances", "prompts"):
        a = {p.relative_to(first): p.read_bytes() for p in (first / sub).rglob("*") if p.is_file()}
        b = {p.relative_to(replay): p.read_bytes() for p in (replay / sub).rglob("*") if p.is_file()}
        assert a and a == b
    assert "SHA-256 verified" in read_json(replay / "manifest.json")["snapshot"]["note"]


def test_network_failure_falls_back_to_synthetic():
    def offline(feed):
        raise requests.ConnectionError("no network")

    snapshot, raw = build_snapshot("usgs", SeedLog(5), fetch=offline)
    assert snapshot.source == "synthetic" and raw is None
    assert "fell back to synthetic" in snapshot.note


def _usgs_run(tmp_path):
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    run_dir = tmp_path / "usgs-run"
    cfg = pipeline.GenerateConfig(seed=11, instances=3, source="usgs", snapshot_file=str(feed))
    pipeline.generate(cfg, run_dir, log=quiet)
    pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)
    return run_dir


def test_usgs_run_has_pending_forecasts(tmp_path):
    run_dir = _usgs_run(tmp_path)
    forecasts = read_json(run_dir / "forecasts.json")
    assert len(forecasts) == 3 and all(f["status"] == "pending" for f in forecasts)
    assert "F1. Forecast" in (run_dir / "prompts" / "inst-000" / "v0.txt").read_text()
    check = read_json(run_dir / "manifest.json")["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"]
    assert "still pending" in (run_dir / "report.md").read_text()


def test_every_usgs_report_line_carries_a_place_note(tmp_path):
    """Planted reports (the phantom included) must not be the only lines without a location note."""
    run_dir = _usgs_run(tmp_path)
    for path in sorted((run_dir / "prompts").rglob("*.txt")):
        digest = path.read_text().split("<digest>")[1].split("</digest>")[0]
        for line in digest.splitlines():
            if line.startswith("[") and " | " in line:
                assert "near: " in line, line
            elif " earthquake struck " in line:
                assert "Location note: " in line, line


def test_resolve_grades_forecasts_with_brier(tmp_path):
    run_dir = _usgs_run(tmp_path)
    end = read_json(run_dir / "forecasts.json")[0]["rule"]["end"]

    summary = pipeline.resolve_run(run_dir, now=end - 10, fetch=lambda rule: [], log=quiet)
    assert summary["newly_resolved"] == 0 and summary["pending"] == 3  # window not closed yet

    hits = iter([[{"id": "us7000zzzz", "properties": {"mag": 4.8}}], [], []])
    summary = pipeline.resolve_run(run_dir, now=end + 3 * 3600, fetch=lambda rule: next(hits), log=quiet)
    assert summary["newly_resolved"] == 3
    forecasts = read_json(run_dir / "forecasts.json")
    assert [f["outcome"] for f in forecasts] == [1, 0, 0]
    models = {m["model"]: m for m in read_json(run_dir / "report.json")["models"]}
    for name in ("oracle", "naive", "mock"):
        assert models[name]["forecast"]["resolved"] == 3
        assert 0.0 <= models[name]["forecast"]["brier"] <= 1.0
    assert models["naive"]["forecast"]["brier"] == 0.25  # it always answers 0.5
    text = (run_dir / "report.md").read_text()
    assert "All 3 forecast question(s) resolved" in text and "still pending" not in text


# ------------------------------------------------- the live source off the legacy path (UsgsFeed)


def feed_from(url: str | None, n: int = 80) -> bytes:
    """A fake USGS summary feed (fake_feed) whose metadata names the feed it came from."""
    doc = json.loads(fake_feed(n)[0])
    if url is not None:
        doc["metadata"]["url"] = url
    return json.dumps(doc).encode()


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content, self.status_code = content, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")


@pytest.fixture
def usgs_server(monkeypatch):
    """requests.get answers as the USGS feed would, with what a test serves, and records every request: no
    test ever reaches the network."""
    calls: list[dict] = []
    served = SimpleNamespace(content=feed_from(USGS_FEEDS["day"]), status=200, error=None)

    def fake_get(url, timeout=None, headers=None):
        calls.append({"url": url, "timeout": timeout, "headers": dict(headers or {})})
        if served.error is not None:
            raise served.error
        return FakeResponse(served.content, served.status)

    monkeypatch.setattr(requests, "get", fake_get)
    return SimpleNamespace(calls=calls, served=served)


def test_the_usgs_live_source_is_fetched_once_frozen_hashed_and_replayed(usgs_server, tmp_path):
    log = SeedLog(1)
    snapshot, raw = build_domain_snapshot(EARTHQUAKES, "live", log)
    assert raw == usgs_server.served.content and log.seeds == {}  # the feed as served; no synthetic draw
    (call,) = usgs_server.calls
    assert call["url"] == USGS_FEEDS["day"] and call["timeout"] > 0 and call["headers"] == {"User-Agent": USER_AGENT}
    events, generated, _ = parse_usgs_geojson(raw)
    assert (snapshot.source, snapshot.domain, snapshot.url, snapshot.raw_sha256, snapshot.note) == (
        "usgs", "earthquakes", USGS_FEEDS["day"], sha256_hex(raw), None)
    assert (snapshot.compiled_at, snapshot.window_hours, snapshot.events) == (generated, 24, events)
    assert build_domain_snapshot(EARTHQUAKES, "usgs", SeedLog(1))[0].events == snapshot.events  # the source's name
    directory = tmp_path / "run" / "snapshot"
    save_snapshot(snapshot, raw, directory, raw_filename=EARTHQUAKES.live.raw_filename)
    frozen = directory / "raw_feed.geojson"
    assert frozen.read_bytes() == raw and load_snapshot(directory).events == snapshot.events
    replay, again = build_domain_snapshot(EARTHQUAKES, "live", SeedLog(2), snapshot_file=str(frozen))
    assert again == raw and replay.events == snapshot.events and replay.events_sha256 == snapshot.events_sha256
    assert (replay.url, replay.window_hours, replay.note) == (USGS_FEEDS["day"], 24,
                                                              "replay of frozen feed raw_feed.geojson (SHA-256 verified)")
    assert len(usgs_server.calls) == 2  # the replay asked nothing
    write_json(directory / "snapshot.json", dict(read_json(directory / "snapshot.json"), raw_sha256="0" * 64))
    assert "WARNING" in build_domain_snapshot(EARTHQUAKES, "live", SeedLog(2), snapshot_file=str(frozen))[0].note


def test_the_usgs_feed_window_is_the_one_its_metadata_names(usgs_server, tmp_path):
    """A feed covers the hour or the day that its metadata.url names (the day feed when it names none), live
    or frozen as a file of its own."""
    for url, window in ((USGS_FEEDS["hour"], 1), (USGS_FEEDS["day"], 24), (None, 24)):
        usgs_server.served.content = feed_from(url)
        snapshot, raw = build_domain_snapshot(EARTHQUAKES, "live", SeedLog(1))
        assert (snapshot.source, snapshot.window_hours) == ("usgs", window) == ("usgs", EARTHQUAKES.live.parse(raw)[2])
        path = tmp_path / f"feed-{window}-{url is None}.geojson"
        path.write_bytes(raw)
        replay, _ = build_domain_snapshot(EARTHQUAKES, "live", SeedLog(1), snapshot_file=str(path))
        assert (replay.window_hours, replay.url, replay.note) == (window, f"file:{path.name}",
                                                                  f"frozen feed file {path.name}")


def test_the_usgs_live_source_falls_back_to_synthetic_earthquakes(usgs_server):
    cases = [
        (requests.ConnectionError("no network"), None, 200, "ConnectionError: no network"),
        (requests.Timeout("too slow"), None, 200, "Timeout: too slow"),
        (None, b"Service Unavailable", 503, "HTTPError: 503 Server Error"),
        (None, b"<html>not json</html>", 200, "JSONDecodeError"),
        (None, feed_from(USGS_FEEDS["day"], n=8), 200, "only 8 usable records"),
    ]
    synthetic = build_domain_snapshot(EARTHQUAKES, "synthetic", SeedLog(3))[0]
    for error, content, status, why in cases:
        usgs_server.served.error, usgs_server.served.content, usgs_server.served.status = error, content, status
        snapshot, raw = build_domain_snapshot(EARTHQUAKES, "live", SeedLog(3))
        assert (snapshot.source, raw, snapshot.domain) == ("synthetic", None, "earthquakes")
        assert snapshot.note.startswith("usgs unavailable (") and why in snapshot.note, snapshot.note
        assert "fell back to synthetic data" in snapshot.note and snapshot.events == synthetic.events
    assert len(usgs_server.calls) == len(cases)  # one request each, never retried


def test_a_usgs_run_off_the_legacy_path_is_live_frozen_and_replays_exactly(usgs_server, tmp_path, monkeypatch):
    """mix earthquakes:news with the USGS source: the day feed is fetched once with evalgen's User-Agent,
    frozen and hashed, the instances ask their forecast question, and a replay of the frozen feed with the
    network blocked regenerates every instance and prompt byte for byte."""
    cfg = pipeline.GenerateConfig(seed=17, instances=3, mix="earthquakes:news", source="usgs")
    live = pipeline.generate(cfg, tmp_path / "live", log=quiet)
    (call,) = usgs_server.calls
    assert call["url"] == USGS_FEEDS["day"] and call["headers"]["User-Agent"] == USER_AGENT
    frozen = live / "snapshot" / "raw_feed.geojson"
    meta = read_json(live / "snapshot" / "snapshot.json")
    assert frozen.read_bytes() == usgs_server.served.content
    assert (meta["source"], meta["domain"], meta["url"], meta["window_hours"], meta["raw_sha256"]) == (
        "usgs", "earthquakes", USGS_FEEDS["day"], 24, sha256_hex(frozen.read_bytes()))
    forecasts = read_json(live / "forecasts.json")
    assert len(forecasts) == 3 and {f["domain"] for f in forecasts} == {"earthquakes"}
    assert all("F1. Forecast" in (live / "prompts" / f"inst-00{i}" / "v0.txt").read_text() for i in range(3))
    check = read_json(live / "manifest.json")["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0

    def unreachable(*args, **kwargs):
        raise requests.ConnectionError("offline: a replay never asks the network")

    monkeypatch.setattr(requests, "get", unreachable)
    replay = pipeline.generate(pipeline.GenerateConfig(seed=17, instances=3, mix="earthquakes:news", source="usgs",
                                                       snapshot_file=str(frozen)), tmp_path / "replay", log=quiet)
    names = sorted(p.relative_to(live).as_posix() for sub in ("instances", "prompts") for p in (live / sub).rglob("*")
                   if p.is_file())
    assert names and all((replay / name).read_bytes() == (live / name).read_bytes() for name in names)
    note = read_json(replay / "manifest.json")["snapshot"]["note"]
    assert note == "replay of frozen feed raw_feed.geojson (SHA-256 verified)"
    usgs_server.served.error = requests.ConnectionError("no network")  # and offline, the run is synthetic
    monkeypatch.setattr(requests, "get", unreachable)
    offline = pipeline.generate(cfg, tmp_path / "offline", log=quiet)
    meta = read_json(offline / "snapshot" / "snapshot.json")
    assert meta["source"] == "synthetic" and meta["note"].startswith("usgs unavailable (ConnectionError")
    assert not (offline / "forecasts.json").exists() and not (offline / "snapshot" / "raw_feed.geojson").exists()
