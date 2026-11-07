"""USGS mode, offline: feed parsing, frozen snapshots, fallback, and delayed-resolution forecasts."""
from __future__ import annotations

import json
import random

import requests

from evalgen import pipeline
from evalgen.seeds import SeedLog
from evalgen.sources import build_snapshot, generate_synthetic, parse_usgs_geojson, region_from_place
from evalgen.util import read_json, sha256_hex

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
