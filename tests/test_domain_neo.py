"""The neo domain (evalgen/domains/neo.py): close approaches of near-Earth objects.

Its fields, units, words and answer types; the seeded synthetic generator; the live adapter on a recorded
JPL CAD response (tests/fixtures/neo, never the network), frozen, hashed, replayed, and falling back to
synthetic approaches; and whole neo:classic runs, synthetic and replayed from the recording, in which
every report and notice reads back exactly, the guards hold, the traps behave, no question repeats, the
oracle scores 100% and metamorphic variants keep every answer."""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from evalgen import pipeline
from evalgen.domains import get_domain
from evalgen.domains.neo import (ASTRONOMICAL_UNIT, AU_KM, CAD_URL, CHATTER, KM_PER_H, LD_KM, LUNAR_DISTANCE,
                                 MILES_PER_HOUR, NEO, RULES, SOURCES, SYNTHETIC_WINDOW_DAYS, TRAP_VOCAB, half_month,
                                 parse_cad)
from evalgen.domains.base import RecordReport, record_rules, stored_places
from evalgen.genres import detect_genre
from evalgen.genres.base import parse_glossary, render_glossary
from evalgen.genres.classic import CLASSIC
from evalgen.models import Instance
from evalgen.ops import DigestData, Step, execute, rows_after_filters, terminal_step
from evalgen.pipeline import GenerateConfig, question_specs
from evalgen.questions import Context, fingerprint, public_templates, round_to_step
from evalgen.report import load_instances
from evalgen.rules import Mishandle
from evalgen.scoring import get_answer_type, is_correct, normalize, same_answer
from evalgen.seeds import SeedLog
from evalgen.solver import solve
from evalgen.sources import build_domain_snapshot, load_snapshot, record_problems, save_snapshot
from evalgen.traps import check_resolution
from evalgen.units import decode, encode
from evalgen.util import read_json, sha256_hex, write_json

from conftest import quiet

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "neo" / "cad_2026-11-08.json"
FIXTURE_BYTES = FIXTURE.read_bytes()
FIXTURE_SHA256 = "15803fc898697fdd24c6f246f5bea36b5e3f5765ebccd79ee22a80b25dedebe9"  # as captured on 2026-11-08
COMPILED_AT = 1_780_000_000
VOLATILE = {"run_id", "seed_commitment", "seed_nonce", "generated_at_utc"}


def utc(*parts: int) -> int:
    return int(datetime(*parts, tzinfo=timezone.utc).timestamp())


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


# ------------------------------------------------------------------------------- the domain


def test_neo_is_a_built_in_domain_without_a_location():
    assert get_domain("neo") is NEO
    NEO.validate()
    assert [spec.key for spec in NEO.fields] == ["dist", "v_rel", "h"] and not NEO.has_location
    assert NEO.roles == {"primary": "v_rel", "secondary": "dist"} and NEO.forecaster is None
    assert [t.template_id for t in public_templates(NEO)] == ["P01", "P03", "P04", "P05", "P06", "P08", "P10", "P11"]
    assert (NEO.noun, NEO.id_label, NEO.source_label, NEO.group_label, NEO.statuses) == (
        "close approach", "approach ID", "catalogue", "body", ("confirmed", "preliminary"))
    assert NEO.synthetic_window_hours == SYNTHETIC_WINDOW_DAYS * 24 == 720
    assert set(TRAP_VOCAB.rereporting_sources) <= set(SOURCES)  # every catalogue that mirrors others lists its own
    assert (NEO.live.source, NEO.live.raw_filename) == ("jpl", "raw_cad.json") and NEO.live.source in SOURCES


def test_answers_are_scored_alike_in_every_unit():
    distance, velocity = get_answer_type("neo_miss_distance"), get_answer_type("neo_velocity")
    assert (distance.tolerance, dict(distance.unit_tolerance)) == ((500.0, 0.005), {"au": (500.0 / AU_KM, 0.005),
                                                                                    "LD": (500.0 / LD_KM, 0.005)})
    assert (velocity.tolerance, dict(velocity.unit_tolerance)) == ((0.01, 0.002), {"km/h": (0.01 * 3600, 0.002),
                                                                                  "mph": (0.01 / 0.00044704, 0.002)})
    assert is_correct("neo_miss_distance", 2.268, "2.279 LD", unit="LD")  # within half a percent ...
    assert not is_correct("neo_miss_distance", 2.268, 2.281, unit="LD")  # ... and no further
    assert is_correct("neo_velocity", 45000.0, "45,090 km/h", unit="km/h")
    assert not is_correct("neo_velocity", 45000.0, 45100, unit="km/h")
    assert is_correct("neo_velocity", 4.91, 4.92, unit="km/s") and not is_correct("neo_velocity", 4.91, 4.93)
    assert normalize("neo_approach_id", "Approach 2026VK2-CA1.") == "2026vk2-ca1"
    assert normalize("neo_approach_id", "2026 VK2") != "2026vk2-ca1"  # a designation alone names no approach
    assert normalize("neo_body", "the Moon") == "moon" and normalize("neo_body", "Earth body") == "earth"


def test_units_convert_exactly_and_every_stored_value_reads_back():
    assert (ASTRONOMICAL_UNIT.to_canonical(1.0), LUNAR_DISTANCE.to_canonical(1.0)) == (149597870.7, 384400.0)
    assert KM_PER_H.to_canonical(3600.0) == 1.0 and MILES_PER_HOUR.to_canonical(1.0) == 0.00044704
    assert MILES_PER_HOUR.to_canonical(3600.0) == pytest.approx(1.609344)  # a mile is 1.609344 km
    assert encode(871745.0, 0, ASTRONOMICAL_UNIT) == "0.005827255" and encode(871745.0, 0, LUNAR_DISTANCE) == "2.267807"
    assert encode(4.91, 2, KM_PER_H) == "17676" and encode(4.91, 2, MILES_PER_HOUR) == "10983"
    rng = random.Random(3)
    for _ in range(400):
        distance, velocity = float(rng.randrange(75_000, 7_480_000)), round(rng.uniform(0.15, 55.0), 2)
        for spec, value in ((NEO.field("dist"), distance), (NEO.field("v_rel"), velocity)):
            for unit in spec.units:
                text = spec.encode(value, unit)
                assert spec.decode(text, unit) == value and "e" not in text, (value, unit, text)
    glossary = NEO.glossary()
    text = render_glossary(glossary)
    assert parse_glossary(text) == glossary and glossary.coord_decimals is None
    for conversion in ("× 149597870.7.", "× 384400.", "× 0.0002777777777777778.", "× 0.00044704."):
        assert conversion in text
    assert float("0.0002777777777777778") == 1 / 3600  # the glossary states the conversion the codec uses
    assert decode("2.267807", 0, LUNAR_DISTANCE) == 871745.0


def test_the_rules_name_every_notice_phrase_and_add_what_close_approaches_need():
    rules = NEO.rules_text()
    shared = record_rules(NEO)
    assert rules.startswith(shared) and rules.endswith("\n")
    numbered = re.compile(r"^(\d+)\. ", flags=re.MULTILINE)
    count = len(numbered.findall(shared)) + len(RULES)  # the shared rules, then this domain's, numbered on
    assert [int(n) for n in numbered.findall(rules)] == list(range(1, count + 1))
    flat = " ".join(rules.split())
    for phrase in (*TRAP_VOCAB.retraction_notices, *TRAP_VOCAB.correction_notices, "time of its close approach",
                   "2026vk2-ca1 and 2026vk2-ca2 are two close approaches"):
        assert phrase in flat, phrase
    assert all(len(line) <= 100 for line in rules.splitlines())
    for line in CHATTER:  # chatter carries no number, ID or value, and fills in nothing but the domain's words
        text = line.format(station="JPL.KTR", h1="01:00", h2="02:30", date="9 September 2026",
                           until="00:30 UTC on 10 September 2026", **NEO.words)
        assert not re.search(r"\d", re.sub(r"\{h[12]\}", "", line)) and "{" not in text
        assert not re.search(NEO.id_pattern, text.lower())


# ---------------------------------------------------------------------------- synthetic data


def test_the_synthetic_generator_is_determined_by_its_seed():
    a = NEO.synthetic(random.Random(8), 120, COMPILED_AT, 720)
    assert a == NEO.synthetic(random.Random(8), 120, COMPILED_AT, 720)
    assert a != NEO.synthetic(random.Random(9), 120, COMPILED_AT, 720)
    for n in (1, 2, 7, 120):
        assert len(NEO.synthetic(random.Random(n), n, COMPILED_AT, 720)) == n
    log = SeedLog(5)
    snapshot, raw = build_domain_snapshot(NEO, "synthetic", log, pool_size=90)
    assert raw is None and list(log.seeds) == ["snapshot/neo/clock", "snapshot/neo/synthetic"]
    assert build_domain_snapshot(NEO, "synthetic", SeedLog(5), pool_size=90)[0] == snapshot
    assert (snapshot.source, snapshot.domain, snapshot.window_hours) == ("synthetic", "neo", 720)
    assert len(snapshot.events) == 90 and record_problems(NEO, snapshot.events) == []


def test_synthetic_approaches_are_shaped_like_the_cad_lists():
    records = NEO.synthetic(random.Random(21), 600, COMPILED_AT, 720)
    assert record_problems(NEO, records) == [] and records == sorted(records, key=lambda r: (r.time, r.id))
    assert all(COMPILED_AT - 720 * 3600 < r.time <= COMPILED_AT - 60 for r in records)
    assert all(r.id == r.id.lower() and r.lat is None and r.qualifiers == {} for r in records)
    bodies = Counter(r.group for r in records)
    assert set(bodies) == {"Earth", "Moon", "Venus", "Mars", "Mercury"} and bodies.most_common(1)[0][0] == "Earth"
    assert 0.55 < bodies["Earth"] / len(records) < 0.8 and set(Counter(r.source for r in records)) == set(SOURCES)
    au = [r.values["dist"] / AU_KM for r in records]
    velocities, magnitudes = [r.values["v_rel"] for r in records], [r.values["h"] for r in records]
    assert 0.00029 < min(au) and max(au) <= 0.05 and 0.015 < statistics.median(au) < 0.035
    assert 0.15 <= min(velocities) and max(velocities) <= 55.0 and 6.0 < statistics.median(velocities) < 13.0
    assert 15.0 <= min(magnitudes) and max(magnitudes) <= 32.0 and 24.0 < statistics.median(magnitudes) < 27.5
    by_object: dict[str, list] = {}
    for r in records:
        by_object.setdefault(r.id.rsplit("-", 1)[0], []).append(r)
    pairs = [rs for rs in by_object.values() if len(rs) > 1]
    assert pairs and all(len(rs) == 2 for rs in pairs)
    for first, second in pairs:  # an object that passes Earth close passes the Moon within hours, as itself
        assert {first.group, second.group} == {"Earth", "Moon"} and (first.id[-4:], second.id[-4:]) == ("-ca1", "-ca2")
        assert first.time <= second.time <= first.time + 28 * 3600 and first.values["h"] == second.values["h"]
    assert all(rs[0].id.endswith("-ca1") for rs in by_object.values())
    year, last_half = datetime.fromtimestamp(COMPILED_AT, tz=timezone.utc).year, half_month(COMPILED_AT)
    found = [m for m in (re.fullmatch(r"(\d{4})([a-y])[a-z]\d*", d) for d in by_object) if m]
    assert len(found) > len(by_object) / 2
    assert all("abcdefghjklmnopqrstuvwxy".index(m[2]) <= last_half for m in found if int(m[1]) == year)
    assert all(int(m[1]) <= year for m in found)
    assert NEO.sample_weight(records[0]) >= 1.0
    assert NEO.sample_weight(replace(records[0], values={**records[0].values, "dist": 0.0})) == 4.0
    assert NEO.sample_weight(replace(records[0], values={**records[0].values, "dist": 0.05 * AU_KM})) == 1.0


def test_fresh_ids_are_new_objects_shaped_like_those_in_the_digest():
    digest = NEO.synthetic(random.Random(4), 24, COMPILED_AT, 720)
    taken = {r.id for r in digest}
    designations = {r.id.rsplit("-", 1)[0] for r in digest}
    fresh = [NEO.new_id("mpc", random.Random(seed), set(taken)) for seed in range(40)]
    assert fresh == [NEO.new_id("neocc", random.Random(seed), set(taken)) for seed in range(40)]  # any catalogue alike
    for ref in fresh:
        assert re.fullmatch(NEO.id_pattern, ref) and ref.endswith("-ca1") and ref.rsplit("-", 1)[0] not in designations
    grown = set(taken)
    ref = NEO.new_id("mpc", random.Random(0), grown)
    assert grown == taken | {ref}
    assert re.fullmatch(r"\d{5}-ca1", NEO.mimic_id("18109-ca1", random.Random(1), set()))
    assert re.fullmatch(r"2026[a-v][a-z]\d*-ca1", NEO.mimic_id("2026vk2-ca2", random.Random(1), set()))
    assert re.fullmatch(r"p2019[a-l][a-z]\d*-ca1", NEO.mimic_id("p2019ld2-ca1", random.Random(1), set()))
    assert re.fullmatch(r"[1-9]p-ca1", NEO.mimic_id("2p-ca1", random.Random(1), set()))


# ------------------------------------------------------------------------------ the live source


def test_the_live_adapter_reads_the_recorded_cad_response():
    assert sha256_hex(FIXTURE_BYTES) == FIXTURE_SHA256
    records, compiled_at, window_hours = NEO.live.parse(FIXTURE_BYTES)
    assert len(records) == 40 and record_problems(NEO, records) == []
    assert (compiled_at, window_hours) == (utc(2026, 12, 6), 28 * 24)  # the whole UTC days, 8 Nov to 5 Dec
    first = records[0]
    assert (first.id, first.group, first.source, first.values) == ("2026vk2-ca1", "Earth", "jpl",
                                                                   {"dist": 871745.0, "v_rel": 4.91, "h": 30.254})
    assert first.time == utc(2026, 11, 8, 1, 55, 43)  # JD 2461352.581157133 TDB (01:56:52) less 69.184 s
    by_id = {r.id: r for r in records}
    assert (by_id["2026vk2-ca2"].group, by_id["2026vk2-ca2"].values["dist"]) == ("Moon", 1051611.0)
    assert (by_id["2019xf2-ca1"].group, by_id["2019xf2-ca2"].group) == ("Moon", "Earth")  # numbered in time order
    assert (by_id["18109-ca1"].group, by_id["152637-ca1"].values["h"]) == ("Mars", 17.87)
    assert Counter(r.group for r in records) == {"Earth": 27, "Venus": 6, "Moon": 4, "Mars": 2, "Mercury": 1}
    assert max(r.values["v_rel"] for r in records) == 34.61 and min(r.values["v_rel"] for r in records) == 0.27


def test_the_live_adapter_keeps_only_rows_it_can_read_and_refuses_what_is_not_cad():
    doc = json.loads(FIXTURE_BYTES)
    fields = doc["fields"]
    row = dict(zip(fields, doc["data"][0]))
    rows = [
        {**row, "des": "C/2026 A1", "h": None},  # a comet: no H, so no record (no value is ever invented)
        {**row, "des": "2P", "body": "Merc", "h": "15.1"},  # a periodic comet, and a body's short name
        {**row, "des": "2024 AB1", "jd": "not a date"},
        {**row, "des": "2024 AB2", "dist": "-0.01"},
        {**row, "des": "2024 AB3", "v_rel": None},
        {**row, "des": "2024 AB4", "dist": "0.0123", "v_rel": "11.006", "h": "24.3456"},
    ]
    raw = json.dumps({**doc, "data": [[r[f] for f in fields] for r in rows] + ["not a row"]}).encode()
    records, _, _ = parse_cad(raw)
    assert [(r.id, r.group, r.values) for r in records] == [
        ("2024ab4-ca1", "Earth", {"dist": float(round(0.0123 * AU_KM)), "v_rel": 11.01, "h": 24.346}),
        ("2p-ca1", "Mercury", {"dist": 871745.0, "v_rel": 4.91, "h": 15.1}),
    ]
    assert record_problems(NEO, records) == []
    for broken, message in ((b"<html>busy</html>", "Expecting value"),
                            (b'{"code": "400", "message": "invalid"}', "no list of fields"),
                            (json.dumps({"fields": ["des", "jd"], "data": []}).encode(), "no dist, v_rel, body, h"),
                            (json.dumps({**doc, "count": 0, "data": None}).encode(), "no usable approach")):
        with pytest.raises(ValueError, match=message):
            parse_cad(broken)


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content, self.status_code = content, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")


@pytest.fixture
def cad_server(monkeypatch):
    """requests.get answers as the CAD API would, with what a test serves, and records every request: no test
    ever reaches the network."""
    calls: list[dict] = []
    served = SimpleNamespace(content=FIXTURE_BYTES, status=200, error=None)

    def fake_get(url, timeout=None, headers=None):
        calls.append({"url": url, "timeout": timeout, "headers": dict(headers or {})})
        if served.error is not None:
            raise served.error
        return FakeResponse(served.content, served.status)

    monkeypatch.setattr(requests, "get", fake_get)
    return SimpleNamespace(calls=calls, served=served)


def test_the_live_source_is_fetched_once_frozen_hashed_and_replayed(cad_server, tmp_path):
    log = SeedLog(1)
    snapshot, raw = build_domain_snapshot(NEO, "live", log)
    assert raw == FIXTURE_BYTES and log.seeds == {}  # no synthetic draw
    (call,) = cad_server.calls
    assert call["url"] == CAD_URL and call["timeout"] > 0 and call["headers"]["User-Agent"].startswith("evalgen/")
    assert "date-min=now" in CAD_URL and "dist-max=0.05" in CAD_URL and "body=ALL" in CAD_URL
    assert (snapshot.source, snapshot.domain, snapshot.url, snapshot.raw_sha256, snapshot.note) == (
        "jpl", "neo", CAD_URL, FIXTURE_SHA256, None)
    assert (snapshot.compiled_at, snapshot.window_hours, len(snapshot.events)) == (utc(2026, 12, 6), 672, 40)
    assert build_domain_snapshot(NEO, "jpl", SeedLog(1))[0].events == snapshot.events  # the source's own name
    directory = tmp_path / "run" / "snapshot"
    save_snapshot(snapshot, raw, directory, raw_filename=NEO.live.raw_filename)
    assert (directory / "raw_cad.json").read_bytes() == FIXTURE_BYTES
    assert load_snapshot(directory).events == snapshot.events
    replay, again = build_domain_snapshot(NEO, "live", SeedLog(2), snapshot_file=str(directory / "raw_cad.json"))
    assert again == raw and replay.events_sha256 == snapshot.events_sha256 and len(cad_server.calls) == 2
    assert (replay.url, replay.note) == (CAD_URL, "replay of frozen feed raw_cad.json (SHA-256 verified)")
    write_json(directory / "snapshot.json", dict(read_json(directory / "snapshot.json"), raw_sha256="0" * 64))
    assert "WARNING" in build_domain_snapshot(NEO, "live", SeedLog(2),
                                              snapshot_file=str(directory / "raw_cad.json"))[0].note


def test_the_live_source_falls_back_to_synthetic_approaches(cad_server):
    few = json.loads(FIXTURE_BYTES)
    few["data"] = few["data"][:12]
    cases = [
        (requests.ConnectionError("no network"), None, 200, "ConnectionError: no network"),
        (requests.Timeout("too slow"), None, 200, "Timeout: too slow"),
        (None, b"Service Unavailable", 503, "HTTPError: 503 Server Error"),
        (None, b"<html>not json</html>", 200, "JSONDecodeError"),
        (None, json.dumps(few).encode(), 200, "only 12 usable records"),
    ]
    for error, content, status, why in cases:
        cad_server.served.error, cad_server.served.content, cad_server.served.status = error, content, status
        snapshot, raw = build_domain_snapshot(NEO, "live", SeedLog(3))
        assert (snapshot.source, raw, snapshot.domain, snapshot.window_hours) == ("synthetic", None, "neo", 720)
        assert snapshot.note.startswith("jpl unavailable (") and why in snapshot.note, snapshot.note
        assert "fell back to synthetic data" in snapshot.note
        assert snapshot.events == build_domain_snapshot(NEO, "synthetic", SeedLog(3))[0].events

    def offline():
        raise requests.ConnectionError("unplugged")

    snapshot, raw = build_domain_snapshot(NEO, "live", SeedLog(3), fetch=offline)  # a fetch passed in instead
    assert (snapshot.source, raw) == ("synthetic", None) and "unplugged" in snapshot.note
    with pytest.raises(ValueError, match="unknown source 'usgs' for neo"):
        build_domain_snapshot(NEO, "usgs", SeedLog(3))


# ------------------------------------------------------------------------------ whole runs


def generate_and_answer(config: GenerateConfig, run_dir: Path) -> Path:
    pipeline.generate(config, run_dir, log=quiet)
    pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)
    return run_dir


@pytest.fixture(scope="module")
def neo_runs(tmp_path_factory):
    """mix neo:classic, generated, answered (oracle, naive, mock), scored and reported: six instances of a
    synthetic snapshot, and four of the recorded CAD response replayed as a frozen feed."""
    root = tmp_path_factory.mktemp("neo-runs")
    return {
        "synthetic": generate_and_answer(GenerateConfig(seed=41, instances=6, mix="neo:classic"), root / "synthetic"),
        "replay": generate_and_answer(GenerateConfig(seed=42, instances=4, mix="neo:classic",
                                                     snapshot_file=str(FIXTURE)), root / "replay"),
    }


def run_records(run: Path) -> dict:
    return {r.id: r for r in load_snapshot(run / "snapshot").events}


def test_neo_runs_live_on_seed_paths_of_their_own(neo_runs):
    for name, run in neo_runs.items():
        manifest = read_json(run / "manifest.json")
        check = manifest["self_check"]
        assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
        assert manifest["config"]["mix"] == "neo:classic" and manifest["snapshot"]["domain"] == "neo"
        seeds = set(manifest["seeds"])
        assert "templates/new/neo" in seeds and "templates/new" not in seeds and "snapshot/clock" not in seeds
        assert ("snapshot/neo/synthetic" in seeds) == (name == "synthetic")
        assert not any(path.endswith("/surface") for path in seeds)  # neo draws no surface words
        for inst in load_instances(run):
            assert (inst.domain, inst.surface) == ("neo", None)
            assert [r.genre for r in inst.renderings] == ["classic"] * 3
    replayed = read_json(neo_runs["replay"] / "manifest.json")["snapshot"]
    assert (replayed["source"], replayed["raw_sha256"], replayed["n_events"]) == ("jpl", FIXTURE_SHA256, 40)
    assert (neo_runs["replay"] / "snapshot" / "raw_cad.json").read_bytes() == FIXTURE_BYTES


def test_every_report_and_notice_reads_back_exactly(neo_runs, tmp_path):
    glossary = NEO.glossary()
    for run in neo_runs.values():
        for inst in load_instances(run):
            reports, notices = printed(inst.reports, inst.notices)
            assert all(isinstance(r, RecordReport) for r in inst.reports)
            for rendering in inst.renderings:
                assert parse_glossary(rendering.prompt) == glossary and NEO.rules_text() in rendering.prompt
                assert detect_genre(rendering.digest) is CLASSIC
                data = CLASSIC.parse(rendering.digest, glossary, NEO)
                assert same_items(data.reports, reports) and same_items(data.notices, notices), rendering.variant
            for chatter in inst.chatter:  # lines without data change nothing
                assert CLASSIC.parse(chatter, glossary, NEO) == DigestData([], [])
            path = run / "instances" / f"{inst.instance_id}.json"  # instance files round trip byte for byte
            write_json(tmp_path / path.name, Instance.from_dict(read_json(path)).to_dict())
            assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_the_oracle_scores_100_percent_on_every_rendering(neo_runs):
    for run in neo_runs.values():
        rows = read_json(run / "scores" / "oracle.json")["rows"]
        assert rows and all(row["correct"] for row in rows) and {row["domain"] for row in rows} == {"neo"}
        for inst in load_instances(run):
            for rendering in inst.renderings:  # read from the text alone, by sniffing, as the self-check reads it
                assert solve(rendering.prompt, question_specs(inst), domain="neo") == inst.answer_key


def assert_guarded(q, records, ctx: Context) -> None:
    """Every parameter of a question keeps its guard from every record of its instance, thresholds lie on
    their unit's steps, time windows on the context's marks, and ranked answers have no near-tie."""
    units = NEO.units
    for raw in q.chain:
        step = Step(raw, units)
        if step.kind in ("at_least", "below"):
            spec = NEO.field(step.field)
            symbol = step.get("unit")
            unit = None if symbol is None else step.unit(symbol)
            assert round_to_step(step["value"], spec.threshold_step if unit is None or unit.step is None
                                 else unit.step) == step["value"], raw
            threshold = step["value"] if unit is None else unit.to_canonical(step["value"])
            assert threshold > 0 or not spec.positive_threshold
            margin = max(spec.margin[0], spec.margin[1] * abs(threshold))
            assert all(abs(r.values[spec.key] - threshold) >= margin for r in records), raw
        elif step.kind == "time_between":
            assert step["start"] % ctx.time_step == 0 == step["end"] % ctx.time_step
            assert all(abs(r.time - edge) >= ctx.time_margin for r in records for edge in (step["start"], step["end"]))
    term = Step(terminal_step(q.chain), units)
    if term.kind in ("argmax", "rank", "time_of_max"):
        rows = rows_after_filters(q.chain, records, units=units)
        top = term["k"] + 1 if term.kind == "rank" else 2  # down to the asked rank, and just below it
        values = sorted((r.values[term.field] for r in rows), reverse=True)[:top]
        assert all(a - b >= NEO.field(term.field).rank_gap - 1e-9 for a, b in zip(values, values[1:])), q.chain


def test_the_guards_hold_and_no_question_repeats(neo_runs):
    templates: Counter = Counter()
    for name, run in neo_runs.items():
        manifest = read_json(run / "manifest.json")
        ctx = Context(manifest["snapshot"]["compiled_at"], manifest["snapshot"]["window_hours"])
        assert (ctx.time_step, ctx.time_margin) == ((10800, 3600) if name == "synthetic" else (10800, 3360))
        by_id = run_records(run)
        prints = []
        for inst in load_instances(run):
            records = [by_id[record_id] for record_id in inst.event_ids]
            assert 3 <= len(inst.questions) <= 5
            for q in inst.questions:
                assert_guarded(q, records, ctx)
                assert execute(q.chain, records, units=NEO.units) == inst.answer_key[q.qid] is not None
                prints.append(fingerprint(q.chain))
                templates[q.template_id] += 1
        assert len(prints) == len(set(prints))  # no question is asked twice in a run
    assert not set(templates) & {"P02", "P07", "P09", "P12", "N02", "N07", "N09", "N12"}  # no location, no space
    assert {t[0] for t in templates} == {"P", "N"} and len(templates) >= 8


def test_the_traps_are_planted_settled_and_misread_as_designed(neo_runs):
    kinds: Counter = Counter()
    for run in neo_runs.values():
        by_id = run_records(run)
        for inst in load_instances(run):
            records = [by_id[record_id] for record_id in inst.event_ids]
            assert check_resolution(inst.reports, inst.notices, records, domain=NEO) == []
            data = DigestData(inst.reports, inst.notices)
            for trap in inst.traps:
                kinds[(trap.kind, trap.form)] += 1
                planted = [r for r in inst.reports if r.trap_id == trap.trap_id]
                if trap.kind == "duplicate":  # an approach the digest does not keep, filed as a re-report
                    (dup,) = planted
                    record = by_id[trap.target]
                    assert dup.status == "preliminary" and dup.source in SOURCES
                    assert (dup.same_as is None) == (trap.form == "same_id") and dup.group == record.group
                    if trap.form == "same_id":  # the object's H as another catalogue fits it, to as many places,
                        # of an object that makes no other approach in the digest (traps._with_siblings)
                        h, places = record.values["h"], stored_places(record.values["h"], 3)
                        assert stored_places(dup.values["h"], 3) <= places and 0.05 - 1e-9 <= abs(dup.values["h"] - h)
                        assert abs(dup.values["h"] - h) <= 0.3 + 10 ** -places
                        stem = dup.ref.rpartition("-")[0]
                        assert [r.ref for r in inst.reports if r.ref.rpartition("-")[0] == stem
                                and r.ref != dup.ref] == []
                    else:  # under another designation, a real one, with its own H
                        assert dup.ref in by_id and dup.ref not in inst.event_ids
                        assert dup.values == by_id[dup.ref].values
                elif trap.kind == "retraction":
                    (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                    assert notice.reason in TRAP_VOCAB.retraction_reasons and planted[0].role == "phantom"
                    assert notice.target == planted[0].ref and re.fullmatch(NEO.id_pattern, planted[0].ref)
                elif trap.form == "correction":
                    (notice,) = [n for n in inst.notices if n.trap_id == trap.trap_id]
                    assert notice.field in ("dist", "v_rel") and planted[0].values[notice.field] == notice.old_value
                    assert notice.new_value == by_id[trap.target].values[notice.field] != notice.old_value
                else:
                    assert trap.kind == "conflicting_value" and planted[0].ref == trap.target
                for qid, wrong in trap.affected.items():  # each careless reading gives its recorded wrong answer
                    q = inst.question(qid)
                    misread = [execute(q.chain, data, mishandle=Mishandle.from_dict(m), units=NEO.units,
                                       resolver=NEO.resolve) for m in trap.mishandles]
                    assert wrong in misread and not same_answer(q.answer_type, wrong, inst.answer_key[qid], unit=q.unit)
    assert {kind for kind, _ in kinds} == {"duplicate", "retraction", "conflicting_value"}
    assert kinds[("duplicate", "same_id")] and kinds[("duplicate", "cross_ref")]
    for run in neo_runs.values():
        models = {m["model"]: m for m in read_json(run / "report.json")["models"]}
        assert models["oracle"]["traps"]["caught"]["rate"] == 1.0 == models["oracle"]["accuracy"]["rate"]
        assert models["naive"]["accuracy"]["rate"] < 1.0 and 0 < models["mock"]["accuracy"]["rate"] < 1.0
        trapped = {(inst.instance_id, qid) for inst in load_instances(run) for t in inst.traps for qid in t.fallen}
        for row in read_json(run / "scores" / "naive.json")["rows"]:  # the naive reader errs only where a trap is
            assert row["correct"] or (row["instance_id"], row["qid"]) in trapped, row


def test_metamorphic_variants_restyle_and_reword_but_keep_every_answer(neo_runs, tmp_path):
    run = generate_and_answer(GenerateConfig(seed=43, instances=3, variants=4, mix="neo:classic"), tmp_path / "v4")
    restated = 0
    for inst in load_instances(run) + load_instances(neo_runs["synthetic"]):
        original = inst.renderings[0]
        for variant in inst.renderings[1:]:
            for key in ("units", "time", "lines", "layout"):  # a variant in the original's genre avoids its look
                assert variant.style[key] != original.style[key], (variant.variant, key)
            assert solve(variant.prompt, question_specs(inst), domain="neo") == inst.answer_key
            for q in inst.questions:
                restated += _restated(q, variant.question_text[q.qid])
    assert restated > 0  # some variants state a threshold in another unit
    scores = read_json(run / "scores" / "oracle.json")["rows"]  # five renderings of every question, all right
    assert len(scores) == 5 * sum(len(inst.questions) for inst in load_instances(run))
    assert all(row["correct"] for row in scores)


def _restated(q, text: str) -> int:
    """How many of a question's thresholds `text` states in the next unit of their field (a metamorphic variant's
    wording), rather than as the chain has them."""
    count = 0
    for raw in q.chain:
        step = Step(raw, NEO.units)
        if step.kind in ("at_least", "below") and step.get("unit") is not None:
            units = list(step.units())
            unit = step.unit(step["unit"])
            other = units[(units.index(unit) + 1) % len(units)]
            places = other.places(NEO.field(step.field).decimals)
            shown = f"{other.from_canonical(unit.to_canonical(step['value'])):.{places}f}"
            count += f"{shown} {other.words_for(shown)}" in text
    return count


def test_runs_are_determined_by_their_seed(neo_runs, tmp_path):
    """The same seed gives the same run: from a synthetic snapshot, and from the frozen bytes a replayed run
    saved (which replay as the recording they were read from did)."""
    saved = neo_runs["replay"] / "snapshot" / "raw_cad.json"
    for name, config in (("synthetic", GenerateConfig(seed=41, instances=6, mix="neo:classic")),
                         ("replay", GenerateConfig(seed=42, instances=4, mix="neo:classic", snapshot_file=str(saved)))):
        again = pipeline.generate(config, tmp_path / name, log=quiet)
        first = neo_runs[name]
        for path in sorted((first / "instances").glob("*.json")) + sorted((first / "prompts").rglob("*.txt")):
            assert (again / path.relative_to(first)).read_bytes() == path.read_bytes(), path.name
        assert (again / "snapshot" / "events.json").read_bytes() == (first / "snapshot" / "events.json").read_bytes()
        a, b = (read_json(run / "manifest.json") for run in (first, again))
        for manifest in (a, b):  # what names the frozen file and when it was read, apart
            manifest["config"].pop("snapshot_file", None)
            for key in (*VOLATILE, "snapshot", "replay"):
                manifest.pop(key)
        assert a == b
    replayed = read_json(tmp_path / "replay" / "manifest.json")["snapshot"]
    assert replayed["note"] == "replay of frozen feed raw_cad.json (SHA-256 verified)"
    assert (replayed["url"], replayed["raw_sha256"]) == ("file:" + FIXTURE.name, FIXTURE_SHA256)
    other = pipeline.generate(GenerateConfig(seed=44, instances=2, mix="neo:classic"), tmp_path / "other", log=quiet)
    assert (other / "instances" / "inst-000.json").read_bytes() != (
        neo_runs["synthetic"] / "instances" / "inst-000.json").read_bytes()


def test_the_legacy_path_never_draws_for_neo(small_run):
    manifest = read_json(small_run / "manifest.json")
    assert "mix" not in manifest["config"] and "domain" not in manifest["snapshot"]
    assert not any("neo" in path for path in manifest["seeds"])
