"""The final audit of the live sources and their replays (0.3.0): what each finding showed, as a test.

Live requests go through a session that trusts nothing in the environment but the proxies it names, and
are bounded as a whole (sweep-network#8, #9); bytes that no parser can read, or values that cannot be real,
fall back or are dropped (#6, #7, #10, code-weather#2, code-neo#2); a replay is verified against every hash
it can find, and what no hash covers is checked against the bytes or the source run (#3, #4, #5); a replay
records no local path off the legacy path (#11); frozen inputs that a run off the legacy path cannot replay
are refused before anything is written (code-pipeline#2, code-traps#1, code-earthquakes#2).

No test here touches the network: every request is answered by a stand-in for requests.Session.send, or
(one test, which needs a real socket) by a server on this machine, at 127.0.0.1.
"""
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import pytest
import requests
from urllib3.exceptions import ProtocolError

import cli
from evalgen import pipeline, sources
from evalgen.adapters import AnthropicAdapter, OpenAIAdapter, Task
from evalgen.domains import all_domains
from evalgen.domains import weather as wx
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.domains.neo import NEO, parse_cad
from evalgen.domains.weather import STATIONS, WEATHER
from evalgen.forecast import fetch_fdsn, resolve_forecasts
from evalgen.mix import MixError
from evalgen.models import Event
from evalgen.pipeline import GenerateConfig
from evalgen.seeds import SeedLog
from evalgen.sources import (USGS_FEEDS, build_domain_snapshot, build_snapshot, failure_text, parse_usgs_geojson,
                             record_problems)
from evalgen.util import read_json, sha256_hex, write_json

from conftest import quiet
from test_usgs_forecast import fake_feed

ROOT = Path(__file__).resolve().parents[1]
NEO_FEED = ROOT / "tests" / "fixtures" / "neo" / "cad_2026-11-08.json"
WEATHER_FEED = ROOT / "tests" / "fixtures" / "weather" / "raw_feed.json"
DEEP = b"[" * 100_000 + b"]" * 100_000  # nested past any recursion limit
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy",
                   "no_proxy")


def response(body: bytes, status: int = 200, url: str = "https://example.invalid/") -> requests.Response:
    """A response as requests.Session.send returns it, its body still to be read from `raw`."""
    answer = requests.Response()
    answer.status_code, answer.url, answer.raw = status, url, io.BytesIO(body)
    answer.reason = "OK" if status < 400 else "Server Error"
    return answer


@pytest.fixture
def sent(monkeypatch):
    """Every request is answered here, never sent: `answer(request)` gives the response (or raises), and
    `seen` records each request with the session that sent it. The environment names no proxy."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    state = {"seen": [], "answer": lambda request: response(b"{}")}

    def send(session, request, **kwargs):
        state["seen"].append({"url": request.url, "headers": dict(request.headers), "trust_env": session.trust_env,
                              "proxies": dict(kwargs.get("proxies") or {}), "timeout": kwargs.get("timeout")})
        return state["answer"](request)

    monkeypatch.setattr(requests.Session, "send", send)
    return state


def generate_frozen(tmp_path: Path, mix: str, feed: Path, name: str = "source", seed: int = 3) -> Path:
    """A run off the legacy path that replays the frozen feed `feed`: its snapshot folder then holds the bytes
    with their snapshot.json, and its manifest the hashes."""
    return pipeline.generate(GenerateConfig(seed=seed, instances=1, variants=0, mix=mix, snapshot_file=str(feed)),
                             tmp_path / name, log=quiet)


def flip_a_digit(path: Path) -> None:
    """Change one digit of the first distance of the CAD fixture: the bytes no longer hash as recorded."""
    data = path.read_bytes()
    assert b'"0.00582725465227592"' in data
    path.write_bytes(data.replace(b'"0.00582725465227592"', b'"0.00592725465227592"', 1))


# ------------------------------------------------------------- sweep-network#3: no hash, no "verified"


@pytest.mark.parametrize("change", ["deleted", "null"])
def test_a_sidecar_without_its_hash_is_refused_not_called_verified(tmp_path, capsys, change):
    run = generate_frozen(tmp_path, "neo:classic", NEO_FEED)
    sidecar, raw = run / "snapshot" / "snapshot.json", run / "snapshot" / "raw_cad.json"
    meta = read_json(sidecar)
    if change == "deleted":
        del meta["raw_sha256"]
    else:
        meta["raw_sha256"] = None
    write_json(sidecar, meta)
    flip_a_digit(raw)
    with pytest.raises(ValueError, match="raw_sha256"):
        build_domain_snapshot(NEO, "live", SeedLog(2), snapshot_file=str(raw))
    out = tmp_path / "replay"
    for option in (["--snapshot-file", str(raw)], ["--snapshot-dir", str(run / "snapshot")]):
        assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "neo:classic", *option,
                         "--out", str(out)]) == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "raw_sha256" in err and not out.exists(), err


def test_a_legacy_replay_of_bytes_that_no_hash_vouches_for_is_refused(tmp_path, capsys):
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    run = pipeline.generate(GenerateConfig(seed=21, instances=1, source="usgs", snapshot_file=str(feed)),
                            tmp_path / "legacy", log=quiet)
    frozen, sidecar = run / "snapshot" / "raw_feed.geojson", run / "snapshot" / "snapshot.json"
    again, _ = build_snapshot("usgs", SeedLog(21), snapshot_file=str(frozen))  # the run as frozen: verified
    assert again.note == "replay of frozen feed raw_feed.geojson (SHA-256 verified)"
    meta = read_json(sidecar)
    write_json(sidecar, {k: v for k, v in meta.items() if k != "raw_sha256"})
    with pytest.raises(ValueError, match="raw_sha256"):
        build_snapshot("usgs", SeedLog(21), snapshot_file=str(frozen))
    write_json(sidecar, dict(meta, raw_sha256="0" * 64))  # 0.2.0 replayed these bytes with a WARNING in the note
    out = tmp_path / "out"
    assert cli.main(["generate", "--seed", "21", "--instances", "1", "--source", "usgs", "--snapshot-file",
                     str(frozen), "--out", str(out)]) == 2
    assert "do not match the raw_sha256" in capsys.readouterr().err and not out.exists()


# ------------------------------------------------- sweep-network#4: what no hash covers is checked


@pytest.mark.parametrize("window", [12, 0, -24, 100000])
def test_a_sidecar_window_that_the_bytes_do_not_give_is_refused(tmp_path, window):
    run = generate_frozen(tmp_path, "weather:classic", WEATHER_FEED)
    sidecar = run / "snapshot" / "snapshot.json"
    write_json(sidecar, dict(read_json(sidecar), window_hours=window))
    with pytest.raises(ValueError, match="window"):
        build_domain_snapshot(WEATHER, "live", SeedLog(2), snapshot_file=str(run / "snapshot" / "raw_feed.json"))
    with pytest.raises(MixError, match="window"):
        pipeline.generate(GenerateConfig(seed=3, instances=1, mix="weather:classic",
                                         snapshot_dir=str(run / "snapshot")), tmp_path / "out", log=quiet)
    assert not (tmp_path / "out").exists()


def test_a_legacy_sidecar_window_is_one_a_legacy_run_records(tmp_path):
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    run = pipeline.generate(GenerateConfig(seed=21, instances=1, source="usgs", snapshot_file=str(feed)),
                            tmp_path / "legacy", log=quiet)
    frozen, sidecar = run / "snapshot" / "raw_feed.geojson", run / "snapshot" / "snapshot.json"
    meta = read_json(sidecar)
    for window in (12, 0, -24):
        write_json(sidecar, dict(meta, window_hours=window))
        with pytest.raises(ValueError, match="window"):
            build_snapshot("usgs", SeedLog(21), snapshot_file=str(frozen))
    write_json(sidecar, dict(meta, window_hours=1))  # as a legacy run of the hour feed records it
    assert build_snapshot("usgs", SeedLog(21), snapshot_file=str(frozen))[0].window_hours == 1


def test_a_synthetic_sidecar_beside_frozen_bytes_is_refused(tmp_path):
    run = generate_frozen(tmp_path, "weather:classic", WEATHER_FEED)
    sidecar = run / "snapshot" / "snapshot.json"
    write_json(sidecar, dict(read_json(sidecar), source="synthetic", raw_sha256=None))
    for config in (dict(snapshot_dir=str(run / "snapshot")), dict(snapshot_file=str(run / "snapshot" / "raw_feed.json"))):
        with pytest.raises(MixError, match="synthetic"):
            pipeline.generate(GenerateConfig(seed=3, instances=1, mix="weather:classic", **config), tmp_path / "out",
                              log=quiet)
    assert not (tmp_path / "out").exists()


def test_provenance_that_the_source_run_does_not_record_is_refused(tmp_path):
    run = generate_frozen(tmp_path, "neo:classic", NEO_FEED)
    sidecar = run / "snapshot" / "snapshot.json"
    meta = read_json(sidecar)
    for forged in ({"url": "https://forged.example/cad.api"}, {"fetched_at": "1999-01-01T00:00:00Z"}):
        write_json(sidecar, dict(meta, **forged))
        for config in (dict(snapshot_dir=str(run / "snapshot")), dict(snapshot_file=str(run / "snapshot" / "raw_cad.json"))):
            with pytest.raises(MixError, match="manifest"):
                pipeline.generate(GenerateConfig(seed=3, instances=1, mix="neo:classic", **config), tmp_path / "out",
                                  log=quiet)
    write_json(sidecar, dict(meta, url="not a url"))  # where no source run vouches for it, its form is checked
    with pytest.raises(ValueError, match="url"):
        build_domain_snapshot(NEO, "live", SeedLog(2), snapshot_file=str(run / "snapshot" / "raw_cad.json"))
    assert not (tmp_path / "out").exists()


# --------------------------------------------- sweep-network#5: the source run's manifest is compared


def test_a_replay_compares_the_hashes_of_the_source_runs_manifest(tmp_path):
    run = generate_frozen(tmp_path, "neo:classic", NEO_FEED)
    manifest_path, sidecar = run / "manifest.json", run / "snapshot" / "snapshot.json"
    manifest = read_json(manifest_path)
    replay = pipeline.generate(GenerateConfig(seed=3, instances=1, variants=0, mix="neo:classic",
                                              snapshot_dir=str(run / "snapshot")), tmp_path / "good", log=quiet)
    assert read_json(replay / "manifest.json")["snapshot"]["note"] == (
        "replay of frozen feed raw_cad.json (SHA-256 verified)")
    write_json(manifest_path, dict(manifest, snapshot=dict(manifest["snapshot"], raw_sha256="0" * 64)))
    with pytest.raises(MixError, match="manifest"):
        pipeline.generate(GenerateConfig(seed=3, instances=1, mix="neo:classic", snapshot_dir=str(run / "snapshot")),
                          tmp_path / "out", log=quiet)
    write_json(manifest_path, dict(manifest, snapshot=dict(manifest["snapshot"], events_sha256="0" * 64)))
    with pytest.raises(MixError, match="events_sha256"):
        pipeline.generate(GenerateConfig(seed=3, instances=1, mix="neo:classic", snapshot_dir=str(run / "snapshot")),
                          tmp_path / "out", log=quiet)
    write_json(manifest_path, manifest)
    raw = run / "snapshot" / "raw_cad.json"
    flip_a_digit(raw)  # the bytes and their snapshot.json changed together: the manifest still says otherwise
    write_json(sidecar, dict(read_json(sidecar), raw_sha256=sha256_hex(raw.read_bytes())))
    with pytest.raises(MixError, match="manifest"):
        pipeline.generate(GenerateConfig(seed=3, instances=1, mix="neo:classic", snapshot_file=str(raw)),
                          tmp_path / "out", log=quiet)
    assert not (tmp_path / "out").exists()


# ----------------------------------------- sweep-network#6, #7: bytes that cannot be real never crash a run


def usgs_with(**props) -> bytes:
    """The fake USGS feed with one more earthquake, whose properties `props` override."""
    doc = json.loads(fake_feed()[0])
    extra = {"type": "Feature", "id": "ci99999999",
             "properties": {"mag": 2.5, "place": "5 km N of Town, CA", "time": 1_793_735_000_000, "magType": "ml",
                            "type": "earthquake", "net": "ci", "status": "reviewed"},
             "geometry": {"type": "Point", "coordinates": [-117.0, 34.0, 8.0]}}
    extra["properties"].update(props)
    doc["features"].append(extra)
    return json.dumps(doc).encode()  # json writes Infinity and NaN as JavaScript does, as a feed might


def test_deeply_nested_bytes_fall_back_on_every_source():
    for domain in [d for d in all_domains() if d.live is not None]:
        snapshot, raw = build_domain_snapshot(domain, "live", SeedLog(1), fetch=lambda: (DEEP, "http://x"))
        assert (snapshot.source, raw) == ("synthetic", None) and "fell back to synthetic data" in snapshot.note
        with pytest.raises(ValueError):
            domain.live.parse(DEEP)
    snapshot, raw = build_snapshot("usgs", SeedLog(1), fetch=lambda feed: (DEEP, "http://x"))  # the legacy path
    assert (snapshot.source, raw) == ("synthetic", None)
    bundle = json.loads(WEATHER_FEED.read_bytes())
    bundle["stations"]["KJFK"]["body"] = DEEP.decode()  # one station's body: that station is left out
    records, _, _ = WEATHER.live.parse(json.dumps(bundle).encode())
    assert sorted(r.source for r in records) == ["KLAX", "KORD", "KSEA"]


def test_a_time_or_magnitude_that_cannot_be_real_is_dropped():
    usable = {e.id for e in parse_usgs_geojson(fake_feed()[0])[0]}
    for props in ({"time": math.inf}, {"time": 1e20}, {"mag": math.inf}, {"mag": math.nan}, {"mag": 25.0}):
        raw = usgs_with(**props)
        events, _, _ = parse_usgs_geojson(raw)
        assert {e.id for e in events} == usable, props  # the absurd earthquake alone is left out
        snapshot, _ = build_snapshot("usgs", SeedLog(1), fetch=lambda feed, raw=raw: (raw, USGS_FEEDS["day"]))
        assert snapshot.source == "usgs" and {e.id for e in snapshot.events} == usable, props
        live, _ = build_domain_snapshot(EARTHQUAKES, "live", SeedLog(1), fetch=lambda raw=raw: (raw, "http://x"))
        assert live.source == "usgs" and {e.id for e in live.events} == usable, props
        assert all(math.isfinite(e.mag) for e in live.events)
    for value in (math.inf, math.nan):
        assert record_problems(EARTHQUAKES, [Event("ci12345678", 1_793_700_000, 34.0, -117.0, 8.0, value, "ml",
                                                   "California", "ci")])


def test_a_close_approach_dated_beyond_any_catalogue_is_dropped():
    doc = json.loads(NEO_FEED.read_bytes())
    doc["data"][0][doc["fields"].index("jd")] = "1e12"
    records, compiled_at, window = parse_cad(json.dumps(doc).encode())
    assert len(records) == 39 and (compiled_at, window) == NEO.live.parse(NEO_FEED.read_bytes())[1:]


def test_a_live_record_that_cannot_be_real_never_ends_a_run(tmp_path, sent):
    raw = usgs_with(mag=math.nan)
    sent["answer"] = lambda request: response(raw)
    run = pipeline.generate(GenerateConfig(seed=5, instances=1, variants=0, mix="earthquakes:news", source="live"),
                            tmp_path / "live", log=quiet)
    meta = read_json(run / "snapshot" / "snapshot.json")
    assert meta["source"] == "usgs" and "NaN" not in (run / "snapshot" / "events.json").read_text()


# ------------------------------------------- sweep-network#8, #9: the session, the deadline, the budget


def test_every_live_request_goes_through_a_session_that_ignores_the_environment(tmp_path, sent, monkeypatch):
    netrc = tmp_path / "netrc"
    netrc.write_text("default login alice password s3cret\n", encoding="utf-8")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("CURL_CA_BUNDLE", str(tmp_path / "no-such-bundle.pem"))
    feed, _ = fake_feed()
    sent["answer"] = lambda request: response(NEO_FEED.read_bytes() if "cad.api" in request.url else feed)
    sources.fetch_usgs("day")
    NEO.live.fetch()
    sent["answer"] = lambda request: response(json.dumps({"type": "FeatureCollection", "features": []}).encode())
    replace(WEATHER.live, stations=("KJFK",)).fetch()
    fetch_fdsn({"start": 1_793_736_000, "end": 1_793_822_400, "min_mag": 4.5, "lat": 34.0, "lon": -117.0,
                "radius_km": 250})
    assert len(sent["seen"]) == 4
    for request in sent["seen"]:
        assert request["trust_env"] is False and "Authorization" not in request["headers"], request
        assert request["headers"]["User-Agent"].startswith("evalgen/") and request["proxies"] == {}


def test_an_api_model_posts_through_a_session_that_ignores_the_environment(tmp_path, sent, monkeypatch):
    """Review (sweep-network#9, for the API models): the adapters posted through module-level requests, whose
    session trusts the environment, so a netrc 'default' entry went to the API hosts as HTTP Basic credentials:
    beside Anthropic's key, and in place of OpenAI's Bearer key, so that every OpenAI call carried the wrong ones."""
    netrc = tmp_path / "netrc"
    netrc.write_text("default login alice password s3cret\n", encoding="utf-8")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sent["answer"] = lambda request: response(json.dumps(
        {"content": [{"type": "text", "text": "{}"}]} if "anthropic" in request.url
        else {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode())
    task = Task("inst-000", "v0", [])
    AnthropicAdapter("claude-test").complete("PROMPT", task)
    OpenAIAdapter("gpt-test").complete("PROMPT", task)
    anthropic, openai = sent["seen"]
    assert anthropic["url"].startswith("https://api.anthropic.com/") and openai["url"].startswith("https://api.openai.com/")
    for request in (anthropic, openai):
        assert request["trust_env"] is False and request["proxies"] == {}, request
    assert "Authorization" not in anthropic["headers"] and anthropic["headers"]["x-api-key"] == "sk-ant-test"
    assert openai["headers"]["Authorization"] == "Bearer sk-test"
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")  # a proxy the environment names is still used
    OpenAIAdapter("gpt-test").complete("PROMPT", task)
    assert sent["seen"][-1]["proxies"] == {"https": "http://proxy.example:3128"}


def test_a_proxy_is_used_only_where_the_environment_names_one(monkeypatch):
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert sources.proxies_for("https://earthquake.usgs.gov/x") == {}
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    assert sources.proxies_for("https://earthquake.usgs.gov/x") == {"https": "http://proxy.example:3128"}
    monkeypatch.setenv("NO_PROXY", "usgs.gov")
    assert sources.proxies_for("https://earthquake.usgs.gov/x") == {}


def slow_body(n: int, pause: float) -> io.RawIOBase:
    """A body that arrives a byte at a time, `pause` seconds apart (a server trickling its answer)."""
    class Trickle(io.RawIOBase):
        left = n

        def readable(self) -> bool:
            return True

        def readinto(self, buffer) -> int:
            if self.left <= 0:
                return 0
            time.sleep(pause)
            buffer[0:1] = b"x"
            self.left -= 1
            return 1

    return Trickle()


def test_a_fetch_is_bounded_as_a_whole_not_read_by_read(sent, monkeypatch):
    monkeypatch.setattr(sources, "FETCH_DEADLINE_S", 1.0)

    def trickling(request):
        answer = response(b"")
        answer.raw = slow_body(40, 0.25)  # ten seconds in all, each read well within the read timeout
        return answer

    for answer in (trickling, lambda request: time.sleep(4) or response(b"{}")):  # a trickle, then a stall
        sent["answer"] = answer
        started = time.monotonic()
        with pytest.raises(requests.Timeout):
            sources.fetch_usgs("day")
        assert time.monotonic() - started < 3.5


def test_a_trickling_server_is_given_up_at_the_deadline_and_its_connection_shut(monkeypatch):
    """With a real socket (a server on this machine, never the network): urllib3 reads a body of known length
    until all of it has come, so only the deadline as a whole stops a server that sends a byte at a time;
    and the connection is shut, so the request's thread stops too instead of reading on."""
    import http.server
    import threading

    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    class Trickling(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                self.send_response(200)
                self.send_header("Content-Length", "60")
                self.end_headers()
                for _ in range(60):  # fifteen seconds in all
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.25)
            except OSError:
                pass

        def log_message(self, *args):
            pass

        def handle_error(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Trickling)
    server.daemon_threads = True
    server.handle_error = lambda *args: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        started = time.monotonic()
        with pytest.raises(sources.FetchTimeout):
            sources.http_get(f"http://127.0.0.1:{server.server_address[1]}/feed", timeout=5.0, deadline=1.0)
        assert time.monotonic() - started < 4.0
        for _ in range(40):
            if not any(t.name == "evalgen-live-request" and t.is_alive() for t in threading.enumerate()):
                break
            time.sleep(0.1)
        assert not any(t.name == "evalgen-live-request" and t.is_alive() for t in threading.enumerate())
    finally:
        server.shutdown()
        server.server_close()


def test_the_nws_fetch_stops_asking_once_its_budget_is_spent(monkeypatch, sent):
    clock = {"now": 0.0}
    monkeypatch.setattr(wx, "_clock", lambda: clock["now"], raising=False)

    def stalled(request):
        clock["now"] += 20.0  # every station waits out its read timeout
        raise requests.ReadTimeout("read timed out")

    sent["answer"] = stalled
    snapshot, raw = build_domain_snapshot(WEATHER, "live", SeedLog(2))
    assert (snapshot.source, raw) == ("synthetic", None) and "no station answered" in snapshot.note
    assert len(sent["seen"]) == WEATHER.live.budget // 20 < len(STATIONS)


# ------------------------------------------------- sweep-network#10: resolve and a reply it cannot read


def test_resolve_keeps_a_forecast_pending_when_the_reply_cannot_be_read(sent):
    rule = {"start": 1_793_736_000, "end": 1_793_822_400, "min_mag": 4.5, "lat": 34.0, "lon": -117.0,
            "radius_km": 250}
    for body in (b"[]", DEEP, b'{"features": 5}', b'{"features": [1]}', b'{"features": [{"properties": 3}]}'):
        sent["answer"] = lambda request, body=body: response(body)
        forecasts = [{"qid": "F1", "rule": dict(rule), "status": "pending", "instance_id": "inst-000"}]
        summary = resolve_forecasts(forecasts, now=rule["end"] + 3 * 86400)
        assert (summary["pending"], summary["newly_resolved"], forecasts[0]["status"]) == (1, 0, "pending"), body
        assert len(summary["errors"]) == 1 and summary["errors"][0].startswith("inst-000: "), summary
    sent["answer"] = lambda request: response(json.dumps({"type": "FeatureCollection", "features": [
        {"id": "us7000zzzz", "properties": {"mag": 4.8}}, {"id": "us7000zzzy", "properties": {"mag": "x"}}]}).encode())
    forecasts = [{"qid": "F1", "rule": dict(rule), "status": "pending", "instance_id": "inst-000"}]
    assert resolve_forecasts(forecasts, now=rule["end"] + 3 * 86400)["newly_resolved"] == 1
    assert (forecasts[0]["outcome"], forecasts[0]["evidence"]) == (1, ["us7000zzzz"])


# ------------------------------------------- sweep-network#11, sweep-determinism#1: no local path recorded


def test_a_replay_off_the_legacy_path_records_no_local_path(tmp_path):
    run = generate_frozen(tmp_path, "neo:classic", NEO_FEED)
    for key, given, recorded in (("snapshot_dir", run / "snapshot", "source/snapshot"),
                                 ("snapshot_file", run / "snapshot" / "raw_cad.json", "source/snapshot/raw_cad.json")):
        replay = pipeline.generate(GenerateConfig(seed=3, instances=1, variants=0, mix="neo:classic",
                                                  **{key: str(given)}), tmp_path / f"replay-{key}", log=quiet)
        manifest = read_json(replay / "manifest.json")
        assert manifest["replay"] == {"source_run": "source", "same_code": True, "same_seed": True}
        assert manifest["config"][key] == recorded
        text = (replay / "manifest.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text
    bare = read_json(run / "manifest.json")  # a frozen file of no run: by its name alone
    assert bare["config"]["snapshot_file"] == "cad_2026-11-08.json" and bare["replay"]["source_run"] is None


# --------------------------------------------- sweep-network#12, #13: proxy notes, a dropped first station


def test_a_failure_note_names_no_proxy(monkeypatch):
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://alice:s3cret@proxy.corp.example:3128")
    error = requests.exceptions.ProxyError(
        "HTTPSConnectionPool(host='earthquake.usgs.gov', port=443): Max retries exceeded with url: "
        "/earthquakes/feed/v1.0/summary/all_day.geojson (Caused by ProxyError('Unable to connect to proxy', "
        "NewConnectionError(\"HTTPSConnection(host='proxy.corp.example', port=3128): Failed to establish a new "
        "connection: [Errno 61] Connection refused\")))")
    note = failure_text(error)
    assert note.startswith("ProxyError: HTTPSConnectionPool(host='earthquake.usgs.gov', port=443)")
    assert "Unable to connect to proxy" in note and "<proxy>" in note
    assert not any(part in note for part in ("proxy.corp.example", "3128", "alice", "s3cret")), note


def test_a_first_station_that_drops_its_connection_is_one_station_down(sent):
    feature = json.loads(json.loads(WEATHER_FEED.read_bytes())["stations"]["KJFK"]["body"])
    body = json.dumps({"type": "FeatureCollection", "features": [feature]}).encode()

    def answer(request):
        if "/KJFK/" in request.url:
            raise requests.ConnectionError(ProtocolError("Connection aborted.", ConnectionResetError(54, "reset")))
        return response(body)

    sent["answer"] = answer
    adapter = replace(WEATHER.live, stations=("KJFK", "KORD"))
    bundle = json.loads(adapter.fetch()[0])
    assert len(sent["seen"]) == 2 and bundle["stations"]["KJFK"]["error"].startswith("ConnectionError: ")
    assert bundle["stations"]["KORD"]["status"] == 200


# ------------------------------------------- code-weather#2, code-neo#2: one bad station, a null designation


def test_one_malformed_station_leaves_the_others_read():
    bundle = json.loads(WEATHER_FEED.read_bytes())
    for body in ({"type": "FeatureCollection", "features": 5},
                 {"type": "FeatureCollection", "features": [dict(json.loads(bundle["stations"]["KJFK"]["body"]),
                                                                 properties=None)]}):
        bundle["stations"]["KJFK"]["body"] = json.dumps(body)
        records, _, _ = WEATHER.live.parse(json.dumps(bundle).encode())
        assert sorted(r.source for r in records) == ["KLAX", "KORD", "KSEA"]
    feature = json.loads(json.loads(WEATHER_FEED.read_bytes())["stations"]["KJFK"]["body"])
    feature["properties"]["temperature"]["qualityControl"] = ["V"]
    bundle["stations"]["KJFK"]["body"] = json.dumps(feature)
    records, _, _ = WEATHER.live.parse(json.dumps(bundle).encode())
    assert sorted(r.source for r in records) == ["KLAX", "KORD", "KSEA"]


def test_a_cad_row_without_a_designation_or_a_body_is_left_out():
    doc = json.loads(NEO_FEED.read_bytes())
    fields = doc["fields"]
    row = dict(zip(fields, doc["data"][0]))
    rows = [{**row, "des": None}, {**row, "des": "2024 AB9", "body": None}, {**row, "des": 2024},
            {**row, "des": "2024 AB8"}]
    records, _, _ = parse_cad(json.dumps({**doc, "data": [[r[k] for k in fields] for r in rows]}).encode())
    assert [(r.id, r.group) for r in records] == [("2024ab8-ca1", "Earth")]


# -------------------------------------- code-pipeline#2, code-traps#1, code-earthquakes#2 and #4: frozen inputs


def test_a_legacy_runs_synthetic_snapshot_is_not_redrawn_off_the_legacy_path(tmp_path):
    legacy = pipeline.generate(GenerateConfig(seed=9, instances=1), tmp_path / "legacy", log=quiet)
    with pytest.raises(MixError, match="legacy run"):
        pipeline.generate(GenerateConfig(seed=9, instances=1, mix="earthquakes:classic",
                                         snapshot_dir=str(legacy / "snapshot")), tmp_path / "out", log=quiet)
    assert not (tmp_path / "out").exists()


def feed_named(tmp_path: Path, period: str, days: int = 1) -> Path:
    """The fake USGS feed as the summary feed of `period` names itself, its earthquakes spread over `days`."""
    doc = json.loads(fake_feed()[0])
    doc["metadata"]["url"] = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_{period}.geojson"
    for i, feature in enumerate(doc["features"]):
        feature["properties"]["time"] -= (i % days) * 86_400_000
    path = tmp_path / f"all_{period}.geojson"
    path.write_bytes(json.dumps(doc).encode())
    return path


def test_a_frozen_feed_is_read_over_the_window_its_name_gives(tmp_path):
    for period, hours in (("hour", 1), ("day", 24), ("week", 168), ("month", 720)):
        events, compiled_at, window = EARTHQUAKES.live.parse(feed_named(tmp_path, period, 7).read_bytes())
        assert window == hours and events and all(compiled_at - hours * 3600 <= e.time <= compiled_at for e in events)
    with pytest.raises(MixError, match="hour"):  # too short for the traps off the legacy path, as --feed hour is
        pipeline.generate(GenerateConfig(seed=11, instances=1, mix="earthquakes:classic",
                                         snapshot_file=str(feed_named(tmp_path, "hour"))), tmp_path / "out", log=quiet)
    assert not (tmp_path / "out").exists()
    week = pipeline.generate(GenerateConfig(seed=7, instances=1, variants=0, mix="earthquakes:classic",
                                            snapshot_file=str(feed_named(tmp_path, "week", 7))), tmp_path / "week",
                             log=quiet)
    manifest = read_json(week / "manifest.json")
    assert manifest["snapshot"]["window_hours"] == 168
    assert manifest["self_check"]["oracle_agreements"] == manifest["self_check"]["oracle_checks"]
    assert "covers the 168 hours before compilation" in (week / "prompts" / "inst-000" / "v0.txt").read_text()


def test_the_shortest_window_leaves_room_for_a_phantom_and_its_notice():
    """A phantom lies a minute or more inside the window and its notice comes NOTICE_DELAY after it, before the
    digest is compiled: the shortest window off the legacy path must hold both, as the hour feed's does not."""
    from evalgen.traps import NOTICE_DELAY

    assert sources.MIN_WINDOW_HOURS * 3600 >= 60 + NOTICE_DELAY[0] + 60 > 1 * 3600


def test_a_replay_says_when_its_forecasts_ask_about_a_window_already_closed(tmp_path):
    logs: list[str] = []
    pipeline.generate(GenerateConfig(seed=31, instances=1, variants=0, mix="earthquakes:news",
                                     snapshot_file=str(feed_named(tmp_path, "day"))), tmp_path / "old", log=logs.append)
    closed = [line for line in logs if "already" in line and "forecast" in line]
    assert len(closed) == 1 and closed[0].startswith("[generate] warning: ") and "2026-11-04" in closed[0]
