"""The command line and the generator under misuse: counts and model specs are checked before anything
is written or asked, misuse ends in `error: ...` with exit status 2 (never a traceback), live sources
fall back on any bytes they cannot read, an offline run regenerates byte for byte, and what the
evaluator publishes (the commitments) and keeps secret (the seed, the nonce) is what the docs say."""
from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

import cli
from evalgen import pipeline, sources
from evalgen.adapters import UnknownModel
from evalgen.domains import all_domains
from evalgen.domains import weather as wx
from evalgen.mix import EXAMPLE_NONCE, NONCE_ADVICE, config_commitment, is_example_nonce, nonce_problem
from evalgen.seeds import SeedLog, guessable_seed
from evalgen.sources import (build_domain_snapshot, build_snapshot, generate_synthetic, parse_usgs_geojson,
                             synthetic_compiled_at)
from evalgen.util import read_json

from conftest import quiet
from test_usgs_forecast import fake_feed

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "private.example.json"
WEATHER_FEED = ROOT / "tests" / "fixtures" / "weather" / "raw_feed.json"
# JSON that no live source serves: before the fix, the earthquake source crashed on most of these
# (AttributeError, TypeError) instead of falling back to synthetic data as the NWS and JPL sources do.
ODD_BODIES = [b"null", b"[]", b"{}", b'"text"', b'{"features": null}', b'{"features": 5}', b'{"features": [1]}',
              b'{"features": [{"properties": "x"}]}', b'{"metadata": "x", "features": []}',
              b'{"metadata": {"generated": [1]}, "features": []}',
              b'{"features": [{"properties": {"mag": [1], "time": 1}, "geometry": {"coordinates": [1, 2, 3]}, '
              b'"id": "us12345678"}]}', b"\xff\xfe"]


def no_traceback(capsys) -> str:
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err, err
    return err


# ------------------------------------------------------------------------------------ counts


@pytest.mark.parametrize("argv", [["--instances", "0"], ["--instances", "-1"], ["--variants", "-1"],
                                  ["--instances", "many"]])
def test_counts_are_checked_before_anything_is_written(tmp_path, argv, capsys):
    """Before the fix, --instances -1 wrote an empty run and exited 0, and --variants -1 wrote the
    snapshots and instances, then died with a traceback."""
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as stopped:
        cli.main(["generate", "--seed", "5", *argv, "--out", str(out)])
    assert stopped.value.code == 2 and "is not a whole number" in capsys.readouterr().err
    assert not out.exists()


def test_the_generator_refuses_a_config_that_no_run_can_have(tmp_path):
    for bad in (dict(instances=0), dict(variants=-1), dict(instances=2.5), dict(events_min=9, events_max=4),
                dict(questions_min=0), dict(pool_size=0), dict(audit_rate=1.5), dict(instances=True)):
        with pytest.raises(ValueError, match="cannot generate this run"):
            pipeline.generate(pipeline.GenerateConfig(seed=5, **bad), tmp_path / "run", log=quiet)
        assert not (tmp_path / "run").exists(), bad
    run = pipeline.generate(pipeline.GenerateConfig(seed=5, instances=1, variants=0, mix="random"), tmp_path / "one",
                            log=quiet)  # no variant at all is a run: the original rendering alone
    assert [len(inst["renderings"]) for inst in map(read_json, sorted((run / "instances").glob("*.json")))] == [1]


def test_path_options_may_be_given_as_paths(tmp_path):
    """Before the fix, a Path in snapshot_file or snapshot_dir went through the whole generation and then
    crashed at the manifest write, leaving a run without a manifest."""
    first = pipeline.generate(pipeline.GenerateConfig(seed=7, instances=1, mix="weather", source="live",
                                                      snapshot_file=WEATHER_FEED), tmp_path / "file", log=quiet)
    replay = pipeline.generate(pipeline.GenerateConfig(seed=7, instances=1, mix="weather", source="live",
                                                       snapshot_dir=first / "snapshot"), tmp_path / "dir", log=quiet)
    # Recorded by the run's folder ("file/snapshot"), or by its own name: no place on the evaluator's machine.
    for run, key, recorded in ((first, "snapshot_file", WEATHER_FEED.name), (replay, "snapshot_dir", "file/snapshot")):
        assert read_json(run / "manifest.json")["config"][key] == recorded
    assert (first / "prompts").is_dir() and (replay / "instances" / "inst-000.json").read_bytes() == (
        first / "instances" / "inst-000.json").read_bytes()


# ------------------------------------------------------------------------------------- models


def test_run_refuses_a_spec_that_names_no_model_before_asking_any(tmp_path, capsys, monkeypatch):
    """Before the fix, `run` skipped every spec it could not build (a typo, a penalty out of range, an
    empty list) and exited 0, so score and report quietly left the model out."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), tmp_path / "run", log=quiet)
    for specs in (["orcale"], ["oracle", "orcale"], ["mock:foo=1"], ["mock:held_out=-0.1"], ["mock:held_out=nan"],
                  ["mock:held_out=abc"], ["oracle:x"], ["naive:anything"], ["oracle", "naive:x"],
                  ["mock:held_out=0.2+held_out=0.3"], ["mock:held_out_genres=0.5+held_out=0.3"]):
        with pytest.raises(UnknownModel):
            pipeline.run_models(run, specs, log=quiet)
        assert cli.main(["run", str(run), "--models", ",".join(specs)]) == 2
        no_traceback(capsys)
    assert not (run / "responses").exists()  # refused before any model was asked, the oracle included
    for models in ("", ",", "anthropic"):  # nothing to ask: no spec, or only a model without its key
        with pytest.raises(ValueError, match="no model to ask"):
            pipeline.run_models(run, [s for s in models.split(",") if s], log=quiet)
        assert cli.main(["run", str(run), "--models", models]) == 2
        assert "no model to ask" in no_traceback(capsys)
    assert pipeline.run_models(run, ["oracle", "anthropic"], log=quiet) == ["oracle"]  # a missing key: skipped


# --------------------------------------------------------------------------------- misuse


def test_misuse_ends_in_an_error_not_a_traceback(tmp_path, capsys):
    """Before the fix, each of these ended in a traceback with exit status 1, which verify-config uses
    for MISMATCH."""
    run = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), tmp_path / "run", log=quiet)
    pipeline.run_models(run, ["oracle"], log=quiet)
    config = tmp_path / "private.json"
    config.write_text(json.dumps({"nonce": secrets.token_hex(16), "domains": {"public": ["earthquakes"], "held_out": []},
                                  "genres": {"public": ["classic"], "held_out": []}}), encoding="utf-8")
    committed = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1, mix="random",
                                                          private_config=str(config)), tmp_path / "committed",
                                  log=quiet)
    (tmp_path / "afile.txt").write_text("x", encoding="utf-8")
    listed = tmp_path / "listed"
    listed.mkdir()
    (listed / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    cases = [["generate", "--seed", "3", "--instances", "1", "--out", str(tmp_path / "afile.txt")],
             ["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config", "",
              "--out", str(tmp_path / "x")],
             ["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config", str(tmp_path),
              "--out", str(tmp_path / "x")],
             ["verify-config", str(config), str(committed)],  # the arguments the wrong way round
             ["verify-config", str(committed), str(tmp_path)],
             ["report", str(listed)], ["score", str(listed)], ["run", str(tmp_path / "afile.txt")]]
    for argv in cases:
        assert cli.main(argv) == 2, argv
        no_traceback(capsys)
    assert cli.main(["verify-config", str(committed), str(config)]) == 0
    # A reply broken off mid-write is a prompt without a reply, which `run` asks again: no traceback in score.
    (run / "responses" / "oracle" / "inst-000__v0.json").write_text('{"mod', encoding="utf-8")
    assert cli.main(["score", str(run)]) == 0
    with pytest.raises(SystemExit) as stopped:
        cli.main(["resolve", str(run), "--now", "garbage"])
    assert stopped.value.code == 2 and "is not an ISO timestamp" in capsys.readouterr().err


# ------------------------------------------------------------------------------ live sources


def test_a_received_run_of_the_wrong_shape_ends_in_an_error(tmp_path, capsys):
    """A run received from someone else may hold files that evalgen did not write as they are. Before the
    fix, a manifest without a seed, with plugins or an inventory of the wrong shape, a scores file or a
    forecasts.json of the wrong shape, or an instance whose answer key is a list, ended in a traceback."""
    base = tmp_path / "base"
    pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), base, log=quiet)
    pipeline.run_models(base, ["oracle"], log=quiet)
    pipeline.score_run(base, log=quiet)

    def edited(name: str, relative: str, change) -> Path:
        run = tmp_path / name
        shutil.copytree(base, run)
        path = run / relative
        data = read_json(path) if path.exists() else None
        path.write_text(json.dumps(change(data)), encoding="utf-8")
        return run

    without = lambda key: lambda d: {k: v for k, v in d.items() if k != key}  # noqa: E731
    with_ = lambda key, value: lambda d: dict(d, **{key: value})  # noqa: E731
    cases = {
        "no-seed": ("manifest.json", without("seed"), "seed"),
        "seed-list": ("manifest.json", with_("seed", [1]), "seed"),
        "plugins-str": ("manifest.json", with_("plugins", "x.py"), "plugins"),
        "plugins-strs": ("manifest.json", with_("plugins", ["x.py"]), "plugins"),
        "inventory-list": ("manifest.json", with_("inventory", [1]), "inventory"),
        "inventory-empty": ("manifest.json", with_("inventory", {}), "inventory"),
        "key-list": ("instances/inst-000.json", with_("answer_key", []), "answer key"),
        "scores-list": ("scores/oracle.json", lambda d: [1], "scores"),
        "scores-no-rows": ("scores/oracle.json", without("rows"), "scores"),
        "forecasts-object": ("forecasts.json", lambda d: {}, "forecasts"),
        "forecasts-ints": ("forecasts.json", lambda d: [1, 2], "forecasts"),
    }
    for name, (relative, change, said) in cases.items():
        run = edited(name, relative, change)
        commands = [["report", str(run)]]  # each step that reads the file
        if not relative.startswith("scores"):  # resolve reads no scores of a run without forecasts
            commands.append(["resolve", str(run)])
        if relative == "manifest.json" or "instances" in relative:
            commands += [["run", str(run), "--models", "oracle", "--force"], ["score", str(run)]]
        for argv in commands:
            assert cli.main(argv) == 2, (name, argv)
            err = no_traceback(capsys)
            assert said in err, (name, argv, err)


def test_frozen_inputs_of_the_wrong_shape_are_refused_before_anything_is_written(tmp_path, capsys):
    """A frozen snapshot's snapshot.json that is not a JSON object, names no source, or gives its window as
    text, is refused before the run folder is made (before the fix: a traceback, some after the folder was
    made); a source run's manifest that is not an object makes the replay one of an unknown run."""
    source = tmp_path / "source"
    assert cli.main(["generate", "--seed", "21", "--instances", "1", "--mix", "weather", "--source", "live",
                     "--snapshot-file", str(WEATHER_FEED), "--out", str(source)]) == 0
    capsys.readouterr()
    sidecar = source / "snapshot" / "snapshot.json"
    good = sidecar.read_text(encoding="utf-8")
    meta = json.loads(good)
    for body in ("[1]", json.dumps({"domain": "weather"}), json.dumps(dict(meta, window_hours="x")), "{nope"):
        sidecar.write_text(body, encoding="utf-8")
        options = [["--snapshot-dir", str(source / "snapshot")]]  # a replay of frozen snapshots reads the source
        if "source" in body or body in ("[1]", "{nope"):  # a bare feed's replay reads its window, origin and hash
            options.append(["--snapshot-file", str(source / "snapshot" / "raw_feed.json")])
        for option in options:
            out = tmp_path / "out"
            assert cli.main(["generate", "--seed", "21", "--instances", "1", "--mix", "weather", "--source", "live",
                             *option, "--out", str(out)]) == 2, (body, option)
            assert "snapshot.json" in no_traceback(capsys) and not out.exists()
    sidecar.write_text(good, encoding="utf-8")
    for body in ("[1]", "{x"):
        (source / "manifest.json").write_text(body, encoding="utf-8")
        out = tmp_path / f"replay-{len(body)}"
        assert cli.main(["generate", "--seed", "21", "--instances", "1", "--mix", "weather", "--source", "live",
                         "--snapshot-dir", str(source / "snapshot"), "--out", str(out)]) == 0
        assert read_json(out / "manifest.json")["replay"]["source_run"] is None
        capsys.readouterr()


def test_a_plugin_whose_own_code_fails_ends_in_an_error(tmp_path, capsys):
    """Before the fix, a plugin with a syntax error, a missing import or an exception at import ended in a
    traceback; now the error names the plugin and what failed."""
    for name, body in (("syntax", "def (:\n"), ("raises", "raise RuntimeError('boom')\n"),
                       ("imports", "import no_such_module_xyz\n")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "plug.py").write_text(body, encoding="utf-8")
        (folder / "private.json").write_text(json.dumps({
            "nonce": secrets.token_hex(16), "domains": {"public": ["earthquakes"], "held_out": []},
            "genres": {"public": ["classic"], "held_out": []}, "plugins": ["plug.py"]}), encoding="utf-8")
        assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config",
                         str(folder / "private.json"), "--out", str(folder / "run")]) == 2
        err = no_traceback(capsys)
        assert "plug.py failed to run" in err and not (folder / "run").exists(), err


def test_every_live_source_falls_back_on_bytes_that_it_cannot_read():
    for domain in [d for d in all_domains() if d.live is not None]:
        for body in ODD_BODIES:
            snapshot, raw = build_domain_snapshot(domain, "live", SeedLog(1), fetch=lambda body=body: (body, "http://x"))
            assert snapshot.source == "synthetic" and raw is None and "fell back to synthetic data" in snapshot.note
    for body in ODD_BODIES:  # the legacy path reads the same feed with the same parser
        if body != b"{}":  # an object without features is an empty feed, which is too small to use
            with pytest.raises(ValueError):
                parse_usgs_geojson(body)
        snapshot, raw = build_snapshot("usgs", SeedLog(1), fetch=lambda feed, body=body: (body, "http://x"))
        assert snapshot.source == "synthetic" and raw is None
    raw, generated = fake_feed()  # a real feed's shape still parses as before
    assert parse_usgs_geojson(raw)[1] == generated


def test_an_offline_live_run_regenerates_byte_for_byte(tmp_path, monkeypatch):
    """When NWS cannot be reached, the note of the synthetic fallback quoted the request URL with its
    start time, so two identical offline runs differed in snapshot.json, the manifest and the report."""
    def unreachable(url, params=None, **kwargs):
        raise requests.ConnectionError(
            f"HTTPSConnectionPool(host='api.weather.gov', port=443): Max retries exceeded with url: "
            f"{url[len(wx.API_ROOT):]}?start={params['start'].replace(':', '%3A')}&limit={params['limit']} (Caused by "
            f"NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id(params):x}>: Failed to "
            "establish a new connection: [Errno 61] Connection refused'))")

    monkeypatch.setattr(sources, "http_get", unreachable)
    runs = []
    for minute in (1, 2):
        monkeypatch.setattr(wx, "_utc_now", lambda minute=minute: datetime(2026, 11, 9, 3, minute, tzinfo=timezone.utc))
        runs.append(pipeline.generate(pipeline.GenerateConfig(seed=7, instances=2, mix="weather", source="live"),
                                      tmp_path / f"run{minute}", log=quiet))
    note = read_json(runs[0] / "snapshot" / "snapshot.json")["note"]
    assert note.startswith("nws unavailable (ConnectionError: ") and "/stations/" in note
    assert "start=" not in note and "0x" not in note and "fell back to synthetic data" in note
    for name in sorted(p.relative_to(runs[0]).as_posix() for p in runs[0].rglob("*") if p.is_file()):
        if name != "manifest.json":
            assert (runs[0] / name).read_bytes() == (runs[1] / name).read_bytes(), name
    manifests = [read_json(run / "manifest.json") for run in runs]
    assert manifests[0]["snapshot"] == manifests[1]["snapshot"]


# ------------------------------------------------------------------- commitments and secrets


def test_generate_prints_the_whole_commitments_and_where_they_are(tmp_path):
    """Before the fix, the console showed 16 hexadecimal digits of each commitment as "publishable": a
    prefix that another seed or config can match in about 2**32 tries."""
    config = tmp_path / "private.json"
    config.write_text(json.dumps({"nonce": secrets.token_hex(16), "domains": {"public": ["earthquakes"],
                                  "held_out": ["neo"]}, "genres": {"public": ["classic"], "held_out": []}}),
                      encoding="utf-8")
    log: list[str] = []
    run = pipeline.generate(pipeline.GenerateConfig(seed=7, instances=1, mix="random", private_config=str(config)),
                            tmp_path / "run", log=log.append)
    manifest = read_json(run / "manifest.json")
    assert manifest["private_config_commitment"] == config_commitment(read_json(config))
    for field in ("seed_commitment", "private_config_commitment"):
        line, = [line for line in log if manifest[field] in line]
        assert re.search(r"\b[0-9a-f]{64}\b", line) and field in line and "publishable" in line


def test_a_small_seed_is_warned_about_on_the_console_only(tmp_path, capsys, monkeypatch):
    """A small seed can be recovered from one prompt in seconds (below: from a list of candidates), so an
    explicit seed of fewer than 48 bits draws a warning, which changes no file of the run. (The fresh seed
    is fixed here: a real one is below 2**47 once in 131,072 draws.)"""
    assert cli.main(["generate", "--seed", "37", "--instances", "1", "--out", str(tmp_path / "small")]) == 0
    assert "a seed this small can be guessed" in capsys.readouterr().err
    monkeypatch.setattr(cli, "fresh_seed", lambda: 16222556419173536875)
    assert cli.main(["generate", "--instances", "1", "--out", str(tmp_path / "fresh")]) == 0
    assert "can be guessed" not in capsys.readouterr().err
    assert [guessable_seed(s) for s in (0, 42, -5, 2 ** 47 - 1, 2 ** 47, 16222556419173536875)] == [
        True, True, True, True, False, False]
    # What the warning says: one prompt gives the seed back. Each candidate's compile time and pool of
    # event IDs follow from the public code alone.
    prompt = (tmp_path / "small" / "prompts" / "inst-000" / "v0.txt").read_text(encoding="utf-8")
    found = [seed for seed in range(64)
             if sum(e.id in prompt for e in generate_synthetic(
                 SeedLog(seed).rng("snapshot", "synthetic"), 360,
                 synthetic_compiled_at(SeedLog(seed).rng("snapshot", "clock")))) >= 10]
    assert found == [37]


def test_only_the_published_example_nonce_skips_the_nonce_check():
    """Before the fix, any nonce that contained "EXAMPLE" (NOT-AN-EXAMPLE-1) skipped the check, so a
    config whose held-out sets anyone could recover from its commitment was accepted."""
    assert read_json(EXAMPLE_CONFIG)["nonce"] == EXAMPLE_NONCE and is_example_nonce(EXAMPLE_NONCE)
    assert nonce_problem(EXAMPLE_NONCE) is None
    for nonce in ("NOT-AN-EXAMPLE-1", "MYEXAMPLE000000", "EXAMPLE", EXAMPLE_NONCE + "-mine", EXAMPLE_NONCE.lower(),
                  "EXAMPLE" + secrets.token_hex(16)):
        assert not is_example_nonce(nonce) and nonce_problem(nonce) == NONCE_ADVICE, nonce
