"""The final audit's findings on the command line, the private run config and received runs (area "cli").

Each test reproduces a finding (its ID opens the docstring) and pins its fix:

  - every error of the command line is `error: ...` with exit status 2 and no traceback, for any malformed
    input or received run, and verify-config's status 1 means MISMATCH, never anything else;
  - a verifier checks a revealed config against the commitment published before the evaluation
    (verify-config --commitment HEX CONFIG), not only against a run's manifest;
  - generate computes and checks everything a run needs (both commitments, the plugins, the snapshots, the
    instances and the self-check) before it deletes (--force) or creates the run folder, and leaves no
    half-made folder behind;
  - off the legacy path, a generator self-check below 100% fails the command.
"""
from __future__ import annotations

import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import cli
from evalgen import genres, mix, pipeline, registry
from evalgen.domains import get_domain
from evalgen.domains.base import _point_label_problems
from evalgen.mix import MixError, config_commitment, plan_mix, read_private_config, verify_config, verify_private_config
from evalgen.models import Instance
from evalgen.pipeline import GenerateConfig, run_name
from evalgen.report import load_instances
from evalgen.util import RunDirError, read_json, read_manifest

from conftest import TOY_PLUGIN, quiet
from test_usgs_forecast import fake_feed

ROOT = Path(__file__).resolve().parents[1]
TOY_GENRE_PLUGIN = ROOT / "tests" / "toy_genre.py"
EXAMPLE_CONFIG = ROOT / "configs" / "private.example.json"
NONCE = "8d41f0c2a97e3b65d1c40e2f9a7b3c58"
NOW = "2030-01-01T00:00:00Z"  # past every forecast window: resolve would grade every pending forecast


def error(capsys) -> str:
    """What the command printed on stderr: an `error: ...`, never a traceback."""
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err, err
    return err


def write_config(path: Path, *, public=("earthquakes", "weather"), held_out=("neo",), public_genres=("classic", "news"),
                 held_out_genres=("email",), nonce=NONCE, **extra) -> Path:
    """A private run config of built-in names (no plugins unless `extra` gives them)."""
    data = {"nonce": nonce, "domains": {"public": list(public), "held_out": list(held_out)},
            "genres": {"public": list(public_genres), "held_out": list(held_out_genres)}, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def edit_json(path: Path, change) -> None:
    data = read_json(path)
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def files_of(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def deep(levels: int) -> str:
    return "[" * levels + "]" * levels


@pytest.fixture(scope="module")
def legacy_run(tmp_path_factory) -> Path:
    """A small legacy run, answered by the oracle and scored: what a received run looks like."""
    run = tmp_path_factory.mktemp("legacy") / "run"
    pipeline.generate(GenerateConfig(seed=5, instances=2), run, log=quiet)
    pipeline.run_models(run, ["oracle"], log=quiet)
    pipeline.score_run(run, log=quiet)
    return run


@pytest.fixture(scope="module")
def config_run(tmp_path_factory) -> tuple[Path, Path]:
    """A random mix under a private run config of built-in names, answered by the oracle and scored."""
    folder = tmp_path_factory.mktemp("configured")
    private = write_config(folder / "private.json")
    run = folder / "run"
    pipeline.generate(GenerateConfig(seed=11, instances=2, variants=1, mix="random", private_config=str(private)), run,
                      log=quiet)
    pipeline.run_models(run, ["oracle"], private_config=str(private), log=quiet)
    pipeline.score_run(run, private_config=str(private), log=quiet)
    return run, private


@pytest.fixture(scope="module")
def forecast_run(tmp_path_factory) -> Path:
    """A legacy run of a frozen USGS feed, whose instances ask forecasts, answered by the oracle and scored."""
    folder = tmp_path_factory.mktemp("forecasts")
    feed = folder / "all_day.geojson"
    feed.write_bytes(fake_feed()[0])
    run = folder / "run"
    pipeline.generate(GenerateConfig(seed=11, instances=3, source="usgs", snapshot_file=str(feed)), run, log=quiet)
    pipeline.run_models(run, ["oracle"], log=quiet)
    pipeline.score_run(run, log=quiet)
    assert read_json(run / "forecasts.json")
    return run


def received(run: Path, where: Path, name: str = "run") -> Path:
    copy_ = where / name
    shutil.copytree(run, copy_)
    return copy_


# ------------------------------------------------------------------ the published commitment


def test_a_revealed_config_is_checked_against_the_commitment_published_before(tmp_path, capsys):
    """code-mix-cli#0: verify-config compared a revealed config only with the commitment in a run's manifest. A
    verifier has no run but one regenerated from the revealed config itself, whose manifest commits to that config:
    a config swapped after the commitment was published verified ("ok", exit 0)."""
    a = write_config(tmp_path / "A.json")
    b = write_config(tmp_path / "B.json", public=("earthquakes", "neo"), held_out=("weather",),
                     public_genres=("classic", "email"), held_out_genres=("news",))
    published = config_commitment(read_json(a))
    argv = ["generate", "--seed", "12345678901234567", "--instances", "1", "--variants", "0", "--mix", "random"]
    assert cli.main([*argv, "--private-config", str(a), "--out", str(tmp_path / "evaluator")]) == 0
    assert published in capsys.readouterr().out  # what the evaluator publishes before the evaluation
    assert cli.main([*argv, "--private-config", str(b), "--out", str(tmp_path / "regenerated")]) == 0
    capsys.readouterr()
    # Against a run, verify-config says that the config and the manifest agree, and on what commitment, which the
    # verifier must compare with the published one.
    assert cli.main(["verify-config", str(tmp_path / "regenerated"), str(b)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ok: ") and config_commitment(read_json(b)) in out and "published" in out
    # Against the published value, the swapped config is a MISMATCH and the revealed one verifies.
    assert cli.main(["verify-config", "--commitment", published, str(b)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("MISMATCH: ") and config_commitment(read_json(b)) in out
    for given in (published, published.upper(), f" {published}\n"):
        assert cli.main(["verify-config", "--commitment", given, str(a)]) == 0
        assert capsys.readouterr().out.startswith("ok: ")
    assert verify_private_config(a, commitment=published) and not verify_private_config(b, commitment=published)
    assert verify_private_config(b, tmp_path / "regenerated") and verify_private_config(a, tmp_path / "evaluator")
    # What is not a commitment is an error, never a MISMATCH; and a check is against a run or a commitment.
    for bad in ("", "xyz", published[:63], published + "0", "g" * 64, "é" * 64, "K" * 64):
        assert cli.main(["verify-config", "--commitment", bad, str(a)]) == 2, bad
        assert "64 hexadecimal digits" in error(capsys)
    for argv in (["verify-config", "--commitment", published, str(tmp_path / "evaluator"), str(a)],
                 ["verify-config", str(a)]):
        assert cli.main(argv) == 2, argv
        assert "--commitment" in error(capsys)
    # The docs that a verifier reads say that the check is against the published value.
    note = read_json(EXAMPLE_CONFIG)["note"]
    for text in (mix.__doc__, note):
        assert "verify-config --commitment" in text and "published before the evaluation" in text


@pytest.mark.parametrize("value", [5, ["x"], "é" * 64, "abc", "AB" * 32, None])
def test_a_received_commitment_that_is_not_one_is_an_error(config_run, tmp_path, capsys, value):
    """code-mix-cli#1, sweep-cli#1, code-core#0: a manifest whose private_config_commitment is not 64 hexadecimal
    digits made secrets.compare_digest raise TypeError: a traceback with exit status 1, which a verifier's script
    reads as MISMATCH, in verify-config, and in run, score and report with --private-config."""
    run, private = config_run
    run = received(run, tmp_path)
    edit_json(run / "manifest.json", lambda m: m.update(private_config_commitment=value))
    for argv in (["verify-config", str(run), str(private)], ["report", str(run), "--private-config", str(private)],
                 ["run", str(run), "--models", "oracle", "--private-config", str(private)],
                 ["score", str(run), "--private-config", str(private)]):
        assert cli.main(argv) == 2, argv
        assert "private_config_commitment" in error(capsys), argv
    with pytest.raises(RunDirError, match="private_config_commitment"):
        read_manifest(run)
    assert verify_config(read_json(private), value) is False  # never an exception, whatever the value


# ------------------------------------------------------------------------ generate --force


def test_generate_force_never_replaces_the_folder_it_is_run_from(tmp_path, monkeypatch, capsys):
    """code-mix-cli#2, sweep-cli#6: `generate --out . --force` inside a run folder removed every file of the run,
    then failed on removing the current folder itself ("[Errno 22] Invalid argument"), and left nothing."""
    run = tmp_path / "runA"
    pipeline.generate(GenerateConfig(seed=5, instances=1), run, log=quiet)
    (run / "sub").mkdir()
    before = files_of(run)
    for where, out in ((run, "."), (run / "sub", ".."), (run, str(run)), (run / "sub", str(run))):
        monkeypatch.chdir(where)
        assert cli.main(["generate", "--seed", "6", "--instances", "1", "--out", out, "--force"]) == 2, (where, out)
        assert "you are in" in error(capsys)
        assert files_of(run) == before and (run / "sub").is_dir()
    monkeypatch.chdir(tmp_path)  # from outside it, --force replaces it
    assert cli.main(["generate", "--seed", "6", "--instances", "1", "--out", str(run), "--force"]) == 0
    assert read_json(run / "manifest.json")["seed"] == 6 and not (run / "sub").exists()
    # A run written into the current (empty) folder is named after that folder, not "" (Path(".").name).
    here = tmp_path / "fresh"
    here.mkdir()
    monkeypatch.chdir(here)
    assert cli.main(["generate", "--seed", "6", "--instances", "1", "--out", "."]) == 0
    assert read_json(here / "manifest.json")["run_id"] == "fresh"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes here")
def test_a_named_pipe_in_place_of_a_manifest_never_holds_generate_up(tmp_path, capsys):
    """sweep-cli#6: `generate --force` read the manifest of the folder it was to replace, to tell whether evalgen
    made it: a named pipe there made it wait for ever."""
    folder = tmp_path / "piped"
    folder.mkdir()
    (folder / "a.txt").write_text("keep", encoding="utf-8")
    os.mkfifo(folder / "manifest.json")
    outcome: list[bool] = []
    worker = threading.Thread(target=lambda: outcome.append(pipeline.is_run_dir(folder)), daemon=True)
    worker.start()
    worker.join(30)
    hung = worker.is_alive()
    if hung:  # let it go: a writer opens the pipe and closes it at once
        os.close(os.open(folder / "manifest.json", os.O_WRONLY | os.O_NONBLOCK))
        worker.join(30)
    assert not hung and outcome == [False]
    assert cli.main(["generate", "--seed", "4", "--instances", "1", "--out", str(folder), "--force"]) == 2
    assert "not an evalgen run directory" in error(capsys)
    assert (folder / "a.txt").read_text(encoding="utf-8") == "keep"


def test_force_replaces_only_a_folder_evalgen_made_and_never_through_a_link(tmp_path, capsys):
    """sweep-cli#6: --force deleted a folder whose `.evalgen-run` was a folder or a link, or whose manifest was any
    object with three keys, and a run folder given by a symbolic link failed with "[Errno None] None"."""
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("x", encoding="utf-8")
    cases = {
        "marker-folder": lambda f: (f / pipeline.RUN_MARKER).mkdir(),
        "marker-link": lambda f: (f / pipeline.RUN_MARKER).symlink_to(elsewhere),
        "hand-written": lambda f: (f / "manifest.json").write_text(
            json.dumps({"evalgen_version": "x", "run_id": "y", "self_check": {}}), encoding="utf-8"),
    }
    for name, plant in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "budget.xlsx").write_text("keep", encoding="utf-8")
        plant(folder)
        before = files_of(folder)
        assert not pipeline.is_run_dir(folder), name
        assert cli.main(["generate", "--seed", "4", "--instances", "1", "--out", str(folder), "--force"]) == 2, name
        assert "not an evalgen run directory" in error(capsys)
        assert files_of(folder) == before and (folder / "budget.xlsx").exists()
    real = pipeline.generate(GenerateConfig(seed=5, instances=1), tmp_path / "real", log=quiet)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    before = files_of(real)
    assert cli.main(["generate", "--seed", "4", "--instances", "1", "--out", str(link), "--force"]) == 2
    assert "symbolic link" in error(capsys) and files_of(real) == before
    # A folder evalgen made is told by its marker, or by its manifest alone, as git keeps a committed run.
    assert pipeline.is_run_dir(real)
    committed = tmp_path / "committed"
    committed.mkdir()
    shutil.copyfile(ROOT / "runs" / "compare-100-v2" / "manifest.json", committed / "manifest.json")
    assert pipeline.is_run_dir(committed)


# ------------------------------------------------------------------- received manifests


MANIFEST_EDITS = {
    "inventory-number": lambda m: m.update(inventory={"domains": {"public": [1], "held_out": []},
                                                      "genres": {"public": ["classic"], "held_out": []}}),
    "inventory-list": lambda m: m.update(inventory={"domains": {"public": [["x"]], "held_out": []},
                                                    "genres": {"public": ["classic"], "held_out": []}}),
    "inventory-split": lambda m: m.update(inventory={"domains": {"public": 5}, "genres": {}}),
    "no-templates": lambda m: m.pop("templates"),
    "templates-number": lambda m: m.update(templates=5),
    "templates-entries": lambda m: m.update(templates={"public": [5], "new": []}),
    "no-seed-commitment": lambda m: m.pop("seed_commitment"),
    "seed-commitment-number": lambda m: m.update(seed_commitment=5),
    "no-audit": lambda m: m.pop("audit"),
    "audit-number": lambda m: m.update(audit=5),
    "audit-text": lambda m: m["audit"].update(population="x"),
    "self-check-number": lambda m: m.update(self_check=5),
    "self-check-text": lambda m: m.update(self_check={"oracle_checks": "x"}),
    "self-check-problems": lambda m: m["self_check"].update(problems=[1]),
    "code-number": lambda m: m.update(code_sha256=5),
    "snapshot-number": lambda m: m.update(snapshot=5),
    "snapshot-far": lambda m: m["snapshot"].update(compiled_at=10 ** 20),
    "no-snapshot": lambda m: m.pop("snapshot"),
    "run-id-list": lambda m: m.update(run_id=["x"]),
}


@pytest.mark.parametrize("name", sorted(MANIFEST_EDITS))
def test_a_received_manifest_of_the_wrong_shape_is_an_error(legacy_run, tmp_path, capsys, name):
    """code-mix-cli#3, code-core#0: read_manifest checked the seed, the instances, the plugins, the config and the
    inventory one level deep; report, run, score and resolve then stopped on a traceback (exit 1) over an inventory
    of numbers or lists, a missing or malformed seed_commitment, templates, audit, self_check, code_sha256 or
    snapshot, and a compile time that no date can print."""
    run = received(legacy_run, tmp_path)
    edit_json(run / "manifest.json", MANIFEST_EDITS[name])
    for argv in (["report", str(run)], ["run", str(run), "--models", "oracle", "--force"], ["score", str(run)],
                 ["resolve", str(run)]):
        assert cli.main(argv) == 2, argv
        assert "manifest.json is not the manifest of an evalgen run" in error(capsys), argv


def test_every_manifest_evalgen_wrote_is_read(legacy_run, config_run, forecast_run):
    """The shape check refuses no manifest that evalgen wrote: those of every version kept in runs/ (0.1.0 to 0.3.0)
    and those of runs generated now, legacy or not, with a private run config or a live feed."""
    paths = sorted(ROOT.glob("runs/*/manifest.json")) + [run / "manifest.json" for run in (
        legacy_run, config_run[0], forecast_run)]
    assert len(paths) >= 8
    for path in paths:
        assert read_manifest(path.parent)["run_id"], path


def test_json_nested_too_deeply_is_an_error_everywhere(config_run, tmp_path, capsys):
    """code-mix-cli#7, sweep-cli#1: json raises RecursionError, not ValueError, for deeply nested text: a config
    nested 200,000 levels made verify-config exit 1 with a traceback (MISMATCH, to a script) and generate exit 1, and
    a manifest with such a key did the same in every step."""
    run, private = config_run
    nested = tmp_path / "deep.json"
    nested.write_text('{"nonce": "%s", "x": %s}' % (NONCE, deep(200000)), encoding="utf-8")
    for argv in (["verify-config", str(run), str(nested)], ["verify-config", "--commitment", "0" * 64, str(nested)],
                 ["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config", str(nested),
                  "--out", str(tmp_path / "x")]):
        assert cli.main(argv) == 2, argv
        assert "nested too deeply" in error(capsys), argv
    assert not (tmp_path / "x").exists()
    copy_ = received(run, tmp_path, "deep-manifest")
    text = (copy_ / "manifest.json").read_text(encoding="utf-8")
    (copy_ / "manifest.json").write_text(text[:-2] + f', "extra": {deep(200000)}}}\n', encoding="utf-8")
    for argv in (["report", str(copy_), "--private-config", str(private)], ["run", str(copy_), "--models", "oracle"],
                 ["score", str(copy_)], ["resolve", str(copy_)], ["verify-config", str(copy_), str(private)]):
        assert cli.main(argv) == 2, argv
        assert "nested too deeply" in error(capsys), argv


def test_a_plugin_path_that_loops_is_an_error(tmp_path, capsys):
    """code-mix-cli#7: Path.resolve raises RuntimeError on a symbolic link that loops on Python 3.9 only, so the same
    config ended in a traceback on 3.9 and in an error on 3.14. Plugin paths are now resolved by os.path.realpath, which
    raises on neither."""
    (tmp_path / "loop.py").symlink_to("loop.py")
    private = write_config(tmp_path / "private.json", plugins=["loop.py"])
    assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config", str(private),
                     "--out", str(tmp_path / "run")]) == 2
    assert "loop.py" in error(capsys) and not (tmp_path / "run").exists()
    assert registry._module_name(tmp_path / "loop.py").startswith("evalgen_plugin_")  # no RuntimeError on any Python


# ------------------------------------------------------------------------- forecasts.json


FORECAST_FILES = {
    "empty-object": [{}],
    "resolved-only": [{"status": "resolved"}],
    "resolved-without-outcome": [{"status": "resolved", "instance_id": "inst-000"}],
    "no-forecaster": [{"status": "pending", "rule": {"end": 1}, "domain": "weather"}],
    "text-end": [{"status": "pending", "rule": {"end": "x"}}],
}
FORECAST_EDITS = {  # one entry of the run's own forecasts changed
    "unknown-domain": lambda f: f.update(domain="nope"),
    "domain-without-forecaster": lambda f: f.update(domain="weather"),
    "domain-list": lambda f: f.update(domain=["x"]),
    "no-status": lambda f: f.pop("status"),
    "end-far": lambda f: f["rule"].update(end=10 ** 20),
    "radius-text": lambda f: f["rule"].update(radius_km="far"),
    "outcome-two": lambda f: f.update(status="resolved", outcome=2),
    "outcome-true": lambda f: f.update(status="resolved", outcome=True),
}


@pytest.mark.parametrize("name", sorted([*FORECAST_FILES, *FORECAST_EDITS]))
def test_a_received_forecasts_file_of_the_wrong_shape_is_an_error(forecast_run, tmp_path, capsys, name):
    """code-mix-cli#4, code-pipeline#0, code-earthquakes#1: read_forecasts took any list of objects, and resolve then
    looked up the domain each one names (KeyError for an unknown one, AttributeError for one that asks no forecasts,
    TypeError for a list) and read keys that may be absent; the report read outcomes that may be absent. Each ended
    in a traceback with exit status 1. Every entry is now checked before anything is resolved or fetched."""
    run = received(forecast_run, tmp_path)
    if name in FORECAST_FILES:
        (run / "forecasts.json").write_text(json.dumps(FORECAST_FILES[name]), encoding="utf-8")
    else:
        edit_json(run / "forecasts.json", lambda forecasts: FORECAST_EDITS[name](forecasts[0]))
    before = files_of(run)
    # The domain a forecast names is looked up by resolve alone (once the run's plugins are loaded): the report never
    # reads it, and reports such a run as it is.
    named_domain = name in ("unknown-domain", "domain-without-forecaster")
    for argv in (["resolve", str(run), "--now", NOW], *([] if named_domain else [["report", str(run)]])):
        assert cli.main(argv) == 2, argv
        assert "forecasts.json" in error(capsys), argv
    assert files_of(run) == before
    if named_domain:
        assert cli.main(["report", str(run)]) == 0


def test_the_forecasts_evalgen_writes_are_read(forecast_run, tmp_path, capsys):
    run = received(forecast_run, tmp_path)
    assert cli.main(["report", str(run)]) == 0
    pending = sum(1 for f in read_json(run / "forecasts.json") if f["status"] == "pending")
    assert pending and cli.main(["resolve", str(run), "--now", "2000-01-01T00:00:00Z"]) == 0  # nothing due yet
    assert f"{pending} still pending" in capsys.readouterr().out


# ------------------------------------------------------------------------- received instances


INSTANCE_EDITS = {
    "affected-qid": lambda d: d["traps"][0].update(affected={"q99": 3}),
    "affected-list": lambda d: d["traps"][0].update(affected=[1]),
    "fallen-list": lambda d: d["traps"][0].update(fallen=[1]),
    "answer-type": lambda d: d["questions"][0].update(answer_type="nonesuch"),
    "chain-number": lambda d: d["questions"][0].update(chain=5),
    "chain-op": lambda d: d["questions"][0].update(chain=[{"op": "nonesuch"}]),
    "features-list": lambda d: d["questions"][0].update(features=[1]),
    "pool-list": lambda d: d["questions"][0].update(pool=["x"]),
    "question-text-list": lambda d: d["renderings"][0].update(question_text=[1]),
    "domain-unknown": lambda d: d.update(domain="atlantis"),
    "genre-unknown": lambda d: d["renderings"][0].update(genre="nonesuch"),
    "prompt-number": lambda d: d["renderings"][0].update(prompt=5),
}


@pytest.mark.parametrize("name", sorted(INSTANCE_EDITS))
def test_a_received_instance_of_the_wrong_shape_is_an_error(legacy_run, tmp_path, capsys, name):
    """code-core#1: load_instances refused only wrong IDs, renderings and missing keys, so a trap that names a
    question the instance lacks (a bare StopIteration from Instance.question), an unregistered answer type, domain or
    genre, a chain that is not a list, or features and question texts of the wrong type ended in a traceback with exit
    status 1 in run, score or report."""
    run = received(legacy_run, tmp_path)
    path = run / "instances" / "inst-000.json"
    assert read_json(path)["traps"]
    edit_json(path, INSTANCE_EDITS[name])
    for argv in (["report", str(run)], ["score", str(run)], ["run", str(run), "--models", "oracle,mock", "--force"]):
        assert cli.main(argv) == 2, argv
        assert "inst-000.json" in error(capsys), argv


def test_an_instance_asked_for_a_question_it_lacks_says_which(legacy_run):
    inst = load_instances(legacy_run)[0]
    with pytest.raises(KeyError, match="q99"):
        inst.question("q99")
    assert Instance.from_dict(inst.to_dict()).to_dict() == inst.to_dict()


# ------------------------------------------------------------------ the private run config


def test_a_config_that_cannot_be_committed_to_is_refused_before_anything_is_written(tmp_path, capsys):
    """code-mix-cli#6, sweep-cli#3: JSON allows "\\ud800", which reads as a lone surrogate that UTF-8 cannot encode.
    The config passed every check; generate then made the run folder (with --force: after deleting the old run) and
    died on the commitment with a message that named neither the file nor the field, leaving the folder behind."""
    good = write_config(tmp_path / "good.json")
    run = tmp_path / "X"
    argv = ["generate", "--seed", "3", "--instances", "1", "--mix", "random"]
    assert cli.main([*argv, "--private-config", str(good), "--out", str(run)]) == 0
    capsys.readouterr()
    before = files_of(run)
    for field, text in (("note", good.read_text(encoding="utf-8")[:-1] + ', "note": "a\\ud800b"}'),
                        ("name", good.read_text(encoding="utf-8").replace('"neo"', '"neo\\ud800"'))):
        bad = tmp_path / f"surrogate-{field}.json"
        bad.write_text(text, encoding="utf-8")
        for force in ([], ["--force"]):
            assert cli.main([*argv, "--private-config", str(bad), "--out", str(run), *force]) == 2
            err = error(capsys)
            assert str(bad) in err and "surrogate" in err
            assert files_of(run) == before
        assert cli.main([*argv, "--private-config", str(bad), "--out", str(tmp_path / "new")]) == 2
        assert "surrogate" in error(capsys) and not (tmp_path / "new").exists()
        for check in (["verify-config", str(run), str(bad)], ["verify-config", "--commitment", "0" * 64, str(bad)]):
            assert cli.main(check) == 2
            assert "surrogate" in error(capsys)
        with pytest.raises(MixError, match="surrogate"):
            read_private_config(bad)


def test_verify_config_calls_only_a_mismatch_a_mismatch(config_run, tmp_path, capsys):
    """sweep-cli#9: a file that is not JSON was a MISMATCH (exit 1), so the right config saved with a byte-order mark
    or as UTF-16 was called "not the private run config", and a script could not tell a bad file from a swapped
    config. JSON readable as UTF-8, -16 or -32, with or without a byte-order mark, is now read as json reads bytes, by
    generate and verify-config alike; a file that is not one JSON value is an error (exit 2)."""
    run, private = config_run
    published = config_commitment(read_json(private))
    text = private.read_text(encoding="utf-8")
    for name, data in (("bom.json", b"\xef\xbb\xbf" + text.encode("utf-8")), ("utf16.json", text.encode("utf-16")),
                       ("utf32.json", text.encode("utf-32"))):
        (tmp_path / name).write_bytes(data)
        assert cli.main(["verify-config", str(run), str(tmp_path / name)]) == 0, name
        assert cli.main(["verify-config", "--commitment", published, str(tmp_path / name)]) == 0, name
        assert read_private_config(tmp_path / name).commitment == published
    capsys.readouterr()
    for name, data in (("not.json", b"not json"), ("empty.json", b""), ("latin1.json", b'{"note": "\xe9"}'),
                       ("long-integer.json", b"1" * 5000), ("two-values.json", b"{} {}")):
        (tmp_path / name).write_bytes(data)
        for argv in (["verify-config", str(run), str(tmp_path / name)],
                     ["verify-config", "--commitment", published, str(tmp_path / name)]):
            assert cli.main(argv) == 2, argv
            assert name in error(capsys)
    (tmp_path / "array.json").write_text("[1, 2]", encoding="utf-8")  # JSON, and not the config: a MISMATCH
    assert cli.main(["verify-config", str(run), str(tmp_path / "array.json")]) == 1
    assert capsys.readouterr().out.startswith("MISMATCH: ")


def test_an_unexpected_failure_is_an_error_and_never_a_mismatch(legacy_run, config_run, monkeypatch, capsys):
    """sweep-cli#1: whatever escapes the checks (a file of a received run, or code that a plugin runs, that no check
    foresaw) is `error: ...` with exit status 2, and verify-config's status 1 can only mean MISMATCH."""
    run, private = config_run

    def broken(*_args, **_kwargs):
        raise TypeError("nobody saw this coming")

    for name in ("revealed_commitment", "run_commitment"):
        with monkeypatch.context() as patch:
            patch.setattr(cli, name, broken)
            assert cli.main(["verify-config", str(run), str(private)]) == 2
            assert "nobody saw this coming" in error(capsys)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "build_report", broken)
        assert cli.main(["report", str(legacy_run)]) == 2
        assert "TypeError: nobody saw this coming" in error(capsys)

    def exits(*_args, **_kwargs):
        sys.exit(0)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "score_run", exits)
        assert cli.main(["score", str(legacy_run)]) == 2
        assert "sys.exit" in error(capsys)


# ------------------------------------------------------------------------------- plugins


BROKEN_DOMAIN = '''\
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("toy_source_of_{name}", {toy!r})
_module = importlib.util.module_from_spec(_spec)
sys.modules["toy_source_of_{name}"] = _module
_spec.loader.exec_module(_module)
DOMAIN = _module.Toy("{name}", located=True)
DOMAIN.fields = (5,)
'''


def test_a_plugin_domain_that_does_not_hold_together_is_a_config_error(tmp_path, capsys):
    """code-mix-cli#5: Domain.validate raised AttributeError on a field that is not a FieldSpec, which
    _register_module did not catch: a traceback with exit status 1 instead of the MixError load_plugin promises."""
    (tmp_path / "broken.py").write_text(BROKEN_DOMAIN.format(name="brokentoy", toy=str(TOY_PLUGIN)), encoding="utf-8")
    private = write_config(tmp_path / "private.json", public=("earthquakes",), held_out=("brokentoy",),
                           public_genres=("classic",), held_out_genres=(), plugins=["broken.py"])
    assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config", str(private),
                     "--out", str(tmp_path / "run")]) == 2
    err = error(capsys)
    assert "broken.py cannot be registered: AttributeError" in err and not (tmp_path / "run").exists()


def toy_genre_variant(name: str, service: str, body: str) -> str:
    """The toy genre plugin's source, defining instead a subclass of its Telex named `name` with `body`."""
    source = TOY_GENRE_PLUGIN.read_text(encoding="utf-8").replace("\nGENRE = Telex()\n", "\n")
    return source + f"\n\nclass Variant(Telex):\n{body}\n\nGENRE = Variant({name!r}, {service!r})\n"


PLUGIN_FAILURES = {
    "raises": ("    def render(self, ctx):\n        raise RuntimeError('render broke')\n", "render broke"),
    "keyerror": ("    def render(self, ctx):\n        return {}['missing']\n", "KeyError"),
    "exits": ("    def render(self, ctx):\n        import sys\n        sys.exit(0)\n", "sys.exit(0)"),
    "sniffs": ("    def sniff(self, digest_text):\n        raise ValueError('sniff broke')\n", "sniff broke"),
}


@pytest.mark.parametrize("name", sorted(PLUGIN_FAILURES))
def test_a_plugin_that_fails_while_the_run_is_built_leaves_everything_as_it_was(tmp_path, capsys, name):
    """sweep-cli#8: a plugin whose code failed during generation (not at import) ended in a traceback once the run
    folder was made (with --force: once the old run was deleted), and a plugin that called sys.exit(0) ended the
    command with status 0 and no run. The run is now built in full before its folder is touched, and a plugin's failure
    is an error that names the plugin."""
    body, said = PLUGIN_FAILURES[name]
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / f"{name}.py").write_text(toy_genre_variant(f"toy-{name}", name.title(), body), encoding="utf-8")
    private = write_config(folder / "private.json", public=("earthquakes",), held_out=(), public_genres=("classic",),
                           held_out_genres=(f"toy-{name}",), plugins=[f"{name}.py"])
    old = pipeline.generate(GenerateConfig(seed=5, instances=1), tmp_path / "old", log=quiet)
    before = files_of(old)
    argv = ["generate", "--seed", "3", "--instances", "1", "--mix", f"earthquakes:toy-{name}", "--private-config",
            str(private)]
    for out, force in ((old, ["--force"]), (tmp_path / "new", [])):
        assert cli.main([*argv, "--out", str(out), *force]) == 2
        err = error(capsys)
        assert f"{name}.py" in err and said in err, err
    assert files_of(old) == before and not (tmp_path / "new").exists()


def test_a_plugin_that_exits_when_it_is_loaded_is_a_config_error(tmp_path, capsys):
    """sweep-cli#8: a plugin calling sys.exit(0) as it ran ended `generate` with status 0, no run and no word."""
    (tmp_path / "exits.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    private = write_config(tmp_path / "private.json", plugins=["exits.py"])
    assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "random", "--private-config", str(private),
                     "--out", str(tmp_path / "run")]) == 2
    err = error(capsys)
    assert "exits.py called sys.exit(0)" in err and not (tmp_path / "run").exists()


def test_a_self_check_below_100_percent_fails_generate_off_the_legacy_path(tmp_path, capsys):
    """code-mix-cli#8: the self-check was counted and logged only, so a held-out genre that its own digests did not
    sniff as theirs (6 of 16 keys reproduced) passed with exit status 0 and the hint to run the models. The run is
    written, to be inspected, and the command fails with an error that says why."""
    fax = toy_genre_variant("fax", "Fax", "    def sniff(self, digest_text):\n        return False\n")
    (tmp_path / "fax.py").write_text(fax, encoding="utf-8")
    private = write_config(tmp_path / "private.json", public=("earthquakes",), held_out=(), public_genres=("classic",),
                           held_out_genres=("fax",), plugins=["fax.py"])
    run = tmp_path / "run"
    assert cli.main(["generate", "--seed", "777777777777777", "--instances", "2", "--variants", "1", "--mix",
                     "earthquakes:fax", "--private-config", str(private), "--out", str(run)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ") and "Traceback" not in captured.err
    assert "self-check" in captured.err and "is recognised as classic" in captured.err
    assert "next:" not in captured.out
    check = read_json(run / "manifest.json")["self_check"]  # written whole, for inspection
    assert check["oracle_agreements"] < check["oracle_checks"] and check["problems"]
    with pytest.raises(pipeline.SelfCheckError, match="self-check"):
        pipeline.generate(GenerateConfig(seed=777777777777777, instances=2, variants=1, mix="earthquakes:fax",
                                         private_config=str(private)), tmp_path / "again", log=quiet)


def test_the_docs_say_what_the_commitment_pins_and_what_a_regeneration_runs():
    """sweep-cli#2, sweep-cli#10, sweep-cli#11: the commitment pins a plugin's own bytes only, so a plugin must be
    self-contained; regenerating a revealed run executes the evaluator's plugin code; and generate's console names
    every domain of the run, held out or not."""
    note = read_json(EXAMPLE_CONFIG)["note"]
    for text in (mix.__doc__, note):
        assert "self-contained" in text and "executes" in text
    assert "console" in mix.__doc__ and "held-out" in mix.__doc__


def test_an_entry_registered_before_any_lookup_comes_after_the_built_in_ones():
    """code-core#2: Registry.register did not discover the built-in entries first, so an entry registered before the
    first lookup was listed (and sniffed) ahead of them, and one with a built-in's name broke every later lookup."""
    module = registry.load_module(TOY_GENRE_PLUGIN)
    built_in = [g.name for g in genres.Registry().all()]
    fresh = genres.Registry()
    fresh.register(module.GENRE)
    assert [g.name for g in fresh.all()] == [*built_in, "toy-telex"]
    clashing = genres.Registry()
    with pytest.raises(ValueError, match="two genres are named 'classic'"):
        clashing.register(module.Telex("classic", "Clash"))
    assert clashing.get("chat").name == "chat" and [g.name for g in clashing.all()] == built_in


def test_a_point_label_cannot_name_the_source_in_a_nested_format_spec():
    """code-domain-base#1: the check read only the top-level fields of a label, so a source nested in a format spec
    ("{number:{source[0]}<4}") passed validate and printed part of the source of the record a point is drawn near."""
    for label in ("COOP site {number:{source[0]}<4}", "station {number:{source}}", "site {number:>{source.upper}}"):
        problems = _point_label_problems(label, "point label", allowed=("number",))
        assert any("names the source" in p for p in problems), label
    assert any("'width'" in p for p in _point_label_problems("site {number:{width}}", "point label",
                                                              allowed=("number",)))
    assert _point_label_problems("station {number:>4}", "point label", allowed=("number",)) == []
    weather = copy.copy(get_domain("weather"))
    weather.spatial = replace(weather.spatial, point_label="COOP site {number:{source[0]}<4}")
    with pytest.raises(ValueError, match="names the source"):
        weather.validate()


# ------------------------------------------------------------------------------ counts and names


def test_counts_have_an_upper_bound(tmp_path, capsys):
    """sweep-cli#7: --instances and --variants had no upper bound: a typo started a job that ran until the disk or the
    memory ran out (415 instance files in 12 s for --instances 1000000000)."""
    out = tmp_path / "run"
    for argv in (["--instances", "1000000000"], ["--instances", str(pipeline.MAX_INSTANCES + 1)],
                 ["--variants", str(pipeline.MAX_VARIANTS + 1)], ["--instances", "1" * 40]):
        for command in (["generate", "--seed", "5"], ["demo"]):
            if command == ["demo"] and argv[0] == "--variants":
                continue
            with pytest.raises(SystemExit) as stopped:
                cli.main([*command, *argv, "--out", str(out)])
            assert stopped.value.code == 2 and "is not a whole number from" in capsys.readouterr().err
            assert not out.exists()
    for bad in (dict(instances=10 ** 9), dict(variants=10 ** 6)):
        assert GenerateConfig(seed=5, **bad).problems()
        with pytest.raises(ValueError, match="cannot generate this run"):
            pipeline.generate(GenerateConfig(seed=5, **bad), out, log=quiet)
    assert not out.exists()


def test_default_run_names_stay_short(tmp_path, monkeypatch, capsys):
    """sweep-cli#12: a run's default folder was named after its whole mix and seed, so a valid mix of every cell, or
    a long seed, failed with "File name too long"; and a long mix was echoed whole in its error."""
    cells = ",".join(f"{d}:{g}" for d in ("earthquakes", "invented", "neo", "weather")
                     for g in ("chat", "classic", "email", "form", "news", "radio", "table"))
    names = [run_name(GenerateConfig(seed=7, mix=cells)), run_name(GenerateConfig(seed=7, mix=cells[:-5] + "radio")),
             run_name(GenerateConfig(seed=int("9" * 300))), run_name(GenerateConfig(seed=7, mix=cells, source="live"))]
    assert all(len(name) <= pipeline.NAME_LIMIT for name in names) and len(set(names)) == len(names)
    assert names[0].startswith("earthquakes_chat_earthquakes_classic") and names[2].startswith("synthetic-seed999")
    assert run_name(GenerateConfig(seed=7)) == "synthetic-seed7"  # a name that fits is the 0.2.0 one
    assert run_name(GenerateConfig(seed=7, mix="weather")) == "weather-synthetic-seed7"
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "--seed", "7", "--instances", "1", "--variants", "0", "--mix", cells]) == 0
    assert (tmp_path / "runs" / names[0] / "manifest.json").is_file()
    capsys.readouterr()
    with pytest.raises(MixError, match="names an entry twice") as refused:
        plan_mix(",".join(["weather"] * 3000))
    assert len(str(refused.value)) < 400


def test_hints_quote_paths_and_seeds_are_plain_digits(tmp_path, capsys):
    """sweep-cli#13: the hints printed paths unquoted (this repository's own path has a space in it), and --seed took
    "1_000" for 1000 and Arabic-Indic digits for 42."""
    out = tmp_path / "dir with space" / "r"
    assert cli.main(["generate", "--seed", "1000", "--instances", "1", "--out", str(out)]) == 0
    assert f"next: python cli.py run {shlex.quote(str(out))}\n" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match=f"run `python cli.py run {shlex.quote(str(out))}` first"):
        pipeline.score_run(out, log=quiet)
    for seed in ("1_000", "٤٢", "0x10", "4.2", "1e3", "", "abc", "1" * 4301):
        with pytest.raises(SystemExit) as stopped:
            cli.main(["generate", "--seed", seed, "--instances", "1", "--out", str(tmp_path / "x")])
        assert stopped.value.code == 2 and "is not a seed" in capsys.readouterr().err, seed
    for seed, value in (("-5", -5), ("+5", 5), ("0", 0), (str(2 ** 64), 2 ** 64)):
        run = tmp_path / f"seed{len(seed)}{seed[0]}"
        assert cli.main(["generate", "--seed", seed, "--instances", "1", "--out", str(run)]) == 0
        assert read_json(run / "manifest.json")["seed"] == value
    assert not (tmp_path / "x").exists()


# ------------------------------------------------------------------- the run folder's lifecycle


def test_a_run_that_fails_while_it_is_built_or_written_leaves_no_folder(tmp_path, monkeypatch, capsys):
    """sweep-network#14: the run folder and its marker were made before any snapshot was built, so a source refused
    or crashed midway left a half-made folder (the marker, or the marker and some snapshots), and the next attempt
    to the same --out needed --force. Everything is now built before the folder is made, and a failure while the
    files are written removes what was written."""
    out = tmp_path / "run"
    real = pipeline.build_domain_snapshot

    def failing(domain, *args, **kwargs):
        if domain.name == "weather":
            raise ValueError("the weather feed could not be read")
        return real(domain, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "build_domain_snapshot", failing)
        assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "earthquakes,weather", "--out",
                         str(out)]) == 2
        assert "weather feed" in error(capsys) and not out.exists()

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    for fails, caught in ((full, OSError), (interrupted, KeyboardInterrupt)):
        with monkeypatch.context() as patch:
            patch.setattr(pipeline, "write_audit_sample", fails)
            with pytest.raises(caught):
                pipeline.generate(GenerateConfig(seed=3, instances=1, mix="earthquakes,weather"), out, log=quiet)
            assert not out.exists()
            out.mkdir()  # a folder that was there, empty, keeps nothing that generate wrote into it
            with pytest.raises(caught):
                pipeline.generate(GenerateConfig(seed=3, instances=1), out, log=quiet)
            assert out.is_dir() and list(out.iterdir()) == []
            out.rmdir()
    assert cli.main(["generate", "--seed", "3", "--instances", "1", "--mix", "earthquakes,weather", "--out",
                     str(out)]) == 0  # nothing was left that needs --force


def test_generate_runs_under_a_console_that_is_not_utf8(tmp_path):
    """sweep-determinism#0: under LC_ALL=en_US.US-ASCII (or de_DE.ISO8859-1) stdout cannot print the "…" of the
    snapshot line, so generate stopped midway (exit 2) and left a half-made run; a non-ASCII path in the hint failed
    the same way."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHONUTF8", "LC_", "LANG"))}
    env.update(PYTHONIOENCODING="ascii", HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
               ALL_PROXY="http://127.0.0.1:9")
    out = tmp_path / "résumé"
    result = subprocess.run([sys.executable, "-B", str(ROOT / "cli.py"), "generate", "--seed", "5", "--instances", "1",
                             "--out", str(out)], cwd=tmp_path, env=env, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stderr.decode("ascii", "replace")
    assert (out / "manifest.json").is_file() and b"next: python cli.py run" in result.stdout
