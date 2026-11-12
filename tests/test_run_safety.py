"""Run directories as untrusted input (docs/DESIGN.md, "A received run cannot reach outside its folder"):
a run received from someone else can name files outside its folder through the IDs of its instances or
plant symbolic links, and no step may follow them; `generate --force` never deletes what the new run
reads; `report` never replaces a report with one built from no data; `resolve` refuses a path that is
not a run; and a prompt without a stored reply counts as unanswered."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import cli
from evalgen import pipeline
from evalgen.mix import config_commitment
from evalgen.report import load_instances
from evalgen.util import RunDirError, read_json, write_in, write_json

from conftest import quiet
from test_usgs_forecast import fake_feed

ROOT = Path(__file__).resolve().parents[1]
WEATHER_FEED = ROOT / "tests" / "fixtures" / "weather" / "raw_feed.json"


@pytest.fixture(scope="module")
def answered(tmp_path_factory) -> Path:
    """A small legacy run, answered by the oracle and scored: what a received run looks like."""
    run = tmp_path_factory.mktemp("answered") / "run"
    pipeline.generate(pipeline.GenerateConfig(seed=3, instances=2), run, log=quiet)
    pipeline.run_models(run, ["oracle"], log=quiet)
    pipeline.score_run(run, log=quiet)
    return run


def received(answered: Path, where: Path) -> Path:
    """A copy of the answered run, as if received from someone else."""
    copy = where / "received" / "run"
    shutil.copytree(answered, copy)
    return copy


def edit_instance(run: Path, name: str, change) -> None:
    path = run / "instances" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def files_under(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


# ---------------------------------------------------------------------- IDs that name files


@pytest.mark.parametrize("change", [
    lambda d, where: d.update(instance_id="../../../outside/pwned"),  # climbs out of responses/<model>/
    lambda d, where: d.update(instance_id=str(where / "abs" / "pwned")),  # an absolute path replaces the folder
    lambda d, where: d.update(instance_id="inst-001"),  # another instance's ID: its replies and key would mix
    lambda d, where: d["renderings"][0].update(variant="v0/../../../../outside2/settings"),
    lambda d, where: d["renderings"][1].update(variant="v0"),  # the same rendering twice
    lambda d, where: d.update(renderings=[]),
], ids=["climbing-id", "absolute-id", "borrowed-id", "climbing-variant", "variant-twice", "no-renderings"])
def test_a_received_run_cannot_name_a_file_outside_its_folder(answered, tmp_path, change, capsys):
    """run builds responses/<model>/<instance>__<variant>.json and score reads it back, from the IDs in the
    instance files: before the fix, an instance ID of "../../../outside/pwned" made `run` write three
    replies outside the run, and a variant of "v0/../../../../outside2/settings" overwrote a JSON file
    there."""
    run = received(answered, tmp_path)
    outside = tmp_path / "received" / "outside2"
    outside.mkdir()
    (outside / "settings.json").write_text('{"keep": true}\n', encoding="utf-8")
    before = files_under(tmp_path)
    edit_instance(run, "inst-000", lambda d: change(d, tmp_path))
    for step in (lambda: pipeline.run_models(run, ["oracle"], force=True, log=quiet),
                 lambda: pipeline.score_run(run, log=quiet), lambda: pipeline.build_report(run, log=quiet)):
        with pytest.raises(RunDirError, match="inst-000.json"):
            step()
    for argv in (["run", str(run), "--models", "oracle", "--force"], ["score", str(run)], ["report", str(run)]):
        assert cli.main(argv) == 2
        assert capsys.readouterr().err.startswith("error: ")
    assert files_under(tmp_path) == before  # nothing written anywhere, inside the run or out
    assert (outside / "settings.json").read_text(encoding="utf-8") == '{"keep": true}\n'


def test_an_instance_file_that_evalgen_did_not_write_is_refused_by_name(answered, tmp_path, capsys):
    run = received(answered, tmp_path)
    (run / "instances" / "inst-000.json").write_text('{"broken', encoding="utf-8")
    assert cli.main(["score", str(run)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "inst-000.json is not an instance that evalgen wrote" in err
    assert "Traceback" not in err


def test_the_instances_are_exactly_those_the_manifest_lists(answered, tmp_path):
    run = received(answered, tmp_path)
    shutil.copy(run / "instances" / "inst-001.json", run / "instances" / "inst-002.json")
    edit_instance(run, "inst-002", lambda d: d.update(instance_id="inst-002"))
    with pytest.raises(RunDirError, match="does not list: inst-002"):
        pipeline.score_run(run, log=quiet)
    for path in (run / "instances" / "inst-002.json", run / "instances" / "inst-001.json"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="holds 1 of the 2 instances its manifest lists"):
        pipeline.run_models(run, ["oracle"], log=quiet)


# --------------------------------------------------------------------------- symbolic links


def test_a_received_run_with_a_symbolic_link_is_refused(answered, tmp_path, capsys):
    """Before the fix, `report` wrote the report into whatever file a planted report.md pointed at, and
    `score` wrote its scores into whatever folder a planted scores/ pointed at."""
    victim = tmp_path / "victim.txt"
    victim.write_text("original\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    cases = {"report.md": lambda run: (run / "report.md").symlink_to(victim),
             "scores": lambda run: (shutil.rmtree(run / "scores"), (run / "scores").symlink_to(elsewhere)),
             "oracle": lambda run: (shutil.rmtree(run / "responses" / "oracle"),
                                    (run / "responses" / "oracle").symlink_to(elsewhere)),
             "manifest.json": lambda run: ((run / "manifest.json").rename(tmp_path / "moved.json"),
                                           (run / "manifest.json").symlink_to(tmp_path / "moved.json"))}
    for name, plant in cases.items():
        run = tmp_path / name / "run"
        shutil.copytree(answered, run)
        plant(run)
        for argv in (["report", str(run)], ["score", str(run)], ["run", str(run), "--models", "oracle", "--force"],
                     ["resolve", str(run)], ["verify-config", str(run), str(victim)]):
            assert cli.main(argv) == 2, (name, argv)
            err = capsys.readouterr().err
            assert err.startswith("error: ") and "is a symbolic link" in err, (name, argv, err)
    assert victim.read_text(encoding="utf-8") == "original\n" and list(elsewhere.iterdir()) == []


def test_the_one_writer_never_writes_through_a_link_or_out_of_the_run(tmp_path):
    root = tmp_path / "run"
    (root / "scores").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    os.link(outside, root / "scores" / "hard.json")  # a hard link in a file's place is replaced, not written through
    write_in(root, root / "scores" / "hard.json", "new\n")
    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert (root / "scores" / "hard.json").read_text(encoding="utf-8") == "new\n"
    (root / "linked.json").symlink_to(outside)
    for path in (root / "linked.json", root / ".." / "x.json", tmp_path / "x.json", root):
        with pytest.raises(RunDirError):
            write_in(root, path, "no\n")
    with pytest.raises(RunDirError, match="symbolic link"):
        write_json(root / "linked.json", {"no": 1})  # without a run directory, still never through a link
    assert outside.read_text(encoding="utf-8") == "outside\n" and not (tmp_path / "x.json").exists()
    # The writer writes exactly what Path.write_text and write_bytes do, and leaves no temporary file.
    write_in(root, root / "a" / "b.txt", "é\nline\n")
    write_in(root, root / "a" / "c.bin", b"\x00\xff")
    (tmp_path / "plain.txt").write_text("é\nline\n", encoding="utf-8")
    assert (root / "a" / "b.txt").read_bytes() == (tmp_path / "plain.txt").read_bytes()
    assert (root / "a" / "c.bin").read_bytes() == b"\x00\xff"
    assert sorted(p.name for p in (root / "a").iterdir()) == ["b.txt", "c.bin"]


# ------------------------------------------------------------------ generate --force and inputs


def test_generate_force_never_deletes_what_the_run_reads(tmp_path, capsys):
    """Before the fix, replaying a run's frozen live feed into that same folder with --force deleted the
    feed first (the folder may hold its only copy), and a private config kept in the folder was deleted
    without a word, although the new manifest committed to it."""
    live = tmp_path / "weather-live"
    assert cli.main(["generate", "--seed", "21", "--instances", "2", "--mix", "weather", "--source", "live",
                     "--snapshot-file", str(WEATHER_FEED), "--out", str(live)]) == 0
    frozen = (live / "snapshot" / "raw_feed.json").read_bytes()
    usgs = tmp_path / "usgs"
    raw, _ = fake_feed()
    feed = tmp_path / "feed.geojson"
    feed.write_bytes(raw)
    assert cli.main(["generate", "--seed", "21", "--instances", "1", "--source", "usgs", "--snapshot-file", str(feed),
                     "--out", str(usgs)]) == 0
    capsys.readouterr()
    config = tmp_path / "inside" / "private.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"nonce": "3f9a6c1e8b2d4f7a0c5e9b1d3f6a8c2e", "domains": {
        "public": ["earthquakes"], "held_out": ["neo"]}, "genres": {"public": ["classic"], "held_out": ["news"]},
        "plugins": []}), encoding="utf-8")
    assert cli.main(["generate", "--seed", "5", "--instances", "1", "--mix", "random", "--private-config", str(config),
                     "--out", str(config.parent)]) == 2  # the folder holds the config and is not a run
    assert "holds what this run reads" in capsys.readouterr().err and config.exists()
    linked = tmp_path / "linked.json"
    linked.symlink_to(live / "snapshot" / "raw_feed.json")  # outside the run, but the file it names is inside
    elsewhere = tmp_path / "elsewhere"  # frozen snapshots outside the run whose feed is a link into it
    elsewhere.mkdir()
    shutil.copy(live / "snapshot" / "snapshot.json", elsewhere / "snapshot.json")
    (elsewhere / "raw_feed.json").symlink_to(live / "snapshot" / "raw_feed.json")
    cases = [(live, ["--mix", "weather", "--source", "live", "--snapshot-dir", str(live / "snapshot")]),
             (live, ["--mix", "weather", "--source", "live", "--snapshot-dir", str(elsewhere)]),
             (live, ["--mix", "weather", "--source", "live", "--snapshot-dir", str(live)]),
             (live, ["--mix", "weather", "--source", "live", "--snapshot-file", str(live / "snapshot" / "raw_feed.json")]),
             (live, ["--mix", "weather", "--source", "live", "--snapshot-file", str(linked)]),
             (usgs, ["--source", "usgs", "--snapshot-file", str(usgs / "snapshot" / "raw_feed.geojson")])]
    pipeline.generate(pipeline.GenerateConfig(seed=5, instances=1, mix="earthquakes"), tmp_path / "with-config",
                      log=quiet)
    shutil.copy(config, tmp_path / "with-config" / "private.json")
    cases.append((tmp_path / "with-config", ["--mix", "random", "--private-config",
                                             str(tmp_path / "with-config" / "private.json")]))
    for run, options in cases:
        before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
        for force in ([], ["--force"]):
            assert cli.main(["generate", "--seed", "21", "--instances", "1", *options, "--out", str(run), *force]) == 2
            err = capsys.readouterr().err
            assert err.startswith(f"error: {run} holds what this run reads") and "--out" in err, err
            assert "use --force" not in err
        assert {path: path.read_bytes() for path in run.rglob("*") if path.is_file()} == before, options
    assert (live / "snapshot" / "raw_feed.json").read_bytes() == frozen
    # Replayed into another folder, the frozen feed regenerates the run, and --force replaces a run that
    # holds none of the inputs.
    again = tmp_path / "again"
    for _ in range(2):
        assert cli.main(["generate", "--seed", "21", "--instances", "2", "--mix", "weather", "--source", "live",
                         "--snapshot-dir", str(live / "snapshot"), "--out", str(again), "--force"]) == 0
    assert (again / "prompts").is_dir() and read_json(again / "manifest.json")["config"]["snapshot_dir"]


# ------------------------------------------------------------------------- committed runs


def test_report_never_replaces_a_report_with_one_built_from_no_data(tmp_path, capsys):
    """git keeps a run's manifest and report only. Before the fix, `report` on a committed run (or on a
    fresh clone) replaced its 41,636-byte report with an empty one, "Instances: 0", and exited 0."""
    committed = tmp_path / "compare-100-v2"
    shutil.copytree(ROOT / "runs" / "compare-100-v2", committed)
    before = {name: (committed / name).read_bytes() for name in ("report.md", "report.json")}
    assert cli.main(["report", str(committed)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "holds 0 of the 100 instances its manifest lists" in err
    assert "regenerate it" in err
    assert {name: (committed / name).read_bytes() for name in before} == before
    for argv in (["run", str(committed), "--models", "oracle"], ["score", str(committed)]):
        assert cli.main(argv) == 2
        capsys.readouterr()
    assert sorted(p.name for p in committed.iterdir()) == sorted(p.name for p in (ROOT / "runs" / "compare-100-v2").iterdir())
    # A run that has no scores yet has no report to write either.
    fresh = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), tmp_path / "fresh", log=quiet)
    assert cli.main(["report", str(fresh)]) == 2
    assert "no scores in" in capsys.readouterr().err and not (fresh / "report.md").exists()


def test_resolve_refuses_a_path_that_is_not_a_run(answered, tmp_path, capsys):
    """Before the fix, `resolve` on a mistyped path said the run had no forecasts and exited 0, so a
    scheduled resolve never resolved anything and never told anyone."""
    (tmp_path / "afile.txt").write_text("x", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    for path in (tmp_path / "no" / "such" / "run", tmp_path / "afile.txt", tmp_path / "empty"):
        assert cli.main(["resolve", str(path)]) == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "is not an evalgen run directory" in err
    assert cli.main(["resolve", str(answered)]) == 0
    assert "--source live with earthquakes in the mix" in capsys.readouterr().out


# ---------------------------------------------------------------------------- missing replies


def test_a_prompt_without_a_reply_counts_as_unanswered(tmp_path):
    """Before the fix, `score` skipped prompts without a stored reply, so accuracy was over the answered
    ones ("5/7 correct" of a run with 13 original questions) and nothing said the run was incomplete; a
    broken reply file stopped `score` with a traceback."""
    run = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=3), tmp_path / "run", log=quiet)
    pipeline.run_models(run, ["oracle"], log=quiet)
    replies = run / "responses" / "oracle"
    (replies / "inst-000__v0.json").unlink()
    (replies / "inst-001__v2.json").unlink()
    (replies / "inst-002__v1.json").write_text('{"model": "oracle", "te', encoding="utf-8")  # broken off
    log: list[str] = []
    pipeline.score_run(run, log=log.append)
    score = read_json(run / "scores" / "oracle.json")
    instances = load_instances(run)
    prompts = sum(len(inst.renderings) for inst in instances)
    originals = sum(len(inst.questions) for inst in instances)
    assert len(score["replies"]) == prompts and len(score["rows"]) == sum(
        len(inst.questions) * len(inst.renderings) for inst in instances)
    missing = [(r["instance_id"], r["variant"]) for r in score["replies"] if r["status"] == "missing"]
    assert missing == [("inst-000", "v0"), ("inst-001", "v2"), ("inst-002", "v1")]
    first = len(instances[0].questions)
    assert f"{originals - first}/{originals} correct on original renderings" in log[0]
    assert any("no stored reply to 3 of 9 prompts" in line for line in log)
    pipeline.build_report(run, log=quiet)
    report = read_json(run / "report.json")
    (oracle,) = report["models"]
    assert oracle["accuracy"]["n"] == originals and oracle["format"]["missing"] == 3
    # Agreement and the traps compare the replies stored: a missing one is a gap in the run, not an answer.
    assert oracle["consistency"]["rate"] == 1.0 and oracle["traps"]["caught"]["rate"] == 1.0
    assert oracle["accuracy_all_renderings"]["n"] == len(score["rows"])
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert "**oracle**: 3 of the run's 9 prompts have no stored reply" in markdown
    assert "3/9 replies missing" in "\n".join(pipeline.summary_lines(run))
    # The next run asks exactly those three prompts again, and the run is whole.
    log = []
    pipeline.run_models(run, ["oracle"], log=log.append)
    assert "oracle: 3 prompt(s) answered" in log[-1]
    pipeline.score_run(run, log=quiet)
    pipeline.build_report(run, log=quiet)
    (oracle,) = read_json(run / "report.json")["models"]
    assert oracle["accuracy"]["rate"] == 1.0 and "missing" not in oracle["format"]
    assert not any(r["status"] == "missing" for r in read_json(run / "scores" / "oracle.json")["replies"])
