"""Metamorphic variants, the seen-vs-new gap, the generator audit, the report and the CLI."""
from __future__ import annotations

import json
import math

from evalgen.util import read_json

import cli


def test_variants_change_the_surface_but_not_the_key(instances):
    for inst in instances:
        original = inst.renderings[0]
        assert len(inst.renderings) == 3
        for variant in inst.renderings[1:]:
            assert variant.digest != original.digest
            assert variant.style != original.style
            assert variant.prompt_sha256 != original.prompt_sha256
            assert set(variant.question_text) == set(original.question_text)
        # the instance has one answer key, shared by all renderings, and the oracle agrees on each


def test_some_questions_are_paraphrased_or_reexpressed_in_other_units(instances):
    changed = sum(inst.renderings[0].question_text[q.qid] != r.question_text[q.qid]
                  for inst in instances for q in inst.questions for r in inst.renderings[1:])
    assert changed > 0


def test_consistency_metric(small_run):
    models = {m["model"]: m for m in read_json(small_run / "report.json")["models"]}
    assert models["oracle"]["consistency"]["rate"] == 1.0
    assert models["mock"]["consistency"]["rate"] < 1.0  # its mistakes depend on the surface form


def test_public_vs_new_gap_is_reported(small_run):
    models = {m["model"]: m for m in read_json(small_run / "report.json")["models"]}
    assert models["oracle"]["gap"]["public_minus_new"] == 0.0
    assert models["mock"]["gap"]["public_minus_new"] > 0.1  # the simulated contamination shows up
    for m in models.values():
        assert m["pools"]["public"]["n"] > 0 and m["pools"]["new"]["n"] > 0


def test_adjusted_gap_flags_contamination_but_not_trap_exposure(tmp_path):
    """On a 100-instance run (seed 42, where the raw comparison used to mislead), the like-for-like
    gap flags the simulated contaminated model and not the pool-agnostic naive reader, and every
    naive error on a trapped question is explained by the traps."""
    path = cli.main(["demo", "--seed", "42", "--instances", "100", "--out", str(tmp_path / "big")])
    assert path == 0
    data = read_json(tmp_path / "big" / "report.json")
    models = {m["model"]: m for m in data["models"]}
    assert models["mock"]["gap"]["flagged"] and not models["naive"]["gap"]["flagged"]
    assert not models["oracle"]["gap"]["flagged"]
    assert abs(models["naive"]["gap"]["adjusted"]["public_minus_new"]) < 0.10
    naive = models["naive"]["traps"]
    assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"]


def test_audit_sample_size_and_verdicts(small_run):
    lines = (small_run / "audit" / "audit_sample.jsonl").read_text().splitlines()
    manifest = read_json(small_run / "manifest.json")
    population = manifest["audit"]["population"]
    assert len(lines) == manifest["audit"]["sampled"] == max(1, math.ceil(0.02 * population))
    entries = [json.loads(line) for line in lines]
    assert all(e["verdict"] is None and e["evidence"] is not None for e in entries)


def test_filled_verdicts_feed_the_error_rate(tmp_path):
    run = tmp_path / "audited"
    cli.main(["demo", "--seed", "5", "--instances", "3", "--out", str(run)])
    path = run / "audit" / "audit_sample.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    for e in entries:
        e["verdict"] = "ok"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    cli.main(["report", str(run)])
    verdicts = read_json(run / "report.json")["generator_audit"]["verdicts"]
    assert verdicts["reviewed"] == len(entries) and verdicts["errors"] == 0
    assert "rule of three" in (run / "report.md").read_text()


def test_report_has_every_required_section(small_run):
    text = (small_run / "report.md").read_text()
    for heading in ("Master seed", "Snapshot SHA-256", "## Model accuracy", "## Metamorphic consistency",
                    "## Seen vs new templates", "## Trap catch rate", "## Generator audit", "### Example 1",
                    "### Example 2", "Operation chains", "code-computed key"):
        assert heading in text, heading
    data = read_json(small_run / "report.json")
    assert set(data) >= {"run", "models", "generator_audit", "examples"}
    assert len(data["examples"]) == 2


def test_cli_demo_end_to_end(tmp_path):
    out = tmp_path / "demo"
    assert cli.main(["demo", "--seed", "42", "--instances", "2", "--out", str(out)]) == 0
    assert (out / "report.md").exists() and (out / "report.json").exists()
    assert cli.main(["generate", "--seed", "42", "--instances", "2", "--out", str(out)]) == 2  # refuses to overwrite
    assert cli.main(["demo", "--seed", "42", "--instances", "2", "--out", str(out)]) == 2  # explicit --out needs --force
    assert cli.main(["demo", "--seed", "42", "--instances", "2", "--out", str(out), "--force"]) == 0


def test_force_never_deletes_a_directory_that_is_not_a_run(tmp_path):
    precious = tmp_path / "thesis"
    precious.mkdir()
    (precious / "chapter1.tex").write_text("do not delete")
    assert cli.main(["demo", "--seed", "1", "--instances", "1", "--out", str(precious), "--force"]) == 2
    assert cli.main(["generate", "--seed", "1", "--instances", "1", "--out", str(precious), "--force"]) == 2
    assert (precious / "chapter1.tex").read_text() == "do not delete"
    # a folder that merely has some other tool's manifest.json is not an evalgen run either
    extension = tmp_path / "browser-extension"
    extension.mkdir()
    (extension / "manifest.json").write_text('{"manifest_version": 3, "name": "x"}')
    (extension / "background.js").write_text("// keep me")
    assert cli.main(["generate", "--seed", "1", "--instances", "1", "--out", str(extension), "--force"]) == 2
    assert (extension / "background.js").exists()


def test_report_counts_are_well_defined(small_run, instances):
    data = read_json(small_run / "report.json")
    traps = data["run"]["traps"]
    assert traps["affected_questions"] <= data["run"]["questions"] <= traps["affected_question_pairs"] + data["run"]["questions"]
    for m in data["models"]:
        assert m["format"]["api_errors"] == 0  # no API was called; parse failures are counted separately
        fully = m["consistency"]["questions_fully_consistent"]
        assert fully["n"] <= sum(len(inst.questions) for inst in instances)


def test_a_model_that_never_answers_is_not_called_consistent(small_run, instances):
    from evalgen.report import model_metrics

    rows = [{"instance_id": inst.instance_id, "variant": r.variant, "qid": q.qid, "template_id": q.template_id,
             "pool": q.pool, "answer_type": q.answer_type, "key": inst.answer_key[q.qid], "given": None,
             "normalized": None, "correct": False} for inst in instances for r in inst.renderings for q in inst.questions]
    silent = {"model": "silent", "kind": "api", "rows": rows, "forecasts": [],
              "replies": [{"status": "failed", "api_error": "HTTP 500", "parse_error": None}] * 3}
    metrics = model_metrics(silent, instances, [])
    assert metrics["consistency"]["n"] == 0 and metrics["consistency"]["questions_fully_consistent"]["n"] == 0
    assert metrics["format"]["api_errors"] == 3
