"""The same seed gives the same output; a different seed gives different instances."""
from __future__ import annotations

import hashlib
from pathlib import Path

from evalgen import pipeline
from evalgen.questions import fingerprint
from evalgen.seeds import SeedLog, derive_seed, fresh_seed, seed_commitment, verify_commitment
from evalgen.util import read_json

from conftest import quiet


def _generated_files(run_dir: Path) -> dict[str, bytes]:
    files = {}
    for sub in ("instances", "prompts", "snapshot", "audit"):
        for path in sorted((run_dir / sub).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(run_dir))] = path.read_bytes()
    return files


def test_seed_derivation_is_plain_sha256():
    expected = int.from_bytes(hashlib.sha256(b"42/instance/0/traps").digest()[:8], "big")
    assert derive_seed(42, "instance", 0, "traps") == expected
    log = SeedLog(42)
    assert log.rng("a", 1).random() == log.rng("a", 1).random()
    assert log.seeds == {"a/1": derive_seed(42, "a", 1)}


def test_same_seed_gives_byte_identical_instances(tmp_path):
    cfg = pipeline.GenerateConfig(seed=123, instances=4)
    a = pipeline.generate(cfg, tmp_path / "a", log=quiet)
    b = pipeline.generate(cfg, tmp_path / "b", log=quiet)
    files_a, files_b = _generated_files(a), _generated_files(b)
    assert files_a and files_a == files_b
    manifest_a, manifest_b = read_json(a / "manifest.json"), read_json(b / "manifest.json")
    assert manifest_a["seeds"] == manifest_b["seeds"]
    assert manifest_a["snapshot"]["events_sha256"] == manifest_b["snapshot"]["events_sha256"]


def test_same_seed_gives_identical_scores_and_metrics(tmp_path):
    reports = []
    for name in ("a", "b"):
        path = pipeline.demo(seed=99, instances=3, run_dir=tmp_path / name, log=quiet)
        reports.append(read_json(path.parent / "report.json"))
    assert reports[0]["models"] == reports[1]["models"]
    # everything about the run matches except its name and the commitment, which embeds a fresh nonce
    a, b = reports
    differs = {"run_id", "seed_commitment"}
    assert {k: v for k, v in a["run"].items() if k not in differs} == {k: v for k, v in b["run"].items() if k not in differs}
    assert a["run"]["seed_commitment"] != b["run"]["seed_commitment"]


def test_seed_commitment_verifies_and_cannot_be_brute_forced(tmp_path):
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=42, instances=1), tmp_path / "c", log=quiet)
    manifest = read_json(run_dir / "manifest.json")
    assert verify_commitment(42, manifest["seed_nonce"], manifest["seed_commitment"])
    assert not verify_commitment(43, manifest["seed_nonce"], manifest["seed_commitment"])
    # without the 128-bit nonce, trying every small seed does not find the commitment
    assert all(seed_commitment(s, "") != manifest["seed_commitment"] for s in range(1000))
    assert max(fresh_seed() for _ in range(4)).bit_length() > 32  # fresh seeds are 64-bit


def test_different_seed_gives_different_instances(tmp_path):
    a = pipeline.generate(pipeline.GenerateConfig(seed=1, instances=3), tmp_path / "a", log=quiet)
    b = pipeline.generate(pipeline.GenerateConfig(seed=2, instances=3), tmp_path / "b", log=quiet)
    prompts_a = [(a / "prompts" / f"inst-00{i}" / "v0.txt").read_text() for i in range(3)]
    prompts_b = [(b / "prompts" / f"inst-00{i}" / "v0.txt").read_text() for i in range(3)]
    assert all(x != y for x, y in zip(prompts_a, prompts_b))
    params_a = [q["chain"] for i in range(3) for q in read_json(a / "instances" / f"inst-00{i}.json")["questions"]]
    params_b = [q["chain"] for i in range(3) for q in read_json(b / "instances" / f"inst-00{i}.json")["questions"]]
    assert params_a != params_b


def test_no_two_instances_share_a_question(tmp_path):
    """Requirement 1: every metric is RNG-drawn and no concrete question is ever asked twice in a run."""
    for seed in (42, 7):
        run_dir = pipeline.generate(pipeline.GenerateConfig(seed=seed, instances=60), tmp_path / f"u{seed}", log=quiet)
        chains = [fingerprint(q["chain"]) for p in sorted((run_dir / "instances").glob("*.json"))
                  for q in read_json(p)["questions"]]
        assert len(chains) >= 60 * 3 and len(chains) == len(set(chains))
