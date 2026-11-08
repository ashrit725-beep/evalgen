"""0.2.0 output is frozen: `cli.py generate --seed 16222556419173536875 --instances 100` with no new
flags must produce byte-identical files in every later version. The hashes below were taken from
evalgen 0.2.0 before the 0.3.0 refactor.

Only manifest.json may differ, and only in fields that differ between two identical runs anyway
(run_id from the output folder, the random seed nonce and its commitment, the generation time) or
that describe the code itself (code_sha256, evalgen_version).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cli

SEED = 16222556419173536875
VOLATILE = {"run_id", "seed_commitment", "seed_nonce", "generated_at_utc", "code_sha256", "evalgen_version"}
ALL_FILES = "6bb81f876e69d34174328729a6c72a4731528f2b317817d73d4bafe59fc0f346"  # 405 files, manifest excluded
MANIFEST = "714c49b8447d4848546c58e3d9ab1786e557ec08a6769e441d4683d065840df2"  # minus VOLATILE keys
FILES = {
    "snapshot/events.json": "59dc898ad3c9c12eee228c1a5817ae588b64fa900fded56f4a16ec42c83fb393",
    "audit/audit_sample.jsonl": "32c5211c95d9f843e3faa7bf59e39d69561e7f837085fb594671ebda3666055a",
    "instances/inst-000.json": "c5dbed2af84a3d1626030dfdf173732a3ecf5899a15c0ac0d1f3dec95ac0a4ab",
    "instances/inst-031.json": "6fefa68621416db0f694d40a1b87001f5c5e5cb50bd51d2ebd212d0dd51b6462",
    "instances/inst-057.json": "a043829bd7a2127552fae2c1825b1675d531235af18d78a1f0c2f28cfdd9607e",
    "instances/inst-099.json": "ec4ac15669f5ab758906fd8b4b5dc5cb76e47dc8a56c05b82e3b3eb0b6a94f65",
    "prompts/inst-000/v0.txt": "b9652bd6c706a6fb335f791e96953a20c0a76187ae3af6fd37e48df2f6f8c3a2",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def run_020(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("regression") / "seed-16222556419173536875"
    assert cli.main(["generate", "--seed", str(SEED), "--instances", "100", "--out", str(out)]) == 0
    return out


@pytest.mark.parametrize("name", sorted(FILES))
def test_pinned_files_are_byte_identical_to_020(run_020, name):
    assert _sha((run_020 / name).read_bytes()) == FILES[name], name


def test_every_output_file_is_byte_identical_to_020(run_020):
    files = sorted(p.relative_to(run_020).as_posix() for p in run_020.rglob("*")
                   if p.is_file() and p.name != "manifest.json")
    assert len(files) == 405
    listing = "".join(f"{f}\t{_sha((run_020 / f).read_bytes())}\n" for f in files)
    assert _sha(listing.encode()) == ALL_FILES


def test_manifest_matches_020_apart_from_volatile_fields(run_020):
    manifest = {k: v for k, v in json.loads((run_020 / "manifest.json").read_text()).items() if k not in VOLATILE}
    assert _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()) == MANIFEST
