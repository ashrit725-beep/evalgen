"""Shared fixtures: one small but complete synthetic run, reused by several test modules."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow plain `pytest` as well as `python -m pytest`
    sys.path.insert(0, str(ROOT))

from evalgen import pipeline  # noqa: E402
from evalgen.report import load_instances  # noqa: E402
from evalgen.sources import load_snapshot  # noqa: E402


def quiet(*_args) -> None:
    pass


@pytest.fixture(scope="session")
def small_run(tmp_path_factory) -> Path:
    """generate -> run (mock, oracle, naive) -> score -> report, for 8 synthetic instances."""
    run_dir = tmp_path_factory.mktemp("runs") / "synthetic-seed7"
    pipeline.generate(pipeline.GenerateConfig(seed=7, instances=8), run_dir, log=quiet)
    pipeline.run_models(run_dir, ["mock", "oracle", "naive"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)
    return run_dir


@pytest.fixture(scope="session")
def instances(small_run):
    return load_instances(small_run)


@pytest.fixture(scope="session")
def snapshot_events(small_run):
    return {e.id: e for e in load_snapshot(small_run / "snapshot").events}
