"""Shared fixtures: one small but complete synthetic run, reused by several test modules, and the
test-only toy domains."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow plain `pytest` as well as `python -m pytest`
    sys.path.insert(0, str(ROOT))

from evalgen import domains, pipeline, scoring  # noqa: E402
from evalgen.report import load_instances  # noqa: E402
from evalgen.sources import load_snapshot  # noqa: E402

TOY_PLUGIN = Path(__file__).resolve().parent / "toy_domain.py"


def quiet(*_args) -> None:
    pass


@pytest.fixture
def answer_registry(monkeypatch):
    """Answer types registered during a test (directly, or by registering a domain) stay in that test."""
    monkeypatch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))


@pytest.fixture
def toys(monkeypatch, answer_registry):
    """The test-only toy domains of toy_domain.py, loaded through the plugin loader, as a held-out
    domain is, into a domain registry of their own: the built-in registry stays as it is, and the toy
    answer types stay registered for the test only. The module's FLAT and DRESSED variants are
    registered beside its DOMAIN."""
    monkeypatch.setattr(domains, "REGISTRY", domains.Registry())
    toy = domains.load_plugin(TOY_PLUGIN)
    module = sys.modules[type(toy).__module__]
    for variant in (module.FLAT, module.DRESSED):
        domains.REGISTRY.register(variant)
    return SimpleNamespace(toy=toy, flat=module.FLAT, dressed=module.DRESSED, module=module)


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
