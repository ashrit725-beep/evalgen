"""Shared fixtures: one small but complete synthetic run, reused by several test modules, and the
test-only toy domains; and what keeps every test from seeing what another registered."""
from __future__ import annotations

import contextlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow plain `pytest` as well as `python -m pytest`
    sys.path.insert(0, str(ROOT))

from evalgen import domains, genres, mix, pipeline, registry, scoring  # noqa: E402
from evalgen.report import load_instances  # noqa: E402
from evalgen.sources import load_snapshot  # noqa: E402

TOY_PLUGIN = Path(__file__).resolve().parent / "toy_domain.py"


def quiet(*_args) -> None:
    pass


# ------------------------------------------------------------------- what a test leaves behind

# What evalgen keeps for the whole process and a test can change: the domain and genre registries (their
# entries, which of them are built in or a private run config's, the files plugins were loaded from, and
# whether the built-in modules have been discovered), the scorer's answer types (registering a domain
# registers its own, into whichever table is current) and the plugin files of each private run config that
# the process has loaded. A test that loads plugins through the command line or a private run config, as a
# real run does, registers them for the rest of the process: a later test then found the toy domain and the
# toy-telex genre beside the built-in ones, and its result depended on which tests ran before it. (The plugin
# modules stay loaded, as in any process: a process runs one version of each plugin file, by its path and its
# bytes, and loading the file again registers the same entries without running it.)
SHARED = ((domains, "REGISTRY"), (genres, "REGISTRY"), (scoring, "ANSWER_TYPES"), (mix, "_CONFIG_PLUGINS"))


def _contents(value: dict | registry.Registry) -> dict:
    """What a table holds (its items), or a registry (its attributes, each container copied: the entries
    themselves are kept, as are the answer types of a table)."""
    if isinstance(value, registry.Registry):
        return {key: copy.copy(item) for key, item in vars(value).items()}
    return dict(value)


@contextlib.contextmanager
def left_as_found() -> Iterator[None]:
    """Put everything in SHARED back as it was, the same objects with the same contents, however the code
    inside changed it (a plugin registered, a registry discovered, a module attribute replaced)."""
    saved = [(owner, name, getattr(owner, name)) for owner, name in SHARED]
    saved = [(owner, name, value, _contents(value)) for owner, name, value in saved]
    try:
        yield
    finally:
        for owner, name, value, contents in saved:
            setattr(owner, name, value)
            held = vars(value) if isinstance(value, registry.Registry) else value
            held.clear()
            held.update(contents)


@pytest.fixture(autouse=True, scope="module")
def _module_leaves_shared_state_as_found() -> Iterator[None]:
    """What a module's own fixtures register (a run made with a private run config) ends with the module."""
    with left_as_found():
        yield


@pytest.fixture(autouse=True)
def _test_leaves_shared_state_as_found() -> Iterator[None]:
    """What a test registers stays in that test, so that no test's outcome depends on the tests run before
    it; what its module's fixtures set up before it is kept for the module's other tests."""
    with left_as_found():
        yield


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
