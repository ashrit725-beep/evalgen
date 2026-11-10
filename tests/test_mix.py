"""Plan phase 4: mixes, the private run config, held-out domains and genres, and the report metrics
they feed (see evalgen.mix).

The tests build against the registries: the earthquake domain and the classic genre, which are built
in, and the test-only toy domain (toy_domain.py) and toy-telex genre (toy_genre.py), which are loaded
through the plugin loader, as held-out sets are, into registries of their own. The 0.2.0 output stays
pinned by test_regression_020.py, and the legacy report is pinned here."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import secrets
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import cli
from evalgen import domains, genres, pipeline, scoring
from evalgen.adapters import AdapterError, MockAdapter, Task, make_adapter
from evalgen.answers import parse_reply
from evalgen.domains import get_domain
from evalgen.domains import weather as wx
from evalgen.domains.weather import STATIONS, WEATHER
from evalgen.genres import get_genre
from evalgen.mix import (HELD_OUT, NONCE_ADVICE, PUBLIC, Inventory, MixError, UnknownName, _genre_plan,
                         config_commitment, config_problems, is_example_nonce, load_plugin, plan_mix, verify_config,
                         verify_private_config)
from evalgen.pipeline import GenerateConfig, question_specs, run_name
from evalgen.report import answer_family, load_instances, model_metrics
from evalgen.scoring import is_correct, normalize
from evalgen.seeds import SeedLog, derive_seed
from evalgen.util import canonical_json, read_json, slugify, write_json

from conftest import TOY_PLUGIN, quiet
from test_domain_neo import FIXTURE_BYTES as CAD_BYTES
from test_domain_weather import FETCHED, nws_feature
from test_domain_weather import FIXTURE as NWS_FIXTURE
from test_usgs_forecast import fake_feed

TOY_GENRE_PLUGIN = Path(__file__).resolve().parent / "toy_genre.py"
PENALTY = 0.45
MODELS = ["oracle", "naive", "mock", f"mock:held_out_domains={PENALTY}", f"mock:held_out_genres={PENALTY}"]

# small_run's report.json and report.md as evalgen wrote them before phase 4 (commit cd7a937, on Python
# 3.9 and 3.14), with the values that differ between two identical runs (the seed commitment) or that
# describe the code (its hash and version) masked as `_masked_report` masks them.
LEGACY_REPORT_JSON = "69d30fc2d56da18527c06fa13bb6bebfdebbd4d92c3c686b42559c0e919b8ed6"
LEGACY_REPORT_MD = "43565797f93ffcdfc348dfcc3edc0091dc2fe6843ae2a91e8576ea0f90d2cff0"

BOTH = '''\
"""A plugin that defines a domain and a genre at once."""
from evalgen.domains.earthquakes import Earthquakes
from evalgen.genres.classic import Classic

LOADS = []  # one entry per execution of this file


class Twin(Earthquakes):
    name = "earthquakes-twin"


class Plain(Classic):
    name = "plain"

    def sniff(self, digest_text):
        return False


DOMAIN = Twin()
GENRE = Plain()
LOADS.append(1)
'''


@pytest.fixture
def registries(monkeypatch, answer_registry):
    """Fresh domain and genre registries, which know only the built-in entries, for one test: what the
    test loads through a plugin or a private run config, and the answer types it registers, stay in it."""
    monkeypatch.setattr(domains, "REGISTRY", domains.Registry())
    monkeypatch.setattr(genres, "REGISTRY", genres.Registry())


def write_config(path: Path, *, public_domains=("earthquakes",), held_out_domains=("toy",), public_genres=("classic",),
                 held_out_genres=("toy-telex",), plugins=(TOY_PLUGIN, TOY_GENRE_PLUGIN), **extra) -> Path:
    """A private run config at `path`, with a fresh nonce and the plugins as absolute paths."""
    data = {"nonce": secrets.token_hex(16),
            "domains": {"public": list(public_domains), "held_out": list(held_out_domains)},
            "genres": {"public": list(public_genres), "held_out": list(held_out_genres)},
            "plugins": [str(Path(p).resolve()) for p in plugins], **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def pairs(plan) -> list[tuple[str, str]]:
    return [(domain.name, genre.name) for domain, genre in plan.options]


def drawn(run: Path) -> list[tuple]:
    """What each instance of a run was drawn as: its domain, its split and the genres of its renderings."""
    return [(inst.domain, inst.split, tuple(r.genre for r in inst.renderings)) for inst in load_instances(run)]


@pytest.fixture(scope="module")
def held_out_run(tmp_path_factory):
    """A random mix over a public and a held-out domain (earthquakes, toy) and genre (classic,
    toy-telex), from a private run config: 40 instances, answered by the reference solvers, the
    simulated model, and the simulated model with a penalty on held-out domains or on held-out genres;
    then scored and reported."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        patch.setattr(genres, "REGISTRY", genres.Registry())
        root = tmp_path_factory.mktemp("held-out")
        config = write_config(root / "private.json")
        log: list[str] = []
        run = pipeline.generate(GenerateConfig(seed=6, instances=40, mix="random", private_config=str(config)),
                                root / "run", log=log.append)
        pipeline.run_models(run, MODELS, log=quiet)
        pipeline.score_run(run, log=quiet)
        pipeline.build_report(run, log=quiet)
        report = read_json(run / "report.json")
        yield SimpleNamespace(run=run, config=config, log=log, report=report, instances=load_instances(run),
                              models={m["model"]: m for m in report["models"]},
                              inventory=Inventory.of_run(read_json(run / "manifest.json")))


@pytest.fixture(scope="module")
def tied_run(tmp_path_factory):
    """An explicit mix that ties the held-out domain to the held-out genre: earthquakes in either genre of
    the private run config, the toy domain in toy-telex alone (seed 6, 40 instances). Answered by the
    simulated model with a penalty on held-out domains, or on held-out genres; scored and reported."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scoring, "ANSWER_TYPES", dict(scoring.ANSWER_TYPES))
        patch.setattr(domains, "REGISTRY", domains.Registry())
        patch.setattr(genres, "REGISTRY", genres.Registry())
        root = tmp_path_factory.mktemp("tied")
        config = write_config(root / "private.json")
        run = pipeline.generate(GenerateConfig(seed=6, instances=40, mix="earthquakes,toy:toy-telex",
                                               private_config=str(config)), root / "run", log=quiet)
        pipeline.run_models(run, MODELS[3:], log=quiet)
        pipeline.score_run(run, log=quiet)
        pipeline.build_report(run, log=quiet)
        report = read_json(run / "report.json")
        yield SimpleNamespace(run=run, instances=load_instances(run), models={m["model"]: m for m in report["models"]},
                              inventory=Inventory.of_run(read_json(run / "manifest.json")))


# ------------------------------------------------------------------------------------ mixes


def test_the_legacy_mix_draws_nothing_and_takes_no_private_config(registries, tmp_path):
    plan = plan_mix("legacy")
    assert plan.legacy and plan.options == () and plan.inventory is None and plan.config is None
    seedlog = SeedLog(3)
    assert plan.draw(0, seedlog, 3) is None and seedlog.seeds == {}
    with pytest.raises(MixError, match="legacy run is the 0.2.0 run"):
        plan_mix("legacy", write_config(tmp_path / "private.json"))


def test_a_mix_lists_every_domain_and_genre_an_instance_may_be_drawn_as(registries, tmp_path):
    config = write_config(tmp_path / "private.json")
    listed = plan_mix("earthquakes:classic,toy", config)  # a bare domain: in any genre the config lists
    assert pairs(listed) == [("earthquakes", "classic"), ("toy", "classic"), ("toy", "toy-telex")]
    assert listed.inventory == Inventory({"earthquakes": PUBLIC, "toy": HELD_OUT},
                                         {"classic": PUBLIC, "toy-telex": HELD_OUT})
    assert pairs(plan_mix("random", config)) == [("earthquakes", "classic"), ("earthquakes", "toy-telex"),
                                                 ("toy", "classic"), ("toy", "toy-telex")]
    assert pairs(plan_mix("toy,earthquakes:classic", config))[0] == ("toy", "classic")  # in the order it names them
    # Without a config, every registered domain and genre is public, and random draws from all of them.
    bare = plan_mix("random")
    assert pairs(bare) == [(d.name, g.name) for d in domains.all_domains() for g in genres.all_genres()]
    assert set(bare.inventory.domains.values()) | set(bare.inventory.genres.values()) == {PUBLIC}
    assert pairs(plan_mix("earthquakes")) == [("earthquakes", g.name) for g in genres.all_genres()]
    # A mix may only name what the config lists, and only names some registry knows.
    toy_only = write_config(tmp_path / "toy-only.json", public_domains=("toy",), held_out_domains=())
    with pytest.raises(MixError, match="earthquakes, which the private run config does not list"):
        plan_mix("earthquakes:classic", toy_only)
    for text in ("atlantis", "toy:sonnet"):
        with pytest.raises(UnknownName, match="unknown"):
            plan_mix(text, config)
    assert issubclass(UnknownName, KeyError) and issubclass(MixError, ValueError)


def test_random_draws_come_from_each_instances_own_seed_path(registries, tmp_path):
    plan = plan_mix("random", write_config(tmp_path / "private.json"))
    seedlog = SeedLog(11)
    assignments = [plan.draw(i, seedlog, 3) for i in range(40)]
    assert sorted(seedlog.seeds) == sorted(f"instance/{i}/mix" for i in range(40))
    for i, assignment in enumerate(assignments):
        rng = random.Random(derive_seed(11, "instance", i, "mix"))  # the domain first, then the original's genre
        assert assignment.domain is rng.choice(plan.domains)
        assert assignment.genres[0] is rng.choice(plan.genres_for(assignment.domain))
        assert assignment.split == (HELD_OUT if assignment.domain.name == "toy" else PUBLIC)
    assert [plan.draw(i, SeedLog(11), 3) for i in range(40)] == assignments
    assert [plan.draw(i, SeedLog(12), 3) for i in range(40)] != assignments
    assert {a.domain.name for a in assignments} == {"earthquakes", "toy"}
    assert {g.name for a in assignments for g in a.genres} == {"classic", "toy-telex"}


def test_random_runs_repeat_exactly_from_their_seed(registries, tmp_path):
    config = write_config(tmp_path / "private.json")

    def generate(where: str, seed: int) -> Path:
        cfg = GenerateConfig(seed=seed, instances=6, mix="random", private_config=str(config))
        return pipeline.generate(cfg, tmp_path / where / "run", log=quiet)

    a, b, c = generate("a", 21), generate("b", 21), generate("c", 22)
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file() and p.name != "manifest.json")
    assert files == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file() and p.name != "manifest.json")
    assert [f for f in files if (a / f).read_bytes() != (b / f).read_bytes()] == []
    volatile = {"seed_commitment", "seed_nonce", "generated_at_utc"}
    manifests = [{k: v for k, v in read_json(run / "manifest.json").items() if k not in volatile} for run in (a, b)]
    assert manifests[0] == manifests[1]
    assert drawn(a) != drawn(c)
    assert all(f"instance/{i}/mix" in manifests[0]["seeds"] for i in range(6))


def test_an_explicit_list_is_honoured(registries, tmp_path):
    config = write_config(tmp_path / "private.json")
    run = pipeline.generate(GenerateConfig(seed=4, instances=12, mix="earthquakes:classic,toy",
                                           private_config=str(config)), tmp_path / "run", log=quiet)
    for domain, split, names in drawn(run):
        if domain == "earthquakes":
            assert split == PUBLIC and names == ("classic",) * 3
        else:  # two genres: the variants in the genre the original is not in
            assert (domain, split) == ("toy", HELD_OUT) and names[1] == names[2] != names[0]
    assert {d for d, _, _ in drawn(run)} == {"earthquakes", "toy"}
    assert {names[0] for d, _, names in drawn(run) if d == "toy"} == {"classic", "toy-telex"}
    manifest = read_json(run / "manifest.json")
    assert manifest["self_check"]["oracle_agreements"] == manifest["self_check"]["oracle_checks"] > 0
    assert manifest["self_check"]["problems"] == []
    # A mix that names one domain and genre draws nothing.
    single = pipeline.generate(GenerateConfig(seed=4, instances=3, mix="toy:toy-telex", private_config=str(config)),
                               tmp_path / "single", log=quiet)
    assert drawn(single) == [("toy", HELD_OUT, ("toy-telex",) * 3)] * 3
    assert not any(path.endswith("/mix") for path in read_json(single / "manifest.json")["seeds"])


def test_variants_are_rendered_in_genres_other_than_the_originals(registries, tmp_path):
    telex, = load_plugin(TOY_GENRE_PLUGIN)
    cable = genres.REGISTRY.register(sys.modules[type(telex).__module__].CABLE)  # a third genre
    config = write_config(tmp_path / "private.json", held_out_genres=("toy-telex", "toy-cable"))
    run = pipeline.generate(GenerateConfig(seed=8, instances=9, mix="toy", private_config=str(config)),
                            tmp_path / "run", log=quiet)
    instances = load_instances(run)
    for inst in instances:  # each other genre before any of them twice
        assert sorted(r.genre for r in inst.renderings) == ["classic", "toy-cable", "toy-telex"]
    assert len({inst.renderings[0].genre for inst in instances}) == 3
    check = read_json(run / "manifest.json")["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    # Every pair of renderings is in two genres, so cross-genre consistency compares all three pairs,
    # where metamorphic consistency compares the original with each variant.
    metrics = model_metrics(crafted(instances, lambda inst, r: True), instances, [], Inventory.of_run(
        read_json(run / "manifest.json")))
    questions = sum(len(inst.questions) for inst in instances)
    assert (metrics["cross_genre_consistency"]["n"], metrics["consistency"]["n"]) == (3 * questions, 2 * questions)
    assert metrics["cross_genre_consistency"]["rate"] == 1.0
    # The genre plan itself: the others in turn, again and again; one genre allowed, one genre throughout.
    classic = get_genre("classic")
    for seed in range(20):
        plan = _genre_plan([classic, telex, cable], 7, random.Random(seed))
        others = sorted({classic, telex, cable} - {plan[0]}, key=lambda g: g.name)
        for turn in (plan[1:3], plan[3:5], plan[5:7]):
            assert sorted(turn, key=lambda g: g.name) == others
    assert _genre_plan([telex], 3, random.Random(0)) == (telex,) * 3


# --------------------------------------------------------------------- the private run config


def test_the_commitment_is_the_sha256_of_the_configs_canonical_json():
    data = {"nonce": "9f" * 16, "domains": {"public": ["earthquakes"], "held_out": ["toy"]},
            "genres": {"public": ["classic"], "held_out": []}, "plugins": []}
    commitment = config_commitment(data)
    assert commitment == hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
    relaid = json.loads(json.dumps(dict(reversed(list(data.items()))), indent=8))  # how the file is laid out does not count
    assert config_commitment(relaid) == commitment and verify_config(relaid, commitment)
    for tampered in (dict(data, nonce="8f" * 16), dict(data, domains={"public": ["earthquakes", "toy"], "held_out": []}),
                     dict(data, note="revealed")):
        assert not verify_config(tampered, commitment)


def test_a_run_commits_to_its_private_config_and_the_revealed_config_verifies(registries, tmp_path):
    config = write_config(tmp_path / "private.json")
    log: list[str] = []
    run = pipeline.generate(GenerateConfig(seed=2, instances=2, mix="random", private_config=str(config)),
                            tmp_path / "run", log=log.append)
    data = json.loads(config.read_text(encoding="utf-8"))
    commitment = config_commitment(data)
    assert read_json(run / "manifest.json")["private_config_commitment"] == commitment
    assert any(commitment[:16] in line and "publishable" in line for line in log)
    assert verify_private_config(config, run)
    relaid = tmp_path / "relaid.json"
    relaid.write_text(json.dumps(dict(reversed(list(data.items()))), indent=8), encoding="utf-8")
    assert verify_private_config(relaid, run)
    for name, text in (("moved.json", json.dumps(dict(data, domains={"public": ["earthquakes", "toy"], "held_out": []}))),
                       ("nonce.json", json.dumps(dict(data, nonce=secrets.token_hex(16)))),
                       ("broken.json", "{not json")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        assert not verify_private_config(tmp_path / name, run), name
    assert cli.main(["verify-config", str(run), str(relaid)]) == 0
    assert cli.main(["verify-config", str(run), str(tmp_path / "moved.json")]) == 1
    # A run without a private config commits to none: every domain and genre in it is public.
    plain = pipeline.generate(GenerateConfig(seed=2, instances=2, mix="earthquakes"), tmp_path / "plain", log=quiet)
    manifest = read_json(plain / "manifest.json")
    assert "private_config_commitment" not in manifest and "plugins" not in manifest
    assert manifest["inventory"]["domains"] == {"public": ["earthquakes"], "held_out": []}
    assert manifest["inventory"]["genres"]["held_out"] == [] and {inst.split for inst in load_instances(plain)} == {PUBLIC}
    legacy = pipeline.generate(GenerateConfig(seed=2, instances=1), tmp_path / "legacy", log=quiet)
    for other in (plain, legacy):
        with pytest.raises(MixError, match="without a private run config"):
            verify_private_config(config, other)
        assert cli.main(["verify-config", str(other), str(config)]) == 2


def test_a_private_config_a_run_cannot_follow_is_refused_before_anything_is_written(registries, tmp_path):
    good = json.loads(write_config(tmp_path / "good.json").read_text(encoding="utf-8"))
    neither = tmp_path / "neither.py"
    neither.write_text("VALUE = 1\n", encoding="utf-8")
    cases = [
        (dict(good, nonce="too short"), "nonce must be at least 32 random hexadecimal digits"),
        ({k: v for k, v in good.items() if k != "genres"}, "it has no 'genres'"),
        (dict(good, seed=7), "'seed' is not a key"),
        (dict(good, domains={"public": ["toy"], "held_out": ["toy"]}), "lists toy more than once"),
        (dict(good, domains={"public": ["toy"]}), "exactly the lists 'public' and 'held_out'"),
        (dict(good, genres={"public": [], "held_out": []}), "it lists no genres"),
        (dict(good, plugins=str(TOY_PLUGIN)), "'plugins' must be a list of paths"),
        (dict(good, plugins=[str(tmp_path / "gone.py")]), "gone.py does not exist"),
        (dict(good, plugins=[str(neither)]), "defines neither DOMAIN nor GENRE"),
        (dict(good, domains={"public": ["earthquakes"], "held_out": ["atlantis"]}), "unknown domain 'atlantis'"),
        ([good], "not a JSON object"),
    ]
    for number, (data, message) in enumerate(cases):
        path = tmp_path / f"bad-{number}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(MixError, match=message):
            pipeline.generate(GenerateConfig(seed=1, mix="random", private_config=str(path)), tmp_path / "run",
                              log=quiet)
    (tmp_path / "text.json").write_text("nonce = 1", encoding="utf-8")
    with pytest.raises(MixError, match="is not JSON"):
        pipeline.generate(GenerateConfig(seed=1, mix="random", private_config=str(tmp_path / "text.json")),
                          tmp_path / "run", log=quiet)
    with pytest.raises(MixError, match="legacy run"):  # a legacy run is the 0.2.0 run, with nothing held out
        pipeline.generate(GenerateConfig(seed=1, private_config=str(tmp_path / "good.json")), tmp_path / "run",
                          log=quiet)
    with pytest.raises(FileNotFoundError):
        pipeline.generate(GenerateConfig(seed=1, mix="random", private_config=str(tmp_path / "missing.json")),
                          tmp_path / "run", log=quiet)
    assert not (tmp_path / "run").exists()


def test_a_config_that_gives_a_key_twice_is_malformed(registries, tmp_path, capsys):
    """JSON readers keep the last of two members with one key, so a revealed config could show a decoy first
    (domains with nothing held out, then the real ones) and still verify against the commitment. A file that
    gives a key twice in any one object is malformed: verify-config refuses to check it (exit 2), and no run
    is generated from it."""
    config = write_config(tmp_path / "private.json")
    run = pipeline.generate(GenerateConfig(seed=2, instances=1, mix="random", private_config=str(config)),
                            tmp_path / "run", log=quiet)
    data = json.loads(config.read_text(encoding="utf-8"))
    body = json.dumps(data)
    decoys = {
        "domains": body.replace('"domains": ', '"domains": {"public": ["earthquakes", "toy"], "held_out": []}, '
                                               '"domains": ', 1),
        "held_out": body.replace('"held_out": ["toy"]', '"held_out": [], "held_out": ["toy"]', 1),
    }
    for key, text in decoys.items():
        assert json.loads(text) == data and verify_config(json.loads(text), config_commitment(data))  # as read before
        path = tmp_path / f"decoy-{key}.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(MixError, match=f"malformed: it gives the key '{key}' twice in one object"):
            verify_private_config(path, run)
        assert cli.main(["verify-config", str(run), str(path)]) == 2
        assert "error:" in capsys.readouterr().err
        with pytest.raises(MixError, match=f"is malformed: it gives the key '{key}' twice"):
            pipeline.generate(GenerateConfig(seed=2, instances=1, mix="random", private_config=str(path)),
                              tmp_path / "refused", log=quiet)
    assert not (tmp_path / "refused").exists()
    assert cli.main(["verify-config", str(run), str(config)]) == 0  # the config itself still verifies


def test_a_config_commits_to_the_bytes_of_the_plugins_it_pins(registries, tmp_path, monkeypatch):
    """A config names its plugins by path, so its commitment covers what a held-out domain or genre is only
    through the SHA-256 it gives a plugin. A pinned plugin whose bytes differ is refused before any of its
    code runs, at generation and at every later step; a plugin given by its path alone loads as before, with
    a warning when it has changed since generation."""
    private = tmp_path / "private"
    private.mkdir()
    domain_file, genre_file = private / "toy_domain.py", private / "toy_genre.py"
    domain_file.write_bytes(TOY_PLUGIN.read_bytes())
    genre_file.write_bytes(TOY_GENRE_PLUGIN.read_bytes())
    sha = hashlib.sha256(domain_file.read_bytes()).hexdigest()
    data = json.loads(write_config(tmp_path / "private.json").read_text(encoding="utf-8"))
    data["plugins"] = [{"path": "private/toy_domain.py", "sha256": sha.upper()}, "private/toy_genre.py"]
    config = tmp_path / "private.json"
    config.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plan = plan_mix("random", config)
    assert plan.config.plugins == (domain_file.resolve(), genre_file.resolve())
    assert plan.config.plugin_sha256 == (sha, None) and plan.inventory.domain_split("toy") == HELD_OUT
    run = pipeline.generate(GenerateConfig(seed=4, instances=2, mix="random", private_config=str(config)),
                            tmp_path / "run", log=quiet)
    manifest = read_json(run / "manifest.json")
    assert manifest["private_config_commitment"] == config_commitment(data)  # the SHA-256 is part of what is committed
    assert [entry["sha256"] for entry in manifest["plugins"]] == [
        sha, hashlib.sha256(genre_file.read_bytes()).hexdigest()]

    marker = tmp_path / "EXECUTED.txt"
    planted = f"\nfrom pathlib import Path as _P\n_P({str(marker)!r}).write_text('ran')\n"
    domain_file.write_text(domain_file.read_text(encoding="utf-8") + planted, encoding="utf-8")
    again = GenerateConfig(seed=4, instances=2, mix="random", private_config=str(config))
    for step in (lambda: pipeline.run_models(run, ["oracle"], private_config=str(config), log=quiet),
                 lambda: pipeline.build_report(run, private_config=str(config), log=quiet),
                 lambda: pipeline.generate(again, tmp_path / "refused", log=quiet)):
        new_process(monkeypatch, [domain_file, genre_file])
        with pytest.raises(MixError, match=f"toy_domain.py is not the file its private run config commits to: its "
                                           f"SHA-256 is [0-9a-f]{{64}}, and the config gives {sha}"):
            step()
    assert not marker.exists() and not (tmp_path / "refused").exists()

    domain_file.write_bytes(TOY_PLUGIN.read_bytes())  # restored: the pinned bytes load again
    genre_file.write_text(genre_file.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    new_process(monkeypatch, [domain_file, genre_file])
    log: list[str] = []
    pipeline.run_models(run, ["oracle"], private_config=str(config), log=log.append)
    assert any("toy_genre.py has changed since the run was generated" in line for line in log)  # unpinned: a warning

    good = {"path": "x.py", "sha256": "ab" * 32}
    for entries in ([{"path": "x.py"}], [dict(good, sha256="ab" * 31)], [dict(good, sha256="zz" * 32)],
                    [dict(good, path="")], [dict(good, size=1)], [dict(good, sha256=None)], [dict(good, path=good)],
                    [5], [""]):
        problems = config_problems(dict(data, plugins=entries))
        assert any("'plugins' must be a list of paths" in p for p in problems), entries
    assert config_problems(dict(data, plugins=[good, "y.py", dict(good, sha256="AB" * 32)])) == []
    assert config_problems({k: v for k, v in data.items() if k != "plugins"}) == []  # a config without plugins


def test_a_plugin_that_its_registry_refuses_is_a_config_error(registries, tmp_path, capsys):
    """A plugin may not take a name that a built-in entry has, nor define something that is no domain: the
    run is refused with a MixError before anything is written, and the command line says so and exits 2."""
    plugins = {
        "clash.py": "from evalgen.domains.earthquakes import Earthquakes\n\n\nclass Clash(Earthquakes):\n"
                    "    pass\n\n\nDOMAIN = Clash()\n",
        "classic.py": "from evalgen.genres.classic import Classic\n\nGENRE = Classic()\n",
        "number.py": "DOMAIN = 5\n",
    }
    messages = {"clash.py": "two domains are named 'earthquakes'", "classic.py": "two genres are named 'classic'",
                "number.py": "evalgen_plugin_[0-9a-f]+ must define DOMAIN, an instance of"}
    for name, source in plugins.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
        config = write_config(tmp_path / f"{name}.json", plugins=(TOY_PLUGIN, tmp_path / name))
        with pytest.raises(MixError, match=f"the plugin .*{name} cannot be registered: {messages[name]}"):
            pipeline.generate(GenerateConfig(seed=1, mix="random", private_config=str(config)), tmp_path / "run",
                              log=quiet)
        out = tmp_path / "cli-run"
        assert cli.main(["generate", "--seed", "1", "--mix", "random", "--private-config", str(config),
                         "--out", str(out)]) == 2
        assert f"error: the plugin {(tmp_path / name).resolve()} cannot be registered" in capsys.readouterr().err
        assert not (tmp_path / "run").exists() and not out.exists()


def test_the_example_config_holds_out_the_test_only_entries_and_says_it_is_an_example(registries):
    path = Path(__file__).resolve().parents[1] / "configs" / "private.example.json"
    data = read_json(path)
    assert "must NOT be published" in data["note"] and "EXAMPLE" in data["nonce"]
    plan = plan_mix("random", path)
    assert plan.inventory == Inventory({"earthquakes": PUBLIC, "toy": HELD_OUT}, {"classic": PUBLIC, "toy-telex": HELD_OUT})
    assert [p.name for p in plan.config.plugins] == ["toy_domain.py", "toy_genre.py"]


def test_a_guessable_nonce_is_refused_and_the_example_nonce_is_flagged(registries, tmp_path):
    """The nonce stops anyone from recovering a small config from its commitment, so it must look drawn:
    32 hexadecimal digits or more, not a run of one digit or a pattern. The published example's nonce,
    which anyone can read, still serves demos, and generate warns loudly that its commitment can be
    guessed."""
    good = json.loads(write_config(tmp_path / "good.json").read_text(encoding="utf-8"))
    guessable = ("0000000000000000", "aaaaaaaaaaaaaaaa", "my-eval-nonce-01", "0" * 32, "ab" * 16,
                 "0123456789abcdef" * 2, "f" * 64, "0011223344556677" * 2, secrets.token_hex(15), " " * 40, 2 ** 130,
                 None)
    for nonce in guessable:
        assert config_problems(dict(good, nonce=nonce)) == [NONCE_ADVICE], nonce
    for nonce in (secrets.token_hex(16), secrets.token_hex(16).upper(), secrets.token_hex(32)):
        assert config_problems(dict(good, nonce=nonce)) == [], nonce
    example = Path(__file__).resolve().parents[1] / "configs" / "private.example.json"
    assert is_example_nonce(read_json(example)["nonce"]) and config_problems(read_json(example)) == []
    for path, warned in ((example, True), (tmp_path / "good.json", False)):
        log: list[str] = []
        pipeline.generate(GenerateConfig(seed=3, instances=1, mix="random", private_config=str(path)),
                          tmp_path / path.stem, log=log.append)
        warnings = [line for line in log if "WARNING" in line]
        assert bool(warnings) == warned
        assert all("EXAMPLE" in line and "recover which domains and genres" in line for line in warnings)


# ----------------------------------------------------------------------------------- plugins


def test_plugins_register_their_domain_and_genre(registries, tmp_path):
    assert "toy" not in [d.name for d in domains.all_domains()]
    assert "toy-telex" not in [g.name for g in genres.all_genres()]
    toy, = load_plugin(TOY_PLUGIN)
    telex, = load_plugin(TOY_GENRE_PLUGIN)
    assert get_domain("toy") is toy and get_genre("toy-telex") is telex
    assert domains.all_domains()[-1] is toy and genres.all_genres()[-1] is telex  # after the built-in entries
    assert load_plugin(TOY_PLUGIN) == [toy] and len(domains.all_domains()) == len(set(domains.all_domains()))
    both = tmp_path / "both.py"
    both.write_text(BOTH, encoding="utf-8")
    twin, plain = load_plugin(both)
    assert get_domain("earthquakes-twin") is twin and get_genre("plain") is plain
    assert sys.modules[type(twin).__module__].LOADS == [1]  # one file, executed once for both registries
    with pytest.raises(MixError, match="does not exist"):
        load_plugin(tmp_path / "gone.py")


def new_process(monkeypatch, plugins=()) -> None:
    """What a later step in a new process starts from: registries that know only the built-in entries,
    and none of the files `plugins` executed yet."""
    monkeypatch.setattr(domains, "REGISTRY", domains.Registry())
    monkeypatch.setattr(genres, "REGISTRY", genres.Registry())
    paths = {Path(p).resolve() for p in plugins}
    for name, module in list(sys.modules.items()):
        if name.startswith("evalgen_plugin_") and Path(getattr(module, "__file__", None) or "").resolve() in paths:
            monkeypatch.delitem(sys.modules, name)


def test_later_steps_load_the_plugins_a_run_was_generated_with(registries, tmp_path, monkeypatch, capsys):
    private = tmp_path / "private"
    private.mkdir()
    plugins = [private / "toy_domain.py", private / "toy_genre.py"]
    for source, copy in zip((TOY_PLUGIN, TOY_GENRE_PLUGIN), plugins):
        copy.write_bytes(source.read_bytes())
    config = write_config(tmp_path / "private.json", plugins=plugins)
    run = pipeline.generate(GenerateConfig(seed=9, instances=3, mix="random", private_config=str(config)),
                            tmp_path / "run", log=quiet)
    assert read_json(run / "manifest.json")["plugins"] == [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in plugins]
    steps = (lambda **kw: pipeline.run_models(run, ["oracle", "mock"], log=quiet, **kw),
             lambda **kw: pipeline.score_run(run, log=quiet, **kw), lambda **kw: pipeline.build_report(run, log=quiet, **kw))
    # Each later step may run in a new process: it loads the plugins from the private run config the
    # evaluator passes again, never from the paths the run names.
    for step in steps:
        new_process(monkeypatch, plugins)
        with pytest.raises(MixError, match="pass its private run config"):
            step()
        step(private_config=str(config))
        assert get_domain("toy").name == "toy" and get_genre("toy-telex").name == "toy-telex"
    # The same process that generated a run has run its plugins already: they are registered again.
    new_process(monkeypatch)
    pipeline.build_report(run, log=quiet)
    report = read_json(run / "report.json")
    oracle = {m["model"]: m for m in report["models"]}["oracle"]
    assert oracle["accuracy"]["rate"] == oracle["accuracy_all_renderings"]["rate"] == 1.0
    assert report["plugins_changed"] == []
    # A plugin edited since generation is loaded as the evaluator asked, and the report says so.
    log: list[str] = []
    plugins[1].write_text(plugins[1].read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    new_process(monkeypatch, plugins)
    pipeline.build_report(run, private_config=str(config), log=log.append)
    assert any("toy_genre.py has changed since the run was generated" in line for line in log)
    assert read_json(run / "report.json")["plugins_changed"] == [str(plugins[1].resolve())]
    assert "these plugins have changed since this run was generated" in (run / "report.md").read_text(encoding="utf-8")
    assert cli.main(["report", str(run), "--private-config", str(config)]) == 0
    capsys.readouterr()
    # A config the run did not commit to is refused, and so is a plugin that is gone.
    other = write_config(tmp_path / "other.json", plugins=plugins)
    with pytest.raises(MixError, match="not the private run config that this run committed to"):
        pipeline.score_run(run, private_config=str(other), log=quiet)
    plugins[0].unlink()
    with pytest.raises(MixError, match="which is gone"):
        pipeline.score_run(run, private_config=str(config), log=quiet)


def test_a_run_directory_never_runs_code_that_it_names(registries, tmp_path, monkeypatch, capsys):
    """A run received from someone else may name any file as one of its plugins, by an absolute path or one
    relative to wherever its reader stands: no later step runs it, whatever SHA-256 the manifest gives,
    unless the evaluator passes the private run config that the run committed to."""
    received = pipeline.generate(GenerateConfig(seed=2, instances=1, mix="earthquakes:classic"),
                                 tmp_path / "received", log=quiet)
    pipeline.run_models(received, ["oracle"], log=quiet)
    pipeline.score_run(received, log=quiet)
    marker = tmp_path / "EXECUTED.txt"
    notes = received / "notes.py"
    notes.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n", encoding="utf-8")
    sha = hashlib.sha256(notes.read_bytes()).hexdigest()
    manifest = read_json(received / "manifest.json")
    monkeypatch.chdir(tmp_path)
    for named in (str(notes), "received/notes.py"):
        write_json(received / "manifest.json", {**manifest, "plugins": [{"path": named, "sha256": sha}]})
        for step in (lambda: pipeline.run_models(received, ["oracle"], force=True, log=quiet),
                     lambda: pipeline.score_run(received, log=quiet), lambda: pipeline.build_report(received, log=quiet)):
            with pytest.raises(MixError, match="pass its private run config"):
                step()
        assert cli.main(["report", str(received)]) == 2 and "never runs code" in capsys.readouterr().err
        config = write_config(tmp_path / "private.json", plugins=[notes])
        with pytest.raises(MixError, match="not the private run config that this run committed to"):
            pipeline.build_report(received, private_config=str(config), log=quiet)
    assert not marker.exists()


# ------------------------------------------------------------------------------ the report


def crafted(instances, right) -> dict:
    """The scores of a model that answers a question in a rendering with its key when right(instance,
    rendering) holds, and gives no answer otherwise."""
    rows = []
    for inst in instances:
        for r in inst.renderings:
            for q in inst.questions:
                key = inst.answer_key[q.qid]
                assert key is not None
                given = key if right(inst, r) else None
                rows.append({"instance_id": inst.instance_id, "variant": r.variant, "qid": q.qid,
                             "template_id": q.template_id, "pool": q.pool, "answer_type": q.answer_type,
                             "unit": q.unit, "key": key, "given": given, "normalized": normalize(q.answer_type, given),
                             "correct": given is not None})
    return {"model": "crafted", "kind": "api", "rows": rows, "replies": [], "forecasts": []}


def coverage(items, side, stratum) -> float:
    """The share of `items` in strata that both sides have."""
    sides = defaultdict(set)
    for item in items:
        sides[stratum(*item)].add(side(*item))
    return sum(1 for item in items if len(sides[stratum(*item)]) == 2) / len(items)


def test_accuracy_by_domain_and_by_genre(held_out_run):
    domain_of = {inst.instance_id: inst.domain for inst in held_out_run.instances}
    genre_of = {(inst.instance_id, r.variant): r.genre for inst in held_out_run.instances for r in inst.renderings}
    for model in MODELS:
        m = held_out_run.models[model]
        rows = read_json(held_out_run.run / "scores" / f"{slugify(model)}.json")["rows"]
        for key, name_of, chosen_rows in (("by_domain", lambda r: domain_of[r["instance_id"]],
                                           [r for r in rows if r["variant"] == "v0"]),
                                          ("by_genre", lambda r: genre_of[(r["instance_id"], r["variant"])], rows)):
            tally = Counter((name_of(r), r["correct"]) for r in chosen_rows)
            names = sorted({name_of(r) for r in chosen_rows})
            assert {n: (e["k"], e["n"]) for n, e in m[key].items()} == {
                n: (tally[n, True], tally[n, True] + tally[n, False]) for n in names}, (model, key)
        assert [(n, e["split"]) for n, e in m["by_domain"].items()] == [("earthquakes", PUBLIC), ("toy", HELD_OUT)]
        assert [(n, e["split"]) for n, e in m["by_genre"].items()] == [("classic", PUBLIC), ("toy-telex", HELD_OUT)]


def test_accuracy_by_template_keeps_each_domains_templates_apart(held_out_run, small_run):
    """Every domain has its own P01-P12 and twins, so off the legacy path accuracy by template is keyed by
    domain and template ("toy/N03"): a run of several domains never pools one domain's P01 with another's.
    A legacy report keeps its 0.2.0 keys, the template IDs alone (and is pinned whole elsewhere)."""
    domain_of = {inst.instance_id: inst.domain for inst in held_out_run.instances}
    for model in MODELS:
        rows = [r for r in read_json(held_out_run.run / "scores" / f"{slugify(model)}.json")["rows"]
                if r["variant"] == "v0"]
        tally = Counter((f"{domain_of[r['instance_id']]}/{r['template_id']}", r["correct"]) for r in rows)
        keys = sorted({key for key, _ in tally})
        by_template = held_out_run.models[model]["by_template"]
        assert list(by_template) == keys and all(re.fullmatch(r"(?:earthquakes|toy)/[PN]\d{2}", key) for key in keys)
        assert {key: (e["k"], e["n"]) for key, e in by_template.items()} == {
            key: (tally[key, True], tally[key, True] + tally[key, False]) for key in keys}
    shared = {key.split("/")[1] for key in keys if key.startswith("toy/")} & {
        key.split("/")[1] for key in keys if key.startswith("earthquakes/")}
    assert shared  # template IDs that both domains ask, kept apart
    for model in read_json(small_run / "report.json")["models"]:
        assert model["by_template"] and all(re.fullmatch(r"[PN]\d{2}", key) for key in model["by_template"])


def test_a_held_out_penalty_is_flagged_by_its_own_gap_and_only_by_it(held_out_run):
    models = held_out_run.models
    plain = models["mock"]
    on_domains, on_genres = models[f"mock:held_out_domains={PENALTY}"], models[f"mock:held_out_genres={PENALTY}"]
    assert not plain["source_gap"]["flagged"] and not plain["genre_gap"]["flagged"]
    assert on_domains["source_gap"]["flagged"] and not on_domains["genre_gap"]["flagged"]
    assert on_genres["genre_gap"]["flagged"] and not on_genres["source_gap"]["flagged"]
    assert on_domains["source_gap"]["adjusted"]["public_minus_held_out"] > 0.25
    assert on_genres["genre_gap"]["adjusted"]["public_minus_held_out"] > 0.25
    for key in ("source_gap", "genre_gap"):
        gap = models["oracle"][key]
        assert gap["public_minus_held_out"] == gap["adjusted"]["public_minus_held_out"] == 0.0 and not gap["flagged"]
        assert 0.5 < gap["adjusted"]["coverage"] <= 1.0
    assert models["oracle"]["cross_genre_consistency"]["rate"] == models["naive"]["cross_genre_consistency"]["rate"] == 1.0
    assert plain["cross_genre_consistency"]["rate"] < 1.0  # its mistakes depend on the surface form
    summary = "\n".join(pipeline.summary_lines(held_out_run.run))
    assert "public minus held out" in summary
    assert "overfitting? (sources)" in summary and "overfitting? (genres)" in summary


def test_a_penalty_is_flagged_by_its_own_gap_even_where_the_mix_ties_the_held_out_sets(tied_run):
    """Every toy instance is rendered in the held-out genre alone, so held-out domains come only in held-out
    genres. Each gap is still compared within the other split: a model weak on held-out genres alone shows
    no source gap, and one weak on held-out domains alone no genre gap; each is flagged by its own."""
    inventory = tied_run.inventory
    toy = [inst for inst in tied_run.instances if inst.domain == "toy"]
    assert toy and all(inventory.genre_split(r.genre) == HELD_OUT for inst in toy for r in inst.renderings)
    assert any(inventory.genre_split(inst.renderings[0].genre) == HELD_OUT  # the public domain in either genre
               for inst in tied_run.instances if inst.domain == "earthquakes")
    on_domains, on_genres = (tied_run.models[f"mock:held_out_{what}={PENALTY}"] for what in ("domains", "genres"))
    assert on_domains["source_gap"]["flagged"] and not on_domains["genre_gap"]["flagged"]
    assert on_genres["genre_gap"]["flagged"] and not on_genres["source_gap"]["flagged"]
    for gap in (on_domains["source_gap"], on_genres["genre_gap"]):
        assert gap["adjusted"]["public_minus_held_out"] > 0.25 and 0.0 < gap["adjusted"]["coverage"] < 1.0
    summary = "\n".join(pipeline.summary_lines(tied_run.run))
    assert "overfitting? (sources)\n" in summary + "\n" and "overfitting? (genres)\n" in summary + "\n"
    assert "overfitting? (sources, genres)" not in summary


def test_a_penalised_mock_answers_like_the_plain_mock_outside_its_held_out_sets(held_out_run):
    def reply(model: str, inst, rendering) -> str:
        return read_json(held_out_run.run / "responses" / slugify(model)
                         / f"{inst.instance_id}__{rendering.variant}.json")["text"]

    differ = Counter()
    for inst in held_out_run.instances:
        for r in inst.renderings:
            plain = reply("mock", inst, r)
            for model, held_out in ((f"mock:held_out_domains={PENALTY}", inst.split == HELD_OUT),
                                    (f"mock:held_out_genres={PENALTY}",
                                     held_out_run.inventory.genre_split(r.genre) == HELD_OUT)):
                same = reply(model, inst, r) == plain
                assert same or held_out, (model, inst.instance_id, r.variant)
                differ[model] += not same
    assert all(differ[model] > 10 for model in MODELS[3:])


def test_the_gaps_compare_public_and_held_out_sets_like_for_like(held_out_run):
    instances, inventory = held_out_run.instances, held_out_run.inventory

    def genre_split(r) -> str:
        return inventory.genre_split(r.genre)

    def difficulty(inst, q) -> tuple:
        return answer_family(q), q.features["trap_exposure"], q.features["filters_matter"]

    originals = [(inst, q) for inst in instances for q in inst.questions]
    renderings = [(inst, r, q) for inst in instances for r in inst.renderings for q in inst.questions]
    public = sum(1 for inst, _ in originals if inst.split == PUBLIC)
    assert 0 < public < len(originals)

    # Right exactly on the public domains: a gap of 100 points, raw and in every stratum.
    by_domain = model_metrics(crafted(instances, lambda inst, r: inst.split == PUBLIC), instances, [], inventory)
    gap = by_domain["source_gap"]
    assert (gap["public"]["k"], gap["public"]["n"], gap["held_out"]["k"], gap["held_out"]["n"]) == (
        public, public, 0, len(originals) - public)
    assert gap["public_minus_held_out"] == 1.0 and gap["adjusted"]["public_minus_held_out"] == pytest.approx(1.0)
    assert gap["flagged"] and gap["adjusted"]["ci95"][0] > 0.5
    assert gap["adjusted"]["coverage"] == pytest.approx(coverage(  # strata of originals: also their genre's split
        originals, lambda inst, q: inst.split, lambda inst, q: (*difficulty(inst, q), genre_split(inst.renderings[0]))))
    # Compared within the split of the instance's domain, the genre gap of a model that is right exactly on
    # the public domains is none at all, whatever share of each domain's renderings is in a public genre.
    assert by_domain["genre_gap"]["adjusted"]["public_minus_held_out"] == pytest.approx(0.0)
    assert not by_domain["genre_gap"]["flagged"]
    assert by_domain["genre_gap"]["public_minus_held_out"] == pytest.approx(
        sum(1 for inst, r, q in renderings if genre_split(r) == PUBLIC and inst.split == PUBLIC)
        / sum(1 for inst, r, q in renderings if genre_split(r) == PUBLIC)
        - sum(1 for inst, r, q in renderings if genre_split(r) == HELD_OUT and inst.split == PUBLIC)
        / sum(1 for inst, r, q in renderings if genre_split(r) == HELD_OUT))

    # Right exactly in the public genres: the genre gap is 100 points, and the source gap is the
    # difference between the domains' shares of originals rendered in a public genre.
    by_genre = model_metrics(crafted(instances, lambda inst, r: genre_split(r) == PUBLIC), instances, [], inventory)
    gap = by_genre["genre_gap"]
    in_public = sum(1 for _, r, _ in renderings if genre_split(r) == PUBLIC)
    assert (gap["public"]["k"], gap["public"]["n"], gap["held_out"]["k"], gap["held_out"]["n"]) == (
        in_public, in_public, 0, len(renderings) - in_public)
    assert gap["public_minus_held_out"] == 1.0 and gap["adjusted"]["public_minus_held_out"] == pytest.approx(1.0)
    assert gap["flagged"]
    assert gap["adjusted"]["coverage"] == pytest.approx(coverage(  # strata of renderings: also original or variant,
        renderings, lambda inst, r, q: genre_split(r),        # and the split of the instance's domain
        lambda inst, r, q: (*difficulty(inst, q), r.variant == "v0", inst.split)))

    def share(split: str) -> float:
        chosen = [inst for inst, _ in originals if inst.split == split]
        return sum(1 for inst in chosen if genre_split(inst.renderings[0]) == PUBLIC) / len(chosen)

    assert by_genre["source_gap"]["public_minus_held_out"] == pytest.approx(share(PUBLIC) - share(HELD_OUT))
    # ... but compared within the split of the original's genre, the source gap of that model is none at all.
    assert by_genre["source_gap"]["adjusted"]["public_minus_held_out"] == pytest.approx(0.0)
    assert not by_genre["source_gap"]["flagged"]

    # A model right everywhere, or nowhere, shows no gap at all.
    for right in (True, False):
        flat = model_metrics(crafted(instances, lambda inst, r: right), instances, [], inventory)
        for key in ("source_gap", "genre_gap"):
            assert flat[key]["public_minus_held_out"] == flat[key]["adjusted"]["public_minus_held_out"] == 0.0
            assert not flat[key]["flagged"]


def test_every_twin_pair_stays_in_step_within_each_domain(held_out_run):
    """Off the legacy path a template runs at most one question ahead of its twin from a run's first
    question, within every domain of the run, so the public and new pools of each domain (and so of the
    public and held-out sets) hold the same mix of final operations."""
    counts = Counter((inst.domain, q.template_id) for inst in held_out_run.instances for q in inst.questions)
    assert {domain for domain, _ in counts} == {"earthquakes", "toy"}
    for (domain, template), n in counts.items():
        twin = ("N" if template.startswith("P") else "P") + template[1:]
        assert abs(n - counts[(domain, twin)]) <= 1, (domain, template, counts)


def test_cross_genre_consistency_compares_renderings_in_different_genres(held_out_run):
    instances = held_out_run.instances
    metrics = model_metrics(crafted(instances, lambda inst, r: r.variant != "v2"), instances, [],
                            held_out_run.inventory)
    compared = agreed = 0
    fully = []
    for inst in instances:
        across = [(a, b) for a, b in itertools.combinations(inst.renderings, 2) if a.genre != b.genre]
        for _ in inst.questions:
            outcomes = [(a.variant != "v2") == (b.variant != "v2") for a, b in across]
            compared += len(outcomes)
            agreed += sum(outcomes)
            fully += [all(outcomes)] if outcomes else []
    c = metrics["cross_genre_consistency"]
    assert (c["k"], c["n"]) == (agreed, compared) and compared > 0
    assert (c["questions_fully_consistent"]["k"], c["questions_fully_consistent"]["n"]) == (sum(fully), len(fully))


def test_the_report_records_the_inventory_the_commitment_and_the_gaps(held_out_run):
    run = held_out_run.report["run"]
    commitment = config_commitment(json.loads(held_out_run.config.read_text(encoding="utf-8")))
    assert run["mix"] == "random" and run["private_config_commitment"] == commitment
    per_domain = Counter(inst.domain for inst in held_out_run.instances)
    per_genre = Counter(r.genre for inst in held_out_run.instances for r in inst.renderings)
    assert run["inventory"] == {
        "domains": {"earthquakes": {"split": PUBLIC, "instances": per_domain["earthquakes"]},
                    "toy": {"split": HELD_OUT, "instances": per_domain["toy"]}},
        "genres": {"classic": {"split": PUBLIC, "renderings": per_genre["classic"]},
                   "toy-telex": {"split": HELD_OUT, "renderings": per_genre["toy-telex"]}}}
    assert set(run["snapshots"]) == {"earthquakes", "toy"}
    for inst in held_out_run.instances:  # each instance records its domain and split, each rendering its genre
        stored = read_json(held_out_run.run / "instances" / f"{inst.instance_id}.json")
        assert stored["domain"] == inst.domain and stored["split"] in (PUBLIC, HELD_OUT)
        assert all(r["genre"] in ("classic", "toy-telex") for r in stored["renderings"])
    markdown = (held_out_run.run / "report.md").read_text(encoding="utf-8")
    for text in ("- **Mix:** `random`", f"earthquakes (public, {per_domain['earthquakes']} instances)",
                 f"toy (held out, {per_domain['toy']} instances)", f"toy-telex (held out, {per_genre['toy-telex']} renderings)",
                 f"committed as `{commitment[:24]}…`", "## Cross-genre consistency",
                 "## Public vs held-out domains and genres", "**Sources** (original renderings):",
                 "**Genres** (every rendering):", "## Accuracy by domain and genre", "| model | earthquakes (public) | toy (held out) |",
                 "| model | classic (public) | toy-telex (held out) |", "in the toy-telex genre"):
        assert text in markdown, text
    assert any(commitment[:16] in line for line in held_out_run.log)


# ------------------------------------------------------------------------------------ legacy


def _masked_report(run: Path) -> tuple[str, str]:
    data = read_json(run / "report.json")
    for key in ("seed_commitment", "code_sha256", "evalgen_version"):
        data["run"][key] = None
    text = (run / "report.md").read_text(encoding="utf-8")
    text = re.sub(r"evalgen \S+ \(the reading rules", "evalgen X (the reading rules", text)
    text = re.sub(r"commitment `[0-9a-f]{24}…`", "commitment `X`", text)
    return (hashlib.sha256(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")).hexdigest(),
            hashlib.sha256(text.encode("utf-8")).hexdigest())


def test_legacy_reports_are_exactly_the_reports_of_before_phase_4(small_run):
    assert _masked_report(small_run) == (LEGACY_REPORT_JSON, LEGACY_REPORT_MD)


def test_legacy_runs_and_reports_carry_none_of_the_new_keys(small_run):
    manifest = read_json(small_run / "manifest.json")
    assert not {"inventory", "private_config_commitment", "plugins", "snapshots"} & set(manifest)
    assert not {"mix", "private_config"} & set(manifest["config"]) and "snapshot" in manifest
    assert not any(path.endswith("/mix") or path.startswith("snapshot/earthquakes") for path in manifest["seeds"])
    for path in sorted((small_run / "instances").glob("inst-*.json")):
        stored = read_json(path)
        assert not {"domain", "split", "surface"} & set(stored)
        assert not any("genre" in r for r in stored["renderings"])
    data = read_json(small_run / "report.json")
    assert not {"mix", "inventory", "private_config_commitment", "snapshots"} & set(data["run"])
    for m in data["models"]:
        assert not {"by_domain", "by_genre", "source_gap", "genre_gap", "cross_genre_consistency"} & set(m)
    assert not any("held out" in line for line in pipeline.summary_lines(small_run))


# ------------------------------------------------------------------------------ the mock model


def test_the_mock_draws_as_before_unless_given_a_held_out_penalty(instances):
    plain = make_adapter("mock", seed=99)
    assert isinstance(plain, MockAdapter) and plain.name == "mock"
    assert (plain.held_out_domains, plain.held_out_genres) == (0.0, 0.0)
    both = make_adapter("mock:held_out=0.5", seed=99)
    assert both.name == "mock:held_out=0.5" and (both.held_out_domains, both.held_out_genres) == (0.5, 0.5)
    each = make_adapter("mock:held_out_domains=0.2+held_out_genres=0.1", seed=99)
    assert each.name == "mock:held_out_domains=0.2+held_out_genres=0.1"
    assert (each.held_out_domains, each.held_out_genres) == (0.2, 0.1)
    for spec in ("mock:held_out", "mock:held_out=lots", "mock:held_out=1.5", "mock:held_out=-0.1", "mock:penalty=0.2"):
        with pytest.raises(AdapterError):
            make_adapter(spec, seed=99)
    full = make_adapter("mock:held_out=1", seed=99)
    for inst in instances[:3]:  # legacy instances: in no held-out set
        for r in inst.renderings:
            def task(**splits) -> Task:
                return Task(inst.instance_id, r.variant, question_specs(inst), hidden=inst, **splits)

            reply = plain.complete(r.prompt, task()).text
            assert both.complete(r.prompt, task()).text == full.complete(r.prompt, task()).text == reply
            for split, genre_split in ((PUBLIC, PUBLIC), (HELD_OUT, None), (None, HELD_OUT), ("elsewhere", "?")):
                assert plain.complete(r.prompt, task(split=split, genre_split=genre_split)).text == reply
            assert both.complete(r.prompt, task(split=PUBLIC, genre_split=PUBLIC)).text == reply
            answers = parse_reply(full.complete(r.prompt, task(split=HELD_OUT)).text, [q.qid for q in inst.questions])
            assert not any(is_correct(q.answer_type, inst.answer_key[q.qid], answers.answers.get(q.qid), unit=q.unit)
                           for q in inst.questions)


# -------------------------------------------------------------------------------------- the CLI


def test_the_cli_generates_and_demos_a_mix_with_a_private_config(registries, tmp_path, capsys):
    config = write_config(tmp_path / "private.json")
    out = tmp_path / "generated"
    assert cli.main(["generate", "--seed", "3", "--instances", "3", "--mix", "random", "--private-config", str(config),
                     "--out", str(out)]) == 0
    assert read_json(out / "manifest.json")["private_config_commitment"] == config_commitment(read_json(config))
    demo = tmp_path / "demo"
    assert cli.main(["demo", "--seed", "3", "--instances", "6", "--mix", "earthquakes,toy:toy-telex",
                     "--private-config", str(config), "--out", str(demo)]) == 0
    printed = capsys.readouterr().out
    assert "public minus held out" in printed and "x-genre" in printed
    assert {d for d, _, _ in drawn(demo)} <= {"earthquakes", "toy"}
    assert all(names == ("toy-telex",) * 3 for d, _, names in drawn(demo) if d == "toy")
    for argv, message in ((["--mix", "toy:sonnet", "--private-config", str(config)], "unknown genre 'sonnet'"),
                          (["--private-config", str(config)], "legacy run"),
                          (["--mix", "random", "--source", "usgs"], "unknown source 'usgs' for")):
        assert cli.main(["generate", "--seed", "3", *argv, "--out", str(tmp_path / "refused")]) == 2
        assert message in capsys.readouterr().err
    assert not (tmp_path / "refused").exists()
    assert run_name(GenerateConfig(seed=7)) == "synthetic-seed7"  # a legacy run keeps its 0.2.0 name
    assert run_name(GenerateConfig(seed=7, mix="toy:toy-telex", source="live")) == "toy_toy-telex-live-seed7"


# ------------------------------------------------------------------ live runs of several domains


class Served:
    """A response of a live API, as the requests library gives it."""

    def __init__(self, content: bytes) -> None:
        self.content, self.status_code = content, 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> object:
        return json.loads(self.content)


def live_apis(calls: list[str]):
    """requests.get as the USGS, NWS and JPL CAD APIs would answer it, from test data, recording each URL:
    a fake USGS day feed, synthetic NWS observations per station and the recorded CAD response."""
    usgs, _ = fake_feed()
    served = WEATHER.synthetic(random.Random(8), 600, int(FETCHED.timestamp()), 24)
    stations = {s.id: {"type": "FeatureCollection", "features": [nws_feature(r) for r in served if r.source == s.id]}
                for s in STATIONS}

    def get(url: str, **kwargs) -> Served:
        calls.append(url)
        if url.startswith("https://earthquake.usgs.gov/"):
            return Served(usgs)
        if url.startswith("https://api.weather.gov/stations/"):
            return Served(json.dumps(stations[url.split("/")[-2]]).encode())
        if url.startswith("https://ssd-api.jpl.nasa.gov/cad.api"):
            return Served(CAD_BYTES)
        raise requests.ConnectionError(f"no such test API: {url}")

    return get


def test_a_live_run_of_several_domains_replays_from_its_frozen_snapshots(tmp_path, monkeypatch):
    """A live run of several domains freezes and hashes one snapshot per domain. Its snapshot folder
    (--snapshot-dir) replays every one, with the network blocked and another clock: each domain from its
    own frozen bytes, and invented, which has no live source, from the synthetic records the seed draws.
    So the run regenerates from its seed byte for byte, as plan constraint 3 asks of every run."""
    calls: list[str] = []
    mix = "earthquakes:classic,weather:news,neo:email,invented:chat"
    monkeypatch.setattr(wx, "_utc_now", lambda: FETCHED)
    with monkeypatch.context() as api:
        api.setattr(requests, "get", live_apis(calls))
        live = pipeline.generate(GenerateConfig(seed=18, instances=5, mix=mix, source="live"), tmp_path / "live",
                                 log=quiet)
    assert len(calls) == 2 + len(STATIONS)  # each source fetched once (the NWS once per station)
    manifest = read_json(live / "manifest.json")
    for name in ("earthquakes", "weather", "neo"):
        frozen = live / "snapshot" / name / get_domain(name).live.raw_filename
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == manifest["snapshots"][name]["raw_sha256"]
    assert manifest["snapshots"]["invented"]["source"] == "synthetic"
    assert {inst.domain for inst in load_instances(live)} == {"earthquakes", "weather", "neo", "invented"}

    def unreachable(*args, **kwargs):
        raise requests.ConnectionError("offline: a replay never asks the network")

    monkeypatch.setattr(requests, "get", unreachable)
    monkeypatch.setattr(requests.Session, "send", unreachable)
    monkeypatch.setattr(wx, "_utc_now", lambda: FETCHED.replace(year=2030))  # nor does it read the clock
    replay = pipeline.generate(GenerateConfig(seed=18, instances=5, mix=mix, source="live",
                                              snapshot_dir=str(live / "snapshot")), tmp_path / "replay", log=quiet)
    files, again = (run_files(run) for run in (live, replay))
    assert sorted(files) == sorted(again) and len(files) >= 5 * (1 + 3) + 4 + 3  # instances, prompts, snapshots
    assert [name for name in files if files[name] != again[name]] == []
    a, b = (read_json(run / "manifest.json") for run in (live, replay))
    assert b["config"] == {**a["config"], "snapshot_dir": str(live / "snapshot")}
    assert b["replay"] == {"source_run": str(live.resolve()), "same_code": True}
    for name, snapshot in a["snapshots"].items():  # the same records, from the same bytes; noted as a replay
        assert {k: v for k, v in snapshot.items() if k not in ("note", "fetched_at")} == {
            k: v for k, v in b["snapshots"][name].items() if k not in ("note", "fetched_at")}
    volatile = {"run_id", "seed_commitment", "seed_nonce", "generated_at_utc", "config", "replay", "snapshots"}
    assert {k: v for k, v in a.items() if k not in volatile} == {k: v for k, v in b.items() if k not in volatile}
    # One frozen file replays one domain only; frozen snapshots must hold every domain of the mix.
    with pytest.raises(MixError, match="--snapshot-dir"):
        pipeline.generate(GenerateConfig(seed=18, mix=mix, snapshot_file=str(live / "snapshot" / "neo" / "raw_cad.json")),
                          tmp_path / "refused", log=quiet)
    with pytest.raises(MixError, match="holds no frozen snapshot of neo"):  # one domain's folder: weather's
        pipeline.generate(GenerateConfig(seed=18, mix="weather:news,neo:email",
                                         snapshot_dir=str(live / "snapshot" / "weather")), tmp_path / "refused", log=quiet)
    with pytest.raises(MixError, match="--snapshot-file"):
        pipeline.generate(GenerateConfig(seed=18, snapshot_dir=str(live / "snapshot")), tmp_path / "refused", log=quiet)
    assert not (tmp_path / "refused").exists()


def test_a_frozen_feed_that_a_run_cannot_replay_is_refused_before_anything_is_written(tmp_path, capsys):
    """A frozen feed replays one domain's live source. A domain without a live source, a file that is not
    there, and another source's bytes (which may even parse, to no record at all) are refused before the run
    directory exists: with a MixError (or FileNotFoundError), which the command line reports as an error with
    exit status 2, off the legacy path and on it (its frozen USGS feed), from --snapshot-file and from the
    frozen snapshots of --snapshot-dir."""
    usgs, _ = fake_feed()
    feeds = {"usgs.geojson": usgs, "cad.json": CAD_BYTES, "nws.json": NWS_FIXTURE.read_bytes(), "junk.json": b"<html>"}
    for name, raw in feeds.items():
        (tmp_path / name).write_bytes(raw)
    cases = [
        (["--mix", "invented:classic"], "usgs.geojson", "invented has no live source whose frozen bytes"),
        (["--mix", "weather:news"], "cad.json", "is not a frozen feed of the live source of weather (nws): ValueError: "
                                                "not a bundle of NWS observations"),
        (["--mix", "neo:classic"], "nws.json", "is not a frozen feed of the live source of neo (jpl): ValueError"),
        (["--mix", "earthquakes:news"], "cad.json", "holds no usable record of the live source of earthquakes (usgs)"),
        ([], "nws.json", "holds no usable record of the live source of earthquakes (usgs)"),  # the legacy path
        ([], "junk.json", "is not a frozen feed of the live source of earthquakes (usgs): JSONDecodeError"),
        (["--mix", "neo:classic"], "missing.json", "the frozen feed"),
    ]
    out = tmp_path / "refused"
    for argv, name, message in cases:
        argv = ["generate", "--seed", "3", "--instances", "2", *argv, "--snapshot-file", str(tmp_path / name),
                "--out", str(out)]
        assert cli.main(argv) == 2, argv
        err = capsys.readouterr().err
        assert err.startswith("error: ") and message in err, err
        assert not out.exists()
    with pytest.raises(MixError, match="the frozen feed cannot be replayed: .* is not a frozen feed of the live source "
                                       "of neo"):
        pipeline.generate(GenerateConfig(seed=3, mix="neo:classic", snapshot_file=str(tmp_path / "usgs.geojson")),
                          out, log=quiet)
    with pytest.raises(FileNotFoundError, match="is not a file"):
        pipeline.generate(GenerateConfig(seed=3, mix="neo:classic", snapshot_file=str(tmp_path)), out, log=quiet)
    # Frozen snapshots whose frozen bytes were damaged since: refused before anything is written, too.
    frozen = pipeline.generate(GenerateConfig(seed=3, instances=1, mix="neo:classic",
                                              snapshot_file=str(tmp_path / "cad.json")), tmp_path / "frozen", log=quiet)
    (frozen / "snapshot" / "raw_cad.json").write_bytes(b"<html>busy</html>")
    with pytest.raises(MixError, match="raw_cad.json is not a frozen feed of the live source of neo"):
        pipeline.generate(GenerateConfig(seed=3, instances=1, mix="neo:classic", snapshot_dir=str(frozen / "snapshot")),
                          out, log=quiet)
    assert not out.exists()


def run_files(run: Path) -> dict[str, bytes]:
    """Every file of a run but those that say when and from where it was made (its manifest, and the
    metadata of its snapshots), by path."""
    return {p.relative_to(run).as_posix(): p.read_bytes() for p in sorted(run.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "snapshot.json")}
