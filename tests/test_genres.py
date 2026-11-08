"""Genres: classic is the 0.2.0 renderer and parser, draw for draw; genre modules and plugins are
discovered like domains; a reader picks the parser of the genre that wrote a digest by sniffing and
falls back to classic, so every legacy digest is read exactly as 0.2.0 read it; and a genre plan
renders an instance's variants in other genres while the oracle still reads every one of them."""
from __future__ import annotations

import importlib
import pkgutil
import random
import sys

import pytest

from evalgen import genres, pipeline
from evalgen.digest import choose_style, render_digest
from evalgen.domains.base import Domain
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.genres import DEFAULT_GENRE, Registry, all_genres, detect_genre, get_genre
from evalgen.genres.base import Glossary, RenderContext, parse_glossary
from evalgen.genres.classic import CLASSIC, GENRE
from evalgen.pipeline import question_specs
from evalgen.report import load_instances
from evalgen.solver import forecast_estimate, parse_digest, read_digest, solve
from evalgen.util import read_json

from conftest import quiet

# ------------------------------------------------------------------------- classic is 0.2.0


def test_classic_draws_and_renders_exactly_as_020(instances):
    for inst in instances:
        original = inst.renderings[0].style
        for rendering in inst.renderings:
            avoid = None if rendering.variant == "v0" else original
            legacy, wrapped = random.Random(rendering.seed), random.Random(rendering.seed)
            style = choose_style(legacy, avoid=avoid)
            digest = render_digest(inst.reports, inst.notices, inst.chatter, compiled_at=inst.compiled_at,
                                   window_hours=inst.window_hours, style=style, rng=legacy)
            assert CLASSIC.choose_style(wrapped, EARTHQUAKES, avoid=avoid) == style == rendering.style
            ctx = RenderContext(inst.reports, inst.notices, inst.chatter, compiled_at=inst.compiled_at,
                                window_hours=inst.window_hours, style=style, rng=wrapped, domain=EARTHQUAKES)
            assert CLASSIC.render(ctx) == digest == rendering.digest
            assert wrapped.getstate() == legacy.getstate()  # the questions are phrased from the same stream next


def test_classic_parses_exactly_as_020(instances):
    glossary = Glossary.of(EARTHQUAKES)
    assert [(f.key, f.label, [u.symbol for u in f.units]) for f in glossary.fields] == [
        ("mag", "magnitude", []), ("depth", "depth", ["km", "mi"])]
    for inst in instances:
        for rendering in inst.renderings:
            legacy = parse_digest(rendering.digest)
            assert CLASSIC.parse(rendering.digest, None) == legacy  # a legacy prompt prints no glossary
            assert CLASSIC.parse(rendering.digest, glossary) == legacy
            assert read_digest(rendering.prompt) == read_digest(rendering.prompt, "classic") == legacy


def test_classic_leaves_other_domains_to_the_generic_record_path(instances):
    class Toy(Domain):
        name = "toy"

    inst = instances[0]
    rendering = inst.renderings[0]
    ctx = RenderContext(inst.reports, inst.notices, inst.chatter, compiled_at=inst.compiled_at,
                        window_hours=inst.window_hours, style=rendering.style, rng=random.Random(0), domain=Toy())
    glossary = Glossary("toy", "item", "items", "item ID", "depot", "yard")
    for attempt in (lambda: CLASSIC.choose_style(random.Random(0), Toy()), lambda: CLASSIC.render(ctx),
                    lambda: CLASSIC.parse(rendering.digest, glossary)):
        with pytest.raises(NotImplementedError, match="phase 1b"):
            attempt()


# --------------------------------------------------------------------- registry and dispatch


def test_dispatch_falls_back_to_classic_on_every_legacy_digest(instances):
    for inst in instances:
        for rendering in inst.renderings:
            assert parse_glossary(rendering.prompt) is None
            assert CLASSIC.sniff(rendering.digest)
            assert detect_genre(rendering.digest) is CLASSIC


def test_the_evalgen_registry_lists_its_modules_in_order_and_defaults_to_classic():
    names = sorted(m.name for m in pkgutil.iter_modules(genres.__path__)
                   if m.name != "base" and not m.name.startswith("_"))
    assert all_genres() == [importlib.import_module(f"evalgen.genres.{name}").GENRE for name in names]
    assert get_genre() is get_genre(DEFAULT_GENRE) is CLASSIC is GENRE
    with pytest.raises(KeyError, match="unknown genre 'sonnet'"):
        get_genre("sonnet")


TOY_GENRE = """\
from evalgen.genres.classic import Classic


class Toy(Classic):
    name = {name!r}


GENRE = Toy()
"""


@pytest.fixture
def toy_genres(tmp_path, monkeypatch):
    """A package laid out like evalgen.genres, whose module order differs from its genre names."""
    root = tmp_path / "toy_genres"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for module, name in (("zeta", "zulu"), ("alpha", "yankee"), ("old", "classic")):
        (root / f"{module}.py").write_text(TOY_GENRE.format(name=name))
    (root / "base.py").write_text("")  # the interface module, not a genre
    (root / "_shared.py").write_text("")  # private helpers, not a genre
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "toy_genres"
    for module in [m for m in sys.modules if m == "toy_genres" or m.startswith("toy_genres.")]:
        del sys.modules[module]


def test_the_registry_discovers_genre_modules_in_sorted_order(toy_genres):
    registry = Registry(toy_genres)
    assert [g.name for g in registry.all()] == ["yankee", "classic", "zulu"]  # alpha, old, zeta
    assert registry.get() is registry.get("classic")


QUOTED_PLUGIN = '''\
from evalgen.genres.classic import Classic


class Quoted(Classic):
    """Every line of the classic digest quoted with '> ', as in an email reply. Read as classic, it
    loses its notices and its bulletin and wire lines; only this genre's parser recovers it all."""

    name = "quoted"

    def render(self, ctx):
        return "\\n".join("> " + line for line in super().render(ctx).split("\\n"))

    def sniff(self, digest_text):
        return all(line.startswith(">") for line in digest_text.split("\\n"))

    def parse(self, digest_text, glossary):
        return super().parse("\\n".join(line[2:] for line in digest_text.split("\\n")), glossary)


GENRE = Quoted()
'''


@pytest.fixture
def quoted(tmp_path, monkeypatch):
    """A private genre plugin, loaded after the built-in genres of a fresh registry."""
    path = tmp_path / "private" / "quoted.py"
    path.parent.mkdir()
    path.write_text(QUOTED_PLUGIN)
    registry = Registry()
    monkeypatch.setattr(genres, "REGISTRY", registry)
    genre = registry.load_plugin(path)
    assert registry.load_plugin(path) is genre and [g.name for g in all_genres()] == ["classic", "quoted"]
    return genre


@pytest.fixture
def quoted_run(quoted, tmp_path):
    """The first instances of the small run, with their first variant in the quoted genre."""
    run_dir = tmp_path / "quoted-run"
    pipeline.generate(pipeline.GenerateConfig(seed=7, instances=3), run_dir, log=quiet,
                      genre_plan=(CLASSIC, quoted, CLASSIC))
    pipeline.run_models(run_dir, ["oracle", "naive", "mock"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)
    return run_dir


def test_sniffing_picks_the_genre_that_wrote_a_digest(quoted, instances):
    inst = instances[0]

    def context() -> RenderContext:
        return RenderContext(inst.reports, inst.notices, inst.chatter, compiled_at=inst.compiled_at,
                             window_hours=inst.window_hours, style=inst.renderings[0].style, rng=random.Random(5),
                             domain=EARTHQUAKES)

    plain, text = CLASSIC.render(context()), quoted.render(context())
    misread = parse_digest(text)  # the classic patterns still read some quoted report lines, never a notice
    assert misread.notices == [] and misread != parse_digest(plain)
    assert not CLASSIC.sniff(text) and not quoted.sniff(plain)
    assert detect_genre(text) is quoted and detect_genre(plain) is CLASSIC
    assert quoted.parse(text, None) == parse_digest(plain)
    assert read_digest(f"<digest>\n{text}\n</digest>") == parse_digest(plain)


# ------------------------------------------------------------------------------ genre plans


def test_a_genre_plan_renders_variants_in_other_genres(quoted_run, instances):
    for inst, legacy in zip(load_instances(quoted_run), instances):
        assert (inst.domain, inst.split) == ("earthquakes", None)
        assert [r.genre for r in inst.renderings] == ["classic", "quoted", "classic"]
        v0, v1, v2 = inst.renderings
        for planned, before in ((v0, legacy.renderings[0]), (v2, legacy.renderings[2])):
            assert (planned.style, planned.digest, planned.question_text) == (
                before.style, before.digest, before.question_text)
        # A variant in another genre does not avoid the original's style; one in its genre does.
        assert v1.style == choose_style(random.Random(v1.seed))
        assert v2.style == choose_style(random.Random(v2.seed), avoid=v0.style)
        assert all(line.startswith("> ") for line in v1.digest.split("\n"))
        assert detect_genre(v1.digest).name == "quoted"


def test_the_oracle_reads_every_genre_of_a_plan(quoted_run):
    check = read_json(quoted_run / "manifest.json")["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    for inst in load_instances(quoted_run):
        specs = question_specs(inst)
        for rendering in inst.renderings:
            for genre in (rendering.genre, None):  # named, or recognised by sniffing
                assert solve(rendering.prompt, specs, domain=inst.domain, genre=genre) == inst.answer_key
    oracle = read_json(quoted_run / "scores" / "oracle.json")["rows"]
    assert oracle and all(row["correct"] for row in oracle)
    assert {(row["domain"], row["genre"]) for row in oracle} == {("earthquakes", "classic"), ("earthquakes", "quoted")}
    naive = read_json(quoted_run / "scores" / "naive.json")["rows"]
    answers: dict = {}
    for row in naive:  # the naive reader depends only on the content, so every genre gives the same answers
        answers.setdefault((row["instance_id"], row["qid"]), set()).add(repr(row["given"]))
    assert all(len(given) == 1 for given in answers.values())
    models = {m["model"]: m for m in read_json(quoted_run / "report.json")["models"]}
    assert models["oracle"]["accuracy_all_renderings"]["rate"] == 1.0 and models["mock"]["accuracy"]["n"] > 0


def test_naming_the_domain_or_the_genre_reads_legacy_prompts_as_020(instances):
    for inst in instances:
        specs = question_specs(inst)
        event = next(r for r in inst.reports if r.role == "primary")
        rule = {"min_mag": 1.0, "lat": event.lat, "lon": event.lon, "radius_km": 400, "digest_window_hours": 24}
        for rendering in inst.renderings:
            for mode in ("oracle", "naive"):
                legacy = solve(rendering.prompt, specs, mode=mode)
                for domain, genre in (("earthquakes", None), (None, "classic"), ("earthquakes", "classic")):
                    assert solve(rendering.prompt, specs, mode=mode, domain=domain, genre=genre) == legacy
            assert (forecast_estimate(rendering.prompt, rule, domain="earthquakes", genre="classic")
                    == forecast_estimate(rendering.prompt, rule))
