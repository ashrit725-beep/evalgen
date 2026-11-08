"""The domain and the genre plan are threaded through generation, prompts, adapters, scoring and
forecast resolution. With the defaults (the legacy path) every call, file and manifest key is the
0.2.0 one; the new options are recorded only when they are set."""
from __future__ import annotations

import copy
import random
import shutil
from dataclasses import asdict, fields, replace

import pytest

from evalgen import pipeline
from evalgen.adapters import MockAdapter, Task
from evalgen.domains.base import Domain, Glossary
from evalgen.domains.earthquakes import EARTHQUAKES, FORECASTER, Earthquakes
from evalgen.genres.base import parse_glossary, render_glossary
from evalgen.genres.classic import CLASSIC
from evalgen.pipeline import GenerateConfig, build_instance, question_specs
from evalgen.prompt import ANSWER_SPECS, INTRO, answer_spec_text, build_prompt
from evalgen.questions import Context, compose_new_templates
from evalgen.report import PACKAGE_DIR, code_fingerprint
from evalgen.rules import READING_RULES
from evalgen.seeds import SeedLog
from evalgen.sources import build_snapshot
from evalgen.util import read_json

from conftest import quiet
from test_usgs_forecast import fake_feed

CONFIG_020 = ["seed", "instances", "source", "feed", "snapshot_file", "variants", "pool_size", "events_min",
              "events_max", "questions_min", "questions_max", "audit_rate"]
SCORE_ROW_020 = ["instance_id", "variant", "qid", "template_id", "pool", "answer_type", "unit", "key", "given",
                 "normalized", "correct"]


class Toy(Domain):
    name = "toy"


# ------------------------------------------------------------------------------ the config


def test_the_manifest_config_keeps_its_020_keys_on_the_default_path(small_run):
    recorded = read_json(small_run / "manifest.json")["config"]
    assert list(recorded) == CONFIG_020
    cfg = GenerateConfig(seed=7, instances=8)
    assert cfg.to_dict() == recorded == {k: v for k, v in asdict(cfg).items() if k in CONFIG_020}


def test_new_options_are_recorded_only_when_set_and_refused_until_phase_4(tmp_path):
    legacy = GenerateConfig(seed=1).to_dict()
    assert GenerateConfig(seed=1, mix="random").to_dict() == {**legacy, "mix": "random"}
    assert GenerateConfig(seed=1, private_config="p.json").to_dict() == {**legacy, "private_config": "p.json"}
    for cfg in (GenerateConfig(seed=1, mix="random"), GenerateConfig(seed=1, private_config="p.json")):
        with pytest.raises(NotImplementedError, match="phase 4"):
            pipeline.generate(cfg, tmp_path / "run", log=quiet)
    assert not (tmp_path / "run").exists()


def test_generation_refuses_a_mix_it_cannot_follow_before_writing_anything(tmp_path):
    for cfg, error in ((GenerateConfig(seed=1, mix="nowhere:classic"), KeyError),
                       (GenerateConfig(seed=1, mix="earthquakes:sonnet"), KeyError),
                       (GenerateConfig(seed=1, mix="earthquakes:classic", feed="hour"), ValueError)):
        with pytest.raises(error):
            pipeline.generate(cfg, tmp_path / "run", log=quiet)
    assert not (tmp_path / "run").exists()
    seedlog = SeedLog(1)
    snapshot, _ = build_snapshot("synthetic", seedlog)
    args = (0, snapshot, compose_new_templates(seedlog.rng("templates", "new")), seedlog, GenerateConfig(seed=1),
            Context(snapshot.compiled_at, snapshot.window_hours), set(), {})
    with pytest.raises(ValueError, match="2 genres for 3 renderings"):
        build_instance(*args, genre_plan=(CLASSIC, CLASSIC))
    with pytest.raises(ValueError, match="needs a genre plan"):
        build_instance(*args, domain=Toy())


def test_the_code_fingerprint_covers_subpackages(tmp_path):
    copy = tmp_path / "evalgen"
    shutil.copytree(PACKAGE_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = code_fingerprint(copy)
    assert before == code_fingerprint()  # relative paths and contents count, not where the package lives
    for name in ("domains/earthquakes.py", "genres/classic.py"):
        source = copy / name
        original = source.read_bytes()
        source.write_bytes(original + b"\n")
        assert code_fingerprint(copy) != before, name
        source.write_bytes(original)
    assert code_fingerprint(copy) == before
    (copy / "genres" / "news.py").write_text("")
    assert code_fingerprint(copy) != before


# ------------------------------------------------------------------------------ the prompt


def test_build_prompt_defaults_to_the_020_wording(instances):
    for inst in instances:
        for r in inst.renderings:
            items = [(q.qid, r.question_text[q.qid], q.answer_type, q.unit) for q in inst.questions]
            assert build_prompt(r.digest, items) == r.prompt
            assert build_prompt(r.digest, items, intro=INTRO, rules=READING_RULES, glossary=None,
                                specs=ANSWER_SPECS) == r.prompt


def test_build_prompt_takes_a_domains_wording_and_glossary():
    glossary = Glossary("item", "items", "item ID", r"it-\d{3}", "depot", "yard", ("checked", "unchecked"), None)
    prompt = build_prompt("DIGEST", [("Q1", "How many?", "count", None), ("Q2", "How deep?", "depth", "mi")],
                          intro="INTRO", rules="RULES\n", glossary=glossary,
                          specs={**ANSWER_SPECS, "count": "a whole number"})
    printed = render_glossary(glossary)  # the one form a prompt's glossary takes: tagged, and read back exactly
    assert printed.startswith("<glossary>\n") and printed.endswith("\n</glossary>\n")
    assert prompt.startswith(f"INTRO\n\nRULES\n\n{printed}\n<digest>\nDIGEST\n</digest>\n\nQuestions:\n"
                             "Q1. How many?\n    Answer type: a whole number.\n"
                             "Q2. How deep?\n    Answer type: a number in miles, rounded to 1 decimal place.\n\n")
    assert parse_glossary(prompt) == glossary
    words = build_prompt("DIGEST", [("Q1", "How warm?", "temperature", "°F"), ("Q2", "Which?", "observation_id", None)],
                         specs={"temperature": "a number in {unit}", "observation_id": "an observation ID"},
                         placeholders={"temperature": "<number>", "observation_id": "\"<observation ID>\""},
                         unit_words={"°F": "degrees Fahrenheit"}, answer_note="IDs are JSON strings.")
    assert "Q1. How warm?\n    Answer type: a number in degrees Fahrenheit.\n" in words
    assert words.endswith("{\"answers\": {\"Q1\": <number>, \"Q2\": \"<observation ID>\"}}\nIDs are JSON strings.\n")
    with pytest.raises(KeyError, match="°F"):  # a unit without words is an error, never an empty word
        answer_spec_text("depth", "°F")


# --------------------------------------------------------------------------- surface words


class Retold(Earthquakes):
    """Earthquakes whose records go by a word drawn for each instance: a domain with surface words."""

    has_surface = True
    WORDS = ("quakes", "tremors", "seismic events", "shocks")

    def draw_surface(self, rng):
        return {"records": rng.choice(self.WORDS)}

    def bind(self, surface):
        if surface is None:
            return self
        bound = copy.copy(self)
        bound.lexicon = replace(self.lexicon, records=surface["records"],
                                all_records=f"{surface['records']} in the digest")
        return bound


def test_surface_words_are_drawn_and_bound_off_the_legacy_path_only():
    def build(domain, genre_plan):
        seedlog = SeedLog(5)
        snapshot, _ = build_snapshot("synthetic", seedlog)
        inst = build_instance(0, snapshot, compose_new_templates(seedlog.rng("templates", "new")), seedlog,
                              GenerateConfig(seed=5), Context(snapshot.compiled_at, snapshot.window_hours), set(), {},
                              domain=domain, genre_plan=genre_plan)
        return inst, seedlog.seeds

    retold = Retold()
    legacy, legacy_seeds = build(retold, None)
    assert legacy.surface is None and "instance/0/surface" not in legacy_seeds
    assert legacy.to_dict() == build(EARTHQUAKES, None)[0].to_dict()  # the legacy path never draws one
    planned, planned_seeds = build(retold, (CLASSIC,) * 3)
    word = random.Random(planned_seeds["instance/0/surface"]).choice(Retold.WORDS)
    assert planned.surface == {"records": word} and planned.to_dict()["surface"] == {"records": word}
    assert all(word in text for r in planned.renderings for text in r.question_text.values())
    check = pipeline.generator_self_check([planned])
    assert check["oracle_agreements"] == check["oracle_checks"] > 0


# ------------------------------------------------------------------------- run and score


def test_tasks_keep_their_positional_fields(instances):
    inst = instances[0]
    task = Task(inst.instance_id, "v0", question_specs(inst), None, inst)
    assert (task.hidden, task.domain, task.genre, task.split) == (inst, None, None, None)
    assert [f.name for f in fields(Task)] == ["instance_id", "variant", "questions", "forecast", "hidden",
                                              "domain", "genre", "split"]


def test_the_mock_answers_any_pool_with_the_draws_of_a_new_template(instances):
    mock = MockAdapter(seed=99)
    for inst in instances[:3]:
        for r in inst.renderings:
            def reply(pool: str) -> str:
                specs = [dict(q, pool=pool) for q in question_specs(inst)]
                return mock.complete(r.prompt, Task(inst.instance_id, r.variant, specs, hidden=inst)).text

            assert reply("held_out") == reply("new")


def test_legacy_score_rows_keep_their_020_keys(small_run):
    for path in sorted((small_run / "scores").glob("*.json")):
        assert all(list(row) == SCORE_ROW_020 for row in read_json(path)["rows"]), path.name


# --------------------------------------------------------------------------------- resolve


def test_forecasts_are_resolved_by_the_forecaster_of_the_domain_that_asked_them(tmp_path, monkeypatch):
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    cfg = GenerateConfig(seed=11, instances=2, source="usgs", snapshot_file=str(feed))
    legacy = pipeline.generate(cfg, tmp_path / "legacy", log=quiet)
    planned = pipeline.generate(replace(cfg, mix="earthquakes:classic"), tmp_path / "planned", log=quiet)
    asked = read_json(legacy / "forecasts.json")
    assert read_json(planned / "forecasts.json") == [dict(f, domain="earthquakes") for f in asked]

    resolved = []

    def resolve(forecasts, now, **kwargs):
        resolved.append([f["instance_id"] for f in forecasts])
        return FORECASTER.resolve(forecasts, now, **kwargs)

    monkeypatch.setattr(Earthquakes, "forecaster", replace(FORECASTER, resolve=resolve))
    after = asked[0]["rule"]["end"] + 3 * 3600
    for run in (legacy, planned):
        summary = pipeline.resolve_run(run, now=after, fetch=lambda rule: [], log=quiet)
        assert (summary["newly_resolved"], summary["pending"]) == (2, 0)
        assert [f["outcome"] for f in read_json(run / "forecasts.json")] == [0, 0]
    assert resolved == [["inst-000", "inst-001"]] * 2
