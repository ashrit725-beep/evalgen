"""The cross matrix of plan phases 2-4: every built-in domain (earthquakes, weather, neo, invented) in
every built-in genre (classic, news, email, chat, table, form, radio).

For each (domain, genre) a small run is generated with the mix "<domain>:<genre>": three instances, each
rendered as its original and two metamorphic variants, which a one-genre mix restyles in that genre.
The run is answered by the oracle and the naive reader, scored and reported. Then, in every cell:

  - every rendering reads back, from its prompt alone (the glossary the prompt prints, and the parser of
    the genre that wrote the digest), to exactly the reports and notices it was rendered from;
  - the oracle scores 100% on every rendering;
  - no question is asked twice in the run;
  - every cell plants re-reports and conflicts, and each instance the traps its plan draws (none or more of
    each kind: traps.draw_plan); every careless reading a trap records gives, from the text of every
    rendering, the wrong answer recorded for it, and holds at the printed precision (the misread digest
    passes the key's guards), and the naive reader is wrong only where a trap explains it;
  - sniffing is exclusive: a genre's digests are claimed by that genre and by no other.

Only classic claims a legacy (0.2.0) digest. Last, a random mix that draws every domain in every genre is
generated a second time, in a new process with another string-hash seed, and comes out byte for byte the
same.

One rendering is not exact by design: earthquakes in the classic genre are the 0.2.0 digest, printed by
the 0.2.0 code byte for byte. It rounds depths to 0.1 km or mi, positions to 0.001° or a hundredth of an
arc-minute and some times down to the minute, and the 0.2.0 parser never reads back place notes,
retraction reasons or the magnitude a correction replaces. The margin guards keep every answer exact
anyway, so that cell is read back to the precision the 0.2.0 digest prints.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from evalgen import domains, genres, pipeline
from evalgen.domains import get_domain
from evalgen.genres import all_genres, detect_genre, get_genre
from evalgen.genres.base import parse_glossary
from evalgen.geo import KM_PER_MILE
from evalgen.mix import Plan, plan_mix
from evalgen.models import Instance, Notice, Report
from evalgen.ops import execute
from evalgen.pipeline import GenerateConfig, question_specs
from evalgen.questions import Context, fingerprint, guards_ok
from evalgen.report import load_instances
from evalgen.rules import Mishandle
from evalgen.scoring import normalize, same_answer
from evalgen.seeds import SeedLog
from evalgen.solver import extract_digest, read_digest, solve
from evalgen.util import read_json

from conftest import ROOT, quiet
from test_usgs_forecast import fake_feed

DOMAINS = ("earthquakes", "weather", "neo", "invented")
GENRES = ("classic", "news", "email", "chat", "table", "form", "radio")
CELLS = tuple(f"{domain}:{genre}" for domain in DOMAINS for genre in GENRES)
SEEDS = {cell: 301 + number for number, cell in enumerate(CELLS)}  # a seed of its own, so cells draw unlike runs
INSTANCES, VARIANTS = 3, 2
MODELS = ("oracle", "naive")
RANDOM_SEED = 812  # a random mix whose first instances draw every domain in every genre (see covering_instances)
VOLATILE = {"seed_commitment", "seed_nonce", "generated_at_utc"}  # differ between two runs of the same seed


@dataclass(frozen=True)
class Cell:
    """The run of one (domain, genre): generated, answered by the reference solvers, scored and reported."""

    domain: str
    genre: str
    run: Path
    instances: list[Instance]
    scores: dict[str, dict]  # model -> its scores/<model>.json
    models: dict[str, dict]  # model -> its entry in report.json


def make_cell(root: Path, name: str) -> Cell:
    """The run of the cell `name` ("<domain>:<genre>") in a folder under `root`."""
    domain, genre = name.split(":")
    cfg = GenerateConfig(seed=SEEDS[name], instances=INSTANCES, variants=VARIANTS, mix=name)
    run = pipeline.generate(cfg, root / f"{domain}-{genre}", log=quiet)
    pipeline.run_models(run, list(MODELS), log=quiet)
    pipeline.score_run(run, log=quiet)
    pipeline.build_report(run, log=quiet)
    return Cell(domain, genre, run, load_instances(run), {m: read_json(run / "scores" / f"{m}.json") for m in MODELS},
                {m["model"]: m for m in read_json(run / "report.json")["models"]})


@pytest.fixture(scope="module")
def cells(tmp_path_factory) -> Callable[[str], Cell]:
    """Each cell's run, made the first time a test asks for it, so that a test of one cell makes one run."""
    root = tmp_path_factory.mktemp("matrix")
    made: dict[str, Cell] = {}

    def cell(name: str) -> Cell:
        if name not in made:
            made[name] = make_cell(root, name)
        return made[name]

    return cell


# ------------------------------------------------------------------------------------ helpers


def printed(inst: Instance) -> tuple[list[Any], list[Notice]]:
    """An instance's reports and notices as every rendering prints them: without the generator's
    bookkeeping (a report's role and trap, a notice's trap), which is never printed."""
    return ([replace(r, role="primary", trap_id=None) for r in inst.reports],
            [replace(n, trap_id=None) for n in inst.notices])


def unmatched(wanted: Sequence[Any], read: Sequence[Any], same: Callable[[Any, Any], bool]) -> tuple[list, list]:
    """The items of `wanted` that no item of `read` matches, and the items of `read` left over, each item
    matching at most one other (repeats count)."""
    rest = list(read)
    missing = []
    for item in wanted:
        match = next((i for i, other in enumerate(rest) if same(item, other)), None)
        if match is None:
            missing.append(item)
        else:
            del rest[match]
    return missing, rest


def printed_by_020(real: Report | Notice, read: Report | Notice) -> bool:
    """Whether `read`, a line of a 0.2.0 earthquake digest as the 0.2.0 parser reads it, is `real` at the
    precision that digest prints: depths to 0.1 km or mi, positions to 0.001° or a hundredth of an
    arc-minute, times to the second or, in some formats, to the minute (truncated). The 0.2.0 parser
    never reads place notes, retraction reasons or the magnitude a correction replaces."""
    if type(real) is not type(read) or not 0 <= real.time - read.time < 60:
        return False
    if isinstance(real, Notice):
        return (real.kind, real.target, real.new_mag, real.mag_type) == (read.kind, read.target, read.new_mag,
                                                                         read.mag_type)
    return ((real.ref, real.network, real.status, real.mag, real.mag_type, real.region, real.same_as)
            == (read.ref, read.network, read.status, read.mag, read.mag_type, read.region, read.same_as)
            and abs(real.lat - read.lat) <= 1e-4 and abs(real.lon - read.lon) <= 1e-4
            and abs(real.depth_km - read.depth_km) <= 0.05 * KM_PER_MILE + 1e-9)


def is_020_digest(cell: Cell) -> bool:
    """Earthquakes in the classic genre: the 0.2.0 digest (see the module docstring)."""
    return (cell.domain, cell.genre) == ("earthquakes", "classic")


def covering_instances(plan: Plan, seed: int, renderings: int, most: int = 60) -> int:
    """How many instances a run of `plan` with `seed` draws before every (domain, genre) of the plan has
    come up (Plan.draw reads each instance's own seed path only, so this draws as the run will)."""
    seedlog, wanted, seen = SeedLog(seed), set(plan.options), set()
    for index in range(most):
        drawn = plan.draw(index, seedlog, renderings)
        seen.update((drawn.domain, genre) for genre in drawn.genres)
        if seen == wanted:
            return index + 1
    raise AssertionError(f"seed {seed} does not draw every domain in every genre within {most} instances: pick "
                         "another RANDOM_SEED")


def run_files(run: Path) -> dict[str, bytes]:
    """Every file of a run but its manifest, by path."""
    return {p.relative_to(run).as_posix(): p.read_bytes() for p in sorted(run.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}


# ----------------------------------------------------------------------------- the matrix


def test_the_matrix_is_every_built_in_domain_and_genre():
    assert sorted(d.name for d in domains.Registry().all()) == sorted(DOMAINS)
    assert sorted(g.name for g in genres.Registry().all()) == sorted(GENRES)


@pytest.mark.parametrize("name", CELLS)
def test_every_rendering_reads_back_to_exactly_its_reports_and_notices(cells, name):
    cell = cells(name)
    manifest = read_json(cell.run / "manifest.json")
    assert manifest["config"]["mix"] == name and manifest["snapshot"]["domain"] == cell.domain
    assert manifest["inventory"] == {"domains": {"public": [cell.domain], "held_out": []},
                                     "genres": {"public": [cell.genre], "held_out": []}}
    domain = get_domain(cell.domain)
    for inst in cell.instances:
        assert (inst.domain, inst.split) == (cell.domain, "public")
        assert [r.genre for r in inst.renderings] == [cell.genre] * (VARIANTS + 1)
        reports, notices = printed(inst)
        for rendering in inst.renderings:
            glossary = parse_glossary(rendering.prompt)  # the prompt's own, as the oracle reads it
            assert glossary == domain.bind(inst.surface).glossary()
            assert extract_digest(rendering.prompt) == rendering.digest
            data = get_genre(cell.genre).parse(rendering.digest, glossary, domain)  # the domain as the run names it
            same = printed_by_020 if is_020_digest(cell) else (lambda a, b: a == b)
            assert unmatched(reports, data.reports, same) == ([], []), (inst.instance_id, rendering.variant)
            assert unmatched(notices, data.notices, same) == ([], []), (inst.instance_id, rendering.variant)
        # The variants are restyled: each one looks unlike the original, and no two are alike.
        assert len({r.digest for r in inst.renderings}) == VARIANTS + 1
        assert all(r.style != inst.renderings[0].style for r in inst.renderings[1:])


@pytest.mark.parametrize("name", CELLS)
def test_the_oracle_scores_100_percent_on_every_rendering(cells, name):
    cell = cells(name)
    check = read_json(cell.run / "manifest.json")["self_check"]  # from the prompt text alone, genre sniffed
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    rows = cell.scores["oracle"]["rows"]
    assert len(rows) == sum(len(inst.questions) * len(inst.renderings) for inst in cell.instances)
    assert all(row["correct"] for row in rows) and {(row["domain"], row["genre"]) for row in rows} == {
        (cell.domain, cell.genre)}
    oracle = cell.models["oracle"]
    assert oracle["accuracy"]["rate"] == oracle["accuracy_all_renderings"]["rate"] == 1.0
    assert oracle["pools"]["public"]["rate"] == oracle["pools"]["new"]["rate"] == 1.0  # P01-P12 and their twins
    assert oracle["consistency"]["rate"] == 1.0


@pytest.mark.parametrize("name", CELLS)
def test_no_question_is_asked_twice(cells, name):
    cell = cells(name)
    asked = [fingerprint(q.chain) for inst in cell.instances for q in inst.questions]
    assert len(asked) == len(set(asked)) >= 3 * INSTANCES
    assert {q.pool for inst in cell.instances for q in inst.questions} == {"public", "new"}  # for the template gap
    for inst in cell.instances:
        for rendering in inst.renderings:
            assert len(set(rendering.question_text.values())) == len(inst.questions)


@pytest.mark.parametrize("name", CELLS)
def test_every_trap_is_planted_and_misread_as_recorded_in_every_rendering(cells, name):
    """Each careless reading a trap records is reproduced from the text of every rendering: the oracle's
    reading of the digest, with that one trap misread, gives exactly the wrong answer recorded for it.
    And it holds at the precision the text prints: the misread digest passes the question's guards, as
    the snapshot does for the key (no record within a threshold's margin, no tie in a ranked or nearest
    answer), so no rounding and no tie-break decides it."""
    cell = cells(name)
    domain = get_domain(cell.domain)
    conflict = "conflicting_magnitude" if cell.domain == "earthquakes" else "conflicting_value"
    # (a digest holds the traps its plan draws, of each kind none or more: traps.draw_plan)
    assert {"duplicate", conflict} <= {t.kind for inst in cell.instances for t in inst.traps}
    for inst in cell.instances:
        assert {t.kind for t in inst.traps} <= {"duplicate", "retraction", conflict}
        bound = domain.bind(inst.surface)
        margin = Context(inst.compiled_at, inst.window_hours).time_margin
        for trap in inst.traps:
            for qid, wrong in trap.affected.items():
                q = inst.question(qid)
                guarded = [execute(q.chain, misread, units=bound.units) for misread in (
                    bound.resolve(inst.reports, inst.notices, Mishandle.from_dict(m)) for m in trap.mishandles)
                    if guards_ok(q.chain, misread, domain=bound, time_margin=margin)]
                assert any(same_answer(q.answer_type, wrong, x, unit=q.unit) for x in guarded), (
                    inst.instance_id, trap.trap_id, qid)
        for rendering in inst.renderings:
            data = read_digest(rendering.prompt, domain=domain)  # the genre that recognises the digest
            units = parse_glossary(rendering.prompt).units
            for trap in inst.traps:
                for qid, wrong in trap.affected.items():
                    q = inst.question(qid)
                    assert not same_answer(q.answer_type, wrong, inst.answer_key[qid], unit=q.unit)
                    readings = [execute(q.chain, data, mishandle=Mishandle.from_dict(m), units=units,
                                        resolver=domain.resolve) for m in trap.mishandles]
                    assert any(same_answer(q.answer_type, wrong, x, unit=q.unit) for x in readings), (
                        inst.instance_id, rendering.variant, trap.trap_id, qid)
                for qid, answers in trap.fallen.items():  # misread with other traps too: never exactly the key
                    q = inst.question(qid)
                    key = normalize(q.answer_type, inst.answer_key[qid])
                    assert answers and all(normalize(q.answer_type, a) != key for a in answers)
    oracle, naive = cell.models["oracle"]["traps"], cell.models["naive"]["traps"]
    assert oracle["caught"]["rate"] == 1.0 and oracle["fell_for_trap"]["k"] == 0
    assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"] > 0  # every miss is a trap's
    trapped = {(inst.instance_id, qid) for inst in cell.instances for t in inst.traps for qid in t.fallen}
    for row in cell.scores["naive"]["rows"]:  # the naive reader is right wherever no trap matters
        assert row["correct"] or (row["instance_id"], row["qid"]) in trapped, row


@pytest.mark.parametrize("name", CELLS)
def test_a_genres_digests_are_claimed_by_that_genre_alone(cells, name):
    cell = cells(name)
    every = all_genres()
    for inst in cell.instances:
        for rendering in inst.renderings:
            assert [g.name for g in every if g.sniff(rendering.digest)] == [cell.genre]
            assert detect_genre(rendering.digest).name == cell.genre


def test_only_classic_claims_a_legacy_digest(instances, tmp_path):
    """The 0.2.0 digests of a legacy run, synthetic or from a frozen USGS feed (place notes, two-decimal
    magnitudes, IDs in each network's sequence), are claimed by classic alone, so the oracle reads every
    one of them as 0.2.0 read it."""
    raw, _ = fake_feed()
    feed = tmp_path / "all_day.geojson"
    feed.write_bytes(raw)
    usgs = pipeline.generate(GenerateConfig(seed=13, instances=3, source="usgs", snapshot_file=str(feed)),
                             tmp_path / "usgs", log=quiet)
    every = all_genres()
    assert len(every) == len(GENRES)
    for inst in [*instances, *load_instances(usgs)]:
        assert inst.domain is None
        for rendering in inst.renderings:
            assert rendering.genre is None and parse_glossary(rendering.prompt) is None
            assert [g.name for g in every if g.sniff(rendering.digest)] == ["classic"]


def test_a_random_mix_of_every_domain_and_genre_repeats_from_its_seed(tmp_path):
    plan = plan_mix("random")
    assert {(d.name, g.name) for d, g in plan.options} == {(d, g) for d in DOMAINS for g in GENRES}
    n = covering_instances(plan, RANDOM_SEED, VARIANTS + 1)
    here = pipeline.generate(GenerateConfig(seed=RANDOM_SEED, instances=n, variants=VARIANTS, mix="random"),
                             tmp_path / "here" / "run", log=quiet)
    # The same run in a new process, whose string hashes (and so its set orders) differ from this one's.
    there = tmp_path / "there" / "run"
    done = subprocess.run([sys.executable, "cli.py", "generate", "--seed", str(RANDOM_SEED), "--instances", str(n),
                           "--variants", str(VARIANTS), "--mix", "random", "--out", str(there)],
                          cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    files, again = run_files(here), run_files(there)
    assert sorted(files) == sorted(again) and [name for name in files if files[name] != again[name]] == []
    manifests = [{k: v for k, v in read_json(run / "manifest.json").items() if k not in VOLATILE}
                 for run in (here, there)]
    assert manifests[0] == manifests[1]
    assert sorted(manifests[0]["snapshots"]) == sorted(DOMAINS)

    instances = load_instances(here)
    drawn = {(inst.domain, r.genre) for inst in instances for r in inst.renderings}
    assert drawn == {(d, g) for d in DOMAINS for g in GENRES}  # every domain in every genre
    used = Counter((inst.domain, q.template_id) for inst in instances for q in inst.questions)
    for (domain, template), n in used.items():  # every twin pair in step within each domain, from the start
        assert abs(n - used[(domain, ("N" if template[0] == "P" else "P") + template[1:])]) <= 1, (domain, template)
    check = manifests[0]["self_check"]
    assert check["oracle_agreements"] == check["oracle_checks"] > 0 and check["problems"] == []
    for inst in instances:
        names = [r.genre for r in inst.renderings]
        assert len(set(names)) == len(names)  # the variants are in genres other than the original's
        specs = question_specs(inst)
        for rendering in inst.renderings:  # each read by the genre that recognises it
            assert detect_genre(rendering.digest).name == rendering.genre
            got = solve(rendering.prompt, specs, domain=inst.domain)
            assert all(same_answer(q.answer_type, got[q.qid], inst.answer_key[q.qid], unit=q.unit)
                       for q in inst.questions)
