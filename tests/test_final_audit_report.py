"""The final audit's findings on the report: its text, its statistics and the docstrings that describe them.

Each test names the finding it pins. Off the legacy path a report says what the code does; a legacy report
keeps 0.2.0's text byte for byte, flaws included (tests/test_mix.py pins one whole, and
tests/test_regression_020.py the legacy run it comes from), so each text test also checks that the 0.2.0
sentence is still there in a legacy report.
"""
from __future__ import annotations

import json
import math
import random
import shutil
from decimal import Decimal
from types import SimpleNamespace

import pytest

from evalgen import pipeline, seeds
from evalgen.domains import get_domain
from evalgen.geo import KM_PER_MILE
from evalgen.mix import HELD_OUT, PUBLIC
from evalgen.pipeline import GenerateConfig
from evalgen.report import _held_out_section, load_instances, model_metrics, split_gap
from evalgen.scoring import difference_ci, get_answer_type, normalize, stratified_difference
from evalgen.seeds import derive_seed
from evalgen.solver import parse_depth_km
from evalgen.units import decode, encode, exact_canonical, find_unit
from evalgen.util import read_json


def quiet(*_args) -> None:
    pass


@pytest.fixture(scope="module")
def random_run(tmp_path_factory):
    """A random mix with every domain and genre public (no private run config): 4 instances of 3 renderings,
    answered by the reference solvers and the simulated model, scored and reported."""
    run = pipeline.generate(GenerateConfig(seed=7, instances=4, variants=2, mix="random"),
                            tmp_path_factory.mktemp("final-audit-report") / "run", log=quiet)
    pipeline.run_models(run, ["oracle", "naive", "mock"], log=quiet)
    pipeline.score_run(run, log=quiet)
    pipeline.build_report(run, log=quiet)
    return SimpleNamespace(run=run, instances=load_instances(run), markdown=(run / "report.md").read_text("utf-8"))


def legacy_markdown(small_run) -> str:
    return (small_run / "report.md").read_text(encoding="utf-8")


def oracle_rows(run) -> list[dict]:
    return read_json(run / "scores" / "oracle.json")["rows"]


# ------------------------------------------------------------------- sweep-seedlog#2, code-core#5: seeds


def test_the_mock_draws_every_answer_through_a_seedlog(tmp_path, monkeypatch):
    """sweep-seedlog#2: every random draw goes through a SeedLog, the simulated model's too. Its generator per
    prompt is derived by a SeedLog, from its seed and the prompt's SHA-256, so it is the generator it always
    was, and it derives again from what a run records: the manifest's "mock" seed and each reply's
    prompt_sha256."""
    run = pipeline.generate(GenerateConfig(seed=5, instances=2), tmp_path / "run", log=quiet)
    derived, made = set(), []
    seed_of = seeds.SeedLog.seed

    def logged(self, *path):
        value = seed_of(self, *path)
        derived.add(value)
        return value

    class Generator(random.Random):
        def __init__(self, x=None):
            made.append(x)
            super().__init__(x)

    monkeypatch.setattr(seeds.SeedLog, "seed", logged)
    monkeypatch.setattr(random, "Random", Generator)
    pipeline.run_models(run, ["mock"], log=quiet)
    monkeypatch.undo()
    replies = [read_json(path) for path in sorted((run / "responses" / "mock").glob("*.json"))]
    assert len(made) == len(replies) == 6  # one generator per prompt: 2 instances of 3 renderings
    assert [x for x in made if x not in derived] == []  # each seeded by a SeedLog
    mock = read_json(run / "manifest.json")["seeds"]["mock"]
    assert sorted(made) == sorted(derive_seed(mock, "mock", reply["prompt_sha256"]) for reply in replies)


def test_the_seed_line_says_which_seeds_manifest_json_lists(random_run, small_run):
    """code-core#5: the per-prompt seeds of the simulated model are derived at run time and listed nowhere, so
    a report off the legacy path no longer says that every derived seed is listed in manifest.json."""
    assert "; every derived seed is listed in `manifest.json`.\n" in legacy_markdown(small_run)  # 0.2.0's
    assert "every derived seed is listed" not in random_run.markdown
    assert ("; every seed that generation derives is listed in `manifest.json`, and the simulated model (`mock`) "
            "derives one per prompt from the listed `mock` seed and the prompt's SHA-256.\n") in random_run.markdown


# --------------------------------------------------------------------------- sweep-stats#1: an empty pool


def test_a_template_gap_with_an_empty_pool_is_na_off_the_legacy_path(random_run, small_run, tmp_path):
    """sweep-stats#1: with no row in one pool there is no template gap. 0.2.0 reported it as a gap of 0 with the
    interval -100 to +100, which a legacy report keeps; off the legacy path it is n/a, as split_gap gives it."""
    public_only = [r for r in oracle_rows(random_run.run) if r["pool"] == "public"]
    for rows in (public_only, []):
        metrics = model_metrics({"model": "x", "kind": "reference", "rows": rows, "replies": []},
                                random_run.instances, [])
        assert metrics["pools"]["new"]["n"] == 0
        assert metrics["gap"] == {"public_minus_new": None, "ci95": None, "adjusted": None, "flagged": False}
    legacy = model_metrics({"model": "x", "kind": "reference", "rows": [], "replies": []}, load_instances(small_run), [])
    assert legacy["gap"] == {"public_minus_new": 0.0, "ci95": [-1.0, 1.0], "adjusted": None, "flagged": False}

    # The report shows it as n/a, beside pools that are n/a.
    run = tmp_path / random_run.run.name
    shutil.copytree(random_run.run, run)
    score = read_json(run / "scores" / "oracle.json")
    (run / "scores" / "x.json").write_text(json.dumps(dict(score, model="x", rows=[])), encoding="utf-8")
    pipeline.build_report(run, log=quiet)
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert "| x | n/a | n/a | n/a | n/a | — |" in markdown and "-100 to +100" not in markdown


# ---------------------------------------------------------------------- sweep-stats#5: interval bounds


def crafted(rows: list[dict]) -> list[dict]:
    """Original rows right on the public pool and wrong on the new one, but for one row of each pool."""
    original = [dict(r) for r in rows if r["variant"] == "v0"]
    for pool, right in (("public", True), ("new", False)):
        chosen = [r for r in original if r["pool"] == pool]
        assert len(chosen) >= 2
        for i, r in enumerate(chosen):
            r["correct"] = right if i else not right
    return original


def test_gap_intervals_stop_at_100_points_off_the_legacy_path(random_run, small_run):
    """sweep-stats#5: a difference of two proportions lies between -100 and +100 points, but its normal
    approximation, and widening it by a design effect, can pass those bounds. Off the legacy path no gap
    interval does; a legacy report keeps 0.2.0's intervals."""
    for run, instances, legacy in ((random_run.run, random_run.instances, False),
                                   (small_run, load_instances(small_run), True)):
        rows = crafted(oracle_rows(run))
        metrics = model_metrics({"model": "x", "kind": "reference", "rows": rows, "replies": []}, instances, [])
        pools = metrics["pools"]
        gap, lo, hi = difference_ci(pools["public"]["k"], pools["public"]["n"], pools["new"]["k"], pools["new"]["n"])
        assert hi > 1.0  # one wrong answer in each pool: the Wald interval passes +100 points
        assert metrics["gap"]["public_minus_new"] == gap
        assert metrics["gap"]["ci95"] == ([lo, hi] if legacy else [lo, 1.0])
        if not legacy:
            assert -1.0 <= metrics["gap"]["adjusted"]["ci95"][0] < metrics["gap"]["adjusted"]["ci95"][1] <= 1.0

    # The source and genre gaps (split_gap, off the legacy path only): one misread held-out digest beside
    # thirty public instances, the +80 points of DESIGN.md whose adjusted interval reached +111; and ten
    # instances a side of one question each, 9 right against 1, whose raw interval reached about +107.
    def rows(public: list[list[bool]], held_out: list[list[bool]]) -> list[dict]:
        return [{"instance_id": f"{side}-{i:03d}", "side": side, "correct": y}
                for side, answers in ((PUBLIC, public), (HELD_OUT, held_out)) for i, answer in enumerate(answers)
                for y in answer]

    one = split_gap(rows([[q != 0 for q in range(5)]] * 30, [[False] * 5]), lambda row: row["side"], lambda row: ())
    assert one["adjusted"]["public_minus_held_out"] == pytest.approx(0.8) and one["adjusted"]["ci95"][1] == 1.0
    assert 0 < one["adjusted"]["ci95"][0] < 0.8 and not one["flagged"]
    few = split_gap(rows([[True]] * 9 + [[False]], [[True]] + [[False]] * 9), lambda row: row["side"], lambda row: ())
    assert difference_ci(9, 10, 1, 10)[2] > 1.0 and few["ci95"][1] == 1.0 and 0 < few["ci95"][0] < 0.8


def test_the_report_says_when_a_raw_interval_has_no_width(random_run, small_run):
    """sweep-stats#5: the raw gap's normal-approximation interval has zero width where neither side's answers
    vary (the oracle's +0.0 pts (+0 to +0)); the adjusted gap's variance is smoothed, so its interval keeps a
    width. A report off the legacy path says so; a legacy report keeps 0.2.0's section text."""
    assert "| oracle | 100.0% " in random_run.markdown and " pts (+0 to +0) | +0.0 pts (-" in random_run.markdown
    sentence = ("A raw interval has zero width where neither pool's answers vary (each pool all right or all wrong). "
                "The adjusted gap's variance adds half a right and half a wrong answer to each pool of each stratum, "
                "so its interval never does. Intervals stop at ±100 points.")
    assert sentence in random_run.markdown and "zero width" not in legacy_markdown(small_run)
    zero = difference_ci(5, 5, 7, 7)
    assert zero == (0.0, 0.0, 0.0) and stratified_difference([(5, 5, 7, 7)])[1] < 0 < stratified_difference(
        [(5, 5, 7, 7)])[2]


# -------------------------------------------------------------- sweep-stats#3, #4: the adjusted gap


def test_the_adjusted_gap_is_weighted_and_its_variance_smoothed_as_documented():
    """sweep-stats#3, #4: each stratum weighs its share of the rows in the strata that both sides have (they
    sum to 1, whatever the coverage), the estimate uses the strata's own proportions, and the variance
    takes each proportion as (k + 0.5) / (n + 1)."""
    gap, lo, hi, coverage = stratified_difference([(1, 1, 0, 1), (5, 5, 0, 0)])
    assert gap == 1.0 and coverage == pytest.approx(2 / 7)  # not 2/7 of a gap of 1: the second stratum is left out

    strata = [(3, 3, 0, 2), (1, 4, 2, 4)]
    covered = sum(n1 + n2 for _, n1, _, n2 in strata)
    estimate = variance = 0.0
    for k1, n1, k2, n2 in strata:
        w = (n1 + n2) / covered
        estimate += w * (k1 / n1 - k2 / n2)
        a1, a2 = (k1 + 0.5) / (n1 + 1), (k2 + 0.5) / (n2 + 1)
        variance += w * w * (a1 * (1 - a1) / n1 + a2 * (1 - a2) / n2)
    half = 1.96 * math.sqrt(variance)
    assert stratified_difference(strata) == pytest.approx((estimate, estimate - half, estimate + half, 1.0))


# ------------------------------------------------------------- sweep-stats#8: a gap that no model has


GENRE_GAP = {"public": {"k": 9, "n": 10, "rate": 0.9, "ci95": [0.6, 0.98]},
             "held_out": {"k": 8, "n": 10, "rate": 0.8, "ci95": [0.5, 0.94]},
             "public_minus_held_out": 0.1, "ci95": [-0.2, 0.4], "adjusted": None, "flagged": False}


def inventory(domains: dict, genres: dict) -> dict:
    return {"inventory": {"domains": {name: {"split": split, "instances": n} for name, (split, n) in domains.items()},
                          "genres": {name: {"split": split, "renderings": n} for name, (split, n) in genres.items()}}}


def test_a_source_or_genre_gap_that_no_model_has_is_explained():
    """sweep-stats#8: a run whose config holds out a domain of which no instance was drawn has no source gap.
    Its section explained the source gap and then showed only the genres' table; it now says why there is
    none, whichever kind is missing, and it is shown whenever the run holds anything out."""
    drawn_none = inventory({"earthquakes": (PUBLIC, 6), "toy": (HELD_OUT, 0)},
                           {"classic": (PUBLIC, 14), "toy-telex": (HELD_OUT, 10)})
    text = "\n".join(_held_out_section([{"model": "oracle", "source_gap": None, "genre_gap": GENRE_GAP}], drawn_none))
    assert "**Sources:** no source gap, since no instance of the held-out domain toy was drawn." in text
    assert "**Genres** (every rendering):" in text and "**Sources** (" not in text

    none_held = inventory({"earthquakes": (PUBLIC, 6)}, {"classic": (PUBLIC, 14), "toy-telex": (HELD_OUT, 10)})
    text = "\n".join(_held_out_section([{"model": "oracle", "source_gap": None, "genre_gap": GENRE_GAP}], none_held))
    assert "**Sources:** no source gap, since this run holds out no domain." in text

    only_held = inventory({"toy": (HELD_OUT, 6)}, {"classic": (PUBLIC, 0), "toy-telex": (HELD_OUT, 12)})
    text = "\n".join(_held_out_section([{"model": "oracle", "source_gap": None, "genre_gap": None}], only_held))
    assert "**Sources:** no source gap, since no instance of a public domain was drawn." in text
    assert "**Genres:** no genre gap, since no rendering in a public genre was drawn." in text
    assert "## Public vs held-out domains and genres" in text

    # Nothing held out, or a legacy run (no inventory): no section, as before.
    public = inventory({"earthquakes": (PUBLIC, 6)}, {"classic": (PUBLIC, 14)})
    assert _held_out_section([{"model": "oracle", "source_gap": None, "genre_gap": None}], public) == []
    assert _held_out_section([{"model": "oracle"}], {}) == []


# ----------------------------------------------------------------- sweep-stats#2: what "fell" compares


def test_fell_counts_answers_within_the_scoring_tolerance_and_says_so(random_run, small_run):
    """sweep-stats#2: a wrong answer counts as falling for a trap when it equals a fallen answer within the
    scoring tolerance (with the fallen answer as reference), not only when it equals one exactly; the text
    off the legacy path says so, and a legacy report keeps 0.2.0's "gave exactly"."""
    instances = load_instances(small_run)
    inst, trap, q = next((inst, trap, inst.question(qid)) for inst in instances for trap in inst.traps
                         for qid in trap.affected if get_answer_type(inst.question(qid).answer_type).kind == "number")
    fallen = normalize(q.answer_type, trap.fallen.get(q.qid, [trap.affected[q.qid]])[0])
    abs_tol, rel_tol = get_answer_type(q.answer_type).tolerance_in(q.unit)
    near = fallen + 0.5 * max(abs_tol, rel_tol * abs(fallen))
    rows = [{"instance_id": inst.instance_id, "variant": r.variant, "qid": other.qid, "pool": other.pool,
             "template_id": other.template_id, "answer_type": other.answer_type, "correct": False,
             "normalized": near if other.qid == q.qid else None} for r in inst.renderings for other in inst.questions]
    metrics = model_metrics({"model": "x", "kind": "reference", "rows": rows, "replies": []}, [inst], [])
    assert near != fallen and metrics["traps"]["fell_for_trap"]["k"] >= len(inst.renderings)

    assert ("*fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other "
            "traps).") in legacy_markdown(small_run)
    assert ("*fell* = gave an answer equal, within the scoring tolerance, to one that a reader produces by missing "
            "that trap (alone or together with other traps).") in random_run.markdown
    assert "gave exactly" not in random_run.markdown


# ------------------------------------------------------------ sweep-roundtrip#1: the resolution invariant


def test_the_self_check_line_says_what_the_resolution_invariant_checks(random_run, small_run):
    """sweep-roundtrip#1: the invariant is checked on each instance's reports and notices before they are
    rendered (generation stops where it fails), not on the digest text: read from its text, the 0.2.0 digest
    gives back the right records at the precision it prints, never the exact snapshot."""
    assert ("Applying the reading rules to each digest recovered the snapshot exactly for 8/8 instances."
            in legacy_markdown(small_run))
    classic = any(inst.domain == "earthquakes" and r.genre == "classic"
                  for inst in random_run.instances for r in inst.renderings)
    expected = ("Applying the reading rules to each instance's reports and notices, before they are rendered, "
                "recovered the snapshot exactly for 4/4 instances (generation stops where they do not)."
                + (" The classic digest of earthquakes, 0.2.0's, reads them back at the precision it prints."
                   if classic else "") + "\n")
    assert expected in random_run.markdown and "to each digest" not in random_run.markdown


# --------------------------------------------------------------------- finish3#1: the naive reader


def test_the_naive_paragraph_says_its_answers_depend_on_the_parsers_read_back(random_run, small_run):
    """finish3#1: naive's answers also depend on how a genre's parser reads a report back: the 0.2.0 parser of
    the classic earthquake digest keeps a depth printed in miles unrounded (6.3 mi is 10.1388672 km) where
    the other parsers round it to the stored 10.1 km, so it can meet a filter's threshold differently."""
    assert parse_depth_km("depth 6.3 mi") == pytest.approx(6.3 * KM_PER_MILE) and round(6.3 * KM_PER_MILE, 3) == 10.139
    assert ("Its answers depend on which reports a digest holds, on what it prints of them and on how its genre's "
            "parser reads them back") in random_run.markdown
    assert "a filter's threshold differently" in random_run.markdown
    assert "`naive` treats every line as a separate quake and ignores notices." in legacy_markdown(small_run)


# ------------------------------------------------------------------ code-core#3, #4, #6: the units codec


def test_the_examples_of_the_units_docstrings_hold():
    """code-core#6: the half rule decides 0.8 °C in kelvins, which is written 273.95: 274 and 273.9 decode to
    0.8 too, but convert to exactly 0.85 and 0.75. (0.7 °C was the docstrings' example, but 273.9 decodes to
    0.8, so the rule never decides it.) code-core#4: one symbol means different things in different fields:
    km/h is the canonical unit of a wind speed, and 1/3600 of the canonical km/s of a relative velocity.
    code-core#3: the classic earthquake digest is the exception to the exact codec."""
    weather, neo = get_domain("weather"), get_domain("neo")
    kelvin, fahrenheit = (find_unit(weather.field("temp").units, symbol) for symbol in ("K", "°F"))
    assert encode(0.8, 1, kelvin) == "273.95"
    assert exact_canonical("274", kelvin) == Decimal("0.85") and exact_canonical("273.9", kelvin) == Decimal("0.75")
    assert decode("274", 1, kelvin) == decode("273.9", 1, kelvin) == 0.8
    assert encode(0.7, 1, kelvin) == "273.85" and decode("273.9", 1, kelvin) == 0.8
    assert encode(21.0, 1) == "21" and encode(21.0, 1, fahrenheit) == "69.8"

    wind, velocity = weather.field("wind"), neo.field("v_rel")
    assert wind.canonical_unit == "km/h" and find_unit(wind.units, "km/h").scale == 1.0
    assert velocity.canonical_unit == "km/s" and find_unit(velocity.units, "km/h").scale == pytest.approx(1 / 3600)
    assert neo.field("dist").canonical_unit == "km" and find_unit(neo.field("dist").units, "au").scale == 149597870.7

    # The 0.2.0 digest prints a depth of 20.2 km as 12.6 mi, and its parser reads back 20.278 km, not 20.2.
    assert f"{20.2 / KM_PER_MILE:.1f}" == "12.6" and parse_depth_km("depth 12.6 mi") == pytest.approx(20.2777344)
