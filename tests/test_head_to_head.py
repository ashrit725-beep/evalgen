"""Head to head: API models answered the same questions, so the report compares them paired."""
from __future__ import annotations

import math

from evalgen import pipeline
from evalgen.adapters import Adapter, Completion, OracleAdapter, make_adapter
from evalgen.report import load_instances
from evalgen.scoring import mcnemar_exact, paired_difference_ci
from evalgen.util import read_json

from conftest import quiet


def test_mcnemar_exact_matches_hand_computed_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert math.isclose(mcnemar_exact(5, 0), 2 / 32)  # two-sided: 2 * (1/2)^5
    assert math.isclose(mcnemar_exact(0, 5), 2 / 32)
    assert math.isclose(mcnemar_exact(4, 1), 2 * (1 + 5) / 32)  # 2 * P(X <= 1), X ~ Bin(5, 1/2)


def test_paired_difference_ci_brackets_the_difference():
    d, lo, hi = paired_difference_ci(both=90, a_only=8, b_only=2, neither=0)
    assert math.isclose(d, 0.06) and lo < d < hi and -1 <= lo and hi <= 1
    d, lo, hi = paired_difference_ci(both=50, a_only=0, b_only=0, neither=50)
    assert d == 0 and lo < 0 < hi  # no disagreements still leaves some uncertainty


class Fake(Adapter):
    """An API-kind model for tests: always right (the oracle's answers), never answering, or right
    only on the original rendering (right="v0")."""

    kind, concurrent = "api", True

    def __init__(self, name: str, right, usage: dict):
        self.name, self.right, self.usage = name, right, usage

    def complete(self, prompt, task):
        right = self.right is True or (self.right == "v0" and task.variant == "v0")
        text = OracleAdapter().complete(prompt, task).text if right else '{"answers": {}}'
        return Completion(text, {"stop_reason": "end_turn", "usage": dict(self.usage)})


def test_report_compares_api_models_head_to_head(tmp_path, monkeypatch):
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=2), tmp_path / "r", log=quiet)
    fakes = {"good": Fake("fake:good", True, {"input_tokens": 1000, "output_tokens": 200}),
             "bad": Fake("fake:bad", False, {"input_tokens": 900, "output_tokens": 50})}
    monkeypatch.setattr(pipeline, "make_adapter",
                        lambda spec, seed=0, max_tokens=0: fakes[spec] if spec in fakes else make_adapter(spec))
    pipeline.run_models(run_dir, ["oracle", "good", "bad"], log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)

    data = read_json(run_dir / "report.json")
    questions = sum(len(inst.questions) for inst in load_instances(run_dir))
    (pair,) = data["head_to_head"]  # only API models are paired; the oracle is a reference solver
    assert (pair["a"], pair["b"]) == ("fake:good", "fake:bad")  # the more accurate model is A
    assert pair["questions"] == questions
    assert (pair["both_right"], pair["a_only"], pair["b_only"], pair["neither"]) == (0, questions, 0, 0)
    assert math.isclose(pair["difference"], 1.0)
    assert pair["mcnemar_p"] == mcnemar_exact(questions, 0) < 0.05
    assert pair["all_renderings"]["a_only"] == questions

    usage = {m["model"]: m["usage"] for m in data["models"]}
    prompts = 2 * 3  # two instances, three renderings each
    assert usage["fake:good"]["input_tokens"] == 1000 * prompts and usage["fake:good"]["output_tokens"] == 200 * prompts
    assert usage["fake:bad"]["median_output_tokens"] == 50 and usage["oracle"] is None

    text = (run_dir / "report.md").read_text()
    assert "## Head to head" in text and "## API usage" in text


def test_every_rendering_row_rewards_robustness(tmp_path, monkeypatch):
    """Two models that tie on the original renderings are separated once one fails the variants."""
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=2), tmp_path / "r", log=quiet)
    usage = {"input_tokens": 1, "output_tokens": 1}
    fakes = {"brittle": Fake("fake:brittle", "v0", usage), "steady": Fake("fake:steady", True, usage)}
    monkeypatch.setattr(pipeline, "make_adapter", lambda spec, seed=0, max_tokens=0: fakes[spec])
    pipeline.run_models(run_dir, list(fakes), log=quiet)
    pipeline.score_run(run_dir, log=quiet)
    pipeline.build_report(run_dir, log=quiet)

    (pair,) = read_json(run_dir / "report.json")["head_to_head"]
    assert pair["a"] == "fake:steady"  # tied on the originals, so robustness decides the order
    assert (pair["a_only"], pair["b_only"], pair["mcnemar_p"]) == (0, 0, 1.0)
    every = pair["all_renderings"]
    assert (every["questions"], every["a_only"], every["b_only"]) == (pair["questions"], pair["questions"], 0)
    assert "every rendering" in (run_dir / "report.md").read_text()


def test_no_head_to_head_without_two_api_models(small_run):
    data = read_json(small_run / "report.json")
    assert data["head_to_head"] == []
    assert "## Head to head" not in (small_run / "report.md").read_text()
