"""Adapters: the API ones build correct requests (tested offline with a fake HTTP session)."""
from __future__ import annotations

import copy
import json

import pytest

from evalgen import pipeline
from evalgen.adapters import Adapter, AdapterError, AnthropicAdapter, Completion, OpenAIAdapter, Task, make_adapter
from evalgen.answers import parse_reply
from evalgen.util import read_json

from conftest import quiet


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code, self._payload = status_code, payload or {}
        self.text = text or json.dumps(self._payload)
        self.headers: dict = {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": copy.deepcopy(json)})
        return self.responses.pop(0)


TASK = Task("inst-000", "v0", [])


def test_anthropic_request_and_reply(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    session = FakeSession([FakeResponse(200, {"content": [{"type": "text", "text": '{"answers": {"Q1": 4}}'}],
                                              "model": "claude-test", "stop_reason": "end_turn"})])
    adapter = AnthropicAdapter("claude-test", session=session)
    completion = adapter.complete("PROMPT", TASK)
    assert parse_reply(completion.text).answers == {"Q1": 4}
    call = session.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "test-key" and call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["json"]["model"] == "claude-test" and call["json"]["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert adapter.name == "anthropic:claude-test"


def test_rejected_temperature_is_dropped_and_retried(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    session = FakeSession([
        FakeResponse(400, text='{"error": {"message": "temperature is not supported for this model"}}'),
        FakeResponse(200, {"content": [{"type": "text", "text": "{}"}]}),
    ])
    AnthropicAdapter("claude-test", session=session).complete("PROMPT", TASK)
    assert "temperature" in session.calls[0]["json"] and "temperature" not in session.calls[1]["json"]


def test_openai_request_and_reply(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    session = FakeSession([FakeResponse(200, {"choices": [{"message": {"content": '{"answers": {"Q1": "tie"}}'},
                                                           "finish_reason": "stop"}], "model": "gpt-test"})])
    completion = OpenAIAdapter("gpt-test", session=session).complete("PROMPT", TASK)
    assert parse_reply(completion.text).answers == {"Q1": "tie"}
    call = session.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["json"]["max_completion_tokens"] > 0


def test_http_errors_become_adapter_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = FakeSession([FakeResponse(401, text="bad key")])
    with pytest.raises(AdapterError):
        OpenAIAdapter("gpt-test", session=session).complete("PROMPT", TASK)


def test_missing_keys_and_unknown_specs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for spec in ("anthropic", "openai:gpt-test", "llama"):
        with pytest.raises(AdapterError):
            make_adapter(spec)


def test_a_broken_json_body_is_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("evalgen.adapters.time.sleep", lambda s: None)

    class Broken(FakeResponse):
        def json(self):
            raise ValueError("Expecting value")

    session = FakeSession([Broken(200, text="<html>gateway</html>"),
                           FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})])
    OpenAIAdapter("gpt-test", session=session).complete("PROMPT", TASK)
    assert len(session.calls) == 2


def test_failed_api_calls_are_retried_by_a_plain_rerun(tmp_path, monkeypatch):
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=2), tmp_path / "r", log=quiet)
    calls = {"n": 0}

    class Flaky(Adapter):
        name, kind = "flaky", "api"

        def complete(self, prompt, task):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise AdapterError("HTTP 429: slow down")
            return Completion('{"answers": {}}')

    monkeypatch.setattr(pipeline, "make_adapter", lambda spec, seed=0, max_tokens=0: Flaky())
    messages = []
    pipeline.run_models(run_dir, ["flaky"], log=messages.append)
    assert "3 failed" in messages[-1]
    pipeline.run_models(run_dir, ["flaky"], log=messages.append)
    assert calls["n"] == 6 + 3  # first run: 6 prompts; second run re-asks only the 3 failures
    pipeline.score_run(run_dir, log=quiet)
    replies = read_json(run_dir / "scores" / "flaky.json")["replies"]
    assert all(r["api_error"] is None for r in replies)


def test_replies_cut_off_at_the_token_limit_are_asked_again(tmp_path, monkeypatch):
    """A model that spends its whole output budget thinking returns no answer; that is a harness
    limit, not a wrong answer, so the reply is flagged and re-asked on the next run."""
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), tmp_path / "r", log=quiet)
    calls = {"n": 0}

    class Thinker(Adapter):
        name, kind = "thinker", "api"

        def complete(self, prompt, task):
            calls["n"] += 1
            if calls["n"] <= 3:
                return Completion("", {"stop_reason": "max_tokens"})
            return Completion('{"answers": {}}', {"stop_reason": "end_turn"})

    monkeypatch.setattr(pipeline, "make_adapter", lambda spec, seed=0, max_tokens=0: Thinker())
    messages = []
    pipeline.run_models(run_dir, ["thinker"], log=messages.append)
    assert "3 failed or cut off" in messages[-1]
    pipeline.score_run(run_dir, log=quiet)
    assert sum(r["truncated"] for r in read_json(run_dir / "scores" / "thinker.json")["replies"]) == 3
    pipeline.run_models(run_dir, ["thinker"], log=messages.append)
    assert calls["n"] == 6  # the second run re-asked exactly the 3 cut-off prompts
    pipeline.score_run(run_dir, log=quiet)
    assert not any(r["truncated"] for r in read_json(run_dir / "scores" / "thinker.json")["replies"])


def test_run_skips_models_without_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), tmp_path / "r", log=quiet)
    messages = []
    done = pipeline.run_models(run_dir, ["oracle", "anthropic"], log=messages.append)
    assert done == ["oracle"] and any("skipping anthropic" in m for m in messages)


def test_reference_adapters_reply_in_strict_json(instances):
    inst = instances[0]
    task = Task(inst.instance_id, "v0", pipeline.question_specs(inst))
    for spec in ("oracle", "naive"):
        reply = parse_reply(make_adapter(spec).complete(inst.renderings[0].prompt, task).text)
        assert reply.status == "strict" and set(reply.answers) == {q.qid for q in inst.questions}
