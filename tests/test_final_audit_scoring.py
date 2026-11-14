"""Final audit of reply parsing and scoring: a reply is untrusted input, so reading it never raises and
takes time linear in its length; a hedge of any kind earns no credit; and every reasonable exact format
the prompt allows still scores."""
from __future__ import annotations

import json
import shutil
import time

import pytest

from evalgen import pipeline
from evalgen.adapters import AdapterError, AnthropicAdapter, OpenAIAdapter, Task, token_usage
from evalgen.answers import normalize_qid, parse_reply
from evalgen.scoring import AnswerType, is_correct, normalize, register_answer_type
from evalgen.util import read_json, slugify

from conftest import quiet

TASK = Task("inst-000", "v0", [])
KINDS = ("count", "magnitude", "distance", "event_id", "region", "utc_time", "probability", "neo_approach_id")


def credited(text: str, answer_types: dict, *, unit: str | None = None) -> dict:
    """Which questions of `text` earn credit: answer_types maps each question to (answer type, key)."""
    answers = parse_reply(text, list(answer_types)).answers
    return {qid: is_correct(t, key, answers.get(qid), unit=unit) for qid, (t, key) in answer_types.items()}


# --------------------------------------------------------------------- crashes (untrusted replies)


@pytest.mark.parametrize("answer_type, value", [
    ("count", 10 ** 400), ("count", "9" * 400), ("count", -10 ** 400), ("count", float("inf")), ("count", "1e999"),
    ("magnitude", 2 * 10 ** 308), ("magnitude", "9" * 400), ("depth", "-" + "9" * 400), ("distance", float("nan")),
    ("probability", "9" * 400 + "/" + "9" * 400), ("probability", "9" * 400 + "%"), ("probability", 10 ** 400),
], ids=lambda value: str(value)[:12])
def test_numbers_beyond_float_range_are_no_answer_and_never_overflow(answer_type, value):
    # 0.3.0 raised OverflowError (int too large to convert to float; round of infinity) or read inf
    assert normalize(answer_type, value) is None
    assert not is_correct(answer_type, 7, value) and not is_correct(answer_type, 0.5, value)


def test_large_finite_numbers_still_read():
    assert normalize("magnitude", 10 ** 308) == 1e308 and normalize("count", 10 ** 15) == 10 ** 15


DEGENERATE = [
    "[" * 25000, "{" * 25000, "[" * 25000 + "]" * 25000, '{"a": ' * 3000 + "1" + "}" * 3000,
    '{"answers": {"Q1": ' + "[" * 3000 + "7" + "]" * 3000 + "}}",
    '{"answers": {"Q1": ' + "9" * 5000 + ', "Q2": 3}}', '```json\n{"answers": {"Q1": ' + "9" * 5000 + "}}\n```",
    'Final: {"answers": {"Q1": ' + "9" * 5000 + ", }}", '{"answers": {"Q' + "1" * 5000 + '": 3, "Q2": 4}}',
    "Q" + "1" * 5000 + ": 3\nQ2: 4", '{"answers": [{"id": ["Q1"], "answer": 7}, {"id": "Q2", "answer": 4}]}',
    '{"answers": [{"id": {"q": 1}, "answer": 7}]}', '{"answers": [{"qid": [1], "answer": 7}, {"id": true}]}',
    '{"answers": {"Q1": NaN, "Q2": Infinity, "Q3": -Infinity, "Q4": 1e999}}', '"' + "\\" * 4001 + '"',
    '{"answers": {"Q1": {"value": 7}, "Q2": [7, [8]], "Q3": {"a": {"b": [1, {"c": 2}]}}}}',
]


@pytest.mark.parametrize("text", DEGENERATE, ids=lambda t: f"{t[:30]!r}...{len(t)}")
def test_degenerate_replies_never_raise_and_give_standard_json(text):
    # 0.3.0 raised RecursionError (deep nesting), ValueError (an integer of more than 4300 digits, in a
    # value or a question ID) and TypeError (a list or object as a list-form ID), and wrote NaN/Infinity
    reply = parse_reply(text, ["Q1", "Q2", "Q3", "Q4"])
    assert reply.status in ("strict", "lenient", "failed")
    json.dumps(reply.answers, allow_nan=False)  # what score records is standard JSON, on any Python
    for value in reply.answers.values():
        for answer_type in KINDS:
            normalize(answer_type, value)
            is_correct(answer_type, 7, value)


def test_deep_nesting_reads_the_same_on_every_python():
    # how deep json.loads can recurse depends on the Python version and the stack: 0.3.0 parsed a reply
    # 500 levels deep on 3.14 and raised on 3.9; now nothing nested deeper than any answer object is parsed
    nested = '{"answers": {"Q1": 7, "Q2": ' + "[" * 500 + "1" + "]" * 500 + "}}"
    reply = parse_reply(nested, ["Q1", "Q2"])
    assert reply.answers.get("Q1") == 7 and normalize("count", reply.answers.get("Q2")) is None


@pytest.mark.parametrize("key", [["Q1"], {"q": 1}, None, True, 1.0, 10 ** 5000, "Q" + "1" * 5000, " " * 20000 + "x"],
                         ids=lambda key: type(key).__name__)
def test_question_ids_of_any_shape_are_read_or_refused(key):
    assert normalize_qid(key) is None


def test_question_ids_still_read():
    assert [normalize_qid(k) for k in ("Q1", " q2 ", "3", 4, "F1", "Q 5", "Q007")] == ["Q1", "Q2", "Q3", "Q4", "F1",
                                                                                       "Q5", "Q7"]


@pytest.mark.parametrize("text", [None, 123, b'{"answers": {"Q1": 7}}', ["Q1: 7"], {"answers": {"Q1": 7}}])
def test_a_reply_that_is_not_text_is_no_reply(text):
    assert parse_reply(text).status == "failed"


# ------------------------------------------------------------------ score: one reply never stops a run


@pytest.fixture
def tiny_run(tmp_path):
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=2), tmp_path / "r", log=quiet)
    pipeline.run_models(run_dir, ["oracle"], log=quiet)
    return run_dir


def test_score_grades_every_reply_whatever_it_holds(tiny_run):
    oracle = tiny_run / "responses" / "oracle"
    evil = tiny_run / "responses" / "evil"
    shutil.copytree(oracle, evil)
    texts = iter([
        '{"answers": [{"id": ["Q1"], "answer": 3}]}', '{"answers": {"Q1": ' + "1" + "0" * 400 + "}}",
        '{"answers": {"Q1": "' + "9" * 401 + '", "Q2": "' + "9" * 401 + '"}}', "[" * 100000 + "]" * 100000,
        '{"answers": {"Q1": ' + "9" * 5000 + "}}", '{"answers": {"Q' + "7" * 5000 + '": 1}}',
    ])
    edits = iter([{"model": 5}, {"kind": {"x": 1}}, {"latency_s": "slow"}, {"meta": {"usage": {"input_tokens": "10",
                  "output_tokens": [1]}}}, {"meta": {"usage": "lots"}}, {"latency_s": float("nan"), "error": {"code": 529}}])
    for path in sorted(evil.glob("*.json")):
        record = read_json(path)
        record.update(model="evil", text=next(texts, "Q1: 7"))
        record.update(next(edits, {}))
        path.write_text(json.dumps(record), encoding="utf-8")  # as a hand edit leaves it (NaN included)
    written = pipeline.score_run(tiny_run, log=quiet)  # 0.3.0: TypeError, OverflowError or RecursionError
    assert {p.name for p in written} == {"evil.json", "oracle.json"}  # the model after it is scored too
    score = read_json(tiny_run / "scores" / "evil.json")
    assert score["model"] == "evil" and isinstance(score["kind"], (str, type(None)))
    assert all(r["status"] in ("strict", "lenient", "failed") for r in score["replies"])
    assert all(r["latency_s"] is None or isinstance(r["latency_s"], float) for r in score["replies"])
    assert all(r["input_tokens"] is None or isinstance(r["input_tokens"], int) for r in score["replies"])
    assert [r["api_error"] for r in score["replies"] if r["api_error"]] == ["{'code': 529}"]
    assert not any(row["correct"] for row in score["rows"] if row["variant"] == "v0" and row["qid"] == "Q1"
                   and row["instance_id"] == "inst-000")
    json.dumps(score, allow_nan=False)
    pipeline.build_report(tiny_run, log=quiet)  # and the report reads what score wrote


def test_token_usage_of_any_shape():
    assert token_usage({"meta": {"usage": {"input_tokens": "10", "output_tokens": [1]}}}) == (0, None)
    assert token_usage({"meta": {"usage": "lots"}}) == (None, None)
    assert token_usage({"meta": None}) == (None, None) and token_usage({"meta": [1]}) == (None, None)
    assert token_usage({"meta": {"usage": {"input_tokens": True, "output_tokens": 5}}}) == (0, 5)
    assert token_usage({"meta": {"usage": {"input_tokens": 7, "cache_read_input_tokens": 3, "output_tokens": 5}}}) == (10, 5)
    assert token_usage({"meta": {"usage": {"prompt_tokens": 7, "completion_tokens": 2}}}) == (7, 2)


# ----------------------------------------------------------------------------- hedges: groups


@pytest.mark.parametrize("answer_type, given", [
    ("region", "Nevada or tie"), ("region", "tie or Nevada"), ("region", "Nevada (or a tie)"),
    ("region", "not a tie"), ("region", "no tie"), ("region", "Nevada, not a tie"), ("region", "Nevada (not tied)"),
    ("region", "No tie: Nevada"), ("region", "isn't a tie"), ("neo_body", "Mars or tie"),
    ("invented_group", "Drern or tie"), ("weather_region", "Pacific or tie"), ("region", "tie (not really)"),
    ("region", "Nevada (tie)"), ("region", "Nevada and Utah are not tied"),
])
def test_a_tie_offered_as_an_alternative_or_negated_is_no_tie(answer_type, given):
    # 0.3.0 read every answer containing "tie" or "tied" as a tie, before testing for "or"
    assert normalize(answer_type, given) != "tie"
    assert not is_correct(answer_type, "tie", given)


@pytest.mark.parametrize("given", ["tie", "Tie.", "TIE", "tied", "It's a tie", "it’s a tie", "a tie", "Equal",
                                   "equal counts", "They are equal", "a draw", "the same", "both", "Neither",
                                   "tie (3 each)", "Tie: 2-2", "tied at 3", "**tie**", "Tie!",
                                   "It's a tie between Nevada and Utah", "tie (Nevada 3, Utah 3)",
                                   "Nevada and Utah are tied"])
def test_tie_phrases_are_ties(given):
    assert normalize("region", given) == "tie" and is_correct("region", "tie", given)


@pytest.mark.parametrize("given", ["Nevada", "nevada.", "**Nevada**", "`Nevada`", "Nevada!", "the Nevada region",
                                   '"Nevada"'])
def test_a_group_in_markdown_or_with_an_exclamation_mark_is_that_group(given):
    assert is_correct("region", "Nevada", given)


# ------------------------------------------------------------------------- hedges: bounds


@pytest.mark.parametrize("answer_type, key, given", [
    ("count", 7, "at least 7"), ("count", 7, "more than 7"), ("count", 7, "7+"), ("count", 7, ">7"),
    ("count", 7, "<= 7"), ("count", 7, "up to 7"), ("count", 7, "at most 7"), ("count", 7, "≥ 7"),
    ("count", 7, "no more than 7"), ("count", 7, "over 7"), ("count", 7, "under 7"), ("count", 7, "nearly 7"),
    ("count", 7, "7 at most"), ("count", 7, "7 or more"), ("count", 7, "at least seven"), ("count", 7, "7 ± 1"),
    ("count", 7, "seven or eight"), ("magnitude", 4.3, "4.3+"), ("magnitude", 4.3, ">4.3"),
    ("magnitude", 4.3, "at least 4.3"), ("distance", 5088.8, "less than 5,088.8 km"), ("depth", 5.9, "5.9 ± 0.2"),
    ("event_id", "nc75012345", "nc75012345 or none"), ("event_id", "nc75012345", "nc75012345 (or its duplicate)"),
    ("event_id", "nc75012345", "either nc75012345 or the PT report"),
    ("neo_approach_id", "2026vk2-ca1", "2026vk2-ca1 or 2026vk2-ca2"),
])
def test_bounds_and_alternatives_are_no_answer(answer_type, key, given):
    unit = "km" if answer_type in ("distance", "depth") else None
    assert not is_correct(answer_type, key, given, unit=unit), given


def test_a_bounded_probability_is_no_answer():
    assert normalize("probability", "<5%") is None and normalize("probability", "at most 0.1") is None


@pytest.mark.parametrize("answer_type, key, given", [
    ("count", 7, "7 events"), ("count", 7, "+7"), ("count", 7, "exactly 7"), ("count", 7, "7 (at least one a duplicate)"),
    ("count", 7, "seven events"), ("magnitude", 4.3, "M4.3"), ("distance", 5088.8, "5,088.8 km"),
    ("depth", -1.1, "−1.1 km"), ("event_id", "nc75012345", "event nc75012345."),
    ("event_id", "or12345678", "OR12345678"),
])
def test_plain_answers_with_words_around_them_still_score(answer_type, key, given):
    unit = "km" if answer_type in ("distance", "depth") else None
    assert is_correct(answer_type, key, given, unit=unit), given


# -------------------------------------------------------------------------- hedges: clocks


@pytest.mark.parametrize("key, given, ok", [
    ("21:47", "9:47 PM", True), ("09:47", "9:47 PM", False), ("21:47", "9:47pm UTC", True),
    ("21:47", "9:47 p.m.", True), ("00:05", "12:05 AM", True), ("12:05", "12:05 AM", False),
    ("12:30", "12:30 PM", True), ("00:30", "12:30 PM", False), ("09:47", "9:47 am", True),
    ("21:47", "21:47 AM", False), ("21:47", "21:47 (9:47 PM)", True),
    ("21:47", "21:47 or later", False), ("21:47", "before 21:47", False), ("21:47", "after 21:47", False),
    ("21:47", "21:47 ± 5 min", False), ("21:47", "no later than 21:47", False), ("21:47", "21:47 at the latest", False),
    ("21:47", "by 21:47", False), ("21:47", "21:47 and later", False), ("21:47", "<21:47", False),
    ("21:47", "21:47:00+00:00", True), ("21:47", "2026-01-14T21:47:00+00:00", True), ("21:47", "21:47 +0000", True),
    ("21:47", "21:47 UTC+0", True), ("21:47", "21:47-00:00", True), ("21:47", "21:47+02:00", False),
    ("19:47", "21:47+02:00", False), ("21:47", "21:47 UTC+1", False), ("21:47", "2026-01-14T21:47:00Z", True),
    ("21:47", "21h47", True), ("21:47", "21:47:59", True), ("21:47", "around 21:47", True),
    ("21:47", "21:47 UTC (the corrected report)", True), ("21:47", "21:47 - 21:52", False),
])
def test_clock_answers(key, given, ok):
    # 0.3.0 dropped AM/PM (9:47 PM read as 09:47), credited single-clock hedges and refused an explicit
    # +00:00 offset (read as a second clock, 00:00)
    assert is_correct("utc_time", key, given) is ok, given


# ------------------------------------------------------------------------------- numbers


def test_leading_dot_decimals_keep_their_value_and_sign():
    # 0.3.0 read ".5" and "-.5" as 5
    assert normalize("depth", ".5") == 0.5 and normalize("depth", "-.5") == -0.5 and normalize("depth", "−.5 km") == -0.5
    assert not is_correct("count", 5, ".5") and not is_correct("count", 5, "-.5")
    assert is_correct("depth", 0.5, ".5", unit="km") and is_correct("depth", -0.5, "-.5", unit="km")
    assert normalize("probability", ".25") == 0.25


@pytest.mark.parametrize("given", ["5 088.8", "5'088.8", "5’088.8", "5,088.8", "5 088.8", "5 088.8",
                                   "5 088.8 km"])
def test_thousands_separators(given):
    assert normalize("distance", given) == 5088.8


def test_fifty_fifty_is_a_half():
    assert normalize("probability", "50/50") == 0.5 and normalize("probability", "fifty-fifty") == 0.5
    assert normalize("probability", "1/4") == 0.25


def test_value_objects_are_no_answer_for_every_kind():
    # 0.3.0 read {"value": ...} through str() for IDs and times, and refused it for numbers and groups
    for answer_type, key in (("count", 7), ("event_id", "nc75012345"), ("utc_time", "21:47"), ("region", "Nevada")):
        assert normalize(answer_type, {"value": key}) is None


def test_several_values_are_an_answer_only_when_they_read_the_same():
    assert normalize("count", [7, "7", 7.0]) == 7 and normalize("count", [7, 8]) is None
    assert normalize("utc_time", ["21:47", "21:47:30Z"]) == 21 * 60 + 47 and normalize("count", []) is None
    assert normalize("count", [7, None]) is None and normalize("count", [[7]]) is None


# ------------------------------------------------------------------- reply parsing: JSON objects


FIVE = {"Q1": ("count", 7), "Q2": ("event_id", "nc75012345"), "Q3": ("distance", 5088.8), "Q4": ("utc_time", "21:47"),
        "Q5": ("region", "Nevada")}
FULL = {"Q1": 7, "Q2": "nc75012345", "Q3": 5088.8, "Q4": "21:47", "Q5": "Nevada"}
ALL = {q: True for q in FIVE}


def wrapped(answers: dict) -> str:
    return json.dumps({"answers": answers})


def test_a_question_given_twice_with_different_values_is_no_answer():
    # 0.3.0 kept the last value and called the reply strict
    for text in ('{"answers": {"Q1": 6, "Q1": 7}}', '{"answers": {"Q1": 6, "1": 7}}', '{"answers": {"q1": 7, "Q1": 6}}',
                 '{"answers": [{"id": "Q1", "answer": 6}, {"id": "Q1", "answer": 7}]}',
                 '{"answers": {"Q1": 6}, "answers": {"Q1": 7}}'):
        reply = parse_reply(text, ["Q1"])
        assert reply.status == "lenient" and "Q1" in (reply.error or ""), text
        assert not is_correct("count", 7, reply.answers["Q1"]) and not is_correct("count", 6, reply.answers["Q1"])
    agreeing = parse_reply('{"answers": {"Q1": 7, "1": "7"}}', ["Q1"])  # given twice, but the same answer
    assert agreeing.status == "lenient" and is_correct("count", 7, agreeing.answers["Q1"])


def test_complete_answer_objects_that_disagree_are_no_answer_where_they_disagree():
    hedge = "If the PT report counts: " + wrapped({**FULL, "Q1": 6}) + "\nOtherwise: " + wrapped(FULL)
    assert credited(hedge, FIVE) == {**ALL, "Q1": False}
    restated = wrapped(FULL) + "\n(My first draft was " + wrapped({**FULL, "Q1": 6, "Q5": "tie"}) + ")"
    assert credited(restated, FIVE) == {**ALL, "Q1": False, "Q5": False}  # 0.3.0 graded the draft
    twice = wrapped(FULL) + "\nTo repeat: " + wrapped(FULL)  # complete objects that agree
    assert credited(twice, FIVE) == ALL


def test_one_object_per_question_is_merged_and_partial_notes_beside_a_complete_object_are_ignored():
    per_question = "\n".join(json.dumps({q: v}) for q, v in FULL.items())
    assert credited(per_question, FIVE) == ALL
    note = wrapped(FULL) + '\nNote: had the PT re-report counted, Q1 would be {"Q1": 6}.'
    assert credited(note, FIVE) == ALL
    clash = '{"Q1": 6}\n{"Q1": 7, "Q2": "nc75012345"}\n' + json.dumps({q: FULL[q] for q in ("Q3", "Q4", "Q5")})
    assert credited(clash, FIVE) == {**ALL, "Q1": False}


def test_the_answer_format_quoted_back_is_not_an_answer():
    neo = {"Q1": ("neo_approach_id", "41572-ca1"), "Q2": ("utc_time", "18:42"), "Q3": ("neo_body", "Moon")}
    template = '{"answers": {"Q1": "<approach ID>", "Q2": "HH:MM", "Q3": "<body or tie>"}}'
    answer = json.dumps({"answers": {"Q1": "41572-ca1", "Q2": "18:42", "Q3": "Moon"}})
    assert credited(answer + "\n\nFormat used: " + template, neo) == {q: True for q in neo}
    assert credited("The format is " + template + ".\n" + answer, neo) == {q: True for q in neo}
    lines = "Q1: 41572-ca1\nQ2: 18:42\nQ3: Moon\nIn the requested format: " + template
    assert credited(lines, neo) == {q: True for q in neo}
    legacy = 'Q1: 7\nQ2: nc75012345\n{"answers": {"Q1": <integer>, "Q2": "<event ID>"}}'
    assert parse_reply(legacy, ["Q1", "Q2"]).answers == {"Q1": 7, "Q2": "nc75012345"}
    assert parse_reply(template, ["Q1", "Q2", "Q3"]).status == "failed"


def test_candidate_positions_are_positions_in_the_reply():
    # 0.3.0 compared positions in the raw text with positions in the comment-stripped text, so an unrelated
    # URL (whose "//" starts a comment) changed which of two objects won
    body = 'Draft: {"answers": {"Q1": 1, "Q2": 2}}\nFinal: {"answers": {"Q1": 5, "Q2": 2} // I said "5\n}\n'
    url = "Notes at https://example.org/a/long/page/about/the/digest/and/its/reading/rules\n"
    assert parse_reply(url + body, ["Q1", "Q2"]).answers == parse_reply(body, ["Q1", "Q2"]).answers
    assert parse_reply(url + body, ["Q1", "Q2"]).answers["Q2"] == 2
    commented = url + 'Partial: {"Q1": 1}\nRest: {"Q2": 2, // I said "2\n"Q3": 3}\n'
    assert parse_reply(commented, ["Q1", "Q2", "Q3"]).answers == {"Q1": 1, "Q2": 2, "Q3": 3}


def test_a_top_level_list_and_a_double_encoded_object_are_read():
    items = [{"id": q, "answer": v} for q, v in FULL.items()]
    for text in (json.dumps(items), json.dumps(wrapped(FULL))):
        reply = parse_reply(text, list(FIVE))
        assert reply.status == "lenient" and credited(text, FIVE) == ALL, text


# ---------------------------------------------------------------------- reply parsing: lines


def test_scraped_lines_hand_the_whole_value_to_the_scorer():
    # 0.3.0 kept only the first number or clock of a line, so hedges earned credit and "5,088.8" was 5
    hedges = ["Q1: 7 or 8", "Q1: 7-8", "Q1: 7 (or 8)", "Q1: at least 7", "Q1: 7+"]
    for line in hedges:
        assert credited(line + "\nQ2: nc75012345", {"Q1": ("count", 7), "Q2": ("event_id", "nc75012345")}) == {
            "Q1": False, "Q2": True}, line
    assert not credited("Q4: 21:47 or 21:52", {"Q4": ("utc_time", "21:47")})["Q4"]
    assert not credited("Q3: 5088.8 to 5300", {"Q3": ("distance", 5088.8)}, unit="km")["Q3"]
    assert not credited('Q2: "nc75012345" or "pt75975146"', {"Q2": ("event_id", "nc75012345")})["Q2"]
    for line in ("Q3: 5,088.8", "Q3: 5 088.8 km", "Q3: 5088.8", "Q3: **5,088.8 km**"):
        assert credited(line, {"Q3": ("distance", 5088.8)}, unit="km") == {"Q3": True}, line
    assert credited("Q1: 44,962 mph", {"Q1": ("neo_velocity", 44962.0)}, unit="mph") == {"Q1": True}


def test_scraped_lines_read_every_reasonable_exact_format():
    neo = {"Q1": ("neo_approach_id", "41572-ca1"), "Q2": ("neo_approach_id", "2026vk2-ca1"),
           "Q3": ("neo_approach_id", "18109-ca2")}
    assert credited("Q1: 41572-ca1\nQ2: 2026vk2-ca1\n**Q3:** 18109-ca2", neo) == {q: True for q in neo}
    assert credited('{"answers": {"Q1": 41572-ca1, "Q2": 2026vk2-ca1, "Q3": 18109-ca2}}', neo) == {q: True for q in neo}
    mixed = {"Q1": ("depth", -1.1), "Q2": ("weather_temperature", -3.2), "Q3": ("region", "Nevada"),
             "Q4": ("utc_time", "21:47"), "Q5": ("count", 7)}
    text = "Q1: −1.1 km\nQ2: −3.2 °C\nQ3: **Nevada**\nQ4: 2026-01-14T21:47Z\nq5: 7"
    assert credited(text, mixed, unit="km") == {q: True for q in mixed}
    assert normalize("probability", parse_reply("Q1: .25").answers["Q1"]) == 0.25
    assert parse_reply("Q1: 1e3\nQ2: 2,217,903 km").answers == {"Q1": 1000.0, "Q2": "2,217,903 km"}
    assert parse_reply("q1: 7\nf1: 0.3").answers == {"Q1": 7, "F1": 0.3}
    assert parse_reply("Q1 => 7\nQ2 = 4.5\nQ3: `nc75012345`").answers == {"Q1": 7, "Q2": 4.5, "Q3": "nc75012345"}


def test_a_value_cut_off_at_the_end_of_the_reply_is_dropped():
    # 0.3.0 let "22", cut off by the token limit, replace the complete 2217903 stated earlier
    cut = "Q1: 12\nQ2: 2217903\nFinal answer:\n" + '{"answers": {"Q1": 12, "Q2": 22'
    assert parse_reply(cut, ["Q1", "Q2"]).answers == {"Q1": 12, "Q2": 2217903}
    assert parse_reply('{"answers": {"Q1": 7, "Q2": 3.4, "Q3": 50', ["Q1", "Q2", "Q3"]).answers == {"Q1": 7, "Q2": 3.4}
    assert parse_reply("Q1: 7\nQ2: 50", ["Q1", "Q2"]).answers == {"Q1": 7, "Q2": 50}  # a reply that just ends


# ----------------------------------------------------------------------------- linear time


@pytest.mark.parametrize("text", [
    "*" * 200000, "Q1: a" + " " * 200000 + "x", "Q1: a" + "\t" * 200000 + "x", '{"answers": {"' + " " * 200000 + 'x": 1}}',
    "```" + " " * 200000 + "x", "Q1: " * 50000, "Q1: “" * 50000, "Q1: \"x Q1: 'y " * 20000, "{}" * 100000,
    '{"answers": {"Q1": "' + "a" * 200000, "/*" * 100000, "//" * 100000 + "\n{}", '{"Q1": 1}' * 20000,
    "Q1: " + "1 " * 100000, "Q1: " + "21:47 " * 40000, "Q1:" + "*" * 200000 + "7",
])
def test_reading_a_reply_takes_linear_time(text):
    # 0.3.0 took 45 s for 64,000 asterisks, and quadratic time in runs of spaces after "Qn: <letter>"
    started = time.monotonic()
    reply = parse_reply(text, ["Q1", "Q2"])
    for value in list(reply.answers.values())[:3]:
        for answer_type in KINDS:
            normalize(answer_type, value)
    assert time.monotonic() - started < 3.0


@pytest.mark.parametrize("value", [" " * 200000 + "x", "1" * 200000, "1 " * 100000, "21:47 " * 40000,
                                   "a" * 100000 + " or b", "(" * 100000 + "tie", "tie " * 50000, "at " * 60000 + "7",
                                   "nc" * 100000, "1-" * 100000, "9:47 " + "p" * 100000, "+" * 100000 + "00:00"])
def test_normalizing_an_answer_takes_linear_time(value):
    started = time.monotonic()
    for answer_type in KINDS:
        normalize(answer_type, value)
    assert time.monotonic() - started < 3.0


# ------------------------------------------------------------------------- API response bodies


class Body:
    def __init__(self, payload):
        self.status_code, self.payload, self.text, self.headers = 200, payload, json.dumps(payload), {}

    def json(self):
        return self.payload


class Session:
    def __init__(self, payload):
        self.payload = payload

    def post(self, url, headers=None, json=None, timeout=None):
        return Body(self.payload)


@pytest.mark.parametrize("payload", [None, [], "text", {"content": "abc"}, {"content": None},
                                     {"content": [{"type": "text", "text": None}]}, {"content": ["Q1: 7"]},
                                     {"content": [None]}, {"content": {"type": "text", "text": "x"}}])
def test_a_malformed_anthropic_body_is_an_adapter_error(monkeypatch, payload):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(AdapterError, match="unexpected response body"):
        AnthropicAdapter("claude-test", session=Session(payload)).complete("PROMPT", TASK)


@pytest.mark.parametrize("payload", [None, [], "text", {"choices": [None]}, {"choices": [{"message": "hi"}]},
                                     {"choices": "x"}, {"choices": []}, {"choices": [{"message": {"content": 7}}]},
                                     {"choices": [{"message": {"content": [{"type": "text", "text": None}]}}]}])
def test_a_malformed_openai_body_is_an_adapter_error(monkeypatch, payload):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(AdapterError, match="unexpected response body"):
        OpenAIAdapter("gpt-test", session=Session(payload)).complete("PROMPT", TASK)


def test_text_blocks_and_content_parts_are_joined_line_by_line(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    blocks = {"content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Q1: 7"},
                          {"type": "text", "text": "Q2: nc75012345"}], "stop_reason": "end_turn"}
    text = AnthropicAdapter("claude-test", session=Session(blocks)).complete("PROMPT", TASK).text
    assert parse_reply(text, ["Q1", "Q2"]).answers == {"Q1": 7, "Q2": "nc75012345"}  # 0.3.0: Q2 lost
    parts = {"choices": [{"message": {"content": [{"type": "text", "text": "Q1: 7"}, {"type": "text", "text": "Q2: 4"}]},
                          "finish_reason": "stop"}]}
    text = OpenAIAdapter("gpt-test", session=Session(parts)).complete("PROMPT", TASK).text
    assert isinstance(text, str) and parse_reply(text, ["Q1", "Q2"]).answers == {"Q1": 7, "Q2": 4}
    empty = {"choices": [{"message": {"content": None, "refusal": "no"}, "finish_reason": "stop"}]}
    assert OpenAIAdapter("gpt-test", session=Session(empty)).complete("PROMPT", TASK).text == ""


def test_run_records_a_malformed_body_as_a_failed_call(tmp_path, monkeypatch):
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=4, instances=1), tmp_path / "r", log=quiet)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    adapter = OpenAIAdapter("gpt-test", session=Session({"choices": [None]}))
    monkeypatch.setattr(pipeline, "make_adapter", lambda spec, seed=0, max_tokens=0: adapter)
    messages = []
    pipeline.run_models(run_dir, ["openai:gpt-test"], log=messages.append)  # 0.3.0: AttributeError
    assert "3 failed" in messages[-1]
    records = [read_json(p) for p in (run_dir / "responses" / slugify(adapter.name)).glob("*.json")]
    assert records and all(r["error"].startswith("AdapterError: unexpected response body") for r in records)


# ------------------------------------------------------------------------------------- IDs


@pytest.mark.parametrize("given, ok", [("WX-482137", True), ("Observation WX-482137.", True), ("WX-4821370", False),
                                       ("WX-482137a", False), ("NWX-482137", False), ("xWX-482137", False),
                                       ("WX-4821375", False)])
def test_ids_are_matched_whole(given, ok):
    # 0.3.0 found "wx-482137" inside "WX-4821370"
    assert is_correct("weather_observation_id", "WX-482137", given) is ok


def test_an_id_pattern_that_cannot_be_wrapped_is_matched_as_it_is(answer_registry):
    # a global inline flag must start a pattern (Python 3.11+ refuses one inside a group)
    register_answer_type(AnswerType("flagged_id", "id", id_pattern=r"(?i)ab-\d{3}"))
    assert normalize("flagged_id", "Record AB-123.") == "ab-123"
