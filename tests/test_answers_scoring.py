"""Strict-JSON answers are parsed robustly, and scoring applies the published tolerances."""
from __future__ import annotations

import pytest

from evalgen.answers import parse_reply
from evalgen.scoring import difference_ci, equivalent, is_correct, normalize, wilson


def test_strict_json():
    reply = parse_reply('{"answers": {"Q1": 3, "Q2": "nc75012345"}}')
    assert reply.status == "strict" and reply.answers == {"Q1": 3, "Q2": "nc75012345"}
    reply = parse_reply('{"answers": [{"id": "Q1", "answer": 9}]}')  # list form is valid JSON too
    assert reply.status == "strict" and reply.answers == {"Q1": 9}


def test_negative_depths_keep_their_sign_even_with_a_typographic_minus():
    # USGS depths can be negative; "−" (U+2212) must not be dropped, or -1.1 would score as 1.1
    for given in (-1.1, "-1.1", "-1.1 km", "−1.1 km"):
        assert normalize("depth", given) == -1.1 and is_correct("depth", -1.1, given), given
    assert not is_correct("depth", -1.1, "1.1 km")
    assert normalize("depth", "3−4 km") is None  # a range written with minus signs is still a hedge


@pytest.mark.parametrize("text, expected", [
    ('Sure! Here you go:\n```json\n{"answers": {"Q1": 3}}\n```\nHope that helps.', {"Q1": 3}),
    ('{"answers": {"Q1": "7", "Q2": "4.6",}}', {"Q1": "7", "Q2": "4.6"}),
    ("{'answers': {'Q1': 2, 'Q2': None}}", {"Q1": 2, "Q2": None}),
    ('Reasoning... {"draft": 1} ... final: {"answers": {"q1": 5, "2": "tie"}}', {"Q1": 5, "Q2": "tie"}),
    ("“answers” is below\n{“answers”: {“Q1”: 4}}", {"Q1": 4}),
])
def test_lenient_recovery(text, expected):
    reply = parse_reply(text)
    assert reply.status == "lenient" and reply.answers == expected


def test_truncated_reply_is_salvaged_by_line_scraping():
    reply = parse_reply('{"answers": {"Q1": 7, "Q2": 3.4, "Q3": "nc7')
    assert reply.status == "lenient" and reply.answers == {"Q1": 7, "Q2": 3.4}


def test_the_most_complete_final_answer_wins():
    expected = ["Q1", "Q2", "Q3"]
    draft_then_commented_final = ('Draft: {"answers": {"Q1": 1, "Q2": 2, "Q3": 3}}\n'
                                  'Final:\n{"answers": {"Q1": 4, /* re-checked */ "Q2": 5, "Q3": 6}}')
    assert parse_reply(draft_then_commented_final, expected).answers == {"Q1": 4, "Q2": 5, "Q3": 6}
    full_then_stray_note = '{"answers": {"Q1": 4, "Q2": 5, "Q3": 6}}\nNote: earlier I wrote {"Q1": 5}.'
    assert parse_reply(full_then_stray_note, expected).answers == {"Q1": 4, "Q2": 5, "Q3": 6}


def test_line_scraping_keeps_clock_times_regions_and_the_last_mention():
    text = "Reasoning: Q1: 3 at first...\n**Q1:** 4\nQ3: Honshu, Japan\nQ4: 21:47"
    assert parse_reply(text).answers == {"Q1": 4, "Q3": "Honshu, Japan", "Q4": "21:47"}
    assert parse_reply('{"answers": {"Q1": 4, "Q4": 21:47}}').answers == {"Q1": 4, "Q4": "21:47"}
    assert parse_reply("**Q1:** **4**").answers == {"Q1": 4}


def test_several_partial_objects_and_cut_off_replies_keep_every_stated_answer():
    expected = ["Q1", "Q2", "Q3"]
    per_question = '{"Q1": 4}\n{"Q2": "nn57149321"}\n{"Q3": "tie"}'
    assert parse_reply(per_question, expected).answers == {"Q1": 4, "Q2": "nn57149321", "Q3": "tie"}
    cut_off = 'Q1: 4\nQ2: nn57149321\nQ3: tie\n\nFinal answer:\n{"answers": {"Q1": 4, "Q2": "nn5714'
    assert parse_reply(cut_off, expected).answers == {"Q1": 4, "Q2": "nn57149321", "Q3": "tie"}
    example_then_answer = ('I will use the requested shape, e.g. {"answers": {"Q1": 0}}.\n'
                           'Final: {"Q1": 4, "Q2": "nc75012345", "Q3": "tie"}')
    assert parse_reply(example_then_answer, expected).answers == {"Q1": 4, "Q2": "nc75012345", "Q3": "tie"}
    complete_then_note = ('{"Q1": 4, "Q2": "nn57149321", "Q3": "tie"}\n'
                          'Note: had the PT re-report been counted, the reply would be {"answers": {"Q1": 5}}.')
    assert parse_reply(complete_then_note, expected).answers == {"Q1": 4, "Q2": "nn57149321", "Q3": "tie"}


@pytest.mark.parametrize("text", ["", "   ", "I cannot answer that.", "{not json at all"])
def test_unusable_replies_fail(text):
    assert parse_reply(text).status == "failed"


@pytest.mark.parametrize("answer_type, key, given, ok", [
    ("count", 7, 7, True), ("count", 7, "7 events", True), ("count", 7, 7.0, True), ("count", 7, "seven", True),
    ("count", 7, 7.5, False), ("count", 7, 8, False), ("count", 7, None, False),
    ("event_id", "nc75012345", "NC75012345", True), ("event_id", "nc75012345", "event nc75012345.", True),
    ("event_id", "nc75012345", "nc75012346", False),
    ("magnitude", 4.3, 4.35, True), ("magnitude", 4.3, 4.36, False), ("magnitude", 4.3, "M4.3", True),
    ("depth", 18.6, 18.9, True), ("depth", 18.6, 19.0, False), ("depth", 100.0, 101.9, True),
    ("distance", 100.0, 103.0, True), ("distance", 100.0, 104.0, False), ("distance", 10.0, 11.5, True),
    ("distance", 10.0, 11.6, False),
    ("region", "Nevada", "nevada", True), ("region", "Honshu, Japan", " honshu,  japan ", True),
    ("region", "tie", "Equal", True), ("region", "Nevada", "Utah", False),
    ("utc_time", "21:47", "21:48", True), ("utc_time", "21:47", "21:49", False),
    ("utc_time", "21:47", "2026-01-14T21:47:00Z", True), ("utc_time", "23:59", "00:00", True),
    ("utc_time", "21:47", "21h47", True), ("utc_time", "21:47", "21:47:59", True),
    # hedged or multi-valued strings are not answers
    ("count", 7, "7 or 8", False), ("count", 3, "3-4", False), ("magnitude", 4.5, "4.5 or 4.7", False),
    ("event_id", "nc75012345", ["nc75012345", "pt75975146"], False),
    ("event_id", "nc75012345", "nc75012345 (same event as pt75975146)", False),
    ("utc_time", "21:47", "21:47 or 21:52", False), ("region", "Nevada", "Nevada or Utah", False),
    # unambiguous strings are read
    ("distance", 5088.8, "5,088.8 km", True), ("count", 7, "7 events", True), ("depth", -0.7, "-0.7 km", True),
    ("region", "Fiji Islands", "Fiji Islands region", True), ("region", "tie", "It's a tie", True),
])
def test_scoring_rules(answer_type, key, given, ok):
    assert is_correct(answer_type, key, given) is ok


def test_probability_normalisation():
    assert normalize("probability", 0.25) == 0.25
    assert normalize("probability", "15%") == pytest.approx(0.15)
    assert normalize("probability", "1 in 3") == pytest.approx(1 / 3)
    assert normalize("probability", "0,3") == pytest.approx(0.3)
    assert normalize("probability", 1.7) is None


def test_consistency_comparison_uses_the_same_tolerances():
    assert equivalent("depth", 18.6, 18.8)
    assert not equivalent("count", 3, 4)
    assert equivalent("count", None, None) and not equivalent("count", 3, None)


def test_confidence_intervals():
    lo, hi = wilson(10, 10)
    assert 0.69 < lo < 0.73 and hi == 1.0
    gap, lo, hi = difference_ci(45, 50, 20, 50)
    assert gap == pytest.approx(0.5) and lo > 0
