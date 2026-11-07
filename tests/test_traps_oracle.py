"""Traps are planted as designed, the oracle handles every one, and the naive baseline does not."""
from __future__ import annotations

from evalgen.ops import DigestData, execute
from evalgen.rules import Mishandle
from evalgen.scoring import same_answer
from evalgen.traps import check_resolution
from evalgen.util import read_json


def test_every_instance_has_all_three_trap_kinds(instances):
    for inst in instances:
        kinds = {t.kind for t in inst.traps}
        assert kinds == {"duplicate", "retraction", "conflicting_magnitude"}, inst.instance_id


def test_reading_rules_recover_the_snapshot(instances, snapshot_events):
    for inst in instances:
        events = [snapshot_events[eid] for eid in inst.event_ids]
        assert check_resolution(inst.reports, inst.notices, events) == [], inst.instance_id


def test_answer_key_is_computed_from_the_snapshot(instances, snapshot_events):
    for inst in instances:
        events = [snapshot_events[eid] for eid in inst.event_ids]
        for q in inst.questions:
            assert execute(q.chain, events) == inst.answer_key[q.qid]


def test_most_traps_are_load_bearing(instances):
    traps = [t for inst in instances for t in inst.traps]
    assert sum(1 for t in traps if t.affected) / len(traps) >= 0.6


def test_recorded_fallen_answers_are_wrong_and_reproducible(instances):
    for inst in instances:
        data = DigestData(inst.reports, inst.notices)
        for trap in inst.traps:
            for qid, fallen in trap.affected.items():
                q = inst.question(qid)
                assert not same_answer(q.answer_type, fallen, inst.answer_key[qid])
                readings = [execute(q.chain, data, mishandle=Mishandle.from_dict(m)) for m in trap.mishandles]
                assert fallen in readings
                assert any(same_answer(q.answer_type, fallen, x) for x in trap.fallen[qid])
            for qid, answers in trap.fallen.items():  # joint misreadings are wrong answers too
                q = inst.question(qid)
                assert answers and all(not same_answer(q.answer_type, a, inst.answer_key[qid]) for a in answers)


def test_generator_self_check_passed(small_run):
    check = read_json(small_run / "manifest.json")["self_check"]
    assert check["oracle_checks"] > 0 and check["oracle_agreements"] == check["oracle_checks"]
    assert check["problems"] == []


def test_oracle_scores_100_percent_on_every_rendering(small_run):
    rows = read_json(small_run / "scores" / "oracle.json")["rows"]
    assert rows and all(r["correct"] for r in rows)


def test_oracle_catches_every_trap_and_naive_falls_for_them(small_run):
    models = {m["model"]: m for m in read_json(small_run / "report.json")["models"]}
    assert models["oracle"]["traps"]["caught"]["rate"] == 1.0
    assert models["oracle"]["traps"]["fell_for_trap"]["k"] == 0
    naive = models["naive"]["traps"]
    assert naive["caught"]["rate"] < 0.25
    # the naive reader misses every trap at once; each of its trap errors is explained by the traps
    assert naive["caught"]["k"] + naive["fell_for_trap"]["k"] == naive["caught"]["n"]
    assert models["naive"]["accuracy"]["rate"] < models["oracle"]["accuracy"]["rate"]


def test_naive_is_right_where_no_trap_matters(small_run, instances):
    """The naive baseline only loses points because of traps (alone or combined): the scoring is fair."""
    rows = read_json(small_run / "scores" / "naive.json")["rows"]
    trapped = {(inst.instance_id, qid) for inst in instances for t in inst.traps for qid in t.fallen}
    for row in rows:
        if (row["instance_id"], row["qid"]) not in trapped:
            assert row["correct"], row
