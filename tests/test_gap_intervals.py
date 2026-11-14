"""The design effect that widens the intervals of the source and genre gaps (scoring.design_effect,
report.split_gap): an instance's rows are read in one digest and asked again in every rendering, so
they are not independent draws, and the intervals treat the instance as the independent unit."""
from __future__ import annotations

import functools
import math
import operator
import random
from fractions import Fraction

import pytest

from evalgen.mix import HELD_OUT, PUBLIC
from evalgen.report import split_gap
from evalgen.scoring import design_effect, difference_ci, widened


def clusters(outcomes: list[list[bool]], first: bool, prefix: str) -> list[tuple]:
    """(cluster, on the first side, correct) rows: one cluster per list of outcomes."""
    return [(f"{prefix}{i}", first, y) for i, cluster in enumerate(outcomes) for y in cluster]


def test_rows_that_are_their_own_clusters_have_no_design_effect_beyond_the_small_sample_factor():
    rows = clusters([[y] for y in [True] * 7 + [False] * 3], True, "a") + clusters(
        [[y] for y in [True] * 4 + [False] * 6], False, "b")
    assert design_effect(rows) == pytest.approx(20 / 19)  # G / (G - 1), over G = 20 clusters


def test_clusters_whose_rows_agree_count_as_one_draw_each():
    """Ten clusters a side, of five rows that all agree: the difference varies as if each cluster were one
    row, five times as much as five times as many independent rows would make it vary."""
    rows = clusters([[y] * 5 for y in [True] * 7 + [False] * 3], True, "a") + clusters(
        [[y] * 5 for y in [True] * 4 + [False] * 6], False, "b")
    assert design_effect(rows) == pytest.approx(5 * 20 / 19)
    # Counting every row twice in its cluster doubles the design effect and halves the independent
    # variance, so the widened interval stays exactly as wide: the rows carry no more information.
    def half_width(rows) -> float:
        k1 = sum(y for _, first, y in rows if first)
        n1 = sum(1 for _, first, _ in rows if first)
        k2 = sum(y for _, first, y in rows if not first)
        n2 = sum(1 for _, first, _ in rows if not first)
        gap, lo, hi = difference_ci(k1, n1, k2, n2)
        lo, hi = widened(gap, lo, hi, math.sqrt(design_effect(rows)))
        return (hi - lo) / 2

    assert half_width(rows + rows) == pytest.approx(half_width(rows))


def test_a_design_effect_is_never_below_one():
    # Every cluster holds one right and one wrong answer on its side: its residuals cancel, so the
    # clusters vary less than independent rows would. That never narrows an interval.
    rows = clusters([[True, False]] * 6, True, "a") + clusters([[True, False]] * 6, False, "b")
    assert design_effect(rows) == 1.0
    assert design_effect(clusters([[True, False, True]], True, "a") + clusters([[False]], False, "a")) == 1.0
    assert design_effect(clusters([[True] * 3] * 4, True, "a") + clusters([[True] * 3] * 4, False, "b")) == 1.0
    assert design_effect(clusters([[True, False]] * 3, True, "a")) == 1.0  # one side only


def test_a_gap_whose_answers_go_together_in_each_instance_gets_an_interval_as_wide_as_its_instances_allow():
    """split_gap widens the raw and the adjusted interval by the design effect of their rows: rows that
    agree within each instance are worth one draw per instance, however many questions it asks."""
    def rows(copies: int) -> list[dict]:
        out = []
        for i in range(30):
            side = PUBLIC if i < 15 else HELD_OUT
            right = (i % 3 != 0) if side == PUBLIC else (i % 2 == 0)
            for q in range(4):
                out += [{"instance_id": f"inst-{i:03d}", "qid": f"Q{q}", "side": side, "correct": right,
                         "stratum": q % 2}] * copies
        return out

    gaps = [split_gap(rows(copies), lambda row: row["side"], lambda row: (row["stratum"],)) for copies in (1, 2)]
    for gap in gaps:
        wald = difference_ci(gap["public"]["k"], gap["public"]["n"], gap["held_out"]["k"], gap["held_out"]["n"])
        assert gap["ci95"][1] - gap["ci95"][0] > 1.9 * (wald[2] - wald[1])  # four agreeing rows a cluster: x2
    once, twice = gaps
    assert twice["public_minus_held_out"] == once["public_minus_held_out"]
    assert twice["ci95"] == pytest.approx(once["ci95"])  # counting each row twice adds no information
    assert twice["adjusted"]["ci95"] == pytest.approx(once["adjusted"]["ci95"], rel=0.05)


def test_a_gap_whose_side_holds_one_instance_is_never_flagged():
    """One held-out instance gets no widening from the design effect (its rows' residuals sum to zero within
    it), so one misread digest beside thirty public instances made a flagged adjusted gap of +80 points. A
    gap is flagged only when each side's compared rows come from two instances or more."""
    def rows(held_out: list[list[bool]]) -> list[dict]:
        out = [{"instance_id": f"inst-{i:03d}", "side": PUBLIC, "correct": q != 0} for i in range(30) for q in range(5)]
        out += [{"instance_id": f"inst-{100 + i:03d}", "side": HELD_OUT, "correct": y}
                for i, answers in enumerate(held_out) for y in answers]
        return out

    one = split_gap(rows([[False] * 5]), lambda row: row["side"], lambda row: ())
    assert one["adjusted"]["public_minus_held_out"] == pytest.approx(0.8)
    assert one["adjusted"]["ci95"][0] > 0 and not one["flagged"]  # the interval excludes 0, but rests on one instance
    three = split_gap(rows([[False] * 5] * 3), lambda row: row["side"], lambda row: ())
    assert three["flagged"]  # three digests misread alike are flagged


def test_the_design_effect_sums_exactly_on_every_python_version():
    """The design effect widens the source and genre gaps' intervals, which report.json gives to the last digit.
    It summed its clusters' squared residuals with the built-in sum(), which adds floats one by one up to
    Python 3.11 and with compensation from 3.12, so a run's report.json differed in its last digits between
    Python 3.9 and 3.14. It sums them exactly rounded (math.fsum): here, where adding one by one gives another
    design effect, it gives the exact sum's."""
    rng = random.Random(0)
    rows = []
    for i in range(60):
        rate, side = rng.random(), rng.random() < 0.5
        rows += [(f"inst-{i:03d}", side if rng.random() < 0.8 else not side, rng.random() < rate)
                 for _ in range(rng.randint(3, 15))]
    (k1, n1), (k2, n2) = ((sum(c for _, f, c in rows if f == side), sum(1 for _, f, _ in rows if f == side))
                          for side in (True, False))
    p1, p2 = k1 / n1, k2 / n2
    totals: dict[str, float] = {}
    for cluster, first, correct in rows:  # the residual totals, formed as design_effect forms them
        totals[cluster] = totals.get(cluster, 0.0) + ((correct - p1) / n1 if first else -(correct - p2) / n2)
    squares = [t * t for t in totals.values()]
    independent, g = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2, len(totals)
    one_by_one = functools.reduce(operator.add, squares, 0.0)  # what sum() gave up to Python 3.11
    exact = float(sum(map(Fraction, squares)))
    assert one_by_one != exact and g / (g - 1) * one_by_one / independent != g / (g - 1) * exact / independent
    assert design_effect(rows) == g / (g - 1) * exact / independent > 1.0
