"""The reading rules must name every wording the digest can print, so no notice is a gotcha."""
from __future__ import annotations

import itertools

from evalgen import pipeline
from evalgen.digest import CHATTER, render_notice
from evalgen.models import Notice
from evalgen.rules import READING_RULES
from evalgen.solver import solve

# the words the rules use to identify each kind of notice
RULE_WORDS = {"retraction": ("retraction", "retracted", "withdrawn"), "correction": ("correction", "revised")}
RULES = " ".join(READING_RULES.split())  # the rules text with its line wrapping undone


def _all_wordings(kind: str) -> list[str]:
    notice = Notice(kind=kind, target="us7000abcd", time=1770530400, reason="quarry blast",
                    new_mag=4.2, old_mag=4.5, mag_type="mb")
    wordings = []
    for form in itertools.count():
        try:
            wordings.append(render_notice(notice, form, "iso"))
        except IndexError:
            return wordings


def test_every_notice_wording_is_named_by_the_reading_rules():
    rules = RULES.lower()
    for kind, words in RULE_WORDS.items():
        assert all(w in rules for w in words), (kind, words)
        wordings = _all_wordings(kind)
        assert len(wordings) >= 3
        for text in wordings:
            assert any(w in text.lower() for w in words), f"the rules do not name this {kind} wording: {text}"


def test_rules_say_how_to_compare_magnitudes_on_different_scales():
    assert any("scale" in line for line in CHATTER)  # digests may remind readers that scales differ ...
    assert "whatever their scale" in RULES  # ... so the rules must say how to compare them


def test_rules_fix_time_window_ends_and_coordinate_signs():
    assert "includes its start and excludes its end" in RULES  # ops.time_between: start <= t < end
    assert "negative for south and west" in RULES


def test_rules_cover_own_network_preliminaries_signed_depths_regions_and_other_lines():
    assert "a PRELIMINARY report from its own network" in RULES  # traps can file it under its own network
    assert "A depth can be negative" in RULES  # USGS hypocentres above sea level; the key keeps the sign
    assert "matched exactly as written" in RULES  # ops.passes: e.region == step["region"]
    assert "carry no data and change nothing" in RULES  # chatter, headers, station status lines
    assert "as printed" not in RULES  # rule 7 must not compete with rules 2 and 3


def test_coordinates_are_stored_at_the_printed_precision(snapshot_events, instances):
    for e in snapshot_events.values():
        assert (round(e.lat, 3), round(e.lon, 3)) == (e.lat, e.lon), e.id
    for inst in instances:
        for r in inst.reports:
            assert (round(r.lat, 3), round(r.lon, 3)) == (r.lat, r.lon), r.ref


def test_the_oracle_reproduces_distance_keys_exactly_from_the_printed_text(instances):
    """Not just within tolerance: a reader working from the printed digits gets the key itself."""
    checked = 0
    for inst in instances:
        specs = pipeline.question_specs(inst)
        for rendering in inst.renderings:
            answers = solve(rendering.prompt, specs, mode="oracle")
            for q in inst.questions:
                if q.answer_type == "distance":
                    assert answers[q.qid] == inst.answer_key[q.qid], (inst.instance_id, rendering.variant, q.qid)
                    checked += 1
    assert checked > 0
