"""The reading rules must name every wording the digest can print, so no notice is a gotcha."""
from __future__ import annotations

import itertools
import re
from collections import Counter

from evalgen import pipeline
from evalgen.digest import CHATTER, render_notice
from evalgen.domains import get_domain
from evalgen.domains.base import REPEAT_RULE
from evalgen.genres import all_genres, chat, email, radio
from evalgen.models import Notice
from evalgen.pipeline import GenerateConfig, build_instance
from evalgen.questions import Context, compose_new_templates
from evalgen.rules import READING_RULES
from evalgen.seeds import SeedLog
from evalgen.solver import solve
from evalgen.sources import build_domain_snapshot

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


# ------------------------------------------------------------- repeats and forwards, in every genre

CHAT_MESSAGE = re.compile(r"^(?:\[\d{2}:\d{2}\] [\w.-]+|\U0001f4cc Pinned by [\w.-]+): ")


def is_repeat(genre: str, line: str) -> bool:
    """Whether a digest line is a copy the rules call a repeat ("a quoted or read-back repeat of a report
    or notice printed elsewhere in the digest"): a quoted line of an email thread, a quote or read-back
    posted in a chat log, or a radio operator reading a report back. No other genre prints repeats."""
    if genre == "email":
        return line.startswith(email.QUOTE)
    if genre == "chat":
        body = CHAT_MESSAGE.sub("", line)
        return body != line and body.startswith(chat.REPEATS)
    if genre == "radio":
        return any(intro in line for intro in radio.READBACK_INTROS)
    return False


def genre_instances(name: str, genre, seed: int = 5) -> list:
    """The first instance of a run about the domain `name` with every rendering in `genre`, as the
    pipeline builds it off the legacy path."""
    domain, log = get_domain(name), SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    new = compose_new_templates(log.rng("templates", "new", domain.name), domain)
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    return [build_instance(0, snapshot, new, log, GenerateConfig(seed=seed), ctx, set(), {}, domain=domain,
                           genre_plan=(genre,) * 3)]


def test_every_notice_is_printed_once_where_the_rules_count_it():
    """The rules of every domain off the legacy path (rule 7 of a record domain's, rule 11 of the
    earthquake rules) call a quote or a read-back of a report or notice printed elsewhere a repeat, which
    is not a new one, and say that a forwarded or relayed report or notice is not a repeat and counts.
    Some genres deliver every notice that way: an email thread forwards each one, from a notices desk
    whose own message never appears. So in every genre, a reader who strikes out every repeat the rules
    name still reads every notice, exactly once: its one copy is never a quote or a read-back."""
    struck = Counter()
    for name in ("earthquakes", "weather", "invented"):
        domain = get_domain(name)
        for genre in all_genres():
            for inst in genre_instances(name, genre):
                bound = domain.bind(inst.surface)
                rules = " ".join(bound.rules_text().split())
                assert REPEAT_RULE in rules and "quoted, forwarded or read-back" not in rules
                assert "A forwarded or relayed report or notice is not a repeat: it counts, once." in rules
                notices = sorted((n.kind, n.target) for n in inst.notices)
                assert notices
                for rendering in inst.renderings:
                    lines = rendering.digest.split("\n")
                    kept = [line for line in lines if not is_repeat(genre.name, line)]
                    struck[genre.name] += len(lines) - len(kept)
                    data = genre.parse("\n".join(kept), bound.glossary(), bound)
                    assert sorted((n.kind, n.target) for n in data.notices) == notices, (name, genre.name)
                    if genre.name == "email":  # each one in a forwarded message, which the rules count
                        forwarded = [i for i, line in enumerate(kept) if line.startswith(email.FORWARD_MARKERS[
                            rendering.style["forwards"]][0])]
                        assert len(forwarded) >= len(notices)
    assert struck["email"] and struck["chat"] and struck["radio"]  # each of them printed repeats to strike out
