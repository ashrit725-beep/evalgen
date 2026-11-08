"""Prompt construction: reading rules + digest + numbered questions + a strict JSON answer format.

The prompt is the ONLY thing an API model sees. Everything it needs is inside: no fetching, no
tools, no scraping. That is what makes scraping skill irrelevant to the score.

``build_prompt`` defaults to the 0.2.0 earthquake wording (introduction, reading rules, answer-type
descriptions and placeholders, unit words, note on the answer format), so a legacy prompt is exactly
the 0.2.0 one. A non-legacy prompt (``domain_prompt``) is worded from its domain instead: an
introduction, the domain's genre-neutral reading rules, the domain's glossary (as render_glossary
prints it, between the rules and the digest), the digest and the questions, each with an answer-type
description and a placeholder made from its type's kind (any registered type) and the domain's words.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .domains.base import Domain, Glossary
from .genres.base import render_glossary
from .geo import UNIT_WORDS
from .rules import READING_RULES
from .scoring import get_answer_type
from .units import LENGTH

ANSWER_SPECS = {
    "count": "an integer",
    "event_id": "an event ID (string)",
    "magnitude": "a number",
    "depth": "a number in {unit}, rounded to 1 decimal place",
    "distance": "a number in {unit}, rounded to 1 decimal place",
    "region": "a region name exactly as written in the digest, or \"tie\"",
    "utc_time": "a UTC time string \"HH:MM\" (24-hour clock)",
    "probability": "a probability between 0 and 1",
}
PLACEHOLDERS = {
    "count": "<integer>",
    "event_id": "\"<event ID>\"",
    "magnitude": "<number>",
    "depth": "<number>",
    "distance": "<number>",
    "region": "\"<region or tie>\"",
    "utc_time": "\"HH:MM\"",
    "probability": "<number from 0 to 1>",
}

INTRO = (
    "You are taking a careful-reading test. Below is a digest of earthquake reports compiled from "
    "several seismic networks, followed by questions about it. Use only the digest and the reading "
    "rules; no outside knowledge, tools or web access is needed or allowed."
)
ANSWER_NOTE = (
    "Numbers must be plain JSON numbers without units. Event IDs, region names and times are JSON "
    "strings. Include every question ID exactly once."
)

# (qid, question text, answer type, unit symbol or None), in display order. Optional, not "X | None":
# this alias is evaluated at import, and Python 3.9 has no such unions.
Item = tuple[str, str, str, Optional[str]]


def answer_spec_text(answer_type: str, unit: str | None, specs: Mapping[str, str] = ANSWER_SPECS,
                     unit_words: Mapping[str, str] = UNIT_WORDS) -> str:
    """How an answer of `answer_type`, asked in `unit` (a symbol of `unit_words`), must be given."""
    return specs[answer_type].format(unit="" if unit is None else unit_words[unit])


def build_prompt(digest: str, items: Sequence[Item], *, intro: str = INTRO, rules: str = READING_RULES,
                 glossary: Glossary | None = None, specs: Mapping[str, str] = ANSWER_SPECS,
                 placeholders: Mapping[str, str] = PLACEHOLDERS, unit_words: Mapping[str, str] = UNIT_WORDS,
                 answer_note: str = ANSWER_NOTE) -> str:
    """items: (qid, question text, answer_type, unit) in display order.

    `rules` ends with a newline, as READING_RULES does; `glossary`, when given, is printed between the
    rules and the digest by render_glossary, the one form a prompt's glossary takes (legacy prompts
    print none). `specs` describes each answer type, with {unit} for the words of the unit asked in
    (`unit_words`, by symbol); `placeholders` shows each type's value in the reply's JSON shape, and
    `answer_note` closes the format."""
    questions = "\n".join(
        f"{qid}. {text}\n    Answer type: {answer_spec_text(answer_type, unit, specs, unit_words)}."
        for qid, text, answer_type, unit in items
    )
    shape = ", ".join(f"\"{qid}\": {placeholders[answer_type]}" for qid, _, answer_type, _ in items)
    dictionary = "" if glossary is None else f"{render_glossary(glossary)}\n"
    return (
        f"{intro}\n\n{rules}\n{dictionary}<digest>\n{digest}\n</digest>\n\nQuestions:\n{questions}\n\n"
        "Answer format: reply with ONE JSON object and nothing else (no explanation, no code fences):\n"
        f"{{\"answers\": {{{shape}}}}}\n"
        f"{answer_note}\n"
    )


# ------------------------------------------------------------------------ non-legacy prompts

# Answer-type descriptions and placeholders by answer kind (scoring.AnswerType.kind); {id_label} and
# {group_label} are the domain's words, and {unit} stays for answer_spec_text to fill in.
KIND_SPECS = {
    "count": "an integer",
    "id": "one {id_label}, as a string",
    "number": "a number",
    "group": "one {group_label} exactly as written in the digest, or \"tie\"",
    "clock": "a UTC time string \"HH:MM\" (24-hour clock)",
    "probability": "a probability between 0 and 1",
}
NUMBER_IN_UNIT = "a number in {unit}, rounded as the question says"  # a number answer asked in a unit
KIND_PLACEHOLDERS = {
    "count": "<integer>",
    "id": "\"<{id_label}>\"",
    "number": "<number>",
    "group": "\"<{group_label} or tie>\"",
    "clock": "\"HH:MM\"",
    "probability": "<number from 0 to 1>",
}


def domain_intro(domain: Domain) -> str:
    return ("You are taking a careful-reading test. Below are the reading rules, a glossary that defines the "
            f"records, and a digest of reports about {domain.plural} compiled from several sources, followed by "
            "questions about it. Use only the digest, the glossary and the reading rules; no outside knowledge, "
            "tools or web access is needed or allowed.")


def domain_answer_note(domain: Domain) -> str:
    id_label = domain.id_label[:1].upper() + domain.id_label[1:]
    return (f"Numbers must be plain JSON numbers without units. {id_label}s, {domain.group_label} names and times "
            "are JSON strings. Include every question ID exactly once.")


def domain_unit_words(domain: Domain) -> dict[str, str]:
    """The words of every unit an answer about the domain's records can be asked in, by symbol: its
    fields' units, and the length units of distances when its records have a location."""
    units = [u for spec in domain.fields for u in spec.units] + (list(LENGTH) if domain.has_location else [])
    return {unit.symbol: unit.words for unit in units}


def domain_answer_spec(answer_type: str, in_unit: bool, domain: Domain) -> str:
    """How an answer of a registered `answer_type` must be given, from its kind and the domain's words
    (`in_unit`: the question asks for it in a unit, which answer_spec_text fills in)."""
    kind = get_answer_type(answer_type).kind
    if kind == "number" and in_unit:
        return NUMBER_IN_UNIT
    return KIND_SPECS[kind].format(id_label=_literal(domain.id_label), group_label=_literal(domain.group_label))


def domain_placeholder(answer_type: str, domain: Domain) -> str:
    """A registered `answer_type`'s value in the reply's JSON shape, in the domain's words."""
    return KIND_PLACEHOLDERS[get_answer_type(answer_type).kind].format(id_label=domain.id_label,
                                                                       group_label=domain.group_label)


def domain_prompt(domain: Domain, digest: str, items: Sequence[Item], glossary: Glossary) -> str:
    """The prompt of a non-legacy rendering: the domain's introduction, its reading rules
    (Domain.rules_text), its glossary, the digest and the questions, whose answer types (any registered
    type) are described and shown from their kinds in the domain's words."""
    types = list(dict.fromkeys(answer_type for _, _, answer_type, _ in items))
    in_unit = {answer_type for _, _, answer_type, unit in items if unit is not None}
    return build_prompt(digest, items, intro=domain_intro(domain), rules=domain.rules_text(), glossary=glossary,
                        specs={t: domain_answer_spec(t, t in in_unit, domain) for t in types},
                        placeholders={t: domain_placeholder(t, domain) for t in types},
                        unit_words=domain_unit_words(domain), answer_note=domain_answer_note(domain))


def _literal(text: str) -> str:
    """`text` inside a str.format template, standing for itself."""
    return text.replace("{", "{{").replace("}", "}}")
