"""Prompt construction: reading rules + digest + numbered questions + a strict JSON answer format.

The prompt is the ONLY thing an API model sees. Everything it needs is inside: no fetching, no
tools, no scraping. That is what makes scraping skill irrelevant to the score.

The introduction, the reading rules, the answer-type descriptions and placeholders, the unit words
and the note on the answer format default to the 0.2.0 earthquake wording, so a legacy prompt is
exactly the 0.2.0 one. Other domains pass their own (deriving them from a domain comes with prompt
assembly, plan phase 1b), and a glossary (the domain's data dictionary, as render_glossary prints
it), when given, goes between the rules and the digest.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .geo import UNIT_WORDS
from .rules import READING_RULES

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


def answer_spec_text(answer_type: str, unit: str | None, specs: Mapping[str, str] = ANSWER_SPECS,
                     unit_words: Mapping[str, str] = UNIT_WORDS) -> str:
    """How an answer of `answer_type`, asked in `unit` (a symbol of `unit_words`), must be given."""
    return specs[answer_type].format(unit="" if unit is None else unit_words[unit])


def build_prompt(digest: str, items: Sequence[tuple[str, str, str, str | None]], *, intro: str = INTRO,
                 rules: str = READING_RULES, glossary: str | None = None,
                 specs: Mapping[str, str] = ANSWER_SPECS, placeholders: Mapping[str, str] = PLACEHOLDERS,
                 unit_words: Mapping[str, str] = UNIT_WORDS, answer_note: str = ANSWER_NOTE) -> str:
    """items: (qid, question text, answer_type, unit) in display order.

    `rules` and `glossary` end with a newline, as READING_RULES does; `specs` describes each answer
    type, with {unit} for the words of the unit asked in (`unit_words`, by symbol); `placeholders`
    shows each type's value in the reply's JSON shape, and `answer_note` closes the format. Without a
    glossary (legacy prompts) none is printed."""
    questions = "\n".join(
        f"{qid}. {text}\n    Answer type: {answer_spec_text(answer_type, unit, specs, unit_words)}."
        for qid, text, answer_type, unit in items
    )
    shape = ", ".join(f"\"{qid}\": {placeholders[answer_type]}" for qid, _, answer_type, _ in items)
    dictionary = "" if glossary is None else f"{glossary}\n"
    return (
        f"{intro}\n\n{rules}\n{dictionary}<digest>\n{digest}\n</digest>\n\nQuestions:\n{questions}\n\n"
        "Answer format: reply with ONE JSON object and nothing else (no explanation, no code fences):\n"
        f"{{\"answers\": {{{shape}}}}}\n"
        f"{answer_note}\n"
    )
