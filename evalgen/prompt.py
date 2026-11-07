"""Prompt construction: reading rules + digest + numbered questions + a strict JSON answer format.

The prompt is the ONLY thing an API model sees. Everything it needs is inside: no fetching, no
tools, no scraping. That is what makes scraping skill irrelevant to the score.
"""
from __future__ import annotations

from typing import Sequence

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


def answer_spec_text(answer_type: str, unit: str | None) -> str:
    return ANSWER_SPECS[answer_type].format(unit=UNIT_WORDS.get(unit or "", ""))


def build_prompt(digest: str, items: Sequence[tuple[str, str, str, str | None]]) -> str:
    """items: (qid, question text, answer_type, unit) in display order."""
    questions = "\n".join(
        f"{qid}. {text}\n    Answer type: {answer_spec_text(answer_type, unit)}."
        for qid, text, answer_type, unit in items
    )
    shape = ", ".join(f"\"{qid}\": {PLACEHOLDERS[answer_type]}" for qid, _, answer_type, _ in items)
    return (
        f"{INTRO}\n\n{READING_RULES}\n<digest>\n{digest}\n</digest>\n\nQuestions:\n{questions}\n\n"
        "Answer format: reply with ONE JSON object and nothing else (no explanation, no code fences):\n"
        f"{{\"answers\": {{{shape}}}}}\n"
        "Numbers must be plain JSON numbers without units. Event IDs, region names and times are JSON "
        "strings. Include every question ID exactly once.\n"
    )
