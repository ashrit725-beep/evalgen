"""The genre interface: how one kind of document presents a digest's content.

An instance's content is fixed before it is rendered: its reports, notices and chatter lines. A
*genre* only decides how that content looks as text (the 0.2.0 bulletin, wire, log and prose lines,
a news article, an email thread, a chat log, a table, a form, a radio transcript), and must read its
own text back to exactly the same reports and notices.

    RenderContext  everything one rendering is drawn from
    Glossary       a prompt's data dictionary: what its records, their fields and units are called
    Genre          the base class; each genre module exports an instance as ``GENRE``
"""
from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..domains.base import Domain, RecordReport, Unit
from ..models import Notice, Report
from ..ops import DigestData


@dataclass(frozen=True)
class GlossaryField:
    """One numeric field as a glossary defines it."""

    key: str  # its key in record values and in question chains: "depth"
    label: str  # its name in the text: "depth"
    units: tuple[Unit, ...] = ()  # the units its values may be printed in, with their conversions


@dataclass(frozen=True)
class Glossary:
    """A prompt's data dictionary: what the records are called, how their ID, source and group are
    labelled, and the label and units of each numeric field. Non-legacy prompts print it (prompt
    assembly, plan phase 1b), so that a reader, and a genre's parser, learn every label and unit from
    the prompt itself: it is the only way to read a domain whose surface words are drawn per
    instance. Legacy (0.2.0) prompts print none."""

    domain: str  # the registry name of the records' domain
    noun: str
    plural: str
    id_label: str
    source_label: str
    group_label: str
    fields: tuple[GlossaryField, ...] = ()

    @classmethod
    def of(cls, domain: Domain) -> Glossary:
        """The glossary of a domain's records."""
        return cls(domain.name, domain.noun, domain.plural, domain.id_label, domain.source_label,
                   domain.group_label, tuple(GlossaryField(spec.key, spec.label, spec.units) for spec in domain.fields))


_GLOSSARY = re.compile(r"<glossary>.*?</glossary>", re.DOTALL)


def render_glossary(glossary: Glossary) -> str:
    """The glossary as a non-legacy prompt prints it, between <glossary> tags, in a form that
    parse_glossary reads back exactly."""
    # The printed form, and the generic classic genre that reads values through it, come with the
    # generic record path (plan phase 1b).
    raise NotImplementedError("printing a glossary comes with the generic record path (plan phase 1b)")


def parse_glossary(prompt: str) -> Glossary | None:
    """The glossary a prompt prints, read back from its text alone; None for a prompt that prints
    none, as every legacy (0.2.0) prompt does."""
    if _GLOSSARY.search(prompt) is None:
        return None
    # Reading a printed glossary comes with its printed form (plan phase 1b).
    raise NotImplementedError("reading a printed glossary comes with the generic record path (plan phase 1b)")


@dataclass(frozen=True)
class RenderContext:
    """What one rendering is drawn from. The lists are shared by every rendering of an instance, so a
    genre must never modify them."""

    reports: Sequence[Report | RecordReport]  # in planting order: what every per-line draw follows
    notices: Sequence[Notice]
    chatter: Sequence[str]  # lines that carry no data
    compiled_at: int  # when the digest was compiled: the end of its time window (Unix seconds, UTC)
    window_hours: int
    style: Mapping[str, Any]  # what the genre's choose_style drew for this rendering
    rng: random.Random  # the rendering's own stream, continued from choose_style
    domain: Domain
    glossary: Glossary | None = None  # what the prompt prints about the records; None on the legacy path


class Genre(ABC):
    """One kind of document a digest can be written as. A genre module subclasses this, implements
    the four methods and exports an instance as ``GENRE`` (see evalgen.genres)."""

    name: str  # registry key: "classic"

    @abstractmethod
    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Draw how one rendering looks. A metamorphic variant in the same genre passes the original's
        style as `avoid`, and must then differ from it."""

    @abstractmethod
    def render(self, ctx: RenderContext) -> str:
        """The digest text of one rendering."""

    @abstractmethod
    def sniff(self, digest_text: str) -> bool:
        """Whether `digest_text` is this genre's output. No genre may recognise another genre's text:
        the oracle picks its parser by sniffing (see evalgen.genres.detect_genre)."""

    @abstractmethod
    def parse(self, digest_text: str, glossary: Glossary | None) -> DigestData:
        """The exact inverse of render: every report and notice, read from the text alone, with the
        labels and units of the prompt's glossary (None: a legacy prompt, which prints none)."""
