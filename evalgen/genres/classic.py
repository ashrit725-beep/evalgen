"""classic: the 0.2.0 digest as a genre.

Bulletin, prose, wire and log report lines, NOTICE, RETRACTION, UPDATE, CORRECTION and Revised
notice lines, a header, section heads and chatter: ``digest.choose_style`` draws the look,
``digest.render_digest`` writes the text and ``solver.parse_digest`` reads it back. On the earthquake
domain this genre calls them unchanged, with the same arguments and the same random stream, so
legacy runs stay byte-identical. The generic version for every other domain, which reads labels and
units through the glossary, comes with the generic record path (plan phase 1b).
"""
from __future__ import annotations

import random
import re
from typing import Any, Mapping

from ..digest import HEADERS, choose_style, render_digest
from ..domains.base import Domain, Glossary
from ..domains.earthquakes import EARTHQUAKES
from ..ops import DigestData
from ..solver import parse_digest
from .base import Genre, RenderContext


def _first_line_pattern(header: str) -> re.Pattern[str]:
    """A header's first line, with its placeholders ({edition}, {window}, {compiled}) as wildcards."""
    literals = re.split(r"\{\w+\}", header.split("\n")[0])
    return re.compile("^" + ".+".join(re.escape(text) for text in literals) + "$")


_HEADER_LINES = tuple(_first_line_pattern(header) for header in HEADERS)


class Classic(Genre):
    name = "classic"

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        _require_earthquakes(domain.name)
        return choose_style(rng, avoid=avoid)

    def render(self, ctx: RenderContext) -> str:
        _require_earthquakes(ctx.domain.name)
        return render_digest(ctx.reports, ctx.notices, ctx.chatter, compiled_at=ctx.compiled_at,
                             window_hours=ctx.window_hours, style=ctx.style, rng=ctx.rng)

    def sniff(self, digest_text: str) -> bool:
        """A classic digest opens with one of the 0.2.0 headers. (Its report lines are no sign: the
        prose and log patterns accept any text before the time, a quoting prefix included.)"""
        first = digest_text.lstrip().split("\n", 1)[0]
        return any(pattern.match(first) for pattern in _HEADER_LINES)

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        _require_earthquakes(domain.name)
        return parse_digest(digest_text)


def _require_earthquakes(domain: str) -> None:
    if domain != EARTHQUAKES.name:
        raise NotImplementedError(f"the classic genre does not render or read {domain} records yet "
                                  "(generic record path, plan phase 1b)")


CLASSIC = Classic()
GENRE = CLASSIC  # what the registry looks up (see evalgen.genres)
