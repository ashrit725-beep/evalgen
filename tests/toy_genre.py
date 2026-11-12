"""A test-only genre. It lives in tests/, never in evalgen/genres, so it is never public: tests load it
through the registry's plugin loader (evalgen.genres.load_plugin, or a private run config's plugins),
as a held-out genre is loaded.

toy-telex: the classic digest of any domain, relayed as a teleprinter message. The message opens with
a ZCZC line that names the relay service and the message's serial number, every line of the digest
follows behind "+ " (a blank line as a lone "+"), and NNNN closes it:

    ZCZC Toy-Telex relay 4821
    + EARTHQUAKE DIGEST (edition 512-C)
    + Compiled 2026-04-12T06:00:00Z; covers the 24 hours before compilation. ...
    +
    + [nc] nc71234567 | 2026-04-12T05:12:19Z | Northern California | ...
    NNNN

It draws its style as classic does (a variant in the same genre avoids the original's), writes the
digest with classic, and reads it back by removing the frame and parsing with classic, so it is exactly
as readable as classic: the oracle recovers every report and notice of every rendering.

The module defines two genres of one class, which differ only in the relay service their frame names
(so neither recognises the other's messages):

    GENRE   "toy-telex", the plugin's genre
    CABLE   "toy-cable", for tests that need a third genre: they load a copy of this file whose GENRE
            is CABLE as a plugin of its own (a private run config lists only what its plugins define)
"""
from __future__ import annotations

import random
from typing import Any, Mapping

from evalgen.domains.base import Domain, Glossary
from evalgen.genres.base import Genre, RenderContext
from evalgen.genres.classic import CLASSIC
from evalgen.ops import DigestData

CLOSING = "NNNN"
PREFIX = "+ "


class Telex(Genre):
    def __init__(self, name: str = "toy-telex", service: str = "Toy-Telex") -> None:
        self.name = name
        self.opening = f"ZCZC {service} relay"  # then the message's serial number

    def choose_style(self, rng: random.Random, domain: Domain,
                     avoid: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return CLASSIC.choose_style(rng, domain, avoid=avoid)

    def render(self, ctx: RenderContext) -> str:
        """The classic digest (which checks itself for a record domain), framed as a telex message."""
        body = CLASSIC.render(ctx).split("\n")
        serial = ctx.rng.randrange(1000, 10000)
        return "\n".join([f"{self.opening} {serial}", *(PREFIX + line if line else PREFIX.rstrip() for line in body),
                          CLOSING])

    def sniff(self, digest_text: str) -> bool:
        lines = digest_text.strip().split("\n")
        return len(lines) >= 2 and lines[0].startswith(self.opening + " ") and lines[-1] == CLOSING

    def parse(self, digest_text: str, glossary: Glossary | None, domain: Domain) -> DigestData:
        """The frame removed, then the classic parser: every report and notice the digest holds."""
        lines = digest_text.strip().split("\n")[1:-1]
        body = "\n".join(line[len(PREFIX):] if line.startswith(PREFIX) else "" for line in lines)
        return CLASSIC.parse(body, glossary, domain)


GENRE = Telex()
CABLE = Telex("toy-cable", "Toy-Cable")
