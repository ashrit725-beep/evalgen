"""A test domain that is earthquakes in everything but its op names (not a test module itself).

GenericQuakes keeps every earthquake field, record type, wording and answer type, but its chains
store the generic kinds ({"op": "at_least", "field": "mag", ...}) instead of the 0.2.0 aliases
("mag_at_least"), and its lexicon is keyed by the generic op names ("at_least(mag)"). Composing,
answering and planting on it must therefore give what earthquakes give, step for step. Subclasses
rename its answer types to stand in for a domain whose types are not built into the scorer.
"""
from __future__ import annotations

from types import MappingProxyType

from evalgen.domains.base import Lexicon
from evalgen.domains.earthquakes import LEXICON, Earthquakes
from evalgen.ops import ALIASES

GENERIC_OP_NAMES = {name: f"{a.kind}({a.field})" if a.field else a.kind for name, a in ALIASES.items()}


def _rekeyed(table):
    return {GENERIC_OP_NAMES.get(name, name): wordings for name, wordings in table.items()}


class GenericQuakes(Earthquakes):
    name = "generic-quakes"
    op_names = MappingProxyType({})
    lexicon = Lexicon(LEXICON.records, LEXICON.all_records, _rekeyed(LEXICON.filters), _rekeyed(LEXICON.questions),
                      legacy=LEXICON.legacy)


GENERIC_QUAKES = GenericQuakes()
