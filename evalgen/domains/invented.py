"""invented: records of a world that does not exist, worded afresh for every instance.

No live source describes these records, so there is nothing to practise on: they are consignments
(or batches, lots, totes, ...) handled at depots (or yards, quays, sectors, ...), each with two
invented quantities, filed by invented agencies. Everything a reader sees of the domain is drawn per
instance from its own stream (the seed path "instance", i, "surface", which only non-legacy runs
create; see Domain.draw_surface):

    words     the record noun and its plural, the ID label, the source label and the group label
    IDs       a prefix of two or three letters, its letter case and its separator: "KX/40512"
    names     a name for every group (depot) and every source (agency), coined from syllables
    fields    a label for each quantity, and its units: a stored unit and one or two others, each
              with an exact stated conversion ("a value in vk is the value in tl × 12")
    traps     the re-reporting agencies, duplicate notes, retraction reasons and notices, and
              correction notices (Surface.trap_vocab), and the chatter lines

So even the surface of the domain is unpredictable, and the glossary each prompt prints is the only
definition of it: what a record is called, what its fields mean and how its units convert. A digest
might read, in the classic genre's bulletin lines:

    [Ostrel Haulage] KX/40512 | 2026-03-02T07:12:19Z | Varnholt | gross heft 15.308 tl | dwell time
        31.5 mr | verified

The values come from fixed distributions of the two roles at each canonical depot (``Depot``), so
margins, rank gaps and threshold steps keep their meaning whatever the surface calls them:

    f1  the primary quantity: log-normal by depot (medians 95 to 340), stored to 1 decimal place
    f2  the secondary quantity: log-normal by depot (medians 15 to 52), stored to 1 decimal place,
        mildly correlated with f1

Their scoring tolerances are relative only, so that they mean the same in every unit a surface can
draw: the scorer registers a field's tolerance once, when the domain is loaded, and applies it to unit
symbols it has never seen. A unit drawn for an instance answers to as many places as keep its last
digit no coarser than a stored step, and its thresholds are rounded to a step near the field's own
(``drawn_unit``), which keeps every guard margin above the rounding of a threshold restated in
another unit.

The canonical world (``DEPOTS``, ``SOURCES``, IDs "iv-NNNNN") is what the snapshot stores; a surface
renames its groups and sources and replaces the ID prefix (Surface.dress). Records have no location,
so the spatial templates and their twins drop out as a pair. The domain has no live source: its
snapshot is always its synthetic one, and asking for a live one falls back to it with a note.
"""
from __future__ import annotations

import math
import random
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Sequence

from ..units import Unit
from .base import (RULES_WIDTH, Domain, FieldSpec, Lexicon, Record, Surface, TrapVocab, record_rules,
                   standard_lexicon)

# ---------------------------------------------------------------------------- the canonical world


@dataclass(frozen=True)
class Depot:
    """One group of the canonical world, as the snapshot names it: its share of the records, the
    sources that file them, and the log-normal distribution (median, sigma) of each quantity there."""

    name: str
    weight: float
    sources: tuple[str, ...]
    f1: tuple[float, float]
    f2: tuple[float, float]


DEPOTS = (
    Depot("alder", 1.6, ("ash", "bay"), (260.0, 0.50), (22.0, 0.40)),
    Depot("birch", 1.3, ("bay",), (180.0, 0.55), (34.0, 0.45)),
    Depot("cedar", 1.0, ("cob", "ash"), (120.0, 0.45), (18.0, 0.40)),
    Depot("dogwood", 0.9, ("dun",), (340.0, 0.50), (41.0, 0.45)),
    Depot("elm", 1.2, ("elk", "cob"), (150.0, 0.60), (27.0, 0.50)),
    Depot("fir", 0.8, ("dun", "elk"), (210.0, 0.45), (15.0, 0.35)),
    Depot("gorse", 0.7, ("ash",), (95.0, 0.50), (30.0, 0.45)),
    Depot("hazel", 0.6, ("elk",), (280.0, 0.55), (52.0, 0.40)),
)
SOURCES = tuple(sorted({source for depot in DEPOTS for source in depot.sources}))
CORRELATION = 0.3  # between the logarithms of a record's two quantities: heavier lots dwell a little longer
FLOORS = {"f1": 10.0, "f2": 2.0}  # no quantity is smaller (the log-normal tails reach them rarely)
WINDOW_HOURS = 72  # a snapshot covers three days of filings
WORKING_HOURS = (6, 20)  # UTC hours when depots are busiest
NIGHT_SHARE = 0.35  # outside them, a drawn time is kept only this share of the time

ID_PREFIX = "iv-"  # canonical IDs: "iv-" and five digits; every surface replaces the prefix
ID_DIGITS = 5
ID_PATTERN = r"[a-z]{2,3}[-/.]\d{5}"  # lower-cased: two or three letters, a separator, five digits

# ---------------------------------------------------------------------------- the two quantities
# The unbound domain's own words: what its glossary prints when no surface is bound (every instance of
# a run is bound to one). Each surface draws its own labels and units in their place.

HEFT = Unit("vk", "varks")
TALLEN = Unit("tl", "tallens", 12.0, decimals=3, step=0.5)
DWELL = Unit("mr", "mirns")
SKELL = Unit("sk", "skells", 0.25, decimals=1, step=5.0)

PRIMARY = FieldSpec(
    key="f1", label="gross heft", canonical_unit="vk", units=(HEFT, TALLEN), decimals=1,
    answer_type="invented_f1", tolerance=(0.0, 0.005), margin=(0.5, 0.004), rank_gap=2.0, quantiles=(0.25, 0.85),
    threshold_step=5.0, jitter=4.0, positive_threshold=True, filters=("at_least", "below"),
    terminals=("argmax", "rank", "max", "time_of_max"), correctable=True,
)
SECONDARY = FieldSpec(
    key="f2", label="dwell time", canonical_unit="mr", units=(DWELL, SKELL), decimals=1,
    answer_type="invented_f2", tolerance=(0.0, 0.02), margin=(0.3, 0.01), rank_gap=0.0, quantiles=(0.25, 0.8),
    threshold_step=1.0, jitter=1.0, positive_threshold=True, filters=("below", "at_least"),
    terminals=("max", "min"), correctable=True,
)

TRAP_VOCAB = TrapVocab(
    rereporting_sources=("fen", "gil", "hob"),
    duplicate_notes=("same consignment as", "duplicate of"),
    retraction_reasons=("ledger echo", "scanner misread", "test manifest"),
    retraction_notices=("voided", "struck from the ledger"),
    correction_notices=("amended to", "restated as"),
)

# ------------------------------------------------------------------------------- surface words
# Pools of English words, and syllables for coining names. Placeholders: {noun} in ID labels and notes.

NOUNS = (("consignment", "consignments"), ("batch", "batches"), ("dispatch", "dispatches"), ("lot", "lots"),
         ("parcel", "parcels"), ("shipment", "shipments"), ("crate", "crates"), ("pallet", "pallets"),
         ("bale", "bales"), ("bundle", "bundles"), ("cask", "casks"), ("drum", "drums"), ("tranche", "tranches"),
         ("carton", "cartons"), ("tote", "totes"), ("coil", "coils"), ("delivery", "deliveries"),
         ("cargo lot", "cargo lots"), ("freight unit", "freight units"), ("sack", "sacks"))
COINED_NOUN_SHARE = 0.4  # how often the record noun is a coined word instead
ID_LABELS = ("{noun} ID", "{noun} number", "waybill", "docket", "tally number", "manifest number",
             "ledger reference", "tracking code", "chit number", "tag number")
SOURCE_LABELS = ("agency", "carrier", "registrar", "tally office", "clearing house", "handler", "broker", "courier",
                 "operator", "bureau", "consolidator", "forwarder")
GROUP_LABELS = ("depot", "yard", "sector", "hub", "terminal", "bay", "quay", "district", "precinct", "dock",
                "warehouse", "outpost", "zone", "berth", "compound")
F1_QUALIFIERS = ("gross", "net", "declared", "tallied", "assessed", "billed", "laden", "rated", "measured", "audited",
                 "manifest", "stated")
F1_HEADS = ("heft", "mass", "weight", "burden", "bulk", "lading", "tonnage", "displacement", "cubage", "volume")
F2_QUALIFIERS = ("dwell", "hold", "transit", "queue", "staging", "handling", "storage", "turnaround", "clearance",
                 "idle", "layover", "berthing")
F2_HEADS = ("time", "span", "interval", "period", "duration", "lag", "spell", "stretch")
PLACE_WORDS = ("Reach", "Cross", "Gate", "Point", "Landing", "Hollow", "Fork", "Rise", "Ford", "Mill", "Halt", "Spur",
               "Bend", "Hythe", "Staithe")
AGENCY_WORDS = ("Freight", "Haulage", "Carriers", "Logistics", "Transit", "Lines", "Couriers", "Cartage", "Shipping",
                "Movers", "Forwarding")
# No radio procedure words ("copy", "relay", "echo"), which a transcript genre prints as chatter.
DUPLICATE_NOTES = ("same {noun} as", "duplicate of", "dup of", "re-entry of", "mirror of", "repeat filing of",
                   "double entry of", "also logged as")
RETRACTION_REASONS = ("ledger echo", "scanner misread", "test manifest", "phantom tally", "mislabelled crate",
                      "double scan", "dummy entry", "training record", "gate sensor fault", "clerical ghost",
                      "stale import", "barcode collision", "misrouted feed", "keying error")
# Both read after "has been" and after an ID ("KX/40512 voided (double scan)."), as notices print them.
RETRACTION_NOTICES = ("voided", "struck from the ledger", "withdrawn", "RETRACTED", "annulled", "rescinded",
                      "expunged", "struck out", "cancelled")
# Both read after "is" and after a label and an ID ("the heft of KX/40512 is amended to 41 vk").
CORRECTION_NOTICES = ("revised to", "amended to", "corrected to", "restated as", "adjusted to", "re-entered as",
                      "reset to")
# Lines without data (digest.make_chatter fills in {h1} and {h2}, and the bound words).
CHATTER = (
    "Editor's note: entries are reproduced exactly as each {source_label} filed them, units included.",
    "Late {plural} may be listed in the next edition instead.",
    "Reminder: each {source_label} files in its own units; the glossary gives every conversion.",
    "This digest is compiled automatically and has not been checked by an editor.",
    "Ledger maintenance ran from {h1} to {h2} UTC; filing resumed afterwards.",
    "Gate traffic at several {group_label}s was heavier than usual.",
    "The {source_label} feeds were briefly out of step between {h1} and {h2} UTC; nothing was lost.",
    "No {group_label} reported a closure in this period.",
)
CHATTER_LINES = 5  # chatter lines a surface draws (make_chatter prints one to three of them)

ONSETS = ("b", "br", "d", "dr", "f", "fl", "g", "gl", "gr", "h", "k", "kl", "kr", "l", "m", "n", "p", "pl", "pr",
          "qu", "r", "s", "sk", "sl", "sn", "st", "t", "tr", "v", "w", "z")
VOWELS = ("a", "e", "i", "o", "u")
MIDDLES = ("b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "nd", "rl", "rn", "st", "sk")
CODAS = ("b", "d", "g", "k", "l", "ll", "m", "n", "nd", "p", "rk", "rm", "rn", "rt", "sk", "st", "t")
# Coined words never contain these: words that give offence.
UNFIT_PARTS = ("fuck", "slut", "nig", "fag", "porn", "dick", "cunt", "shit", "rape", "nazi", "kill", "dead", "bum",
               "tit", "wank", "cum", "sex", "god", "hell", "anal", "anus", "piss", "turd", "jew", "gay", "homo",
               "tard", "slag", "nob", "gun", "drug", "pot", "hag", "git", "wog", "fart", "prat", "slob", "snot",
               "sod", "pig")
# Nor are coined words and unit symbols ever one of these: a real unit's name or symbol (so that no
# outside knowledge applies to an invented unit), a word the digests print around the data ("lists",
# "ref", "PM"), a word the scorer reads specially ("tie", "or"), or a word of the pools above.
REAL_UNITS = frozenset((
    "ac", "amp", "atm", "au", "bar", "barn", "bbl", "bit", "btu", "bu", "byte", "cal", "cc", "cd", "ch", "cm", "ct",
    "ctn", "cup", "cwt", "dag", "dam", "day", "db", "deg", "dl", "dm", "doz", "dr", "dram", "dwt", "dyn", "dz", "ea",
    "ell", "erg", "ev", "fen", "ft", "ftm", "fur", "gal", "gb", "gi", "gill", "gm", "gr", "gram", "gro", "ha", "hand",
    "hm", "hp", "hr", "hrs", "hz", "in", "jin", "kan", "kat", "kb", "ken", "kg", "kgs", "kl", "km", "kn", "kt", "kw",
    "last", "lb", "lbs", "ld", "li", "link", "lk", "lm", "lt", "ltr", "ly", "mb", "mg", "mi", "mil", "mile", "min",
    "ml", "mm", "mo", "mol", "ms", "mt", "mu", "nit", "nm", "nmi", "ns", "oz", "pa", "pc", "pcs", "pk", "pkt", "ps",
    "psi", "pt", "qt", "rad", "rd", "rem", "rod", "rood", "rpm", "sec", "sen", "span", "sr", "st", "sun", "tb", "tod",
    "tola", "ton", "tun", "us", "vt", "wk", "yd", "yr"))
DIGEST_WORDS = frozenset((
    "list", "lists", "ref", "src", "status", "group", "loc", "place", "note", "near", "wire", "digest", "edition",
    "notice", "update", "revised", "report", "utc", "gmt", "am", "pm", "id", "no", "nr", "was", "now", "not", "tie",
    "tied", "or", "and", "the", "same", "both", "even", "draw", "equal", "neither"))
POOL_WORDS = frozenset(word for phrase in (
    *(word for pair in NOUNS for word in pair), *ID_LABELS, *SOURCE_LABELS, *GROUP_LABELS, *F1_QUALIFIERS, *F1_HEADS,
    *F2_QUALIFIERS, *F2_HEADS, *PLACE_WORDS, *AGENCY_WORDS, *DUPLICATE_NOTES, *RETRACTION_REASONS,
    *RETRACTION_NOTICES, *CORRECTION_NOTICES) for word in re.findall(r"[a-z]+", phrase.casefold()))
RESERVED = REAL_UNITS | DIGEST_WORDS | POOL_WORDS
# A drawn unit is 1 of it = `scale` stored units: one larger unit, one smaller, or one of each.
LARGER_SCALES = (2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 25.0, 40.0, 50.0, 60.0, 100.0)
SMALLER_SCALES = (0.5, 0.25, 0.2, 0.125, 0.1, 0.05, 0.04)
STEPS = (0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0,
         25.0, 50.0, 100.0, 200.0, 250.0, 500.0)  # the thresholds of a drawn unit are rounded to one of these
ID_LETTERS = "bcdfghjklmnpqrstvwxz"
ID_SEPARATORS = ("-", "/", ".")


def drawn_unit(spec: FieldSpec, symbol: str, words: str, scale: float) -> Unit:
    """A unit drawn for `spec` in which one unit is `scale` stored units, exactly. Its answers and
    restated thresholds take as many places as keep their last digit no coarser than one stored step
    (a stored value of 183.7 is 15.308 in a unit of 12: one step of 0.001 there is 0.012), and its
    thresholds are rounded to the step of STEPS nearest the field's own threshold step in it."""
    places = max(0, spec.decimals + math.ceil(math.log10(scale) - 1e-9))
    step = min(STEPS, key=lambda s: abs(math.log(s * scale / spec.threshold_step)))
    return Unit(symbol, words, scale, decimals=places, step=step)


def _plural(word: str) -> str:
    return word + ("es" if word.endswith(("s", "x", "z", "ch", "sh")) else "s")


def _literal(text: str) -> str:
    """`text` inside a str.format template, standing for itself."""
    return text.replace("{", "{{").replace("}", "}}")


class _Coiner:
    """Draws the words of one surface from its stream, in a fixed order, and never uses a word twice
    (in any letter case): no name, label, unit or ID prefix of an instance can be taken for another.
    A coined word or symbol is never RESERVED either."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def fresh(self, *words: str) -> bool:
        """None of `words` is used yet."""
        return all(word.casefold() not in self.used for word in words)

    def take(self, *words: str) -> None:
        """Mark `words` used, and every word inside them ("gross heft": "gross" and "heft" too)."""
        for word in words:
            self.used.add(word.casefold())
            self.used.update(re.findall(r"[a-z]+", word.casefold()))

    def pick(self, pool: Sequence[str], **fill: str) -> str:
        """A phrase of `pool` (filled in with `fill`) that is not used yet."""
        options = [phrase.format(**fill) for phrase in pool]
        choice = self.rng.choice([phrase for phrase in options if self.fresh(phrase)])
        self.take(choice)
        return choice

    def coin(self, syllables: int) -> str:
        """A new lower-case word of `syllables` syllables: onset, vowel, (middle, vowel)..., coda."""
        while True:
            parts = [self.rng.choice(ONSETS), self.rng.choice(VOWELS)]
            for _ in range(syllables - 1):
                parts += [self.rng.choice(MIDDLES), self.rng.choice(VOWELS)]
            word = "".join(parts) + self.rng.choice(CODAS)
            if self.fresh(word) and word not in RESERVED and not any(part in word for part in UNFIT_PARTS):
                self.take(word)
                return word

    def name(self, words: Sequence[str], share: float) -> str:
        """A proper name: a coined word of one or two syllables, capitalised, followed `share` of the
        time by a word of `words` ("Kessel Reach")."""
        base = self.coin(self.rng.choice((1, 2))).capitalize()
        name = f"{base} {self.rng.choice(words)}" if self.rng.random() < share else base
        self.take(name)
        return name

    def unit(self) -> tuple[str, str]:
        """A coined unit: its symbol (two or three letters, from its name) and its words, the plural
        of its name ("vk", "varks")."""
        while True:
            name = self.coin(1)
            candidates = (name[0] + name[-1], name[:2], name[0] + name[-2:], name[:3])
            symbol = next((s for s in candidates if s.isalpha() and 2 <= len(s) <= 3 and self.fresh(s)
                           and s not in RESERVED), None)
            if symbol is not None and _plural(name) not in RESERVED:
                self.take(symbol, _plural(name))
                return symbol, _plural(name)

    def prefix(self) -> str:
        """An ID prefix: two or three letters in one case, and a separator ("KX/", "vrn-")."""
        while True:
            letters = "".join(self.rng.choice(ID_LETTERS) for _ in range(self.rng.choice((2, 3))))
            if self.fresh(letters) and letters not in RESERVED:
                self.take(letters)
                return (letters.upper() if self.rng.random() < 0.6 else letters) + self.rng.choice(ID_SEPARATORS)


def _label(coiner: _Coiner, qualifiers: Sequence[str], heads: Sequence[str], bare: float) -> str:
    """A field label: `bare` of the time a head alone ("tonnage"), else a qualifier and a head ("gross
    heft") three times in four, or a coined qualifier and a head ("brask weight")."""
    rng = coiner.rng
    roll = rng.random()
    while True:
        head = rng.choice(heads)
        if roll < bare:
            label = head
        elif roll < bare + 0.75 * (1.0 - bare):
            label = f"{rng.choice(qualifiers)} {head}"
        else:
            label = f"{coiner.coin(1)} {head}"
        if coiner.fresh(label):
            coiner.take(label)
            return label


def _draw_units(spec: FieldSpec, coiner: _Coiner) -> tuple[Unit, ...]:
    """A field's units for one instance: a stored unit, then a larger one, a smaller one, or both, in
    a drawn order."""
    rng = coiner.rng
    stored = Unit(*coiner.unit())
    scales = [rng.choice(LARGER_SCALES), rng.choice(SMALLER_SCALES)]
    rng.shuffle(scales)
    scales = scales[:rng.choice((1, 2))]
    return (stored, *(drawn_unit(spec, *coiner.unit(), scale) for scale in scales))


# ---------------------------------------------------------------------------------------- domain


class Invented(Domain):
    """Consignments at depots, worded per instance by a drawn surface (see the module docstring)."""

    name = "invented"
    noun = "consignment"
    plural = "consignments"
    id_label = "waybill"
    source_label = "agency"
    group_label = "depot"

    fields = (PRIMARY, SECONDARY)
    roles = {"primary": PRIMARY.key, "secondary": SECONDARY.key}
    synthetic_window_hours = WINDOW_HOURS

    id_pattern = ID_PATTERN
    id_prefix = ID_PREFIX
    id_answer_type = "invented_id"
    group_answer_type = "invented_group"
    statuses = ("verified", "provisional")
    trap_vocab = TRAP_VOCAB
    chatter = CHATTER
    has_surface = True

    def __init__(self) -> None:
        self.lexicon = self.surface_lexicon()

    # -------------------------------------------------------------------------------- records

    def synthetic(self, rng: random.Random, n: int, compiled_at: int, window_hours: int) -> list[Record]:
        """`n` records of the canonical world (its depots, agencies and IDs, whatever surface this
        domain is bound to): a depot by its weight, one of its agencies, both quantities from the
        depot's distributions, and a time in the window, busiest in the working hours."""
        weights = [depot.weight for depot in DEPOTS]
        taken: set[str] = set()
        records = []
        for _ in range(n):
            depot = rng.choices(DEPOTS, weights=weights, k=1)[0]
            source = rng.choice(depot.sources)
            z1 = rng.gauss(0.0, 1.0)
            z2 = CORRELATION * z1 + math.sqrt(1.0 - CORRELATION ** 2) * rng.gauss(0.0, 1.0)
            values = {spec.key: round(max(FLOORS[spec.key], median * math.exp(sigma * z)), spec.decimals) + 0.0
                      for spec, (median, sigma), z in ((PRIMARY, depot.f1, z1), (SECONDARY, depot.f2, z2))}
            when = _filing_time(rng, compiled_at, window_hours)
            records.append(Record(id=_fresh_id(ID_PREFIX, rng, taken), time=when, lat=None, lon=None, group=depot.name,
                                  source=source, values=values))
        return sorted(records, key=lambda r: (r.time, r.id))

    def sample_weight(self, record: Record) -> float:
        """Heavier consignments make the digest more often: the weight grows as the root of f1."""
        return math.sqrt(record.values["f1"] / 100.0)

    def new_id(self, source: str, rng: random.Random, taken: set[str]) -> str:
        return _fresh_id(self.id_prefix, rng, taken)

    def mimic_id(self, example: str, rng: random.Random, taken: set[str]) -> str:
        return _fresh_id(example[:-ID_DIGITS], rng, taken)

    # ------------------------------------------------------------------------------- surfaces

    def draw_surface(self, rng: random.Random) -> dict[str, Any]:
        """The words of one instance (see the module docstring), every one of them different."""
        coiner = _Coiner(rng)
        if rng.random() < COINED_NOUN_SHARE:
            noun = coiner.coin(rng.choice((1, 2)))
            plural = _plural(noun)
            coiner.take(plural)
        else:
            noun, plural = rng.choice([pair for pair in NOUNS if coiner.fresh(*pair)])
            coiner.take(noun, plural)
        id_label = coiner.pick(ID_LABELS, noun=noun)
        source_label = coiner.pick(SOURCE_LABELS)
        group_label = coiner.pick(GROUP_LABELS)
        labels = {PRIMARY.key: _label(coiner, F1_QUALIFIERS, F1_HEADS, bare=0.25),
                  SECONDARY.key: _label(coiner, F2_QUALIFIERS, F2_HEADS, bare=0.0)}
        units = {spec.key: _draw_units(spec, coiner) for spec in self.fields}
        id_prefix = coiner.prefix()
        groups = {depot.name: coiner.name(PLACE_WORDS, 0.35) for depot in DEPOTS}
        sources = {source: coiner.name(AGENCY_WORDS, 0.5) for source in SOURCES}
        relays = (coiner.name(AGENCY_WORDS, 0.5), coiner.name(AGENCY_WORDS, 0.5), rng.choice(sorted(sources.values())))
        trap_vocab = TrapVocab(
            rereporting_sources=relays,
            duplicate_notes=tuple(note.format(noun=noun) for note in rng.sample(DUPLICATE_NOTES, 2)),
            retraction_reasons=tuple(rng.sample(RETRACTION_REASONS, 3)),
            retraction_notices=tuple(rng.sample(RETRACTION_NOTICES, 2)),
            correction_notices=tuple(rng.sample(CORRECTION_NOTICES, 2)),
        )
        chatter = tuple(rng.sample(CHATTER, CHATTER_LINES))
        return Surface(noun=noun, plural=plural, id_label=id_label, source_label=source_label,
                       group_label=group_label, labels=labels, units=units, id_prefix=id_prefix, groups=groups,
                       sources=sources, trap_vocab=trap_vocab, chatter=chatter).to_dict()

    def surface_lexicon(self) -> Lexicon:
        """The standard wording of the bound words, with a few more wordings of their own, so that
        questions vary as the words do."""
        from ..questions import op_name  # imported here: questions imports the domains

        lexicon = standard_lexicon(self)
        noun, id_label, group = (_literal(word) for word in (self.noun, self.id_label, self.group_label))
        filters: dict[str, tuple[str, ...]] = {}
        questions: dict[str, tuple[str, ...]] = {
            op_name(self, "count"): ("What is the number of separate {records}?",),
            op_name(self, "compare_groups"): (
                f"Comparing only {{records}}: does {{a}} or {{b}} have more of them? Answer \"tie\" if the counts "
                "are equal.",),
        }
        for spec in self.fields:
            label = _literal(spec.label)
            a_label = ("an " if spec.label[:1].lower() in "aeiou" else "a ") + label
            as_unit, rounding = (", in {unit}," if spec.units else ""), (" {rounding}" if spec.units else "")
            more = {
                "at_least": (f"with {a_label} of {{value}} or more",),
                "below": (f"with {a_label} of less than {{value}}",),
                "argmax": (f"Which of the {{records}} has the greatest {label}? Give its {id_label}.",),
                "rank": (f"Which of the {{records}} has the {{nth}}-greatest {label}? Give its {id_label}.",),
                "max": (f"Give the largest {label}{as_unit} found among the {{records}}.{rounding}",),
                "min": (f"Give the smallest {label}{as_unit} found among the {{records}}.{rounding}",),
                "time_of_max": (f"Give the UTC time (HH:MM) at which the {noun} with the greatest {label} among the "
                                "{records} was recorded.",),
            }
            filters.update((op_name(self, kind, spec.key), more[kind]) for kind in spec.filters)
            questions.update((op_name(self, kind, spec.key), more[kind]) for kind in spec.terminals)
        filters[op_name(self, "in_group")] = (f"at the {group} {{group}}",)
        return Lexicon(lexicon.records, lexicon.all_records,
                       {op: wordings + filters.get(op, ()) for op, wordings in lexicon.filters.items()},
                       {op: wordings + questions.get(op, ()) for op, wordings in lexicon.questions.items()})

    def rules_text(self) -> str:
        """The reading rules of every record domain (record_rules), and one more: the glossary is the
        only definition of an invented domain's words, and a value in another unit stands for its
        stored value."""
        rules = record_rules(self)
        number = len(re.findall(r"^\d+\. ", rules, flags=re.MULTILINE)) + 1
        rule = (f"The glossary is the only definition of the words in this digest: what a {self.noun} is, what each "
                "field label means, and every unit with its conversion. A value written in a unit other than its "
                "field's stored unit stands for that value converted into the stored unit and rounded to the decimal "
                "places the glossary gives for the field.")
        wrapped = textwrap.fill(f"{number}. {rule}", width=RULES_WIDTH, subsequent_indent=" " * (len(str(number)) + 2),
                                break_long_words=False, break_on_hyphens=False)
        return rules + wrapped + "\n"


def _fresh_id(prefix: str, rng: random.Random, taken: set[str]) -> str:
    """A new ID: `prefix` and five digits, not in `taken` (and added to it)."""
    while True:
        ref = f"{prefix}{rng.randrange(10 ** (ID_DIGITS - 1), 10 ** ID_DIGITS)}"
        if ref not in taken:
            taken.add(ref)
            return ref


def _filing_time(rng: random.Random, compiled_at: int, window_hours: int) -> int:
    """A record's time in the window ending at `compiled_at`: uniform, but kept only NIGHT_SHARE of the
    time outside the working hours (UTC)."""
    while True:
        when = compiled_at - rng.randrange(60, window_hours * 3600)
        hour = when // 3600 % 24
        if WORKING_HOURS[0] <= hour < WORKING_HOURS[1] or rng.random() < NIGHT_SHARE:
            return when


INVENTED = Invented()
DOMAIN = INVENTED  # what the registry looks up (see evalgen.domains)
