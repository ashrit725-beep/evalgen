"""Units, and the exact codec of the numbers written in them.

A field's values are stored in its canonical unit, rounded to the field's places. A digest may print a
value in any unit of its field, and a reader converts the printed number back and rounds it to the
stored places. Converting back is rarely exact by itself (most tenths of a degree Celsius have no
decimal form in Fahrenheit that converts back to them), so the codec is a pair:

    encode  a stored value written in a unit, with the fewest decimals that decode back to it exactly
    decode  the stored value that a number written in a unit stands for

Both work in exact decimal arithmetic, with the scale and offset exactly as a glossary prints them
(``plain_number``), as a careful reader with pencil and paper would, never in binary floating point.
And encode never writes a number whose conversion falls exactly halfway between two stored values
(273.9 K is 0.75 °C): it takes more decimals instead (273.85 K for 0.7 °C), so every printed number
lies strictly closer to its stored value than to any other, and a reader recovers it whichever way
they round a half. Every genre prints numbers with ``encode`` and every parser reads them with
``decode``, so the text-only oracle recovers each stored value exactly, in affine units (°F) as in
scaled ones, and so does any reader who converts as the glossary says. A number without a unit (a
magnitude, a latitude) uses the same codec with no unit.

    Unit          a unit a value is printed or asked in, and its affine map to the canonical unit
    KM, MI        kilometres and miles: LENGTH, the units of depths, radii and distances
    ExactValues   encode and decode at the stored places of one field (FieldSpec, GlossaryField)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal
from typing import Any, Iterable, Mapping

from .geo import KM_PER_MILE

MAX_PLACES = 15  # a value that needs more decimals than this in some unit cannot be printed in it
# Decimal arithmetic wide enough that no conversion of a printed number is ever rounded: a number of up
# to MAX_PLACES decimals times a scale of up to 17 significant digits, plus an offset.
EXACT = Context(prec=80, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class Unit:
    """A unit a field's value can be printed or asked in: canonical = value × scale + offset.

    Scale and offset are relative to the field's canonical unit, so one symbol can mean different
    things in different fields: km is the canonical unit of a depth, but 1 / 149597870.7 of the
    canonical au of a miss distance. The offset makes the map affine, as °F needs. A unit whose values
    are on another scale than the field's stored ones (inches of mercury beside hectopascals) can set
    its own precision, scoring tolerance and threshold step; the others use the field's (a tolerance
    converted into the unit, see FieldSpec.unit_tolerances)."""

    symbol: str  # printed after a number: "km", "mi", "°F"
    words: str  # written in a question: "km", "miles", "degrees Fahrenheit"
    scale: float = 1.0
    offset: float = 0.0
    decimals: int | None = None  # places of answers and restated thresholds in this unit (None: the field's)
    tolerance: tuple[float, float] | None = None  # (abs, rel) of answers in this unit (None: the field's)
    step: float | None = None  # thresholds asked in this unit are rounded to it (None: the field's threshold_step)
    one: str | None = None  # the words after a number of exactly 1: "mile", "degree Celsius" (None: `words`)

    def words_for(self, number: str) -> str:
        """The words after `number` as a question writes it: `one` after 1 or -1 ("1 degree Celsius"),
        else `words` ("1.5 degrees Celsius", "0 knots")."""
        return self.one if self.one is not None and number.lstrip("-") == "1" else self.words

    def to_canonical(self, value: float) -> float:
        return value * self.scale + self.offset

    def from_canonical(self, value: float) -> float:
        return (value - self.offset) / self.scale

    def places(self, default: int) -> int:
        """Decimal places of a value written in this unit: its own, or else `default` (the field's)."""
        return default if self.decimals is None else self.decimals

    def for_readers(self) -> Unit:
        """The unit as a reader of a digest knows it (a glossary prints it): its symbol, its words and
        its conversion, without the precision, tolerance and step that only the generator uses."""
        return Unit(self.symbol, self.words, self.scale, self.offset)

    def to_dict(self) -> dict[str, Any]:
        """The unit as JSON (a drawn surface stores its units), without the attributes left at default."""
        defaults = Unit("", "")
        return {name: list(value) if isinstance(value, tuple) else value for name, value in asdict(self).items()
                if name in ("symbol", "words") or value != getattr(defaults, name)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Unit:
        tolerance = d.get("tolerance")
        return cls(**{**d, "tolerance": None if tolerance is None else tuple(tolerance)})


KM = Unit("km", "km")
MI = Unit("mi", "miles", KM_PER_MILE, one="mile")
LENGTH = (KM, MI)  # depths, radii and distances: stored in km, printed in km or miles


def find_unit(units: Iterable[Unit], symbol: str) -> Unit:
    """The unit printed as `symbol` among `units`."""
    for unit in units:
        if unit.symbol == symbol:
            return unit
    raise ValueError(f"unknown unit {symbol!r}")


def exact_canonical(text: str, unit: Unit | None = None) -> Decimal:
    """The number `text`, written in `unit` (None: the canonical unit or no unit), converted exactly into
    the canonical unit: text × scale + offset in decimal arithmetic, with the scale and the offset as a
    glossary prints them (plain_number writes a float's shortest exact decimal, its repr)."""
    number = Decimal(text)
    if unit is None:
        return number
    return EXACT.add(EXACT.multiply(number, Decimal(repr(unit.scale))), Decimal(repr(unit.offset)))


def _stored(exact: Decimal, decimals: int) -> Decimal:
    """An exact canonical value rounded to the stored places (halves to even, which encode never prints)."""
    return exact.quantize(Decimal(1).scaleb(-decimals), context=EXACT)


def decode(text: str, decimals: int, unit: Unit | None = None) -> float:
    """The stored value that the number `text`, written in `unit` (None: the canonical unit or no unit),
    stands for: the number converted exactly to the canonical unit and rounded to the stored `decimals`."""
    return float(_stored(exact_canonical(text, unit), decimals)) + 0.0  # + 0.0: never -0.0


def encode(value: float, decimals: int, unit: Unit | None = None) -> str:
    """A value stored to `decimals` places, written in `unit` (None: the canonical unit or no unit) with
    the fewest decimals that decode back to it, and whose exact conversion is strictly nearer to it than
    to any other stored value: "21" or "69.8" for 21.0 °C, "273.85" (not "273.9", which converts to
    0.75 exactly) for 0.7 °C in kelvins; never in exponent notation and never as "-0". A value with
    more places than are stored has no such form (ValueError)."""
    shown = value if unit is None else unit.from_canonical(value)
    half = Decimal(5).scaleb(-decimals - 1)
    for places in range(MAX_PLACES + 1):
        text = f"{shown:.{places}f}"
        exact = exact_canonical(text, unit)
        stored = _stored(exact, decimals)
        if float(stored) == value and EXACT.abs(EXACT.subtract(exact, stored)) != half:
            return text.lstrip("-") if Decimal(text) == 0 else text
    raise ValueError(f"{value!r} cannot be printed in {unit.symbol if unit else 'its canonical unit'} so that it "
                     f"reads back exactly: is it stored to {decimals} decimal places?")


class ExactValues:
    """The codec at the stored places of one field: FieldSpec, and GlossaryField for a reader that
    knows only the prompt."""

    decimals: int  # places of stored values, in the canonical unit

    def decode(self, text: str, unit: Unit | None = None) -> float:
        """The stored value that the number `text`, written in `unit`, stands for (see ``decode``)."""
        return decode(text, self.decimals, unit)

    def encode(self, value: float, unit: Unit | None = None) -> str:
        """A stored value written in `unit` with the fewest decimals that decode back to it (``encode``)."""
        return encode(value, self.decimals, unit)
