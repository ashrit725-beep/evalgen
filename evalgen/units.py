"""Units, and the exact codec of the numbers written in them.

A field's values are stored in its canonical unit, rounded to the field's places. A digest may print a
value in any unit of its field, and a reader converts the printed number back and rounds it to the
stored places. Converting back is rarely exact by itself (most tenths of a degree Celsius have no
decimal form in Fahrenheit that converts back to them), so the codec is a pair:

    encode  a stored value written in a unit, with the fewest decimals that decode back to it exactly
    decode  the stored value that a number written in a unit stands for

Every genre prints numbers with ``encode`` and every parser reads them with ``decode``, so the
text-only oracle recovers each stored value exactly, in affine units (°F) as in scaled ones. A number
without a unit (a magnitude, a latitude) uses the same codec with no unit.

    Unit          a unit a value is printed or asked in, and its affine map to the canonical unit
    KM, MI        kilometres and miles: LENGTH, the units of depths, radii and distances
    ExactValues   encode and decode at the stored places of one field (FieldSpec, GlossaryField)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .geo import KM_PER_MILE

MAX_PLACES = 15  # a value that needs more decimals than this in some unit cannot be printed in it


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
MI = Unit("mi", "miles", KM_PER_MILE)
LENGTH = (KM, MI)  # depths, radii and distances: stored in km, printed in km or miles


def find_unit(units: Iterable[Unit], symbol: str) -> Unit:
    """The unit printed as `symbol` among `units`."""
    for unit in units:
        if unit.symbol == symbol:
            return unit
    raise ValueError(f"unknown unit {symbol!r}")


def decode(text: str, decimals: int, unit: Unit | None = None) -> float:
    """The stored value that the number `text`, written in `unit` (None: the canonical unit or no unit),
    stands for: the number converted to the canonical unit and rounded to the stored `decimals`."""
    printed = float(text)
    return round(printed if unit is None else unit.to_canonical(printed), decimals)


def encode(value: float, decimals: int, unit: Unit | None = None) -> str:
    """A value stored to `decimals` places, written in `unit` (None: the canonical unit or no unit) with
    the fewest decimals that decode back to it: "21" or "69.8" for 21.0 °C, never in exponent notation
    and never as "-0". A value with more places than are stored has no such form (ValueError)."""
    shown = value if unit is None else unit.from_canonical(value)
    for places in range(MAX_PLACES + 1):
        text = f"{shown:.{places}f}"
        if decode(text, decimals, unit) == value:
            return text.lstrip("-") if float(text) == 0 else text
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
