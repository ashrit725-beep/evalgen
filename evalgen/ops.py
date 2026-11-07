"""Question operations and the code-only answer key.

A question's meaning is an *operation chain*, a list of small steps such as

    [{"op": "dedupe"},
     {"op": "mag_at_least", "value": 3.1},
     {"op": "in_region", "region": "Nevada"},
     {"op": "count"}]

The answer key is simply ``execute(chain, snapshot_events)``: plain Python over the frozen
snapshot, with no LLM and no text parsing. The same executor also powers the reference solvers
(fed with what they parsed from the digest text) and trap grading (fed with a digest read the
wrong way, see ``rules.Mishandle``).

Chains always have the shape: dedupe, zero or more filters, one terminal, optional post-steps
(convert, round).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .geo import from_km, haversine_km, to_km
from .models import Event, Notice, Report
from .rules import Mishandle, naive_events, resolve
from .util import utc_iso

EPS = 1e-9

FILTER_OPS = (
    "mag_at_least",
    "mag_below",
    "depth_shallower_than",
    "depth_at_least",
    "within_distance",
    "time_between",
    "in_region",
)
TERMINAL_OPS = (
    "count",
    "argmax_mag",
    "rank_mag",
    "max_mag",
    "max_depth",
    "min_depth",
    "nearest_to",
    "nearest_distance",
    "compare_regions",
    "time_of_largest",
)
ANSWER_TYPES = {
    "count": "count",
    "argmax_mag": "event_id",
    "rank_mag": "event_id",
    "nearest_to": "event_id",
    "max_mag": "magnitude",
    "max_depth": "depth",
    "min_depth": "depth",
    "nearest_distance": "distance",
    "compare_regions": "region",
    "time_of_largest": "utc_time",
}


@dataclass
class DigestData:
    """Reports and notices as read from a digest, before the reading rules are applied."""

    reports: list[Report]
    notices: list[Notice]


def distance_km(step: dict, event: Event) -> float:
    return haversine_km(step["lat"], step["lon"], event.lat, event.lon)


def passes(step: dict, e: Event) -> bool:
    op = step["op"]
    if op == "mag_at_least":
        return e.mag >= step["value"] - EPS
    if op == "mag_below":
        return e.mag < step["value"] - EPS
    if op == "depth_shallower_than":
        return e.depth_km < to_km(step["value"], step["unit"]) - EPS
    if op == "depth_at_least":
        return e.depth_km >= to_km(step["value"], step["unit"]) - EPS
    if op == "within_distance":
        return distance_km(step, e) <= to_km(step["radius"], step["unit"]) + EPS
    if op == "time_between":
        return step["start"] <= e.time < step["end"]
    if op == "in_region":
        return e.region == step["region"]
    raise ValueError(f"not a filter op: {op!r}")


def by_magnitude(rows: Iterable[Event]) -> list[Event]:
    """Largest first; ties broken by earlier time, then ID (the guards make ties impossible)."""
    return sorted(rows, key=lambda e: (-e.mag, e.time, e.id))


def terminal(step: dict, rows: Sequence[Event]) -> Any:
    op = step["op"]
    if op == "count":
        return len(rows)
    if op == "compare_regions":
        a = sum(1 for e in rows if e.region == step["a"])
        b = sum(1 for e in rows if e.region == step["b"])
        return step["a"] if a > b else step["b"] if b > a else "tie"
    if not rows:
        return None
    if op == "argmax_mag":
        return by_magnitude(rows)[0].id
    if op == "rank_mag":
        ordered = by_magnitude(rows)
        return ordered[step["k"] - 1].id if len(ordered) >= step["k"] else None
    if op == "max_mag":
        return max(e.mag for e in rows)
    if op == "max_depth":
        return max(e.depth_km for e in rows)
    if op == "min_depth":
        return min(e.depth_km for e in rows)
    if op == "nearest_to":
        return min(rows, key=lambda e: (distance_km(step, e), e.id)).id
    if op == "nearest_distance":
        return min(distance_km(step, e) for e in rows)
    if op == "time_of_largest":  # UTC, truncated to the minute
        return datetime.fromtimestamp(by_magnitude(rows)[0].time, tz=timezone.utc).strftime("%H:%M")
    raise ValueError(f"not a terminal op: {op!r}")


def _dedupe(rows: Any, mishandle: Mishandle | None) -> list[Event]:
    if isinstance(rows, DigestData):
        return resolve(rows.reports, rows.notices, mishandle)
    return list(rows)  # snapshot events are already one row per earthquake


def _as_events(rows: Any) -> list[Event]:
    if isinstance(rows, DigestData):  # dedupe was skipped: every report line counts
        return naive_events(rows.reports)
    return list(rows)


def rows_after_filters(chain: Sequence[dict], data: Any, *, mishandle: Mishandle | None = None,
                       skip: Sequence[str] = ()) -> list[Event]:
    """The events that survive dedupe and every filter (the question's 'evidence')."""
    rows = data
    for step in chain:
        op = step["op"]
        if op in skip:
            continue
        if op == "dedupe":
            rows = _dedupe(rows, mishandle)
        elif op in FILTER_OPS:
            rows = [e for e in _as_events(rows) if passes(step, e)]
    return _as_events(rows)


def execute(chain: Sequence[dict], data: Any, *, mishandle: Mishandle | None = None,
            skip: Sequence[str] = ()) -> Any:
    """Run an operation chain over snapshot events (answer key) or parsed digest data (solvers)."""
    rows = rows_after_filters(chain, data, mishandle=mishandle, skip=skip)
    value: Any = None
    for step in chain:
        op = step["op"]
        if op in skip or op == "dedupe" or op in FILTER_OPS:
            continue
        if op in TERMINAL_OPS:
            value = terminal(step, rows)
        elif op == "convert":
            value = None if value is None else from_km(value, step["to"])
        elif op == "round":
            value = None if value is None else round(value, step["ndigits"])
        else:
            raise ValueError(f"unknown op: {op!r}")
    return value


def terminal_step(chain: Sequence[dict]) -> dict:
    return next(s for s in chain if s["op"] in TERMINAL_OPS)


def answer_spec(chain: Sequence[dict]) -> tuple[str, str | None]:
    """(answer_type, unit) of a chain, e.g. ("depth", "mi")."""
    answer_type = ANSWER_TYPES[terminal_step(chain)["op"]]
    unit = next((s["to"] for s in chain if s["op"] == "convert"), None)
    return answer_type, unit


def describe_step(step: dict) -> str:
    op = step["op"]
    if op == "mag_at_least":
        return f"mag ≥ {step['value']:g}"
    if op == "mag_below":
        return f"mag < {step['value']:g}"
    if op == "depth_shallower_than":
        return f"depth < {step['value']:g} {step['unit']}"
    if op == "depth_at_least":
        return f"depth ≥ {step['value']:g} {step['unit']}"
    if op == "within_distance":
        return f"within {step['radius']:g} {step['unit']} of ({step['lat']:.3f}, {step['lon']:.3f})"
    if op == "time_between":
        return f"time in [{utc_iso(step['start'])}, {utc_iso(step['end'])})"
    if op == "in_region":
        return f"region = {step['region']}"
    if op == "rank_mag":
        return f"rank_mag(k={step['k']})"
    if op in ("nearest_to", "nearest_distance"):
        return f"{op}({step['lat']:.3f}, {step['lon']:.3f})"
    if op == "compare_regions":
        return f"compare_regions({step['a']} vs {step['b']})"
    if op == "convert":
        return f"convert → {step['to']}"
    if op == "round":
        return f"round({step['ndigits']})"
    return op


def describe(chain: Sequence[dict]) -> str:
    return " → ".join(describe_step(s) for s in chain)
