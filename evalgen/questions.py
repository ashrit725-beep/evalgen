"""Question templates and the composer that turns them into concrete, guarded questions.

A *template* is a skeleton: which filters, which final operation. A *question* is a template whose
every parameter (thresholds, radii, units, points, time windows, regions) was drawn by the RNG and
then checked by margin guards, so that rounding in the digest can never flip the correct answer.
No concrete question is ever used twice in a run.

Two pools:
  public  PUBLIC_TEMPLATES below. Treat them as published: a model may have been trained on them.
  new     one "twin" per public template, composed at run time: same final operation and the same
          number of filters, but a different filter combination that is not a public skeleton.
Because the twins mirror the public pool's answer types and filter counts, a large accuracy gap
between the pools points at overfitting to the public skeletons rather than at difficulty.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from .digest import MONTHS_LONG
from .geo import UNIT_WORDS, destination, from_km, to_km
from .models import Event, Question
from .ops import FILTER_OPS, TERMINAL_OPS, answer_spec, distance_km, execute, rows_after_filters, terminal_step

# Margin guards. No event may sit this close to a threshold, so display rounding (0.1 km or mi,
# 0.001 degrees, minute-precision times) can never change a correct answer.
DEPTH_MARGIN_KM, DEPTH_MARGIN_REL = 0.6, 0.03
DISTANCE_MARGIN_KM, DISTANCE_MARGIN_REL = 3.0, 0.06
TIME_MARGIN_S = 120
MAG_GAP = 0.1 - 1e-9  # ranked answers: every gap down to the asked rank must be at least 0.1
NEAREST_GAP_KM, NEAREST_GAP_REL = 3.0, 0.08  # the nearest quake must beat the runner-up clearly
NEAREST_MAX_KM = 500.0  # keeps "nearest" questions local, so the 3% distance tolerance stays tight
RADII = {
    "km": (25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300),
    "mi": (15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200),
}


@dataclass(frozen=True)
class Template:
    template_id: str
    pool: str  # "public" or "new"
    filters: tuple[str, ...]
    terminal: str
    twin_of: str | None = None  # for new templates: the public template whose profile it mirrors

    @property
    def signature(self) -> tuple[tuple[str, ...], str]:
        return tuple(sorted(self.filters)), self.terminal

    def describe(self) -> str:
        return " → ".join(["dedupe", *self.filters, self.terminal])

    def to_dict(self) -> dict:
        return {"template_id": self.template_id, "pool": self.pool, "filters": list(self.filters),
                "terminal": self.terminal, "twin_of": self.twin_of, "skeleton": self.describe()}


PUBLIC_TEMPLATES: tuple[Template, ...] = (
    Template("P01", "public", ("mag_at_least",), "count"),
    Template("P02", "public", ("within_distance",), "count"),
    Template("P03", "public", ("in_region",), "argmax_mag"),
    Template("P04", "public", ("mag_at_least",), "max_depth"),
    Template("P05", "public", ("mag_at_least",), "compare_regions"),
    Template("P06", "public", ("depth_shallower_than", "time_between"), "count"),
    Template("P07", "public", ("in_region",), "nearest_to"),
    Template("P08", "public", ("in_region",), "time_of_largest"),
    Template("P09", "public", ("within_distance",), "rank_mag"),
    Template("P10", "public", ("depth_at_least",), "max_mag"),
    Template("P11", "public", ("time_between",), "min_depth"),
    Template("P12", "public", ("mag_below",), "nearest_distance"),
)

_EXCLUSIVE = (frozenset({"mag_at_least", "mag_below"}), frozenset({"depth_shallower_than", "depth_at_least"}))
# Filter/terminal pairs where the filter can never change the answer (the winner always passes it).
_MONOTONE = {
    ("mag_at_least", "max_mag"), ("mag_at_least", "argmax_mag"), ("mag_at_least", "time_of_largest"),
    ("mag_at_least", "rank_mag"), ("depth_at_least", "max_depth"), ("depth_shallower_than", "min_depth"),
}


def _compatible(filters: Sequence[str], terminal: str) -> bool:
    chosen = set(filters)
    if any(pair <= chosen for pair in _EXCLUSIVE):
        return False
    if any((f, terminal) in _MONOTONE for f in chosen):
        return False
    if terminal == "compare_regions" and "in_region" in chosen:
        return False
    if terminal in ("nearest_to", "nearest_distance") and "within_distance" in chosen:
        return False
    return True


def compose_new_templates(rng: random.Random) -> list[Template]:
    """One 'new this run' twin per public template: same terminal and filter count, new filters."""
    public_signatures = {t.signature for t in PUBLIC_TEMPLATES}
    templates: list[Template] = []
    seen: set = set()
    for base in PUBLIC_TEMPLATES:
        for _ in range(500):
            filters = tuple(rng.sample(FILTER_OPS, len(base.filters)))
            candidate = Template(f"N{base.template_id[1:]}", "new", filters, base.terminal, twin_of=base.template_id)
            if (_compatible(filters, base.terminal) and candidate.signature not in public_signatures
                    and candidate.signature not in seen):
                seen.add(candidate.signature)
                templates.append(candidate)
                break
    return templates


# --------------------------------------------------------------------- drawing parameters


@dataclass(frozen=True)
class Context:
    compiled_at: int
    window_hours: int


def _quantile_pick(values: Sequence[float], rng: random.Random, lo: float = 0.25, hi: float = 0.85) -> float:
    ordered = sorted(values)
    return ordered[int(rng.uniform(lo, hi) * (len(ordered) - 1))]


def _region_counts(events: Sequence[Event]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.region] = counts.get(e.region, 0) + 1
    return sorted(counts.items())


def _draw_point(events: Sequence[Event], rng: random.Random, taken_labels: set[str]) -> dict:
    """A named point near a random quake. Labels are unique within an instance."""
    anchor = rng.choice(events)
    lat, lon = destination(anchor.lat, anchor.lon, rng.uniform(5.0, 60.0), rng.uniform(0.0, 360.0))
    label = f"station {anchor.network.upper()}-{rng.randrange(10, 100)}"
    while label in taken_labels:
        label = f"station {anchor.network.upper()}-{rng.randrange(10, 1000)}"
    return {"lat": round(lat, 3), "lon": round(lon, 3), "label": label}


def _draw_window(rng: random.Random, ctx: Context) -> dict | None:
    total_min = ctx.window_hours * 60
    duration = rng.randint(max(10, total_min // 8), max(20, total_min // 2))
    duration -= duration % 5
    first = -(-(ctx.compiled_at - total_min * 60) // 300) * 300  # first 5-minute mark in the window
    latest = ctx.compiled_at - duration * 60
    if latest <= first:
        return None
    start = first + 300 * rng.randrange(0, max(1, (latest - first) // 300))
    return {"op": "time_between", "start": start, "end": start + duration * 60}


def draw_filter(op: str, events: Sequence[Event], rng: random.Random, ctx: Context,
                taken_labels: set[str]) -> dict | None:
    if op in ("mag_at_least", "mag_below"):
        return {"op": op, "value": round(_quantile_pick([e.mag for e in events], rng), 1)}
    if op in ("depth_shallower_than", "depth_at_least"):
        unit = rng.choice(("km", "mi"))
        depth_km = _quantile_pick([e.depth_km for e in events], rng, 0.25, 0.8) + rng.uniform(-1.5, 1.5)
        value = from_km(depth_km, unit)
        value = float(round(value)) if rng.random() < 0.5 else round(value, 1)
        return {"op": op, "value": value, "unit": unit} if value > 0 else None
    if op == "within_distance":
        unit = rng.choice(("km", "mi"))
        return {"op": op, **_draw_point(events, rng, taken_labels), "radius": rng.choice(RADII[unit]), "unit": unit}
    if op == "time_between":
        return _draw_window(rng, ctx)
    if op == "in_region":
        regions = [r for r, c in _region_counts(events) if c >= 2]
        return {"op": op, "region": rng.choice(regions)} if regions else None
    raise ValueError(f"unknown filter {op!r}")


def draw_terminal(op: str, events: Sequence[Event], rng: random.Random, taken_labels: set[str]) -> list[dict] | None:
    if op in ("count", "argmax_mag", "max_mag", "time_of_largest"):
        return [{"op": op}]
    if op == "rank_mag":
        return [{"op": op, "k": rng.choice((2, 3))}]
    if op in ("max_depth", "min_depth"):
        return [{"op": op}, {"op": "convert", "to": rng.choice(("km", "mi"))}, {"op": "round", "ndigits": 1}]
    if op == "nearest_to":
        return [{"op": op, **_draw_point(events, rng, taken_labels)}]
    if op == "nearest_distance":
        return [{"op": op, **_draw_point(events, rng, taken_labels)}, {"op": "convert", "to": rng.choice(("km", "mi"))},
                {"op": "round", "ndigits": 1}]
    if op == "compare_regions":
        counts = _region_counts(events)
        regions = [r for r, c in counts if c >= 2] or [r for r, _ in counts]
        if len(regions) < 2:
            return None
        a, b = rng.sample(regions, 2)
        return [{"op": op, "a": a, "b": b}]
    raise ValueError(f"unknown terminal {op!r}")


def guards_ok(chain: Sequence[dict], events: Sequence[Event]) -> bool:
    """Reject parameters that sit too close to any event, or answers that are not unique."""
    for step in chain:
        op = step["op"]
        if op in ("depth_shallower_than", "depth_at_least"):
            y = to_km(step["value"], step["unit"])
            margin = max(DEPTH_MARGIN_KM, DEPTH_MARGIN_REL * abs(y))
            if any(abs(e.depth_km - y) < margin for e in events):
                return False
        elif op == "within_distance":
            r = to_km(step["radius"], step["unit"])
            margin = max(DISTANCE_MARGIN_KM, DISTANCE_MARGIN_REL * r)
            if any(abs(distance_km(step, e) - r) < margin for e in events):
                return False
        elif op == "time_between":
            if any(abs(e.time - edge) < TIME_MARGIN_S for e in events for edge in (step["start"], step["end"])):
                return False
    rows = rows_after_filters(chain, events)
    term = terminal_step(chain)
    op = term["op"]
    mags = sorted((e.mag for e in rows), reverse=True)
    if op in ("argmax_mag", "time_of_largest"):
        return bool(rows) and (len(mags) < 2 or mags[0] - mags[1] >= MAG_GAP)
    if op == "rank_mag":  # no ties anywhere above or at the asked rank, so every reading of "k-th" agrees
        k = term["k"]
        top = mags[:k + 1]
        return len(mags) >= k and all(a - b >= MAG_GAP for a, b in zip(top, top[1:]))
    if op in ("nearest_to", "nearest_distance"):
        d = sorted(distance_km(term, e) for e in rows)
        return (bool(d) and d[0] <= NEAREST_MAX_KM
                and (len(d) < 2 or d[1] - d[0] >= max(NEAREST_GAP_KM, NEAREST_GAP_REL * d[0])))
    if op in ("max_mag", "max_depth", "min_depth"):
        return bool(rows)
    if op == "compare_regions":
        return any(e.region in (term["a"], term["b"]) for e in rows)
    return True


def _interesting(chain: Sequence[dict], events: Sequence[Event], value: Any, rng: random.Random) -> bool:
    op = terminal_step(chain)["op"]
    if op == "count":
        return 1 <= value < len(events)
    if op == "compare_regions":
        return value != "tie" or rng.random() < 0.25  # ties are legitimate but kept rare
    return value is not None


def fingerprint(chain: Sequence[dict]) -> str:
    """Identity of a concrete question. Station labels are cosmetic and left out, and the two regions
    of a comparison are unordered ("A or B?" is the same question as "B or A?")."""
    steps = []
    for step in chain:
        canonical = {k: v for k, v in step.items() if k != "label"}
        if canonical["op"] == "compare_regions":
            canonical["a"], canonical["b"] = sorted((canonical["a"], canonical["b"]))
        steps.append(canonical)
    return json.dumps(steps, sort_keys=True)


def instantiate(template: Template, events: Sequence[Event], rng: random.Random, ctx: Context, *,
                used_chains: set[str], taken_labels: set[str], attempts: int = 40) -> Question | None:
    """Draw parameters for a template until every guard passes and the question is new to the run."""
    for _ in range(attempts):
        chain: list[dict] = [{"op": "dedupe"}]
        complete = True
        for op in template.filters:
            step = draw_filter(op, events, rng, ctx, taken_labels)
            if step is None:
                complete = False
                break
            chain.append(step)
        tail = draw_terminal(template.terminal, events, rng, taken_labels) if complete else None
        if tail is None:
            continue
        chain.extend(tail)
        if fingerprint(chain) in used_chains or not guards_ok(chain, events):
            continue
        value = execute(chain, events)
        if value is None or not _interesting(chain, events, value, rng):
            continue
        answer_type, unit = answer_spec(chain)
        used_chains.add(fingerprint(chain))
        taken_labels.update(step["label"] for step in chain if "label" in step)
        return Question(qid="", template_id=template.template_id, pool=template.pool, chain=chain,
                        answer_type=answer_type, unit=unit,
                        evidence=[e.id for e in rows_after_filters(chain, events)])
    return None


def compose_questions(events: Sequence[Event], public: Sequence[Template], new: Sequence[Template],
                      rng: random.Random, ctx: Context, n_questions: int, used_chains: set[str],
                      usage: dict[str, int]) -> list[Question]:
    """3-5 questions per instance, balanced between the public and the new pool.

    `usage` counts how often each template has been used so far in the run. A template may only run
    at most one question ahead of its twin, so the two pools keep the same mix of final operations
    even when run-wide uniqueness exhausts a template with few possible parameters."""
    twin = {t.template_id: t.twin_of for t in new if t.twin_of}
    twin.update({public_id: new_id for new_id, public_id in list(twin.items())})
    pools = ["public", "new"] * (n_questions // 2) + ([rng.choice(("public", "new"))] if n_questions % 2 else [])
    rng.shuffle(pools)
    questions: list[Question] = []
    used_templates: list[str] = []
    taken_labels: set[str] = set()
    for pool in pools + ["public", "new"] * 2:  # spare slots in case some templates do not fit
        if len(questions) >= n_questions:
            break
        candidates = [t for t in (public if pool == "public" else new)
                      if t.template_id not in used_templates
                      and usage.get(t.template_id, 0) <= usage.get(twin.get(t.template_id, ""), 1 << 30)]
        rng.shuffle(candidates)
        for template in candidates[:6]:
            question = instantiate(template, events, rng, ctx, used_chains=used_chains, taken_labels=taken_labels)
            if question is not None:
                question.qid = f"Q{len(questions) + 1}"
                questions.append(question)
                used_templates.append(template.template_id)
                usage[template.template_id] = usage.get(template.template_id, 0) + 1
                break
    return questions


# -------------------------------------------------------------------------------- phrasing


def fmt_num(x: Any) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:g}" if isinstance(x, float) else str(x)


def _qty(value: float, unit: str, alt_units: bool) -> str:
    """'50 miles', or in a metamorphic variant the same quantity in the other unit: '80.5 km'."""
    if alt_units:
        other = "mi" if unit == "km" else "km"
        return f"{from_km(to_km(value, unit), other):.1f} {UNIT_WORDS[other]}"
    return f"{fmt_num(value)} {UNIT_WORDS[unit]}"


def _place(step: dict, rng: random.Random) -> str:
    lat, lon = step["lat"], step["lon"]
    if rng.random() < 0.5:
        return f"{step['label']} ({abs(lat):.3f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.3f}°{'E' if lon >= 0 else 'W'})"
    return f"{step['label']} at ({lat:.3f}, {lon:.3f})"


def _when(ts: int, style: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if style == 0:
        return f"{dt:%Y-%m-%d %H:%M} UTC"
    return f"{dt:%H:%M} UTC on {dt.day} {MONTHS_LONG[dt.month - 1]} {dt.year}"


def _filter_phrase(step: dict, rng: random.Random, alt_units: bool) -> str:
    op = step["op"]
    if op == "mag_at_least":
        v = fmt_num(step["value"])
        return rng.choice((f"of magnitude {v} or greater", f"with magnitude at least {v}", f"of magnitude {v} or more"))
    if op == "mag_below":
        v = fmt_num(step["value"])
        return rng.choice((f"with magnitude below {v}", f"smaller than magnitude {v}"))
    if op == "depth_shallower_than":
        q = _qty(step["value"], step["unit"], alt_units)
        return rng.choice((f"shallower than {q}", f"with a depth of less than {q}"))
    if op == "depth_at_least":
        q = _qty(step["value"], step["unit"], alt_units)
        return rng.choice((f"at least {q} deep", f"with a depth of {q} or more"))
    if op == "within_distance":
        q = _qty(step["radius"], step["unit"], alt_units)
        p = _place(step, rng)
        return rng.choice((f"within {q} of {p}", f"no more than {q} from {p}"))
    if op == "time_between":
        style = rng.randrange(2)
        a, b = _when(step["start"], style), _when(step["end"], style)
        return rng.choice((f"that occurred between {a} and {b}", f"that were recorded from {a} up to {b}"))
    if op == "in_region":
        r = step["region"]
        return rng.choice((f"in {r}", f"located in {r}", f"whose region is listed as {r}"))
    raise ValueError(f"unknown filter {op!r}")


def _join(clauses: list[str]) -> str:
    if len(clauses) <= 1:
        return "".join(clauses)
    return ", ".join(clauses[:-1]) + " and " + clauses[-1]


ORDINALS = {2: "second", 3: "third"}


def phrase_question(question: Question, rng: random.Random, alt_units: bool = False) -> str:
    """English wording for a question. Different RNG draws give paraphrases with the same meaning."""
    clauses = [_filter_phrase(s, rng, alt_units) for s in question.chain if s["op"] in FILTER_OPS]
    clauses.sort(key=lambda c: c.startswith(("that ", "whose ")))  # relative clauses read best last
    group = f"earthquakes {_join(clauses)}" if clauses else "earthquakes in the digest"
    term = terminal_step(question.chain)
    op = term["op"]
    unit = UNIT_WORDS.get(question.unit or "", "")
    if op == "count":
        options = (f"How many distinct {group} are there?", f"Count the distinct {group}.",
                   f"What is the number of separate {group}?")
    elif op == "argmax_mag":
        options = (f"Among the {group}, which one had the largest magnitude? Give its event ID.",
                   f"What is the event ID of the strongest of the {group}?")
    elif op == "rank_mag":
        nth = ORDINALS[term["k"]]
        options = (f"Among the {group}, which one has the {nth}-largest magnitude? Give its event ID.",
                   f"Rank the {group} by magnitude, largest first: which event is {nth}? Give its event ID.")
    elif op == "max_mag":
        options = (f"What was the largest magnitude among the {group}?",
                   f"Among the {group}, what was the highest magnitude recorded?")
    elif op == "max_depth":
        options = (f"What was the greatest depth, in {unit}, among the {group}? Round to 1 decimal place.",
                   f"Among the {group}, how deep was the deepest one, in {unit}? Round to 1 decimal place.")
    elif op == "min_depth":
        options = (f"What was the smallest depth, in {unit}, among the {group}? Round to 1 decimal place.",
                   f"Among the {group}, how deep was the shallowest one, in {unit}? Round to 1 decimal place.")
    elif op == "nearest_to":
        p = _place(term, rng)
        options = (f"Which of the {group} was closest to {p}? Give its event ID.",
                   f"Among the {group}, give the event ID of the one nearest to {p}.")
    elif op == "nearest_distance":
        p = _place(term, rng)
        options = (f"How far from {p}, in {unit}, was the closest of the {group}? Round to 1 decimal place.",
                   f"Among the {group}, what is the distance in {unit} from {p} to the nearest one? Round to 1 decimal place.")
    elif op == "compare_regions":
        a, b = term["a"], term["b"]
        options = (f"Counting only {group}, which region has more of them: {a} or {b}? Answer \"tie\" if the counts are equal.",
                   f"Which region had more {group}: {a} or {b}? Say \"tie\" if they are equal.")
    elif op == "time_of_largest":
        options = (f"At what time (UTC, HH:MM) did the largest of the {group} occur?",
                   f"When, in UTC (HH:MM), was the strongest of the {group} recorded?")
    else:
        raise ValueError(f"unknown terminal {op!r}")
    return rng.choice(options)
