"""Question templates and the composer that turns them into concrete, guarded questions.

A *template* is a skeleton: which filters, which final operation. A *question* is a template whose
every parameter (thresholds, radii, units, points, time windows, groups) was drawn by the RNG and
then checked by margin guards, so that rounding in the digest can never flip the correct answer.
No concrete question is ever used twice in a run.

Two pools:
  public  P01-P12 below. Treat them as published: a model may have been trained on them.
  new     one "twin" per public template, composed at run time: same final operation and the same
          number of filters, but a different filter combination that is not a public skeleton.
Because the twins mirror the public pool's answer types and filter counts, a large accuracy gap
between the pools points at overfitting to the public skeletons rather than at difficulty.

Templates are role-based. P01-P12 are written for the fields a domain puts in the *primary* and
*secondary* roles (earthquake magnitude and depth, see ``Domain.roles``) and take the domain's own op
names, so the earthquake templates are exactly the 0.2.0 ones; a domain whose records have no
location has no spatial templates (nor their twins). Parameters, guards and wording come from the
domain: its field specs (quantiles, jitter, unit steps, margins, rank gaps), its spatial spec (radii,
distance guards) and its lexicon (its own, or ``standard_lexicon``); time windows and their guard
scale with the length of the instance's window (Context). Every function takes the domain as a
keyword that defaults to earthquakes, which draws, guards and words exactly as 0.2.0 did.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .digest import MONTHS_LONG
from .domains.base import Domain, FieldSpec, ResolvedRecord, Spatial
from .domains.earthquakes import EARTHQUAKES
from .geo import destination
from .models import Question
from .ops import (ALIASES, Step, answer_spec, canonical_threshold, distance_km, execute, role, rows_after_filters,
                  terminal_step)
from .units import LENGTH, Unit, find_unit
from .util import plain_number

TIME_MARGIN_S = 120  # no record within 2 minutes of a time window's ends (digests print times to the minute)
TIME_STEP_S = 300  # a time window's ends lie on 5-minute marks
# Time windows over a longer span scale both, per day (see Context): their ends lie on the first of
# these marks that is at least 5 minutes a day.
TIME_STEPS_S = (300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)
RANKS = (2, 3)  # rank questions ask for the second or the third largest
FLOAT_SLACK = 1e-9  # rank gaps: values exactly `rank_gap` apart pass despite float error
SPATIAL_KINDS = ("within_distance", "nearest_to", "nearest_distance")
TOP_KINDS = ("max", "argmax", "rank", "time_of_max")  # terminals that answer from a field's largest values
BOTTOM_KINDS = ("min",)  # terminals that answer from its smallest values


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


# --------------------------------------------------------------------------------- op names
# Templates name each operation by its op name: a domain's own name for a kind on a field (the 0.2.0
# names for earthquakes, e.g. "mag_at_least"), else "kind(field)" ("at_least(temp)"), or the bare kind
# for a kind that reads no field ("count", "time_between").


def op_name(domain: Domain, kind: str, field: str | None = None) -> str:
    """The op name of a kind on a field in a domain: ('at_least', 'mag') is 'mag_at_least' for earthquakes."""
    named = domain.op_names.get((kind, field))
    if named is not None:
        return named
    return f"{kind}({field})" if field else kind


def parse_op(name: str) -> tuple[str, str | None]:
    """(kind, field) of an op name: 'mag_at_least' -> ('at_least', 'mag'), 'count' -> ('count', None)."""
    alias = ALIASES.get(name)
    if alias is not None:
        return alias.kind, alias.field
    kind, _, field = name.partition("(")
    return kind, field.rstrip(")") or None


def step_op(step: Mapping[str, Any]) -> str:
    """The op name of a chain step."""
    return f"{step['op']}({step['field']})" if "field" in step else step["op"]


def _step(name: str, **params: Any) -> dict:
    """A chain step for an op name. A 0.2.0 name keeps its 0.2.0 parameter keys ('region' for a
    group); a generic kind stores its field after the op: {"op": "at_least", "field": "temp", ...}."""
    alias = ALIASES.get(name)
    if alias is not None:
        return {"op": name, **{alias.keys.get(key, key): value for key, value in params.items()}}
    kind, field = parse_op(name)
    return {"op": kind, **({"field": field} if field else {}), **params}


# -------------------------------------------------------------------------------- templates

# P01-P12 in role terms: "(primary)" and "(secondary)" stand for the fields in those roles.
ROLE_TEMPLATES: tuple[Template, ...] = (
    Template("P01", "public", ("at_least(primary)",), "count"),
    Template("P02", "public", ("within_distance",), "count"),
    Template("P03", "public", ("in_group",), "argmax(primary)"),
    Template("P04", "public", ("at_least(primary)",), "max(secondary)"),
    Template("P05", "public", ("at_least(primary)",), "compare_groups"),
    Template("P06", "public", ("below(secondary)", "time_between"), "count"),
    Template("P07", "public", ("in_group",), "nearest_to"),
    Template("P08", "public", ("in_group",), "time_of_max(primary)"),
    Template("P09", "public", ("within_distance",), "rank(primary)"),
    Template("P10", "public", ("at_least(secondary)",), "max(primary)"),
    Template("P11", "public", ("time_between",), "min(secondary)"),
    Template("P12", "public", ("below(primary)",), "nearest_distance"),
)


def _bind(token: str, domain: Domain) -> str:
    """A role template's op on a domain: 'at_least(primary)' is 'mag_at_least' for earthquakes."""
    kind, role_name = parse_op(token)
    if role_name is None:
        return op_name(domain, kind)
    spec = domain.field(domain.roles[role_name])
    if kind not in spec.filters + spec.terminals:
        raise ValueError(f"{domain.name}: its {role_name} field {spec.key!r} has no {kind!r} operation")
    return op_name(domain, kind, spec.key)


def public_templates(domain: Domain = EARTHQUAKES) -> tuple[Template, ...]:
    """P01-P12 on a domain, under its op names. Without locations, the spatial templates (and so
    their twins) are left out."""
    out = []
    for t in ROLE_TEMPLATES:
        if not domain.has_location and any(parse_op(op)[0] in SPATIAL_KINDS for op in (*t.filters, t.terminal)):
            continue
        out.append(replace(t, filters=tuple(_bind(op, domain) for op in t.filters), terminal=_bind(t.terminal, domain)))
    return tuple(out)


def filter_ops(domain: Domain = EARTHQUAKES) -> tuple[str, ...]:
    """Every filter a twin may use, in the order twins sample them: each field's threshold filters in
    its spec's order, then the distance, time and group filters (ops.FILTER_OPS for earthquakes)."""
    names = [op_name(domain, kind, spec.key) for spec in domain.fields for kind in spec.filters]
    if domain.has_location:
        names.append(op_name(domain, "within_distance"))
    return (*names, op_name(domain, "time_between"), op_name(domain, "in_group"))


def exclusive_pairs(domain: Domain = EARTHQUAKES) -> frozenset[frozenset[str]]:
    """Filters a twin never combines: a lower and an upper threshold on one field."""
    return frozenset(frozenset(op_name(domain, kind, spec.key) for kind in ("at_least", "below"))
                     for spec in domain.fields if {"at_least", "below"} <= set(spec.filters))


def monotone_pairs(domain: Domain = EARTHQUAKES) -> frozenset[tuple[str, str]]:
    """(filter, terminal) pairs where the filter can never change the answer: the record with the
    largest value passes every lower threshold on that field, and the smallest every upper one."""
    pairs: set[tuple[str, str]] = set()
    for spec in domain.fields:
        for kind, ends in (("at_least", TOP_KINDS), ("below", BOTTOM_KINDS)):
            if kind in spec.filters:
                pairs.update((op_name(domain, kind, spec.key), op_name(domain, end, spec.key))
                             for end in ends if end in spec.terminals)
    return frozenset(pairs)


# The 0.2.0 module-level names, for earthquakes (the pipeline and the tests import them).
PUBLIC_TEMPLATES = public_templates(EARTHQUAKES)
_EXCLUSIVE = exclusive_pairs(EARTHQUAKES)
_MONOTONE = monotone_pairs(EARTHQUAKES)
NEAREST_MAX_KM = EARTHQUAKES.spatial.nearest_max_km


def _compatible(filters: Sequence[str], terminal: str, exclusive: frozenset[frozenset[str]] = _EXCLUSIVE,
                monotone: frozenset[tuple[str, str]] = _MONOTONE) -> bool:
    chosen = set(filters)
    if any(pair <= chosen for pair in exclusive):
        return False
    if any((f, terminal) in monotone for f in chosen):
        return False
    kinds = {parse_op(f)[0] for f in chosen}
    end = parse_op(terminal)[0]
    if end == "compare_groups" and "in_group" in kinds:
        return False
    if end in ("nearest_to", "nearest_distance") and "within_distance" in kinds:  # one point per chain
        return False
    return True


def compose_new_templates(rng: random.Random, domain: Domain = EARTHQUAKES) -> list[Template]:
    """One 'new this run' twin per public template: same terminal and filter count, new filters."""
    public = public_templates(domain)
    population = filter_ops(domain)
    exclusive, monotone = exclusive_pairs(domain), monotone_pairs(domain)
    public_signatures = {t.signature for t in public}
    templates: list[Template] = []
    seen: set = set()
    for base in public:
        for _ in range(500):
            filters = tuple(rng.sample(population, len(base.filters)))
            candidate = Template(f"N{base.template_id[1:]}", "new", filters, base.terminal, twin_of=base.template_id)
            if (_compatible(filters, base.terminal, exclusive, monotone)
                    and candidate.signature not in public_signatures and candidate.signature not in seen):
                seen.add(candidate.signature)
                templates.append(candidate)
                break
    return templates


# --------------------------------------------------------------------- drawing parameters


@dataclass(frozen=True)
class Context:
    """The time window an instance's records lie in, which time-window questions are drawn from.

    A window of a day or less (every earthquake digest) is asked about on 5-minute marks, and no
    record may lie within 2 minutes of a window's ends. A longer window (weeks of close approaches)
    scales both by its length in days: a month is asked about on 3-hour marks and kept an hour clear
    of every record, so that, for its span, its windows are as coarse and as clear as a day's."""

    compiled_at: int
    window_hours: int

    @property
    def days(self) -> int:
        """The window's length in days, rounded up; at least 1."""
        return max(1, -(-self.window_hours // 24))

    @property
    def time_step(self) -> int:
        """Seconds between the marks a time window's ends lie on."""
        wanted = TIME_STEP_S * self.days
        return next((step for step in TIME_STEPS_S if step >= wanted), -(-wanted // 86400) * 86400)

    @property
    def time_margin(self) -> int:
        """No record may lie within this many seconds of a time window's ends."""
        return TIME_MARGIN_S * self.days


def _spatial(domain: Domain) -> Spatial:
    if domain.spatial is None:
        raise ValueError(f"{domain.name} records have no location")
    return domain.spatial


def _pick_unit(units: Sequence[Unit], rng: random.Random) -> str:
    """The symbol of the unit to ask in, drawn when there is a choice."""
    return rng.choice(units).symbol if len(units) > 1 else units[0].symbol


def _quantile_pick(values: Sequence[float], rng: random.Random, lo: float, hi: float) -> float:
    ordered = sorted(values)
    return ordered[int(rng.uniform(lo, hi) * (len(ordered) - 1))]


def round_to_step(value: float, step: float) -> float:
    """`value` rounded to a multiple of `step`. A power-of-ten step rounds to decimal places exactly
    as round() does: 0.1 gives round(value, 1), and 1 gives float(round(value))."""
    _, digits, exponent = Decimal(repr(step)).normalize().as_tuple()
    places = -exponent
    if digits == (1,):
        return float(round(value)) if places == 0 else round(value, places)
    return round(round(value / step) * step, max(0, places))


def _draw_threshold(spec: FieldSpec, records: Sequence[ResolvedRecord], rng: random.Random) -> dict | None:
    """A threshold on a field: a quantile of the records' values, jittered, converted to a drawn unit
    (affinely, for °F) and rounded to that unit's step (or, on a coin flip, to the field's coarse
    step). Its parameters, or None when it must be positive and is not; every draw is made either way.
    A threshold of zero is +0, never -0, so that it is written as the question states it."""
    symbol = _pick_unit(spec.units, rng) if spec.units else None
    unit = None if symbol is None else find_unit(spec.units, symbol)
    value = _quantile_pick([r.values[spec.key] for r in records], rng, *spec.quantiles)
    if spec.jitter:
        value += rng.uniform(-spec.jitter, spec.jitter)
    if unit is not None:
        value = unit.from_canonical(value)
    coarse = spec.coarse_step is not None and rng.random() < 0.5
    step = spec.threshold_step if unit is None or unit.step is None else unit.step
    value = round_to_step(value, spec.coarse_step if coarse else step) + 0.0
    if spec.positive_threshold and value <= 0:
        return None
    return {"value": value} if symbol is None else {"value": value, "unit": symbol}


def _group_counts(records: Sequence[ResolvedRecord]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.group] = counts.get(r.group, 0) + 1
    return sorted(counts.items())


def _draw_point(records: Sequence[ResolvedRecord], rng: random.Random, taken_labels: set[str],
                spatial: Spatial) -> dict:
    """A named point near a random record. Labels are unique within an instance."""
    anchor = rng.choice(records)
    lat, lon = destination(anchor.lat, anchor.lon, rng.uniform(*spatial.point_offset_km), rng.uniform(0.0, 360.0))
    label = spatial.point_label.format(source=anchor.source.upper(), number=rng.randrange(10, 100))
    while label in taken_labels:
        label = spatial.point_label.format(source=anchor.source.upper(), number=rng.randrange(10, 1000))
    return {"lat": round(lat, 3), "lon": round(lon, 3), "label": label}


def _draw_window(rng: random.Random, ctx: Context) -> dict | None:
    """A window an eighth to a half of the instance's window long, its ends on the context's marks."""
    total_min = ctx.window_hours * 60
    step = ctx.time_step
    duration = rng.randint(max(10, total_min // 8), max(20, total_min // 2))
    duration -= duration % (step // 60)
    first = -(-(ctx.compiled_at - total_min * 60) // step) * step  # first mark in the window
    latest = ctx.compiled_at - duration * 60
    if latest <= first:
        return None
    start = first + step * rng.randrange(0, max(1, (latest - first) // step))
    return {"start": start, "end": start + duration * 60}


def _answer_steps(units: Sequence[Unit], decimals: int, rng: random.Random) -> list[dict]:
    """An answer with a unit is converted into a drawn unit and rounded to that unit's places (the
    field's `decimals` unless the unit sets its own); a unitless one is left as it is."""
    if not units:
        return []
    unit = find_unit(units, _pick_unit(units, rng))
    return [{"op": "convert", "to": unit.symbol}, {"op": "round", "ndigits": unit.places(decimals)}]


def draw_filter(op: str, records: Sequence[ResolvedRecord], rng: random.Random, ctx: Context,
                taken_labels: set[str], *, domain: Domain = EARTHQUAKES) -> dict | None:
    kind, field = parse_op(op)
    if kind in ("at_least", "below"):
        params = _draw_threshold(domain.field(field), records, rng)
        return None if params is None else _step(op, **params)
    if kind == "within_distance":
        spatial = _spatial(domain)
        unit = _pick_unit(LENGTH, rng)
        point = _draw_point(records, rng, taken_labels, spatial)
        return _step(op, **point, radius=rng.choice(spatial.radii[unit]), unit=unit)
    if kind == "time_between":
        window = _draw_window(rng, ctx)
        return None if window is None else _step(op, **window)
    if kind == "in_group":
        groups = [g for g, c in _group_counts(records) if c >= 2]
        return _step(op, group=rng.choice(groups)) if groups else None
    raise ValueError(f"unknown filter {op!r}")


def draw_terminal(op: str, records: Sequence[ResolvedRecord], rng: random.Random, taken_labels: set[str], *,
                  domain: Domain = EARTHQUAKES) -> list[dict] | None:
    kind, field = parse_op(op)
    if kind in ("count", "argmax", "time_of_max"):
        return [_step(op)]
    if kind == "rank":
        return [_step(op, k=rng.choice(RANKS))]
    if kind in ("max", "min"):
        spec = domain.field(field)
        return [_step(op), *_answer_steps(spec.units, spec.decimals, rng)]
    if kind == "nearest_to":
        return [_step(op, **_draw_point(records, rng, taken_labels, _spatial(domain)))]
    if kind == "nearest_distance":
        spatial = _spatial(domain)
        nearest = _step(op, **_draw_point(records, rng, taken_labels, spatial))
        return [nearest, *_answer_steps(LENGTH, spatial.decimals, rng)]
    if kind == "compare_groups":
        counts = _group_counts(records)
        groups = [g for g, c in counts if c >= 2] or [g for g, _ in counts]
        if len(groups) < 2:
            return None
        a, b = rng.sample(groups, 2)
        return [_step(op, a=a, b=b)]
    raise ValueError(f"unknown terminal {op!r}")


def guards_ok(chain: Sequence[dict], records: Sequence[ResolvedRecord], *, domain: Domain = EARTHQUAKES,
              time_margin: int = TIME_MARGIN_S) -> bool:
    """Reject parameters that sit too close to any record, or answers that are not unique. Thresholds
    keep their field's margin, radii the domain's distance margin and time windows `time_margin`
    seconds (the instance's window's, Context.time_margin)."""
    units = domain.units
    for raw in chain:
        s = Step(raw, units)
        if s.kind in ("at_least", "below"):
            spec = domain.field(s.field)
            y = canonical_threshold(s)
            margin = max(spec.margin[0], spec.margin[1] * abs(y))
            if any(abs(r.values[s.field] - y) < margin for r in records):
                return False
        elif s.kind == "within_distance":
            spatial = _spatial(domain)
            radius = s.unit(s["unit"]).to_canonical(s["radius"])
            margin = max(spatial.margin[0], spatial.margin[1] * radius)
            if any(abs(distance_km(raw, r) - radius) < margin for r in records):
                return False
        elif s.kind == "time_between":
            if any(abs(r.time - edge) < time_margin for r in records for edge in (s["start"], s["end"])):
                return False
    rows = rows_after_filters(chain, records, units=units)
    term = Step(terminal_step(chain), units)
    if term.kind in ("argmax", "time_of_max", "rank"):
        values = sorted((r.values[term.field] for r in rows), reverse=True)
        gap = domain.field(term.field).rank_gap - FLOAT_SLACK
        if term.kind == "rank":  # no ties anywhere above or at the asked rank, so every reading of "k-th" agrees
            top = values[:term["k"] + 1]
            return len(values) >= term["k"] and all(a - b >= gap for a, b in zip(top, top[1:]))
        return bool(rows) and (len(values) < 2 or values[0] - values[1] >= gap)
    if term.kind in ("nearest_to", "nearest_distance"):
        spatial = _spatial(domain)
        d = sorted(distance_km(term.raw, r) for r in rows)
        return (bool(d) and d[0] <= spatial.nearest_max_km
                and (len(d) < 2 or d[1] - d[0] >= max(spatial.nearest_gap[0], spatial.nearest_gap[1] * d[0])))
    if term.kind in ("max", "min"):
        return bool(rows)
    if term.kind == "compare_groups":
        return any(r.group in (term["a"], term["b"]) for r in rows)
    return True


def _interesting(chain: Sequence[dict], records: Sequence[ResolvedRecord], value: Any, rng: random.Random) -> bool:
    kind = Step(terminal_step(chain)).kind
    if kind == "count":
        return 1 <= value < len(records)
    if kind == "compare_groups":
        return value != "tie" or rng.random() < 0.25  # ties are legitimate but kept rare
    return value is not None


def fingerprint(chain: Sequence[dict]) -> str:
    """Identity of a concrete question. Point labels are cosmetic and left out, and the two groups
    of a comparison are unordered ("A or B?" is the same question as "B or A?")."""
    steps = []
    for step in chain:
        canonical = {k: v for k, v in step.items() if k != "label"}
        if Step(step).kind == "compare_groups":
            canonical["a"], canonical["b"] = sorted((canonical["a"], canonical["b"]))
        steps.append(canonical)
    return json.dumps(steps, sort_keys=True)


def instantiate(template: Template, records: Sequence[ResolvedRecord], rng: random.Random, ctx: Context, *,
                used_chains: set[str], taken_labels: set[str], attempts: int = 40,
                domain: Domain = EARTHQUAKES) -> Question | None:
    """Draw parameters for a template until every guard passes and the question is new to the run."""
    for _ in range(attempts):
        chain: list[dict] = [{"op": "dedupe"}]
        complete = True
        for op in template.filters:
            step = draw_filter(op, records, rng, ctx, taken_labels, domain=domain)
            if step is None:
                complete = False
                break
            chain.append(step)
        tail = draw_terminal(template.terminal, records, rng, taken_labels, domain=domain) if complete else None
        if tail is None:
            continue
        chain.extend(tail)
        if fingerprint(chain) in used_chains or not guards_ok(chain, records, domain=domain,
                                                              time_margin=ctx.time_margin):
            continue
        value = execute(chain, records, units=domain.units)
        if value is None or not _interesting(chain, records, value, rng):
            continue
        answer_type, unit = answer_spec(chain, domain)
        used_chains.add(fingerprint(chain))
        taken_labels.update(step["label"] for step in chain if "label" in step)
        return Question(qid="", template_id=template.template_id, pool=template.pool, chain=chain,
                        answer_type=answer_type, unit=unit,
                        evidence=[r.id for r in rows_after_filters(chain, records, units=domain.units)])
    return None


def compose_questions(records: Sequence[ResolvedRecord], public: Sequence[Template], new: Sequence[Template],
                      rng: random.Random, ctx: Context, n_questions: int, used_chains: set[str],
                      usage: dict[str, int], *, domain: Domain = EARTHQUAKES) -> list[Question]:
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
            question = instantiate(template, records, rng, ctx, used_chains=used_chains, taken_labels=taken_labels,
                                   domain=domain)
            if question is not None:
                question.qid = f"Q{len(questions) + 1}"
                questions.append(question)
                used_templates.append(template.template_id)
                usage[template.template_id] = usage.get(template.template_id, 0) + 1
                break
    return questions


# -------------------------------------------------------------------------------- phrasing

ORDINALS = {2: "second", 3: "third"}
RELATIVE_CLAUSES = ("that ", "whose ")  # clause openings that read best at the end


def fmt_num(x: Any) -> str:
    """A threshold or radius as a question states it: exactly the chain's value, 12 for 12.0 (and 0
    for -0.0), never rounded or in exponent notation."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return plain_number(x)


def _quantity(value: float, unit: Unit, units: Sequence[Unit], decimals: int, alt_units: bool) -> str:
    """'50 miles', or in a metamorphic variant the same quantity in the next unit, to that unit's
    places (`decimals` unless it sets its own): '80.5 km'."""
    if alt_units:
        other = units[(units.index(unit) + 1) % len(units)]
        return f"{other.from_canonical(unit.to_canonical(value)):.{other.places(decimals)}f} {other.words}"
    return f"{fmt_num(value)} {unit.words}"


def _place(step: Mapping[str, Any], rng: random.Random) -> str:
    lat, lon = step["lat"], step["lon"]
    if rng.random() < 0.5:
        return f"{step['label']} ({abs(lat):.3f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.3f}°{'E' if lon >= 0 else 'W'})"
    return f"{step['label']} at ({lat:.3f}, {lon:.3f})"


def _when(ts: int, style: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if style == 0:
        return f"{dt:%Y-%m-%d %H:%M} UTC"
    return f"{dt:%H:%M} UTC on {dt.day} {MONTHS_LONG[dt.month - 1]} {dt.year}"


def _wordings(table: Mapping[str, tuple[str, ...]], step: Mapping[str, Any], domain: Domain) -> tuple[str, ...]:
    name = step_op(step)
    if name not in table:
        raise ValueError(f"the {domain.name} lexicon has no wording for {name!r}")
    return table[name]


def _filter_phrase(step: Mapping[str, Any], rng: random.Random, alt_units: bool, domain: Domain) -> str:
    s = Step(step, domain.units)
    if s.kind in ("at_least", "below"):
        symbol = s.get("unit")
        value = s["value"]
        values = {"value": fmt_num(value) if symbol is None else
                  _quantity(value, s.unit(symbol), s.units(), domain.field(s.field).decimals, alt_units)}
    elif s.kind == "within_distance":
        values = {"value": _quantity(s["radius"], s.unit(s["unit"]), LENGTH, _spatial(domain).decimals, alt_units),
                  "place": _place(step, rng)}
    elif s.kind == "time_between":
        style = rng.randrange(2)
        values = {"start": _when(s["start"], style), "end": _when(s["end"], style)}
    elif s.kind == "in_group":
        values = {"group": s["group"]}
    else:
        raise ValueError(f"unknown filter {s.op!r}")
    return rng.choice(_wordings(domain.lexicon.filters, step, domain)).format(**values)


def _join(clauses: list[str]) -> str:
    if len(clauses) <= 1:
        return "".join(clauses)
    return ", ".join(clauses[:-1]) + " and " + clauses[-1]


def rounding_text(chain: Sequence[Mapping[str, Any]]) -> str:
    """The rounding a chain asks for, as a sentence ("Round to 1 decimal place."); "" without one."""
    places = next((step["ndigits"] for step in chain if step["op"] == "round"), None)
    if places is None:
        return ""
    if places == 0:
        return "Round to a whole number."
    return f"Round to {places} decimal place{'' if places == 1 else 's'}."


def phrase_question(question: Question, rng: random.Random, alt_units: bool = False, *,
                    domain: Domain = EARTHQUAKES) -> str:
    """Wording for a question. Different RNG draws give paraphrases with the same meaning."""
    lexicon = domain.lexicon
    clauses = [_filter_phrase(s, rng, alt_units, domain) for s in question.chain if role(s["op"]) == "filter"]
    clauses.sort(key=lambda c: c.startswith(RELATIVE_CLAUSES))  # relative clauses read best last
    term = terminal_step(question.chain)
    s = Step(term, domain.units)
    values = {"records": f"{lexicon.records} {_join(clauses)}" if clauses else lexicon.all_records,
              "unit": s.unit(question.unit).words if question.unit else "",
              "rounding": rounding_text(question.chain)}
    if s.kind == "rank":
        values["nth"] = ORDINALS[s["k"]]
    elif s.kind in ("nearest_to", "nearest_distance"):
        values["place"] = _place(term, rng)
    elif s.kind == "compare_groups":
        values.update(a=s["a"], b=s["b"])
    return rng.choice(_wordings(lexicon.questions, term, domain)).format(**values)
