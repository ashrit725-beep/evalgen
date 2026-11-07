"""Generator guards: every generated item must have exactly one defensible answer, and the two
template pools must differ only in their filter combinations (fixes from the multi-agent review)."""
from __future__ import annotations

import random
import re

import pytest

from evalgen import pipeline
from evalgen.geo import COMPASS16, compass_point, destination, from_km, haversine_km
from evalgen.models import Event
from evalgen.ops import distance_km, rows_after_filters, terminal_step
from evalgen.questions import (_MONOTONE, NEAREST_MAX_KM, PUBLIC_TEMPLATES, compose_new_templates, fingerprint,
                               guards_ok)
from evalgen.report import load_instances
from evalgen.scoring import is_correct
from evalgen.traps import TrapConfig, inject_traps
from evalgen.util import read_json

from conftest import quiet


def quake(i, mag, lat=35.0, lon=-118.0, region="Alpha", network="aa", event_id=None, place=None):
    return Event(id=event_id or f"{network}{1000000 + i}", time=1_790_000_000 + 600 * i, lat=lat, lon=lon,
                 depth_km=5.0, mag=mag, mag_type="ml", region=region, network=network, place=place)


RANK3 = [{"op": "dedupe"}, {"op": "rank_mag", "k": 3}]


def test_rank_guard_rejects_ties_anywhere_down_to_the_asked_rank():
    assert not guards_ok(RANK3, [quake(i, m) for i, m in enumerate([3.0, 3.0, 2.5, 2.0])])  # tie for first
    assert not guards_ok(RANK3, [quake(i, m) for i, m in enumerate([3.0, 3.0, 2.5])])
    assert not guards_ok(RANK3, [quake(i, m) for i, m in enumerate([3.0, 2.9, 2.5, 2.45])])  # 3rd vs 4th too close
    assert guards_ok(RANK3, [quake(i, m) for i, m in enumerate([3.0, 2.9, 2.5, 2.0])])


def test_nearest_distance_needs_a_clear_and_local_winner():
    chain = [{"op": "dedupe"}, {"op": "nearest_distance", "lat": 35.0, "lon": -118.0, "label": "x"},
             {"op": "convert", "to": "km"}, {"op": "round", "ndigits": 1}]
    assert not guards_ok(chain, [quake(0, 2.0, 35.10), quake(1, 2.1, 35.11)])  # 11.1 vs 12.2 km: too close
    assert not guards_ok(chain, [quake(0, 2.0, 45.0), quake(1, 2.1, 46.0)])  # 1112 km away: not local
    assert guards_ok(chain, [quake(0, 2.0, 35.10), quake(1, 2.1, 35.30)])


def test_new_templates_are_twins_of_public_ones():
    public = {t.template_id: t for t in PUBLIC_TEMPLATES}
    public_signatures = {t.signature for t in PUBLIC_TEMPLATES}
    terminals = {t.terminal for t in PUBLIC_TEMPLATES}
    assert len(terminals) == 10  # the public pool covers every final operation
    for seed in range(40):
        new = compose_new_templates(random.Random(seed))
        assert len(new) == len(PUBLIC_TEMPLATES) and len({t.signature for t in new}) == len(new)
        for t in new:
            base = public[t.twin_of]
            assert t.terminal == base.terminal and len(t.filters) == len(base.filters)
            assert t.signature not in public_signatures
            assert not any((f, t.terminal) in _MONOTONE for f in t.filters)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """120 synthetic instances over three seeds, with their canonical events."""
    out = []
    for seed in (1, 2, 3):
        run_dir = pipeline.generate(pipeline.GenerateConfig(seed=seed, instances=40),
                                    tmp_path_factory.mktemp(f"seed{seed}"), log=quiet)
        events = {e["id"]: Event(**e) for e in read_json(run_dir / "snapshot" / "events.json")}
        out += [(inst, [events[i] for i in inst.event_ids]) for inst in load_instances(run_dir)]
    return out


def test_every_question_has_a_drawn_filter_and_no_monotone_pair(generated):
    for inst, _ in generated:
        for q in inst.questions:
            filters = [s["op"] for s in q.chain if s["op"] not in ("dedupe", "convert", "round")][:-1]
            assert filters, (inst.instance_id, q.qid)
            assert not any((f, terminal_step(q.chain)["op"]) in _MONOTONE for f in filters)


def test_rank_questions_have_no_ties_down_to_the_asked_rank(generated):
    checked = 0
    for inst, events in generated:
        for q in inst.questions:
            term = terminal_step(q.chain)
            if term["op"] == "rank_mag":
                mags = sorted((e.mag for e in rows_after_filters(q.chain, events)), reverse=True)[:term["k"] + 1]
                assert all(a - b >= 0.1 - 1e-9 for a, b in zip(mags, mags[1:])), (inst.instance_id, q.qid, mags)
                checked += 1
    assert checked > 0


def test_nearest_distance_runner_up_is_never_scored_correct(generated):
    checked = 0
    for inst, events in generated:
        for q in inst.questions:
            term = terminal_step(q.chain)
            if term["op"] == "nearest_distance":
                d = sorted(distance_km(term, e) for e in rows_after_filters(q.chain, events))
                assert d[0] <= NEAREST_MAX_KM
                if len(d) > 1:
                    runner_up = round(from_km(d[1], q.unit), 1)
                    assert not is_correct("distance", inst.answer_key[q.qid], runner_up)
                checked += 1
    assert checked > 0


def test_station_labels_are_unique_within_an_instance(generated):
    for inst, _ in generated:
        labels = {}
        for q in inst.questions:
            for step in q.chain:
                if "label" in step:
                    assert labels.setdefault(step["label"], (step["lat"], step["lon"])) == (step["lat"], step["lon"])


TOWN = (33.917, -116.787)  # "Cabazon, CA"


def _feed_style_events(ids):
    """USGS-shaped events: real place notes measured from one town, 2-decimal magnitudes, given IDs."""
    rng = random.Random(0)
    events = []
    for i, event_id in enumerate(ids):
        km, bearing = rng.uniform(4, 40), rng.uniform(0, 360)
        lat, lon = destination(*TOWN, km, bearing)
        events.append(Event(id=event_id, time=1_790_000_000 + 3000 * i, lat=round(lat, 4), lon=round(lon, 4),
                            depth_km=8.0, mag=round(1.0 + 0.37 * i, 2), mag_type="ml", region="California",
                            network=re.match(r"[a-z]+", event_id).group(0)[:2],
                            place=f"{round(km)} km {compass_point(bearing)} of Cabazon, CA"))
    return events


def _implied_town(report):
    km, bearing, _ = re.match(r"(\d+) km ([NSEW]+) of (.+)", report.place).groups()
    return destination(report.lat, report.lon, float(km), (COMPASS16.index(bearing) * 22.5 + 180) % 360)


def test_planted_reports_fit_the_feed_style_numeric_ids():
    """Sequential numeric IDs (CI style, step 8): the phantom's ID falls inside the sequence."""
    events = _feed_style_events([f"ci{41572000 + 8 * i}" for i in range(12)])
    for seed in range(20):
        reports, _, _ = inject_traps(events, [], random.Random(seed), compiled_at=1_790_040_000, window_hours=24,
                                     config=TrapConfig(id_style="sequence"))
        phantom = next(r for r in reports if r.role == "phantom")
        value = int(phantom.ref[2:])
        assert 41572000 - 4 * 8 <= value <= 41572000 + 8 * 11 + 4 * 8 and (value - 41572000) % 8 == 0
        # its place note implies the same town as the real notes, up to the 16-point compass rounding
        town = _implied_town(phantom)
        assert haversine_km(*town, *TOWN) <= 0.25 * haversine_km(*TOWN, phantom.lat, phantom.lon) + 1.5
        assert all(r.place for r in reports)  # no report is the odd one out without a place


def test_planted_reports_fit_the_feed_style_alphanumeric_ids():
    """Time-coded IDs (AK style): the phantom shares its time neighbour's long prefix."""
    letters = "abcdefghijkl"
    events = _feed_style_events([f"aka2026vu{letters[i]}{letters[(3 * i) % 12]}q" for i in range(12)])
    for seed in range(20):
        reports, _, _ = inject_traps(events, [], random.Random(seed), compiled_at=1_790_040_000, window_hours=24,
                                     config=TrapConfig(id_style="sequence"))
        phantom = next(r for r in reports if r.role == "phantom")
        assert phantom.ref.startswith("aka2026vu") and len(phantom.ref) == len(events[0].id)


def test_pools_stay_matched_per_twin_pair_even_when_templates_run_out(tmp_path):
    """Run-wide uniqueness exhausts region-only templates; each twin pair must stay in step."""
    run_dir = pipeline.generate(pipeline.GenerateConfig(seed=42, instances=300), tmp_path / "big", log=quiet)
    counts: dict[str, int] = {}
    terminals: dict[tuple[str, str], int] = {}
    for inst in load_instances(run_dir):
        for q in inst.questions:
            counts[q.template_id] = counts.get(q.template_id, 0) + 1
            key = (terminal_step(q.chain)["op"], q.pool)
            terminals[key] = terminals.get(key, 0) + 1
    for public in PUBLIC_TEMPLATES:
        twin = "N" + public.template_id[1:]
        assert abs(counts.get(public.template_id, 0) - counts.get(twin, 0)) <= 1, (public.template_id, counts)
    for op in {op for op, _ in terminals}:
        assert abs(terminals.get((op, "public"), 0) - terminals.get((op, "new"), 0)) <= 3, (op, terminals)


def test_swapped_region_comparisons_are_the_same_question():
    a = [{"op": "dedupe"}, {"op": "compare_regions", "a": "Utah", "b": "Nevada"}]
    b = [{"op": "dedupe"}, {"op": "compare_regions", "a": "Nevada", "b": "Utah"}]
    assert fingerprint(a) == fingerprint(b)
