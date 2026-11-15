"""The final audit of trap placement off the legacy path (evalgen 0.3.0). Each test pins one finding on the
instances that showed it: built, as a run's first instance is, with a run's fresh state, they printed fewer
reports than their size, shared a careless reading's key between two reports, left a made-up weather observation
or a close approach's twin beside a planted report, dropped a planned correction, or left out of a trap's
affected questions a careless reading that the key's own guards pass."""
from __future__ import annotations

import functools
import json
import math
import operator
import random
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from conftest import quiet
from evalgen import pipeline, traps
from evalgen.domains.base import Record, stored_places
from evalgen.domains.earthquakes import EARTHQUAKES
from evalgen.domains.invented import INVENTED
from evalgen.domains.neo import NEO, designation_of
from evalgen.domains.weather import WEATHER
from evalgen.genres import get_genre
from evalgen.models import Event, Question
from evalgen.ops import DigestData, execute
from evalgen.pipeline import GenerateConfig, build_instance
from evalgen.questions import Context, compose_new_templates, guards_ok
from evalgen.report import load_instances
from evalgen.rules import Mishandle
from evalgen.scoring import same_answer
from evalgen.seeds import SeedLog
from evalgen.sources import build_domain_snapshot
from evalgen.traps import TrapPlan, draw_plan

CFG = GenerateConfig(seed=0)


@functools.lru_cache(maxsize=None)
def _snapshot(domain, seed):
    log = SeedLog(seed)
    snapshot, _ = build_domain_snapshot(domain, "synthetic", log)
    return snapshot, compose_new_templates(log.rng("templates", "new", domain.name), domain)


def built(domain, seed: int, index: int):
    """Instance `index` of a run about `domain` off the legacy path (classic), built with a fresh run's state, its
    snapshot, its bound domain, and the questions whose place was drawn near a report (reanchor_nearest)."""
    snapshot, new = _snapshot(domain, seed)
    moved: set = set()
    original = pipeline.reanchor_nearest

    def recording(questions, *args, **kwargs):
        out = original(questions, *args, **kwargs)
        moved.update(q.qid for q in out)
        return out

    pipeline.reanchor_nearest = recording
    try:
        inst = build_instance(index, snapshot, new, SeedLog(seed), GenerateConfig(seed=seed, variants=0),
                              Context(snapshot.compiled_at, snapshot.window_hours), set(), {}, domain=domain,
                              genre_plan=(get_genre("classic"),))
    finally:
        pipeline.reanchor_nearest = original
    return inst, snapshot, domain.bind(inst.surface), moved


def drawn_size(seed: int, index: int) -> int:
    """The size an instance draws for its digest (pipeline.sample_with_slots, its first draw)."""
    return SeedLog(seed).rng("instance", index, "events").randint(CFG.events_min, CFG.events_max)


def planned(seed: int, index: int) -> Counter:
    """The traps an instance of a built-in domain plans, by form (every built-in domain has a correctable field,
    so it plants the plan as drawn: TrapPlan.for_domain)."""
    plan = draw_plan(SeedLog(seed).rng("instance", index, "plan"))
    return +Counter({"phantom": plan.retractions, "cross_ref": plan.cross_refs, "same_id": plan.same_ids,
                     "second_report": plan.second_reports, "correction": plan.corrections})


def unsettled(inst, bound) -> list:
    return [g for g in map(bound.to_generic, inst.reports) if g.status != bound.statuses[0]]


def made_up(inst, snapshot, bound) -> list:
    """The planted reports (re-reports, phantoms, second reports) that stand for no record of the snapshot: made
    beside a real record when no slot could take them (traps._virtual_slot)."""
    pool = {(r.time, r.source, r.group, r.place) for r in map(bound.dress, snapshot.events)}
    return [g for g in map(bound.to_generic, inst.reports)
            if (g.role in ("duplicate", "phantom") or (g.role == "conflict" and g.status != bound.statuses[0]))
            and (g.time, g.source, g.group, g.place) not in pool]


# --------------------------------------------------------------------------- code-traps#0: split keys


@pytest.mark.parametrize("domain, seed, index", [(WEATHER, 77, 336), (EARTHQUAKES, 77, 336), (INVENTED, 77, 2710)])
def test_no_two_planted_reports_share_a_careless_readings_key(domain, seed, index):
    """A careless reader counts an unsettled report as a record of its own by its ID, source and status
    (Mishandle.split). Two re-reports under one record's ID from one source shared that key (in weather, inst-336:
    T6 and T9 on WX-549134 from KOKC), so misreading "only" one of them split both: each trap was credited with
    the other's effect, and a reader who counted one line alone fell to no trap. Every key now names one report,
    and each trap's split reading separates exactly its own line."""
    inst, _, bound, _ = built(domain, seed, index)
    keys = Counter((g.ref, g.source, g.status) for g in unsettled(inst, bound))
    assert all(count == 1 for count in keys.values()), [k for k, c in keys.items() if c > 1]
    for trap in inst.traps:
        for reading in trap.mishandles:
            for key in Mishandle.from_dict(reading).split:
                assert keys[tuple(key)] == 1, (trap.trap_id, key)


# ------------------------------------------------------------------ code-traps#2, code-traps#3: sizes


def test_a_neo_digest_prints_every_report_its_plan_adds():
    """In neo a report under a record's own ID needs a record whose object makes no other approach in the digest
    or its slots; inst-098 of seed 5 kept 9 records, none such, and its 3 second reports and 5 re-reports under
    their records' IDs were dropped: it printed 13 reports for a size of 21. Such a report is now filed under its
    slot's own ID with a note naming its record, so the digest prints its size, and holds as many traps as its
    plan draws."""
    inst, _, bound, _ = built(NEO, 5, 98)
    assert len(inst.reports) == drawn_size(5, 98) == 21
    plan = planned(5, 98)
    forms = Counter(t.form for t in inst.traps)
    assert len(inst.traps) == sum(plan.values())
    assert forms["phantom"] == plan["phantom"] and forms["correction"] == plan["correction"]
    assert forms["cross_ref"] == plan["cross_ref"] + plan["same_id"] + plan["second_report"]
    added = len(inst.reports) - len(inst.event_ids)  # one settling report a record, and the planted lines
    assert added == sum(plan[f] for f in ("phantom", "cross_ref", "same_id", "second_report")) == 12
    for g in unsettled(inst, bound):
        assert g.same_as is not None and designation_of(g.ref) != designation_of(g.same_as)


def test_a_domain_without_a_correctable_field_makes_room_for_its_second_reports(toys):
    """A domain may have no correctable field (FieldSpec.correctable), and then settles every conflict by a second
    report: its planned corrections, which a digest made no room for (a correction adds no report), each added a
    report, and digests printed up to 31 reports. The plan as such a domain plants it (TrapPlan.for_domain) turns
    them into second reports, which the digest makes room for."""
    nocorr = toys.module.Toy("toy-nocorr", located=True)
    nocorr.fields = tuple(replace(spec, correctable=False) for spec in nocorr.fields)
    plan = TrapPlan(retractions=1, cross_refs=1, same_ids=2, second_reports=1, corrections=3)
    assert plan.for_domain(nocorr) == TrapPlan(retractions=1, cross_refs=1, same_ids=2, second_reports=4)
    assert plan.for_domain(toys.toy) is plan and plan.for_domain(EARTHQUAKES) is plan
    corrected = 0
    for index in range(6):
        inst, _, _, _ = built(nocorr, 7, index)
        raw = draw_plan(SeedLog(7).rng("instance", index, "plan"))
        corrected += raw.corrections
        assert len(inst.reports) == drawn_size(7, index), inst.instance_id
        assert Counter(t.form for t in inst.traps)["second_report"] == raw.second_reports + raw.corrections
        assert not any(t.form == "correction" for t in inst.traps)
    assert corrected >= 5


# ----------------------------------------------------------------- code-traps#4, code-neo#0: neo IDs


def test_a_report_under_an_approach_id_never_leaves_its_slots_twin_beside_it():
    """A report filed under a record's approach ID takes a slot's time and values. In inst-000 of seed 404,
    2026dj30-ca1's re-report stood in slot 2026bl97-ca2, whose sibling 2026bl97-ca1, the object's approach to
    the Moon 8.8 hours earlier at about its speed and distance, was printed: a near-perfect twin under another
    designation and H, which no real report has (a reader who took the report of a shared ID with such a twin
    for the planted one was right 54% of the time). Its slot now makes no other approach in the digest or its
    slots, as its record makes none."""
    checked = 0
    for index in range(3):
        inst, snapshot, bound, _ = built(NEO, 404, index)
        slots = {(r.time, r.source, r.group, r.values["dist"], r.values["v_rel"]): r for r in snapshot.events}
        printed = Counter(designation_of(g.ref) for g in map(bound.to_generic, inst.reports))
        for g in unsettled(inst, bound):
            if g.same_as is not None:
                continue
            slot = slots.get((g.time, g.source, g.group, g.values["dist"], g.values["v_rel"]))
            if slot is None:  # made up beside its record (traps._virtual_slot), under a new designation
                continue
            checked += 1
            assert printed[designation_of(slot.id)] == 0, (inst.instance_id, g.ref, slot.id)
    assert checked >= 5


def _cad_twice(path, tmp_path):
    """The recorded CAD response, plus two objects that pass one body twice within its 60 days."""
    doc = json.loads(path.read_text())
    fields = doc["fields"]

    def row(des, when, body, dist, v, h):
        jd = datetime(*when, tzinfo=timezone.utc).timestamp() / 86400 + 2440587.5 + 69.184 / 86400
        values = {"des": des, "orbit_id": "1", "jd": repr(jd), "cd": "x", "dist": repr(dist), "dist_min": repr(dist),
                  "dist_max": repr(dist), "v_rel": repr(v), "v_inf": repr(v), "t_sigma_f": "< 00:01", "body": body,
                  "h": h, "fullname": f"       ({des})"}
        return [values[f] for f in fields]

    doc["data"] += [row("2026 WA3", (2026, 11, 18, 6, 0, 0), "Moon", 0.0041, 0.93, "30.112"),
                    row("2026 WA3", (2026, 11, 23, 18, 0, 0), "Moon", 0.0062, 1.07, "30.112"),
                    row("2026 WB7", (2026, 11, 14, 3, 0, 0), "Earth", 0.0021, 1.21, "31.04"),
                    row("2026 WB7", (2026, 11, 20, 15, 0, 0), "Earth", 0.0034, 1.33, "31.04")]
    doc["count"] = len(doc["data"])
    out = tmp_path / "cad_twice.json"
    out.write_text(json.dumps(doc))
    return out


def test_a_cross_reference_never_names_another_approach_of_its_own_object(tmp_path):
    """Rule 10 counts 2026wa3-ca1 and 2026wa3-ca2 as two records; a cross-referenced re-report under one with a
    note naming the other (seed 6, inst-014 of a CAD response with an object passing the Moon twice: "2026wa3-ca2
    ... preliminary | mirror of 2026wa3-ca1") made the key, which follows the note, disagree with rule 10, and the
    trap recorded rule 10's count as the careless one. A note now never names an approach of its own object."""
    from pathlib import Path

    feed = _cad_twice(Path(__file__).resolve().parent / "fixtures" / "neo" / "cad_2026-11-08.json", tmp_path)
    run = tmp_path / "twice"
    pipeline.generate(GenerateConfig(seed=6, instances=15, variants=0, mix="neo:classic", snapshot_file=str(feed)),
                      run, log=quiet)
    notes = 0
    for inst in load_instances(run):
        for r in inst.reports:
            if r.same_as is not None:
                notes += 1
                assert designation_of(r.ref) != designation_of(r.same_as), (inst.instance_id, r.ref, r.same_as)
    assert notes >= 5


def test_a_rereported_h_without_decimals_moves_as_another_catalogue_fits_it():
    """Under its record's designation a report gives the object's H as another catalogue fits it, 0.05 to 0.3
    off. An H stored without decimals (22.0, an older object's H to one place that ends in a zero) moved by a
    whole magnitude, to 21.0 or 23.0; it is now written to one place at least."""
    slot = Record("2026ab-ca1", 0, None, None, "Earth", "mpc", {"dist": 2e6, "v_rel": 6.0, "h": 27.3})
    for h in (22.0, 22.3, 21.07, 27.481):
        record = Record("12345-ca1", 0, None, None, "Earth", "jpl", {"dist": 1e6, "v_rel": 5.0, "h": h})
        for seed in range(60):
            got = NEO.rereport_values(record, slot, random.Random(seed), same_id=True)["h"]
            assert 0.05 - 1e-9 <= abs(got - h) <= 0.3 + 1e-9, (h, got)
            assert stored_places(got, 3) <= max(1, stored_places(h, 3)), (h, got)


# ----------------------------------------------------------------- code-weather#0: made-up observations


@pytest.mark.parametrize("seed, index", [(606, 47), (77, 336)])
def test_a_heavy_plan_files_its_weather_reports_in_real_observations_places(seed, index):
    """Where no record was free of traps, a re-report shared a record in a slot made beside a real one: in inst-047
    of seed 606, WX-670842 was re-reported at San Francisco at 15:25 UTC (KSFO reports at :51 and on five-minute
    marks, in whole degrees there) at 26.9 °C, 16 °C off the station's observation 26 minutes later. It now
    shares the record in a slot's place, an observation of the snapshot."""
    inst, snapshot, bound, _ = built(WEATHER, seed, index)
    assert made_up(inst, snapshot, bound) == []
    assert sum(1 for t in inst.traps if t.kind == "duplicate") > len(inst.event_ids) - 3  # records shared


# ------------------------------------------------------------- code-pipeline#3: every planned correction


@pytest.mark.parametrize("domain, seed, index", [(WEATHER, 77, 336), (WEATHER, 606, 47), (INVENTED, 77, 2710),
                                                 (NEO, 77, 2883)])
def test_a_heavy_plan_holds_every_trap_it_draws(domain, seed, index):
    """Corrections, planted last, need a record free of traps, and a heavy plan's re-reports took every record:
    these instances dropped every planned correction. The re-reports now leave as many records free as the
    corrections need, where they can share one, and a correction otherwise takes a record that only re-reports
    carry, none of which changes its values when misread. Their readings and the correction's act on rows of
    their own, so misreading both changes what misreading each changes, no more. Review: neo inst-2883 of seed 77
    still dropped its third correction, its one record free of traps having no room for a wrong value (an
    approach whose values are tied to its twin's): a correction that finds no place is now looked for again on
    the records that only re-reports carry."""
    inst, _, bound, _ = built(domain, seed, index)
    assert Counter(t.form for t in inst.traps) == planned(seed, index)
    assert len(inst.reports) == drawn_size(seed, index)
    corrections = [t for t in inst.traps if t.form == "correction"]
    for c in corrections:
        beside = [t for t in inst.traps if t is not c and t.target == c.target]
        assert all(t.kind == "duplicate" for t in beside)
        assert not any(Mishandle.from_dict(m).prefer_preliminary_for for t in beside for m in t.mishandles)
        alone = bound.resolve(inst.reports, inst.notices, Mishandle.from_dict(c.mishandles[0]))
        for t in beside:
            for m in t.mishandles:
                both = bound.resolve(inst.reports, inst.notices, Mishandle.merged([c.mishandles[0], m]))
                added = [r for r in bound.resolve(inst.reports, inst.notices, Mishandle.from_dict(m))
                         if r not in bound.resolve(inst.reports, inst.notices)]
                assert sorted(map(repr, both)) == sorted(map(repr, alone + added))


@pytest.mark.parametrize("index", [336, 799, 1229, 2615, 2672])
def test_a_correction_is_dropped_only_where_every_record_carries_a_conflict_or_one_report_under_its_id(index):
    """Review: what is left of code-pipeline#3, as it is. A re-report shares a record to leave one free for each
    correction to come only where a slot of the record's group and another source lets it; where none does, it takes
    a free record, and the plan's last correction can find no record that is free of traps or carries only re-reports
    none of which changes its values when misread (_valued). It is dropped, and nothing else is: the digest holds
    every other trap its plan draws and prints its drawn size. These are the instances of seed 77 (0-2999, built as
    a run's first, on synthetic data) whose plans of ten or more added reports drop a correction in any built-in
    domain: all five in earthquakes, with 10 to 13 added reports on 7 to 10 records. traps.py gives the rate, which
    is no domain's alone (a run of neo on a crafted CAD feed and one of the toy domain drop one too)."""
    domain, seed = EARTHQUAKES, 77
    inst, _, bound, _ = built(domain, seed, index)
    want, got = planned(seed, index), Counter(t.form for t in inst.traps)
    assert want - got == Counter({"correction": 1}) and not got - want
    assert len(inst.reports) == drawn_size(seed, index)
    conflicted = {t.target for t in inst.traps if t.kind.startswith("conflicting")}
    naming: dict = {}
    for g in unsettled(inst, bound):
        naming.setdefault(g.same_as or g.ref, []).append(g.same_as is None)
    valued = {record for record, own in naming.items() if own == [True]}
    assert set(inst.event_ids) <= conflicted | valued  # no record a correction could have taken


# --------------------------------------------------------- code-questions#0, sweep-guarantees#0: readings


@pytest.mark.parametrize("domain, seed, index", [(WEATHER, 11, 8), (WEATHER, 11, 91), (EARTHQUAKES, 11, 90)])
def test_every_reading_the_keys_own_guards_pass_is_counted(domain, seed, index):
    """A question whose place is drawn near a report once the traps are planted keeps no bound on how far its
    nearest record lies, but its readings were held to the domain's 500 km: a reading that changed its answer,
    with the misread nearest record beyond 500 km, was left out of the trap's affected questions and of the
    question's trap exposure (seed 11, weather inst-008 T3 Q3). Every reading that changes an answer and passes the
    key's own guards is now counted."""
    inst, _, bound, moved = built(domain, seed, index)
    assert moved
    data = DigestData(inst.reports, inst.notices)
    margin = Context(inst.compiled_at, inst.window_hours).time_margin
    counted = 0
    for trap in inst.traps:
        for reading in trap.mishandles:
            mishandle = Mishandle.from_dict(reading)
            records = bound.resolve(inst.reports, inst.notices, mishandle)
            for q in inst.questions:
                answer = execute(q.chain, data, mishandle=mishandle, units=bound.units, resolver=bound.resolve)
                if same_answer(q.answer_type, answer, inst.answer_key[q.qid], unit=q.unit):
                    continue
                if guards_ok(q.chain, records, domain=bound, time_margin=margin,
                             nearest_max_km=math.inf if q.qid in moved else None):
                    assert q.qid in trap.affected, (inst.instance_id, trap.trap_id, q.qid)
                    counted += q.qid in moved
    assert counted >= 1
    for q in inst.questions:
        assert q.features["trap_exposure"] == min(2, sum(1 for t in inst.traps if q.qid in t.affected))


def _reading(question, answer, guarded):
    return traps._Reading(question, answer, guarded)


def test_a_placement_whose_unguarded_reading_changes_an_answer_comes_last():
    """A trap counted as not load-bearing changed answers through readings that fail the key's guards (random mix,
    seed 316, inst-039 T8: a re-report 0.5% inside a threshold's 2% margin, which every careless reader counts):
    any placement with a reading that failed a guard ranked alike. One whose failing readings change no answer
    now beats one whose failing readings do, before its leads and scopes, so such a trap is planted only where
    every placement tried changes an answer that way."""
    q = Question("Q1", "P01", "public", [{"op": "count"}], "count")
    key = {"Q1": 3}
    assert traps._rank([_reading(q, 3, True)], key, True) == 1  # changes nothing, every reading holds
    assert traps._rank([_reading(q, 4, True)], key, True) == 3
    assert traps._rank([_reading(q, 3, True), _reading(q, 3, False)], key, True) == 0
    assert traps._rank([_reading(q, 3, True), _reading(q, 4, False)], key, True) == -1
    assert traps._rank([_reading(q, 4, True)], key, False) == 2  # the 0.2.0 ranks are as they were
    assert traps._rank([_reading(q, 3, True)], key, False) == 0
    target = (1, 1)
    quiet_one = traps._fit(0, (3, 3), target, traps.GUARDED_TOP, leads_like=False)
    loud_one = traps._fit(-1, target, target, traps.GUARDED_TOP, leads_like=True)
    assert quiet_one > loud_one
    assert traps._fit(1, (3, 3), target, traps.GUARDED_TOP, False) > quiet_one


# --------------------------------------------------------------------------- the nits


def test_place_notes_add_their_estimates_in_one_order_on_every_python():
    """A planted earthquake's place note averages the positions that the feed's notes imply for its town. The
    built-in sum() of floats adds them one by one up to Python 3.11 and compensates its rounding from 3.12, which
    can change the last bit; they are now added one by one on every Python, as 0.2.0 added them on 3.9."""
    values = [1e16, 1.0, -1e16]
    assert traps._in_order(values) == functools.reduce(operator.add, values) == 0.0
    town = "Lower Lake, CA"
    events = [Event(f"nc7{i}000000", 0, 38.9 + 0.01 * i, -122.6 + 0.01 * i, 5.0, 2.0, "md", "Northern California",
                    "nc", place=f"{3 + i} km {d} of {town}") for i, d in enumerate(("E", "NE", "N", "SSE"))]
    calls = []
    original = traps._in_order

    def spy(seq):
        calls.append(list(seq))
        return original(seq)

    traps._in_order = spy
    try:
        note = traps._place_for(events[0], 38.95, -122.55, events)
    finally:
        traps._in_order = original
    assert note.endswith(f" of {town}") and len(calls) == 2 and all(len(c) == 4 for c in calls)


def test_the_legacy_paired_difference_is_quoted_as_measured():
    """The module docstring quoted the legacy paired difference as 2.2 points [-0.5, +5.0]; the final
    measurement (and DESIGN.md) give [-0.6, +4.9]."""
    assert "2.2 points [-0.6, +4.9]" in traps.__doc__ and "[-0.5, +5.0]" not in traps.__doc__
