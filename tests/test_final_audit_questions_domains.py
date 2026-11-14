"""The final audit's fixes to questions, chatter and domain details.

Chatter off the legacy path: a status line names a window that holds no report made at its station,
whichever source filed it (a weather observation that a relay feed filed stands at its station), and never
names a relay feed; a line that names a window and no station holds no report at all; a window that ends on a
later day prints its end's date; no line uses a word that the rules define; and the earthquake status line
says when its station was offline, with the date, not that its readings were left out. The legacy path keeps
the 0.2.0 lines, draw for draw. Then the synthetic earthquake world of few events, the exact scales that
settle a rounding, filter clauses joined as English joins them, point numbers apart from radio call signs,
rank gaps only where a ranked terminal reads them, the threshold check of Domain.validate (and the toy
domain's temperature, which it refused) and what a question's identity holds.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from evalgen import pipeline
from evalgen.digest import CHATTER, make_chatter, timed_chatter
from evalgen.domains import all_domains
from evalgen.domains.base import GENERIC_CHATTER, RecordReport
from evalgen.domains.earthquakes import EARTHQUAKES, GENRE_CHATTER, synthetic_world
from evalgen.domains.invented import CHATTER as INVENTED_CHATTER
from evalgen.domains.invented import CORRECTION_NOTICES, DUPLICATE_NOTES, RETRACTION_NOTICES
from evalgen.domains.neo import AU_KM, NEO
from evalgen.domains.weather import STATION_BY_ID, WEATHER, station_of
from evalgen.genres.radio import CALL_SIGNS
from evalgen.models import Question
from evalgen.questions import POINT_NUMBER_OFFSET, _draw_point, _exact, _halfway, fingerprint, phrase_question
from evalgen.units import LENGTH

from conftest import quiet

COMPILED_AT = 1_780_000_000  # 20:26:40 UTC
DATED = re.compile(r"from (?P<h1>\d{2}:\d{2}) UTC on (?P<d1>\d{1,2} [A-Z][a-z]+ \d{4}) until (?P<h2>\d{2}:\d{2}) UTC"
                   r"(?: on (?P<d2>\d{1,2} [A-Z][a-z]+ \d{4}))?")
STATION = re.compile(r"\b(?P<station>[A-Z0-9]{2,12})\.[A-Z]{3}\b")


def _at(date: str, clock: str) -> int:
    day = datetime.strptime(date, "%d %B %Y").replace(tzinfo=timezone.utc)
    return int((day + timedelta(hours=int(clock[:2]), minutes=int(clock[3:]))).timestamp())


def window(line: str) -> tuple[int, int] | None:
    """The first and last minute of the window a line names, read as a reader reads it: the end on the date it
    gives, else on the start's date (so an end date left out across midnight reads as an end before the start)."""
    found = DATED.search(line)
    if found is None:
        return None
    return _at(found["d1"], found["h1"]), _at(found["d2"] or found["d1"], found["h2"])


def generated(tmp_path: Path, mix: str, seed: int, instances: int) -> list[dict]:
    out = tmp_path / f"{mix.replace(':', '-')}-{seed}"
    pipeline.generate(pipeline.GenerateConfig(seed=seed, instances=instances, variants=0, mix=mix), out, log=quiet)
    return [json.loads(p.read_text()) for p in sorted((out / "instances").glob("inst-*.json"))]


def as_report(d: dict) -> RecordReport:
    return RecordReport(ref=d["ref"], source=d["source"], status=d["status"], time=d["time"], lat=d["lat"],
                        lon=d["lon"], group=d["group"], values=d["values"], place=d.get("place"))


# --------------------------------------------------------------------------- status lines


@pytest.mark.parametrize("seed", [18, 9])
def test_a_weather_status_line_keeps_clear_of_every_observation_made_at_its_station(tmp_path, seed):
    """ "Sensor KIAH.PGH was offline for maintenance from 02:00 UTC on 23 February 2026 until 03:02 UTC" beside an
    observation made at KIAH at 02:30 that MADIS filed read as a reason to doubt it: the window was kept clear only
    of the reports that KIAH itself filed. A status line now names a station of the digest, never a relay feed
    (which has no sensors), and its window holds no observation made at that station, whichever feed filed it.
    (With 12 instances the old lines named a relay feed three times and held a relayed KPHL observation in seed
    18, and in seed 9 left out the date of an end that fell on the next day.)"""
    lines = crossed = 0
    for inst in generated(tmp_path, "weather:classic", seed, 12):
        reports = [as_report(r) for r in inst["reports"]]
        for line in inst["chatter"]:
            named = STATION.search(line)
            if named is None:
                continue
            lines += 1
            station = named["station"]
            assert station in STATION_BY_ID, line  # a station, never MADIS, NOAAPORT or AWOSNET
            first, last = window(line)
            assert first < last, line  # an end on the next day gives its date
            crossed += "UTC on" in line.split("until", 1)[1]
            held = [r.ref for r in reports if station_of(r).id == station and first <= r.time < last + 60]
            assert held == [], (line, held)
    assert lines >= 3 and (crossed >= 1 or seed != 9), (lines, crossed)


def test_status_station_is_where_a_record_was_made_in_weather_and_its_source_elsewhere():
    kiah = STATION_BY_ID["KIAH"]
    relayed = RecordReport(ref="WX-362253", source="MADIS", status="verified", time=COMPILED_AT, lat=kiah.lat,
                           lon=kiah.lon, group=kiah.region, values={}, place=kiah.name)
    assert WEATHER.status_station(relayed) == "KIAH"
    assert WEATHER.status_station(replace(relayed, source="KIAH")) == "KIAH"
    nowhere = replace(relayed, place=None, lat=0.0, lon=0.0)  # at no known station: its feed
    assert WEATHER.status_station(nowhere) == "MADIS"
    assert EARTHQUAKES.status_station(replace(relayed, source="ci")) == "ci" and NEO.status_station(relayed) == "MADIS"


def test_a_relayed_observation_takes_its_stations_windows(tmp_path):
    """Observations made at KIAH that MADIS relayed, every ten minutes of the day, leave no window for a line
    about a KIAH sensor: every such line names KJFK, whose digest holds none, or is left out."""
    kiah = STATION_BY_ID["KIAH"]
    relayed = [RecordReport(ref=f"WX-{100000 + k}", source="MADIS", status="verified", time=t, lat=kiah.lat,
                            lon=kiah.lon, group=kiah.region, values={}, place=kiah.name)
               for k, t in enumerate(range(COMPILED_AT - 21 * 3600, COMPILED_AT, 600))]
    filed = [(WEATHER.status_station(r), r.time) for r in relayed]
    lines = (WEATHER.chatter[0], WEATHER.chatter[0].replace("Sensor", "Gauge"), WEATHER.chatter[0].replace("Sensor",
                                                                                                         "Probe"))
    named = set()
    for seed in range(40):
        for text, _ in timed_chatter(random.Random(seed), COMPILED_AT, ["KIAH", "KJFK"], lines, WEATHER.words, filed):
            named.add(STATION.search(text)["station"])
    assert named == {"KJFK"}


def test_a_window_line_without_a_station_keeps_clear_of_every_report():
    """ "Ledger maintenance ran from 03:00 to 03:45 UTC; filing resumed afterwards" stood beside a lot filed at 03:20
    (invented, QSV-64676): only lines about a station were kept clear. Now a line that names a window holds no
    report of any source; and a window printed without its date holds none at that time of any day."""
    dated = ("Maintenance ran from {h1} UTC on {date} until {until}; filing resumed afterwards.",
             "Late lots may be listed in the next edition.", "Gate traffic was heavier than usual.")
    undated = ("Maintenance ran from {h1} to {h2} UTC; filing resumed afterwards.", *dated[1:])
    filed = [(source, COMPILED_AT - minutes * 60) for source in ("ash", "bay") for minutes in range(5, 20 * 60, 47)]
    earlier = [(source, t - 86400) for source, t in filed]  # the same times of day, a day before
    seen = 0
    for seed in range(60):
        for text, after in timed_chatter(random.Random(seed), COMPILED_AT, ["ash", "bay"], dated, None, filed):
            if text.startswith("Maintenance"):
                first, last = window(text)
                assert after == last and not any(first <= t < last + 60 for _, t in filed), text
                seen += 1
        for text, _ in timed_chatter(random.Random(seed), COMPILED_AT, ["ash", "bay"], undated, None, earlier):
            if text.startswith("Maintenance"):
                h1, h2 = re.findall(r"\d{2}:\d{2}", text)
                start = int(h1[:2]) * 60 + int(h1[3:])
                length = (int(h2[:2]) * 60 + int(h2[3:]) - start) % 1440 + 1
                assert not any((t // 60 - start) % 1440 < length for _, t in earlier), text
                seen += 1
    assert seen >= 40, seen
    # A line that names no window draws exactly as make_chatter's do.
    plain = make_chatter(random.Random(5), COMPILED_AT, ["ash"], dated[1:] + ("Reminder: units vary.",))
    timed = timed_chatter(random.Random(5), COMPILED_AT, ["ash"], dated[1:] + ("Reminder: units vary.",), None, filed)
    assert [text for text, _ in timed] == plain


def test_a_window_that_ends_on_a_later_day_prints_its_end_date():
    """ "The catalogue mirror JPL.KTR was offline from 23:30 UTC on 9 September 2026 until 00:15 UTC" left out the
    date of an end that fell on the next day. {until} gives it then, and only then, in make_chatter and in
    timed_chatter alike (make_chatter's lines draw as before)."""
    compiled = COMPILED_AT - COMPILED_AT % 86400 + 86400 + 90 * 60  # 01:30 UTC: windows open 23:30 to 06:30 UTC
    line = ("The {source_label} {station} was offline for maintenance from {h1} UTC on {date} until {until}.",
            "Reminder: units vary.", "Late records may be listed in the next edition.")
    crossed = same_day = 0
    for seed in range(80):
        texts = make_chatter(random.Random(seed), compiled, ["jpl"], line, NEO.words)
        timed = timed_chatter(random.Random(seed), compiled, ["jpl"], line, NEO.words, [])
        assert [text for text, _ in timed] == texts
        for text, after in timed:
            if "offline" not in text:
                continue
            first, last = window(text)
            found = DATED.search(text)
            assert first < last and after == last, text
            if found["d2"] is not None:
                crossed += 1
                assert _at(found["d1"], "00:00") + 86400 == _at(found["d2"], "00:00"), text
            else:
                same_day += 1
                assert datetime.fromtimestamp(last, tz=timezone.utc).date() == datetime.fromtimestamp(
                    first, tz=timezone.utc).date(), text
    assert crossed and same_day, (crossed, same_day)
    a, b = random.Random(9), random.Random(9)  # the 0.2.0 lines draw as they did, and print as they did
    assert make_chatter(a, COMPILED_AT, ["us", "ci"]) == make_chatter(b, COMPILED_AT, ["us", "ci"], CHATTER)
    assert CHATTER[0] == ("Station {station} was offline from {h1} to {h2} UTC for maintenance; its readings are not "
                          "included.")


# Words that the reading rules give a meaning of their own: the kinds of report and notice, and the words of the
# notes and notices they quote, besides the words any line may use (and the records' own nouns).
RULE_TERMS = {"duplicate", "duplicates", "dup", "re-report", "repeat", "repeats", "retraction", "retracted",
              "correction", "corrected", "notice", "relay", "relayed", "forwarded", "quoted", "read-back",
              "superseded", "mirror", "withdrawn", "voided", "cancelled", "revised", "amended", "restated"}
PLAIN = {"same", "as", "of", "to", "from", "the", "was", "not", "an", "a", "now", "is", "in", "and", "or", "by",
         "its", "it", "this", "out", "also", "earthquake", "event", "magnitude", "observation", "approach", "catalogue"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:-[a-z]+)*", re.sub(r"\{[a-z_0-9]+\}", " ", text).lower()))


def _rule_words(phrases) -> set[str]:
    return {w for phrase in phrases for w in _words(phrase)} - PLAIN | RULE_TERMS


@pytest.mark.parametrize("name", ["earthquakes", "weather", "neo", "invented", "generic"])
def test_no_chatter_line_uses_a_word_that_the_rules_define(name):
    """ "The catalogue mirror JPL.KTR was offline ..." used "mirror", the word of the note "mirror of" that rule 1
    quotes; and an invented surface could draw "struck from the ledger" or "double entry of" beside "Ledger
    maintenance" and "units vary from entry to entry". No line off the legacy path uses a word of a note or notice
    its domain may print (every pool, for invented), nor a word that names a kind of report or notice."""
    if name == "invented":
        lines, phrases = INVENTED_CHATTER, (*DUPLICATE_NOTES, *RETRACTION_NOTICES, *CORRECTION_NOTICES)
    else:
        domain = {d.name: d for d in all_domains()}.get(name, EARTHQUAKES)
        lines = GENERIC_CHATTER if name == "generic" else domain.chatter
        vocab = domain.trap_vocab
        phrases = (*vocab.duplicate_notes, *vocab.retraction_notices, *vocab.correction_notices)
    forbidden = _rule_words(phrases)
    for line in lines:
        assert not _words(line) & forbidden, (line, _words(line) & forbidden)
        assert "not included" not in line, line  # no line says what a digest holds


def test_the_earthquake_status_line_gives_its_date_off_the_legacy_path_and_the_legacy_line_stays(tmp_path):
    """Off the legacy path an earthquake digest printed the 0.2.0 status line, "Station CI.WBN was offline from 15:18
    to 16:19 UTC for maintenance; its readings are not included.", which says that the readings were left out and
    gives no date. It now says when the station was offline, with the date; the other 0.2.0 lines stay, and a
    legacy digest prints the 0.2.0 lines, drawn as 0.2.0 drew them."""
    assert EARTHQUAKES.chatter is GENRE_CHATTER and GENRE_CHATTER[1:] == CHATTER[1:]
    assert GENRE_CHATTER[0] == "Station {station} was offline for maintenance from {h1} UTC on {date} until {until}."
    status = re.compile(r"Station [A-Z]{2}\.[A-Z]{3} was offline for maintenance from \d{2}:\d{2} UTC on "
                        r"\d{1,2} [A-Z][a-z]+ \d{4} until \d{2}:\d{2} UTC(?: on \d{1,2} [A-Z][a-z]+ \d{4})?\.")
    printed = 0
    for inst in generated(tmp_path, "earthquakes:classic", 31, 8):
        networks = {r["network"] for r in inst["reports"]}  # an earthquake report keeps its 0.2.0 form
        for line in inst["chatter"]:
            assert line in GENRE_CHATTER[1:] or status.fullmatch(line), line
            if line.startswith("Station"):
                printed += 1
                station = STATION.search(line)["station"].lower()
                first, last = window(line)
                assert station in networks and not any(r["network"] == station and first <= r["time"] < last + 60
                                                       for r in inst["reports"]), line
    assert printed >= 1
    legacy = tmp_path / "legacy"  # the 0.2.0 status line (tests/test_regression_020.py pins every legacy byte)
    pipeline.generate(pipeline.GenerateConfig(seed=31, instances=6, variants=0), legacy, log=quiet)
    old = re.compile(r"Station [A-Z]{2}\.[A-Z]{3} was offline from \d{2}:\d{2} to \d{2}:\d{2} UTC for maintenance; its "
                     r"readings are not included\.")
    lines = [line for path in sorted((legacy / "instances").glob("inst-*.json"))
             for line in json.loads(path.read_text())["chatter"]]
    assert all(line in CHATTER[1:] or old.fullmatch(line) for line in lines) and any(map(old.fullmatch, lines))


def test_every_genre_reads_a_digest_whose_chatter_gives_an_end_date_as_no_data(tmp_path):
    """A status line whose window ends the next day prints two dates ("from 23:00 UTC on 22 April 2026 until
    00:48 UTC on 23 April 2026"): every genre still reads it as no data, and its rendering back exactly."""
    from evalgen.genres import all_genres
    from evalgen.genres.base import RenderContext, check_round_trip

    inst = generated(tmp_path, "weather:classic", 9, 1)[0]
    reports, notices = [as_report(r) for r in inst["reports"]], []
    compiled = inst["compiled_at"]
    end = compiled - 3 * 3600
    lines = [template.format(station="KIAH.PGH", h1="23:00", h2="00:48", date="22 April 2026",
                             until="00:48 UTC on 23 April 2026", **WEATHER.words)
             for template in (WEATHER.chatter[0], GENRE_CHATTER[0], NEO.chatter[0], INVENTED_CHATTER[4],
                              INVENTED_CHATTER[6], GENERIC_CHATTER[0])]
    glossary = WEATHER.glossary()
    for genre in all_genres():
        for seed in range(3):
            rng = random.Random(seed)
            style = genre.choose_style(rng, WEATHER)
            ctx = RenderContext(reports, notices, lines, compiled_at=compiled, window_hours=inst["window_hours"],
                                style=style, rng=rng, domain=WEATHER, glossary=glossary,
                                chatter_after=[end] * len(lines))
            text = genre.render(ctx)
            check_round_trip(genre, text, ctx, glossary)
            assert sum("00:48 UTC on 23 April 2026" in line for line in text.splitlines()) >= 1, genre.name


def test_an_invented_window_line_holds_no_lot_of_the_digest(tmp_path):
    """The invented lines that name a window give its date, and hold no report of any agency (QSV-64676 was filed
    inside "Ledger maintenance ran from ... to ... UTC; filing resumed afterwards")."""
    printed = 0
    for inst in generated(tmp_path, "invented:classic", 9206, 10):
        for line in inst["chatter"]:
            if "{" in line or not re.search(r"\d{2}:\d{2}", line):
                continue
            first, last = window(line)
            assert first < last and not any(first <= r["time"] < last + 60 for r in inst["reports"]), line
            printed += 1
    assert printed >= 2, printed


# ----------------------------------------------------------------------- domain details


def test_a_synthetic_earthquake_world_of_few_events_never_fails(tmp_path):
    """A world whose first events were all below magnitude 2 drew its mainshocks from the list it was growing,
    beside weights of the old length, and raised ValueError (seed 7, 10 events; a run of seed 10 with a pool of
    10). It holds at most `n` events, as its docstring now says."""
    first = synthetic_world(random.Random(7), 10, COMPILED_AT, 24)
    assert 1 <= len(first) <= 10
    for seed in range(120):
        for n in (1, 4, 8, 10, 12, 13, 20):
            world = synthetic_world(random.Random(seed), n, COMPILED_AT, 24)
            assert len(world) <= n and len({e.id for e in world}) == len(world)
    assert len(synthetic_world(random.Random(3), 360, COMPILED_AT, 24)) == 360
    small = pipeline.GenerateConfig(seed=10, instances=1, variants=0, mix="earthquakes:classic", pool_size=10)
    run = pipeline.generate(small, tmp_path / "small", log=quiet)
    assert (run / "instances" / "inst-000.json").is_file()


def test_only_a_field_a_ranked_terminal_reads_has_a_rank_gap(toys):
    """Weather gave its wind and its pressure rank gaps that nothing read (no question ranks them); a field whose
    terminals rank nothing has none, and every field a ranked terminal reads has one."""
    for domain in (*all_domains(), toys.toy):
        for spec in domain.fields:
            ranked = {"argmax", "rank", "time_of_max"} & set(spec.terminals)
            assert (spec.rank_gap > 0) == bool(ranked), (domain.name, spec.key, spec.rank_gap)


def test_validate_refuses_a_field_whose_thresholds_can_never_clear_their_margin(toys):
    """A threshold lies at most the field's jitter and half the step of its unit from the record it is drawn from:
    the toy's temperature (no jitter, half-degree and whole-°F steps, a 0.3 °C margin) was never asked about with a
    threshold. Domain.validate refuses such a field in any unit, and the toy's temperature, now with a jitter, is
    asked about in both."""
    spec = toys.flat.field("temp")
    assert spec.jitter > 0
    variant = type("Stiff", (type(toys.flat),), {})("toy-stiff", located=False)
    variant.fields = (replace(spec, jitter=0.0), variant.fields[1])
    with pytest.raises(ValueError, match=re.escape("no threshold on temp asked in °C clears its guard margin 0.3")):
        variant.validate()
    with pytest.raises(ValueError, match=re.escape("no threshold on temp asked in °F clears its guard margin 0.3")):
        variant.validate()
    for domain in (*all_domains(), toys.toy, toys.flat):
        domain.validate()
    from evalgen.questions import _draw_threshold, guards_ok

    passed = set()
    for seed in range(120):
        rng = random.Random(seed)
        records = toys.flat.synthetic(rng, 24, COMPILED_AT, 168)
        params = _draw_threshold(toys.flat.field("temp"), records, rng)
        chain = [{"op": "dedupe"}, {"op": "at_least", "field": "temp", **params}, {"op": "count"}]
        if guards_ok(chain, records, domain=toys.flat):
            passed.add(params["unit"])
    assert passed == {"°C", "°F"}


# ------------------------------------------------------------------------------ questions


def test_an_exact_scale_is_the_decimal_a_glossary_prints_or_the_ratio_a_repeating_one_rounds():
    """_exact read 1.609344 as 955675237141/593829061494 and the au as its binary rounding: it now gives the
    decimal a glossary prints and the codec computes with, and the simple ratio of a scale that only a repeating
    decimal writes. No halfway decision moves in the ranges the built-in fields store."""
    assert _exact(1.609344) == Fraction(25146, 15625) and _exact(AU_KM) == Fraction(1495978707, 10)
    assert _exact(33.86389) == Fraction(3386389, 100000) and _exact(0.00044704) == Fraction(1397, 3125000)
    assert (_exact(5 / 9), _exact(-160 / 9), _exact(1 / 3600)) == (Fraction(5, 9), Fraction(-160, 9), Fraction(1, 3600))
    assert (_exact(-273.15), _exact(384400.0), _exact(0.0)) == (Fraction(-5463, 20), 384400, 0)

    def binary(x: float) -> Fraction:  # the former reading
        return Fraction(x).limit_denominator(10 ** 12)

    mile, au, mph = LENGTH[1], NEO.field("dist").units[1], WEATHER.field("wind").units[1]
    for unit, values, places in ((mile, range(-100, 8000), 1), (mph, range(0, 4000), 1),
                                 (au, range(0, 7_600_000, 37), 6)):
        for n in values:
            value = n / 10 if unit is not au else float(n)
            exact = Fraction(repr(value))
            assert (_halfway(exact / _exact(unit.scale) * 10 ** places)
                    == _halfway(exact / binary(unit.scale) * 10 ** places)), (unit.symbol, value)


WEATHER_TWO = Question("Q5", "N01", "new", [{"op": "dedupe"}, {"op": "below", "field": "temp", "value": 283.0,
                                                                 "unit": "K"},
                                             {"op": "below", "field": "wind", "value": 10.0, "unit": "mph"},
                                             {"op": "count"}], "count")
QUAKES_TWO = Question("Q1", "N06", "new", [{"op": "dedupe"}, {"op": "mag_at_least", "value": 3.0},
                                           {"op": "time_between", "start": COMPILED_AT, "end": COMPILED_AT + 3600},
                                           {"op": "count"}], "count")
MIXED = re.compile(r"\b(?:with|of|at|in|within|colder|smaller|shallower|located|recorded|taken) [^?]*? and "
                   r"(?:whose|that) ")


def test_filter_clauses_join_as_english_joins_them_off_the_legacy_path():
    """ "observations with a temperature below 283 kelvins and whose wind speed was less than 10 miles per hour"
    joined a relative clause to one that is not. Off the legacy path a relative clause beside another kind takes a
    wording of its op that is not one, or follows without "and" when its op has none ("earthquakes of magnitude 3
    or greater that occurred between ..."); the draws stay the same, and legacy prompts keep the 0.2.0 wording."""
    weather = [phrase_question(WEATHER_TWO, random.Random(seed), domain=WEATHER, lexicon=WEATHER.prompt_lexicon())
               for seed in range(60)]
    assert not any(MIXED.search(text) for text in weather), [t for t in weather if MIXED.search(t)][:3]
    assert any("whose temperature" in text and "and whose wind" in text for text in weather)  # two of a kind: "and"
    lexicon = EARTHQUAKES.prompt_lexicon()
    for seed in range(40):
        genre_rng, legacy_rng = random.Random(seed), random.Random(seed)
        genre = phrase_question(QUAKES_TWO, genre_rng, lexicon=lexicon)
        legacy = phrase_question(QUAKES_TWO, legacy_rng)
        assert genre_rng.getstate() == legacy_rng.getstate()  # the same draws
        assert " and that " not in genre and re.search(r"(?:or greater|or more|at least 3) that occurred", genre), genre
        assert " and that " in legacy, legacy  # the 0.2.0 wording
    assert phrase_question(QUAKES_TWO, random.Random(1)) == (
        "How many distinct earthquakes of magnitude 3 or greater and that were recorded from 2026-05-28 20:26 UTC "
        "up to 2026-05-28 21:26 UTC are there?")


def test_a_point_off_the_legacy_path_shares_no_number_with_a_call_sign(tmp_path):
    """ "How far from station 84 ..." beside a transcript whose Unit 84 read out Puerto Rico's reports: a point's
    number (10 to 99) could be a call sign's (1 to 99). Off the legacy path it is drawn as before and printed 100
    higher; the legacy label keeps the 0.2.0 number."""
    records = [e for e in synthetic_world(random.Random(2), 40, COMPILED_AT, 24)]
    spatial = EARTHQUAKES.spatial
    for seed in range(30):
        strict = _draw_point(records, random.Random(seed), set(), spatial, strict=True)
        legacy = _draw_point(records, random.Random(seed), set(), spatial)
        number = int(strict["label"].split()[-1])
        assert POINT_NUMBER_OFFSET + 10 <= number < POINT_NUMBER_OFFSET + 100 and strict["lat"] == legacy["lat"]
        assert re.fullmatch(r"station [A-Z]{2}-(\d{2})", legacy["label"])
        assert int(legacy["label"].split("-")[1]) == number - POINT_NUMBER_OFFSET  # the same draw
    taken = {f"station {POINT_NUMBER_OFFSET + n}" for n in range(10, 100)}
    again = _draw_point(records, random.Random(0), taken, spatial, strict=True)
    assert again["label"] not in taken and int(again["label"].split()[-1]) >= POINT_NUMBER_OFFSET + 100
    sign = re.compile(r"\b(?:" + "|".join(w for words in CALL_SIGNS.values() for w in words) + r") (\d{1,3})\b",
                      re.IGNORECASE)
    labels = calls = 0
    for inst in generated(tmp_path, "earthquakes:radio", 9313, 4):
        signs = {int(n) for n in sign.findall(inst["renderings"][0]["digest"])}
        calls += len(signs)
        for q in inst["questions"]:
            for step in q["chain"]:
                if "label" in step:
                    labels += 1
                    assert int(step["label"].split()[-1]) not in signs, (step["label"], sorted(signs))
    assert labels >= 1 and calls >= 4 and max(signs) < POINT_NUMBER_OFFSET


def test_a_questions_identity_holds_the_unit_and_rounding_of_its_answer():
    """The same filters and terminal asked in another unit are another question (fingerprint): a later instance of
    a run may ask "the shallowest depth in Tonga, in miles" after "in km", about its own records."""
    km = [{"op": "dedupe"}, {"op": "in_region", "region": "Tonga"}, {"op": "min_depth"}, {"op": "convert", "to": "km"},
          {"op": "round", "ndigits": 1}]
    mi = [*km[:3], {"op": "convert", "to": "mi"}, {"op": "round", "ndigits": 1}]
    assert fingerprint(km) != fingerprint(mi) and fingerprint(km) == fingerprint([dict(s) for s in km])
    assert "unit" in fingerprint.__doc__ and "rounding" in fingerprint.__doc__
