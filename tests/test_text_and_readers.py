"""Fixes from the 0.3.0 audit of misleading text and from the late checks of blind readers (chat, form
and radio prompts), each pinned where code says or prints it:

  - the command line describes what evalgen does now, demo says when it goes online, and the next step
    after generating a run with plugins names the private run config it needs;
  - a replay is named and recorded as a replay: it keeps the time its frozen bytes were fetched, and
    says when its seed differs from the run it replays (whose synthetic domains it then draws anew);
  - reports and logs of runs off the legacy path count records, not events;
  - no value answer rests on how a tie is rounded, and HH:MM answers say that seconds are dropped;
  - chatter claims nothing that a digest's reports could contradict, and says nothing false of units;
  - radio transcripts name no "Dispatch" that is not the net's control, and the session line says what
    the reports are about; forms are numbered in filing order and cite the original under one label;
  - the reading rules call a re-report a duplicate, say which record a relayed report is, and define
    the mile wherever distances can be asked in miles.

Legacy prompts, reports and chatter keep their 0.2.0 text (tests/test_regression_020.py pins them).
"""
from __future__ import annotations

import json
import random
import re
import secrets
from pathlib import Path

import pytest

import cli
from evalgen import pipeline
from evalgen.digest import CHATTER as CHATTER_020
from evalgen.digest import make_chatter
from evalgen.domains import get_domain
from evalgen.domains.base import GENERIC_CHATTER, REPEAT_RULE, Record, standard_lexicon
from evalgen.domains.invented import CHATTER as INVENTED_CHATTER
from evalgen.domains.invented import SECONDARY, drawn_unit
from evalgen.domains.neo import CHATTER as NEO_CHATTER
from evalgen.domains.weather import CHATTER as WEATHER_CHATTER
from evalgen.ops import execute
from evalgen.pipeline import GenerateConfig, run_name
from evalgen.prompt import ANSWER_SPECS, KIND_SPECS
from evalgen.questions import strict_answer_ok
from evalgen.report import load_instances
from evalgen.util import read_json

from conftest import ROOT, TOY_PLUGIN, quiet

NEO_FEED = ROOT / "tests" / "fixtures" / "neo" / "cad_2026-11-08.json"
TOY_GENRE_PLUGIN = ROOT / "tests" / "toy_genre.py"


# ------------------------------------------------------------------------------------ command line


def test_the_command_line_describes_every_domain_and_when_a_demo_goes_online(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    top = " ".join(capsys.readouterr().out.split())
    assert "on earthquake digests." not in top and "several domains in several genres" in top
    with pytest.raises(SystemExit):
        cli.main(["demo", "--help"])
    demo = " ".join(capsys.readouterr().out.split())
    assert "fully offline" not in top + demo and "offline with the default synthetic source" in top
    assert "offline with the default synthetic source" in demo and "usgs and live fetch each feed once" in demo
    with pytest.raises(SystemExit):
        cli.main(["generate", "--help"])
    generate = " ".join(capsys.readouterr().out.split())
    assert "whatever --source says" in generate and "drawn again from --seed" in generate
    assert "--source live --snapshot-dir" not in (ROOT / "cli.py").read_text(encoding="utf-8")  # the replay example
    assert "Everything offline" not in pipeline.demo.__doc__


def test_the_next_step_after_a_run_with_plugins_names_its_private_config(tmp_path, capsys):
    config = tmp_path / "private.json"
    config.write_text(json.dumps({"nonce": secrets.token_hex(16),
                                  "domains": {"public": ["earthquakes"], "held_out": ["toy"]},
                                  "genres": {"public": ["classic"], "held_out": ["toy-telex"]},
                                  "plugins": [str(TOY_PLUGIN), str(TOY_GENRE_PLUGIN)]}), encoding="utf-8")
    out = tmp_path / "run"
    assert cli.main(["generate", "--seed", "11", "--instances", "1", "--mix", "random", "--private-config",
                     str(config), "--out", str(out)]) == 0
    hint = capsys.readouterr().out.strip().splitlines()[-1]
    assert hint == f"next: python cli.py run {out} --private-config {config}"
    plain = tmp_path / "plain"
    assert cli.main(["generate", "--seed", "11", "--instances", "1", "--out", str(plain)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"next: python cli.py run {plain}"


# ------------------------------------------------------------------------------------------ replays


def test_a_replay_is_named_as_one_and_whatever_source_it_is_given():
    assert run_name(GenerateConfig(seed=7)) == "synthetic-seed7"  # legacy, as in 0.2.0
    assert run_name(GenerateConfig(seed=7, snapshot_file="feed.geojson")) == "synthetic-seed7"  # a 0.2.0 replay
    for source in ("synthetic", "live"):
        assert run_name(GenerateConfig(seed=7, mix="random", source=source, snapshot_dir="x")) == "random-replay-seed7"
        assert run_name(GenerateConfig(seed=7, mix="neo:news", source=source, snapshot_file="cad.json")) == (
            "neo_news-replay-seed7")
    assert run_name(GenerateConfig(seed=7, mix="neo:news", source="live")) == "neo_news-live-seed7"


def test_a_replay_keeps_the_fetch_time_and_says_when_its_seed_is_another(tmp_path):
    """A live run (here the frozen CAD file of the neo fixture, with invented beside it, which has no live
    source) replays from its snapshot folder: neo from its frozen bytes, fetched when the run fetched them
    (not when the replay ran), and invented drawn again from the seed. With another seed the replay says
    that invented is drawn anew, and records it (replay.same_seed)."""
    first = pipeline.generate(GenerateConfig(seed=31, instances=2, mix="neo:news", snapshot_file=str(NEO_FEED)),
                              tmp_path / "first", log=quiet)
    assert read_json(first / "snapshot" / "snapshot.json")["fetched_at"] is None  # a bare frozen file: unknown
    sidecar = first / "snapshot" / "snapshot.json"
    meta = read_json(sidecar)
    sidecar.write_text(json.dumps({**meta, "fetched_at": "2026-11-08T02:40:00Z"}), encoding="utf-8")
    mix = "neo:news,invented:chat"
    frozen = tmp_path / "frozen"
    (frozen / "neo").mkdir(parents=True)
    for name in ("snapshot.json", "raw_cad.json", "events.json"):
        (frozen / "neo" / name).write_bytes((first / "snapshot" / name).read_bytes())
    invented = pipeline.generate(GenerateConfig(seed=31, instances=1, mix="invented:chat"), tmp_path / "inv",
                                 log=quiet)
    (frozen / "invented").mkdir()
    for name in ("snapshot.json", "events.json"):
        (frozen / "invented" / name).write_bytes((invented / "snapshot" / name).read_bytes())

    logs: list[str] = []
    same = pipeline.generate(GenerateConfig(seed=31, instances=2, mix=mix, snapshot_dir=str(frozen)),
                             tmp_path / "same", log=logs.append)
    snapshots = read_json(same / "manifest.json")["snapshots"]
    assert snapshots["neo"]["fetched_at"] == "2026-11-08T02:40:00Z"  # as frozen, not the time of the replay
    assert read_json(same / "manifest.json")["replay"]["same_seed"] is None  # the folder is no run of its own
    assert not any("another seed" in line for line in logs)

    source = pipeline.generate(GenerateConfig(seed=31, instances=2, mix=mix, snapshot_dir=str(frozen)),
                               tmp_path / "source", log=quiet)
    logs.clear()
    again = pipeline.generate(GenerateConfig(seed=31, instances=2, mix=mix, snapshot_dir=str(source / "snapshot")),
                              tmp_path / "again", log=logs.append)
    assert read_json(again / "manifest.json")["replay"]["same_seed"] is True
    other = pipeline.generate(GenerateConfig(seed=32, instances=2, mix=mix, snapshot_dir=str(source / "snapshot")),
                              tmp_path / "other", log=logs.append)
    assert read_json(other / "manifest.json")["replay"]["same_seed"] is False
    warning = [line for line in logs if "another seed" in line]
    assert len(warning) == 1 and "(invented)" in warning[0]
    for name, same_bytes in (("neo", True), ("invented", False)):
        a, b = ((run / "snapshot" / name / "events.json").read_bytes() for run in (source, other))
        assert (a == b) is same_bytes


# ------------------------------------------------------------------------------- records, not events


def test_runs_off_the_legacy_path_count_records_and_legacy_ones_events(tmp_path, small_run):
    logs: list[str] = []
    run = pipeline.generate(GenerateConfig(seed=5, instances=2, mix="neo:table"), tmp_path / "neo", log=logs.append)
    assert any(re.search(r"snapshot: synthetic, \d+ records,", line) for line in logs)
    pipeline.run_models(run, ["oracle"], log=quiet)
    pipeline.score_run(run, log=quiet)
    pipeline.build_report(run, log=quiet)
    assert re.search(r"\d+ records in the snapshot pool", (run / "report.md").read_text(encoding="utf-8"))
    assert re.search(r"\d+ events in the snapshot pool", (small_run / "report.md").read_text(encoding="utf-8"))


def test_a_run_without_forecasts_says_which_runs_ask_them(tmp_path):
    run = pipeline.generate(GenerateConfig(seed=5, instances=1, mix="neo:table"), tmp_path / "neo", log=quiet)
    logs: list[str] = []
    assert pipeline.resolve_run(run, log=logs.append) == {}
    assert "--snapshot-file or --snapshot-dir" in logs[0] and "--source live with earthquakes" in logs[0]


# ------------------------------------------------------------------------------------------ answers


def test_no_value_answer_rests_on_how_a_tie_is_rounded():
    """68.3 in a unit of 4 is exactly 17.075, halfway between two answers to 2 places: binary floating
    point rounds it down (17.07), and half up or half even give 17.08. Off the legacy path such a question
    is redrawn (questions.strict_answer_ok); 68.2, which is exactly 17.05, is kept."""
    invented = get_domain("invented")
    domain = invented.bind(invented.draw_surface(random.Random(6)))  # its second field has a unit of 4
    spec = domain.fields[1]
    unit = next(u for u in spec.units if u.scale == 4.0)
    assert unit == drawn_unit(SECONDARY, unit.symbol, unit.words, 4.0) and unit.places(spec.decimals) == 2
    chain = [{"op": "dedupe"}, {"op": "max", "field": spec.key}, {"op": "convert", "to": unit.symbol},
             {"op": "round", "ndigits": unit.places(spec.decimals)}]

    def record(value: float) -> Record:
        return Record(id="ab-12345", time=1_779_990_000, lat=None, lon=None, group="g", source="s",
                      values={domain.fields[0].key: 10.0, spec.key: value}, qualifiers={}, place=None)

    assert execute(chain, [record(68.3)], units=domain.units) == 17.07  # the binary rounding of 17.075
    assert not strict_answer_ok(chain, [record(68.3)], domain=domain)
    assert strict_answer_ok(chain, [record(68.2)], domain=domain)  # exactly 17.05


def test_a_time_answer_says_that_seconds_are_dropped_off_the_legacy_path():
    assert "seconds are dropped, not rounded" in KIND_SPECS["clock"]
    assert ANSWER_SPECS["utc_time"] == "a UTC time string \"HH:MM\" (24-hour clock)"  # 0.2.0, word for word


def test_time_of_max_questions_name_the_records_before_the_verb():
    for name in ("weather", "invented"):
        domain = get_domain(name)
        lexicon = domain.prompt_lexicon() if name == "weather" else standard_lexicon(domain)
        for op, wordings in lexicon.questions.items():
            if op.startswith("time_of_max"):
                assert all(w.startswith("Among the {records}, ") for w in wordings[:2]), (name, wordings)


# ------------------------------------------------------------------------------------------ chatter


def test_chatter_claims_nothing_that_the_reports_could_contradict():
    for lines in (GENERIC_CHATTER, WEATHER_CHATTER, NEO_CHATTER, INVENTED_CHATTER):
        for line in lines:
            assert "included" not in line or "Editor's note" in line, line  # no "nothing it sent then is included"
            assert "own units" not in line and "units their" not in line, line  # units vary from entry to entry
    assert any("{date}" in line for line in WEATHER_CHATTER)
    rng = random.Random(4)
    for _ in range(40):
        for line in make_chatter(rng, 1_780_000_000, ["KJFK"], WEATHER_CHATTER, get_domain("weather").words):
            if line.startswith("Sensor"):
                assert re.fullmatch(r"Sensor KJFK\.[A-Z]{3} was offline for maintenance from \d{2}:\d{2} UTC on "
                                    r"\d{1,2} [A-Z][a-z]+ 2026 until \d{2}:\d{2} UTC\.", line), line
    # The 0.2.0 lines are untouched, and draw as they did.
    assert CHATTER_020[0] == ("Station {station} was offline from {h1} to {h2} UTC for maintenance; its readings "
                              "are not included.")
    a, b = random.Random(9), random.Random(9)
    make_chatter(a, 1_780_000_000, ["us", "ci"])
    make_chatter(b, 1_780_000_000, ["us", "ci"], WEATHER_CHATTER, get_domain("weather").words)
    assert a.getstate() == b.getstate()


# ------------------------------------------------------------------------------------- radio and form


def _instances(domain: str, genre: str, seed: int, tmp_path: Path):
    run = pipeline.generate(GenerateConfig(seed=seed, instances=4, variants=1, mix=f"{domain}:{genre}"),
                            tmp_path / f"{domain}-{genre}-{seed}", log=quiet)
    return load_instances(run)


def test_a_radio_transcript_names_no_dispatch_but_its_control(tmp_path):
    seen = set()
    for seed, domain in enumerate(("neo", "weather", "invented", "earthquakes"), start=40):
        for inst in _instances(domain, "radio", seed, tmp_path):
            for rendering in inst.renderings:
                text = rendering.digest
                control = rendering.style["control"]
                seen.add(control)
                assert "Dispatch broadcast" not in text and "dispatch broadcasts" not in text
                assert not text.startswith("Dispatch log") and "traffic for the" not in text
                assert re.search(r"Session of .+: reports for the ", text)
                if control != "Dispatch":
                    assert "Dispatch" not in text and "DISPATCH" not in text
    assert len(seen) >= 2


def test_forms_are_numbered_in_filing_order_and_cite_the_original_alike(tmp_path):
    """An amendment is issued after the report it amends was sent, so its serial number is above the one it
    cites as the original, wherever the batch prints it; every frame labels that number alike."""
    checked = 0
    for seed, domain in enumerate(("weather", "neo", "invented", "earthquakes"), start=50):
        for inst in _instances(domain, "form", seed, tmp_path):
            for rendering in inst.renderings:
                text = rendering.digest
                assert "Original serial no." not in text and "Original ref. no." not in text
                for block in text.split("\n\n"):
                    if "AMENDMENT" not in block:
                        continue
                    own = re.search(r"(?:No\.|Serial no\.|· no\.|Ref\. no\.) (\d[\d/-]*\d)", block)
                    cited = re.search(r"Original form(?: no\.)?:?[ .]*(\d[\d/-]*\d)", block)
                    if own and cited:
                        assert _serial(own.group(1)) > _serial(cited.group(1)), block
                        checked += 1
    assert checked >= 3


def _serial(number: str) -> int:
    """A form's serial in any frame's style: 004417 (boxed), 26-07631 (ruled), 7468 (numbered), 04510/26 (dotted)."""
    ruled, dotted = re.fullmatch(r"\d{2}-(\d+)", number), re.fullmatch(r"(\d+)/\d{2}", number)
    return int((ruled or dotted).group(1)) if ruled or dotted else int(number)


# ------------------------------------------------------------------------------------------ rules


def test_the_rules_call_a_re_report_a_duplicate_and_define_the_mile():
    for name in ("weather", "neo", "invented"):
        rules = " ".join(get_domain(name).rules_text().split())
        assert "it duplicates" in rules and "it repeats" not in rules, name
        assert REPEAT_RULE in rules and "rule 1 still says which record that report is" in rules
    weather = " ".join(get_domain("weather").rules_text().split())
    assert "1 mile = 1.609344 km" in weather  # distances can be asked in miles
    assert "1 mile" not in " ".join(get_domain("neo").rules_text().split())  # neo has no positions
