"""One evaluation cycle, end to end.

    generate  snapshot -> instances (digest + traps + questions + key + metamorphic renderings)
    run       send every rendering's prompt to each model and store the raw replies
    score     parse replies (robust JSON) and grade every answer automatically
    report    accuracy, consistency, seen-vs-new gap, trap catch rate, audit -> report.md / .json
    resolve   later: grade pending forecast questions against the USGS catalogue

A run's mix (GenerateConfig.mix, see evalgen.mix) says which domains and genres its instances come
from; the default, legacy, is the 0.2.0 run: earthquakes in the classic genre. The evaluator's private
run config (GenerateConfig.private_config) holds some of them out; the manifest commits to it by hash
and lists its plugins, which every later step loads again.

Run directory layout:
    manifest.json          seeds, config, snapshot hash, templates, generator self-check (+ the inventory
                           of domains and genres off the legacy path, and the private config's commitment)
    snapshot/              events.json, snapshot.json (+ the frozen feed of a live source, raw_feed.geojson for USGS);
                           a run of several domains keeps one such folder per domain, snapshot/<domain>/
    instances/inst-*.json  full instance: reports, notices, traps, questions, key, renderings
    prompts/inst-*/v*.txt  the exact prompt texts sent to models
    responses/<model>/     raw replies         scores/<model>.json  graded rows
    audit/                 ~2% human spot-check sample
    forecasts.json         pending / resolved forecasts (USGS mode)
    report.md, report.json
"""
from __future__ import annotations

import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Sequence

import requests

from . import __version__
from .adapters import DEFAULT_MAX_TOKENS, AdapterError, MissingKey, Task, is_truncated, make_adapter, token_usage
from .answers import parse_reply
from .audit import write_audit_sample
from .digest import make_chatter
from .domains import get_domain
from .domains.base import Domain
from .domains.earthquakes import EARTHQUAKES
from .genres import detect_genre
from .genres.base import Genre, RenderContext
from .genres.classic import CLASSIC
from .mix import LEGACY, Inventory, MixError, Plan, load_run_plugins, plan_mix, plugin_record
from .models import Instance, Question, Rendering
from .ops import execute, role
from .prompt import build_prompt, domain_prompt
from .questions import Context, Template, compose_new_templates, compose_questions, phrase_question, public_templates
from .report import code_fingerprint, load_instances, read_run_json, read_scores, write_report
from .rules import READING_RULES
from .scoring import is_correct, normalize, same_answer
from .seeds import SeedLog, derive_seed, fresh_nonce, seed_commitment
from .solver import extract_digest, solve
from .sources import LIVE, Snapshot, build_domain_snapshot, build_snapshot, frozen_feed, read_frozen_feed, save_snapshot
from .traps import TrapConfig, inject_traps
from .util import read_json, read_manifest, sha256_hex, slugify, write_in, write_json

RUNS_DIR = Path("runs")
RUN_MARKER = ".evalgen-run"  # lets --force recognise (and only delete) evalgen run directories
Log = Callable[[str], None]


@dataclass
class GenerateConfig:
    seed: int
    instances: int = 20
    source: str = "synthetic"  # or a live source: "usgs" (legacy), "live" or the domain's own (any other mix)
    feed: str = "day"  # the USGS feed of a legacy run: "hour" or "day"
    snapshot_file: str | None = None  # frozen bytes of one live source, replayed
    variants: int = 2  # metamorphic re-renderings per instance (besides the original)
    pool_size: int = 360  # synthetic snapshot size
    events_min: int = 20  # records per digest
    events_max: int = 28
    questions_min: int = 3
    questions_max: int = 5
    audit_rate: float = 0.02
    # Which domains and genres the instances come from (evalgen.mix): "legacy" (0.2.0: earthquakes,
    # classic), a list such as "toy:classic" or "weather,neo:email", or "random"; and the evaluator's
    # private run config (held-out domains and genres, and the plugins that define them).
    mix: str = "legacy"
    private_config: str | None = None  # path of the private run config (JSON)
    # The frozen snapshots of an earlier run off the legacy path (its snapshot folder), which replay every
    # domain of the mix as that run read it: from its frozen live bytes, or synthetic (sources.frozen_feed).
    snapshot_dir: str | None = None

    OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = ("mix", "private_config", "snapshot_dir")  # recorded only when set
    PATHS: ClassVar[tuple[str, ...]] = ("snapshot_file", "private_config", "snapshot_dir")

    def __post_init__(self) -> None:
        """A path option given as a Path is kept as its text, as the command line gives it, which is what the
        manifest records."""
        for name in self.PATHS:
            if getattr(self, name) is not None:
                setattr(self, name, os.fspath(getattr(self, name)))

    def problems(self) -> list[str]:
        """What no run can be generated with: no instance, fewer than no variant, an empty snapshot pool,
        record and question ranges that are empty, or an audit rate that is not a share."""
        def whole(value: Any, least: int) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value >= least

        problems = [f"{name} must be a whole number of {least} or more, not {getattr(self, name)!r}"
                    for name, least in (("instances", 1), ("variants", 0), ("pool_size", 1), ("events_min", 1),
                                        ("events_max", 1), ("questions_min", 1), ("questions_max", 1))
                    if not whole(getattr(self, name), least)]
        for low, high in (("events_min", "events_max"), ("questions_min", "questions_max")):
            if not problems and getattr(self, low) > getattr(self, high):
                problems.append(f"{low} ({getattr(self, low)}) is more than {high} ({getattr(self, high)})")
        if isinstance(self.audit_rate, bool) or not isinstance(self.audit_rate, (int, float)) or not (
                0 <= self.audit_rate <= 1):
            problems.append(f"audit_rate must be a share between 0 and 1, not {self.audit_rate!r}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """The config as the manifest records it: asdict without the 0.3.0 options that keep their
        defaults, so that a legacy run records exactly the 0.2.0 config. The private run config is
        recorded by its file name alone: the manifest commits to its content (private_config_commitment),
        and its folder names a place on the evaluator's machine, which a published run should not."""
        defaults = {f.name: f.default for f in fields(self)}
        recorded = {k: v for k, v in asdict(self).items() if k not in self.OPTIONAL_KEYS or v != defaults[k]}
        if recorded.get("private_config") is not None:
            recorded["private_config"] = Path(recorded["private_config"]).name
        return recorded


# ------------------------------------------------------------------------------ generate


def _weighted_sample(items: Sequence[Any], k: int, rng: random.Random, weight: Callable[[Any], float]) -> list[Any]:
    """Weighted sampling without replacement (Efraimidis-Spirakis): heavier records are likelier."""
    keyed = sorted(((rng.random() ** (1.0 / weight(e)), i) for i, e in enumerate(items)), reverse=True)
    return [items[i] for _, i in keyed[:k]]


def sample_events(pool: Sequence[Any], rng: random.Random, n_min: int, n_max: int, *,
                  domain: Domain = EARTHQUAKES) -> list[Any]:
    """The records for one digest: mostly from 3-4 focus groups, so questions have substance. The
    domain says what a record's group is and how heavily it weighs (Domain.sample_group and
    sample_weight; earthquakes: its region, and 1.5 ** magnitude, so larger quakes are likelier)."""
    group, weight = domain.sample_group, domain.sample_weight
    n = min(len(pool), rng.randint(n_min, n_max))
    counts = Counter(group(e) for e in pool)
    eligible = sorted(r for r, c in counts.items() if c >= 4)
    focus = set(rng.sample(eligible, min(len(eligible), rng.randint(3, 4)))) if eligible else set()
    in_focus = [e for e in pool if group(e) in focus]
    others = [e for e in pool if group(e) not in focus]
    n_focus = min(len(in_focus), round(n * 0.8))
    chosen = (_weighted_sample(in_focus, n_focus, rng, weight)
              + _weighted_sample(others, min(len(others), n - n_focus), rng, weight))
    if len(chosen) < n:
        chosen_ids = {e.id for e in chosen}
        chosen += _weighted_sample([e for e in pool if e.id not in chosen_ids], n - len(chosen), rng, weight)
    return sorted(chosen, key=lambda e: (e.time, e.id))


def question_specs(inst: Instance) -> list[dict]:
    """What reference solvers get: each question's chain, never its answer."""
    return [{"qid": q.qid, "chain": q.chain, "answer_type": q.answer_type, "unit": q.unit, "pool": q.pool}
            for q in inst.questions]


def forecast_spec(inst: Instance) -> dict | None:
    return {"qid": inst.forecast["qid"], "rule": inst.forecast["rule"]} if inst.forecast else None


def _recorded(**labels: str | None) -> dict[str, str]:
    """The labels that are set: a non-legacy run records its domain and genres next to its scores
    and forecasts, while a legacy run records none, so its files keep their 0.2.0 keys."""
    return {name: value for name, value in labels.items() if value is not None}


@dataclass
class Composition:
    """What an instance is about, before its traps and renderings: its domain as the instance words it
    (bound to the instance's surface words, when the domain draws them), those words, the instance's
    records (the snapshot's, as the instance names them) and its questions with their answer key."""

    domain: Domain
    surface: dict[str, Any] | None
    events: list[Any]
    questions: list[Question]
    answer_key: dict[str, Any]


def compose_instance(index: int, snapshot: Snapshot, new_templates: Sequence[Template], seedlog: SeedLog,
                     cfg: GenerateConfig, ctx: Context, used_chains: set[str], usage: dict[str, int], *,
                     domain: Domain = EARTHQUAKES, legacy: bool = True) -> Composition:
    """An instance's records, questions and answer key. Off the legacy path, a domain with surface
    words draws them first (seed path "instance", i, "surface", which legacy runs never create) and
    the bound domain renames the sampled records (Domain.dress); everything after uses the bound
    domain and the renamed records, and every twin pair stays in step from the run's first question
    (compose_questions, strict_twins). The key is code over the snapshot's records."""
    surface = None
    if not legacy and domain.has_surface:
        surface = domain.draw_surface(seedlog.rng("instance", index, "surface"))
        domain = domain.bind(surface)
    sample = sample_events(snapshot.events, seedlog.rng("instance", index, "events"), cfg.events_min,
                           cfg.events_max, domain=domain)
    events = [domain.dress(e) for e in sample]
    q_rng = seedlog.rng("instance", index, "questions")
    questions = compose_questions(events, public_templates(domain), new_templates, q_rng, ctx,
                                  q_rng.randint(cfg.questions_min, cfg.questions_max), used_chains, usage,
                                  domain=domain, strict_twins=not legacy)
    answer_key = {q.qid: execute(q.chain, events, units=domain.units) for q in questions}
    return Composition(domain, surface, events, questions, answer_key)


def build_instance(index: int, snapshot: Snapshot, new_templates: Sequence[Template], seedlog: SeedLog,
                   cfg: GenerateConfig, ctx: Context, used_chains: set[str], usage: dict[str, int], *,
                   domain: Domain = EARTHQUAKES, genre_plan: Sequence[Genre] | None = None,
                   split: str | None = None) -> Instance:
    """One test item about `domain`: its records, questions, answer key (compose_instance), traps and
    renderings.

    `genre_plan` gives the genre of each rendering, the original first. None is the legacy path:
    classic everywhere, with the 0.2.0 calls, random draws and output. Otherwise the instance records
    its domain and `split` (whether the domain is public or held out in this run) and each rendering
    its genre, a variant avoids the original's style only when it is in the original's genre, and
    every prompt is a non-legacy one (prompt.domain_prompt), with the domain's glossary, which the
    genre renders the records through, and questions worded by Domain.prompt_lexicon. A domain with
    surface words draws them for the instance (only off the legacy path, so the legacy seed paths never
    change), and every step uses the bound domain."""
    if genre_plan is None and domain.name != EARTHQUAKES.name:
        raise ValueError(f"only earthquakes have a legacy rendering: {domain.name} needs a genre plan")
    if genre_plan is not None and len(genre_plan) != cfg.variants + 1:
        raise ValueError(f"the genre plan names {len(genre_plan)} genres for {cfg.variants + 1} renderings")
    composed = compose_instance(index, snapshot, new_templates, seedlog, cfg, ctx, used_chains, usage,
                                domain=domain, legacy=genre_plan is None)
    domain, events, questions, answer_key = composed.domain, composed.events, composed.questions, composed.answer_key
    units = domain.units
    glossary = None if genre_plan is None else domain.glossary()
    lexicon = None if genre_plan is None else domain.prompt_lexicon()  # None: the 0.2.0 wording
    # Off the legacy path a trap counts only careless readings that hold at the printed precision (traps).
    trap_config = TrapConfig(id_style="sequence" if snapshot.source == "usgs" else "random",
                             guarded=genre_plan is not None)
    reports, notices, traps = inject_traps(events, questions, seedlog.rng("instance", index, "traps"),
                                           compiled_at=snapshot.compiled_at, window_hours=snapshot.window_hours,
                                           config=trap_config, domain=domain)
    for q in questions:
        unfiltered = [s for s in q.chain if role(s["op"]) != "filter"]
        q.features = {
            "trap_exposure": min(2, sum(1 for t in traps if q.qid in t.affected)),
            "filters_matter": not same_answer(q.answer_type, execute(unfiltered, events, units=units),
                                              answer_key[q.qid], unit=q.unit),
        }
    chatter = make_chatter(seedlog.rng("instance", index, "chatter"), snapshot.compiled_at,
                           sorted({e.source for e in events}), domain.chatter, domain.words)
    forecaster = domain.forecaster
    forecast = None
    if forecaster is not None and snapshot.source == forecaster.source:
        forecast = forecaster.make(events, seedlog.rng("instance", index, "forecast"), snapshot.compiled_at,
                                   snapshot.window_hours)

    renderings: list[Rendering] = []
    original: tuple[str, dict] | None = None  # the genre and style of the original rendering
    for v in range(cfg.variants + 1):
        seed = seedlog.seed("instance", index, "render", v)
        rng = random.Random(seed)
        genre = CLASSIC if genre_plan is None else genre_plan[v]
        avoid = original[1] if original is not None and original[0] == genre.name else None
        style = genre.choose_style(rng, domain, avoid=avoid)
        digest = genre.render(RenderContext(reports, notices, chatter, compiled_at=snapshot.compiled_at,
                                            window_hours=snapshot.window_hours, style=style, rng=rng, domain=domain,
                                            glossary=glossary))
        texts = {q.qid: phrase_question(q, rng, alt_units=v > 0 and rng.random() < 0.5, domain=domain,
                                        lexicon=lexicon)
                 for q in questions}
        items = [(q.qid, texts[q.qid], q.answer_type, q.unit) for q in questions]
        if forecast:
            items.append((forecast["qid"], forecaster.text(forecast), "probability", None))
        if glossary is None:  # the legacy (0.2.0) prompt, with the 0.2.0 reading rules
            prompt = build_prompt(digest, items, rules=READING_RULES)
        else:
            prompt = domain_prompt(domain, digest, items, glossary)
        renderings.append(Rendering(variant=f"v{v}", seed=seed, style=style, digest=digest, question_text=texts,
                                    prompt=prompt, prompt_sha256=sha256_hex(prompt),
                                    genre=None if genre_plan is None else genre.name))
        original = original or (genre.name, style)

    return Instance(
        instance_id=f"inst-{index:03d}", seed=seedlog.seed("instance", index), compiled_at=snapshot.compiled_at,
        window_hours=snapshot.window_hours, event_ids=[e.id for e in events], reports=reports, notices=notices,
        chatter=chatter, traps=traps, questions=questions, answer_key=answer_key, renderings=renderings,
        forecast=forecast, domain=None if genre_plan is None else domain.name,
        split=None if genre_plan is None else split, surface=composed.surface,
    )


def generator_self_check(instances: Sequence[Instance], readers: Sequence[Genre] | None = None) -> dict:
    """The generator audits itself: a solver that sees ONLY the prompt text must reproduce the key.
    Each rendering is read in the domain the run recorded for its instance (a legacy run records none,
    and its prompts are read as 0.2.0 read them), by the genre that recognises the digest, which must
    be the genre it was rendered in; through the glossary the prompt prints, for a non-legacy run.
    The genre is sniffed among `readers` when they are given: the run's own (mix.Plan.readers), so a
    genre that this process registered for something else, such as an earlier run's private plugin,
    never changes the check or appears in it. Without them, among every registered genre."""
    checks = agreements = 0
    problems: list[str] = []
    for inst in instances:
        specs = question_specs(inst)
        for rendering in inst.renderings:
            got = solve(rendering.prompt, specs, mode="oracle", domain=inst.domain, readers=readers)
            if rendering.genre is not None:
                sniffed = detect_genre(extract_digest(rendering.prompt), readers).name
                if sniffed != rendering.genre:
                    problems.append(f"{inst.instance_id}/{rendering.variant}: a {rendering.genre} digest is "
                                    f"recognised as {sniffed}")
            for q in inst.questions:
                checks += 1
                if same_answer(q.answer_type, got[q.qid], inst.answer_key[q.qid], unit=q.unit):
                    agreements += 1
                else:
                    problems.append(f"{inst.instance_id}/{rendering.variant}/{q.qid}: digest-only oracle "
                                    f"{got[q.qid]!r} vs key {inst.answer_key[q.qid]!r}")
    # inject_traps raises if the reading rules ever fail to recover the snapshot, so reaching
    # this point means the invariant held for every instance.
    return {"oracle_checks": checks, "oracle_agreements": agreements,
            "resolution_invariant_instances": len(instances), "problems": problems[:50]}


def is_run_dir(path: Path) -> bool:
    """True only for directories evalgen created (its marker file, or its own manifest)."""
    if (path / RUN_MARKER).exists():
        return True
    try:
        manifest = read_json(path / "manifest.json")
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and {"evalgen_version", "run_id", "self_check"} <= set(manifest)


def _instance_records(instances: Sequence[Instance], snapshots: Mapping[str, Snapshot],
                      domains: Mapping[str, Domain]) -> dict[str, dict[str, Any]]:
    """Each instance's records by ID: those of its domain's snapshot (a legacy instance names no domain:
    it is about earthquakes), as the instance names them (renamed by its surface words, when it has them)."""
    by_id = {name: {e.id: e for e in snapshot.events} for name, snapshot in snapshots.items()}
    records = {}
    for inst in instances:
        name = inst.domain or EARTHQUAKES.name
        records[inst.instance_id] = by_id[name] if inst.surface is None else {
            r.id: r for r in map(domains[name].bind(inst.surface).dress, snapshots[name].events)}
    return records


def _replay_check(frozen: str | None, log: Log) -> dict | None:
    """When replaying another run's frozen feed (a file of its snapshot folder) or its frozen snapshots
    (the folder), note whether that run used the same code."""
    if not frozen:
        return None
    path = Path(frozen).resolve()
    source_run = next((folder for folder in [path, *path.parents][:4]
                       if is_run_dir(folder) and (folder / "manifest.json").exists()), None)
    if source_run is None:
        return {"source_run": None, "same_code": None}
    same = read_json(source_run / "manifest.json").get("code_sha256") == code_fingerprint()
    if not same:
        log("[generate] warning: the source run was generated with different code, so a replay "
            "with the same seed will not reproduce its instances exactly")
    return {"source_run": str(source_run), "same_code": same}


def run_name(cfg: GenerateConfig) -> str:
    """The default name of a run's directory: <source>-seed<seed>, and for any mix but legacy the mix
    first ("random-synthetic-seed7"), so that it never takes the legacy run's place."""
    name = f"{cfg.source}-seed{cfg.seed}"
    return name if cfg.mix.strip() == LEGACY else f"{slugify(cfg.mix)}-{name}"


def _check_sources(plan: Plan, cfg: GenerateConfig) -> dict[str, Path | None]:
    """Refuse, before anything is written, sources that a run off the legacy path cannot use: the USGS
    hour feed, a frozen snapshot file for several domains, frozen snapshots that lack a domain of the
    mix, a live source that a domain lacks, and frozen bytes that its domain's live source cannot replay
    (a domain without one, or another source's bytes: _check_frozen). With frozen snapshots
    (cfg.snapshot_dir), return what replays each domain (sources.frozen_feed: its frozen bytes, or None
    for synthetic records)."""
    if cfg.feed != GenerateConfig.feed:
        raise MixError("the USGS feed is chosen in legacy runs only: off the legacy path its live source is the "
                       "day feed")
    if cfg.snapshot_file and cfg.snapshot_dir:
        raise MixError("give a frozen snapshot file or the frozen snapshots of a run, not both")
    if cfg.snapshot_file and len(plan.domains) > 1:
        raise MixError("a frozen snapshot file replays the live source of one domain, and this mix draws several: "
                       "replay the frozen snapshots of a run of several domains with --snapshot-dir RUN/snapshot")
    for domain in plan.domains:
        if cfg.source not in ("synthetic", LIVE) and (domain.live is None or domain.live.source != cfg.source):
            raise MixError(f"unknown source {cfg.source!r} for {domain.name}: use synthetic or {LIVE}")
    if cfg.snapshot_file:
        _check_frozen(plan.domains[0], cfg.snapshot_file)
    if not cfg.snapshot_dir:
        return {}
    if not Path(cfg.snapshot_dir).is_dir():
        raise FileNotFoundError(f"the frozen snapshots {cfg.snapshot_dir} do not exist")
    try:
        feeds = {domain.name: frozen_feed(cfg.snapshot_dir, domain) for domain in plan.domains}
    except ValueError as exc:
        raise MixError(str(exc)) from None
    for domain in plan.domains:
        if feeds[domain.name] is not None:
            _check_frozen(domain, feeds[domain.name])
    return feeds


def _check_frozen(domain: Domain, path: str | Path) -> None:
    """Refuse, before anything is written, frozen bytes that `domain` cannot replay: MixError for a domain
    without a live source, or bytes that are not its live source's (sources.read_frozen_feed);
    FileNotFoundError for a file that is not there."""
    try:
        read_frozen_feed(domain, path)
    except ValueError as exc:
        raise MixError(f"the frozen feed cannot be replayed: {exc}") from None


def _inputs_in(run_dir: Path, cfg: GenerateConfig, plan: Plan, frozen: Mapping[str, Path | None]) -> list[str]:
    """What a run reads that lies in the folder `run_dir`, as given: the frozen feed or the frozen snapshots it
    replays (`frozen`: each domain's feed, from _check_sources), with the snapshot.json beside each feed, its
    private run config and the config's plugins. Replacing `run_dir` would delete them: the frozen bytes
    before they are read (a run folder may hold the only copy of a live feed), the config after the new
    manifest has committed to it. A path is in `run_dir` as written or as it resolves: deleting the folder
    removes the entry below it, and the file that a link elsewhere points to below it."""
    feeds = [Path(cfg.snapshot_file)] if cfg.snapshot_file else []
    feeds += [Path(feed) for feed in frozen.values() if feed is not None]
    inputs = [Path(p) for p in (cfg.snapshot_dir, cfg.private_config) if p is not None]
    inputs += feeds + [feed.with_name("snapshot.json") for feed in feeds]
    if plan.config is not None:
        named = [e if isinstance(e, str) else e["path"] for e in plan.config.data.get("plugins", [])]
        inputs += [plan.config.path.parent / name for name in named] + list(plan.config.plugins)

    def within(path: Path) -> bool:
        return any(Path(where(path)).is_relative_to(where(run_dir)) for where in (os.path.abspath, os.path.realpath))

    return list(dict.fromkeys(str(path) for path in inputs if within(path)))


def _snapshot_entry(snapshots: Mapping[str, Snapshot]) -> dict[str, Any]:
    """The manifest's record of a run's snapshots: "snapshot", as in 0.2.0, for a run of one domain;
    "snapshots", by domain, for a run of several."""
    if len(snapshots) == 1:
        return {"snapshot": next(iter(snapshots.values())).meta()}
    return {"snapshots": {name: snapshot.meta() for name, snapshot in snapshots.items()}}


def _templates_entry(plan: Plan, new_templates: Mapping[str, Sequence[Template]]) -> dict[str, list[dict]]:
    """The manifest's record of a run's templates: P01-P12 and their twins, each tagged with its
    domain in a run of several domains (P01 is one skeleton in every domain; its twin differs)."""
    if plan.legacy or len(plan.domains) == 1:
        new, = new_templates.values()
        domain = EARTHQUAKES if plan.legacy else plan.domains[0]
        return {"public": [t.to_dict() for t in public_templates(domain)], "new": [t.to_dict() for t in new]}
    return {"public": [dict(t.to_dict(), domain=d.name) for d in plan.domains for t in public_templates(d)],
            "new": [dict(t.to_dict(), domain=d.name) for d in plan.domains for t in new_templates[d.name]]}


def _mix_entry(plan: Plan) -> dict[str, Any]:
    """What the manifest records about the mix off the legacy path: the run's inventory and, when a
    private run config was given, its commitment, whether it keeps a published EXAMPLE's nonce, and the
    plugins that later steps load again."""
    if plan.legacy:
        return {}
    entry: dict[str, Any] = {"inventory": plan.inventory.to_dict()}
    if plan.config is not None:
        entry["private_config_commitment"] = plan.config.commitment  # publishable, like the seed commitment
        if plan.config.example:  # a published example's config: for demos, its held-out sets public
            entry["private_config_example"] = True
        entry["plugins"] = plugin_record(plan.config)
    return entry


def generate(cfg: GenerateConfig, run_dir: Path | None = None, *, force: bool = False, log: Log = print) -> Path:
    """Write a run: snapshots, instances, prompts, the generator's self-check, the audit sample and
    the manifest. The run's mix (cfg.mix) says which domains and genres its instances are drawn from,
    and the private run config (cfg.private_config) which of them are held out; the default is the
    legacy (0.2.0) run, and any other mix is generated on seed paths of its own (see evalgen.mix). The
    mix and the config are read, and their plugins loaded, before anything is written. A run off the
    legacy path replays an earlier run's frozen snapshots (cfg.snapshot_dir), one per domain, so that a
    live run of several domains regenerates exactly from its seed, without the network.

    Every file goes through the run directory's one writer (util.write_in). An existing run directory is
    replaced only with `force`, only if evalgen made it, and never while it holds what this run reads (a
    frozen feed or snapshot folder, the private run config or a plugin), which replacing it would delete."""
    problems = cfg.problems()
    if problems:
        raise ValueError("cannot generate this run: " + "; ".join(problems))
    plan = plan_mix(cfg.mix, cfg.private_config)
    frozen: dict[str, Path | None] = {}
    if not plan.legacy:
        frozen = _check_sources(plan, cfg)
    elif cfg.source == LIVE:
        raise MixError(f"a legacy run reads synthetic data or the USGS feed: the {LIVE} source of each domain is "
                       "read off the legacy path (give a mix)")
    elif cfg.snapshot_dir:
        raise MixError("a legacy run replays a frozen USGS feed with --snapshot-file; --snapshot-dir replays the "
                       "frozen snapshots of a run off the legacy path")
    elif cfg.snapshot_file:  # the frozen USGS feed of a legacy run, read as the replay reads it
        _check_frozen(EARTHQUAKES, cfg.snapshot_file)
    run_dir = Path(run_dir) if run_dir else RUNS_DIR / run_name(cfg)
    if run_dir.exists() and not run_dir.is_dir():
        raise FileExistsError(f"{run_dir} is a file, not a run directory")
    if run_dir.exists() and any(run_dir.iterdir()):
        held = _inputs_in(run_dir, cfg, plan, frozen)
        if held:
            raise FileExistsError(f"{run_dir} holds what this run reads ({', '.join(held)}), which replacing it would "
                                  "delete: write the run to another folder with --out")
        if not force:
            raise FileExistsError(f"{run_dir} already exists (use --force to overwrite, or --out)")
        if not is_run_dir(run_dir):
            raise FileExistsError(f"{run_dir} is not an evalgen run directory; refusing to delete it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_in(run_dir, run_dir / RUN_MARKER, "evalgen run directory\n")

    seedlog = SeedLog(cfg.seed)
    nonce = fresh_nonce()
    commitment = seed_commitment(cfg.seed, nonce)
    # The commitments in full: what the evaluator publishes before the evaluation is the whole value (a prefix
    # of 16 hexadecimal digits could be matched by another seed or config in about 2**32 tries).
    log(f"[generate] master seed {cfg.seed} (publishable commitment {commitment}: seed_commitment in manifest.json)")
    if plan.config is not None:
        log(f"[generate] private run config committed as {plan.config.commitment} (publishable: "
            "private_config_commitment in manifest.json)")
        if plan.config.example:
            log("[generate] WARNING: the private run config keeps the nonce of a published EXAMPLE, so anyone can "
                "recover which domains and genres it holds out from its commitment. Use it for demos only: a real "
                "config needs a nonce of its own (python -c \"import secrets; print(secrets.token_hex(16))\").")
    domains = [EARTHQUAKES] if plan.legacy else plan.domains
    snapshots: dict[str, Snapshot] = {}
    for domain in domains:
        if plan.legacy:  # the 0.2.0 snapshot and seed paths
            snapshot, raw = build_snapshot(cfg.source, seedlog, feed=cfg.feed, pool_size=cfg.pool_size,
                                           snapshot_file=cfg.snapshot_file)
        elif cfg.snapshot_dir:  # replay each domain as the frozen run read it: its frozen bytes, or synthetic
            feed = frozen[domain.name]
            snapshot, raw = build_domain_snapshot(domain, "synthetic" if feed is None else LIVE, seedlog,
                                                  pool_size=cfg.pool_size,
                                                  snapshot_file=None if feed is None else str(feed))
        else:
            snapshot, raw = build_domain_snapshot(domain, cfg.source, seedlog, pool_size=cfg.pool_size,
                                                  snapshot_file=cfg.snapshot_file)
        folder = run_dir / "snapshot" if len(domains) == 1 else run_dir / "snapshot" / domain.name
        if raw is None or plan.legacy:
            save_snapshot(snapshot, raw, folder, root=run_dir)
        else:  # frozen bytes of the domain's live source
            save_snapshot(snapshot, raw, folder, raw_filename=domain.live.raw_filename, root=run_dir)
        snapshots[domain.name] = snapshot
        about = "" if len(domains) == 1 else f" ({domain.name})"
        if snapshot.note:
            log(f"[generate] note{about}: {snapshot.note}")
        log(f"[generate] snapshot{about}: {snapshot.source}, {len(snapshot.events)} events, "
            f"sha256 {snapshot.events_sha256[:16]}…")

    replay = _replay_check(cfg.snapshot_file or cfg.snapshot_dir, log)
    if plan.legacy:
        new_templates = {EARTHQUAKES.name: compose_new_templates(seedlog.rng("templates", "new"), EARTHQUAKES)}
    else:
        new_templates = {d.name: compose_new_templates(seedlog.rng("templates", "new", d.name), d) for d in domains}
    contexts = {name: Context(s.compiled_at, s.window_hours) for name, s in snapshots.items()}
    used_chains: set[str] = set()  # no concrete question is ever asked twice in a run
    # Questions per template so far, per domain: keeps each domain's public/twin pairs in step, so that
    # the template pools stay balanced within every domain (and so between public and held-out ones).
    usage: dict[str, dict[str, int]] = {name: {} for name in snapshots}
    instances = []
    for index in range(cfg.instances):
        drawn = plan.draw(index, seedlog, cfg.variants + 1)
        domain = EARTHQUAKES if drawn is None else drawn.domain
        inst = build_instance(index, snapshots[domain.name], new_templates[domain.name], seedlog, cfg,
                              contexts[domain.name], used_chains, usage[domain.name], domain=domain,
                              genre_plan=None if drawn is None else drawn.genres,
                              split=None if drawn is None else drawn.split)
        instances.append(inst)
        write_json(run_dir / "instances" / f"{inst.instance_id}.json", inst.to_dict(), root=run_dir)
        for rendering in inst.renderings:
            write_in(run_dir, run_dir / "prompts" / inst.instance_id / f"{rendering.variant}.txt", rendering.prompt)

    self_check = generator_self_check(instances, plan.readers or None)
    records = _instance_records(instances, snapshots, {d.name: d for d in domains})
    audit = write_audit_sample(run_dir, instances, records, seedlog.rng("audit"), cfg.audit_rate)
    forecasts = [dict(inst.forecast, instance_id=inst.instance_id, **_recorded(domain=inst.domain))
                 for inst in instances if inst.forecast]
    if forecasts:
        write_json(run_dir / "forecasts.json", forecasts, root=run_dir)
    seedlog.seed("mock")  # the simulated model's seed, so it is logged too
    write_json(run_dir / "manifest.json", {
        "evalgen_version": __version__,
        "code_sha256": code_fingerprint(),
        "run_id": run_dir.name,
        "seed": cfg.seed,
        "seed_commitment": commitment,  # publishable before the evaluation
        "seed_nonce": nonce,  # secret until the reveal, like the seed itself (this file is private)
        "config": replace(cfg, mix=plan.mix).to_dict(),  # the mix in canonical form: " legacy " records none
        **_mix_entry(plan),
        **_snapshot_entry(snapshots),
        "replay": replay,
        "templates": _templates_entry(plan, new_templates),
        "instances": [inst.instance_id for inst in instances],
        "self_check": self_check,
        "audit": audit,
        "seeds": seedlog.seeds,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, root=run_dir)
    n_questions = sum(len(inst.questions) for inst in instances)
    log(f"[generate] {len(instances)} instances, {n_questions} questions, {cfg.variants + 1} renderings each; "
        f"oracle self-check {self_check['oracle_agreements']}/{self_check['oracle_checks']}; "
        f"audit sample {audit['sampled']}/{audit['population']} -> {run_dir}")
    return run_dir


# ----------------------------------------------------------------------------------- run


def run_models(run_dir: Path, specs: Sequence[str], *, workers: int = 4, force: bool = False,
               max_tokens: int = DEFAULT_MAX_TOKENS, private_config: str | Path | None = None,
               log: Log = print) -> list[str]:
    """Ask each model (an adapter spec) every rendering's prompt, keeping replies already stored. A
    run's plugins are loaded first, from its private run config (mix.load_run_plugins); the simulated
    model also learns whether each instance's domain and each rendering's genre are public or held out
    (Task.split, Task.genre_split).

    Every spec is read before any model runs: one that names no model, or options that its model does
    not take, is refused (adapters.UnknownModel, a ValueError). An API model whose key is not set is
    skipped with a note, and a run in which no model is left to ask raises ValueError, so a typing slip
    never passes for a finished run."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    load_run_plugins(manifest, private_config, log=log)
    inventory = Inventory.of_run(manifest)
    instances = load_instances(run_dir, manifest)
    adapters, skipped = [], []
    for spec in specs:
        try:
            adapters.append(make_adapter(spec, seed=derive_seed(manifest["seed"], "mock"), max_tokens=max_tokens))
        except MissingKey as exc:
            log(f"[run] skipping {spec}: {exc}")
            skipped.append(f"{spec}: {exc}")
    if not adapters:
        raise ValueError("no model to ask" + (f" ({'; '.join(skipped)})" if skipped else
                                              ": name one or more with --models"))
    finished = []
    for adapter in adapters:
        note = adapter.note()  # e.g. a simulated penalty that a chance of 0 cuts short
        if note:
            log(f"[run] note: {note}")
        out_dir = run_dir / "responses" / slugify(adapter.name)
        jobs = [(inst, r) for inst in instances for r in inst.renderings
                if force or not _has_reply(out_dir / f"{inst.instance_id}__{r.variant}.json")]

        def answer(job: tuple[Instance, Rendering]) -> bool:
            inst, rendering = job
            genre_split = None if inventory is None or rendering.genre is None else inventory.genre_split(rendering.genre)
            task = Task(inst.instance_id, rendering.variant, question_specs(inst), forecast_spec(inst),
                        hidden=inst if adapter.kind == "simulated" else None, domain=inst.domain,
                        genre=rendering.genre, split=inst.split, genre_split=genre_split)
            record = {"model": adapter.name, "kind": adapter.kind, "instance_id": inst.instance_id,
                      "variant": rendering.variant, "prompt_sha256": rendering.prompt_sha256}
            if adapter.kind == "api":
                started = time.monotonic()
                try:
                    completion = adapter.complete(rendering.prompt, task)
                    record.update(text=completion.text, error=None, meta=completion.meta)
                except (AdapterError, requests.RequestException, ValueError) as exc:
                    record.update(text=None, error=f"{type(exc).__name__}: {exc}", meta={})
                record["latency_s"] = round(time.monotonic() - started, 2)
            else:
                completion = adapter.complete(rendering.prompt, task)
                record.update(text=completion.text, error=None, meta=completion.meta)
            write_json(out_dir / f"{inst.instance_id}__{rendering.variant}.json", record, root=run_dir)
            return record["error"] is None and not is_truncated(record)

        if adapter.concurrent and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes = list(pool.map(answer, jobs))
        else:
            outcomes = [answer(job) for job in jobs]
        failed = outcomes.count(False)
        log(f"[run] {adapter.name}: {len(outcomes) - failed} prompt(s) answered"
            + (f", {failed} failed or cut off at the token limit (run again to retry only those)" if failed else "")
            + f" -> {out_dir}")
        finished.append(adapter.name)
    return finished


def _stored_reply(path: Path) -> dict | None:
    """The reply stored at `path`, as `run` writes it: a JSON object whose text is a string (or None) and
    whose meta is an object; None when there is none, or when it cannot be read as one (a reply edited or
    copied by hand, or broken off), so that `score` counts its prompt as unanswered and the next `run`
    asks it again."""
    if not path.is_file():
        return None
    try:
        record = read_json(path)
    except (ValueError, OSError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("text"), (str, type(None))) or not isinstance(
            record.get("meta") or {}, dict):
        return None
    return record


def _has_reply(path: Path) -> bool:
    """A stored reply counts as done unless its API call failed or the reply was cut off at the
    output-token limit; those prompts are asked again on the next run, as is a reply that cannot be read."""
    record = _stored_reply(path)
    return record is not None and record.get("error") is None and not is_truncated(record)


# --------------------------------------------------------------------------------- score


def score_run(run_dir: Path, *, private_config: str | Path | None = None, log: Log = print) -> list[Path]:
    """Grade every stored reply. A run's plugins are loaded first, from its private run config: their
    domains' answer types score the answers.

    Every prompt of the run is scored for every model that has replies: a prompt without a reply that can
    be read (a run that was cut short, a reply removed or broken since) counts as unanswered, as a failed
    API call does, so that accuracy and the gaps are always over every question of the run. Its reply is
    recorded with the status "missing" and its rows are marked "missing", so that the metrics that compare a
    model's own answers (consistency, the traps) leave it out; the report and the console summary say how
    many there are."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    responses = run_dir / "responses"
    if not responses.is_dir():
        raise FileNotFoundError(f"no responses in {run_dir}; run `python cli.py run {run_dir}` first")
    load_run_plugins(manifest, private_config, log=log)
    instances = load_instances(run_dir, manifest)
    written = []
    for model_dir in sorted(p for p in responses.iterdir() if p.is_dir()):
        rows, replies, forecasts = [], [], []
        model, kind = model_dir.name, None
        for inst in instances:
            for rendering in inst.renderings:
                record = _stored_reply(model_dir / f"{inst.instance_id}__{rendering.variant}.json")
                if record is None:  # unanswered: no reply that can be read
                    replies.append({"instance_id": inst.instance_id, "variant": rendering.variant, "status": "missing",
                                    "api_error": None, "parse_error": "no stored reply", "truncated": False,
                                    "stale_prompt": False, "input_tokens": None, "output_tokens": None,
                                    "latency_s": None})
                    parsed = parse_reply(None)
                else:
                    model, kind = record.get("model", model), record.get("kind", kind)
                    expected = [q.qid for q in inst.questions] + ([inst.forecast["qid"]] if inst.forecast else [])
                    parsed = parse_reply(record.get("text"), expected)
                    input_tokens, output_tokens = token_usage(record)
                    replies.append({"instance_id": inst.instance_id, "variant": rendering.variant,
                                    "status": parsed.status, "api_error": record.get("error"),
                                    "parse_error": parsed.error, "truncated": is_truncated(record),
                                    "stale_prompt": record.get("prompt_sha256") != rendering.prompt_sha256,
                                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                                    "latency_s": record.get("latency_s")})
                for q in inst.questions:
                    given = parsed.answers.get(q.qid)
                    key = inst.answer_key[q.qid]
                    rows.append({"instance_id": inst.instance_id, "variant": rendering.variant, "qid": q.qid,
                                 "template_id": q.template_id, "pool": q.pool, "answer_type": q.answer_type,
                                 "unit": q.unit, "key": key, "given": given,
                                 "normalized": normalize(q.answer_type, given),
                                 "correct": is_correct(q.answer_type, key, given, unit=q.unit),
                                 **_recorded(domain=inst.domain, genre=rendering.genre),
                                 **({"missing": True} if record is None else {})})
                if inst.forecast:
                    forecasts.append({"instance_id": inst.instance_id, "variant": rendering.variant,
                                      "probability": normalize("probability", parsed.answers.get(inst.forecast["qid"]))})
        per_instance: dict[str, dict] = {}
        for row in rows:
            entry = per_instance.setdefault(row["instance_id"], {"v0": [], "all": []})
            entry["all"].append(row["correct"])
            if row["variant"] == "v0":
                entry["v0"].append(row["correct"])
        per_instance_scores = {iid: {k: (sum(v) / len(v) if v else None) for k, v in e.items()}
                               for iid, e in per_instance.items()}
        out = run_dir / "scores" / f"{model_dir.name}.json"
        write_json(out, {"model": model, "kind": kind, "rows": rows, "replies": replies, "forecasts": forecasts,
                         "per_instance": per_instance_scores}, root=run_dir)
        correct = sum(r["correct"] for r in rows if r["variant"] == "v0")
        total = sum(1 for r in rows if r["variant"] == "v0")
        log(f"[score] {model}: {correct}/{total} correct on original renderings -> {out}")
        missing = sum(1 for r in replies if r["status"] == "missing")
        if missing:
            log(f"[score] warning: {model} has no stored reply to {missing} of {len(replies)} prompts, which count as "
                f"unanswered: `python cli.py run {run_dir}` with this model asks only those")
        written.append(out)
    return written


# ------------------------------------------------------------------------- report & resolve


def build_report(run_dir: Path, *, private_config: str | Path | None = None, log: Log = print) -> Path:
    """Write report.md and report.json. The report loads a run's plugins from its private run config,
    and records any plugin that changed since generation."""
    path = write_report(Path(run_dir), private_config=private_config, log=log)
    log(f"[report] {path} (+ report.json)")
    return path


def resolve_run(run_dir: Path, *, now: int | None = None, fetch: Callable | None = None,
                private_config: str | Path | None = None, log: Log = print) -> dict:
    """Grade the pending forecasts whose window has passed, each with the forecaster of the domain
    that asked it, and refresh the report. A run's plugins are loaded from its private run config. A path
    that is not a run raises FileNotFoundError, so that a scheduled resolve with a mistyped path fails
    instead of reporting a run without forecasts."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    path = run_dir / "forecasts.json"
    if not path.exists():
        log("[resolve] this run has no forecast questions (only earthquake instances from the live USGS feed ask "
            "one: --source usgs, or --source live with earthquakes in the mix)")
        return {}
    load_run_plugins(manifest, private_config, log=log)  # a plugin domain resolves its own forecasts
    forecasts = read_run_json(path)
    by_domain: dict[str | None, list[dict]] = {}  # a legacy run records no domain: its forecasts are earthquake ones
    for forecast in forecasts:
        by_domain.setdefault(forecast.get("domain"), []).append(forecast)
    when = int(time.time()) if now is None else now
    parts = [get_domain(name).forecaster.resolve(group, when, **({"fetch": fetch} if fetch is not None else {}))
             for name, group in (by_domain or {None: []}).items()]
    summary = parts[0]
    for part in parts[1:]:  # counts add up and error lists join
        summary = {key: summary[key] + part[key] for key in summary}
    write_json(path, forecasts, root=run_dir)
    if read_scores(run_dir):
        write_report(run_dir, private_config=private_config)
    log(f"[resolve] {summary['newly_resolved']} newly resolved, {summary['resolved']} resolved in total, "
        f"{summary['pending']} still pending" + (f"; errors: {summary['errors']}" if summary["errors"] else ""))
    return summary


def demo(seed: int = 42, instances: int = 20, *, source: str = "synthetic", mix: str = LEGACY,
         private_config: str | None = None, run_dir: Path | None = None, force: bool = False,
         log: Log = print) -> Path:
    """Everything offline, no API keys: generate -> run (mock, oracle, naive) -> score -> report.

    `mix` and `private_config` are generate's (GenerateConfig). The default directory
    runs/demo-<source>-seed<seed> (with the mix first for any mix but legacy) is overwritten on every
    demo; an explicit run_dir is only overwritten with force=True (and only if it is an evalgen run
    directory)."""
    cfg = GenerateConfig(seed=seed, instances=instances, source=source, mix=mix, private_config=private_config)
    default = RUNS_DIR / f"demo-{run_name(cfg)}"
    run_dir = Path(run_dir) if run_dir else default
    generate(cfg, run_dir, force=force or run_dir == default, log=log)
    run_models(run_dir, ["mock", "oracle", "naive"], private_config=private_config, log=log)
    score_run(run_dir, private_config=private_config, log=log)
    return build_report(run_dir, private_config=private_config, log=log)


def _pct(x: float | None) -> str:
    return "   n/a" if x is None else f"{100 * x:6.1f}%"


def _points(gap: dict | None, key: str) -> str:
    """An adjusted gap in points for the console ("+12.3"), or n/a."""
    adjusted = (gap or {}).get("adjusted") or {}
    return "     n/a" if adjusted.get(key) is None else f"{100 * adjusted[key]:+7.1f} "


def summary_lines(run_dir: Path) -> list[str]:
    """A compact console summary of report.json: per model, accuracy, consistency, the template pools
    and their adjusted gap, and traps; off the legacy path, also the adjusted source and genre gaps
    and cross-genre consistency, when the run has them; then the head-to-head comparisons."""
    data = read_json(Path(run_dir) / "report.json")
    lines = [f"{'model':<28} {'accuracy':>9} {'consist.':>9} {'public':>8} {'new':>8} {'adj.gap':>8} "
             f"{'traps':>8}  flag"]
    for m in data["models"]:
        note = "overfitting?" if m["gap"]["flagged"] else ""
        if m["format"]["api_errors"]:
            note = f"{m['format']['api_errors']}/{m['format']['replies']} API calls failed"
        if m["format"].get("truncated"):
            note = (note + "; " if note else "") + f"{m['format']['truncated']}/{m['format']['replies']} replies cut off"
        if m["format"].get("missing"):  # prompts without a stored reply, counted as unanswered
            note = (note + "; " if note else "") + f"{m['format']['missing']}/{m['format']['replies']} replies missing"
        lines.append(f"{m['model']:<28} {_pct(m['accuracy']['rate']):>9} {_pct(m['consistency']['rate']):>9} "
                     f"{_pct(m['pools']['public']['rate']):>8} {_pct(m['pools']['new']['rate']):>8} "
                     f"{_points(m['gap'], 'public_minus_new'):>8} {_pct(m['traps']['caught']['rate']):>8}  {note}")
    if any(m.get("source_gap") or m.get("genre_gap") or m.get("cross_genre_consistency") for m in data["models"]):
        lines += ["", f"{'public minus held out':<28} {'sources':>9} {'genres':>9} {'x-genre':>9}  flag"]
        for m in data["models"]:
            flagged = [what for what, key in (("sources", "source_gap"), ("genres", "genre_gap"))
                       if (m.get(key) or {}).get("flagged")]
            consistency = (m.get("cross_genre_consistency") or {}).get("rate")
            lines.append(f"{m['model']:<28} {_points(m.get('source_gap'), 'public_minus_held_out'):>9} "
                         f"{_points(m.get('genre_gap'), 'public_minus_held_out'):>9} {_pct(consistency):>9}  "
                         + (f"overfitting? ({', '.join(flagged)})" if flagged else ""))
        lines.append("(adjusted gaps in points: domains on original renderings, genres on every rendering; "
                     "x-genre: agreement between renderings in different genres)")
    for p in data.get("head_to_head", []):
        for label, c in (("original rendering", p), ("every rendering", p.get("all_renderings"))):
            if c:
                lines.append(f"head to head ({label}), {p['a']} minus {p['b']}: {100 * c['difference']:+.1f} pts "
                             f"(95% CI {100 * c['ci95'][0]:+.1f} to {100 * c['ci95'][1]:+.1f}); only A right "
                             f"{c['a_only']}, only B right {c['b_only']}; McNemar p = {c['mcnemar_p']:.2g}")
    return lines
