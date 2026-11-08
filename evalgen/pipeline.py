"""One evaluation cycle, end to end.

    generate  snapshot -> instances (digest + traps + questions + key + metamorphic renderings)
    run       send every rendering's prompt to each model and store the raw replies
    score     parse replies (robust JSON) and grade every answer automatically
    report    accuracy, consistency, seen-vs-new gap, trap catch rate, audit -> report.md / .json
    resolve   later: grade pending forecast questions against the USGS catalogue

A run's mix (GenerateConfig.mix, see evalgen.mix) says which domain and genres its instances come
from; the default, legacy, is the 0.2.0 run: earthquakes in the classic genre.

Run directory layout:
    manifest.json          seeds, config, snapshot hash, templates, generator self-check
    snapshot/              events.json, snapshot.json (+ the frozen feed of a live source, raw_feed.geojson for USGS)
    instances/inst-*.json  full instance: reports, notices, traps, questions, key, renderings
    prompts/inst-*/v*.txt  the exact prompt texts sent to models
    responses/<model>/     raw replies         scores/<model>.json  graded rows
    audit/                 ~2% human spot-check sample
    forecasts.json         pending / resolved forecasts (USGS mode)
    report.md, report.json
"""
from __future__ import annotations

import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Sequence

import requests

from . import __version__
from .adapters import DEFAULT_MAX_TOKENS, AdapterError, Task, is_truncated, make_adapter, token_usage
from .answers import parse_reply
from .audit import write_audit_sample
from .digest import make_chatter
from .domains import get_domain
from .domains.base import Domain
from .domains.earthquakes import EARTHQUAKES
from .genres import detect_genre
from .genres.base import Genre, RenderContext
from .genres.classic import CLASSIC
from .mix import single_choice
from .models import Instance, Question, Rendering
from .ops import execute, role
from .prompt import build_prompt, domain_prompt
from .questions import Context, Template, compose_new_templates, compose_questions, phrase_question, public_templates
from .report import code_fingerprint, load_instances, write_report
from .scoring import is_correct, normalize, same_answer
from .seeds import SeedLog, derive_seed, fresh_nonce, seed_commitment
from .solver import extract_digest, solve
from .sources import Snapshot, build_domain_snapshot, build_snapshot, save_snapshot
from .traps import TrapConfig, inject_traps
from .util import read_json, sha256_hex, slugify, write_json

RUNS_DIR = Path("runs")
RUN_MARKER = ".evalgen-run"  # lets --force recognise (and only delete) evalgen run directories
Log = Callable[[str], None]


@dataclass
class GenerateConfig:
    seed: int
    instances: int = 20
    source: str = "synthetic"  # or a live source: "usgs" (legacy), "live" or the domain's own (any other mix)
    feed: str = "day"  # the USGS feed of a legacy run: "hour" or "day"
    snapshot_file: str | None = None
    variants: int = 2  # metamorphic re-renderings per instance (besides the original)
    pool_size: int = 360  # synthetic snapshot size
    events_min: int = 20  # records per digest
    events_max: int = 28
    questions_min: int = 3
    questions_max: int = 5
    audit_rate: float = 0.02
    # Which domain and genres the instances come from (evalgen.mix): "legacy" (0.2.0: earthquakes,
    # classic), a list such as "toy:classic" or "weather,neo:email", or "random"; and the evaluator's
    # private run config (held-out domains and genres, plugins). The per-instance draws over a mix
    # that leaves a choice, and the private config, come with plan phase 4.
    mix: str = "legacy"
    private_config: str | None = None  # path of the private run config (JSON)

    OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = ("mix", "private_config")  # recorded only when not default

    def to_dict(self) -> dict[str, Any]:
        """The config as the manifest records it: asdict without the 0.3.0 options that keep their
        defaults, so that a legacy run records exactly the 0.2.0 config."""
        defaults = {f.name: f.default for f in fields(self)}
        return {k: v for k, v in asdict(self).items() if k not in self.OPTIONAL_KEYS or v != defaults[k]}


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
    domain and the renamed records. The key is code over the snapshot's records."""
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
                                  domain=domain)
    answer_key = {q.qid: execute(q.chain, events, units=domain.units) for q in questions}
    return Composition(domain, surface, events, questions, answer_key)


def build_instance(index: int, snapshot: Snapshot, new_templates: Sequence[Template], seedlog: SeedLog,
                   cfg: GenerateConfig, ctx: Context, used_chains: set[str], usage: dict[str, int], *,
                   domain: Domain = EARTHQUAKES, genre_plan: Sequence[Genre] | None = None) -> Instance:
    """One test item about `domain`: its records, questions, answer key (compose_instance), traps and
    renderings.

    `genre_plan` gives the genre of each rendering, the original first. None is the legacy path:
    classic everywhere, with the 0.2.0 calls, random draws and output. Otherwise the instance records
    its domain and each rendering its genre, a variant avoids the original's style only when it is in
    the original's genre, and every prompt is a non-legacy one (prompt.domain_prompt), with the
    domain's glossary, which the genre renders the records through. A domain with surface words draws
    them for the instance (only off the legacy path, so the legacy seed paths never change), and every
    step uses the bound domain."""
    if genre_plan is None and domain.name != EARTHQUAKES.name:
        raise ValueError(f"only earthquakes have a legacy rendering: {domain.name} needs a genre plan")
    if genre_plan is not None and len(genre_plan) != cfg.variants + 1:
        raise ValueError(f"the genre plan names {len(genre_plan)} genres for {cfg.variants + 1} renderings")
    composed = compose_instance(index, snapshot, new_templates, seedlog, cfg, ctx, used_chains, usage,
                                domain=domain, legacy=genre_plan is None)
    domain, events, questions, answer_key = composed.domain, composed.events, composed.questions, composed.answer_key
    units = domain.units
    glossary = None if genre_plan is None else domain.glossary()
    trap_config = TrapConfig(id_style="sequence" if snapshot.source == "usgs" else "random")
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
        texts = {q.qid: phrase_question(q, rng, alt_units=v > 0 and rng.random() < 0.5, domain=domain)
                 for q in questions}
        items = [(q.qid, texts[q.qid], q.answer_type, q.unit) for q in questions]
        if forecast:
            items.append((forecast["qid"], forecaster.text(forecast), "probability", None))
        if glossary is None:
            prompt = build_prompt(digest, items, rules=domain.rules_text())
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
        forecast=forecast, domain=None if genre_plan is None else domain.name, surface=composed.surface,
    )


def generator_self_check(instances: Sequence[Instance]) -> dict:
    """The generator audits itself: a solver that sees ONLY the prompt text must reproduce the key.
    Each rendering is read in the domain the run recorded for its instance (a legacy run records none,
    and its prompts are read as 0.2.0 read them), by the genre that recognises the digest, which must
    be the genre it was rendered in; through the glossary the prompt prints, for a non-legacy run."""
    checks = agreements = 0
    problems: list[str] = []
    for inst in instances:
        specs = question_specs(inst)
        for rendering in inst.renderings:
            got = solve(rendering.prompt, specs, mode="oracle", domain=inst.domain)
            if rendering.genre is not None:
                sniffed = detect_genre(extract_digest(rendering.prompt)).name
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


def _instance_records(instances: Sequence[Instance], snapshot: Snapshot, domain: Domain) -> dict[str, dict[str, Any]]:
    """Each instance's records by ID: the snapshot's records, as the instance names them (renamed by
    its surface words, when it has them)."""
    by_id = {e.id: e for e in snapshot.events}
    return {inst.instance_id: by_id if inst.surface is None else
            {r.id: r for r in map(domain.bind(inst.surface).dress, snapshot.events)} for inst in instances}


def _replay_check(snapshot_file: str | None, log: Log) -> dict | None:
    """When replaying another run's frozen feed, note whether that run used the same code."""
    if not snapshot_file:
        return None
    source_run = Path(snapshot_file).resolve().parent.parent
    if not is_run_dir(source_run) or not (source_run / "manifest.json").exists():
        return {"source_run": None, "same_code": None}
    same = read_json(source_run / "manifest.json").get("code_sha256") == code_fingerprint()
    if not same:
        log("[generate] warning: the source run was generated with different code, so a replay "
            "with the same seed will not reproduce its instances exactly")
    return {"source_run": str(source_run), "same_code": same}


def generate(cfg: GenerateConfig, run_dir: Path | None = None, *, force: bool = False, log: Log = print) -> Path:
    """Write a run: snapshot, instances, prompts, the generator's self-check, the audit sample and
    the manifest. The run's mix (cfg.mix) names its domain and genre; the default is the legacy (0.2.0)
    run, and any other mix is generated on seed paths of its own (see evalgen.mix)."""
    if cfg.private_config is not None:
        raise NotImplementedError("the private run config comes with plan phase 4")
    choice = single_choice(cfg.mix)
    if choice is not None and cfg.feed != GenerateConfig.feed:
        raise ValueError("the USGS feed is chosen in legacy runs only: off the legacy path its live source is the "
                         "day feed")
    run_dir = Path(run_dir) if run_dir else RUNS_DIR / f"{cfg.source}-seed{cfg.seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{run_dir} already exists (use --force to overwrite, or --out)")
        if not is_run_dir(run_dir):
            raise FileExistsError(f"{run_dir} is not an evalgen run directory; refusing to delete it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_MARKER).write_text("evalgen run directory\n", encoding="utf-8")

    seedlog = SeedLog(cfg.seed)
    nonce = fresh_nonce()
    commitment = seed_commitment(cfg.seed, nonce)
    log(f"[generate] master seed {cfg.seed} (publishable commitment {commitment[:16]}…)")
    if choice is None:  # legacy: the 0.2.0 snapshot and seed paths
        domain, genre_plan = EARTHQUAKES, None
        snapshot, raw = build_snapshot(cfg.source, seedlog, feed=cfg.feed, pool_size=cfg.pool_size,
                                       snapshot_file=cfg.snapshot_file)
        save_snapshot(snapshot, raw, run_dir / "snapshot")
    else:
        domain, genre = choice
        genre_plan = (genre,) * (cfg.variants + 1)
        snapshot, raw = build_domain_snapshot(domain, cfg.source, seedlog, pool_size=cfg.pool_size,
                                              snapshot_file=cfg.snapshot_file)
        if raw is None:
            save_snapshot(snapshot, raw, run_dir / "snapshot")
        else:  # frozen bytes of the domain's live source
            save_snapshot(snapshot, raw, run_dir / "snapshot", raw_filename=domain.live.raw_filename)
    if snapshot.note:
        log(f"[generate] note: {snapshot.note}")
    log(f"[generate] snapshot: {snapshot.source}, {len(snapshot.events)} events, sha256 {snapshot.events_sha256[:16]}…")

    replay = _replay_check(cfg.snapshot_file, log)
    template_seeds = ("templates", "new") if choice is None else ("templates", "new", domain.name)
    new_templates = compose_new_templates(seedlog.rng(*template_seeds), domain)
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    used_chains: set[str] = set()  # no concrete question is ever asked twice in a run
    usage: dict[str, int] = {}  # questions per template so far; keeps each public/twin pair in step
    instances = []
    for index in range(cfg.instances):
        inst = build_instance(index, snapshot, new_templates, seedlog, cfg, ctx, used_chains, usage,
                              domain=domain, genre_plan=genre_plan)
        instances.append(inst)
        write_json(run_dir / "instances" / f"{inst.instance_id}.json", inst.to_dict())
        for rendering in inst.renderings:
            path = run_dir / "prompts" / inst.instance_id / f"{rendering.variant}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendering.prompt, encoding="utf-8")

    self_check = generator_self_check(instances)
    audit = write_audit_sample(run_dir, instances, _instance_records(instances, snapshot, domain),
                               seedlog.rng("audit"), cfg.audit_rate)
    forecasts = [dict(inst.forecast, instance_id=inst.instance_id, **_recorded(domain=inst.domain))
                 for inst in instances if inst.forecast]
    if forecasts:
        write_json(run_dir / "forecasts.json", forecasts)
    seedlog.seed("mock")  # the simulated model's seed, so it is logged too
    write_json(run_dir / "manifest.json", {
        "evalgen_version": __version__,
        "code_sha256": code_fingerprint(),
        "run_id": run_dir.name,
        "seed": cfg.seed,
        "seed_commitment": commitment,  # publishable before the evaluation
        "seed_nonce": nonce,  # secret until the reveal, like the seed itself (this file is private)
        "config": cfg.to_dict(),
        "snapshot": snapshot.meta(),
        "replay": replay,
        "templates": {"public": [t.to_dict() for t in public_templates(domain)],
                      "new": [t.to_dict() for t in new_templates]},
        "instances": [inst.instance_id for inst in instances],
        "self_check": self_check,
        "audit": audit,
        "seeds": seedlog.seeds,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    n_questions = sum(len(inst.questions) for inst in instances)
    log(f"[generate] {len(instances)} instances, {n_questions} questions, {cfg.variants + 1} renderings each; "
        f"oracle self-check {self_check['oracle_agreements']}/{self_check['oracle_checks']}; "
        f"audit sample {audit['sampled']}/{audit['population']} -> {run_dir}")
    return run_dir


# ----------------------------------------------------------------------------------- run


def run_models(run_dir: Path, specs: Sequence[str], *, workers: int = 4, force: bool = False,
               max_tokens: int = DEFAULT_MAX_TOKENS, log: Log = print) -> list[str]:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    instances = load_instances(run_dir)
    finished = []
    for spec in specs:
        try:
            adapter = make_adapter(spec, seed=derive_seed(manifest["seed"], "mock"), max_tokens=max_tokens)
        except AdapterError as exc:
            log(f"[run] skipping {spec}: {exc}")
            continue
        out_dir = run_dir / "responses" / slugify(adapter.name)
        jobs = [(inst, r) for inst in instances for r in inst.renderings
                if force or not _has_reply(out_dir / f"{inst.instance_id}__{r.variant}.json")]

        def answer(job: tuple[Instance, Rendering]) -> bool:
            inst, rendering = job
            task = Task(inst.instance_id, rendering.variant, question_specs(inst), forecast_spec(inst),
                        hidden=inst if adapter.kind == "simulated" else None, domain=inst.domain,
                        genre=rendering.genre, split=inst.split)
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
            write_json(out_dir / f"{inst.instance_id}__{rendering.variant}.json", record)
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


def _has_reply(path: Path) -> bool:
    """A stored reply counts as done unless its API call failed or the reply was cut off at the
    output-token limit; those prompts are asked again on the next run."""
    if not path.exists():
        return False
    try:
        record = read_json(path)
    except (ValueError, OSError):
        return False
    return record.get("error") is None and not is_truncated(record)


# --------------------------------------------------------------------------------- score


def score_run(run_dir: Path, *, log: Log = print) -> list[Path]:
    run_dir = Path(run_dir)
    responses = run_dir / "responses"
    if not responses.exists():
        raise FileNotFoundError(f"no responses in {run_dir}; run `python cli.py run {run_dir}` first")
    instances = load_instances(run_dir)
    written = []
    for model_dir in sorted(p for p in responses.iterdir() if p.is_dir()):
        rows, replies, forecasts = [], [], []
        model, kind = model_dir.name, None
        for inst in instances:
            for rendering in inst.renderings:
                path = model_dir / f"{inst.instance_id}__{rendering.variant}.json"
                if not path.exists():
                    continue
                record = read_json(path)
                model, kind = record.get("model", model), record.get("kind", kind)
                expected = [q.qid for q in inst.questions] + ([inst.forecast["qid"]] if inst.forecast else [])
                parsed = parse_reply(record.get("text"), expected)
                input_tokens, output_tokens = token_usage(record)
                replies.append({"instance_id": inst.instance_id, "variant": rendering.variant, "status": parsed.status,
                                "api_error": record.get("error"), "parse_error": parsed.error,
                                "truncated": is_truncated(record),
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
                                 **_recorded(domain=inst.domain, genre=rendering.genre)})
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
                         "per_instance": per_instance_scores})
        correct = sum(r["correct"] for r in rows if r["variant"] == "v0")
        total = sum(1 for r in rows if r["variant"] == "v0")
        log(f"[score] {model}: {correct}/{total} correct on original renderings -> {out}")
        written.append(out)
    return written


# ------------------------------------------------------------------------- report & resolve


def build_report(run_dir: Path, *, log: Log = print) -> Path:
    path = write_report(Path(run_dir))
    log(f"[report] {path} (+ report.json)")
    return path


def resolve_run(run_dir: Path, *, now: int | None = None, fetch: Callable | None = None, log: Log = print) -> dict:
    """Grade the pending forecasts whose window has passed, each with the forecaster of the domain
    that asked it, and refresh the report."""
    run_dir = Path(run_dir)
    path = run_dir / "forecasts.json"
    if not path.exists():
        log("[resolve] this run has no forecast questions (they are only generated with --source usgs)")
        return {}
    forecasts = read_json(path)
    by_domain: dict[str | None, list[dict]] = {}  # a legacy run records no domain: its forecasts are earthquake ones
    for forecast in forecasts:
        by_domain.setdefault(forecast.get("domain"), []).append(forecast)
    when = int(time.time()) if now is None else now
    parts = [get_domain(name).forecaster.resolve(group, when, **({"fetch": fetch} if fetch is not None else {}))
             for name, group in (by_domain or {None: []}).items()]
    summary = parts[0]
    for part in parts[1:]:  # counts add up and error lists join
        summary = {key: summary[key] + part[key] for key in summary}
    write_json(path, forecasts)
    if (run_dir / "scores").exists():
        write_report(run_dir)
    log(f"[resolve] {summary['newly_resolved']} newly resolved, {summary['resolved']} resolved in total, "
        f"{summary['pending']} still pending" + (f"; errors: {summary['errors']}" if summary["errors"] else ""))
    return summary


def demo(seed: int = 42, instances: int = 20, *, source: str = "synthetic", run_dir: Path | None = None,
         force: bool = False, log: Log = print) -> Path:
    """Everything offline, no API keys: generate -> run (mock, oracle, naive) -> score -> report.

    The default directory runs/demo-<source>-seed<seed> is overwritten on every demo; an explicit
    run_dir is only overwritten with force=True (and only if it is an evalgen run directory)."""
    default = RUNS_DIR / f"demo-{source}-seed{seed}"
    run_dir = Path(run_dir) if run_dir else default
    generate(GenerateConfig(seed=seed, instances=instances, source=source), run_dir,
             force=force or run_dir == default, log=log)
    run_models(run_dir, ["mock", "oracle", "naive"], log=log)
    score_run(run_dir, log=log)
    return build_report(run_dir, log=log)


def summary_lines(run_dir: Path) -> list[str]:
    """A compact console summary of report.json."""
    data = read_json(Path(run_dir) / "report.json")
    lines = [f"{'model':<28} {'accuracy':>9} {'consist.':>9} {'public':>8} {'new':>8} {'adj.gap':>8} "
             f"{'traps':>8}  flag"]
    for m in data["models"]:
        def pct(x):
            return "   n/a" if x is None else f"{100 * x:6.1f}%"
        adjusted = (m["gap"].get("adjusted") or {}).get("public_minus_new")
        gap = "     n/a" if adjusted is None else f"{100 * adjusted:+7.1f} "
        note = "overfitting?" if m["gap"]["flagged"] else ""
        if m["format"]["api_errors"]:
            note = f"{m['format']['api_errors']}/{m['format']['replies']} API calls failed"
        if m["format"].get("truncated"):
            note = (note + "; " if note else "") + f"{m['format']['truncated']}/{m['format']['replies']} replies cut off"
        lines.append(f"{m['model']:<28} {pct(m['accuracy']['rate']):>9} {pct(m['consistency']['rate']):>9} "
                     f"{pct(m['pools']['public']['rate']):>8} {pct(m['pools']['new']['rate']):>8} "
                     f"{gap:>8} {pct(m['traps']['caught']['rate']):>8}  {note}")
    for p in data.get("head_to_head", []):
        for label, c in (("original rendering", p), ("every rendering", p.get("all_renderings"))):
            if c:
                lines.append(f"head to head ({label}), {p['a']} minus {p['b']}: {100 * c['difference']:+.1f} pts "
                             f"(95% CI {100 * c['ci95'][0]:+.1f} to {100 * c['ci95'][1]:+.1f}); only A right "
                             f"{c['a_only']}, only B right {c['b_only']}; McNemar p = {c['mcnemar_p']:.2g}")
    return lines
