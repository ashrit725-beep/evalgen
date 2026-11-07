"""One evaluation cycle, end to end.

    generate  snapshot -> instances (digest + traps + questions + key + metamorphic renderings)
    run       send every rendering's prompt to each model and store the raw replies
    score     parse replies (robust JSON) and grade every answer automatically
    report    accuracy, consistency, seen-vs-new gap, trap catch rate, audit -> report.md / .json
    resolve   later: grade pending forecast questions against the USGS catalogue

Run directory layout:
    manifest.json          seeds, config, snapshot hash, templates, generator self-check
    snapshot/              events.json, snapshot.json (+ raw_feed.geojson in USGS mode)
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import requests

from . import __version__
from .adapters import DEFAULT_MAX_TOKENS, AdapterError, Task, is_truncated, make_adapter, token_usage
from .answers import parse_reply
from .audit import write_audit_sample
from .digest import choose_style, make_chatter, render_digest
from .forecast import forecast_text, make_forecast, resolve_forecasts
from .models import Event, Instance, Rendering
from .ops import FILTER_OPS, execute
from .prompt import build_prompt
from .questions import PUBLIC_TEMPLATES, Context, Template, compose_new_templates, compose_questions, phrase_question
from .report import code_fingerprint, load_instances, write_report
from .scoring import is_correct, normalize, same_answer
from .seeds import SeedLog, derive_seed, fresh_nonce, seed_commitment
from .solver import solve
from .sources import Snapshot, build_snapshot, save_snapshot
from .traps import TrapConfig, inject_traps
from .util import read_json, sha256_hex, slugify, write_json

RUNS_DIR = Path("runs")
RUN_MARKER = ".evalgen-run"  # lets --force recognise (and only delete) evalgen run directories
Log = Callable[[str], None]


@dataclass
class GenerateConfig:
    seed: int
    instances: int = 20
    source: str = "synthetic"
    feed: str = "day"
    snapshot_file: str | None = None
    variants: int = 2  # metamorphic re-renderings per instance (besides the original)
    pool_size: int = 360  # synthetic snapshot size
    events_min: int = 20  # earthquakes per digest
    events_max: int = 28
    questions_min: int = 3
    questions_max: int = 5
    audit_rate: float = 0.02


# ------------------------------------------------------------------------------ generate


def _weighted_sample(items: Sequence[Event], k: int, rng: random.Random) -> list[Event]:
    """Weighted sampling without replacement (Efraimidis-Spirakis); larger quakes are likelier."""
    keyed = sorted(((rng.random() ** (1.0 / 1.5 ** e.mag), i) for i, e in enumerate(items)), reverse=True)
    return [items[i] for _, i in keyed[:k]]


def sample_events(pool: Sequence[Event], rng: random.Random, n_min: int, n_max: int) -> list[Event]:
    """The earthquakes for one digest: mostly from 3-4 focus regions, so questions have substance."""
    n = min(len(pool), rng.randint(n_min, n_max))
    counts = Counter(e.region for e in pool)
    eligible = sorted(r for r, c in counts.items() if c >= 4)
    focus = set(rng.sample(eligible, min(len(eligible), rng.randint(3, 4)))) if eligible else set()
    in_focus = [e for e in pool if e.region in focus]
    others = [e for e in pool if e.region not in focus]
    n_focus = min(len(in_focus), round(n * 0.8))
    chosen = _weighted_sample(in_focus, n_focus, rng) + _weighted_sample(others, min(len(others), n - n_focus), rng)
    if len(chosen) < n:
        chosen_ids = {e.id for e in chosen}
        chosen += _weighted_sample([e for e in pool if e.id not in chosen_ids], n - len(chosen), rng)
    return sorted(chosen, key=lambda e: (e.time, e.id))


def question_specs(inst: Instance) -> list[dict]:
    """What reference solvers get: each question's chain, never its answer."""
    return [{"qid": q.qid, "chain": q.chain, "answer_type": q.answer_type, "unit": q.unit, "pool": q.pool}
            for q in inst.questions]


def forecast_spec(inst: Instance) -> dict | None:
    return {"qid": inst.forecast["qid"], "rule": inst.forecast["rule"]} if inst.forecast else None


def build_instance(index: int, snapshot: Snapshot, new_templates: Sequence[Template], seedlog: SeedLog,
                   cfg: GenerateConfig, ctx: Context, used_chains: set[str], usage: dict[str, int]) -> Instance:
    events = sample_events(snapshot.events, seedlog.rng("instance", index, "events"), cfg.events_min, cfg.events_max)
    q_rng = seedlog.rng("instance", index, "questions")
    questions = compose_questions(events, PUBLIC_TEMPLATES, new_templates, q_rng, ctx,
                                  q_rng.randint(cfg.questions_min, cfg.questions_max), used_chains, usage)
    answer_key = {q.qid: execute(q.chain, events) for q in questions}  # the key: code over the snapshot
    trap_config = TrapConfig(id_style="sequence" if snapshot.source == "usgs" else "random")
    reports, notices, traps = inject_traps(events, questions, seedlog.rng("instance", index, "traps"),
                                           compiled_at=snapshot.compiled_at, window_hours=snapshot.window_hours,
                                           config=trap_config)
    for q in questions:
        unfiltered = [s for s in q.chain if s["op"] not in FILTER_OPS]
        q.features = {
            "trap_exposure": min(2, sum(1 for t in traps if q.qid in t.affected)),
            "filters_matter": not same_answer(q.answer_type, execute(unfiltered, events), answer_key[q.qid]),
        }
    chatter = make_chatter(seedlog.rng("instance", index, "chatter"), snapshot.compiled_at,
                           sorted({e.network for e in events}))
    forecast = None
    if snapshot.source == "usgs":
        forecast = make_forecast(events, seedlog.rng("instance", index, "forecast"), snapshot.compiled_at,
                                 snapshot.window_hours)

    renderings: list[Rendering] = []
    original_style = None
    for v in range(cfg.variants + 1):
        seed = seedlog.seed("instance", index, "render", v)
        rng = random.Random(seed)
        style = choose_style(rng, avoid=original_style)
        digest = render_digest(reports, notices, chatter, compiled_at=snapshot.compiled_at,
                               window_hours=snapshot.window_hours, style=style, rng=rng)
        texts = {q.qid: phrase_question(q, rng, alt_units=v > 0 and rng.random() < 0.5) for q in questions}
        items = [(q.qid, texts[q.qid], q.answer_type, q.unit) for q in questions]
        if forecast:
            items.append((forecast["qid"], forecast_text(forecast), "probability", None))
        prompt = build_prompt(digest, items)
        renderings.append(Rendering(variant=f"v{v}", seed=seed, style=style, digest=digest, question_text=texts,
                                    prompt=prompt, prompt_sha256=sha256_hex(prompt)))
        original_style = original_style or style

    return Instance(
        instance_id=f"inst-{index:03d}", seed=seedlog.seed("instance", index), compiled_at=snapshot.compiled_at,
        window_hours=snapshot.window_hours, event_ids=[e.id for e in events], reports=reports, notices=notices,
        chatter=chatter, traps=traps, questions=questions, answer_key=answer_key, renderings=renderings,
        forecast=forecast,
    )


def generator_self_check(instances: Sequence[Instance]) -> dict:
    """The generator audits itself: a solver that sees ONLY the digest text must reproduce the key."""
    checks = agreements = 0
    problems: list[str] = []
    for inst in instances:
        specs = question_specs(inst)
        for rendering in inst.renderings:
            got = solve(rendering.prompt, specs, mode="oracle")
            for q in inst.questions:
                checks += 1
                if same_answer(q.answer_type, got[q.qid], inst.answer_key[q.qid]):
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
    snapshot, raw = build_snapshot(cfg.source, seedlog, feed=cfg.feed, pool_size=cfg.pool_size,
                                   snapshot_file=cfg.snapshot_file)
    save_snapshot(snapshot, raw, run_dir / "snapshot")
    if snapshot.note:
        log(f"[generate] note: {snapshot.note}")
    log(f"[generate] snapshot: {snapshot.source}, {len(snapshot.events)} events, sha256 {snapshot.events_sha256[:16]}…")

    replay = _replay_check(cfg.snapshot_file, log)
    new_templates = compose_new_templates(seedlog.rng("templates", "new"))
    ctx = Context(snapshot.compiled_at, snapshot.window_hours)
    used_chains: set[str] = set()  # no concrete question is ever asked twice in a run
    usage: dict[str, int] = {}  # questions per template so far; keeps each public/twin pair in step
    instances = []
    for index in range(cfg.instances):
        inst = build_instance(index, snapshot, new_templates, seedlog, cfg, ctx, used_chains, usage)
        instances.append(inst)
        write_json(run_dir / "instances" / f"{inst.instance_id}.json", inst.to_dict())
        for rendering in inst.renderings:
            path = run_dir / "prompts" / inst.instance_id / f"{rendering.variant}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendering.prompt, encoding="utf-8")

    self_check = generator_self_check(instances)
    audit = write_audit_sample(run_dir, instances, {e.id: e for e in snapshot.events}, seedlog.rng("audit"),
                               cfg.audit_rate)
    forecasts = [dict(inst.forecast, instance_id=inst.instance_id) for inst in instances if inst.forecast]
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
        "config": asdict(cfg),
        "snapshot": snapshot.meta(),
        "replay": replay,
        "templates": {"public": [t.to_dict() for t in PUBLIC_TEMPLATES], "new": [t.to_dict() for t in new_templates]},
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
                        hidden=inst if adapter.kind == "simulated" else None)
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
                                 "correct": is_correct(q.answer_type, key, given)})
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
    run_dir = Path(run_dir)
    path = run_dir / "forecasts.json"
    if not path.exists():
        log("[resolve] this run has no forecast questions (they are only generated with --source usgs)")
        return {}
    forecasts = read_json(path)
    summary = resolve_forecasts(forecasts, int(time.time()) if now is None else now,
                                **({"fetch": fetch} if fetch is not None else {}))
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
