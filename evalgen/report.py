"""The report: report.md for people, report.json for machines.

Per model: accuracy (95% CI), mean per-instance score, accuracy over all renderings, metamorphic
consistency, seen-vs-new template gap, trap catch rate, JSON format compliance and, for USGS runs,
forecast Brier scores, plus token use for API models. Per pair of API models: a paired head-to-head
comparison (exact McNemar test). Per run: seed and seed commitment, snapshot hash, trap inventory,
generator audit, and two example instances with digest, questions, answer key and model answers.

A run off the legacy path (any mix but legacy, see evalgen.mix) also gets, per model, accuracy by
domain (original renderings) and by genre (every rendering, by its own genre), the source gap (public
minus held-out domains) and the genre gap (public minus held-out genres), each adjusted like the
template gap, and cross-genre consistency; and, per run, its mix, its inventory of domains and genres
and the commitment to its private run config. A legacy report has none of these, so it is exactly the
0.2.0 report.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable, Mapping

from .audit import audit_status
from .domains import get_domain
from .mix import HELD_OUT, LEGACY, PUBLIC, RANDOM, SPLITS, Inventory, load_run_plugins
from .models import Instance, Question
from .ops import KINDS, Step, describe, terminal_step
from .scoring import (brier, difference_ci, equivalent, mcnemar_exact, normalize, paired_difference_ci,
                      stratified_difference, wilson)
from .util import read_json, utc_iso, write_json

PACKAGE_DIR = Path(__file__).resolve().parent  # the evalgen sources that code_fingerprint covers
GAP_FLAG = 0.10  # flag a public-minus-new gap above 10 points whose 95% CI excludes zero
TRAP_KINDS = ("duplicate", "retraction", "conflicting_magnitude")  # the 0.2.0 kinds, in their 0.2.0 order
ANSWER_TYPE_ORDER = ("count", "event_id", "magnitude", "depth", "distance", "region", "utc_time")
MODEL_ORDER = {"oracle": 0, "naive": 1, "mock": 2}


def load_instances(run_dir: Path) -> list[Instance]:
    return [Instance.from_dict(read_json(p)) for p in sorted((Path(run_dir) / "instances").glob("inst-*.json"))]


def legacy_run(instances: Iterable[Instance]) -> bool:
    """Whether a run is a legacy (0.2.0) one: its instances record no domain."""
    return all(inst.domain is None for inst in instances)


def trap_kinds(instances: Iterable[Instance]) -> tuple[str, ...]:
    """The trap kinds a report counts. A legacy report lists the 0.2.0 kinds, planted or not; any
    other run lists the kinds it planted (a generic domain's conflicts are "conflicting_value"), the
    0.2.0 kinds in their order first and then the others by name."""
    instances = list(instances)
    if legacy_run(instances):
        return TRAP_KINDS
    planted = {t.kind for inst in instances for t in inst.traps}
    return tuple(k for k in TRAP_KINDS if k in planted) + tuple(sorted(planted - set(TRAP_KINDS)))


def answer_type_order(answer_types: Iterable[str]) -> list[str]:
    """The answer types that occur, in report order: the 0.2.0 types in their 0.2.0 order, then a
    domain's own types (temperatures, record IDs, groups) by name."""
    present = set(answer_types)
    return [t for t in ANSWER_TYPE_ORDER if t in present] + sorted(present - set(ANSWER_TYPE_ORDER))


def code_fingerprint(root: Path = PACKAGE_DIR) -> str:
    """SHA-256 over the evalgen sources, recorded at generation so a report can tell if code changed:
    every .py file under `root`, subpackages (domains, genres) included, by its relative path."""
    digest = hashlib.sha256()
    for name in sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": list(wilson(k, n)) if n else None}


def _flagged(adjusted: tuple[float, float, float, float] | None) -> bool:
    """The flag of every gap (templates, sources, genres): an adjusted gap above GAP_FLAG whose 95% CI
    excludes zero."""
    return bool(adjusted is not None and adjusted[0] > GAP_FLAG and adjusted[1] > 0)


# ------------------------------------------------------------------------------- metrics


def model_metrics(score: dict, instances: list[Instance], forecasts: list[dict],
                  inventory: Inventory | None = None) -> dict:
    """Every metric of one model's scores. A run off the legacy path also gets the metrics of its
    domains and genres (held_out_metrics), which read each genre's split from `inventory` (the run's,
    recorded in its manifest; without one, every genre is public)."""
    rows = score["rows"]
    index = {(r["instance_id"], r["variant"], r["qid"]): r for r in rows}
    original = [r for r in rows if r["variant"] == "v0"]
    out: dict[str, Any] = {"model": score["model"], "kind": score.get("kind")}

    out["accuracy"] = _rate(sum(r["correct"] for r in original), len(original))
    out["accuracy_all_renderings"] = _rate(sum(r["correct"] for r in rows), len(rows))
    per_instance: dict[str, list[bool]] = {}
    for r in original:
        per_instance.setdefault(r["instance_id"], []).append(r["correct"])
    out["per_instance"] = {iid: sum(v) / len(v) for iid, v in per_instance.items()}
    out["instance_mean"] = mean(out["per_instance"].values()) if per_instance else None

    pools = {}
    for pool in ("public", "new"):
        chosen = [r for r in original if r["pool"] == pool]
        pools[pool] = _rate(sum(r["correct"] for r in chosen), len(chosen))
    gap, lo, hi = difference_ci(pools["public"]["k"], pools["public"]["n"], pools["new"]["k"], pools["new"]["n"])
    # Like-for-like gap: compare the pools within strata of equal code-computed difficulty (how many
    # traps change the answer, and whether the filters matter), so that a difference in trap exposure
    # between the pools cannot masquerade as overfitting. The flag uses this adjusted gap.
    features = {(inst.instance_id, q.qid): q.features for inst in instances for q in inst.questions}
    cells: dict[tuple, list[int]] = {}
    for r in original:
        f = features.get((r["instance_id"], r["qid"])) or {}
        cell = cells.setdefault((f.get("trap_exposure", 0), f.get("filters_matter", True)), [0, 0, 0, 0])
        offset = 0 if r["pool"] == "public" else 2
        cell[offset] += r["correct"]
        cell[offset + 1] += 1
    adjusted = stratified_difference(list(cells.values()))
    out["pools"] = pools
    out["gap"] = {"public_minus_new": gap, "ci95": [lo, hi],
                  "adjusted": None if adjusted is None else
                  {"public_minus_new": adjusted[0], "ci95": [adjusted[1], adjusted[2]], "coverage": adjusted[3]},
                  "flagged": _flagged(adjusted)}

    out["by_answer_type"] = {}
    for answer_type in answer_type_order(r["answer_type"] for r in original):
        chosen = [r for r in original if r["answer_type"] == answer_type]
        out["by_answer_type"][answer_type] = _rate(sum(r["correct"] for r in chosen), len(chosen))
    # By template: by its ID in a legacy run, as 0.2.0 keyed it; off the legacy path by domain and ID
    # ("weather/N01"), since every domain has a P01 of its own, and its twins differ by domain.
    domain_of = {} if legacy_run(instances) else {inst.instance_id: inst.domain for inst in instances}
    by_template: dict[str, list[dict]] = {}
    for r in original:
        key = r["template_id"] if not domain_of else f"{domain_of[r['instance_id']]}/{r['template_id']}"
        by_template.setdefault(key, []).append(r)
    out["by_template"] = {key: _rate(sum(r["correct"] for r in chosen), len(chosen))
                          for key, chosen in sorted(by_template.items())}

    # Metamorphic consistency: needs no ground truth, only the model's own answers.
    pairs = consistent = 0
    fully: list[bool] = []
    for inst in instances:
        for q in inst.questions:
            base = index.get((inst.instance_id, "v0", q.qid))
            if base is None:
                continue
            compared, all_agree = 0, True
            for rendering in inst.renderings[1:]:
                other = index.get((inst.instance_id, rendering.variant, q.qid))
                if other is None or (base["normalized"] is None and other["normalized"] is None):
                    continue
                compared += 1
                same = equivalent(q.answer_type, base["normalized"], other["normalized"], unit=q.unit)
                consistent += same
                all_agree = all_agree and same
            pairs += compared
            if compared:  # a question never answered anywhere is not evidence of consistency
                fully.append(all_agree)
    out["consistency"] = {**_rate(consistent, pairs), "questions_fully_consistent": _rate(sum(fully), len(fully))}

    # Trap catch rate over (load-bearing trap, affected question, rendering) triples. "fell" means the
    # answer is one a reader produces by misreading this trap, alone or together with other traps.
    tally = {kind: Counter() for kind in trap_kinds(instances)}
    for inst in instances:
        for trap in inst.traps:
            for qid in trap.affected:
                q = inst.question(qid)
                fallen = [normalize(q.answer_type, v) for v in trap.fallen.get(qid, [trap.affected[qid]])]
                for rendering in inst.renderings:
                    row = index.get((inst.instance_id, rendering.variant, qid))
                    if row is None:
                        continue
                    if row["correct"]:
                        tally[trap.kind]["caught"] += 1
                    elif row["normalized"] is not None and any(
                            f is not None and equivalent(q.answer_type, row["normalized"], f, f, unit=q.unit)
                            for f in fallen):
                        tally[trap.kind]["fell"] += 1
                    else:
                        tally[trap.kind]["other_error"] += 1
    total = sum(sum(c.values()) for c in tally.values())
    out["traps"] = {
        "caught": _rate(sum(c["caught"] for c in tally.values()), total),
        "fell_for_trap": _rate(sum(c["fell"] for c in tally.values()), total),
        "by_kind": {kind: {**_rate(c["caught"], sum(c.values())), "fell": c["fell"], "other_error": c["other_error"]}
                    for kind, c in tally.items()},
    }

    statuses = Counter(reply["status"] for reply in score["replies"])
    out["format"] = {"replies": len(score["replies"]), "strict": statuses["strict"], "lenient": statuses["lenient"],
                     "failed": statuses["failed"], "api_errors": sum(1 for r in score["replies"] if r.get("api_error")),
                     "truncated": sum(1 for r in score["replies"] if r.get("truncated")),
                     "first_api_error": next((r["api_error"] for r in score["replies"] if r.get("api_error")), None)}

    # Tokens as each API reported them, for the replies that were scored (output includes thinking).
    used = [r for r in score["replies"] if r.get("output_tokens") is not None]
    latencies = [r["latency_s"] for r in used if r.get("latency_s") is not None]
    out["usage"] = None if not used else {
        "replies": len(used),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in used),
        "output_tokens": sum(r["output_tokens"] for r in used),
        "median_output_tokens": median(r["output_tokens"] for r in used),
        "median_latency_s": median(latencies) if latencies else None,
    }

    # Forecasts (v0 renderings). An unanswered forecast is scored as 0.5, i.e. no information.
    resolved = {f["instance_id"]: f for f in forecasts if f.get("status") == "resolved"}
    given = [f for f in score.get("forecasts", []) if f["variant"] == "v0"]
    scored = [brier(0.5 if f["probability"] is None else f["probability"], resolved[f["instance_id"]]["outcome"])
              for f in given if f["instance_id"] in resolved]
    out["forecast"] = {"asked": len(given), "answered": sum(1 for f in given if f["probability"] is not None),
                       "resolved": len(scored), "brier": mean(scored) if scored else None}
    if not legacy_run(instances):  # a legacy report keeps exactly its 0.2.0 keys
        out.update(held_out_metrics(rows, index, instances, inventory))
    return out


# ------------------------------------------------------------------ domains, genres, held-out sets


def answer_family(q: Question) -> str:
    """What a question's answer is, alike in every domain: what its terminal operation answers with
    (count, id, field, distance, group or utc_time; ops.KINDS). Strata of the source and genre gaps."""
    return KINDS[Step(terminal_step(q.chain)).kind].answer


def held_out_metrics(rows: list[dict], index: Mapping[tuple, dict], instances: list[Instance],
                     inventory: Inventory | None = None) -> dict:
    """The metrics of a run off the legacy path, from one model's score rows (`index`: the rows by
    (instance, variant, question)):

      by_domain        accuracy on the original renderings, by the instance's domain
      by_genre         accuracy on every rendering, by the rendering's own genre
      source_gap       public minus held-out domains, on the original renderings
      genre_gap        public minus held-out genres, on every rendering
      cross_genre_consistency  agreement between renderings of a question in different genres

    Both gaps are compared like for like, as the template gap is: within strata of equal
    code-computed difficulty, here the answer's family, the trap exposure and whether the filters
    matter, and for genres also whether the rendering is the original (a variant may be reworded and
    in other units). Each gap is also adjusted for the other split: the source gap's strata hold the
    split of the original rendering's genre, and the genre gap's the split of the instance's domain.
    A mix may tie held-out domains to held-out genres ("earthquakes,toy:toy-telex" renders the toy
    domain in the toy genre alone), and then a model weak on held-out genres alone would otherwise show
    a source gap too, and one weak on held-out domains a genre gap; where the two splits are entirely
    confounded, no stratum compares them and the adjusted gap is n/a rather than a flag on the wrong
    set. (The template pool needs no stratum: every instance asks about as many public as new
    questions, and each domain's twin pairs are kept in step.) An instance's split is the one it
    records; a genre's comes from `inventory`."""
    inventory = inventory or Inventory({}, {})
    by_id = {inst.instance_id: inst for inst in instances}
    questions = {(inst.instance_id, q.qid): q for inst in instances for q in inst.questions}
    genre_of = {(inst.instance_id, r.variant): r.genre for inst in instances for r in inst.renderings}
    rows = [r for r in rows if (r["instance_id"], r["qid"]) in questions]
    original = [r for r in rows if r["variant"] == "v0"]

    def domain(row: dict) -> str | None:
        return by_id[row["instance_id"]].domain

    def domain_split(row: dict) -> str:
        inst = by_id[row["instance_id"]]
        return inst.split or inventory.domain_split(inst.domain)

    def genre(row: dict) -> str | None:
        return genre_of.get((row["instance_id"], row["variant"]))

    def genre_split(row: dict) -> str | None:
        name = genre(row)
        return None if name is None else inventory.genre_split(name)

    def difficulty(row: dict) -> tuple:
        q = questions[(row["instance_id"], row["qid"])]
        return answer_family(q), q.features.get("trap_exposure", 0), q.features.get("filters_matter", True)

    def source_stratum(row: dict) -> tuple:  # rows of original renderings, whose genre is the original's
        return (*difficulty(row), genre_split(row))

    def genre_stratum(row: dict) -> tuple:
        return (*difficulty(row), "original" if row["variant"] == "v0" else "variant", domain_split(row))

    return {
        "by_domain": _accuracy_by(original, domain, domain_split),
        "by_genre": _accuracy_by(rows, genre, genre_split),
        "source_gap": split_gap(original, domain_split, source_stratum),
        "genre_gap": split_gap(rows, genre_split, genre_stratum),
        "cross_genre_consistency": cross_genre_consistency(index, instances),
    }


def _split_rank(split: str | None) -> int:
    return SPLITS.index(split) if split in SPLITS else len(SPLITS)


def _accuracy_by(rows: list[dict], name_of: Callable[[dict], str | None],
                 split_of: Callable[[dict], str | None]) -> dict:
    """Accuracy over `rows` by a name (a domain, a genre), each with its split: public names first,
    then held-out ones, each by name."""
    tally: dict[str, list] = {}
    for row in rows:
        name = name_of(row)
        if name is None:
            continue
        entry = tally.setdefault(name, [split_of(row), 0, 0])
        entry[1] += row["correct"]
        entry[2] += 1
    order = sorted(tally, key=lambda name: (_split_rank(tally[name][0]), name))
    return {name: {"split": tally[name][0], **_rate(tally[name][1], tally[name][2])} for name in order}


def split_gap(rows: list[dict], split_of: Callable[[dict], str | None],
              stratum_of: Callable[[dict], tuple]) -> dict | None:
    """Accuracy on the public side minus accuracy on the held-out side (`split_of` says which side a
    row is on): raw, with a 95% Wald interval, and adjusted (scoring.stratified_difference) within the
    strata that `stratum_of` gives, with its coverage, and flagged as the template gap is. None unless
    both sides have rows."""
    sides = {PUBLIC: [0, 0], HELD_OUT: [0, 0]}
    cells: dict[tuple, list[int]] = {}
    for row in rows:
        side = split_of(row)
        if side not in sides:
            continue
        sides[side][0] += row["correct"]
        sides[side][1] += 1
        cell = cells.setdefault(stratum_of(row), [0, 0, 0, 0])
        offset = 0 if side == PUBLIC else 2
        cell[offset] += row["correct"]
        cell[offset + 1] += 1
    (k1, n1), (k2, n2) = sides[PUBLIC], sides[HELD_OUT]
    if not n1 or not n2:
        return None
    gap, lo, hi = difference_ci(k1, n1, k2, n2)
    adjusted = stratified_difference(list(cells.values()))
    return {"public": _rate(k1, n1), "held_out": _rate(k2, n2), "public_minus_held_out": gap, "ci95": [lo, hi],
            "adjusted": None if adjusted is None else
            {"public_minus_held_out": adjusted[0], "ci95": [adjusted[1], adjusted[2]], "coverage": adjusted[3]},
            "flagged": _flagged(adjusted)}


def cross_genre_consistency(index: Mapping[tuple, dict], instances: list[Instance]) -> dict | None:
    """Agreement between renderings of the same question in different genres: every pair of an
    instance's renderings whose genres differ, compared like metamorphic consistency (no ground truth;
    a missing reply, or a pair that neither answered, is skipped). None when no instance has
    renderings in two genres."""
    pairs = consistent = 0
    fully: list[bool] = []
    mixed = False
    for inst in instances:
        across = [(a.variant, b.variant) for a, b in itertools.combinations(inst.renderings, 2)
                  if a.genre is not None and b.genre is not None and a.genre != b.genre]
        if not across:
            continue
        mixed = True
        for q in inst.questions:
            compared, all_agree = 0, True
            for va, vb in across:
                a, b = index.get((inst.instance_id, va, q.qid)), index.get((inst.instance_id, vb, q.qid))
                if a is None or b is None or (a["normalized"] is None and b["normalized"] is None):
                    continue
                compared += 1
                same = equivalent(q.answer_type, a["normalized"], b["normalized"], unit=q.unit)
                consistent += same
                all_agree = all_agree and same
            pairs += compared
            if compared:  # a question never answered anywhere is not evidence of consistency
                fully.append(all_agree)
    if not mixed:
        return None
    return {**_rate(consistent, pairs), "questions_fully_consistent": _rate(sum(fully), len(fully))}


def _paired_cells(right_a: dict, right_b: dict) -> dict:
    """The paired 2x2 table over the questions both models have, with the difference and McNemar test."""
    common = right_a.keys() & right_b.keys()
    cells = Counter((bool(right_a[k]), bool(right_b[k])) for k in common)
    both, a_only, b_only, neither = cells[True, True], cells[True, False], cells[False, True], cells[False, False]
    difference, lo, hi = paired_difference_ci(both, a_only, b_only, neither)
    return {"questions": len(common), "both_right": both, "a_only": a_only, "b_only": b_only, "neither": neither,
            "difference": difference, "ci95": [lo, hi], "mcnemar_p": mcnemar_exact(a_only, b_only)}


def _right_in_every_rendering(score: dict, renderings: dict[str, int]) -> dict:
    """Per question: answered correctly in every rendering of its instance (a missing reply counts as wrong)."""
    answers: dict[tuple, list[bool]] = {}
    for r in score["rows"]:
        answers.setdefault((r["instance_id"], r["qid"]), []).append(r["correct"])
    return {k: len(v) == renderings.get(k[0], 0) and all(v) for k, v in answers.items()}


def paired_comparison(a: dict, b: dict, renderings: dict[str, int]) -> dict:
    """Model A against model B, question by question; questions are the independent units. The
    questions that exactly one of them got right carry the information, and the exact McNemar test
    asks whether those split evenly. Top-level fields use the original renderings; `all_renderings`
    counts a question as right only if every rendering of it was answered correctly."""
    original = [{(r["instance_id"], r["qid"]): r["correct"] for r in s["rows"] if r["variant"] == "v0"} for s in (a, b)]
    return {"a": a["model"], "b": b["model"], **_paired_cells(*original),
            "all_renderings": _paired_cells(_right_in_every_rendering(a, renderings),
                                            _right_in_every_rendering(b, renderings))}


def _snapshot_summary(snapshot: Mapping[str, Any]) -> dict:
    """A snapshot's provenance, as the run summary words it."""
    return {
        "source": snapshot["source"],
        "snapshot_sha256": snapshot["events_sha256"],
        "raw_feed_sha256": snapshot.get("raw_sha256"),
        "snapshot_note": snapshot.get("note"),
        "snapshot_url": snapshot.get("url"),
        "pool_events": snapshot["n_events"],
        "compiled_at": utc_iso(snapshot["compiled_at"]),
        "window_hours": snapshot["window_hours"],
    }


def _snapshot_fields(manifest: Mapping[str, Any]) -> dict:
    """The run summary's snapshot fields: its snapshot's, or for a run of several domains (one
    snapshot each, listed under "snapshots") their sources and notes together and their pools added
    up; the hashes, URL and windows are then each domain's own."""
    if "snapshot" in manifest:
        return _snapshot_summary(manifest["snapshot"])
    snapshots = manifest["snapshots"]
    notes = [f"{name}: {s['note']}" for name, s in snapshots.items() if s.get("note")]
    return {"source": ", ".join(dict.fromkeys(s["source"] for s in snapshots.values())), "snapshot_sha256": None,
            "raw_feed_sha256": None, "snapshot_note": "; ".join(notes) or None, "snapshot_url": None,
            "pool_events": sum(s["n_events"] for s in snapshots.values()), "compiled_at": None, "window_hours": None}


def _ordered(splits: Mapping[str, str]) -> list[tuple[str, str]]:
    """(name, split) pairs, public ones first, each split in its own order."""
    return sorted(splits.items(), key=lambda item: _split_rank(item[1]))


def mix_summary(manifest: Mapping[str, Any], instances: list[Instance]) -> dict:
    """What the run summary says about a run off the legacy path: its mix; its inventory, each domain
    and genre with its split and how many instances (domains) or renderings (genres) it has; the
    commitment to its private run config (None without one); and, for a run of several domains, each
    domain's snapshot. A run that records no inventory is read from its instances (all public)."""
    inventory = Inventory.of_run(manifest) or Inventory(
        {inst.domain: inst.split or PUBLIC for inst in instances if inst.domain is not None},
        {r.genre: PUBLIC for inst in instances for r in inst.renderings if r.genre is not None})
    per_domain = Counter(inst.domain for inst in instances)
    per_genre = Counter(r.genre for inst in instances for r in inst.renderings)
    summary: dict[str, Any] = {
        "mix": manifest.get("config", {}).get("mix", LEGACY),
        "inventory": {
            "domains": {name: {"split": split, "instances": per_domain[name]} for name, split in
                        _ordered(inventory.domains)},
            "genres": {name: {"split": split, "renderings": per_genre[name]} for name, split in
                       _ordered(inventory.genres)},
        },
        "private_config_commitment": manifest.get("private_config_commitment"),
    }
    if "snapshots" in manifest:
        summary["snapshots"] = {name: _snapshot_summary(s) for name, s in manifest["snapshots"].items()}
    return summary


def run_summary(manifest: dict, instances: list[Instance], forecasts: list[dict]) -> dict:
    questions = [q for inst in instances for q in inst.questions]
    traps = [t for inst in instances for t in inst.traps]
    summary = {
        "run_id": manifest["run_id"],
        "evalgen_version": manifest.get("evalgen_version"),  # not the generation time: report.json stays deterministic
        "seed": manifest["seed"],
        "seed_commitment": manifest["seed_commitment"],
        **_snapshot_fields(manifest),
        "instances": len(instances),
        "questions": len(questions),
        "questions_by_pool": dict(Counter(q.pool for q in questions)),
        "renderings_per_instance": len(instances[0].renderings) if instances else 0,
        "prompts": sum(len(inst.renderings) for inst in instances),
        "traps": {
            "total": len(traps),
            "by_kind": {kind: sum(1 for t in traps if t.kind == kind) for kind in trap_kinds(instances)},
            "load_bearing": sum(1 for t in traps if t.affected),
            "affected_questions": len({(inst.instance_id, qid) for inst in instances for t in inst.traps
                                       for qid in t.affected}),
            "affected_question_pairs": sum(len(t.affected) for t in traps),
        },
        "code_sha256": manifest.get("code_sha256"),
        "forecasts": {"total": len(forecasts), "resolved": sum(1 for f in forecasts if f.get("status") == "resolved")},
        "templates": manifest["templates"],
    }
    if not legacy_run(instances):  # a legacy summary keeps exactly its 0.2.0 keys
        summary.update(mix_summary(manifest, instances))
    return summary


def compute(run_dir: Path, *, private_config: str | Path | None = None,
            log: Callable[[str], None] | None = None) -> dict:
    """Everything report.json holds. The domains and genres of the run's private plugins are loaded first,
    from its private run config (mix.load_run_plugins); a run with plugins records next to code_changed
    which of them changed since generation (plugins_changed), and a run without keeps its 0.2.0 keys."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    plugins_changed = load_run_plugins(manifest, private_config, log=log)
    instances = load_instances(run_dir)
    forecasts = read_json(run_dir / "forecasts.json") if (run_dir / "forecasts.json").exists() else []
    scores = [read_json(p) for p in sorted((run_dir / "scores").glob("*.json"))] if (run_dir / "scores").exists() else []
    inventory = Inventory.of_run(manifest)
    models = [model_metrics(s, instances, forecasts, inventory) for s in scores]
    models.sort(key=lambda m: (MODEL_ORDER.get(m["model"], 9), m["model"]))
    rank = {m["model"]: (-(m["accuracy"]["rate"] or 0.0), -(m["accuracy_all_renderings"]["rate"] or 0.0))
            for m in models}
    api = sorted((s for s in scores if s.get("kind") == "api"), key=lambda s: (*rank[s["model"]], s["model"]))
    renderings = {inst.instance_id: len(inst.renderings) for inst in instances}
    return {
        "run": run_summary(manifest, instances, forecasts),
        "code_changed": bool(manifest.get("code_sha256")) and manifest.get("code_sha256") != code_fingerprint(),
        **({"plugins_changed": plugins_changed} if manifest.get("plugins") else {}),
        "models": models,
        # the more accurate model is A, so a positive difference reads "A is better"
        "head_to_head": [paired_comparison(a, b, renderings) for a, b in itertools.combinations(api, 2)],
        "generator_audit": {"self_check": manifest["self_check"], "human_sample": manifest["audit"],
                            "verdicts": audit_status(run_dir)},
        "forecasts": forecasts,
        "examples": [inst.instance_id for inst in pick_examples(instances)],
    }


def pick_examples(instances: list[Instance], n: int = 2) -> list[Instance]:
    """Instances with the most load-bearing traps (ties: earliest), for the report."""
    ranked = sorted(instances, key=lambda inst: (-sum(1 for t in inst.traps if t.affected), -len(inst.questions), inst.instance_id))
    return ranked[:n]


# ------------------------------------------------------------------------------ markdown


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(rate: dict | None) -> str:
    if not rate or rate.get("ci95") is None:
        return "n/a"
    lo, hi = rate["ci95"]
    return f"{100 * lo:.0f}–{100 * hi:.0f}%"


def _p_value(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def _seconds(s: float | None) -> str:
    return "—" if s is None else f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


def _frac(rate: dict) -> str:
    return f"{_pct(rate['rate'])} ({rate['k']}/{rate['n']})" if rate["n"] else "n/a"


def _cell(value: Any) -> str:
    text = "—" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _words(name: str) -> str:
    """A key as words: "conflicting_magnitude" -> "conflicting magnitude"."""
    return name.replace("_", " ")


def _nouns(inst: Instance) -> tuple[str, str]:
    """What an instance calls one of its records and several (its domain's words, or its surface's)."""
    domain = get_domain(inst.domain).bind(inst.surface)
    return domain.noun, domain.plural


def _record_word(instances: Iterable[Instance]) -> str:
    """What a report calls one of the run's records: "quake" in a legacy run (as 0.2.0 did), else the
    word its instances share ("reading"), or "record" when their words differ."""
    words = {"quake" if inst.domain is None else _nouns(inst)[0] for inst in instances}
    return words.pop() if len(words) == 1 else "record"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out + [""]


def _show_answer(value: Any, answer_type: str) -> str:
    return "—" if value is None else str(value)


def _in_genre(genre: str | None, prefix: str = " ") -> str:
    """", in the classic genre" for a rendering off the legacy path; "" for a legacy one."""
    return "" if genre is None else f"{prefix}in the {genre} genre"


def _split_words(split: str | None) -> str:
    return "held out" if split == HELD_OUT else split or PUBLIC


def _gap_text(value: float, ci95: list[float]) -> str:
    return f"{100 * value:+.1f} pts ({100 * ci95[0]:+.0f} to {100 * ci95[1]:+.0f})"


def _sources_lines(run: dict) -> list[str]:
    """The data sources of a run of several domains: each domain's snapshot."""
    lines = [f"- **Data sources:** one snapshot per domain, {run['pool_events']} records in all."]
    for name, s in run["snapshots"].items():
        lines.append(f"  - **{name}:** {s['source']}" + (f" ({s['snapshot_url']})" if s.get("snapshot_url") else "")
                     + f"; {s['pool_events']} records in the snapshot pool; window: {s['window_hours']} h ending "
                     f"{s['compiled_at']}; snapshot SHA-256 `{s['snapshot_sha256']}`"
                     + (f"; raw feed SHA-256 `{s['raw_feed_sha256']}`" if s.get("raw_feed_sha256") else "")
                     + (f"; note: {s['snapshot_note']}" if s.get("snapshot_note") else "") + ".")
    return lines


def _mix_lines(run: dict, run_dir: Path) -> list[str]:
    """The run bullets of a run off the legacy path: its mix, its domains and genres with their splits,
    and its private run config."""
    inventory = run["inventory"]

    def listed(entries: Mapping[str, dict], count: str) -> str:
        return ", ".join(f"{name} ({_split_words(e['split'])}, {e[count]} {count if e[count] != 1 else count[:-1]})"
                         for name, e in entries.items())

    how = ("each instance's domain, and the genre of each of its renderings, are drawn from the seed"
           if run["mix"] == RANDOM else "instances come from the domains and genres it names, drawn from the seed "
                                        "where it leaves a choice")
    lines = [] if run["mix"] == LEGACY else [  # (instances built outside a mix, as tests build some, have none)
        f"- **Mix:** `{run['mix']}`: {how}; a variant is rendered in another genre than its original whenever the run "
        "allows one."]
    lines += [f"- **Domains:** {listed(inventory['domains'], 'instances')}.",
              f"- **Genres:** {listed(inventory['genres'], 'renderings')}."]
    commitment = run.get("private_config_commitment")
    if commitment:
        lines.append(f"- **Private run config:** committed as `{commitment[:24]}…`, the SHA-256 of its canonical JSON, "
                     "which can be published before the evaluation; once the config is revealed, "
                     f"`python cli.py verify-config {run_dir} CONFIG` checks it. The domains and genres it holds out "
                     "are marked held out above.")
    else:
        lines.append("- **Private run config:** none, so every domain and genre is public.")
    return lines


def _cross_genre_section(models: list[dict]) -> list[str]:
    """The cross-genre consistency of a run whose instances are rendered in several genres; nothing otherwise."""
    if not any(m.get("cross_genre_consistency") for m in models):
        return []
    md = ["## Cross-genre consistency", "",
          "A variant is rendered in another genre than its original whenever the run allows one: the same reports and "
          "notices, written as another kind of document. Correct answers cannot depend on the genre, so every pair of "
          "renderings of a question in two different genres should agree. Like the metamorphic check, this needs no "
          "ground truth.", ""]
    rows = []
    for m in models:
        c = m.get("cross_genre_consistency")
        rows.append([m["model"], "n/a", "n/a", "n/a"] if c is None else
                    [m["model"], _frac(c), _ci(c), _frac(c["questions_fully_consistent"])])
    return md + _table(["model", "consistent cross-genre pairs", "95% CI", "questions consistent across genres"], rows)


def _held_out_section(models: list[dict]) -> list[str]:
    """The source and genre gaps of a run with held-out domains or genres; nothing otherwise."""
    if not any(m.get("source_gap") or m.get("genre_gap") for m in models):
        return []
    md = ["## Public vs held-out domains and genres", "",
          "The private run config holds some domains and genres out. In a real evaluation they are defined in private "
          "plugins, so no model can have been tuned to them, while the public ones are in the published code. The "
          "*source gap* compares accuracy on the original renderings of instances from public and from held-out "
          "domains; the *genre gap* compares renderings in public and in held-out genres. Like the template gap, each "
          "is adjusted: compared within strata of equal code-computed difficulty (the kind of answer asked for, how "
          "many traps change it, and whether the filters matter; for genres, also whether the rendering is the "
          "original or a reworded variant), and within the same split of the other kind: originals rendered in "
          "genres of one split for the source gap, renderings of instances from domains of one split for the genre "
          "gap. So a mix that renders held-out domains only in held-out genres cannot make a model weak on one look "
          "weak on the other; where the two are never apart, the adjusted gap is n/a. A large positive adjusted gap "
          f"flags a model tuned to the public domains or genres (flag: > {int(GAP_FLAG * 100)} points with a 95% CI "
          "that excludes 0). It is a flag, not proof.",
          ""]
    for key, title, what, rendered in (("source_gap", "Sources", "domains", "original renderings"),
                                       ("genre_gap", "Genres", "genres", "every rendering")):
        if not any(m.get(key) for m in models):
            continue
        rows = []
        for m in models:
            gap = m.get(key)
            if gap is None:
                rows.append([m["model"], "n/a", "n/a", "n/a", "n/a", "n/a", "—"])
                continue
            adjusted = gap["adjusted"]
            rows.append([m["model"], _frac(gap["public"]), _frac(gap["held_out"]),
                         _gap_text(gap["public_minus_held_out"], gap["ci95"]),
                         "n/a" if adjusted is None else _gap_text(adjusted["public_minus_held_out"], adjusted["ci95"]),
                         "n/a" if adjusted is None else _pct(adjusted["coverage"]),
                         "⚠ overfitting?" if gap["flagged"] else "—"])
        md += [f"**{title}** ({rendered}):", ""]
        md += _table(["model", f"public {what}", f"held-out {what}", "raw gap (95% CI)", "adjusted gap (95% CI)",
                      "compared like for like", "flag"], rows)
    return md


def _by_domain_and_genre_section(models: list[dict]) -> list[str]:
    """Accuracy by domain and by genre, for a run off the legacy path; nothing otherwise."""
    if not any("by_domain" in m for m in models):
        return []
    md = ["## Accuracy by domain and genre", ""]
    for key, title in (("by_domain", "By domain (original renderings):"),
                       ("by_genre", "By genre (every rendering, by its own genre):")):
        splits = {name: entry["split"] for m in models for name, entry in m[key].items()}
        names = sorted(splits, key=lambda name: (_split_rank(splits[name]), name))
        md += [title, ""]
        md += _table(["model"] + [f"{name} ({_split_words(splits[name])})" for name in names],
                     [[m["model"]] + [_frac(m[key][name]) if name in m[key] else "—" for name in names]
                      for m in models])
    return md


def render_markdown(data: dict, run_dir: Path) -> str:
    run, models, audit = data["run"], data["models"], data["generator_audit"]
    instances = {inst.instance_id: inst for inst in load_instances(run_dir)}
    scores = {s["model"]: s for s in (read_json(p) for p in sorted((Path(run_dir) / "scores").glob("*.json")))}
    t = run["traps"]
    planted = ", ".join(f"{_words(kind)} {n}" for kind, n in t["by_kind"].items())
    pools = run["questions_by_pool"]
    md = [f"# Live evaluation report: `{run['run_id']}`", "",
          "Every instance in this run was generated fresh from the seed below, every answer key was computed by "
          "code from the frozen snapshot, and every score was computed automatically. No item was labelled by hand.", "",
          "## Run", ""]
    md += [
        f"- **Generated with:** evalgen {run.get('evalgen_version') or '?'} (the reading rules and templates are "
        "versioned with the code).",
        f"- **Master seed:** `{run['seed']}` (commitment `{run['seed_commitment'][:24]}…`); every derived seed is listed in `manifest.json`.",
    ]
    if run.get("snapshots"):  # a run of several domains: one snapshot each
        md += _sources_lines(run)
    else:
        md += [
            f"- **Data source:** {run['source']}" + (f" ({run['snapshot_url']})" if run.get("snapshot_url") else "")
            + f"; {run['pool_events']} events in the snapshot pool; window: {run['window_hours']} h ending {run['compiled_at']}.",
            f"- **Snapshot SHA-256:** `{run['snapshot_sha256']}`"
            + (f"; raw feed SHA-256 `{run['raw_feed_sha256']}`" if run.get("raw_feed_sha256") else ""),
        ]
        if run.get("snapshot_note"):
            md.append(f"- **Note:** {run['snapshot_note']}")
    if data.get("code_changed"):
        md.append("- **⚠ Note:** the evalgen code has changed since this run was generated (with evalgen "
                  f"{run.get('evalgen_version') or '?'}, code `{(run.get('code_sha256') or 'unknown')[:16]}…`). This report "
                  "scores the stored prompts and replies; regenerating the run from its seed reproduces it exactly only "
                  "with the code that generated it.")
    if data.get("plugins_changed"):
        md.append("- **⚠ Note:** these plugins have changed since this run was generated: "
                  + ", ".join(f"`{path}`" for path in data["plugins_changed"]) + ". This report reads the run's "
                  "domains and genres with them as they are now; regenerating the run from its seed reproduces it "
                  "exactly only with the plugins that generated it.")
    md += [
        f"- **Instances:** {run['instances']}; **questions:** {run['questions']} ({pools.get('public', 0)} from public "
        f"templates, {pools.get('new', 0)} from new-this-run templates); **renderings per instance:** "
        f"{run['renderings_per_instance']} (original + {run['renderings_per_instance'] - 1} metamorphic variants); "
        f"**prompts:** {run['prompts']}.",
        f"- **Traps planted:** {t['total']} ({planted}); {t['load_bearing']} are load-bearing: "
        f"mishandling them changes the answers of {t['affected_questions']} of the {run['questions']} questions "
        f"({t['affected_question_pairs']} trap–question pairs). The rest stay in as plain noise."]
    if "inventory" in run:  # a run off the legacy path
        md += _mix_lines(run, run_dir)
    md += [""]

    md += ["## Model accuracy (original renderings)", ""]
    md += _table(["model", "kind", "accuracy", "95% CI", "mean instance score", "accuracy, all renderings",
                  "JSON strict / lenient / failed (API errors)"],
                 [[m["model"], m["kind"], _frac(m["accuracy"]), _ci(m["accuracy"]), _pct(m["instance_mean"]),
                   _pct(m["accuracy_all_renderings"]["rate"]),
                   f"{m['format']['strict']} / {m['format']['lenient']} / {m['format']['failed']} "
                   f"({m['format']['api_errors']})"] for m in models])
    md += ["`oracle` reads only the digest text and applies the reading rules, so it should score 100%. That shows every "
           f"item is answerable from the text. `naive` treats every line as a separate {_record_word(instances.values())} "
           "and ignores notices. `mock` is a *simulated* model with built-in benchmark contamination, used to exercise "
           "the metrics without API keys; it is not a language model.", ""]
    for m in models:
        f = m["format"]
        if f["api_errors"]:
            first = (f.get("first_api_error") or "").replace("`", "'")[:200]
            md += [f"> ⚠ **{m['model']}**: {f['api_errors']} of {f['replies']} API calls failed, so its scores count "
                   f"those prompts as unanswered; they say nothing about the model itself. First error: `{first}`. "
                   "Running `cli.py run` again retries only the failed prompts.", ""]
        if f.get("truncated"):
            md += [f"> ⚠ **{m['model']}**: {f['truncated']} of {f['replies']} replies were cut off at the output-token "
                   "limit, usually because the model was still thinking, so those prompts count as unanswered. Raise "
                   "`--max-tokens` and run `cli.py run` again: it re-asks only those prompts.", ""]

    if data.get("head_to_head"):
        md += ["## Head to head (paired)", "",
               "The API models answered the same questions, so they are compared question by question, with questions "
               "as the independent units. Only the questions that exactly one model got right say which is better; the "
               "exact McNemar test asks whether those split evenly. The first row uses the original renderings. The "
               "second counts a question as right only if every rendering of it was answered correctly, so it also "
               "rewards robustness to surface changes (units, formats, line order, wording).", ""]
        rows = []
        for p in data["head_to_head"]:
            for label, c in (("original rendering", p), ("every rendering", p["all_renderings"])):
                rows.append([p["a"], p["b"], label, c["questions"], c["both_right"], c["a_only"], c["b_only"],
                             c["neither"], f"{100 * c['difference']:+.1f} pts ({100 * c['ci95'][0]:+.1f} to "
                             f"{100 * c['ci95'][1]:+.1f})", _p_value(c["mcnemar_p"])])
        md += _table(["A", "B", "right on", "questions", "both right", "only A right", "only B right", "both wrong",
                      "A minus B (95% CI)", "McNemar p"], rows)
    used = [m for m in models if m.get("usage")]
    if used:
        md += ["## API usage", "",
               "Tokens as each API reported them, for the replies that were scored (an attempt that failed or was cut "
               "off and then asked again is not counted). Output tokens include any thinking before the answer.", ""]
        md += _table(["model", "replies", "input tokens", "output tokens", "median output tokens per reply",
                      "median latency"],
                     [[m["model"], m["usage"]["replies"], f"{m['usage']['input_tokens']:,}",
                       f"{m['usage']['output_tokens']:,}", f"{m['usage']['median_output_tokens']:,.0f}",
                       _seconds(m["usage"]["median_latency_s"])] for m in used])

    md += ["## Metamorphic consistency", "",
           "Each instance is re-rendered with the same snapshot but different units, line order, formats and question "
           "wording. Correct answers cannot change, so any change in a model's answer is an inconsistency. This needs no "
           "ground truth.", ""]
    md += _table(["model", "consistent answer pairs", "95% CI", "questions consistent across all variants"],
                 [[m["model"], _frac(m["consistency"]), _ci(m["consistency"]),
                   _frac(m["consistency"]["questions_fully_consistent"])] for m in models])
    md += _cross_genre_section(models)

    md += ["## Seen vs new templates", "",
           "Public templates are published (a model may have trained on them). Each new template is composed for this "
           "run as a twin of one public template: the same final operation and number of filters, but a different "
           "filter combination, and each twin pair is used equally often. The *adjusted* gap compares the pools like "
           "for like, within strata of equal code-computed difficulty (how many traps change the answer, and whether "
           "the filters matter), so that the pools' different trap exposure is not mistaken for overfitting. A large "
           f"positive adjusted gap flags overfitting to the published skeletons (flag: > {int(GAP_FLAG * 100)} points "
           "with a 95% CI that excludes 0). It is a flag, not proof.", ""]

    def adjusted(m: dict) -> str:
        a = m["gap"].get("adjusted")
        if not a:
            return "n/a"
        return f"{100 * a['public_minus_new']:+.1f} pts ({100 * a['ci95'][0]:+.0f} to {100 * a['ci95'][1]:+.0f})"

    md += _table(["model", "public (seen)", "new this run", "raw gap (95% CI)", "adjusted gap (95% CI)", "flag"],
                 [[m["model"], _frac(m["pools"]["public"]), _frac(m["pools"]["new"]),
                   f"{100 * m['gap']['public_minus_new']:+.1f} pts ({100 * m['gap']['ci95'][0]:+.0f} to "
                   f"{100 * m['gap']['ci95'][1]:+.0f})", adjusted(m),
                   "⚠ overfitting?" if m["gap"]["flagged"] else "—"] for m in models])
    md += _held_out_section(models)

    md += ["## Trap catch rate", "",
           "Counted over (load-bearing trap, affected question, rendering) triples. *Caught* = answered correctly; "
           "*fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other traps).", ""]
    kinds = list(t["by_kind"])
    md += _table(["model", "caught (overall)", *(_words(k) for k in kinds), "fell for the trap"],
                 [[m["model"], _frac(m["traps"]["caught"])]
                  + [_pct(m["traps"]["by_kind"][k]["rate"]) for k in kinds]
                  + [_frac(m["traps"]["fell_for_trap"])] for m in models])

    types = answer_type_order(a for m in models for a in m["by_answer_type"])
    md += ["## Accuracy by answer type (original renderings)", ""]
    md += _table(["model"] + types, [[m["model"]] + [_frac(m["by_answer_type"][a]) if a in m["by_answer_type"] else "—"
                                                    for a in types] for m in models])
    md += _by_domain_and_genre_section(models)

    sc, sample, verdicts = audit["self_check"], audit["human_sample"], audit["verdicts"]
    md += ["## Generator audit", "",
           f"- **Automated self-check:** for {sc['oracle_agreements']}/{sc['oracle_checks']} question-renderings, the "
           f"digest-only oracle agreed with the code-computed key. Applying the reading rules to each digest recovered the "
           f"snapshot exactly for {sc['resolution_invariant_instances']}/{run['instances']} instances."
           + (f" Problems: {'; '.join(sc['problems'][:5])}" if sc["problems"] else ""),
           f"- **Human spot-check sample:** {sample['sampled']} of {sample['population']} generated items "
           f"({_pct(sample['sampled'] / sample['population'] if sample['population'] else None)}; target {_pct(sample['rate'])}) → "
           "`audit/audit_sample.jsonl` (checklist: `audit/audit_sample.md`).",
           ]
    if verdicts.get("reviewed"):
        line = (f"- **Verdicts:** {verdicts['reviewed']}/{verdicts['sampled']} reviewed, {verdicts['errors']} generator "
                f"error(s): estimated error rate {_pct(verdicts['error_rate'])} (95% CI {_ci({'ci95': verdicts['error_rate_ci95']})})")
        if verdicts.get("rule_of_three_upper95") is not None:
            line += f"; with zero errors the rule of three bounds it below {_pct(verdicts['rule_of_three_upper95'])}"
        md += [line + "."]
    else:
        md += [f"- **Verdicts:** 0/{verdicts['sampled']} reviewed so far. Fill in `verdict` (ok / error) in the JSONL file "
               "and re-run `cli.py report` to get an estimated generator error rate."]
    md += [""]

    if data["forecasts"]:
        md += ["## Delayed-resolution forecasts (USGS mode)", ""]
        pending = sum(1 for f in data["forecasts"] if f.get("status") != "resolved")
        outcomes = [f["outcome"] for f in data["forecasts"] if f.get("status") == "resolved"]
        if pending:
            md += [f"{len(data['forecasts'])} forecast question(s); {pending} still pending. Run `python cli.py resolve "
                   f"{run_dir}` after the window closes (+2 h grace) to grade them.", ""]
        else:
            md += [f"All {len(outcomes)} forecast question(s) resolved against the USGS catalogue "
                   f"({sum(outcomes)} with a qualifying earthquake). Unanswered forecasts count as 0.5.", ""]
        md += _table(["model", "forecasts answered (v0)", "resolved", "Brier score (lower is better)"],
                     [[m["model"], f"{m['forecast']['answered']}/{m['forecast']['asked']}", m["forecast"]["resolved"],
                       "pending" if m["forecast"]["brier"] is None else f"{m['forecast']['brier']:.3f}"] for m in models])

    md += ["## Templates", ""]
    templates = run["templates"]["public"] + run["templates"]["new"]
    if any("domain" in x for x in templates):  # several domains: each has its own twins (and binds P01-P12)
        md += _table(["domain", "id", "pool", "twin of", "skeleton"],
                     [[x.get("domain") or "—", x["template_id"], x["pool"], x.get("twin_of") or "—", x["skeleton"]]
                      for x in templates])
    else:
        md += _table(["id", "pool", "twin of", "skeleton"],
                     [[x["template_id"], x["pool"], x.get("twin_of") or "—", x["skeleton"]] for x in templates])

    md += ["## Per-instance scores (original rendering)", "", "<details><summary>Show table</summary>", ""]
    model_names = [m["model"] for m in models]
    md += _table(["instance", "questions", "load-bearing traps"] + model_names,
                 [[iid, len(inst.questions), sum(1 for t in inst.traps if t.affected)]
                  + [_pct(m["per_instance"].get(iid)) for m in models] for iid, inst in instances.items()])
    md += ["</details>", ""]

    md += ["## Example instances", ""]
    for number, iid in enumerate(data["examples"], start=1):
        md += _example(number, instances[iid], scores, model_names)
    return "\n".join(md)


def _example(number: int, inst: Instance, scores: dict, model_names: list[str]) -> list[str]:
    v0 = inst.renderings[0]
    rows_by_model = {name: {(r["instance_id"], r["variant"], r["qid"]): r for r in scores[name]["rows"]} for name in model_names}
    def count(n: int, word: str, plural: str | None = None) -> str:
        return f"{n} {word if n == 1 else plural or word + 's'}"

    records = ("earthquake", "earthquakes") if inst.domain is None else _nouns(inst)
    md = [f"### Example {number}: `{inst.instance_id}` ({count(len(inst.event_ids), *records)}, "
          f"{count(len(inst.reports), 'report line')}, {count(len(inst.notices), 'notice')}, "
          f"{count(len(inst.questions), 'question')})", "",
          f"**Digest (original rendering `v0`{_in_genre(v0.genre, ', ')}, exactly as sent to the models):**", "",
          "```text", v0.digest, "```", "",
          "**Questions, code-computed key and model answers (`v0`):**", ""]
    table_rows = []
    for q in inst.questions:
        cells = [q.qid, f"{q.template_id} ({q.pool})", v0.question_text[q.qid], _show_answer(inst.answer_key[q.qid], q.answer_type)]
        for name in model_names:
            row = rows_by_model[name].get((inst.instance_id, "v0", q.qid))
            cells.append("—" if row is None else f"{_show_answer(row['given'], q.answer_type)} {'✓' if row['correct'] else '✗'}")
        table_rows.append(cells)
    md += _table(["Q", "template", "question", "key"] + model_names, table_rows)
    md += ["**Operation chains (the machine-readable meaning of each question):**", ""]
    md += [f"- {q.qid}: `{describe(q.chain)}`" for q in inst.questions] + [""]
    md += ["**Traps in this digest:**", ""]
    for trap in inst.traps:
        effect = ", ".join(f"{qid} → {_show_answer(v, inst.question(qid).answer_type)}" for qid, v in trap.affected.items())
        md.append(f"- {trap.trap_id} *{trap.kind.replace('_', ' ')}* ({trap.form}): {trap.description}. "
                  + (f"If mishandled: {effect}." if effect else "Inert here (changes no answer)."))
    md += [""]
    if len(inst.renderings) > 1:
        v1 = inst.renderings[1]
        md += [f"<details><summary>Metamorphic variant `{v1.variant}` of the same instance{_in_genre(v1.genre, ', ')} "
               "(same key)</summary>", "",
               "```text", v1.digest, "```", ""]
        md += [f"- {qid}: {text}" for qid, text in v1.question_text.items()] + ["", "</details>", ""]
    return md


def write_report(run_dir: Path, *, private_config: str | Path | None = None,
                 log: Callable[[str], None] | None = None) -> Path:
    """Write report.json and report.md (see compute for `private_config`)."""
    run_dir = Path(run_dir)
    data = compute(run_dir, private_config=private_config, log=log)
    write_json(run_dir / "report.json", data)
    path = run_dir / "report.md"
    path.write_text(render_markdown(data, run_dir), encoding="utf-8")
    return path
