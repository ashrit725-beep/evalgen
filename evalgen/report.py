"""The report: report.md for people, report.json for machines.

Per model: accuracy (95% CI), mean per-instance score, accuracy over all renderings, metamorphic
consistency, seen-vs-new template gap, trap catch rate, JSON format compliance and, for USGS runs,
forecast Brier scores, plus token use for API models. Per pair of API models: a paired head-to-head
comparison (exact McNemar test). Per run: seed and seed commitment, snapshot hash, trap inventory,
generator audit, and two example instances with digest, questions, answer key and model answers.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from .audit import audit_status
from .models import Instance
from .ops import describe
from .scoring import (brier, difference_ci, equivalent, mcnemar_exact, normalize, paired_difference_ci,
                      stratified_difference, wilson)
from .util import read_json, utc_iso, write_json

GAP_FLAG = 0.10  # flag a public-minus-new gap above 10 points whose 95% CI excludes zero
TRAP_KINDS = ("duplicate", "retraction", "conflicting_magnitude")
ANSWER_TYPE_ORDER = ("count", "event_id", "magnitude", "depth", "distance", "region", "utc_time")
MODEL_ORDER = {"oracle": 0, "naive": 1, "mock": 2}


def load_instances(run_dir: Path) -> list[Instance]:
    return [Instance.from_dict(read_json(p)) for p in sorted((Path(run_dir) / "instances").glob("inst-*.json"))]


def code_fingerprint() -> str:
    """SHA-256 over the evalgen sources, recorded at generation so a report can tell if code changed."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": list(wilson(k, n)) if n else None}


# ------------------------------------------------------------------------------- metrics


def model_metrics(score: dict, instances: list[Instance], forecasts: list[dict]) -> dict:
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
                  "flagged": bool(adjusted is not None and adjusted[0] > GAP_FLAG and adjusted[1] > 0)}

    out["by_answer_type"] = {}
    for answer_type in ANSWER_TYPE_ORDER:
        chosen = [r for r in original if r["answer_type"] == answer_type]
        if chosen:
            out["by_answer_type"][answer_type] = _rate(sum(r["correct"] for r in chosen), len(chosen))
    out["by_template"] = {}
    for template_id in sorted({r["template_id"] for r in original}):
        chosen = [r for r in original if r["template_id"] == template_id]
        out["by_template"][template_id] = _rate(sum(r["correct"] for r in chosen), len(chosen))

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
                same = equivalent(q.answer_type, base["normalized"], other["normalized"])
                consistent += same
                all_agree = all_agree and same
            pairs += compared
            if compared:  # a question never answered anywhere is not evidence of consistency
                fully.append(all_agree)
    out["consistency"] = {**_rate(consistent, pairs), "questions_fully_consistent": _rate(sum(fully), len(fully))}

    # Trap catch rate over (load-bearing trap, affected question, rendering) triples. "fell" means the
    # answer is one a reader produces by misreading this trap, alone or together with other traps.
    tally = {kind: Counter() for kind in TRAP_KINDS}
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
                            f is not None and equivalent(q.answer_type, row["normalized"], f, f) for f in fallen):
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
    return out


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


def run_summary(manifest: dict, instances: list[Instance], forecasts: list[dict]) -> dict:
    questions = [q for inst in instances for q in inst.questions]
    traps = [t for inst in instances for t in inst.traps]
    return {
        "run_id": manifest["run_id"],
        "evalgen_version": manifest.get("evalgen_version"),  # not the generation time: report.json stays deterministic
        "seed": manifest["seed"],
        "seed_commitment": manifest["seed_commitment"],
        "source": manifest["snapshot"]["source"],
        "snapshot_sha256": manifest["snapshot"]["events_sha256"],
        "raw_feed_sha256": manifest["snapshot"].get("raw_sha256"),
        "snapshot_note": manifest["snapshot"].get("note"),
        "snapshot_url": manifest["snapshot"].get("url"),
        "pool_events": manifest["snapshot"]["n_events"],
        "compiled_at": utc_iso(manifest["snapshot"]["compiled_at"]),
        "window_hours": manifest["snapshot"]["window_hours"],
        "instances": len(instances),
        "questions": len(questions),
        "questions_by_pool": dict(Counter(q.pool for q in questions)),
        "renderings_per_instance": len(instances[0].renderings) if instances else 0,
        "prompts": sum(len(inst.renderings) for inst in instances),
        "traps": {
            "total": len(traps),
            "by_kind": {kind: sum(1 for t in traps if t.kind == kind) for kind in TRAP_KINDS},
            "load_bearing": sum(1 for t in traps if t.affected),
            "affected_questions": len({(inst.instance_id, qid) for inst in instances for t in inst.traps
                                       for qid in t.affected}),
            "affected_question_pairs": sum(len(t.affected) for t in traps),
        },
        "code_sha256": manifest.get("code_sha256"),
        "forecasts": {"total": len(forecasts), "resolved": sum(1 for f in forecasts if f.get("status") == "resolved")},
        "templates": manifest["templates"],
    }


def compute(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    instances = load_instances(run_dir)
    forecasts = read_json(run_dir / "forecasts.json") if (run_dir / "forecasts.json").exists() else []
    scores = [read_json(p) for p in sorted((run_dir / "scores").glob("*.json"))] if (run_dir / "scores").exists() else []
    models = [model_metrics(s, instances, forecasts) for s in scores]
    models.sort(key=lambda m: (MODEL_ORDER.get(m["model"], 9), m["model"]))
    rank = {m["model"]: (-(m["accuracy"]["rate"] or 0.0), -(m["accuracy_all_renderings"]["rate"] or 0.0))
            for m in models}
    api = sorted((s for s in scores if s.get("kind") == "api"), key=lambda s: (*rank[s["model"]], s["model"]))
    renderings = {inst.instance_id: len(inst.renderings) for inst in instances}
    return {
        "run": run_summary(manifest, instances, forecasts),
        "code_changed": bool(manifest.get("code_sha256")) and manifest.get("code_sha256") != code_fingerprint(),
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


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out + [""]


def _show_answer(value: Any, answer_type: str) -> str:
    return "—" if value is None else str(value)


def render_markdown(data: dict, run_dir: Path) -> str:
    run, models, audit = data["run"], data["models"], data["generator_audit"]
    instances = {inst.instance_id: inst for inst in load_instances(run_dir)}
    scores = {s["model"]: s for s in (read_json(p) for p in sorted((Path(run_dir) / "scores").glob("*.json")))}
    t = run["traps"]
    pools = run["questions_by_pool"]
    md = [f"# Live evaluation report: `{run['run_id']}`", "",
          "Every instance in this run was generated fresh from the seed below, every answer key was computed by "
          "code from the frozen snapshot, and every score was computed automatically. No item was labelled by hand.", "",
          "## Run", ""]
    md += [
        f"- **Generated with:** evalgen {run.get('evalgen_version') or '?'} (the reading rules and templates are "
        "versioned with the code).",
        f"- **Master seed:** `{run['seed']}` (commitment `{run['seed_commitment'][:24]}…`); every derived seed is listed in `manifest.json`.",
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
    md += [
        f"- **Instances:** {run['instances']}; **questions:** {run['questions']} ({pools.get('public', 0)} from public "
        f"templates, {pools.get('new', 0)} from new-this-run templates); **renderings per instance:** "
        f"{run['renderings_per_instance']} (original + {run['renderings_per_instance'] - 1} metamorphic variants); "
        f"**prompts:** {run['prompts']}.",
        f"- **Traps planted:** {t['total']} (duplicate {t['by_kind']['duplicate']}, retraction {t['by_kind']['retraction']}, "
        f"conflicting magnitude {t['by_kind']['conflicting_magnitude']}); {t['load_bearing']} are load-bearing: "
        f"mishandling them changes the answers of {t['affected_questions']} of the {run['questions']} questions "
        f"({t['affected_question_pairs']} trap–question pairs). The rest stay in as plain noise.", ""]

    md += ["## Model accuracy (original renderings)", ""]
    md += _table(["model", "kind", "accuracy", "95% CI", "mean instance score", "accuracy, all renderings",
                  "JSON strict / lenient / failed (API errors)"],
                 [[m["model"], m["kind"], _frac(m["accuracy"]), _ci(m["accuracy"]), _pct(m["instance_mean"]),
                   _pct(m["accuracy_all_renderings"]["rate"]),
                   f"{m['format']['strict']} / {m['format']['lenient']} / {m['format']['failed']} "
                   f"({m['format']['api_errors']})"] for m in models])
    md += ["`oracle` reads only the digest text and applies the reading rules, so it should score 100%. That shows every "
           "item is answerable from the text. `naive` treats every line as a separate quake and ignores notices. "
           "`mock` is a *simulated* model with built-in benchmark contamination, used to exercise the metrics without "
           "API keys; it is not a language model.", ""]
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

    md += ["## Trap catch rate", "",
           "Counted over (load-bearing trap, affected question, rendering) triples. *Caught* = answered correctly; "
           "*fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other traps).", ""]
    md += _table(["model", "caught (overall)", "duplicate", "retraction", "conflicting magnitude", "fell for the trap"],
                 [[m["model"], _frac(m["traps"]["caught"])]
                  + [_pct(m["traps"]["by_kind"][k]["rate"]) for k in TRAP_KINDS]
                  + [_frac(m["traps"]["fell_for_trap"])] for m in models])

    types = [a for a in ANSWER_TYPE_ORDER if any(a in m["by_answer_type"] for m in models)]
    md += ["## Accuracy by answer type (original renderings)", ""]
    md += _table(["model"] + types, [[m["model"]] + [_frac(m["by_answer_type"][a]) if a in m["by_answer_type"] else "—"
                                                    for a in types] for m in models])

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
    md += _table(["id", "pool", "twin of", "skeleton"],
                 [[x["template_id"], x["pool"], x.get("twin_of") or "—", x["skeleton"]]
                  for x in run["templates"]["public"] + run["templates"]["new"]])

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
    def count(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    md = [f"### Example {number}: `{inst.instance_id}` ({count(len(inst.event_ids), 'earthquake')}, "
          f"{count(len(inst.reports), 'report line')}, {count(len(inst.notices), 'notice')}, "
          f"{count(len(inst.questions), 'question')})", "",
          "**Digest (original rendering `v0`, exactly as sent to the models):**", "", "```text", v0.digest, "```", "",
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
        md += [f"<details><summary>Metamorphic variant `{v1.variant}` of the same instance (same key)</summary>", "",
               "```text", v1.digest, "```", ""]
        md += [f"- {qid}: {text}" for qid, text in v1.question_text.items()] + ["", "</details>", ""]
    return md


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    data = compute(run_dir)
    write_json(run_dir / "report.json", data)
    path = run_dir / "report.md"
    path.write_text(render_markdown(data, run_dir), encoding="utf-8")
    return path
