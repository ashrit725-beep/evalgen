"""Generator audit: a ~2% random sample of items for human spot-checking.

The point is to estimate the *generator's own* error rate (ambiguous wording, a wrong key, an
unfair trap), not to label data: every answer is already computed by code. Auditors set the
``verdict`` field to "ok" or "error" in audit/audit_sample.jsonl; ``cli.py report`` then estimates
the error rate with a Wilson interval (with zero errors found, the rule of three gives a quick
95% upper bound of 3/n).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

from .models import Event, Instance
from .ops import describe
from .scoring import wilson
from .util import utc_iso

OK_WORDS = {"ok", "correct", "pass", "fine"}
ERROR_WORDS = {"error", "wrong", "ambiguous", "fail", "bad"}


def write_audit_sample(run_dir: Path, instances: Sequence[Instance], events_by_id: dict[str, Event],
                       rng: random.Random, rate: float = 0.02) -> dict:
    items = [(inst, q) for inst in instances for q in inst.questions]
    n = min(len(items), max(1, math.ceil(rate * len(items)))) if items else 0
    picks = sorted(rng.sample(range(len(items)), n)) if n else []
    entries = []
    for i in picks:
        inst, q = items[i]
        entries.append({
            "item": f"{inst.instance_id}/{q.qid}",
            "instance_id": inst.instance_id,
            "qid": q.qid,
            "template_id": q.template_id,
            "pool": q.pool,
            "question": inst.renderings[0].question_text[q.qid],
            "answer_type": q.answer_type,
            "unit": q.unit,
            "operation_chain": describe(q.chain),
            "answer_key": inst.answer_key[q.qid],
            "evidence": [_event_row(events_by_id[eid]) for eid in q.evidence],
            "traps_affecting": [
                {"trap_id": t.trap_id, "kind": t.kind, "description": t.description, "fallen_answer": t.affected[q.qid]}
                for t in inst.traps if q.qid in t.affected
            ],
            "prompt_file": f"prompts/{inst.instance_id}/v0.txt",
            "verdict": None,
            "auditor": "",
            "notes": "",
        })
    audit_dir = Path(run_dir) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with open(audit_dir / "audit_sample.jsonl", "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (audit_dir / "audit_sample.md").write_text(_markdown(entries, instances), encoding="utf-8")
    return {"population": len(items), "sampled": n, "rate": rate, "file": "audit/audit_sample.jsonl"}


def _event_row(e: Event) -> dict:
    return {"id": e.id, "time_utc": utc_iso(e.time), "lat": e.lat, "lon": e.lon, "depth_km": e.depth_km,
            "mag": e.mag, "mag_type": e.mag_type, "region": e.region}


def _markdown(entries: list[dict], instances: Sequence[Instance]) -> str:
    by_id = {inst.instance_id: inst for inst in instances}
    out = ["# Generator audit sample", "",
           "For each item, check that the question is unambiguous, that the answer key is right given the",
           "digest and the reading rules, and that any trap is fair. Record the verdict (`ok` or `error`) in",
           "`audit_sample.jsonl`; the report turns verdicts into an estimated generator error rate.", ""]
    for entry in entries:
        out += [f"## {entry['item']} ({entry['template_id']}, {entry['pool']} template)", "",
                f"**Question:** {entry['question']}", "",
                f"**Operation chain:** `{entry['operation_chain']}`", "",
                f"**Answer key:** `{entry['answer_key']}` ({entry['answer_type']}{', ' + entry['unit'] if entry['unit'] else ''})", "",
                "**Evidence (events passing the filters, canonical values):**", ""]
        out += [f"- {row['id']} {row['time_utc']} M{row['mag']:g} {row['mag_type']}, depth {row['depth_km']:g} km, "
                f"({row['lat']:.3f}, {row['lon']:.3f}), {row['region']}" for row in entry["evidence"]] or ["- (none)"]
        if entry["traps_affecting"]:
            out += ["", "**Traps that change this answer if mishandled:**", ""]
            out += [f"- {t['trap_id']} {t['kind']}: {t['description']} → fallen answer `{t['fallen_answer']}`"
                    for t in entry["traps_affecting"]]
        out += ["", "<details><summary>Digest (original rendering)</summary>", "", "```text",
                by_id[entry["instance_id"]].renderings[0].digest, "```", "", "</details>", "",
                "- [ ] verdict: ok / error", ""]
    return "\n".join(out)


def audit_status(run_dir: Path) -> dict:
    path = Path(run_dir) / "audit" / "audit_sample.jsonl"
    if not path.exists():
        return {"sampled": 0, "reviewed": 0, "errors": 0}
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts = [str(e.get("verdict") or "").strip().lower() for e in entries]
    reviewed = [v for v in verdicts if v in OK_WORDS | ERROR_WORDS]
    errors = sum(1 for v in reviewed if v in ERROR_WORDS)
    status = {"sampled": len(entries), "reviewed": len(reviewed), "errors": errors}
    if reviewed:
        status["error_rate"] = errors / len(reviewed)
        status["error_rate_ci95"] = list(wilson(errors, len(reviewed)))
        if errors == 0:
            status["rule_of_three_upper95"] = min(1.0, 3 / len(reviewed))
    return status
