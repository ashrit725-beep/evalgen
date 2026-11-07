#!/usr/bin/env python3
"""evalgen command line.

    python cli.py demo --seed 42 --instances 20          # everything offline, no API keys
    python cli.py generate --seed 7 --instances 20 [--source usgs]
    python cli.py run runs/synthetic-seed7 --models mock,oracle,naive,anthropic,openai:gpt-4.1
    python cli.py score runs/synthetic-seed7
    python cli.py report runs/synthetic-seed7
    python cli.py resolve runs/usgs-seed7                  # later: grade forecast questions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalgen import pipeline
from evalgen.seeds import fresh_seed
from evalgen.util import parse_utc_iso


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="Live, fine-tuning-resistant LLM evaluation on earthquake digests.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="build a fresh, seeded instance set (snapshot, digests, questions, key)")
    g.add_argument("--seed", type=int, help="master seed (default: a fresh random seed, which is logged)")
    g.add_argument("--instances", type=int, default=20)
    g.add_argument("--source", choices=("synthetic", "usgs"), default="synthetic")
    g.add_argument("--feed", choices=("hour", "day"), default="day", help="USGS feed window (usgs source only)")
    g.add_argument("--snapshot-file", help="use a frozen USGS GeoJSON file instead of fetching the live feed")
    g.add_argument("--variants", type=int, default=2, help="metamorphic re-renderings per instance")
    g.add_argument("--out", help="run directory (default: runs/<source>-seed<seed>)")
    g.add_argument("--force", action="store_true", help="overwrite an existing run directory")

    r = sub.add_parser("run", help="send every prompt to one or more models and store the replies")
    r.add_argument("run_dir")
    r.add_argument("--models", default="mock,oracle,naive",
                   help="comma-separated: oracle, naive, mock, anthropic[:model], openai[:model]")
    r.add_argument("--workers", type=int, default=4, help="parallel requests for API models")
    r.add_argument("--force", action="store_true", help="ask again even if a stored reply exists")
    r.add_argument("--max-tokens", type=int, default=16000,
                   help="output-token limit per reply for API models (models that think need room)")

    s = sub.add_parser("score", help="parse and grade the stored replies")
    s.add_argument("run_dir")

    rep = sub.add_parser("report", help="write report.md and report.json")
    rep.add_argument("run_dir")

    res = sub.add_parser("resolve", help="grade pending forecast questions against the USGS catalogue")
    res.add_argument("run_dir")
    res.add_argument("--now", help="pretend the current UTC time is this ISO timestamp (for testing)")

    d = sub.add_parser("demo", help="generate + run (mock, oracle, naive) + score + report, fully offline")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--instances", type=int, default=20)
    d.add_argument("--source", choices=("synthetic", "usgs"), default="synthetic")
    d.add_argument("--out", help="run directory (default: runs/demo-<source>-seed<seed>, overwritten each time)")
    d.add_argument("--force", action="store_true", help="overwrite an existing --out run directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            seed = fresh_seed() if args.seed is None else args.seed
            cfg = pipeline.GenerateConfig(seed=seed, instances=args.instances, source=args.source, feed=args.feed,
                                          snapshot_file=args.snapshot_file, variants=args.variants)
            run_dir = pipeline.generate(cfg, Path(args.out) if args.out else None, force=args.force)
            print(f"next: python cli.py run {run_dir}")
        elif args.command == "run":
            specs = [s.strip() for s in args.models.split(",") if s.strip()]
            pipeline.run_models(Path(args.run_dir), specs, workers=args.workers, force=args.force,
                                max_tokens=args.max_tokens)
        elif args.command == "score":
            pipeline.score_run(Path(args.run_dir))
        elif args.command == "report":
            pipeline.build_report(Path(args.run_dir))
            print("\n".join(pipeline.summary_lines(Path(args.run_dir))))
        elif args.command == "resolve":
            now = parse_utc_iso(args.now) if args.now else None
            pipeline.resolve_run(Path(args.run_dir), now=now)
        elif args.command == "demo":
            path = pipeline.demo(args.seed, args.instances, source=args.source,
                                 run_dir=Path(args.out) if args.out else None, force=args.force)
            print()
            print("\n".join(pipeline.summary_lines(path.parent)))
            print(f"\nreport: {path}")
    except (FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
