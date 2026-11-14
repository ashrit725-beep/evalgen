#!/usr/bin/env python3
"""evalgen command line.

    python cli.py demo --seed 42 --instances 20          # offline with the default synthetic source, no API keys
    python cli.py generate --seed 7 --instances 20 [--source usgs]   # a small seed: demos and examples only
    python cli.py generate --mix random --private-config private.json   # an evaluation: a fresh seed, held-out sets
    python cli.py run runs/random-synthetic-seed7 --private-config private.json   # (and score, report: plugins)
    python cli.py generate --seed 7 --mix random --snapshot-dir runs/X/snapshot   # replay a live run (its seed)
    python cli.py run runs/synthetic-seed7 --models mock,oracle,naive,anthropic,openai:gpt-4.1
    python cli.py score runs/synthetic-seed7
    python cli.py report runs/synthetic-seed7
    python cli.py resolve runs/usgs-seed7                  # later: grade forecast questions
    python cli.py verify-config runs/random-synthetic-seed7 private.json   # after revealing the config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalgen import pipeline
from evalgen.mix import verify_private_config
from evalgen.seeds import fresh_seed, guessable_seed
from evalgen.util import parse_utc_iso, read_manifest

SOURCES = ("synthetic", "usgs", "live")
SEED_HELP = ("master seed (default: a fresh random 64-bit seed, which is logged). Leave it out for an evaluation: a "
             "small seed such as 42 is for demos and reproductions, since anyone can guess it, or recover it from a "
             "single prompt in seconds")
SMALL_SEED_WARNING = ("warning: a seed this small can be guessed, and recovered from any one prompt of the run in "
                      "seconds, which gives away every instance: use it for demos and reproductions, and leave --seed "
                      "out for an evaluation (a fresh 64-bit seed is drawn and logged)")
MIX_HELP = ("which domains and genres instances come from: legacy (0.2.0: earthquakes in the classic genre; the "
            "default), random, or a list such as 'weather,neo:email' (a bare domain: in any genre)")
CONFIG_HELP = ("the evaluator's private run config (JSON): held-out domains and genres and the plugins that define "
               "them; its hash is committed in the manifest (see configs/private.example.json)")
DESCRIPTION = ("Live, fine-tuning-resistant LLM evaluation on seeded record digests: earthquakes in the 0.2.0 "
               "classic style by default, or (--mix) several domains in several genres, some held out by a private "
               "run config.")
DEMO_HELP = ("generate + run (mock, oracle, naive) + score + report: offline with the default synthetic source "
             "(usgs and live fetch the live feeds once, with a synthetic fallback)")
DEMO_SOURCE_HELP = ("synthetic data (the default; never touches the network), the USGS feed (usgs: earthquakes), or "
                    "each domain's own live source (live; off the legacy path). usgs and live fetch each feed once and "
                    "fall back to synthetic data when it cannot be read")
SOURCE_HELP = (DEMO_SOURCE_HELP + "; a replay (--snapshot-dir, or --snapshot-file off the legacy path) reads its "
               "frozen snapshots instead, whatever --source says")
SNAPSHOT_DIR_HELP = ("replay the frozen snapshots of an earlier run off the legacy path (its snapshot folder): a domain "
                     "that run read live is replayed from its frozen bytes, whatever the seed; a domain it drew "
                     "synthetic records for (by choice, or after a fallback) is drawn again from --seed, so it matches "
                     "that run only with that run's seed")
LOAD_HELP = ("the private run config the run was generated with, for a run with private plugins: its plugins are "
             "loaded from it (evalgen never runs code that a run directory names by itself)")


def _count(least: int):
    """An argparse type: a whole number of at least `least`."""
    def count(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            value = None
        if value is None or value < least:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of {least} or more")
        return value
    return count


def _config_hint(run_dir: Path, private_config: str | None) -> str:
    """The private run config that the next steps of a run with plugins need (" --private-config PATH"),
    or nothing for a run without plugins: they load the plugins from it, never from the run itself."""
    if private_config is None:
        return ""
    try:
        plugins = read_manifest(run_dir).get("plugins")
    except (OSError, ValueError):
        return ""
    return f" --private-config {private_config}" if plugins else ""


def _utc_time(text: str) -> int:
    """An argparse type: an ISO timestamp in UTC (or with its offset), as Unix seconds."""
    try:
        return parse_utc_iso(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an ISO timestamp such as 2026-01-14T06:00:00Z") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="build a fresh, seeded instance set (snapshot, digests, questions, key)")
    g.add_argument("--seed", type=int, help=SEED_HELP)
    g.add_argument("--instances", type=_count(1), default=20)
    g.add_argument("--source", choices=SOURCES, default="synthetic", help=SOURCE_HELP)
    g.add_argument("--feed", choices=("hour", "day"), default="day", help="USGS feed window (usgs source only)")
    g.add_argument("--snapshot-file", help="use a frozen USGS GeoJSON file (or a domain's frozen live feed) instead of "
                                           "fetching the live feed")
    g.add_argument("--snapshot-dir", help=SNAPSHOT_DIR_HELP)
    g.add_argument("--variants", type=_count(0), default=2, help="metamorphic re-renderings per instance")
    g.add_argument("--mix", default="legacy", help=MIX_HELP)
    g.add_argument("--private-config", help=CONFIG_HELP)
    g.add_argument("--out", help="run directory (default: runs/<source>-seed<seed>, with the mix first for any mix "
                                 "but legacy)")
    g.add_argument("--force", action="store_true", help="overwrite an existing run directory")

    r = sub.add_parser("run", help="send every prompt to one or more models and store the replies")
    r.add_argument("run_dir")
    r.add_argument("--models", default="mock,oracle,naive",
                   help="comma-separated: oracle, naive, mock[:held_out=P], anthropic[:model], openai[:model]; the "
                        "simulated mock's penalty on held-out sets is held_out=P (domains and genres), "
                        "held_out_domains=P or held_out_genres=P, joined with + for two penalties "
                        "(mock:held_out_domains=0.2+held_out_genres=0.3)")
    r.add_argument("--workers", type=_count(1), default=4, help="parallel requests for API models")
    r.add_argument("--force", action="store_true", help="ask again even if a stored reply exists")
    r.add_argument("--max-tokens", type=_count(1), default=16000,
                   help="output-token limit per reply for API models (models that think need room)")
    r.add_argument("--private-config", help=LOAD_HELP)

    s = sub.add_parser("score", help="parse and grade the stored replies")
    s.add_argument("run_dir")
    s.add_argument("--private-config", help=LOAD_HELP)

    rep = sub.add_parser("report", help="write report.md and report.json")
    rep.add_argument("run_dir")
    rep.add_argument("--private-config", help=LOAD_HELP)

    res = sub.add_parser("resolve", help="grade pending forecast questions against the USGS catalogue")
    res.add_argument("run_dir")
    res.add_argument("--now", type=_utc_time, help="pretend the current UTC time is this ISO timestamp (for testing)")
    res.add_argument("--private-config", help=LOAD_HELP)

    d = sub.add_parser("demo", help=DEMO_HELP, description=DEMO_HELP)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--instances", type=_count(1), default=20)
    d.add_argument("--source", choices=SOURCES, default="synthetic", help=DEMO_SOURCE_HELP)
    d.add_argument("--mix", default="legacy", help=MIX_HELP)
    d.add_argument("--private-config", help=CONFIG_HELP)
    d.add_argument("--out", help="run directory (default: runs/demo-<source>-seed<seed>, with the mix before the "
                                 "source for any mix but legacy, and heldout before the mix with --private-config; "
                                 "overwritten each time)")
    d.add_argument("--force", action="store_true", help="overwrite an existing --out run directory")

    v = sub.add_parser("verify-config", help="check a revealed private run config against a run's commitment")
    v.add_argument("run_dir")
    v.add_argument("config", help="the private run config (JSON) the evaluator revealed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command. Misuse ends with `error: ...` on stderr and exit status 2, never a traceback: a path
    that is not a run, a file where a folder belongs, a run that evalgen will not read (a received one with
    a link in it, say), a file that is not JSON, a mix, config or model that a run cannot follow. For
    verify-config, status 1 means only MISMATCH."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            seed = fresh_seed() if args.seed is None else args.seed
            cfg = pipeline.GenerateConfig(seed=seed, instances=args.instances, source=args.source, feed=args.feed,
                                          snapshot_file=args.snapshot_file, variants=args.variants, mix=args.mix,
                                          private_config=args.private_config, snapshot_dir=args.snapshot_dir)
            run_dir = pipeline.generate(cfg, Path(args.out) if args.out else None, force=args.force)
            if args.seed is not None and guessable_seed(args.seed):  # on the console only: no file changes with it
                print(SMALL_SEED_WARNING, file=sys.stderr)
            print(f"next: python cli.py run {run_dir}" + _config_hint(run_dir, args.private_config))
        elif args.command == "run":
            specs = [s.strip() for s in args.models.split(",") if s.strip()]
            pipeline.run_models(Path(args.run_dir), specs, workers=args.workers, force=args.force,
                                max_tokens=args.max_tokens, private_config=args.private_config)
        elif args.command == "score":
            pipeline.score_run(Path(args.run_dir), private_config=args.private_config)
        elif args.command == "report":
            pipeline.build_report(Path(args.run_dir), private_config=args.private_config)
            print("\n".join(pipeline.summary_lines(Path(args.run_dir))))
        elif args.command == "resolve":
            pipeline.resolve_run(Path(args.run_dir), now=args.now, private_config=args.private_config)
        elif args.command == "demo":
            path = pipeline.demo(args.seed, args.instances, source=args.source, mix=args.mix,
                                 private_config=args.private_config, run_dir=Path(args.out) if args.out else None,
                                 force=args.force)
            print()
            print("\n".join(pipeline.summary_lines(path.parent)))
            print(f"\nreport: {path}")
        elif args.command == "verify-config":
            if verify_private_config(args.config, args.run_dir):
                print(f"ok: {args.config} is the private run config that {args.run_dir} committed to")
                return 0
            print(f"MISMATCH: {args.config} is not the private run config that {args.run_dir} committed to")
            return 1
    except (OSError, ValueError) as exc:  # FileNotFoundError, IsADirectoryError, ...; MixError, RunDirError, ...
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
