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
    python cli.py verify-config --commitment HEX private.json   # after the reveal: HEX is the published commitment
    python cli.py verify-config runs/random-synthetic-seed7 private.json   # the config and a run agree
"""
from __future__ import annotations

import argparse
import re
import reprlib
import secrets
import shlex
import sys
from pathlib import Path

from evalgen import pipeline
from evalgen.mix import MixError, read_commitment, revealed_commitment, run_commitment
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
VERIFY_HELP = ("check a revealed private run config against the commitment published before the evaluation "
               "(--commitment HEX CONFIG), or against a run's manifest (RUN_DIR CONFIG), which says only that the "
               "config and that run agree: ok (exit status 0), MISMATCH (1) or error (2)")
COMMITMENT_HELP = ("the private_config_commitment that the evaluator published before the evaluation (64 hexadecimal "
                   "digits): the value a revealed config must hash to. Without it, CONFIG is checked against RUN_DIR's "
                   "manifest, and the commitment printed must be the published one")
VERIFY_USAGE = ("give the commitment published before the evaluation (verify-config --commitment HEX CONFIG) or the run "
                "whose manifest committed to the config (verify-config RUN_DIR CONFIG), one of the two")
WHOLE = re.compile(r"[+-]?[0-9]+")  # a whole number in plain ASCII digits: not "1_000", nor another script's digits
MOST_DIGITS = 4300  # what Python 3.11+ reads of an integer: 3.9 reads more, and every version reads alike here


def _whole(text: str) -> int | None:
    """`text` as a whole number written in plain digits (an optional sign, then ASCII digits), or None: int()
    also takes "1_000" and the digits of other scripts (Arabic-Indic ones, say), which nobody means as a seed
    or a count."""
    text = text.strip()
    if not WHOLE.fullmatch(text) or len(text.lstrip("+-")) > MOST_DIGITS:
        return None
    return int(text)


def _seed(text: str) -> int:
    """An argparse type: a master seed, a whole number in plain digits."""
    value = _whole(text)
    if value is None:
        raise argparse.ArgumentTypeError(f"{reprlib.repr(text)} is not a seed: a seed is a whole number written in "
                                         "plain digits, such as 42")
    return value


def _count(least: int, most: int | None = None):
    """An argparse type: a whole number in plain digits of at least `least` (and at most `most`)."""
    def count(text: str) -> int:
        value = _whole(text)
        if value is None or value < least or (most is not None and value > most):
            span = f"of {least} or more" if most is None else f"from {least} to {most}"
            raise argparse.ArgumentTypeError(f"{reprlib.repr(text)} is not a whole number {span}")
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
    return f" --private-config {shlex.quote(private_config)}" if plugins else ""


def _verify_config(run_dir: str | None, config: str, commitment: str | None) -> int:
    """verify-config: 0 when the config hashes to the commitment (the published one given, or the one a run
    records), 1 (MISMATCH) when it does not, and an error (MixError, ...: exit status 2) for anything else,
    so that 1 always means MISMATCH."""
    if (run_dir is None) == (commitment is None):
        raise MixError(VERIFY_USAGE)
    expected = read_commitment(commitment) if commitment is not None else run_commitment(run_dir)
    actual = revealed_commitment(config)
    if commitment is not None:
        if secrets.compare_digest(actual, expected):
            print(f"ok: {config} is the private run config committed to as {expected}")
            return 0
        print(f"MISMATCH: {config} is not the private run config committed to as {expected}: its commitment is {actual}")
        return 1
    if secrets.compare_digest(actual, expected):
        print(f"ok: {config} is the private run config that {run_dir} committed to, as {expected}. A run's manifest "
              "commits to the config that made the run, whichever that was: check that this is the commitment "
              "published before the evaluation, or check the config against the published value itself with "
              f"python cli.py verify-config --commitment HEX {shlex.quote(config)}")
        return 0
    print(f"MISMATCH: {config} is not the private run config that {run_dir} committed to, as {expected}: its "
          f"commitment is {actual}")
    return 1


def _console_never_fails() -> None:
    """Print what the console cannot show as escapes (\\u2026), never fail on it: under a console encoding
    that is not UTF-8 (LC_ALL=en_US.US-ASCII, or PYTHONIOENCODING=ascii), a path, a name or a line that it
    cannot encode would otherwise stop a command midway."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):  # not a text stream of io's (a capture, or closed)
            pass


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
    g.add_argument("--seed", type=_seed, help=SEED_HELP)
    g.add_argument("--instances", type=_count(1, pipeline.MAX_INSTANCES), default=20)
    g.add_argument("--source", choices=SOURCES, default="synthetic", help=SOURCE_HELP)
    g.add_argument("--feed", choices=("hour", "day"), default="day", help="USGS feed window (usgs source only)")
    g.add_argument("--snapshot-file", help="use a frozen USGS GeoJSON file (or a domain's frozen live feed) instead of "
                                           "fetching the live feed")
    g.add_argument("--snapshot-dir", help=SNAPSHOT_DIR_HELP)
    g.add_argument("--variants", type=_count(0, pipeline.MAX_VARIANTS), default=2,
                   help="metamorphic re-renderings per instance")
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
    d.add_argument("--seed", type=_seed, default=42)
    d.add_argument("--instances", type=_count(1, pipeline.MAX_INSTANCES), default=20)
    d.add_argument("--source", choices=SOURCES, default="synthetic", help=DEMO_SOURCE_HELP)
    d.add_argument("--mix", default="legacy", help=MIX_HELP)
    d.add_argument("--private-config", help=CONFIG_HELP)
    d.add_argument("--out", help="run directory (default: runs/demo-<source>-seed<seed>, with the mix before the "
                                 "source for any mix but legacy, and heldout before the mix with --private-config; "
                                 "overwritten each time)")
    d.add_argument("--force", action="store_true", help="overwrite an existing --out run directory")

    v = sub.add_parser("verify-config", help=VERIFY_HELP, description=VERIFY_HELP)
    v.add_argument("--commitment", metavar="HEX", help=COMMITMENT_HELP)
    v.add_argument("run_dir", nargs="?", metavar="RUN_DIR", help="a run generated with the config (without --commitment)")
    v.add_argument("config", metavar="CONFIG", help="the private run config (JSON) the evaluator revealed")
    return parser


def _command(args: argparse.Namespace) -> int:
    """Run the command that `args` name, and return its exit status."""
    if args.command == "generate":
        seed = fresh_seed() if args.seed is None else args.seed
        cfg = pipeline.GenerateConfig(seed=seed, instances=args.instances, source=args.source, feed=args.feed,
                                      snapshot_file=args.snapshot_file, variants=args.variants, mix=args.mix,
                                      private_config=args.private_config, snapshot_dir=args.snapshot_dir)
        run_dir = pipeline.generate(cfg, Path(args.out) if args.out else None, force=args.force)
        if args.seed is not None and guessable_seed(args.seed):  # on the console only: no file changes with it
            print(SMALL_SEED_WARNING, file=sys.stderr)
        print(f"next: python cli.py run {shlex.quote(str(run_dir))}" + _config_hint(run_dir, args.private_config))
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
        return _verify_config(args.run_dir, args.config, args.commitment)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one command. Misuse ends with `error: ...` on stderr and exit status 2, never a traceback: a path
    that is not a run, a file where a folder belongs, a run that evalgen will not read (a received one with
    a link in it, or a file of the wrong shape), a file that is not JSON, a mix, config or model that a run
    cannot follow, a generator self-check that failed off the legacy path. So does a failure that no check
    foresaw (a received file, or code that a plugin runs, of a shape nothing expected; or a plugin that calls
    sys.exit), named by its type. For verify-config, status 1 means only MISMATCH. What the console cannot
    encode is printed as escapes, never a failure."""
    _console_never_fails()
    args = _parser().parse_args(argv)
    try:
        return _command(args)
    except (OSError, ValueError) as exc:  # FileNotFoundError, IsADirectoryError, ...; MixError, RunDirError, ...
        print(f"error: {exc}", file=sys.stderr)
    except SystemExit as exc:  # evalgen never ends a command so: code that a plugin ran did
        print(f"error: code that this command ran called sys.exit({exc.code!r}), which evalgen never does: a "
              "plugin's, most likely", file=sys.stderr)
    except Exception as exc:  # a file of a received run, or a plugin's code, that no check foresaw
        print(f"error: {type(exc).__name__}: {exc} (unexpected: a file of the run or a plugin is not as evalgen writes "
              "or expects it, or this is a bug in evalgen)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
