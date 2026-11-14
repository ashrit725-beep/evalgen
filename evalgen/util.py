"""Small shared helpers: canonical hashing, JSON files, the one writer of every file in a run
directory and the one reader of its manifest, UTC timestamps, exact number text."""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import reprlib
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

_TEMPORARY = itertools.count()  # numbers the temporary files of write_in, so that no two writes share one
MANIFEST_KEYS = ("evalgen_version", "run_id", "self_check")  # in every manifest that evalgen writes
SHA256_HEX = re.compile(r"[0-9a-f]{64}")  # a SHA-256 (a commitment) as evalgen writes it: 64 lower-case hex digits
LAST_SECOND = 253402300799  # 9999-12-31T23:59:59Z, the last moment that utc_iso can print
_SPLITS = ("public", "held_out")  # an inventory's lists (evalgen.mix.SPLITS, which imports this module)
_SHOWN = reprlib.Repr()  # what an error shows of a value read from a file: short, whatever its size or depth
_SHOWN.maxstring = _SHOWN.maxother = 40


class RunDirError(ValueError):
    """A run directory that evalgen will not read or write: a symbolic link in it, a path that leaves it,
    a manifest that evalgen did not write, or an instance file whose IDs are not the ones its name gives.
    A run received from someone else could hold any of these, to make a step read or overwrite a file
    outside the run directory. The command line reports it and exits with status 2."""


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal data always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def inside(root: Path, path: Path) -> Path:
    """`path`, checked to lie in the directory `root`: a path that is not below `root` as written (an
    absolute path elsewhere, or one that climbs out with '..'), and one that passes through a symbolic
    link at any step below `root`, raise RunDirError. `root` itself may be reached through links: the
    evaluator names it."""
    root, path = Path(root), Path(path)
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise RunDirError(f"{path} lies outside the run directory {root}") from None
    if not parts or os.pardir in parts:
        raise RunDirError(f"{path} is not a file in the run directory {root}")
    step = root
    for part in parts:
        step = step / part
        if step.is_symlink():
            raise RunDirError(f"{step} is a symbolic link: evalgen never reads or writes through a link in a run "
                              "directory, which could point anywhere")
    return path


def check_run_dir(run_dir: Path) -> None:
    """Refuse (RunDirError) a run directory that holds a symbolic link, at any depth: a run received from
    someone else could link a file or a folder anywhere, and a later step would read or write it there. It
    refuses as well an entry that is neither a file nor a folder (a named pipe, a socket, a device), which a
    step would block on, or read from without end, where it expects a file. A run directory that does not
    exist passes: the step that reads it says what is missing."""
    folders = [Path(run_dir)] if Path(run_dir).is_dir() else []
    while folders:
        with os.scandir(folders.pop()) as listing:
            entries = sorted(listing, key=lambda entry: entry.name)
        for entry in entries:
            if entry.is_symlink():
                raise RunDirError(f"{entry.path} is a symbolic link: evalgen does not read or write a run directory "
                                  "that holds one, since a run received from someone else could point it anywhere")
            if entry.is_dir(follow_symlinks=False):
                folders.append(Path(entry.path))
            elif not entry.is_file(follow_symlinks=False):
                raise RunDirError(f"{entry.path} is neither a file nor a folder: evalgen does not read or write a run "
                                  "directory that holds one (a named pipe would block the step that reads it)")


def write_in(root: Path, path: Path, data: str | bytes) -> None:
    """Write `data` (text, as UTF-8, or bytes) to the file `path` in the run directory `root`: the one way
    evalgen writes into a run directory. The path must lie in `root`, with no symbolic link on the way or
    in its place (inside). The missing folders are made, and the data goes whole into a new file beside
    `path`, which then takes its name (os.replace): a link or a hard link that takes the name meanwhile is
    replaced, never written through, and no reader sees half a file. Text is written as Path.write_text
    writes it, so the bytes are the same."""
    path = inside(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        temporary = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}-{next(_TEMPORARY)}.tmp")
        try:
            fd = os.open(temporary, flags, 0o666)  # the mode a new file gets from open(), umask applied
            break
        except FileExistsError:  # left by a write that was killed: take the next number
            continue
    try:
        with (os.fdopen(fd, "wb") if isinstance(data, bytes) else os.fdopen(fd, "w", encoding="utf-8")) as out:
            out.write(data)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def write_json(path: Path, obj: Any, *, root: Path | None = None) -> None:
    """Write `obj` as indented JSON to `path` (write_in), in the run directory `root`; without one, in the
    file's own folder, so that a symbolic link in the file's place is refused, never written through."""
    path = Path(path)
    write_in(path.parent if root is None else root, path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    """The JSON in the file `path`, read as UTF-8. ValueError for text that is not UTF-8 or not JSON, and for
    JSON nested too deeply to read (json itself raises RecursionError, which no caller would expect)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except RecursionError:
        raise ValueError("its JSON is nested too deeply to read") from None


def shown(value: Any) -> str:
    """A value read from a file, as an error message shows it: its repr, cut short however long or deep it is."""
    return _SHOWN.repr(value)


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """The manifest of the run in `run_dir`, read as every step after generate reads it: the directory may
    hold no symbolic link (check_run_dir), and its manifest.json must be one that evalgen wrote, a JSON
    object with MANIFEST_KEYS and every key that a later step reads in the shape evalgen writes it
    (manifest_problems). FileNotFoundError for a folder (or a file) that is not a run, so that a mistyped
    path stops a step instead of passing for a run with nothing to do; RunDirError otherwise."""
    run_dir = Path(run_dir)
    check_run_dir(run_dir)
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} is not an evalgen run directory: it has no manifest.json")
    try:
        manifest = read_json(path)
    except ValueError as exc:  # not UTF-8, not JSON, or nested too deeply to read
        raise RunDirError(f"{path} is not JSON ({exc})") from None
    problems = manifest_problems(manifest)
    if problems:
        raise RunDirError(f"{path} is not the manifest of an evalgen run: {'; '.join(problems)}")
    return manifest


def manifest_problems(manifest: Any) -> list[str]:
    """What keeps `manifest` (JSON as read) from being the manifest of a run as evalgen writes it: [] for one
    that is. Every step after generate reads a run's manifest through read_manifest, which refuses one with
    problems, and `generate --force` replaces only a folder whose manifest has none (pipeline.is_run_dir)."""
    if not isinstance(manifest, dict):
        return ["it is not a JSON object"]
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        return [f"it has no {', '.join(missing)}"]
    return _manifest_problems(manifest)


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _snapshot_ok(snapshot: Any) -> bool:
    """A snapshot's record in a manifest (sources.Snapshot.meta), as the report reads it: its source and hash,
    its pool size, and a compile time and window that it prints."""
    return (isinstance(snapshot, dict) and isinstance(snapshot.get("source"), str)
            and isinstance(snapshot.get("events_sha256"), str) and _whole(snapshot.get("n_events"))
            and _whole(snapshot.get("compiled_at")) and 0 <= snapshot["compiled_at"] <= LAST_SECOND
            and _whole(snapshot.get("window_hours"))
            and all(isinstance(snapshot.get(key), (str, type(None))) for key in ("raw_sha256", "url", "note")))


def _template_ok(template: Any) -> bool:
    """A template's record in a manifest (questions.Template.to_dict), as the report's table prints it."""
    return (isinstance(template, dict) and all(isinstance(template.get(key), str) for key in ("template_id", "pool",
                                                                                             "skeleton"))
            and isinstance(template.get("twin_of"), (str, type(None))) and isinstance(template.get("domain", ""), str))


def _manifest_problems(manifest: dict[str, Any]) -> list[str]:
    """What is wrong with the keys of a manifest that later steps index: its run ID and version (text), the
    seed (a whole number) and its commitment (64 hexadecimal digits), the instances (a list of IDs), the
    generator's self-check (its counts and problems), its human audit sample (its counts and rate), its
    templates (public and new, each with its ID, pool and skeleton), its snapshot (or one per domain: source,
    hash, pool size, and a compile time that a report can print) and the code's hash; and where they are
    given, the commitment to a private run config (64 hexadecimal digits), the plugins (files, each with its
    path), the inventory (lists of names, public and held out, of domains and of genres) and the config (an
    object). evalgen writes them so, in every version; a run received from someone else may not, and a step
    would then stop on a traceback."""
    problems = []
    for key in ("run_id", "evalgen_version"):
        if not isinstance(manifest.get(key), str):
            problems.append(f"its {key} is {shown(manifest.get(key))}, not text")
    seed = manifest.get("seed")
    if not _whole(seed):
        problems.append(f"its seed is {shown(seed)}, not a whole number")
    for key in ("seed_commitment", "private_config_commitment"):
        if (key in manifest or key == "seed_commitment") and not (
                isinstance(manifest.get(key), str) and SHA256_HEX.fullmatch(manifest[key])):
            problems.append(f"its {key} is {shown(manifest.get(key))}, not 64 hexadecimal digits")
    instances = manifest.get("instances")
    if not _strings(instances):
        problems.append("it lists no instances by their IDs")
    check = manifest.get("self_check")
    if not (isinstance(check, dict) and all(_whole(check.get(key)) for key in (
            "oracle_checks", "oracle_agreements", "resolution_invariant_instances")) and _strings(check.get("problems"))):
        problems.append("its self_check is not the generator's counts and problems")
    audit = manifest.get("audit")
    if not (isinstance(audit, dict) and _whole(audit.get("population")) and _whole(audit.get("sampled"))
            and _real(audit.get("rate"))):
        problems.append("its audit is not the counts and rate of a human audit sample")
    templates = manifest.get("templates")
    if not (isinstance(templates, dict) and all(isinstance(templates.get(pool), list) and all(map(
            _template_ok, templates[pool])) for pool in ("public", "new"))):
        problems.append("its templates are not the run's public and new templates")
    if "snapshot" not in manifest and "snapshots" not in manifest:
        problems.append("it records no snapshot")
    if "snapshot" in manifest and not _snapshot_ok(manifest["snapshot"]):
        problems.append("its snapshot is not the record of a snapshot")
    snapshots = manifest.get("snapshots")
    if "snapshots" in manifest and not (isinstance(snapshots, dict) and snapshots and all(
            map(_snapshot_ok, snapshots.values()))):
        problems.append("its snapshots are not the records of a snapshot per domain")
    if not isinstance(manifest.get("code_sha256"), (str, type(None))):
        problems.append("its code_sha256 is not text")
    plugins = manifest.get("plugins")
    if plugins is not None and (not isinstance(plugins, list) or not all(
            isinstance(entry, dict) and isinstance(entry.get("path"), str) for entry in plugins)):
        problems.append("its plugins are not files, each with its path")
    inventory = manifest.get("inventory")
    if inventory is not None and (not isinstance(inventory, dict) or not all(
            isinstance(inventory.get(kind), dict) and set(inventory[kind]) <= set(_SPLITS)
            and all(_strings(names) and all(names) for names in inventory[kind].values())
            for kind in ("domains", "genres"))):
        problems.append("its inventory is not lists of names, public and held out, of domains and of genres")
    if "config" in manifest and not isinstance(manifest["config"], dict):
        problems.append("its config is not an object")
    return problems


def plain_number(x: float | int) -> str:
    """A number as text, exactly as stored and never in exponent notation: 3.1, 12 (for 12.0), -0,
    0.00005, 1234567.5. It agrees with format(x, "g") wherever that is exact; "g" keeps only six
    significant digits and switches to exponents for small and large values."""
    if isinstance(x, float):
        text = format(Decimal(repr(x)), "f")
        return text[:-2] if text.endswith(".0") else text
    return str(x)


def utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_iso(text: str) -> int:
    """Parse '2026-01-14T06:00:00Z' (or with +00:00) into Unix seconds."""
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def slugify(name: str) -> str:
    """Model spec -> safe directory name, e.g. 'anthropic:claude-x' -> 'anthropic_claude-x'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"
