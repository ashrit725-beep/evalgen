"""Which domains and genres a run's instances come from (GenerateConfig.mix), and which of them the
evaluator holds out (GenerateConfig.private_config, the private run config).

Mixes:

    legacy        0.2.0, the default: earthquakes in the classic genre, on the 0.2.0 seed paths
    random        every domain and genre of the run's inventory (see below)
    toy:classic   every instance about the toy domain, and every rendering in the classic genre
    toy           the toy domain, in any genre
    a,b:news      a list: each entry is a domain, alone or with ":" and a genre

Any mix but legacy is recorded in the manifest, and puts the run on seed paths of its own (its
snapshots, its twin templates, its instances' surfaces), so the legacy draws never move. Each instance
then records its domain and whether that domain is public or held out (its split), and each rendering
its genre. A mix that leaves a choice draws every instance from ``seedlog.rng("instance", i, "mix")``,
a path that no other run creates: first its domain (each domain of the mix alike), then the genre of
its original rendering (each genre the mix allows for that domain alike), then the genre of each
metamorphic variant: one other than the original's whenever the mix allows one, every other genre in
turn before any of them twice. A mix that names a single domain and genre draws nothing.

The private run config is a JSON file the evaluator keeps private until after the evaluation:

    {"nonce": "<at least 32 random hexadecimal digits: secrets.token_hex(16)>",
     "domains": {"public": ["earthquakes"], "held_out": ["tides"]},
     "genres": {"public": ["classic"], "held_out": ["fax"]},
     "plugins": [{"path": "plugins/tides.py", "sha256": "<the SHA-256 of the file's bytes>"},
                 {"path": "plugins/fax.py", "sha256": "<...>"}],
     "note": "anything (optional)"}

Plugins are Python files outside the public repository (paths relative to the config) that define a
domain (``DOMAIN``), a genre (``GENRE``) or both. Each is given by its path and the SHA-256 of its
bytes, or by its path alone (a string). They are loaded before any name is looked up, and a plugin
whose bytes differ from the SHA-256 its config gives is refused, unexecuted, at generation and at every
later step; a run records each plugin's path as the config names it, and the SHA-256 of its bytes (it
records the config itself by its file name alone: a run names no folder of the evaluator's machine that
the revealed config does not). Every later step (run, score, report,
resolve) loads them again from the config, which the evaluator passes again (``--private-config``)
and the run's commitment authenticates (``load_run_plugins``): evalgen never executes a file that a
run directory names by itself, so reporting on a run received from someone else runs none of its
code. With a config, a run draws only from the domains and genres the config lists, with their
splits; without one, every registered domain and genre is public. The config is read strictly: a
file that is not JSON, or that gives a key twice in one object (JSON readers keep one of the two, so
such a file could show one config and verify as another), is refused.

Like the seed, the config is committed by hash: the manifest records ``sha256(canonical_json(config))``
(``config_commitment``), and the random nonce stops anyone from guessing a small config from its hash.
So a nonce must look drawn, not typed (``nonce_problem``): 32 hexadecimal digits or more, not a run of
one digit or a pattern. The one exception is a nonce that says it is an EXAMPLE, as the published
configs/private.example.json does: it is accepted, so that the demo of held-out sets works offline,
but generate warns loudly that its commitment can be guessed.

What the evaluator publishes, and when:

    before the evaluation  the two commitments that generate records in the run's manifest: the seed's
                           (seed_commitment) and the config's (private_config_commitment). Nothing
                           else: not the manifest, which holds the seed, nor the config or its plugins.
    after the evaluation   the config, its plugin files, and the seed with its nonce. Anyone can then
                           check the config against its commitment (``verify_private_config``, ``cli.py
                           verify-config``) and regenerate the run, which checks every plugin against
                           the SHA-256 its config gives before running it.

A config names its plugins by path, so the commitment pins what a held-out domain or genre is only
through each plugin's SHA-256: with it, the commitment covers the plugins' content, which can no longer
change once the commitment is published; a plugin given by its path alone is committed to by its path
only, and could be edited after the commitment without anyone noticing.
"""
from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from . import domains, genres, registry
from .domains import all_domains, get_domain
from .domains.base import Domain
from .genres import all_genres, get_genre
from .genres.base import Genre
from .seeds import SeedLog
from .util import canonical_json, read_json, sha256_hex

LEGACY = "legacy"
RANDOM = "random"
PUBLIC = "public"
HELD_OUT = "held_out"
SPLITS = (PUBLIC, HELD_OUT)  # the order in which inventories, reports and tables list them

CONFIG_KEYS = ("nonce", "domains", "genres", "plugins", "note")  # every key a private run config may have
REQUIRED_KEYS = ("nonce", "domains", "genres")
# A nonce is 128 random bits or more, written in hexadecimal (secrets.token_hex(16)): anything less, or
# a word or pattern someone typed, would let a small config be recovered from its hash by brute force.
MIN_NONCE_DIGITS = 32
# A random draw of 32 hexadecimal digits shows fewer than 8 different digits about once in 30 million.
MIN_NONCE_DISTINCT_DIGITS = 8
EXAMPLE_MARK = "EXAMPLE"  # in the nonce of a published example config, whose commitment anyone can guess
NONCE_ADVICE = ("its nonce must be at least 32 random hexadecimal digits: draw one with "
                "python -c \"import secrets; print(secrets.token_hex(16))\"")
PLUGIN_KEYS = ("path", "sha256")  # a plugin given with the SHA-256 of its bytes
SHA256_DIGITS = 64

Log = Optional[Callable[[str], None]]  # Optional, not "X | None": this alias is evaluated at import (Python 3.9)


class MixError(ValueError):
    """A mix, or a private run config, that a run cannot follow. Generation raises it before writing
    anything, and the command line reports it and exits with status 2."""


class UnknownName(MixError, KeyError):
    """A mix or a private run config names a domain or a genre that no registry has."""

    def __str__(self) -> str:  # the message itself, not KeyError's quoted repr of it
        return str(self.args[0]) if self.args else ""


class DuplicateKey(MixError):
    """A private run config gives a key twice in one JSON object."""


# ------------------------------------------------------------------------------------ entries


@dataclass(frozen=True)
class Entry:
    """One entry of a mix: a domain, and the genre its instances are rendered in (None: any genre)."""

    domain: str
    genre: str | None = None


def _domain(name: str) -> Domain:
    try:
        return get_domain(name)
    except KeyError as exc:
        raise UnknownName(str(exc.args[0])) from None


def _genre(name: str) -> Genre:
    try:
        return get_genre(name)
    except KeyError as exc:
        raise UnknownName(str(exc.args[0])) from None


def parse_mix(text: str) -> tuple[Entry, ...]:
    """The entries of a mix that lists them, with every name checked (an unknown one raises
    UnknownName, a KeyError); () for legacy. The random mix has no entries of its own: it draws from
    the run's whole inventory (``plan_mix``)."""
    text = text.strip()
    if text == LEGACY:
        return ()
    if text == RANDOM:
        raise MixError("the random mix lists no entries: it draws from every domain and genre of the run (plan_mix)")
    entries = []
    for part in text.split(","):
        domain, colon, genre = (piece.strip() for piece in part.partition(":"))
        if not domain or (colon and not genre):
            raise MixError(f"mix entry {part.strip()!r} is not a domain or domain:genre")
        _domain(domain)
        if genre:
            _genre(genre)
        entries.append(Entry(domain, genre or None))
    if len(set(entries)) != len(entries):
        raise MixError(f"the mix {text!r} names an entry twice")
    return tuple(entries)


def choices(entries: Sequence[Entry], genre_pool: Sequence[Genre] | None = None) -> list[tuple[Domain, Genre]]:
    """Every (domain, genre) an instance of the mix may be, in the order the entries name them. A bare
    domain may be in every genre of `genre_pool` (every registered genre by default), in its order."""
    pool = all_genres() if genre_pool is None else list(genre_pool)
    out: list[tuple[Domain, Genre]] = []
    for entry in entries:
        domain = _domain(entry.domain)
        for genre in [_genre(entry.genre)] if entry.genre else pool:
            if (domain, genre) not in out:
                out.append((domain, genre))
    return out


# ---------------------------------------------------------------------- the private run config


def is_example_nonce(nonce: Any) -> bool:
    """Whether a nonce says it is a published example's (configs/private.example.json): such a config
    serves demos, and its commitment can be guessed."""
    return isinstance(nonce, str) and EXAMPLE_MARK in nonce


def nonce_problem(nonce: Any) -> str | None:
    """What is wrong with a private run config's nonce, or None. A nonce must look drawn rather than
    typed: at least MIN_NONCE_DIGITS hexadecimal digits, at least MIN_NONCE_DISTINCT_DIGITS different
    ones, and not one shorter run of digits repeated ("abab..."). A nonce that says it is an EXAMPLE
    is accepted (generate warns that its commitment can be guessed). No check can prove a nonce random:
    these refuse the guessable nonces that a config gets by accident, a word, a constant or a pattern."""
    if is_example_nonce(nonce):
        return None
    digits = nonce.strip().lower() if isinstance(nonce, str) else ""
    if (len(digits) < MIN_NONCE_DIGITS or any(c not in "0123456789abcdef" for c in digits)
            or len(set(digits)) < MIN_NONCE_DISTINCT_DIGITS or digits in (digits + digits)[1:-1]):
        return NONCE_ADVICE
    return None


def _plugin_entry_ok(entry: Any) -> bool:
    """A plugin as a config may give it: a path, or {"path": <a path>, "sha256": <64 hexadecimal digits>}."""
    def is_path(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    if isinstance(entry, dict):
        digest = entry.get("sha256")
        return (set(entry) == set(PLUGIN_KEYS) and is_path(entry["path"]) and isinstance(digest, str)
                and len(digest) == SHA256_DIGITS and all(c in "0123456789abcdef" for c in digest.lower()))
    return is_path(entry)


def config_problems(data: Any) -> list[str]:
    """What is wrong with the shape of a private run config (names are checked once its plugins are
    loaded): its keys, a nonce that looks drawn (``nonce_problem``), public and held-out lists of
    distinct names for domains and for genres (a name in one list only, and at least one name of each
    kind), plugins as a list of paths, each alone or with its SHA-256, and an optional note."""
    if not isinstance(data, dict):
        return ["it is not a JSON object"]
    problems = [f"it has no {key!r}" for key in REQUIRED_KEYS if key not in data]
    problems += [f"{key!r} is not a key of a private run config" for key in data if key not in CONFIG_KEYS]
    nonce = nonce_problem(data["nonce"]) if "nonce" in data else None
    if nonce is not None:
        problems.append(nonce)
    for kind in ("domains", "genres"):
        if kind not in data:
            continue
        splits = data[kind]
        if not isinstance(splits, dict) or set(splits) != set(SPLITS):
            problems.append(f"{kind!r} must have exactly the lists 'public' and 'held_out'")
            continue
        named: list[str] = []
        for split in SPLITS:
            names = splits[split]
            if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() for name in names):
                problems.append(f"{kind}.{split} must be a list of names")
                continue
            named += names
        twice = sorted({name for name in named if named.count(name) > 1})
        if twice:
            problems.append(f"{kind} lists {', '.join(twice)} more than once")
        if not named:
            problems.append(f"it lists no {kind}")
    plugins = data.get("plugins", [])
    if not isinstance(plugins, list) or not all(map(_plugin_entry_ok, plugins)):
        problems.append("'plugins' must be a list of paths, each alone or as {\"path\": <path>, \"sha256\": <the "
                        "SHA-256 of the file, 64 hexadecimal digits>}")
    if not isinstance(data.get("note", ""), str):
        problems.append("'note' must be a string")
    return problems


@dataclass(frozen=True)
class PrivateConfig:
    """A private run config as read and checked (``read_private_config``)."""

    path: Path
    data: Mapping[str, Any]  # the JSON object exactly as read: what the commitment hashes
    domains: Mapping[str, str]  # name -> split, as the config lists them
    genres: Mapping[str, str]
    plugins: tuple[Path, ...]  # absolute paths
    # The SHA-256 the config gives each plugin's bytes, in lower case, in the order of `plugins`; None
    # for a plugin given by its path alone.
    plugin_sha256: tuple[str | None, ...] = ()

    @property
    def commitment(self) -> str:
        return config_commitment(self.data)

    @property
    def example(self) -> bool:
        """The config keeps a published example's nonce, so anyone can guess it from its commitment."""
        return is_example_nonce(self.data["nonce"])

    def pinned(self) -> list[tuple[Path, str | None]]:
        """Each plugin's path and the SHA-256 the config commits its bytes to (None: none)."""
        return list(zip(self.plugins, self.plugin_sha256 or (None,) * len(self.plugins)))


def _distinct_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """The members of a JSON object, refusing a key given twice: json.loads would keep the last of the
    two, so a file could show a decoy first and still read, and verify, as another config."""
    members: dict[str, Any] = {}
    for key, value in pairs:
        if key in members:
            raise DuplicateKey(f"it gives the key {key!r} twice in one object")
        members[key] = value
    return members


def _read_config_json(path: Path) -> Any:
    """The JSON of a config file, read strictly: DuplicateKey for a key given twice in any one object,
    ValueError for text that is not JSON (or not UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_distinct_members)


def read_private_config(path: str | Path) -> PrivateConfig:
    """Read and check a private run config (``config_problems``); a missing file raises
    FileNotFoundError, a malformed one MixError (not JSON, a key given twice, or a shape that
    config_problems refuses). Plugin paths are relative to the config's folder."""
    path = Path(path)
    try:
        data = _read_config_json(path)
    except DuplicateKey as exc:
        raise MixError(f"the private run config {path} is malformed: {exc}") from None
    except ValueError as exc:
        raise MixError(f"the private run config {path} is not JSON ({exc})") from None
    problems = config_problems(data)
    if problems:
        raise MixError(f"the private run config {path} cannot be used: " + "; ".join(problems))
    entries = data.get("plugins", [])
    return PrivateConfig(
        path=path, data=data,
        domains={name: split for split in SPLITS for name in data["domains"][split]},
        genres={name: split for split in SPLITS for name in data["genres"][split]},
        plugins=tuple((path.parent / (e if isinstance(e, str) else e["path"])).resolve() for e in entries),
        plugin_sha256=tuple(None if isinstance(e, str) else e["sha256"].lower() for e in entries),
    )


def config_commitment(config: Mapping[str, Any]) -> str:
    """The commitment to a private run config: the SHA-256 of its canonical JSON (sorted keys, no
    whitespace), so that only its content counts, not how the file is laid out."""
    return sha256_hex(canonical_json(config))


def verify_config(config: Mapping[str, Any], commitment: str) -> bool:
    """Whether a revealed config is the one a commitment was made to."""
    return secrets.compare_digest(config_commitment(config), commitment)


def verify_private_config(path: str | Path, run_dir: str | Path) -> bool:
    """Whether the private run config at `path` is the one the run in `run_dir` committed to (its
    manifest's private_config_commitment). A run generated without a config raises MixError; a file
    that is not JSON is not the config. A file that gives a key twice in one object raises MixError: it
    reads as two configs (the first value of the key or the last), so whether it is the committed one
    has no answer. The plugins' bytes are checked when a run loads them (see the module docstring)."""
    manifest = read_json(Path(run_dir) / "manifest.json")
    recorded = manifest.get("private_config_commitment")
    if recorded is None:
        raise MixError(f"{run_dir} was generated without a private run config, so it committed to none")
    try:
        data = _read_config_json(Path(path))
    except DuplicateKey as exc:
        raise MixError(f"{path} is malformed: {exc}, so it cannot be checked against a commitment") from None
    except ValueError:
        return False
    return verify_config(data, recorded)


# ------------------------------------------------------------------------------------ plugins

Entrant = Union[Domain, Genre]  # Union, not "X | Y": this alias is evaluated at import (Python 3.9)

# The plugin files of each private run config whose plugins this process has loaded (plan_mix, or
# load_run_plugins with the config), by the config's commitment: what a later step of the same process
# registers again without the config. A run names its plugins as its config does, often relative to the
# config's folder, so the run alone does not say where they are on this machine.
_CONFIG_PLUGINS: dict[str, tuple[Path, ...]] = {}


def load_plugin(path: str | Path, sha256: str | None = None) -> list[Entrant]:
    """Register what a plugin file defines: its domain (``DOMAIN``), its genre (``GENRE``) or both,
    each in its registry. With `sha256` (what a private run config commits the file to), a file whose
    bytes have another SHA-256 is refused before any of it runs, and what runs is exactly the bytes
    that were checked. Raises MixError for a file that is gone, refused, defines neither entry, or
    defines one that its registry refuses (a name a built-in or another plugin already has, or a
    domain that does not hold together)."""
    path = Path(path).resolve()
    if not path.is_file():
        raise MixError(f"the plugin {path} does not exist")
    source = path.read_bytes()
    if sha256 is not None and sha256_hex(source) != sha256.lower():
        raise MixError(f"the plugin {path} is not the file its private run config commits to: its SHA-256 is "
                       f"{sha256_hex(source)}, and the config gives {sha256.lower()}")
    module = registry.load_module(path, source)
    loaded: list[Entrant] = []
    try:
        if hasattr(module, "DOMAIN"):
            loaded.append(domains.load_plugin(path))
        if hasattr(module, "GENRE"):
            loaded.append(genres.load_plugin(path))
    except (TypeError, ValueError) as exc:  # the registry refuses the entry
        raise MixError(f"the plugin {path} cannot be registered: {exc}") from None
    if not loaded:
        raise MixError(f"the plugin {path} defines neither DOMAIN nor GENRE")
    return loaded


def plugin_record(config: PrivateConfig) -> list[dict[str, str]]:
    """The plugins of a config as a manifest records them, in the config's order: each file's path as
    the config names it (relative to the config, unless the config gives an absolute path), so that a
    run names no folder of the evaluator's machine that the revealed config does not, and the SHA-256
    of its bytes."""
    named = [entry if isinstance(entry, str) else entry["path"] for entry in config.data.get("plugins", [])]
    return [{"path": name, "sha256": sha256_hex(path.read_bytes())} for name, path in zip(named, config.plugins)]


def load_run_plugins(manifest: Mapping[str, Any], private_config: str | Path | None = None,
                     log: Log = None) -> list[str]:
    """Register the plugins a run was generated with (its manifest lists them, each with its SHA-256),
    so that a later step, maybe in a new process, can read, answer, score and report the run's domains
    and genres; and return those whose bytes have changed since generation, as the manifest names them
    (plugin_record), which a report records. A legacy run, or any run without plugins, needs none.

    evalgen never runs code that a run directory names by itself: a run received from someone else
    could name any file. A plugin is loaded from the evaluator's private run config (`private_config`):
    the config the run was generated with, as its commitment proves, whose plugin paths are relative
    to the config. A plugin that the config gives with a SHA-256 is refused (MixError) if its bytes
    differ from it, as at generation: the commitment covers it. One given by its path alone whose bytes
    differ from the manifest's record is loaded all the same, as the evaluator asked, with a warning to
    `log`. Without a config, only plugins that this process has run already, as the plugins of the config
    that the run committed to (it generated the run, or loaded them for an earlier step), are registered
    again (_CONFIG_PLUGINS), and none of their code runs again; any other plugin, and a config the run did
    not commit to, raise MixError, as does a plugin that is gone."""
    entries = list(manifest.get("plugins") or ())
    recorded = manifest.get("private_config_commitment")
    if private_config is not None:
        config = read_private_config(private_config)
        if recorded is None or not verify_config(config.data, recorded):
            raise MixError(f"{private_config} is not the private run config that this run committed to")
        plugins = config.pinned()
        if len(plugins) != len(entries):
            raise MixError("the run's manifest does not list the plugins of its private run config")
    else:
        paths = _CONFIG_PLUGINS.get(recorded) if isinstance(recorded, str) else None
        if paths is None or len(paths) != len(entries):
            paths = (None,) * len(entries)
        strange = [str(entry["path"]) for entry, path in zip(entries, paths)
                   if path is None or not registry.executed(path)]
        if strange:
            raise MixError(f"the run was generated with the plugins {', '.join(strange)}: pass its private run config "
                           "(--private-config) to load them. evalgen never runs code that a run directory names by "
                           "itself")
        plugins = [(path, None) for path in paths]
    changed = []
    for (path, sha256), entry in zip(plugins, entries):
        if not path.is_file():
            raise MixError(f"the run was generated with the plugin {path}, which is gone: restore it to read the run")
        if sha256 is None and entry.get("sha256") and sha256_hex(path.read_bytes()) != entry["sha256"]:
            changed.append(str(entry["path"]))
            if log is not None:
                log(f"warning: the plugin {path} has changed since the run was generated")
        load_plugin(path, sha256=sha256)
    if private_config is not None and plugins:
        _CONFIG_PLUGINS[recorded] = tuple(path for path, _ in plugins)
    return changed


# ---------------------------------------------------------------------------------- inventory


@dataclass(frozen=True)
class Inventory:
    """The domains and genres a run draws its instances from, each public or held out. A manifest
    records it for every run off the legacy path."""

    domains: Mapping[str, str]  # name -> split, in the order the run draws them
    genres: Mapping[str, str]

    def domain_split(self, name: str) -> str:
        """A domain's split; one the inventory does not list is public, as every domain of a run
        without a private config is."""
        return self.domains.get(name, PUBLIC)

    def genre_split(self, name: str) -> str:
        return self.genres.get(name, PUBLIC)

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        """{"domains": {"public": [...], "held_out": [...]}, "genres": {...}}"""
        return {kind: {split: [name for name, s in names.items() if s == split] for split in SPLITS}
                for kind, names in (("domains", self.domains), ("genres", self.genres))}

    @classmethod
    def from_dict(cls, d: Mapping[str, Mapping[str, Sequence[str]]]) -> Inventory:
        return cls(*({name: split for split in SPLITS for name in d[kind].get(split, ())}
                     for kind in ("domains", "genres")))

    @classmethod
    def of_run(cls, manifest: Mapping[str, Any]) -> Inventory | None:
        """The inventory a run's manifest records; None for a legacy run (and for a run generated before
        inventories were recorded)."""
        recorded = manifest.get("inventory")
        return None if recorded is None else cls.from_dict(recorded)


# --------------------------------------------------------------------------------- draws


@dataclass(frozen=True)
class Assignment:
    """What one instance is drawn as: its domain, that domain's split, and the genre of each of its
    renderings, the original's first."""

    domain: Domain
    split: str
    genres: tuple[Genre, ...]


@dataclass(frozen=True)
class Plan:
    """A run's mix, read against the registries and the private run config: every (domain, genre) an
    instance may be drawn as (none on the legacy path), the run's inventory and the config."""

    mix: str
    options: tuple[tuple[Domain, Genre], ...]
    inventory: Inventory | None = None  # None on the legacy path
    config: PrivateConfig | None = None

    @property
    def legacy(self) -> bool:
        return not self.options

    @property
    def domains(self) -> list[Domain]:
        """The domains instances may be about, in the order the mix names them (registry order for random)."""
        return list(dict.fromkeys(domain for domain, _ in self.options))

    def genres_for(self, domain: Domain) -> list[Genre]:
        """The genres a rendering of an instance about `domain` may be in."""
        return [genre for d, genre in self.options if d is domain]

    def draw(self, index: int, seedlog: SeedLog, renderings: int) -> Assignment | None:
        """The domain, split and genres of instance `index` with `renderings` renderings; None on the
        legacy path. Only a mix that leaves a choice draws, from seed path ("instance", index, "mix")."""
        if self.legacy:
            return None
        if len(self.options) == 1:
            domain, genre = self.options[0]
            return Assignment(domain, self._split(domain), (genre,) * renderings)
        rng = seedlog.rng("instance", index, "mix")
        domain = rng.choice(self.domains)
        allowed = self.genres_for(domain)
        return Assignment(domain, self._split(domain), _genre_plan(allowed, renderings, rng))

    def _split(self, domain: Domain) -> str:
        return PUBLIC if self.inventory is None else self.inventory.domain_split(domain.name)


def _genre_plan(allowed: Sequence[Genre], renderings: int, rng: random.Random) -> tuple[Genre, ...]:
    """The original's genre, drawn from `allowed`, then each variant's: a genre other than the
    original's whenever there is one, all of the others in turn before any of them twice."""
    original = rng.choice(list(allowed))
    others = [genre for genre in allowed if genre is not original]
    plan, unused = [original], []
    for _ in range(renderings - 1):
        if not others:
            plan.append(original)
            continue
        if not unused:
            unused = list(others)
        plan.append(unused.pop(rng.randrange(len(unused))))
    return tuple(plan)


def plan_mix(mix: str, private_config: str | Path | None = None) -> Plan:
    """Read a run's mix and its private run config (path) before anything is generated: load the
    config's plugins, check every name, and list what instances may be drawn as. Without a config,
    every registered domain and genre is public; with one, the mix may only name what the config
    lists, and random draws from all of that. A legacy run takes no config: it is the 0.2.0 run, with
    nothing held out. Raises MixError (or UnknownName) for anything a run could not follow."""
    text = mix.strip()
    config = None if private_config is None else read_private_config(private_config)
    if text == LEGACY:
        if config is not None:
            raise MixError("a legacy run is the 0.2.0 run, with nothing held out: give the private run config "
                           "together with a mix (random, or a list of domains and genres)")
        return Plan(LEGACY, ())
    if config is None:
        domain_splits = {d.name: PUBLIC for d in all_domains()}
        genre_splits = {g.name: PUBLIC for g in all_genres()}
    else:
        for path, sha256 in config.pinned():  # a plugin whose bytes are not those the config gives is refused
            load_plugin(path, sha256=sha256)
        if config.plugins:
            _CONFIG_PLUGINS[config.commitment] = config.plugins
        for name in config.domains:
            _domain(name)
        for name in config.genres:
            _genre(name)
        domain_splits, genre_splits = dict(config.domains), dict(config.genres)
    pool = [g for g in all_genres() if g.name in genre_splits]  # registry order, whatever the config's order
    if text == RANDOM:
        options = [(d, g) for d in all_domains() if d.name in domain_splits for g in pool]
    else:
        entries = parse_mix(text)
        outside = sorted({e.domain for e in entries if e.domain not in domain_splits}
                         | {e.genre for e in entries if e.genre is not None and e.genre not in genre_splits})
        if outside:
            raise MixError(f"the mix {text!r} names {', '.join(outside)}, which the private run config does not list")
        options = choices(entries, pool)
    used_domains = dict.fromkeys(d.name for d, _ in options)
    used_genres = dict.fromkeys(g.name for _, g in options)
    inventory = Inventory({name: domain_splits[name] for name in used_domains},
                          {name: genre_splits[name] for name in used_genres})
    return Plan(text, tuple(options), inventory, config)


def single_choice(mix: str) -> tuple[Domain, Genre] | None:
    """The one (domain, genre) of every instance of a run with this mix (and no private config), or
    None for a legacy run. A mix that leaves a choice raises MixError: its instances are drawn one by
    one (Plan.draw)."""
    plan = plan_mix(mix)
    if plan.legacy:
        return None
    if len(plan.options) > 1:
        names = ", ".join(f"{domain.name}:{genre.name}" for domain, genre in plan.options)
        raise MixError(f"the mix {mix!r} leaves a choice ({names}): each instance is drawn from it (Plan.draw)")
    return plan.options[0]
