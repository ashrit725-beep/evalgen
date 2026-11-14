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

What a mix allows is put in a canonical order before anything is drawn: the domains and the genres
built into evalgen in module order, then those of the private run config's plugins in the config's
order (and, without a config, any other entry the list names, by name). The order in which a list
names its entries therefore never changes a run, a list that allows one domain in one genre twice
("weather,weather:email": a bare domain is in every genre) is refused, and the manifest records the
mix in that canonical form (Plan.mix). Nothing that a process loaded for an earlier run changes what
a later one may draw or in which order.

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
splits, and a config may list only what evalgen or its own plugins define, so that the config and its
plugins regenerate the run in any process. Without one, every domain and genre is public and a run
draws from the built-in ones; a list may also name an entry that the caller registered itself, but
never one that a private run config loaded into the process (``Registry.is_private``), which is drawn
only under its config. A process runs each plugin file once: a plugin whose file has changed since the
process ran it is refused, and a run records the SHA-256 of the bytes that ran. The config is read
strictly: a file that is not JSON, or that gives a key twice in one object (JSON readers keep one of
the two, so such a file could show one config and verify as another), is refused, and so is one that
no commitment can hash: JSON nested too deeply to read, or text with a lone surrogate escape such as
"\\ud800", which no UTF-8 encodes (``read_config_data``). JSON is read as json reads bytes: UTF-8, -16 or
-32, with or without a byte-order mark, none of which changes the commitment.

Like the seed, the config is committed by hash: the manifest records ``sha256(canonical_json(config))``
(``config_commitment``), and the random nonce stops anyone from guessing a small config from its hash.
So a nonce must look drawn, not typed (``nonce_problem``): 32 hexadecimal digits or more, not a run of
one digit or a pattern. The one exception is the nonce of the published configs/private.example.json
(EXAMPLE_NONCE), which anyone can read: it is accepted, so that the demo of held-out sets works offline,
but generate warns loudly that its commitment can be guessed. Any other nonce goes through the check,
one that only says EXAMPLE included.

What the evaluator publishes, and when:

    before the evaluation  the two commitments that generate records in the run's manifest: the seed's
                           (seed_commitment) and the config's (private_config_commitment). Nothing
                           else: not the manifest, which holds the seed, nor the config or its plugins.
    after the evaluation   the config, its plugin files, and the seed with its nonce. Anyone can then
                           check the config against the commitment published before the evaluation
                           (``cli.py verify-config --commitment HEX CONFIG``: ``verify_private_config``
                           with `commitment`) and regenerate the run, which checks every plugin against
                           the SHA-256 its config gives before running it.

The check is against the published value: a run's manifest commits to the config that made the run,
whichever that was, so ``cli.py verify-config RUN_DIR CONFIG`` says only that a config and a run agree
(a run regenerated from a swapped config agrees with it). It prints the commitment it checked against,
which must be the one published before the evaluation.

A config names its plugins by path, so the commitment pins what a held-out domain or genre is only
through each plugin's SHA-256: with it, the commitment covers the plugin file's own bytes, which can no
longer change once the commitment is published; a plugin given by its path alone is committed to by its
path only, and could be edited after the commitment without anyone noticing. The SHA-256 covers that one
file and nothing it reads, so a plugin must be self-contained: no reads of other files (beside it or
elsewhere), of the environment, the clock or the network, and imports only from evalgen and Python's
standard library. Otherwise two runs under one commitment can hold out different domains or genres.

Regenerating a revealed run executes the evaluator's plugin code, in-process and with the rights of
whoever runs it (generate and demo run a config's plugins before any name is checked; run, score, report
and resolve run them only from a config that the run's commitment authenticates). The SHA-256 pins which
code runs, not that it is safe: a verifier reads the plugins first, or regenerates in a sandbox.
verify-config runs no plugin.

generate's console names every domain of the run, held-out ones included (each snapshot's line), and the
manifest and the default run folder name the mix: like the manifest, its console log is private until
the evaluation is over.
"""
from __future__ import annotations

import json
import os
import random
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from . import domains, genres, registry
from .domains import get_domain
from .domains.base import Domain
from .genres import get_genre
from .genres.base import Genre
from .seeds import SeedLog
from .util import SHA256_HEX, canonical_json, read_manifest, sha256_hex, shown

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
# The nonce of the published example config (configs/private.example.json), whose commitment anyone can
# guess: the only nonce that is exempt from the check, so that the example demos the held-out metrics.
EXAMPLE_NONCE = "EXAMPLE-ONLY-draw-your-own-with-secrets.token_hex"
NONCE_ADVICE = ("its nonce must be at least 32 random hexadecimal digits: draw one with "
                "python -c \"import secrets; print(secrets.token_hex(16))\"")
PLUGIN_KEYS = ("path", "sha256")  # a plugin given with the SHA-256 of its bytes
SHA256_DIGITS = 64
# Python 3.11 and later refuse to read an integer of more digits (sys.int_info.default_max_str_digits) and 3.9
# reads it: a config is read the same way by every Python, so a longer one is refused by all of them.
MAX_INT_DIGITS = 4300
COMMITMENT_TEXT = re.compile(r"[0-9a-fA-F]{64}")  # a commitment as a verifier may give it, in either case
SHOWN_MIX = 80  # characters of a mix that an error message quotes

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


def shown_mix(text: str) -> str:
    """A mix as an error message quotes it: whole, or its first SHOWN_MIX characters when it is longer (a list of
    every cell, or a mistake thousands of entries long, is not echoed whole)."""
    return repr(text) if len(text) <= SHOWN_MIX else f"{text[:SHOWN_MIX]!r}... ({len(text)} characters)"


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
            raise MixError(f"mix entry {shown_mix(part.strip())} is not a domain or domain:genre")
        _domain(domain)
        if genre:
            _genre(genre)
        entries.append(Entry(domain, genre or None))
    if len(set(entries)) != len(entries):
        raise MixError(f"the mix {shown_mix(text)} names an entry twice")
    return tuple(entries)


def entry_text(entry: Entry) -> str:
    """An entry as a mix writes it: "domain" or "domain:genre"."""
    return entry.domain if entry.genre is None else f"{entry.domain}:{entry.genre}"


def choices(entries: Sequence[Entry], genre_pool: Sequence[Genre] | None = None) -> list[tuple[Domain, Genre]]:
    """Every (domain, genre) an instance of the mix may be, in the order the entries name them. A bare
    domain may be in every genre of `genre_pool` (every built-in genre by default), in its order. Two
    entries that allow the same (domain, genre), such as weather and weather:email, raise MixError: the
    second adds nothing, so it could only be a way of writing one mix in two forms."""
    pool = genres.REGISTRY.builtins() if genre_pool is None else list(genre_pool)
    out: dict[tuple[str, str], tuple[Domain, Genre]] = {}
    allowed_by: dict[tuple[str, str], Entry] = {}
    for entry in entries:
        domain = _domain(entry.domain)
        for genre in [_genre(entry.genre)] if entry.genre else pool:
            key = (domain.name, genre.name)
            if key in out:
                first = allowed_by[key]
                why = " (a bare domain is in every genre)" if first.genre is None or entry.genre is None else ""
                raise MixError(f"the mix allows {domain.name}:{genre.name} twice, as {entry_text(first)} and as "
                               f"{entry_text(entry)}{why}: name it once")
            out[key], allowed_by[key] = (domain, genre), entry
    return list(out.values())


def _rank(order: Sequence[Any]) -> Callable[[str], tuple]:
    """A sort key on the names of domains (or genres): an entry of `order` by its place there, and any
    other (a mix without a private run config may name an entry its caller registered) after them, by
    name. So no order depends on the order in which this process registered its entries."""
    place = {entry.name: index for index, entry in enumerate(order)}
    return lambda name: (0, place[name], "") if name in place else (1, 0, name)


def _canonical_text(entries: Sequence[Entry], domain_rank: Callable[[str], tuple],
                    genre_rank: Callable[[str], tuple]) -> str:
    """A listed mix in its canonical form: its entries in canonical order (by domain, a bare domain
    before that domain's genres), as entry_text writes them."""
    def key(entry: Entry) -> tuple:
        return domain_rank(entry.domain), (-1, 0, "") if entry.genre is None else genre_rank(entry.genre)

    return ",".join(entry_text(entry) for entry in sorted(entries, key=key))


# ---------------------------------------------------------------------- the private run config


def is_example_nonce(nonce: Any) -> bool:
    """Whether a nonce is the published example's (configs/private.example.json, EXAMPLE_NONCE): such a
    config serves demos, and its commitment can be guessed. Exactly that nonce: one that merely contains
    "EXAMPLE" is checked like any other, so that no word in a nonce can let it skip the check."""
    return nonce == EXAMPLE_NONCE


def nonce_problem(nonce: Any) -> str | None:
    """What is wrong with a private run config's nonce, or None. A nonce must look drawn rather than
    typed: at least MIN_NONCE_DIGITS hexadecimal digits, at least MIN_NONCE_DISTINCT_DIGITS different
    ones, and not one shorter run of digits repeated ("abab..."). The published example's nonce is
    accepted (is_example_nonce; generate warns that its commitment can be guessed). No check can prove a
    nonce random: these refuse the guessable nonces that a config gets by accident, a word, a constant or
    a pattern."""
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


def _bounded_int(text: str) -> int:
    """An integer of a config's JSON, read alike by every Python: one of more than MAX_INT_DIGITS digits is
    refused (ValueError), as Python 3.11 and later refuse it and 3.9 does not."""
    digits = len(text.lstrip("-"))
    if digits > MAX_INT_DIGITS:
        raise ValueError(f"it holds an integer of {digits} digits, more than the {MAX_INT_DIGITS} that JSON readers take")
    return int(text)


def read_config_data(path: str | Path) -> Any:
    """The JSON value of a private run config file, read strictly, as generate and verify-config read it:
    FileNotFoundError for a path that is not a file (a folder, or nothing: "" names the current folder);
    MixError for a file that is not one JSON value (as json reads bytes: UTF-8, -16 or -32, with or without
    a byte-order mark, none of which changes the commitment), that is nested too deeply to read (json raises
    RecursionError for it), that gives a key twice in one object (it reads as two configs: the first value
    of the key or the last, so which one was committed to has no answer), or that no commitment can hash:
    text with a lone surrogate escape such as "\\ud800", which json reads and no UTF-8 encodes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"the private run config {path} is not a file")
    try:
        data = json.loads(path.read_bytes(), object_pairs_hook=_distinct_members, parse_int=_bounded_int)
    except DuplicateKey as exc:
        raise MixError(f"the private run config {path} is malformed: {exc}") from None
    except RecursionError:
        raise MixError(f"the private run config {path} is not JSON that evalgen reads: it is nested too deeply") from None
    except ValueError as exc:  # not JSON, not text in an encoding that JSON allows, or an integer too long
        raise MixError(f"the private run config {path} is not JSON ({exc})") from None
    try:
        config_commitment(data)
    except UnicodeEncodeError:
        raise MixError(f"the private run config {path} cannot be committed to: it holds a lone surrogate escape (such as "
                       "\\ud800), which is no character, so no UTF-8 and no commitment holds it") from None
    except (RecursionError, ValueError) as exc:  # nested past what json writes, though not past what it reads
        raise MixError(f"the private run config {path} cannot be committed to: {type(exc).__name__}: {exc}") from None
    return data


def read_private_config(path: str | Path) -> PrivateConfig:
    """Read and check a private run config: its JSON (``read_config_data``: a missing file raises
    FileNotFoundError, one that cannot be committed to MixError) and its shape (``config_problems``,
    MixError). Plugin paths are relative to the config's folder, and resolved with os.path.realpath,
    which no symbolic link makes raise on any Python (Path.resolve raises RuntimeError on a loop on 3.9)."""
    path = Path(path)
    data = read_config_data(path)
    problems = config_problems(data)
    if problems:
        raise MixError(f"the private run config {path} cannot be used: " + "; ".join(problems))
    entries = data.get("plugins", [])
    return PrivateConfig(
        path=path, data=data,
        domains={name: split for split in SPLITS for name in data["domains"][split]},
        genres={name: split for split in SPLITS for name in data["genres"][split]},
        plugins=tuple(Path(os.path.realpath(path.parent / (e if isinstance(e, str) else e["path"]))) for e in entries),
        plugin_sha256=tuple(None if isinstance(e, str) else e["sha256"].lower() for e in entries),
    )


def config_commitment(config: Mapping[str, Any]) -> str:
    """The commitment to a private run config: the SHA-256 of its canonical JSON (sorted keys, no
    whitespace), so that only its content counts, not how the file is laid out."""
    return sha256_hex(canonical_json(config))


def read_commitment(text: str) -> str:
    """A commitment as a verifier gives it (the private_config_commitment published before the
    evaluation): 64 hexadecimal digits, in either case, blanks around them ignored; returned as evalgen
    writes it, in lower case. Anything else raises MixError: it is no commitment, so no config can match it."""
    digits = text.strip() if isinstance(text, str) else ""
    if not COMMITMENT_TEXT.fullmatch(digits):
        raise MixError(f"{shown(text)} is not a commitment: a commitment is 64 hexadecimal digits, as generate prints it")
    return digits.lower()


def run_commitment(run_dir: str | Path) -> str:
    """The commitment to its private run config that the run in `run_dir` records (its manifest's
    private_config_commitment, which util.read_manifest checks is 64 hexadecimal digits). A run generated
    without a config raises MixError; a `run_dir` that is not a run raises FileNotFoundError, or RunDirError
    for one that evalgen will not read (util.read_manifest)."""
    recorded = read_manifest(run_dir).get("private_config_commitment")
    if recorded is None:
        raise MixError(f"{run_dir} was generated without a private run config, so it committed to none")
    return recorded


def revealed_commitment(path: str | Path) -> str:
    """The commitment of the private run config at `path`, read as generate reads it (``read_config_data``):
    what a verifier compares with the commitment published before the evaluation."""
    return config_commitment(read_config_data(path))


def verify_config(config: Any, commitment: Any) -> bool:
    """Whether a revealed config is the one a commitment was made to. A commitment that is not 64
    hexadecimal digits (a received manifest may hold anything) matches no config, and so does a config
    that no commitment can hash: neither raises."""
    if not isinstance(commitment, str) or not COMMITMENT_TEXT.fullmatch(commitment):
        return False
    try:
        actual = config_commitment(config)
    except (RecursionError, TypeError, ValueError):  # a lone surrogate (UnicodeEncodeError), or what is no JSON
        return False
    return secrets.compare_digest(actual, commitment.lower())


def verify_private_config(path: str | Path, run_dir: str | Path | None = None, *,
                          commitment: str | None = None) -> bool:
    """Whether the private run config at `path` is the one committed to: as `commitment` (the value
    published before the evaluation, ``read_commitment``), which is the check a verifier needs, or as the
    run in `run_dir` records it (``run_commitment``), which says only that the config and that run agree. A
    run regenerated from a config agrees with it, whatever was published, so the run's commitment must
    itself be the published one. Give one of the two: MixError otherwise, for a commitment that is not
    one, for a run generated without a config, and for a file that cannot be read as one config
    (``read_config_data``). The plugins' bytes are checked when a run loads them (see the module
    docstring)."""
    if (run_dir is None) == (commitment is None):
        raise MixError("check a config against the commitment published before the evaluation (commitment), or "
                       "against a run's manifest (run_dir): give one of the two")
    expected = read_commitment(commitment) if commitment is not None else run_commitment(run_dir)
    return secrets.compare_digest(revealed_commitment(path), expected)


# ------------------------------------------------------------------------------------ plugins

Entrant = Union[Domain, Genre]  # Union, not "X | Y": this alias is evaluated at import (Python 3.9)

# The plugin files of each private run config whose plugins this process has loaded (plan_mix, or
# load_run_plugins with the config), by the config's commitment: what a later step of the same process
# registers again without the config. A run names its plugins as its config does, often relative to the
# config's folder, so the run alone does not say where they are on this machine.
_CONFIG_PLUGINS: dict[str, tuple[Path, ...]] = {}


def load_plugin(path: str | Path, sha256: str | None = None, *, private: bool = False) -> list[Entrant]:
    """Register what a plugin file defines: its domain (``DOMAIN``), its genre (``GENRE``) or both,
    each in its registry. With `sha256` (what a private run config commits the file to), a file whose
    bytes have another SHA-256 is refused before any of it runs, and what runs is exactly the bytes
    that were checked. A file that this process has run already is not run again, and its bytes must
    still be those that ran: a process runs one version of a plugin (registry.load_module). With
    `private` (a private run config's plugin), each entry is marked private as soon as it is registered
    (_register_module). Raises MixError for a file that is gone (or a link that loops: the path is
    resolved with os.path.realpath, which raises on no Python), refused, changed since this process ran
    it, whose own code fails or ends the process (sys.exit) as it runs, that defines neither entry, or
    that defines one that its registry refuses (a name a built-in or another plugin already has, or a
    domain that does not hold together, whatever Domain.validate raises on it)."""
    path = Path(os.path.realpath(path))
    if not path.is_file():
        raise MixError(f"the plugin {path} does not exist")
    source = path.read_bytes()
    if sha256 is not None and sha256_hex(source) != sha256.lower():
        raise MixError(f"the plugin {path} is not the file its private run config commits to: its SHA-256 is "
                       f"{sha256_hex(source)}, and the config gives {sha256.lower()}")
    try:
        module = registry.load_module(path, source)
    except registry.PluginChanged as exc:
        raise MixError(str(exc)) from None
    except SystemExit as exc:  # a plugin defines entries; it never ends the process that loads it
        raise MixError(f"the plugin {path} called sys.exit({exc.code!r}) as it ran: a plugin defines DOMAIN or GENRE, "
                       "and never ends the command that loads it") from None
    except Exception as exc:  # the plugin's own code failed: a syntax error, a missing import, what it raised
        raise MixError(f"the plugin {path} failed to run: {type(exc).__name__}: {exc}") from exc
    return _register_module(path, module, private=private)


def _register_module(path: Path, module: Any, *, private: bool = False) -> list[Entrant]:
    """Register what an executed plugin module defines, each entry in its registry (load_plugin). With
    `private`, each entry is marked private (_mark_private) as soon as it is registered: a config whose
    loading fails later (a genre of this file, or another plugin, refused) has run this file all the
    same, and its entries stay registered, so they must never be drawable by a run without the config."""
    loaded: list[Entrant] = []
    try:
        if hasattr(module, "DOMAIN"):
            loaded.append(domains.load_plugin(path))
            if private:
                _mark_private(loaded[-1:])
        if hasattr(module, "GENRE"):
            loaded.append(genres.load_plugin(path))
            if private:
                _mark_private(loaded[-1:])
    except (TypeError, ValueError) as exc:  # the registry refuses the entry
        raise MixError(f"the plugin {path} cannot be registered: {exc}") from None
    except Exception as exc:  # a domain whose attributes are not what Domain.validate reads (a field that is a number)
        raise MixError(f"the plugin {path} cannot be registered: {type(exc).__name__}: {exc}") from None
    if not loaded:
        raise MixError(f"the plugin {path} defines neither DOMAIN nor GENRE")
    return loaded


def plugin_file(entry: Entrant) -> Path | None:
    """The plugin file that defines `entry`, a domain or a genre (registry.Registry.origin, resolved); None for
    a built-in entry, and for one that its caller registered directly."""
    entries = domains.REGISTRY if isinstance(entry, Domain) else genres.REGISTRY
    return None if entries.is_builtin(entry.name) else entries.origin(entry.name)


def _mark_private(entries: Sequence[Entrant]) -> None:
    """Mark entries that a private run config's plugins define (Registry.mark_private): no run without
    that config draws them."""
    for entry in entries:
        (domains.REGISTRY if isinstance(entry, Domain) else genres.REGISTRY).mark_private(entry)


def _load_config_plugins(config: PrivateConfig) -> tuple[list[Domain], list[Genre]]:
    """Load the plugins of a private run config, each checked against the SHA-256 the config gives it
    (load_plugin); mark what they define as private (_mark_private, entry by entry as each registers, so
    that a config refused halfway leaves nothing drawable without it) and remember them for later steps
    of this process (_CONFIG_PLUGINS). Returns the domains and the genres they define, in the config's
    plugin order."""
    found: list[Entrant] = []
    for path, sha256 in config.pinned():  # a plugin whose bytes are not those the config gives is refused
        found += load_plugin(path, sha256=sha256, private=True)
    if config.plugins:
        _CONFIG_PLUGINS[config.commitment] = config.plugins
    plugin_domains = {entry.name: entry for entry in found if isinstance(entry, Domain)}
    plugin_genres = {entry.name: entry for entry in found if isinstance(entry, Genre)}
    return list(plugin_domains.values()), list(plugin_genres.values())


def plugin_record(config: PrivateConfig) -> list[dict[str, str]]:
    """The plugins of a config as a manifest records them, in the config's order: each file's path as
    the config names it (relative to the config, unless the config gives an absolute path), so that a
    run names no folder of the evaluator's machine that the revealed config does not, and the SHA-256
    of the bytes this process ran for it (registry.executed_sha256): the code that made the run, even if
    the file has been edited since. A plugin this process has not run is recorded by its file's bytes."""
    named = [entry if isinstance(entry, str) else entry["path"] for entry in config.data.get("plugins", [])]
    return [{"path": name, "sha256": registry.executed_sha256(path) or sha256_hex(path.read_bytes())}
            for name, path in zip(named, config.plugins)]


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
    `log`; and one that this process ran already, but whose file has changed since, is refused, since a
    process runs one version of a plugin (load_plugin). Without a config, only plugins that this process
    has run already, as the plugins of the config that the run committed to (it generated the run, or
    loaded them for an earlier step), are registered again (_CONFIG_PLUGINS), as they ran: none of their
    code runs again, and a plugin counts as changed only if the code that ran differs from the run's
    record. Any other plugin, and a config the run did not commit to, raise MixError, as does a plugin
    that is gone, and a run that draws on a domain or genre that neither evalgen nor these plugins
    define (it was registered in the process that generated the run, by its caller or by a plugin that
    process loaded)."""
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
        if private_config is not None:
            load_plugin(path, sha256=sha256, private=True)
        else:  # what this process ran for the run's config, as it ran (registry.executed checked it ran)
            _register_module(path, registry.load_module(path), private=True)
        if sha256 is None and entry.get("sha256") and registry.executed_sha256(path) != entry["sha256"]:
            changed.append(str(entry["path"]))
            if log is not None:
                log(f"warning: the plugin {path} has changed since the run was generated")
    if private_config is not None and plugins:
        _CONFIG_PLUGINS[recorded] = tuple(path for path, _ in plugins)
    unknown = _unknown_entries(manifest)
    if unknown:
        them, they = ("it", "it was") if len(unknown) == 1 else ("them", "they were")
        raise MixError(f"the run draws on {', '.join(unknown)}, which this process does not know: neither evalgen nor a "
                       f"plugin of the run's private run config defines {them}, so {they} registered in the process "
                       "that generated the run (by its caller or by a plugin it loaded), and a process that reads the run "
                       f"must register {them} too")
    return changed


def _unknown_entries(manifest: Mapping[str, Any]) -> list[str]:
    """The domains and genres in a run's inventory that no registry of this process knows."""
    inventory = manifest.get("inventory") or {}
    unknown = []
    for kind, lookup in (("domains", get_domain), ("genres", get_genre)):
        for name in (name for split in SPLITS for name in inventory.get(kind, {}).get(split, ())):
            try:
                lookup(name)
            except KeyError:
                unknown.append(name)
    return unknown


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
    """A run's mix, read against the registries and the private run config: the mix in its canonical
    form, every (domain, genre) an instance may be drawn as in canonical order (none on the legacy path;
    see plan_mix), the run's inventory, the config, and the genres a reader of the run's prompts may
    meet, which the generator's self-check sniffs among (genres.detect_genre): the built-in genres in
    module order, then those of the config's plugins in the config's plugin order, then any other genre
    the mix names (one its caller registered). They are what a new process that loads the config knows,
    whatever this process has registered besides."""

    mix: str  # canonical: "legacy", "random", or the entries of a list in canonical order
    options: tuple[tuple[Domain, Genre], ...]
    inventory: Inventory | None = None  # None on the legacy path
    config: PrivateConfig | None = None
    readers: tuple[Genre, ...] = ()  # none on the legacy path, whose prompts are read as 0.2.0 read them

    @property
    def legacy(self) -> bool:
        return not self.options

    @property
    def domains(self) -> list[Domain]:
        """The domains instances may be about, in canonical order: however the mix orders its entries."""
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


def _listed(listed: Mapping[str, str], builtins: Sequence[Any], plugin_entries: Sequence[Any],
            lookup: Callable[[str], Any], noun: str) -> list[Any]:
    """The domains (or genres) a private run config lists, in canonical order: the built-in ones in
    module order, then those its plugins define in the config's plugin order. A name that no registry
    knows raises UnknownName. One that is registered but neither built in nor defined by the config's
    plugins (its caller registered it, or another config's plugin did) raises MixError: the config and
    its plugins could not regenerate the run in another process."""
    for name in listed:
        lookup(name)
    order = [entry for entry in builtins if entry.name in listed]
    order += [entry for entry in plugin_entries if entry.name in listed]
    stray = [name for name in listed if name not in {entry.name for entry in order}]
    if stray:
        raise MixError(f"the private run config lists the {noun}{'s' if len(stray) > 1 else ''} {', '.join(stray)}, "
                       "which neither evalgen nor any of the config's plugins defines: a config's plugins must define "
                       "everything it lists that is not built in, so that they regenerate the run in any process")
    return order


def plan_mix(mix: str, private_config: str | Path | None = None) -> Plan:
    """Read a run's mix and its private run config (path) before anything is generated: load the
    config's plugins, check every name, and list what instances may be drawn as, in canonical order
    (the module docstring). Without a config, every domain and genre is public: random draws from the
    built-in ones, a bare domain's renderings are in the built-in genres, and a list may also name an
    entry that its caller registered, but none that a private run config loaded into this process. With
    a config, which may list only what evalgen or its own plugins define, the mix may only name what the
    config lists, and random draws from all of that. A legacy run takes no config: it is the 0.2.0 run,
    with nothing held out. Raises MixError (or UnknownName) for anything a run could not follow."""
    text = mix.strip()
    config = None if private_config is None else read_private_config(private_config)
    if text == LEGACY:
        if config is not None:
            raise MixError("a legacy run is the 0.2.0 run, with nothing held out: give the private run config "
                           "together with a mix (random, or a list of domains and genres)")
        return Plan(LEGACY, ())
    if config is None:  # the same in every process, whatever it has loaded
        domain_order, genre_order = domains.REGISTRY.builtins(), genres.REGISTRY.builtins()
        readers = list(genre_order)
    else:
        plugin_domains, plugin_genres = _load_config_plugins(config)
        domain_order = _listed(config.domains, domains.REGISTRY.builtins(), plugin_domains, _domain, "domain")
        genre_order = _listed(config.genres, genres.REGISTRY.builtins(), plugin_genres, _genre, "genre")
        readers = genres.REGISTRY.builtins() + plugin_genres
    domain_rank, genre_rank = _rank(domain_order), _rank(genre_order)
    if text == RANDOM:
        options = [(d, g) for d in domain_order for g in genre_order]
        canonical = RANDOM
    else:
        entries = parse_mix(text)
        if config is None:
            private = sorted({e.domain for e in entries if domains.REGISTRY.is_private(e.domain)}
                             | {e.genre for e in entries if e.genre is not None and genres.REGISTRY.is_private(e.genre)})
            if private:
                raise MixError(f"the mix {shown_mix(text)} names {', '.join(private)}, which a private run config loaded "
                               "into this process: a run draws from a config's plugins only under that config "
                               "(--private-config)")
        else:
            outside = sorted({e.domain for e in entries if e.domain not in config.domains}
                             | {e.genre for e in entries if e.genre is not None and e.genre not in config.genres})
            if outside:
                raise MixError(f"the mix {shown_mix(text)} names {', '.join(outside)}, which the private run config does "
                               "not list")
        options = sorted(choices(entries, genre_order), key=lambda o: (domain_rank(o[0].name), genre_rank(o[1].name)))
        canonical = _canonical_text(entries, domain_rank, genre_rank)
    used_domains = dict.fromkeys(d.name for d, _ in options)
    used_genres = dict.fromkeys(g.name for _, g in options)
    inventory = Inventory({name: PUBLIC if config is None else config.domains[name] for name in used_domains},
                          {name: PUBLIC if config is None else config.genres[name] for name in used_genres})
    known = {genre.name for genre in readers}
    for _, genre in options:  # a genre that the mix names and its caller registered
        if genre.name not in known:
            known.add(genre.name)
            readers.append(genre)
    return Plan(canonical, tuple(options), inventory, config, tuple(readers))


def single_choice(mix: str) -> tuple[Domain, Genre] | None:
    """The one (domain, genre) of every instance of a run with this mix (and no private config), or
    None for a legacy run. A mix that leaves a choice raises MixError: its instances are drawn one by
    one (Plan.draw). Only the tests call it: a run plans its mix with plan_mix and draws each instance
    with Plan.draw."""
    plan = plan_mix(mix)
    if plan.legacy:
        return None
    if len(plan.options) > 1:
        names = ", ".join(f"{domain.name}:{genre.name}" for domain, genre in plan.options)
        raise MixError(f"the mix {shown_mix(mix)} leaves a choice ({names}): each instance is drawn from it (Plan.draw)")
    return plan.options[0]
