"""Discovery-based registries of named entries: the domains (evalgen.domains) and the genres
(evalgen.genres).

Every public module of a registry's package defines one entry under a fixed name (``DOMAIN``,
``GENRE``). Modules are discovered on the first lookup, in sorted order, so importing the package
never imports an entry module (entry modules import the rest of evalgen, which may in turn import
the package). Private plugins, Python files kept outside the public repository for held-out
entries, follow the same convention and are loaded by path with ``load_plugin``.

A registry is shared by the whole process, so it remembers where each entry came from: the package's
own entries (``builtins``) and the entries that a private run config's plugins define (``private``,
marked by evalgen.mix). What a run may draw is decided from those (evalgen.mix): ``random`` and a bare
domain draw from the built-in entries, and with a private run config from the built-in entries and
those of the config's own plugins, never from whatever else the process has registered. A list may
also name an entry that its caller registered itself, but a run without a config never draws an entry
that a private run config loaded. So what a run draws does not depend on what an earlier step of the
same process loaded.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import pkgutil
import sys
import weakref
from pathlib import Path
from types import ModuleType
from typing import Generic, Protocol, TypeVar


class Named(Protocol):
    name: str


T = TypeVar("T", bound=Named)


class Registry(Generic[T]):
    """Entries by name: the built-in modules of a package in sorted order, then plugins in the order
    they were loaded. A name is never taken twice, so a plugin cannot shadow a built-in entry. The
    built-in entries (``builtins``) and those of private run configs (``is_private``) are told apart
    from the rest."""

    def __init__(self, package: str, kind: type[T], *, attribute: str, default: str) -> None:
        self.package = package
        self.kind = kind  # what every entry is an instance of: Domain, Genre
        self.attribute = attribute  # the name each module defines its entry under: "DOMAIN", "GENRE"
        self.default = default  # the entry a lookup without a name gets
        self.noun = kind.__name__.lower()  # "domain", "genre"
        self._entries: dict[str, T] = {}
        self._builtin: set[str] = set()  # the names of the package's own entries
        self._private: set[str] = set()  # the names of entries that a private run config's plugins define
        self._discovered = False

    def get(self, name: str | None = None) -> T:
        self._discover()
        name = name or self.default
        if name not in self._entries:
            raise KeyError(f"unknown {self.noun} {name!r} (known: {', '.join(self._entries) or 'none'})")
        return self._entries[name]

    def all(self) -> list[T]:
        """Every registered entry: the built-in ones in module order, then the others in the order they
        were registered, which depends on what this process has loaded (see ``builtins``)."""
        self._discover()
        return list(self._entries.values())

    def builtins(self) -> list[T]:
        """The package's own entries, in module order: the same in every process, whatever plugins it
        has loaded. A run without a private run config draws from these."""
        self._discover()
        return [entry for name, entry in self._entries.items() if name in self._builtin]

    def is_builtin(self, name: str) -> bool:
        """Whether `name` is one of the package's own entries, whose code is public, rather than a
        plugin's (or no entry at all)."""
        self._discover()
        return name in self._builtin

    def mark_private(self, entry: T) -> None:
        """Record that `entry`, a registered plugin entry, is defined by a plugin of a private run config
        (evalgen.mix): held-out code of the evaluator's, which a run may draw only under that config."""
        if self._entries.get(entry.name) is not entry or entry.name in self._builtin:
            raise ValueError(f"{entry.name!r} is not a plugin {self.noun} of this registry")
        self._private.add(entry.name)

    def is_private(self, name: str) -> bool:
        """Whether `name` is an entry that a private run config's plugins define (``mark_private``)."""
        return name in self._private

    def register(self, entry: T) -> T:
        if not isinstance(entry, self.kind):
            raise TypeError(f"not a {self.kind.__name__}: {entry!r}")
        known = self._entries.get(entry.name)
        if known is not None and known is not entry:
            raise ValueError(f"two {self.noun}s are named {entry.name!r}")
        self._admit(entry)
        self._entries[entry.name] = entry
        return entry

    def _admit(self, entry: T) -> None:
        """Called before an entry is added: a registry of a particular kind checks it here (raising
        refuses it) and sets up what comes with it."""

    def load_plugin(self, path: str | Path) -> T:
        """Register the entry defined in a Python file outside the package. The private run config
        names these files (evalgen.mix); loading the same file again returns the same entry."""
        path = Path(path).resolve()
        self._discover()  # built-ins first, so that a plugin cannot take a built-in name
        return self.register(self._entry_in(load_module(path)))

    def _discover(self) -> None:
        """Register the package's entry modules in sorted order, all or none: when one fails to import
        or is refused, the registry is left as it was and the next lookup tries again (and fails the
        same way), so a lookup never quietly misses the entries after a broken module."""
        if self._discovered:
            return
        self._discovered = True  # set first, as a recursion guard: an entry module may look entries up
        before = dict(self._entries)
        try:
            package = importlib.import_module(self.package)
            for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
                if info.name != "base" and not info.name.startswith("_"):
                    entry = self.register(self._entry_in(importlib.import_module(f"{self.package}.{info.name}")))
                    self._builtin.add(entry.name)
        except BaseException:
            self._entries = before
            self._builtin.clear()
            self._discovered = False
            raise

    def _entry_in(self, module: ModuleType) -> T:
        entry = getattr(module, self.attribute, None)
        if not isinstance(entry, self.kind):
            raise TypeError(f"{module.__name__} must define {self.attribute}, an instance of "
                            f"{self.kind.__module__}.{self.kind.__qualname__}")
        return entry


class PluginChanged(ImportError):
    """A plugin file whose bytes are not those that this process executed for it. One process runs one
    version of a plugin (its entries are registered under their names), so a new version needs a new
    process."""


# The SHA-256 of the bytes each plugin module was executed from (load_module), kept with the module
# object, so that what a run records of a plugin is the code that ran, not the file as it is later.
_EXECUTED_SHA256: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _module_name(path: Path) -> str:
    """The name a plugin file's module is registered under in sys.modules: one per resolved path."""
    return "evalgen_plugin_" + hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def load_module(path: str | Path, source: bytes | None = None) -> ModuleType:
    """The module of a plugin file, executed once per process, so that a file defining both a domain
    and a genre is loaded once for both registries. `source` is the file's bytes as the caller read
    (and checked) them: what runs is exactly those bytes, never a later state of the file or a
    bytecode cache beside it. Without it, the file is read once, here. A file that this process has
    executed already is not run again: with `source`, those bytes must be the ones it executed, or
    PluginChanged is raised (a module that ran other bytes is never passed off as these); without
    `source`, the module is returned as it ran."""
    path = Path(path).resolve()
    module_name = _module_name(path)
    module = sys.modules.get(module_name)
    if module is None:
        return _exec_file(module_name, path, source)
    if source is not None and hashlib.sha256(source).hexdigest() != _EXECUTED_SHA256.get(module):
        raise PluginChanged(f"the plugin {path} has changed since this process ran it: a process runs one version "
                            "of a plugin, so load the new one in a new process")
    return module


def executed(path: str | Path) -> bool:
    """Whether this process has already executed the plugin file at `path` (load_module): loading it
    again registers its entries without running any of its code."""
    return _module_name(Path(path)) in sys.modules


def executed_sha256(path: str | Path) -> str | None:
    """The SHA-256 of the bytes this process executed for the plugin file at `path` (load_module), or
    None if it has not executed it: what a run made by this process ran, whatever the file holds now."""
    module = sys.modules.get(_module_name(Path(path)))
    return None if module is None else _EXECUTED_SHA256.get(module)


def _exec_file(module_name: str, path: Path, source: bytes | None) -> ModuleType:
    """Execute a plugin's source as a module of its own, and keep the SHA-256 of the bytes it ran. The
    bytes are compiled here, so no bytecode cache is read or written next to a private plugin."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a plugin from {path}")
    source = path.read_bytes() if source is None else source
    code = compile(source, str(path), "exec", dont_inherit=True)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses in the plugin look their module up here
    try:
        exec(code, module.__dict__)
    except BaseException:
        del sys.modules[module_name]
        raise
    _EXECUTED_SHA256[module] = hashlib.sha256(source).hexdigest()
    return module
