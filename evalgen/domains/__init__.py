"""Domains: the sources of records that evalgen builds tests from (see ``base`` for the interface).

Every public module of this package is one built-in domain and exports ``DOMAIN``, an instance of
``base.Domain``. Modules are discovered on the first lookup, in sorted order, so importing this
package never imports a domain module (domain modules import the rest of evalgen, which may in turn
import this package). Private plugins, Python files kept outside the public repository for held-out
domains, follow the same convention and are loaded by path with ``load_plugin``.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

from .base import Domain

DEFAULT_DOMAIN = "earthquakes"  # the 0.2.0 domain: the legacy path, and any lookup without a name


class Registry:
    """Domains by name: the built-in modules of a package in sorted order, then plugins in the order
    they were loaded. A name is never taken twice, so a plugin cannot shadow a built-in domain."""

    def __init__(self, package: str = __name__) -> None:
        self.package = package
        self._domains: dict[str, Domain] = {}
        self._discovered = False

    def get(self, name: str | None = None) -> Domain:
        self._discover()
        name = name or DEFAULT_DOMAIN
        if name not in self._domains:
            raise KeyError(f"unknown domain {name!r} (known: {', '.join(self._domains) or 'none'})")
        return self._domains[name]

    def all(self) -> list[Domain]:
        self._discover()
        return list(self._domains.values())

    def register(self, domain: Domain) -> Domain:
        if not isinstance(domain, Domain):
            raise TypeError(f"not a Domain: {domain!r}")
        known = self._domains.get(domain.name)
        if known is not None and known is not domain:
            raise ValueError(f"two domains are named {domain.name!r}")
        self._domains[domain.name] = domain
        return domain

    def load_plugin(self, path: str | Path) -> Domain:
        """Register the domain defined in a Python file outside the package. The private run config
        names these files (plan phase 4); loading the same file again returns the same domain."""
        path = Path(path).resolve()
        self._discover()  # built-ins first, so that a plugin cannot take a built-in name
        module_name = "evalgen_plugin_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        module = sys.modules.get(module_name) or _exec_file(module_name, path)
        return self.register(_domain_in(module))

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True  # set first: a domain module that looks domains up must not recurse
        package = importlib.import_module(self.package)
        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if info.name != "base" and not info.name.startswith("_"):
                self.register(_domain_in(importlib.import_module(f"{self.package}.{info.name}")))


def _exec_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses in the plugin look their module up here
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def _domain_in(module: ModuleType) -> Domain:
    domain = getattr(module, "DOMAIN", None)
    if not isinstance(domain, Domain):
        raise TypeError(f"{module.__name__} must define DOMAIN, an instance of evalgen.domains.base.Domain")
    return domain


REGISTRY = Registry()


def get_domain(name: str | None = None) -> Domain:
    """The domain called `name`; earthquakes when no name is given."""
    return REGISTRY.get(name)


def all_domains() -> list[Domain]:
    """Built-in domains in module order, then loaded plugins."""
    return REGISTRY.all()


def load_plugin(path: str | Path) -> Domain:
    """Load and register a private domain plugin (a Python file that defines ``DOMAIN``)."""
    return REGISTRY.load_plugin(path)
