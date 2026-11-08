"""Domains: the sources of records that evalgen builds tests from (see ``base`` for the interface).

Every public module of this package is one built-in domain and exports ``DOMAIN``, an instance of
``base.Domain``. Modules are discovered on the first lookup, in sorted order, so importing this
package never imports a domain module (domain modules import the rest of evalgen, which may in turn
import this package). Private plugins, Python files kept outside the public repository for held-out
domains, follow the same convention and are loaded by path with ``load_plugin`` (see
evalgen.registry, which the genres share). A domain is validated when it is registered, and its
answer types are registered with the scorer at the same time.
"""
from __future__ import annotations

from pathlib import Path

from .. import registry
from ..scoring import register_answer_type
from .base import Domain

DEFAULT_DOMAIN = "earthquakes"  # the 0.2.0 domain: the legacy path, and any lookup without a name


class Registry(registry.Registry[Domain]):
    """Domains by name: the built-in modules of a package in sorted order, then plugins in the order
    they were loaded. A name is never taken twice, so a plugin cannot shadow a built-in domain."""

    def __init__(self, package: str = __name__) -> None:
        super().__init__(package, Domain, attribute="DOMAIN", default=DEFAULT_DOMAIN)

    def _admit(self, entry: Domain) -> None:
        """Validate a domain (Domain.validate) before it is added, and register its answer types, so
        that the scorer reads every answer about its records. (They stay registered if a later module
        of the same discovery fails: registering them again on the next attempt is harmless.)"""
        entry.validate()
        for answer_type in entry.answer_types():  # validate() checked that none conflicts
            register_answer_type(answer_type)


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
