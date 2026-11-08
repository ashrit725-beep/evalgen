"""Genres: the kinds of document a digest can be written as (see ``base`` for the interface).

Every public module of this package is one built-in genre and exports ``GENRE``, an instance of
``base.Genre``; ``classic`` is the 0.2.0 digest. Modules are discovered on the first lookup, in
sorted order, and private plugins (held-out genres kept outside the public repository) are loaded by
path with ``load_plugin``, exactly as domains are (see evalgen.registry).

A reader of a prompt picks the parser of the genre that wrote it with ``detect_genre``: the first
other genre whose ``sniff`` recognises the digest, or else classic, so a legacy digest is always read
as 0.2.0 read it.
"""
from __future__ import annotations

from pathlib import Path

from .. import registry
from .base import Genre

DEFAULT_GENRE = "classic"  # the 0.2.0 genre: the legacy path, and the fallback of detect_genre


class Registry(registry.Registry[Genre]):
    """Genres by name: the built-in modules of a package in sorted order, then plugins in the order
    they were loaded. A name is never taken twice, so a plugin cannot shadow a built-in genre."""

    def __init__(self, package: str = __name__) -> None:
        super().__init__(package, Genre, attribute="GENRE", default=DEFAULT_GENRE)


REGISTRY = Registry()


def get_genre(name: str | None = None) -> Genre:
    """The genre called `name`; classic when no name is given."""
    return REGISTRY.get(name)


def all_genres() -> list[Genre]:
    """Built-in genres in module order, then loaded plugins."""
    return REGISTRY.all()


def load_plugin(path: str | Path) -> Genre:
    """Load and register a private genre plugin (a Python file that defines ``GENRE``)."""
    return REGISTRY.load_plugin(path)


def detect_genre(digest_text: str) -> Genre:
    """The genre that wrote `digest_text`: the first genre other than classic whose sniff recognises
    it, or classic when none does."""
    for genre in all_genres():
        if genre.name != DEFAULT_GENRE and genre.sniff(digest_text):
            return genre
    return get_genre(DEFAULT_GENRE)
