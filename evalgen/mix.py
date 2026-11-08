"""Which domains and genres a run's instances come from: GenerateConfig.mix.

    legacy        0.2.0, the default: earthquakes in the classic genre, on the 0.2.0 seed paths
    toy:classic   every instance about the toy domain, and every rendering in the classic genre
    toy           the toy domain, in any genre
    a,b:news      a list: each entry is a domain, alone or with ":" and a genre

Any mix but legacy is recorded in the manifest, and puts the run on seed paths of its own (its
snapshot, its twin templates, its instances' surfaces), so the legacy draws never move. Each instance
then records its domain and each rendering its genre.

A mix is read in two steps: ``parse_mix`` checks the names against the domain and genre registries,
and ``choices`` lists every (domain, genre) pair an instance may be drawn as. Drawing that pair per
instance ("random", or a list that leaves a choice) comes with plan phase 4, together with the private
run config; until then ``single_choice`` gives the one pair a run can have.
"""
from __future__ import annotations

from dataclasses import dataclass

from .domains import get_domain
from .domains.base import Domain
from .genres import all_genres, get_genre
from .genres.base import Genre

LEGACY = "legacy"
RANDOM = "random"


@dataclass(frozen=True)
class Entry:
    """One entry of a mix: a domain, and the genre its instances are rendered in (None: any genre)."""

    domain: str
    genre: str | None = None


def parse_mix(text: str) -> tuple[Entry, ...]:
    """The entries of a mix, with every name checked (an unknown one raises KeyError); () for legacy."""
    text = text.strip()
    if text == LEGACY:
        return ()
    if text == RANDOM:
        raise NotImplementedError("the random mix comes with the mix draws of plan phase 4")
    entries = []
    for part in text.split(","):
        domain, colon, genre = (piece.strip() for piece in part.partition(":"))
        if not domain or (colon and not genre):
            raise ValueError(f"mix entry {part.strip()!r} is not a domain or domain:genre")
        get_domain(domain)
        if genre:
            get_genre(genre)
        entries.append(Entry(domain, genre or None))
    if len(set(entries)) != len(entries):
        raise ValueError(f"the mix {text!r} names an entry twice")
    return tuple(entries)


def choices(entries: tuple[Entry, ...]) -> list[tuple[Domain, Genre]]:
    """Every (domain, genre) an instance of the mix may be, in the order the entries name them (a bare
    domain: in every genre, in registry order)."""
    out: list[tuple[Domain, Genre]] = []
    for entry in entries:
        domain = get_domain(entry.domain)
        for genre in [get_genre(entry.genre)] if entry.genre else all_genres():
            if (domain, genre) not in out:
                out.append((domain, genre))
    return out


def single_choice(text: str) -> tuple[Domain, Genre] | None:
    """The domain and genre of every instance of a run with this mix, or None for a legacy run. A mix
    that leaves a choice needs the per-instance draws of plan phase 4 (NotImplementedError)."""
    entries = parse_mix(text)
    if not entries:
        return None
    options = choices(entries)
    if len(options) > 1:
        names = ", ".join(f"{domain.name}:{genre.name}" for domain, genre in options)
        raise NotImplementedError(f"the mix {text!r} leaves a choice ({names}); drawing each instance's domain and "
                                  "genre comes with plan phase 4")
    return options[0]
