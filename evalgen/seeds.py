"""Deterministic, logged seeding.

Every random choice in evalgen flows from ONE master seed. A component asks for its own RNG by a
readable path, e.g. ``seedlog.rng("instance", 3, "traps")``. The child seed is derived with
SHA-256 (not Python's salted ``hash()``), so it is identical across runs, machines and Python
versions, and every derived seed is recorded in the run manifest.

Threat-model note: the method is public, so the *seed* is the only secret, and it must be one that
nobody can guess. Draw a fresh seed at evaluation time (``fresh_seed``, 64 random bits: what
``cli.py generate`` does when --seed is left out) and publish ``seed_commitment(seed, nonce)``
beforehand if you want. The random nonce makes the commitment impossible to brute-force, even for a
small seed such as 42, but it protects only the commitment: the instances are a function of the seed
and the public code, so anyone can generate those of a small seed in advance, and any one prompt of a
run (its compile time and record IDs) gives a small seed back by brute force in seconds. Small seeds
are for demos, tests and reproductions (``guessable_seed``). Reveal the seed and nonce afterwards so
anyone can verify the commitment and regenerate the exact instances.
"""
from __future__ import annotations

import hashlib
import random
import secrets

# A seed of fewer bits than this can be enumerated: a fresh 64-bit seed has fewer once in 131,072 draws.
GUESSABLE_SEED_BITS = 48


def derive_seed(master: int, *path: object) -> int:
    key = "/".join([str(master)] + [str(p) for p in path])
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


class SeedLog:
    """Hands out child RNGs and remembers every seed it derived."""

    def __init__(self, master: int) -> None:
        self.master = int(master)
        self.seeds: dict[str, int] = {}

    def seed(self, *path: object) -> int:
        value = derive_seed(self.master, *path)
        self.seeds["/".join(str(p) for p in path)] = value
        return value

    def rng(self, *path: object) -> random.Random:
        return random.Random(self.seed(*path))


def seed_commitment(seed: int, nonce: str) -> str:
    """A hash you can publish before an evaluation and verify after revealing the seed and nonce."""
    return hashlib.sha256(f"evalgen-seed:{seed}:{nonce}".encode("utf-8")).hexdigest()


def verify_commitment(seed: int, nonce: str, commitment: str) -> bool:
    return secrets.compare_digest(seed_commitment(seed, nonce), commitment)


def fresh_seed() -> int:
    """A new unpredictable 64-bit master seed (used when --seed is not given; it is logged)."""
    return secrets.randbits(64)


def guessable_seed(seed: int) -> bool:
    """Whether a master seed is small enough to enumerate (fewer than GUESSABLE_SEED_BITS bits, either
    sign): fine for a demo or a reproduction, never the secret of an evaluation, since one prompt of the
    run recovers it by brute force."""
    return int(seed).bit_length() < GUESSABLE_SEED_BITS


def fresh_nonce() -> str:
    return secrets.token_hex(16)
