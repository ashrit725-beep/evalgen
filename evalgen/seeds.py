"""Deterministic, logged seeding.

Every random choice in evalgen flows from ONE master seed. A component asks for its own RNG by a
readable path, e.g. ``seedlog.rng("instance", 3, "traps")``. The child seed is derived with
SHA-256 (not Python's salted ``hash()``), so it is identical across runs, machines and Python
versions, and every derived seed is recorded in the run manifest.

Threat-model note: the method is public, so the *seed* is the only secret. Draw a fresh seed at
evaluation time and publish ``seed_commitment(seed, nonce)`` beforehand if you want. The random
nonce makes the commitment impossible to brute-force even for a small seed such as 42. Reveal the
seed and nonce afterwards so anyone can verify the commitment and regenerate the exact instances.
"""
from __future__ import annotations

import hashlib
import random
import secrets


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


def fresh_nonce() -> str:
    return secrets.token_hex(16)
