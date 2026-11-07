"""Small shared helpers: canonical hashing, JSON files, UTC timestamps."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal data always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
