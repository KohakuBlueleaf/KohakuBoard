"""Run ID generation, friendly run naming, and path slug validation."""

from __future__ import annotations

import importlib.resources as resources
import re
import secrets
from functools import lru_cache

_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{3,31}$")
_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


@lru_cache(maxsize=1)
def _load_word_lists() -> tuple[tuple[str, ...], tuple[str, ...]]:
    pkg = resources.files("kohakuboard.data.friendly_words")
    predicates = tuple(
        line.strip()
        for line in (pkg / "predicates.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    objects = tuple(
        line.strip()
        for line in (pkg / "objects.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return predicates, objects


def generate_run_id(length: int = 4) -> str:
    """Generate a lowercase base36 run identifier."""
    return "".join(secrets.choice(_BASE36_ALPHABET) for _ in range(length))


def generate_friendly_name() -> str:
    """Generate a memorable `predicate-object` run name."""
    predicates, objects = _load_word_lists()
    return f"{secrets.choice(predicates)}-{secrets.choice(objects)}"


def validate_run_id(run_id: str) -> str:
    """Validate an explicit run_id to ensure safe directory naming."""
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"run_id must be 4..32 chars matching [a-zA-Z0-9][a-zA-Z0-9_-]*, got {run_id!r}"
        )
    return run_id


def validate_slug(value: str, field_name: str) -> str:
    """Validate project or annotation slugs against path traversal and invalid chars."""
    if not _SLUG_RE.match(value) or ".." in value:
        raise ValueError(
            f"{field_name} must be 1..64 safe path chars without '..', got {value!r}"
        )
    return value
