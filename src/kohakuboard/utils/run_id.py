"""Utilities for generating friendly run identifiers."""

from __future__ import annotations

import importlib.resources as resources
import random
import re
import string
from typing import Final

FRIENDLY_WORDS_PACKAGE = "kohakuboard.data.friendly_words"
BASE36_ALPHABET: Final[str] = string.ascii_lowercase + string.digits

_WORD_CACHE: dict[str, list[str]] = {}


def _load_word_list(filename: str) -> list[str]:
    """Load and cache word lists shipped with the package."""
    if filename not in _WORD_CACHE:
        file_path = resources.files(FRIENDLY_WORDS_PACKAGE).joinpath(filename)
        with file_path.open("r", encoding="utf-8") as handle:
            _WORD_CACHE[filename] = [line.strip() for line in handle if line.strip()]
    return _WORD_CACHE[filename]


def generate_annotation_id(length: int = 4) -> str:
    """Generate a short annotation/run_id (default 4 chars)."""
    return "".join(random.choices(BASE36_ALPHABET, k=length))


def generate_friendly_name() -> str:
    """Generate a friendly human-readable run name."""
    predicate = random.choice(_load_word_list("predicates.txt"))
    obj = random.choice(_load_word_list("objects.txt"))
    return f"{predicate.title()} {obj.title()}"


def sanitize_annotation(value: str) -> str:
    """Sanitize annotation to filesystem-friendly form (lowercase, _, - only)."""
    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("_-")
