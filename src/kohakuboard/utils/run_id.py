"""Utilities for generating friendly run identifiers."""

from __future__ import annotations

import importlib.resources as resources
import random
import re
import string
from pathlib import Path
from typing import Final, Tuple

FRIENDLY_WORDS_PACKAGE = "kohakuboard.data.friendly_words"
BASE36_ALPHABET: Final[str] = string.ascii_lowercase + string.digits
RUN_ID_ALPHABET: Final[str] = string.ascii_lowercase + string.digits

_WORD_CACHE: dict[str, list[str]] = {}


def _load_word_list(filename: str) -> list[str]:
    """Load and cache word lists shipped with the package."""
    if filename not in _WORD_CACHE:
        file_path = resources.files(FRIENDLY_WORDS_PACKAGE).joinpath(filename)
        with file_path.open("r", encoding="utf-8") as handle:
            _WORD_CACHE[filename] = [line.strip() for line in handle if line.strip()]
    return _WORD_CACHE[filename]


def generate_annotation_id(length: int = 4) -> str:
    """Generate a short annotation suffix (default 4 chars)."""
    return "".join(random.choices(BASE36_ALPHABET, k=length))


def generate_run_id(length: int = 4) -> str:
    """Generate a compact run identifier (default 4 chars)."""
    return "".join(random.choices(RUN_ID_ALPHABET, k=length))


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


def split_run_dir_name(dir_name: str) -> Tuple[str, str | None]:
    """Split a run directory name into (run_id, annotation).

    Args:
        dir_name: Directory basename, e.g. "abc123_my-run"

    Returns:
        Tuple of (run_id, annotation). Annotation is None if missing.
    """
    if "_" not in dir_name:
        return dir_name, None

    run_id, annotation = dir_name.split("_", 1)
    return run_id, annotation or None


def build_run_dir_name(run_id: str, annotation: str | None) -> str:
    """Build directory name from run_id + annotation."""
    if annotation:
        return f"{run_id}_{annotation}"
    return run_id


def find_run_dir_by_id(project_dir: Path, run_id: str) -> Path | None:
    """Find run directory within project_dir that matches run_id prefix."""
    direct = project_dir / run_id
    if direct.exists():
        return direct

    if not project_dir.exists():
        return None

    for entry in project_dir.iterdir():
        if not entry.is_dir():
            continue
        entry_run_id, _ = split_run_dir_name(entry.name)
        if entry_run_id == run_id:
            return entry

    return None
