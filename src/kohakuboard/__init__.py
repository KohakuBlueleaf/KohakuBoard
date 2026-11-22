"""KohakuBoard v2 — local-first, zero-overhead ML experiment tracking."""

from __future__ import annotations

from kohakuboard import errors
from kohakuboard._rs import SCHEMA_VERSION, version as _rs_version
from kohakuboard.client import (
    Artifact,
    Board,
    Histogram,
    KernelDensity,
    Media,
    Table,
    TensorLog,
    add_tags,
    alert,
    config_update,
    current,
    define_metric,
    finish,
    flush,
    init,
    log,
    progress,
    set_notes,
    snapshot,
    step,
    summary_update,
)
from kohakuboard.reader import (
    ProjectReader,
    ProjectSummary,
    RunReader,
    RunSummary,
    ScalarSlice,
)

__version__: str = _rs_version()

__all__ = [
    "SCHEMA_VERSION",
    "Artifact",
    "Board",
    "Histogram",
    "KernelDensity",
    "Media",
    "ProjectReader",
    "ProjectSummary",
    "RunReader",
    "RunSummary",
    "ScalarSlice",
    "Table",
    "TensorLog",
    "__version__",
    "add_tags",
    "alert",
    "config_update",
    "current",
    "define_metric",
    "errors",
    "finish",
    "flush",
    "init",
    "log",
    "progress",
    "set_notes",
    "snapshot",
    "step",
    "summary_update",
]
