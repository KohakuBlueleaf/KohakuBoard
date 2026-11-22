"""Module-level active `Board` registry (`kohakuboard.init()` / `kohakuboard.log()`)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from kohakuboard.client.board import Board
from kohakuboard.errors import ConfigError

_LOCK = threading.Lock()
_ACTIVE_STACK: list[Board] = []


def _prune_closed_locked() -> None:
    while _ACTIVE_STACK and _ACTIVE_STACK[-1].closed:
        _ACTIVE_STACK.pop()


def _unregister_board(board: Board) -> None:
    with _LOCK:
        for idx in range(len(_ACTIVE_STACK) - 1, -1, -1):
            if _ACTIVE_STACK[idx] is board:
                _ACTIVE_STACK.pop(idx)
                break


def current() -> Board | None:
    """Return the currently active `Board` initialized via `kohakuboard.init()`, or `None`."""
    with _LOCK:
        _prune_closed_locked()
        return _ACTIVE_STACK[-1] if _ACTIVE_STACK else None


def _require_active(action: str) -> Board:
    board = current()
    if board is None:
        raise ConfigError(
            f"Cannot call kohakuboard.{action}(): project/board has not been initialized. "
            f"Call kohakuboard.init(...) first, or call .{action}() directly on a Board(...) instance."
        )
    return board


def init(
    *,
    name: str | None = None,
    project: str = "default",
    run_id: str | None = None,
    annotation: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    config: dict[str, Any] | None = None,
    dir: str | Path | None = None,
    mode: Literal["local", "remote", "hybrid", "memory", "disabled"] = "local",
    resume: Literal["never", "allow", "must"] = "never",
    tui: Any = False,
    capture_output: bool = True,
    system_metrics: bool | float = True,
    rank_zero_only: bool = True,
    remote_url: str | None = None,
    remote_token: str | None = None,
    flush_interval_ms: int = 500,
    queue_capacity: int = 65_536,
    overflow: Literal["block", "drop_oldest"] = "block",
    reinit: bool = False,
) -> Board:
    """Initialize a `Board` and register it as the module-level active run.

    Works both standalone (`kobo.init(...); kobo.log(...)`) and as a context manager
    (`with kobo.init(...) as board:`).
    """
    existing = current()
    if existing is not None:
        if reinit:
            existing.finish()
        else:
            raise ConfigError(
                "An active KohakuBoard run is already initialized. "
                "Call kohakuboard.finish() first, pass reinit=True, "
                "or construct Board(...) directly for concurrent runs."
            )

    board = Board(
        name=name,
        project=project,
        run_id=run_id,
        annotation=annotation,
        group=group,
        job_type=job_type,
        tags=tags,
        notes=notes,
        config=config,
        dir=dir,
        mode=mode,
        resume=resume,
        tui=tui,
        capture_output=capture_output,
        system_metrics=system_metrics,
        rank_zero_only=rank_zero_only,
        remote_url=remote_url,
        remote_token=remote_token,
        flush_interval_ms=flush_interval_ms,
        queue_capacity=queue_capacity,
        overflow=overflow,
    )
    board._on_finish_callbacks.append(_unregister_board)
    with _LOCK:
        _ACTIVE_STACK.append(board)
    return board


def log(
    data: Mapping[str, Any] | None = None,
    *,
    step: int | None = None,
    global_step: int | None = None,
    commit: bool = True,
    **metrics: Any,
) -> None:
    """Log metrics/payloads to the active run initialized via `kohakuboard.init()`."""
    _require_active("log").log(
        data,
        step=step,
        global_step=global_step,
        commit=commit,
        **metrics,
    )


def step(increment: int = 1, *, value: int | None = None) -> int:
    """Advance `global_step` on the active run."""
    return _require_active("step").step(increment, value=value)


def config_update(patch: dict[str, Any]) -> None:
    """Merge hyperparameters into the active run's config."""
    _require_active("config_update").config_update(patch)


def summary_update(patch: dict[str, float]) -> None:
    """Set explicit scalar summary values on the active run."""
    _require_active("summary_update").summary_update(patch)


def add_tags(*tags: str) -> None:
    """Add tags to the active run."""
    _require_active("add_tags").add_tags(*tags)


def set_notes(markdown: str) -> None:
    """Update markdown notes on the active run."""
    _require_active("set_notes").set_notes(markdown)


def define_metric(
    pattern: str,
    *,
    step_metric: str | None = None,
    summary: Literal["last", "min", "max", "mean", "none"] = "last",
    hidden: bool = False,
) -> None:
    """Register per-metric x-axis/summary preferences on the active run."""
    _require_active("define_metric").define_metric(
        pattern,
        step_metric=step_metric,
        summary=summary,
        hidden=hidden,
    )


def alert(
    title: str,
    text: str,
    level: Literal["info", "warn", "error"] = "info",
) -> None:
    """Log a structured alert event on the active run."""
    _require_active("alert").alert(title, text, level=level)


def progress(
    current_val: int,
    total: int | None = None,
    *,
    label: str = "train",
    unit: str = "step",
) -> None:
    """Update live progress tracking on the active run."""
    _require_active("progress").progress(current_val, total, label=label, unit=unit)


def snapshot() -> dict[str, Any]:
    """Return the live in-memory snapshot dictionary of the active run."""
    return _require_active("snapshot").snapshot()


def flush(timeout: float = 30.0) -> None:
    """Block until all queued events on the active run are committed to SQLite."""
    _require_active("flush").flush(timeout=timeout)


def finish(
    exit_code: int = 0,
    *,
    state: Literal["finished", "crashed", "interrupted"] | None = None,
    timeout: float = 30.0,
) -> None:
    """Idempotently finish and pop the current active run (no-op if none is active)."""
    board = current()
    if board is not None:
        board.finish(exit_code=exit_code, state=state, timeout=timeout)
