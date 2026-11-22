"""Typed exception hierarchy for KohakuBoard."""

from __future__ import annotations


class KohakuBoardError(Exception):
    """Base exception for all KohakuBoard errors."""


class ConfigError(KohakuBoardError):
    """Invalid configuration, argument, or unsupported mode."""


class StorageError(KohakuBoardError):
    """SQLite / KohakuVault I/O, corruption, or schema mismatch error."""


class ClosedBoardError(KohakuBoardError):
    """Raised when logging or flushing on an already-finished Board."""


class FlushTimeoutError(KohakuBoardError):
    """Raised when `board.flush()` exceeds its timeout."""


class RunNotFoundError(KohakuBoardError):
    """Raised when a requested run directory or run_id does not exist."""


class RunExistsError(KohakuBoardError):
    """Raised when creating a run with `resume='never'` collides with an existing run."""


class SyncError(KohakuBoardError):
    """Raised on remote synchronization authentication, network, or protocol errors."""


def translate_rs_error(exc: Exception) -> KohakuBoardError:
    """Convert a native PyO3 exception from `kohakuboard._rs` into a typed KohakuBoardError."""
    msg = str(exc)
    if isinstance(exc, TimeoutError):
        return FlushTimeoutError(msg)
    if isinstance(exc, KeyError):
        return RunNotFoundError(msg.strip("'\""))
    if isinstance(exc, ValueError):
        if "board is closed" in msg:
            return ClosedBoardError(msg)
        return ConfigError(msg)
    return StorageError(msg)
