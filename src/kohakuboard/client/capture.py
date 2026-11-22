"""Non-invasive stdout/stderr tee forwarding console lines to `_LoggerEngine`."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TextIO


class _TeeStream:
    """Line-buffered stream wrapper forwarding writes to the original stream and logger."""

    def __init__(
        self,
        original: TextIO,
        stream_name: str,
        emit_line: Callable[[int, str, str], None],
    ) -> None:
        self._original = original
        self._stream_name = stream_name
        self._emit_line = emit_line
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        written = self._original.write(data)
        if not data:
            return written
        lines_to_emit: list[str] = []
        with self._lock:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                cleaned = line.rstrip("\r")
                if cleaned:
                    lines_to_emit.append(cleaned)
        if lines_to_emit:
            ts_ms = int(time.time() * 1000)
            for line in lines_to_emit:
                self._emit_line(ts_ms, self._stream_name, line)
        return written

    def flush(self) -> None:
        self._original.flush()

    def flush_pending(self) -> None:
        with self._lock:
            rem = self._buf.rstrip("\r")
            self._buf = ""
        if rem:
            self._emit_line(int(time.time() * 1000), self._stream_name, rem)

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class OutputCapture:
    """Install and deterministically restore stdout/stderr tees for a `Board`."""

    def __init__(self, emit_line: Callable[[int, str, str], None]) -> None:
        self._emit_line = emit_line
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None
        self._tee_stdout: _TeeStream | None = None
        self._tee_stderr: _TeeStream | None = None
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._tee_stdout = _TeeStream(self._orig_stdout, "stdout", self._emit_line)
        self._tee_stderr = _TeeStream(self._orig_stderr, "stderr", self._emit_line)
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._tee_stdout is not None:
            self._tee_stdout.flush_pending()
        if self._tee_stderr is not None:
            self._tee_stderr.flush_pending()
        if sys.stdout is self._tee_stdout and self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if sys.stderr is self._tee_stderr and self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        self._tee_stdout = None
        self._tee_stderr = None
        self._orig_stdout = None
        self._orig_stderr = None
