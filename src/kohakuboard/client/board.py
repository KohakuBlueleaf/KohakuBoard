"""User-facing `Board` experiment logger backed by the Rust `_LoggerEngine`."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import platform
import sys
import time
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from kohakuboard._rs import _LoggerEngine
from kohakuboard.client.capture import OutputCapture
from kohakuboard.client.names import (
    generate_friendly_name,
    generate_run_id,
    validate_run_id,
    validate_slug,
)
from kohakuboard.client.types import (
    Artifact,
    Histogram,
    KernelDensity,
    Media,
    Table,
    TensorLog,
)
from kohakuboard.errors import (
    ClosedBoardError,
    ConfigError,
    RunExistsError,
    RunNotFoundError,
    translate_rs_error,
)
from kohakuboard.tui.config import resolve_tui_config

_SCALAR_TYPES = (int, float, np.integer, np.floating)
_ACTIVE_BOARDS: set[weakref.ref[Board]] = set()
_ATEXIT_REGISTERED = False
_EXCEPTHOOK_INSTALLED = False
_UNCAUGHT_EXC_TYPE: type[BaseException] | None = None


def _install_excepthook() -> None:
    global _EXCEPTHOOK_INSTALLED
    if _EXCEPTHOOK_INSTALLED:
        return
    prev_hook = sys.excepthook

    def _chained_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> None:
        global _UNCAUGHT_EXC_TYPE
        _UNCAUGHT_EXC_TYPE = exc_type
        prev_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _chained_excepthook
    _EXCEPTHOOK_INSTALLED = True


def _cleanup_active_boards() -> None:
    if _UNCAUGHT_EXC_TYPE is None:
        exit_code = 0
        final_state: Literal["finished", "crashed", "interrupted"] = "finished"
    elif issubclass(_UNCAUGHT_EXC_TYPE, KeyboardInterrupt):
        exit_code = 130
        final_state = "interrupted"
    else:
        exit_code = 1
        final_state = "crashed"

    for ref in list(_ACTIVE_BOARDS):
        board = ref()
        if board is not None and not board.closed:
            with contextlib.suppress(Exception):
                board.finish(exit_code=exit_code, state=final_state)


def _register_board(board: Board) -> weakref.ref[Board]:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_cleanup_active_boards)
        _ATEXIT_REGISTERED = True
    _install_excepthook()

    def _on_gc(ref: weakref.ref[Board]) -> None:
        _ACTIVE_BOARDS.discard(ref)

    ref = weakref.ref(board, _on_gc)
    _ACTIVE_BOARDS.add(ref)
    return ref


def _is_nonzero_rank() -> bool:
    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                if int(val) > 0:
                    return True
            except ValueError:
                pass
    return False


def _collect_python_env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }


def _find_existing_run_dir(project_dir: Path, run_id: str) -> Path | None:
    if not project_dir.exists():
        return None
    exact = project_dir / run_id
    if exact.is_dir():
        return exact
    prefix = f"{run_id}_"
    for entry in sorted(project_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith(prefix):
            return entry
    return None


class Board:
    """Local-first, zero-overhead ML experiment tracker backed by Rust + KohakuVault 1.2."""

    def __init__(
        self,
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
    ) -> None:
        if mode in ("remote", "hybrid"):
            raise ConfigError(
                f"mode={mode!r} is not yet implemented. "
                "Use mode='local', 'memory', or 'disabled'."
            )
        if mode not in ("local", "memory", "disabled"):
            raise ConfigError(f"Unsupported mode {mode!r}")
        if resume not in ("never", "allow", "must"):
            raise ConfigError(f"Unsupported resume policy {resume!r}")
        if overflow not in ("block", "drop_oldest"):
            raise ConfigError(f"Unsupported overflow policy {overflow!r}")
        tui_cfg = resolve_tui_config(tui)
        _ = (remote_url, remote_token)

        try:
            self.project = validate_slug(project, "project")
            if annotation is not None:
                annotation = validate_slug(annotation, "annotation")
            explicit_id = run_id is not None
            self.run_id = validate_run_id(run_id) if run_id is not None else generate_run_id()
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        self.name = name or generate_friendly_name()
        self.annotation = annotation
        self.group = group
        self.job_type = job_type
        self.tags = list(tags or [])
        self.notes = notes or ""
        self.config: dict[str, Any] = dict(config or {})
        self.mode = mode

        self._disabled = mode == "disabled" or (rank_zero_only and _is_nonzero_rank())
        self._closed = False
        self._engine: _LoggerEngine | None = None
        self._capture: OutputCapture | None = None
        self._tui: Any = None
        self._weakref: weakref.ref[Board] | None = None
        self._on_finish_callbacks: list[Callable[[Board], None]] = []
        self._step = -1
        self._global_step = 0
        self._explicit_global_step = False
        self._has_pending_commit = False
        self.run_dir: Path | None = None

        if self._disabled:
            return

        if system_metrics is False:
            sysmon_ms: int | None = None
        elif system_metrics is True:
            sysmon_ms = 2000
        elif isinstance(system_metrics, (int, float)) and system_metrics > 0:
            sysmon_ms = max(250, int(float(system_metrics) * 1000))
        else:
            raise ConfigError("system_metrics must be bool or positive number of seconds")

        should_resume = False
        if mode == "local":
            base = (
                Path(dir) if dir is not None else Path(os.environ.get("KOBO_DIR", "./kohakuboard"))
            )
            project_dir = base / self.project
            existing = _find_existing_run_dir(project_dir, self.run_id)
            dir_name = f"{self.run_id}_{self.annotation}" if self.annotation else self.run_id
            if existing is not None:
                if resume == "never":
                    if explicit_id:
                        raise RunExistsError(
                            f"Run '{self.run_id}' already exists at {existing}; "
                            "pass resume='allow' or resume='must' to resume."
                        )
                    # Auto-generated id collision: regenerate cleanly.
                    while existing is not None:
                        self.run_id = generate_run_id()
                        existing = _find_existing_run_dir(project_dir, self.run_id)
                    dir_name = (
                        f"{self.run_id}_{self.annotation}" if self.annotation else self.run_id
                    )
                    self.run_dir = project_dir / dir_name
                else:
                    self.run_dir = existing
                    should_resume = (existing / "run.db").exists()
            else:
                if resume == "must":
                    raise RunNotFoundError(
                        f"Cannot resume run '{self.run_id}' in {project_dir}: "
                        "run directory not found"
                    )
                self.run_dir = project_dir / dir_name
            run_dir_arg: str | None = str(self.run_dir)
        else:
            if resume == "must":
                raise ConfigError("resume='must' is not supported in mode='memory'")
            run_dir_arg = None

        try:
            self._engine = _LoggerEngine(
                run_dir=run_dir_arg,
                run_id=self.run_id,
                name=self.name,
                project=self.project,
                annotation=self.annotation,
                group=self.group,
                job_type=self.job_type,
                tags=self.tags,
                notes=self.notes,
                config_json=json.dumps(self.config, default=str),
                env_json=json.dumps(_collect_python_env(), default=str),
                resume=should_resume,
                flush_interval_ms=int(flush_interval_ms),
                queue_capacity=int(queue_capacity),
                overflow=overflow,
                system_metrics_interval_ms=sysmon_ms,
            )
        except Exception as exc:
            raise translate_rs_error(exc) from exc

        self._step = self._engine.initial_last_step
        self._global_step = self._engine.initial_last_global_step
        self._explicit_global_step = self._step >= 0 and self._global_step != self._step
        self._weakref = _register_board(self)

        tui_started = False
        if tui_cfg is not None:
            try:
                from kohakuboard.tui.adapter import LiveTuiHandle  # noqa: PLC0415
            except ModuleNotFoundError as exc:
                raise ConfigError(
                    'Terminal UI requires kohakutui. Install with: pip install "kohakuboard[tui]"'
                ) from exc
            handle = LiveTuiHandle(self, tui_cfg)
            tui_started = handle.start()
            self._tui = handle

        route_to_tui = tui_started and tui_cfg is not None and tui_cfg.capture_stdio
        if capture_output or route_to_tui:
            self._capture = OutputCapture(
                self._emit_console_line,
                passthrough=not route_to_tui,
                on_raw=self._tui.write_log if route_to_tui and self._tui is not None else None,
            )
            self._capture.start()

    @property
    def tui(self) -> Any:
        """Active `LiveTuiHandle` when `tui` is enabled, else `None`."""
        return self._tui

    @property
    def disabled(self) -> bool:
        """True when logging is a zero-overhead no-op (disabled or non-zero DDP rank)."""
        return self._disabled

    @property
    def closed(self) -> bool:
        """True once `finish()` has been called."""
        return self._closed

    @property
    def current_step(self) -> int:
        """Current internal step counter."""
        return max(self._step, 0)

    @property
    def global_step(self) -> int:
        """Current training global_step counter."""
        return self._global_step

    def _require_open(self) -> _LoggerEngine:
        if self._closed or self._engine is None:
            raise ClosedBoardError("Cannot log to a finished Board")
        return self._engine

    def _emit_console_line(self, ts_ms: int, stream: str, line: str) -> None:
        if self._closed or self._engine is None:
            return
        with contextlib.suppress(Exception):
            self._engine.log_lines([(ts_ms, max(self._step, 0), stream, line)])

    def step(self, increment: int = 1, *, value: int | None = None) -> int:
        """Advance `global_step` (or set it explicitly via `value=`) and seal pending commits."""
        if self._disabled:
            return self._global_step
        engine = self._require_open()
        self._explicit_global_step = True
        if value is not None:
            self._global_step = int(value)
        else:
            self._global_step += int(increment)
        self._has_pending_commit = False
        engine.set_step_counters(max(self._step, 0), self._global_step)
        return self._global_step

    def log(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        step: int | None = None,
        global_step: int | None = None,
        commit: bool = True,
        **metrics: Any,
    ) -> None:
        """Log scalars, media, tables, histograms, KDEs, tensors, or artifacts."""
        if self._disabled:
            return
        engine = self._require_open()

        if step is not None:
            cur_step = int(step)
            self._step = max(self._step, cur_step)
        elif self._has_pending_commit:
            cur_step = max(self._step, 0)
        else:
            self._step += 1
            cur_step = self._step

        self._has_pending_commit = not commit
        if global_step is not None:
            cur_gstep = int(global_step)
            self._global_step = max(self._global_step, cur_gstep)
            self._explicit_global_step = True
        elif self._explicit_global_step:
            cur_gstep = self._global_step
        else:
            cur_gstep = cur_step
            self._global_step = cur_step
        ts_ms = int(time.time() * 1000)

        # Ultra-fast path: single mapping or kwargs of pure scalar numbers.
        if data is not None and not metrics:
            items = data.items()
        elif data is None:
            items = metrics.items()
        else:
            merged = dict(data)
            merged.update(metrics)
            items = merged.items()

        scalars: list[tuple[str, float]] = []
        rich: list[tuple[str, Any]] = []
        for k, v in items:
            if type(v) is float or type(v) is int or isinstance(v, _SCALAR_TYPES):
                scalars.append((k, float(v)))
            else:
                rich.append((k, v))

        try:
            if scalars:
                engine.log_scalars(cur_step, cur_gstep, ts_ms, scalars)
            if rich:
                self._log_rich_items(engine, cur_step, cur_gstep, ts_ms, rich)
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def _log_rich_items(
        self,
        engine: _LoggerEngine,
        step: int,
        global_step: int,
        ts_ms: int,
        items: list[tuple[str, Any]],
    ) -> None:
        extra_scalars: list[tuple[str, float]] = []
        for key, value in items:
            match value:
                case Mapping():
                    nested_rich: list[tuple[str, Any]] = []
                    for sub_k, sub_v in value.items():
                        full_key = f"{key}/{sub_k}"
                        if isinstance(sub_v, _SCALAR_TYPES):
                            extra_scalars.append((full_key, float(sub_v)))
                        else:
                            nested_rich.append((full_key, sub_v))
                    if nested_rich:
                        self._log_rich_items(engine, step, global_step, ts_ms, nested_rich)
                case Media():
                    self._send_media(engine, step, global_step, ts_ms, key, 0, value)
                case list() if value and all(isinstance(x, Media) for x in value):
                    for idx, media_item in enumerate(value):
                        self._send_media(engine, step, global_step, ts_ms, key, idx, media_item)
                case Table():
                    cols_json, row_count, rows_json = value.serialize(
                        lambda m: self._stage_table_media(engine, m)
                    )
                    engine.log_table(step, global_step, ts_ms, key, cols_json, row_count, rows_json)
                case Histogram():
                    if value.values is not None:
                        engine.log_histogram_raw(
                            step,
                            global_step,
                            ts_ms,
                            key,
                            value.to_raw_f32_bytes(),
                            int(value.bins),
                            value.precision,
                            value.range,
                            value.percentiles,
                        )
                    else:
                        engine.log_histogram_precomputed(
                            step,
                            global_step,
                            ts_ms,
                            key,
                            [float(e) for e in value.edges or ()],
                            [int(c) for c in value.counts or ()],
                            value.precision,
                        )
                case KernelDensity():
                    if value.values is not None:
                        bw = None if value.bandwidth == "auto" else float(value.bandwidth)
                        engine.log_kde_raw(
                            step,
                            global_step,
                            ts_ms,
                            key,
                            value.to_raw_f32_bytes(),
                            int(value.num_points),
                            bw,
                            bool(value.approximate),
                            value.range,
                            value.percentiles,
                        )
                    else:
                        grid = [float(x) for x in value.grid or ()]
                        dens = [float(x) for x in value.density or ()]
                        r_min = (
                            float(value.range[0]) if value.range else (min(grid) if grid else 0.0)
                        )
                        r_max = (
                            float(value.range[1]) if value.range else (max(grid) if grid else 1.0)
                        )
                        bw_val = 0.0 if value.bandwidth == "auto" else float(value.bandwidth)
                        engine.log_kde_precomputed(
                            step,
                            global_step,
                            ts_ms,
                            key,
                            r_min,
                            r_max,
                            bw_val,
                            int(value.sample_count or len(grid)),
                            grid,
                            dens,
                        )
                case TensorLog():
                    raw_bytes, dtype, shape_json, compress, meta_json = value.prepare()
                    engine.log_tensor(
                        step,
                        global_step,
                        ts_ms,
                        key,
                        raw_bytes,
                        dtype,
                        shape_json,
                        compress,
                        meta_json,
                    )
                case Artifact():
                    name, kind, uri, digest, size_bytes, meta_json = value.prepare()
                    engine.log_artifact(
                        step,
                        global_step,
                        ts_ms,
                        name or key,
                        kind,
                        uri,
                        digest,
                        size_bytes,
                        meta_json,
                    )
                case None:
                    continue
                case _:
                    raise ConfigError(
                        f"Unsupported metric value type for '{key}': {type(value).__name__}"
                    )

        if extra_scalars:
            engine.log_scalars(step, global_step, ts_ms, extra_scalars)

    def _send_media(
        self,
        engine: _LoggerEngine,
        step: int,
        global_step: int,
        ts_ms: int,
        name: str,
        idx: int,
        media: Media,
    ) -> None:
        kind, payload = media.prepare()
        if kind == "pixels":
            pixels, width, height, channels, fmt, quality, caption, meta_json = payload
            engine.log_media_pixels(
                step,
                global_step,
                ts_ms,
                name,
                idx,
                pixels,
                width,
                height,
                channels,
                fmt,
                quality,
                caption,
                meta_json,
            )
        else:
            media_type, raw, ext, duration_ms, caption, meta_json = payload
            engine.log_media_encoded(
                step,
                global_step,
                ts_ms,
                name,
                idx,
                media_type,
                raw,
                ext,
                duration_ms,
                caption,
                meta_json,
            )

    def _stage_table_media(self, engine: _LoggerEngine, media: Media) -> dict[str, Any]:
        kind, payload = media.prepare()
        if kind == "pixels":
            pixels, width, height, channels, fmt, quality, caption, _ = payload
            blob_key, mime, w, h, size = engine.stage_media_pixels(
                pixels, width, height, channels, fmt, quality
            )
        else:
            media_type, raw, ext, _, caption, _ = payload
            blob_key, mime, w, h, size = engine.stage_media_encoded(raw, ext, media_type)
        return {
            "$media": blob_key,
            "mime": mime,
            "width": w,
            "height": h,
            "size_bytes": size,
            "caption": caption,
        }

    def config_update(self, patch: dict[str, Any]) -> None:
        """Merge additional hyperparameters into the run config."""
        if self._disabled:
            return
        engine = self._require_open()
        self.config.update(patch)
        try:
            engine.patch_config(json.dumps(patch, default=str))
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def summary_update(self, patch: dict[str, float]) -> None:
        """Set explicit summary scalar values."""
        if self._disabled:
            return
        engine = self._require_open()
        clean = {str(k): float(v) for k, v in patch.items()}
        try:
            engine.patch_summary(
                max(self._step, 0), self._global_step, int(time.time() * 1000), clean
            )
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def add_tags(self, *tags: str) -> None:
        """Add tags to the run."""
        if self._disabled or not tags:
            return
        engine = self._require_open()
        for t in tags:
            if t not in self.tags:
                self.tags.append(t)
        try:
            engine.patch_meta(add_tags=list(tags))
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def set_notes(self, markdown: str) -> None:
        """Update run markdown notes."""
        if self._disabled:
            return
        engine = self._require_open()
        self.notes = markdown
        try:
            engine.patch_meta(notes=markdown)
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def define_metric(
        self,
        pattern: str,
        *,
        step_metric: str | None = None,
        summary: Literal["last", "min", "max", "mean", "none"] = "last",
        hidden: bool = False,
    ) -> None:
        """Register W&B-style per-metric x-axis and summary display preferences."""
        if self._disabled:
            return
        if summary not in ("last", "min", "max", "mean", "none"):
            raise ConfigError(f"Unsupported summary mode {summary!r}")
        engine = self._require_open()
        try:
            engine.define_metric(pattern, step_metric=step_metric, summary=summary, hidden=hidden)
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def alert(
        self,
        title: str,
        text: str,
        level: Literal["info", "warn", "error"] = "info",
    ) -> None:
        """Log a structured alert / milestone event."""
        if self._disabled:
            return
        if level not in ("info", "warn", "error"):
            raise ConfigError(f"Unsupported alert level {level!r}")
        engine = self._require_open()
        try:
            engine.log_alert(
                max(self._step, 0),
                self._global_step,
                int(time.time() * 1000),
                level,
                title,
                text,
            )
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def progress(
        self,
        current: int,
        total: int | None = None,
        *,
        label: str = "train",
        unit: str = "step",
    ) -> None:
        """Update live progress bar state without polluting scalar charts."""
        if self._disabled:
            return
        engine = self._require_open()
        try:
            engine.progress(
                label,
                max(0, int(current)),
                None if total is None else max(0, int(total)),
                unit,
                int(time.time() * 1000),
            )
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def snapshot(self) -> dict[str, Any]:
        """Return the live in-memory run snapshot dictionary."""
        if self._disabled or self._engine is None:
            return {
                "run_id": self.run_id,
                "name": self.name,
                "project": self.project,
                "state": "disabled",
                "metrics": {},
            }
        return json.loads(self._engine.snapshot_json())

    def flush(self, timeout: float = 30.0) -> None:
        """Block (with GIL released) until all prior logged events are committed to SQLite."""
        if self._disabled:
            return
        engine = self._require_open()
        try:
            engine.flush(max(1, int(timeout * 1000)))
        except Exception as exc:
            raise translate_rs_error(exc) from exc

    def finish(
        self,
        exit_code: int = 0,
        *,
        state: Literal["finished", "crashed", "interrupted"] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Idempotently flush, checkpoint WAL, restore stdout/stderr, and close handles."""
        if self._closed:
            return
        self._closed = True
        final_state = state or ("finished" if exit_code == 0 else "crashed")
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._tui is not None:
            with contextlib.suppress(Exception):
                self._tui.finish(state=final_state)
            self._tui = None
        if self._weakref is not None:
            _ACTIVE_BOARDS.discard(self._weakref)
            self._weakref = None
        for cb in self._on_finish_callbacks:
            cb(self)
        self._on_finish_callbacks.clear()
        if self._engine is not None:
            try:
                self._engine.finish(final_state, int(exit_code), max(1, int(timeout * 1000)))
            except Exception as exc:
                raise translate_rs_error(exc) from exc

    def __enter__(self) -> Board:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if exc_type is None:
            self.finish(exit_code=0, state="finished")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finish(exit_code=130, state="interrupted")
        else:
            self.finish(exit_code=1, state="crashed")
