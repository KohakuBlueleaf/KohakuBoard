"""Project-wide `.catalog.db` reader for fast run discovery, filtering, and sorting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kohakuboard._rs import _Catalog
from kohakuboard.errors import RunNotFoundError, translate_rs_error
from kohakuboard.reader.run_reader import RunReader


@dataclass(slots=True)
class ProjectSummary:
    """Summary entry for a project indexed in `.catalog.db`."""

    name: str
    run_count: int
    running_count: int
    latest_updated_ms: int


@dataclass(slots=True)
class RunSummary:
    """Indexed run metadata, config, and scalar summary snapshot."""

    project: str
    run_id: str
    dir_name: str
    path: Path
    name: str
    annotation: str | None
    group: str | None
    job_type: str | None
    tags: list[str]
    notes: str
    state: str
    created_at_ms: int
    updated_at_ms: int
    finished_at_ms: int | None
    exit_code: int | None
    last_step: int
    last_global_step: int
    config: dict[str, Any]
    summary: dict[str, Any]


def _extract_dotted(mapping: dict[str, Any], path: str) -> Any:
    cur: Any = mapping
    if path in mapping:
        return mapping[path]
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _summary_metric_value(summary: dict[str, Any], key: str) -> float | None:
    entry = summary.get(key)
    if isinstance(entry, dict):
        val = entry.get("last")
        return float(val) if isinstance(val, (int, float)) else None
    if isinstance(entry, (int, float)):
        return float(entry)
    return None


class ProjectReader:
    """Read-only project/run catalog browser over `{data_dir}/.catalog.db`."""

    def __init__(self, data_dir: str | Path, *, project: str | None = None) -> None:
        self._data_dir = Path(data_dir)
        self._default_project = project
        try:
            self._catalog = _Catalog(str(self._data_dir))
        except Exception as exc:  # noqa: BLE001
            raise translate_rs_error(exc) from exc

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def refresh(self, *, force_rebuild: bool = False) -> int:
        """Incrementally update `.catalog.db` (or rebuild from scratch) and return updated count."""
        try:
            return self._catalog.refresh(force_rebuild)
        except Exception as exc:  # noqa: BLE001
            raise translate_rs_error(exc) from exc

    def list_projects(self, *, refresh: bool = True) -> list[ProjectSummary]:
        if refresh:
            self.refresh()
        rows = json.loads(self._catalog.list_projects_json())
        return [
            ProjectSummary(
                name=r["name"],
                run_count=r["run_count"],
                running_count=r["running_count"],
                latest_updated_ms=r["latest_updated_ms"],
            )
            for r in rows
        ]

    def list_runs(
        self,
        *,
        project: str | None = None,
        state: str | None = None,
        group: str | None = None,
        search: str | None = None,
        sort: str = "-updated_at_ms",
        limit: int | None = None,
        offset: int = 0,
        refresh: bool = True,
    ) -> list[RunSummary]:
        """List runs with optional project/state/group/search filtering and sorting.

        `sort` supports `-updated_at_ms`, `created_at_ms`, `name`, `last_step`,
        `summary.<metric>`, or `config.<key>` (prefix with `-` for descending).
        """
        if refresh:
            self.refresh()

        proj = project if project is not None else self._default_project
        descending = sort.startswith("-")
        sort_key = sort[1:] if descending else sort
        custom_sort = sort_key.startswith(("summary.", "config."))

        raw_rows = json.loads(
            self._catalog.list_runs_json(
                project=proj,
                state=state,
                group=group,
                search=search,
                sort_by=None if custom_sort else sort_key,
                descending=descending,
                limit=None if custom_sort else limit,
                offset=0 if custom_sort else offset,
            )
        )
        runs = [
            RunSummary(
                project=r["project"],
                run_id=r["run_id"],
                dir_name=r["dir_name"],
                path=Path(r["path"]),
                name=r["name"],
                annotation=r["annotation"],
                group=r["group"],
                job_type=r["job_type"],
                tags=r["tags"],
                notes=r["notes"],
                state=r["state"],
                created_at_ms=r["created_at_ms"],
                updated_at_ms=r["updated_at_ms"],
                finished_at_ms=r["finished_at_ms"],
                exit_code=r["exit_code"],
                last_step=r["last_step"],
                last_global_step=r["last_global_step"],
                config=r["config"],
                summary=r["summary"],
            )
            for r in raw_rows
        ]

        if custom_sort:
            if sort_key.startswith("summary."):
                metric = sort_key[len("summary.") :]
                runs.sort(
                    key=lambda r: (
                        _summary_metric_value(r.summary, metric) is None,
                        _summary_metric_value(r.summary, metric) or 0.0,
                    ),
                    reverse=descending,
                )
            elif sort_key.startswith("config."):
                cfg_key = sort_key[len("config.") :]
                runs.sort(
                    key=lambda r: (
                        _extract_dotted(r.config, cfg_key) is None,
                        str(_extract_dotted(r.config, cfg_key) or ""),
                    ),
                    reverse=descending,
                )
            if offset:
                runs = runs[offset:]
            if limit is not None:
                runs = runs[:limit]

        return runs

    def open_run(self, run_id: str, *, project: str | None = None) -> RunReader:
        """Resolve `(project, run_id)` via `.catalog.db` and open a read-only `RunReader`."""
        proj = project if project is not None else (self._default_project or "default")
        path = self._catalog.resolve_run_path(proj, run_id)
        if path is None:
            self.refresh()
            path = self._catalog.resolve_run_path(proj, run_id)
        if path is None:
            raise RunNotFoundError(f"Run '{run_id}' not found in project '{proj}'")
        return RunReader(path)

    def __enter__(self) -> ProjectReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None
