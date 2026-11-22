"""Pure read/write service layer over `kohakuboard.reader` (ProjectReader + RunReader)."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from kohakuboard.errors import ConfigError, RunNotFoundError
from kohakuboard.reader import ProjectReader, RunReader

_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".json": "application/json",
    ".gz": "application/gzip",
}


def _mime_for_blob_key(key: str) -> str:
    lower = key.lower()
    for ext, mime in _EXT_MIME.items():
        if lower.endswith(ext):
            return mime
    guessed, _ = mimetypes.guess_type(key)
    return guessed or "application/octet-stream"


def _validate_segment(val: str, name: str) -> str:
    if not val or "/" in val or "\\" in val or ".." in val or val.startswith("."):
        raise ConfigError(f"Invalid {name}: {val!r}")
    return val


class BoardService:
    """Thread-safe orchestration over `.catalog.db` and pooled read-only `RunReader`s."""

    def __init__(self, data_dir: str | Path, *, max_open_readers: int = 32) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = ProjectReader(self.data_dir)
        self._max_readers = max(4, max_open_readers)
        self._readers: OrderedDict[str, RunReader] = OrderedDict()
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            for reader in self._readers.values():
                reader.close()
            self._readers.clear()

    def _resolve_run_dir(self, project: str, run_id: str) -> Path:
        proj = _validate_segment(project, "project")
        rid = _validate_segment(run_id, "run_id")
        with self._catalog.open_run(rid, project=proj) as rr:
            return rr.run_dir

    def _get_reader(self, project: str, run_id: str) -> RunReader:
        proj = _validate_segment(project, "project")
        rid = _validate_segment(run_id, "run_id")
        key = f"{proj}/{rid}"
        with self._lock:
            cached = self._readers.get(key)
            if cached is not None:
                if (cached.run_dir / "run.db").exists():
                    self._readers.move_to_end(key)
                    return cached
                cached.close()
                self._readers.pop(key, None)

        reader = self._catalog.open_run(rid, project=proj)
        with self._lock:
            self._readers[key] = reader
            self._readers.move_to_end(key)
            while len(self._readers) > self._max_readers:
                _, evicted = self._readers.popitem(last=False)
                evicted.close()
        return reader

    def _evict_reader(self, project: str, run_id: str) -> None:
        key = f"{project}/{run_id}"
        with self._lock:
            reader = self._readers.pop(key, None)
            if reader is not None:
                reader.close()

    def list_projects(self, *, refresh: bool = True) -> list[dict[str, Any]]:
        projects = self._catalog.list_projects(refresh=refresh)
        return [
            {
                "name": p.name,
                "run_count": p.run_count,
                "running_count": p.running_count,
                "latest_updated_ms": p.latest_updated_ms,
            }
            for p in projects
        ]

    def list_runs(
        self,
        project: str | None = None,
        *,
        state: str | None = None,
        group: str | None = None,
        search: str | None = None,
        sort: str = "-updated_at_ms",
        limit: int | None = None,
        offset: int = 0,
        refresh: bool = True,
    ) -> list[dict[str, Any]]:
        if project is not None:
            _validate_segment(project, "project")
        runs = self._catalog.list_runs(
            project=project,
            state=state,
            group=group,
            search=search,
            sort=sort,
            limit=limit,
            offset=offset,
            refresh=refresh,
        )
        return [
            {
                "project": r.project,
                "run_id": r.run_id,
                "dir_name": r.dir_name,
                "name": r.name,
                "annotation": r.annotation,
                "group": r.group,
                "job_type": r.job_type,
                "tags": r.tags,
                "notes": r.notes,
                "state": r.state,
                "created_at_ms": r.created_at_ms,
                "updated_at_ms": r.updated_at_ms,
                "finished_at_ms": r.finished_at_ms,
                "exit_code": r.exit_code,
                "last_step": r.last_step,
                "last_global_step": r.last_global_step,
                "config": r.config,
                "summary": r.summary,
            }
            for r in runs
        ]

    def get_run_detail(self, project: str, run_id: str) -> dict[str, Any]:
        reader = self._get_reader(project, run_id)
        run_dir = reader.run_dir
        run_db = run_dir / "run.db"
        blobs_db = run_dir / "blobs.db"
        return {
            "info": reader.info,
            "config": reader.config,
            "env": reader.env,
            "summary": reader.summary,
            "catalog": reader.catalog,
            "metric_definitions": reader.metric_definitions,
            "scalar_names": reader.scalar_names,
            "histogram_names": reader.histogram_names,
            "kde_names": reader.kde_names,
            "media_names": reader.media_names,
            "table_names": reader.table_names,
            "tensor_names": reader.tensor_names,
            "storage": {
                "run_db_bytes": run_db.stat().st_size if run_db.exists() else 0,
                "blobs_db_bytes": blobs_db.stat().st_size if blobs_db.exists() else 0,
            },
        }

    def batch_run_meta(self, project: str, run_ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for rid in run_ids:
            try:
                out[rid] = self.get_run_detail(project, rid)
            except RunNotFoundError:
                continue
        return out

    def patch_run(self, project: str, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        proj = _validate_segment(project, "project")
        rid = _validate_segment(run_id, "run_id")
        reader = self._get_reader(proj, rid)
        run_dir = reader.run_dir
        manifest_path = run_dir / "run.json"
        if not manifest_path.exists():
            raise RunNotFoundError(f"Manifest run.json missing for {proj}/{rid}")

        info = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "name" in patch and patch["name"] is not None:
            info["name"] = str(patch["name"])
        if "group" in patch:
            info["group"] = None if patch["group"] in (None, "") else str(patch["group"])
        if "notes" in patch and patch["notes"] is not None:
            info["notes"] = str(patch["notes"])
        if "tags" in patch and isinstance(patch["tags"], list):
            info["tags"] = [str(t) for t in patch["tags"] if str(t).strip()]
        info["updated_at_ms"] = int(time.time() * 1000)

        tmp_path = run_dir / "run.json.tmp"
        tmp_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        tmp_path.replace(manifest_path)
        self._catalog.refresh()
        return info

    def delete_run(self, project: str, run_id: str) -> bool:
        proj = _validate_segment(project, "project")
        rid = _validate_segment(run_id, "run_id")
        run_dir = self._resolve_run_dir(proj, rid)
        if not run_dir.resolve().is_relative_to(self.data_dir):
            raise ConfigError("Refusing to delete path outside data_dir")
        self._evict_reader(proj, rid)
        shutil.rmtree(run_dir)
        self._catalog.refresh()
        return True

    def scalar_page_frame(
        self,
        project: str,
        run_id: str,
        metric: str,
        *,
        before: int | None = None,
        after: int | None = None,
        limit: int = 8192,
        x_dtype: Literal["f64", "i64"] = "f64",
    ) -> bytes:
        reader = self._get_reader(project, run_id)
        return reader.scalar_page_frame(
            run_id,
            metric,
            before=before,
            after=after,
            limit=max(1, min(limit, 262_144)),
            x_dtype=x_dtype,
        )

    async def stream_scalars_chunks(
        self,
        project: str,
        run_ids: list[str],
        metrics: list[str],
        *,
        x_dtype: Literal["f64", "i64"] = "f64",
        first_chunk: int = 8192,
        max_chunk: int = 65536,
    ) -> AsyncIterator[bytes]:
        """Yield newest-first aligned binary chunk frames round-robin across (run, metric)."""
        first = max(256, min(first_chunk, 65_536))
        cap = max(first, min(max_chunk, 262_144))

        def _build_plans() -> list[tuple[str, str, RunReader, list[tuple[int, int]]]]:
            plans: list[tuple[str, str, RunReader, list[tuple[int, int]]]] = []
            for rid in run_ids:
                try:
                    reader = self._get_reader(project, rid)
                except RunNotFoundError:
                    continue
                for metric in metrics:
                    ranges = reader.scalar_newest_first_plan(
                        metric,
                        first_chunk=first,
                        max_chunk=cap,
                    )
                    if ranges:
                        plans.append((rid, metric, reader, ranges))
            return plans

        plans = await asyncio.to_thread(_build_plans)
        if not plans:
            return

        max_rounds = max(len(p[3]) for p in plans)
        for round_idx in range(max_rounds):
            for rid, metric, reader, ranges in plans:
                if round_idx >= len(ranges):
                    continue
                start, end = ranges[round_idx]
                frame = await asyncio.to_thread(
                    reader.scalar_range_frame,
                    rid,
                    metric,
                    start,
                    end,
                    x_dtype=x_dtype,
                )
                yield frame

    def media_events(
        self,
        project: str,
        run_id: str,
        name: str,
        *,
        before_id: int | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.media(name, before_id=before_id, limit=max(1, min(limit, 200)))

    def get_blob(self, project: str, run_id: str, key: str) -> tuple[bytes, str] | None:
        _validate_segment(key, "blob_key")
        reader = self._get_reader(project, run_id)
        data = reader.blob(key)
        if data is None:
            return None
        return data, _mime_for_blob_key(key)

    def tables(
        self,
        project: str,
        run_id: str,
        name: str,
        *,
        before_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.tables(name, before_id=before_id, limit=max(1, min(limit, 50)))

    def distributions_page(
        self,
        project: str,
        run_id: str,
        kind: Literal["histogram", "kde"],
        name: str,
        *,
        before: int | None = None,
        limit: int = 64,
    ) -> dict[str, Any]:
        reader = self._get_reader(project, run_id)
        cat = reader.catalog
        if kind == "histogram":
            meta = cat.get("histograms", {}).get(name)
            total = int(meta["count"]) if meta else 0
            end = total if before is None else min(before, total)
            start = max(0, end - max(1, min(limit, 256)))
            records = reader.histograms(name, start=start, count=end - start) if end > start else []
        else:
            meta = cat.get("kdes", {}).get(name)
            total = int(meta["count"]) if meta else 0
            end = total if before is None else min(before, total)
            start = max(0, end - max(1, min(limit, 256)))
            records = reader.kdes(name, start=start, count=end - start) if end > start else []

        # Return newest-first so UI renders latest distributions immediately.
        records.reverse()
        return {
            "kind": kind,
            "name": name,
            "total": total,
            "start": start,
            "end": end,
            "done": start == 0,
            "next_before": start if start > 0 else None,
            "items": records,
        }

    def tensors(
        self,
        project: str,
        run_id: str,
        name: str,
        *,
        before_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.tensors(name, before_id=before_id, limit=max(1, min(limit, 100)))

    def artifacts(
        self,
        project: str,
        run_id: str,
        *,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.artifacts(before_id=before_id, limit=max(1, min(limit, 200)))

    def alerts(
        self,
        project: str,
        run_id: str,
        *,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.alerts(before_id=before_id, limit=max(1, min(limit, 200)))

    def logs_tail(
        self,
        project: str,
        run_id: str,
        *,
        before_id: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.logs(before_id=before_id, limit=max(1, min(limit, 1000)))

    def logs_search(
        self,
        project: str,
        run_id: str,
        query: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        reader = self._get_reader(project, run_id)
        return reader.search_logs(query, limit=max(1, min(limit, 200)))

    async def sse_project_events(
        self,
        project: str,
        *,
        poll_interval: float = 1.5,
    ) -> AsyncIterator[str]:
        """Yield SSE `run_update` frames whenever runs in `project` change."""
        _validate_segment(project, "project")
        prev_watermarks: dict[str, tuple[str, int, int]] = {}
        yield f"event: hello\ndata: {json.dumps({'project': project})}\n\n"
        while True:
            runs = await asyncio.to_thread(self.list_runs, project, refresh=True)
            changed: list[dict[str, Any]] = []
            seen: set[str] = set()
            for r in runs:
                rid = r["run_id"]
                seen.add(rid)
                wm = (r["state"], r["last_step"], r["updated_at_ms"])
                if prev_watermarks.get(rid) != wm:
                    prev_watermarks[rid] = wm
                    changed.append(
                        {
                            "run_id": rid,
                            "state": r["state"],
                            "last_step": r["last_step"],
                            "last_global_step": r["last_global_step"],
                            "updated_at_ms": r["updated_at_ms"],
                        }
                    )
            removed = [rid for rid in list(prev_watermarks) if rid not in seen]
            for rid in removed:
                prev_watermarks.pop(rid, None)

            if changed or removed:
                payload = json.dumps({"project": project, "changed": changed, "removed": removed})
                yield f"event: runs\ndata: {payload}\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(poll_interval)
