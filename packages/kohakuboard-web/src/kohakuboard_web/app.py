"""FastAPI application factory for KohakuBoard Web (local and unified API + SPA)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from kohakuboard import SCHEMA_VERSION
from kohakuboard import __version__ as core_version
from kohakuboard.errors import ConfigError, KohakuBoardError, RunNotFoundError
from kohakuboard_web.access import LoopbackGuardMiddleware
from kohakuboard_web.service import BoardService

DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"


class RunPatchBody(BaseModel):
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


class BatchMetaBody(BaseModel):
    run_ids: list[str]


def _parse_range_header(range_header: str, total_size: int) -> tuple[int, int] | None:
    if not range_header.startswith("bytes="):
        return None
    spec = range_header[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    if start_s == "":
        suffix = int(end_s)
        start = max(0, total_size - suffix)
        end = total_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else total_size - 1
    if start < 0 or end < start or start >= total_size:
        return None
    return start, min(end, total_size - 1)


def create_app(
    data_dir: str | Path,
    *,
    static_dir: str | Path | None = None,
    enforce_loopback: bool = True,
    allow_testserver: bool = False,
) -> FastAPI:
    """Create the unified KohakuBoard FastAPI application."""
    service = BoardService(data_dir)
    resolved_static = Path(static_dir) if static_dir is not None else DEFAULT_STATIC_DIR

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="KohakuBoard", version=core_version, lifespan=lifespan)
    app.state.service = service

    if enforce_loopback:
        app.add_middleware(LoopbackGuardMiddleware, allow_testserver=allow_testserver)

    @app.exception_handler(RunNotFoundError)
    async def _not_found(_req: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ConfigError)
    async def _bad_req(_req: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KohakuBoardError)
    async def _board_err(_req: Request, exc: KohakuBoardError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=500)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/system/info")
    async def system_info() -> dict[str, Any]:
        return {
            "version": core_version,
            "schema_version": SCHEMA_VERSION,
            "mode": "local",
            "data_dir": str(service.data_dir),
            "ui_bundled": (resolved_static / "index.html").exists(),
        }

    @app.get("/api/projects")
    async def list_projects(refresh: bool = True) -> list[dict[str, Any]]:
        return await asyncio.to_thread(service.list_projects, refresh=refresh)

    @app.get("/api/projects/{project}/runs")
    async def list_runs(
        project: str,
        state: str | None = None,
        group: str | None = None,
        search: str | None = None,
        sort: str = "-updated_at_ms",
        limit: int | None = None,
        offset: int = 0,
        refresh: bool = True,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.list_runs,
            project,
            state=state,
            group=group,
            search=search,
            sort=sort,
            limit=limit,
            offset=offset,
            refresh=refresh,
        )

    @app.post("/api/projects/{project}/runs/batch/meta")
    async def batch_meta(project: str, body: BatchMetaBody) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(service.batch_run_meta, project, body.run_ids)

    @app.get("/api/projects/{project}/runs/scalars/stream")
    async def stream_scalars(
        project: str,
        runs: str = Query(..., description="Comma-separated run_ids"),
        metrics: str = Query(..., description="Comma-separated metric names"),
        x_dtype: Literal["f64", "i64"] = "f64",
        first_chunk: int = 8192,
        max_chunk: int = 65536,
    ) -> StreamingResponse:
        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
        if not run_ids or not metric_names:
            raise HTTPException(status_code=400, detail="Both 'runs' and 'metrics' are required")
        return StreamingResponse(
            service.stream_scalars_chunks(
                project,
                run_ids,
                metric_names,
                x_dtype=x_dtype,
                first_chunk=first_chunk,
                max_chunk=max_chunk,
            ),
            media_type="application/vnd.kobo.chunks",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/projects/{project}/events")
    async def project_events(project: str) -> StreamingResponse:
        return StreamingResponse(
            service.sse_project_events(project),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/projects/{project}/runs/{run_id}")
    async def get_run(project: str, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(service.get_run_detail, project, run_id)

    @app.patch("/api/projects/{project}/runs/{run_id}")
    async def patch_run(project: str, run_id: str, body: RunPatchBody) -> dict[str, Any]:
        return await asyncio.to_thread(
            service.patch_run,
            project,
            run_id,
            body.model_dump(exclude_unset=True),
        )

    @app.delete("/api/projects/{project}/runs/{run_id}")
    async def delete_run(project: str, run_id: str) -> dict[str, bool]:
        ok = await asyncio.to_thread(service.delete_run, project, run_id)
        return {"deleted": ok}

    @app.get("/api/projects/{project}/runs/{run_id}/scalars/{metric:path}/page")
    async def scalar_page(
        project: str,
        run_id: str,
        metric: str,
        before: int | None = None,
        after: int | None = None,
        limit: int = 8192,
        x_dtype: Literal["f64", "i64"] = "f64",
    ) -> Response:
        frame = await asyncio.to_thread(
            service.scalar_page_frame,
            project,
            run_id,
            metric,
            before=before,
            after=after,
            limit=limit,
            x_dtype=x_dtype,
        )
        return Response(
            content=frame,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/projects/{project}/runs/{run_id}/media/{name:path}")
    async def media_events(
        project: str,
        run_id: str,
        name: str,
        before_id: int | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.media_events,
            project,
            run_id,
            name,
            before_id=before_id,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/blobs/{key}")
    async def get_blob(project: str, run_id: str, key: str, request: Request) -> Response:
        found = await asyncio.to_thread(service.get_blob, project, run_id, key)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Blob '{key}' not found")
        data, mime = found
        etag = f'"{key}"'
        base_headers = {
            "ETag": etag,
            "Cache-Control": "public, max-age=31536000, immutable",
            "Accept-Ranges": "bytes",
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=base_headers)

        range_hdr = request.headers.get("range")
        if range_hdr:
            parsed = _parse_range_header(range_hdr, len(data))
            if parsed is None:
                return Response(
                    status_code=416,
                    headers={**base_headers, "Content-Range": f"bytes */{len(data)}"},
                )
            start, end = parsed
            chunk = data[start : end + 1]
            return Response(
                content=chunk,
                status_code=206,
                media_type=mime,
                headers={
                    **base_headers,
                    "Content-Range": f"bytes {start}-{end}/{len(data)}",
                },
            )
        return Response(content=data, media_type=mime, headers=base_headers)

    @app.get("/api/projects/{project}/runs/{run_id}/tables/{name:path}")
    async def tables(
        project: str,
        run_id: str,
        name: str,
        before_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.tables,
            project,
            run_id,
            name,
            before_id=before_id,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/distributions/{kind}/{name:path}/page")
    async def distributions_page(
        project: str,
        run_id: str,
        kind: Literal["histogram", "kde"],
        name: str,
        before: int | None = None,
        limit: int = 64,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            service.distributions_page,
            project,
            run_id,
            kind,
            name,
            before=before,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/tensors/{name:path}")
    async def tensors(
        project: str,
        run_id: str,
        name: str,
        before_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.tensors,
            project,
            run_id,
            name,
            before_id=before_id,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/artifacts")
    async def artifacts(
        project: str,
        run_id: str,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.artifacts,
            project,
            run_id,
            before_id=before_id,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/alerts")
    async def alerts(
        project: str,
        run_id: str,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.alerts,
            project,
            run_id,
            before_id=before_id,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/logs")
    async def logs_tail(
        project: str,
        run_id: str,
        before_id: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            service.logs_tail,
            project,
            run_id,
            before_id=before_id,
            limit=limit,
        )

    @app.get("/api/projects/{project}/runs/{run_id}/logs/search")
    async def logs_search(
        project: str,
        run_id: str,
        q: str = Query(..., min_length=1),
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(service.logs_search, project, run_id, q, limit=limit)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> Response:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        index_html = resolved_static / "index.html"
        if not index_html.exists():
            return JSONResponse(
                {
                    "status": "api-only",
                    "detail": "Bundled web UI not built yet. Run `npm run build` in apps/web.",
                },
                status_code=200,
            )
        candidate = (resolved_static / full_path).resolve()
        if (
            full_path
            and candidate.is_file()
            and candidate.is_relative_to(resolved_static.resolve())
        ):
            return FileResponse(candidate)
        return FileResponse(index_html)

    return app
