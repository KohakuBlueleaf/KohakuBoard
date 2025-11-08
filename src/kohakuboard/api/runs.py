"""Run data access API endpoints

Unified API for accessing run data (scalars, media, tables, histograms).
Works in both local and remote modes with project-based organization.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from kohakuvault import KVault

from kohakuboard.utils.board_reader import BoardReader, DEFAULT_LOCAL_PROJECT
from kohakuboard.utils.run_id import sanitize_annotation
from kohakuboard.config import cfg
from kohakuboard.logger import logger_api


router = APIRouter()


class BatchSummaryRequest(BaseModel):
    run_ids: list[str]


class BatchScalarsRequest(BaseModel):
    run_ids: list[str]
    metrics: list[str]


class RunUpdateRequest(BaseModel):
    name: str | None = None
    annotation: str | None = None


def get_run_path(project: str, run_id: str) -> Path:
    """Resolve run path in local mode."""
    base_dir = Path(cfg.app.board_data_dir)

    if project == DEFAULT_LOCAL_PROJECT:
        run_path = base_dir / project / run_id
        if not run_path.exists():
            run_path = base_dir / run_id
    else:
        run_path = base_dir / project / run_id

    if not run_path.exists():
        raise HTTPException(404, detail={"error": "Run not found"})

    return run_path


@router.get("/projects/{project}/runs/{run_id}/status")
async def get_run_status(project: str, run_id: str):
    """Get run status with latest update timestamp

    Returns minimal info for polling (last update time, row counts)
    """
    run_path = get_run_path(project, run_id)

    # Check metadata for creation time
    metadata_file = run_path / "metadata.json"
    if metadata_file.exists():
        # Use asyncio.to_thread to avoid blocking
        def read_metadata():
            with open(metadata_file, "r") as f:
                return json.load(f)

        metadata = await asyncio.to_thread(read_metadata)
    else:
        metadata = {}

    # Get row count and last update from storage
    metrics_count = 0
    last_updated = metadata.get("created_at")

    try:
        reader = BoardReader(run_path)

        # Check if hybrid backend (has get_latest_step method)
        if hasattr(reader, "get_latest_step"):
            # Hybrid backend - get latest from steps table
            latest_step_info = reader.get_latest_step()
            if latest_step_info:
                metrics_count = latest_step_info.get("step", 0) + 1  # step count
                # Convert timestamp ms to ISO string
                ts_ms = latest_step_info.get("timestamp")
                if ts_ms:
                    last_updated = datetime.fromtimestamp(
                        ts_ms / 1000, tz=timezone.utc
                    ).isoformat()

    except Exception as e:
        logger_api.warning(f"Failed to get status: {e}")

    return {
        "run_id": run_id,
        "project": project,
        "metrics_count": metrics_count,
        "last_updated": last_updated,
    }


@router.patch("/projects/{project}/runs/{run_id}")
async def update_run(
    project: str,
    run_id: str,
    payload: RunUpdateRequest,
):
    """Update run metadata (local mode only)."""

    if cfg.app.mode != "local":
        raise HTTPException(
            405, detail={"error": "Updating runs is only supported in local mode"}
        )

    if payload.name is None and payload.annotation is None:
        raise HTTPException(400, detail={"error": "No update fields provided"})

    run_path = get_run_path(project, run_id)
    metadata_file = run_path / "metadata.json"

    if not metadata_file.exists():
        raise HTTPException(404, detail={"error": "metadata.json not found"})

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    changed = False
    response_data = {
        "run_id": metadata.get("run_id", run_id),
        "name": metadata.get("name", metadata.get("run_id", run_id)),
        "finished_at": metadata.get("finished_at"),
    }

    if payload.name is not None:
        new_name = payload.name.strip()
        if not new_name:
            raise HTTPException(400, detail={"error": "Run name cannot be empty"})
        metadata["name"] = new_name
        response_data["name"] = new_name
        changed = True

    if payload.annotation is not None:
        finished_at = metadata.get("finished_at")
        if not finished_at:
            raise HTTPException(
                400,
                detail={
                    "error": "Annotation can only be changed after the run is finished"
                },
            )

        sanitized = sanitize_annotation(payload.annotation)
        if not sanitized:
            raise HTTPException(
                400,
                detail={
                    "error": "Annotation must contain letters, numbers, '-' or '_'"
                },
            )

        current_annotation = metadata.get("run_id", run_path.name)
        if sanitized != current_annotation:
            target_path = run_path.parent / sanitized
            if target_path.exists():
                raise HTTPException(
                    409,
                    detail={
                        "error": f"Annotation '{sanitized}' already exists in project '{project}'"
                    },
                )
            run_path.rename(target_path)
            run_path = target_path

        metadata["run_id"] = sanitized
        metadata["board_id"] = sanitized
        response_data["run_id"] = sanitized
        changed = True

    if not changed:
        return response_data

    with open(run_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return response_data


@router.get("/projects/{project}/runs/{run_id}/summary")
async def get_run_summary(
    project: str,
    run_id: str,
):
    """Get run summary with metadata and available data

    Args:
        project: Project name
        run_id: Run ID
        current_user: Current user (optional)

    Returns:
        dict: Run summary with metadata, counts, available metrics/media/tables
        Same format as experiments API for compatibility
    """
    logger_api.info(f"Fetching summary for {project}/{run_id}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    summary = reader.get_summary()

    # Return in same format as experiments API for frontend compatibility
    metadata = summary["metadata"]

    return {
        "experiment_id": run_id,  # For compatibility with ConfigurableChartCard
        "project": project,
        "run_id": run_id,
        "experiment_info": {
            "id": run_id,
            "name": metadata.get("name", run_id),
            "description": f"Config: {metadata.get('config', {})}",
            "status": "completed",
            "total_steps": summary["metrics_count"],
            "duration": "N/A",
            "created_at": metadata.get("created_at", ""),
        },
        "total_steps": summary["metrics_count"],
        "available_data": {
            "scalars": summary["available_metrics"],
            "media": summary["available_media"],
            "tables": summary["available_tables"],
            "histograms": summary["available_histograms"],
        },
    }


@router.get("/projects/{project}/runs/{run_id}/metadata")
async def get_run_metadata(
    project: str,
    run_id: str,
):
    """Get run metadata"""
    logger_api.info(f"Fetching metadata for {project}/{run_id}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    metadata = reader.get_metadata()

    return metadata


@router.get("/projects/{project}/runs/{run_id}/scalars")
async def get_available_scalars(
    project: str,
    run_id: str,
):
    """Get list of available scalar metrics"""
    logger_api.info(f"Fetching available scalars for {project}/{run_id}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    metrics = reader.get_available_metrics()

    return {"metrics": metrics}


@router.get("/projects/{project}/runs/{run_id}/scalars/{metric:path}")
async def get_scalar_data(
    project: str,
    run_id: str,
    metric: str,
    limit: int | None = Query(None, description="Maximum number of data points"),
):
    """Get scalar data for a specific metric

    Note: metric can contain slashes (e.g., "train/loss")
    FastAPI path parameter automatically URL-decodes it
    """
    logger_api.info(f"Fetching scalar data for {project}/{run_id}/{metric}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    data = reader.get_scalar_data(metric, limit=limit)

    # data is now columnar format: {steps: [], global_steps: [], timestamps: [], values: []}
    return {"metric": metric, **data}


@router.get("/projects/{project}/runs/{run_id}/media")
async def get_available_media(
    project: str,
    run_id: str,
):
    """Get list of available media log names"""
    logger_api.info(f"Fetching available media for {project}/{run_id}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    media_names = reader.get_available_media_names()

    return {"media": media_names}


@router.get("/projects/{project}/runs/{run_id}/media/{name:path}")
async def get_media_data(
    project: str,
    run_id: str,
    name: str,
    limit: int | None = Query(None, description="Maximum number of entries"),
):
    """Get media data for a specific log name"""
    logger_api.info(f"Fetching media data for {project}/{run_id}/{name}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    data = reader.get_media_data(name, limit=limit)

    # Transform to same format as experiments API
    media_entries = []
    for entry in data:
        media_entries.append(
            {
                "name": entry.get("media_id", ""),
                "step": entry.get("step", 0),
                "type": entry.get("type", "image"),
                "url": f"/api/projects/{project}/runs/{run_id}/media/files/{entry.get('filename', '')}",
                "caption": entry.get("caption", ""),
                "width": entry.get("width"),
                "height": entry.get("height"),
            }
        )

    return {"experiment_id": run_id, "media_name": name, "data": media_entries}


@router.get("/projects/{project}/runs/{run_id}/media/files/{filename}")
async def get_media_file(
    project: str,
    run_id: str,
    filename: str,
):
    """Serve media file (image/video/audio) from SQLite KV storage"""
    logger_api.info(f"Serving media file: {project}/{run_id}/{filename}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    # Basic path safety
    if "/" in filename or ".." in filename:
        raise HTTPException(400, detail={"error": "Invalid media filename"})

    media_data = reader.get_media_data(filename)

    if not media_data:
        kv_path = run_path / "media" / "blobs.db"
        if kv_path.exists():
            try:
                media_kv = KVault(str(kv_path))
                try:
                    media_data = media_kv.get(filename)
                finally:
                    media_kv.close()
            except Exception as exc:
                logger_api.warning(
                    f"KVault fallback failed for {run_id}/{filename}: {exc}"
                )

    if not media_data:
        raise HTTPException(404, detail={"error": "Media file not found"})

    # Determine media type from extension
    suffix = Path(filename).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
    }

    media_type = media_types.get(suffix, "application/octet-stream")

    return Response(
        content=media_data,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.get("/projects/{project}/runs/{run_id}/tables")
async def get_available_tables(
    project: str,
    run_id: str,
):
    """Get list of available table log names"""
    logger_api.info(f"Fetching available tables for {project}/{run_id}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    table_names = reader.get_available_table_names()

    return {"tables": table_names}


@router.get("/projects/{project}/runs/{run_id}/tables/{name:path}")
async def get_table_data(
    project: str,
    run_id: str,
    name: str,
    limit: int | None = Query(None, description="Maximum number of entries"),
):
    """Get table data for a specific log name"""
    logger_api.info(f"Fetching table data for {project}/{run_id}/{name}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    data = reader.get_table_data(name, limit=limit)

    return {"experiment_id": run_id, "table_name": name, "data": data}


@router.get("/projects/{project}/runs/{run_id}/histograms")
async def get_available_histograms(
    project: str,
    run_id: str,
):
    """Get list of available histogram log names"""
    logger_api.info(f"Fetching available histograms for {project}/{run_id}")

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    histogram_names = reader.get_available_histogram_names()

    return {"histograms": histogram_names}


@router.get("/projects/{project}/runs/{run_id}/histograms/{name:path}")
async def get_histogram_data(
    project: str,
    run_id: str,
    name: str,
    limit: int | None = Query(None, description="Maximum number of entries"),
    bins: int | None = Query(
        None,
        ge=8,
        le=4096,
        description="Override bin count when sampling KDE entries",
    ),
    range_min: float | None = Query(
        None, description="Override minimum value for KDE resampling"
    ),
    range_max: float | None = Query(
        None, description="Override maximum value for KDE resampling"
    ),
):
    """Get histogram data for a specific log name"""
    logger_api.info(f"Fetching histogram data for {project}/{run_id}/{name}")

    if range_min is not None and range_max is not None and range_min >= range_max:
        raise HTTPException(400, detail={"error": "range_min must be < range_max"})

    run_path = get_run_path(project, run_id)
    reader = BoardReader(run_path)
    data = reader.get_histogram_data(
        name,
        limit=limit,
        bins=bins,
        range_min=range_min,
        range_max=range_max,
    )

    return {"experiment_id": run_id, "histogram_name": name, "data": data}


def should_include_metric(metric_name: str) -> bool:
    """Filter out params/ and gradients/ metrics."""
    return not metric_name.startswith("params/") and not metric_name.startswith(
        "gradients/"
    )


@router.post("/projects/{project}/runs/batch/summary")
async def batch_get_run_summaries(
    project: str,
    batch_request: BatchSummaryRequest = Body(...),
):
    """Batch fetch summaries for multiple runs.

    Args:
        project: Project name
        batch_request: List of run IDs to fetch

    Returns:
        dict: Map of run_id -> summary data (with params/gradients filtered)
    """
    logger_api.info(
        f"Batch fetching summaries for {len(batch_request.run_ids)} runs in {project}"
    )

    async def fetch_one_summary(run_id: str) -> tuple[str, dict | None]:
        try:
            run_path = get_run_path(project, run_id)
            reader = BoardReader(run_path)

            def get_summary_sync():
                return reader.get_summary()

            summary = await asyncio.to_thread(get_summary_sync)

            # Filter out params/ and gradients/ from all metric types
            filtered_summary = summary.copy()
            filtered_summary["available_metrics"] = [
                m for m in summary["available_metrics"] if should_include_metric(m)
            ]
            filtered_summary["available_media"] = [
                m for m in summary["available_media"] if should_include_metric(m)
            ]
            filtered_summary["available_tables"] = [
                m for m in summary["available_tables"] if should_include_metric(m)
            ]
            filtered_summary["available_histograms"] = [
                m for m in summary["available_histograms"] if should_include_metric(m)
            ]

            # Return in same format as single summary endpoint
            metadata = filtered_summary["metadata"]
            return run_id, {
                "experiment_id": run_id,
                "project": project,
                "run_id": run_id,
                "experiment_info": {
                    "id": run_id,
                    "name": metadata.get("name", run_id),
                    "description": f"Config: {metadata.get('config', {})}",
                    "status": "completed",
                    "total_steps": filtered_summary["metrics_count"],
                    "duration": "N/A",
                    "created_at": metadata.get("created_at", ""),
                },
                "total_steps": filtered_summary["metrics_count"],
                "available_data": {
                    "scalars": filtered_summary["available_metrics"],
                    "media": filtered_summary["available_media"],
                    "tables": filtered_summary["available_tables"],
                    "histograms": filtered_summary["available_histograms"],
                },
            }
        except Exception as e:
            logger_api.warning(f"Failed to fetch summary for {run_id}: {e}")
            return run_id, None

    # Fetch all summaries concurrently
    results = await asyncio.gather(
        *[fetch_one_summary(run_id) for run_id in batch_request.run_ids]
    )

    # Build result map (exclude None values)
    summaries = {run_id: summary for run_id, summary in results if summary is not None}

    logger_api.info(
        f"Successfully fetched {len(summaries)}/{len(batch_request.run_ids)} summaries"
    )

    return summaries


@router.post("/projects/{project}/runs/batch/scalars")
async def batch_get_scalar_data(
    project: str,
    body: BatchScalarsRequest = Body(...),
):
    """Batch fetch scalar data for multiple runs and metrics.

    Args:
        project: Project name
        body: List of run IDs and metrics to fetch

    Returns:
        dict: Nested map of run_id -> metric -> data
    """
    logger_api.info(
        f"Batch fetching {len(body.metrics)} metrics for {len(body.run_ids)} runs in {project}"
    )

    # Filter out params/ and gradients/ metrics
    filtered_metrics = [m for m in body.metrics if should_include_metric(m)]

    if not filtered_metrics:
        logger_api.warning("No valid metrics after filtering params/gradients")
        return {}

    async def fetch_one_metric(
        run_id: str, metric: str
    ) -> tuple[str, str, dict | None]:
        try:
            run_path = get_run_path(project, run_id)
            reader = BoardReader(run_path)

            def get_scalar_sync():
                data = reader.get_scalar_data(metric, limit=None)
                logger_api.debug(
                    f"Fetched {metric} for {run_id}: "
                    f"steps={len(data.get('steps', []))}, "
                    f"values={len(data.get('values', []))}"
                )
                return data

            data = await asyncio.to_thread(get_scalar_sync)
            return run_id, metric, data
        except Exception as e:
            logger_api.warning(f"Failed to fetch {metric} for {run_id}: {e}")
            return run_id, metric, None

    # Fetch all combinations concurrently
    tasks = []
    for run_id in body.run_ids:
        for metric in filtered_metrics:
            tasks.append(fetch_one_metric(run_id, metric))

    results = await asyncio.gather(*tasks)

    # Build nested result map
    scalar_data: dict[str, dict[str, dict]] = {}
    for run_id, metric, data in results:
        if data is not None:
            if run_id not in scalar_data:
                scalar_data[run_id] = {}
            scalar_data[run_id][metric] = data

    logger_api.info(f"Successfully fetched data for {len(scalar_data)} runs")

    return scalar_data
