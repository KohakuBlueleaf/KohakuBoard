"""Run data access API endpoints

Unified API for accessing run data (scalars, media, tables, histograms).
Works in both local and remote modes with project-based organization.
"""

import asyncio
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from kohakuboard.utils.board_reader import BoardReader, DEFAULT_LOCAL_PROJECT
from kohakuboard.utils.run_id import (
    find_run_dir_by_id,
    split_run_dir_name,
    build_run_dir_name,
)
from kohakuboard_server.auth import get_optional_user
from kohakuboard_server.auth.permissions import check_board_read_permission
from kohakuboard_server.config import cfg
from kohakuboard_server.db import Board, User
from kohakuboard_server.logger import logger_api

router = APIRouter()


class BatchSummaryRequest(BaseModel):
    run_ids: list[str]


class BatchScalarsRequest(BaseModel):
    run_ids: list[str]
    metrics: list[str]


def _resolve_run_path(
    project: str, run_id: str, current_user: User | None
) -> tuple[Path, Board | None]:
    base_dir = Path(cfg.app.board_data_dir)

    if cfg.app.mode == "local":
        checked = set()

        def iter_candidate_dirs():
            primary = base_dir / project
            yield primary
            if project == DEFAULT_LOCAL_PROJECT:
                yield base_dir

            users_root = base_dir / "users"
            if users_root.exists():
                for owner_dir in users_root.iterdir():
                    if not owner_dir.is_dir():
                        continue
                    yield owner_dir / project

        for candidate_dir in iter_candidate_dirs():
            key = str(candidate_dir.resolve(strict=False))
            if key in checked:
                continue
            checked.add(key)

            if not candidate_dir.exists():
                continue

            run_path = find_run_dir_by_id(candidate_dir, run_id)
            if run_path:
                return run_path, None

        raise HTTPException(404, detail={"error": "Run not found"})

    else:  # remote mode
        # Get board from DB (don't filter by owner - check permissions instead)
        board = Board.get_or_none(
            (Board.project_name == project) & (Board.run_id == run_id)
        )
        if not board:
            raise HTTPException(404, detail={"error": "Run not found"})

        # Check read permission (works for owner, org members, and public boards)
        check_board_read_permission(board, current_user)

        return base_dir / board.storage_path, board


async def get_run_path(
    project: str, run_id: str, current_user: User | None
) -> tuple[Path, Board | None]:
    """Async wrapper that resolves run path without blocking the event loop."""
    return await asyncio.to_thread(_resolve_run_path, project, run_id, current_user)


async def _call_in_thread(func, *args, **kwargs):
    """Run blocking storage operation in a thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


@router.get("/projects/{project}/runs/{run_id}/status")
async def get_run_status(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get run status with latest update timestamp"""
    logger_api.info(f"start run_status {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    folder_run_id, annotation = split_run_dir_name(run_path.name)

    # Check metadata for creation time without blocking loop
    metadata_file = run_path / "metadata.json"

    def read_metadata():
        if not metadata_file.exists():
            return {}
        with open(metadata_file, "r") as f:
            return json.load(f)

    metadata = await asyncio.to_thread(read_metadata)

    # Get row count and last update from storage
    metrics_count = 0
    last_updated = metadata.get("created_at")

    try:
        reader = BoardReader(run_path)

        # Check if hybrid backend (has get_latest_step method)
        if hasattr(reader, "get_latest_step"):
            # Hybrid backend - get latest from steps table
            latest_step_info = await _call_in_thread(reader.get_latest_step)
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

    result = {
        "run_id": folder_run_id,
        "project": project,
        "metrics_count": metrics_count,
        "last_updated": last_updated,
        "annotation": annotation,
    }
    return result


@router.get("/projects/{project}/runs/{run_id}/summary")
async def get_run_summary(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
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
    logger_api.info(f"start run_summary {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    summary = await _call_in_thread(reader.get_summary)

    folder_run_id, annotation = split_run_dir_name(run_path.name)
    metadata = dict(summary.get("metadata") or {})
    metadata.setdefault("run_id", folder_run_id)
    metadata["annotation"] = annotation
    summary["metadata"] = metadata

    result = {
        "experiment_id": folder_run_id,  # For compatibility with ConfigurableChartCard
        "project": project,
        "run_id": folder_run_id,
        "experiment_info": {
            "id": folder_run_id,
            "name": metadata.get("name", folder_run_id),
            "description": f"Config: {metadata.get('config', {})}",
            "status": "completed",
            "total_steps": summary["metrics_count"],
            "duration": "N/A",
            "created_at": metadata.get("created_at", ""),
            "annotation": annotation,
        },
        "total_steps": summary["metrics_count"],
        "available_data": {
            "scalars": summary["available_metrics"],
            "media": summary["available_media"],
            "tables": summary["available_tables"],
            "histograms": summary["available_histograms"],
            "tensors": summary.get("available_tensors", []),
            "kernel_density": summary.get("available_kernel_density", []),
        },
    }
    return result


@router.get("/projects/{project}/runs/{run_id}/metadata")
async def get_run_metadata(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get run metadata"""
    logger_api.info(f"start run_metadata {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    metadata = await _call_in_thread(reader.get_metadata)
    metadata = dict(metadata or {})
    folder_run_id, annotation = split_run_dir_name(run_path.name)
    metadata.setdefault("run_id", folder_run_id)
    metadata["annotation"] = annotation
    return metadata


@router.get("/projects/{project}/runs/{run_id}/scalars")
async def get_available_scalars(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get list of available scalar metrics"""
    logger_api.info(f"start scalars_list {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    metrics = await _call_in_thread(reader.get_available_metrics)
    return {"metrics": metrics}


@router.get("/projects/{project}/runs/{run_id}/scalars/{metric:path}")
async def get_scalar_data(
    project: str,
    run_id: str,
    metric: str,
    limit: int | None = Query(None, description="Maximum number of data points"),
    current_user: User | None = Depends(get_optional_user),
):
    """Get scalar data for a specific metric

    Note: metric can contain slashes (e.g., "train/loss")
    FastAPI path parameter automatically URL-decodes it
    """
    logger_api.info(f"start scalar_data {project}/{run_id}/{metric}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    data = await _call_in_thread(reader.get_scalar_data, metric, limit=limit)

    # data is now columnar format: {steps: [], global_steps: [], timestamps: [], values: []}
    return {"metric": metric, **data}


@router.get("/projects/{project}/runs/{run_id}/media")
async def get_available_media(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get list of available media log names"""
    logger_api.info(f"start media_list {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    media_names = await _call_in_thread(reader.get_available_media_names)

    return {"media": media_names}


@router.get("/projects/{project}/runs/{run_id}/media/{name:path}")
async def get_media_data(
    project: str,
    run_id: str,
    name: str,
    limit: int | None = Query(None, description="Maximum number of entries"),
    current_user: User | None = Depends(get_optional_user),
):
    """Get media data for a specific log name"""
    logger_api.info(f"start media_data {project}/{run_id}/{name}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    data = await _call_in_thread(reader.get_media_data, name, limit=limit)

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
    current_user: User | None = Depends(get_optional_user),
):
    """Serve media file (image/video/audio) from SQLite KV storage"""
    logger_api.info(f"start media_file {project}/{run_id}/{filename}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    media_data = await _call_in_thread(reader.get_media_data, filename)

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

    response = Response(
        content=media_data,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
    return response


@router.get("/projects/{project}/runs/{run_id}/tables")
async def get_available_tables(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get list of available table log names"""
    logger_api.info(f"start tables_list {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    table_names = await _call_in_thread(reader.get_available_table_names)
    return {"tables": table_names}


@router.get("/projects/{project}/runs/{run_id}/tables/{name:path}")
async def get_table_data(
    project: str,
    run_id: str,
    name: str,
    limit: int | None = Query(None, description="Maximum number of entries"),
    current_user: User | None = Depends(get_optional_user),
):
    """Get table data for a specific log name"""
    logger_api.info(f"start table_data {project}/{run_id}/{name}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    data = await _call_in_thread(reader.get_table_data, name, limit=limit)
    return {"experiment_id": run_id, "table_name": name, "data": data}


@router.get("/projects/{project}/runs/{run_id}/histograms")
async def get_available_histograms(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get list of available histogram log names"""
    logger_api.info(f"start hist_list {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    histogram_names = await _call_in_thread(reader.get_available_histogram_names)
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
    current_user: User | None = Depends(get_optional_user),
):
    """Get histogram data for a specific log name"""
    logger_api.info(f"start hist_data {project}/{run_id}/{name}")
    if range_min is not None and range_max is not None and range_min >= range_max:
        raise HTTPException(400, detail={"error": "range_min must be < range_max"})

    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    data = await _call_in_thread(
        reader.get_histogram_data,
        name,
        limit=limit,
        bins=bins,
        range_min=range_min,
        range_max=range_max,
    )

    return {"experiment_id": run_id, "histogram_name": name, "data": data}


@router.get("/projects/{project}/runs/{run_id}/tensors")
async def get_available_tensors(
    project: str,
    run_id: str,
    current_user: User | None = Depends(get_optional_user),
):
    """Get list of available tensor log names."""
    logger_api.info(f"start tensor_list {project}/{run_id}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    tensor_names = await _call_in_thread(reader.get_available_tensor_names)
    return {"tensors": tensor_names}


@router.get("/projects/{project}/runs/{run_id}/tensors/{name:path}")
async def get_tensor_data(
    project: str,
    run_id: str,
    name: str,
    include_data: bool = Query(
        False, description="When true, include base64-encoded tensor payloads"
    ),
    current_user: User | None = Depends(get_optional_user),
):
    """Get tensor metadata (and optionally payload) for a specific log name."""
    logger_api.info(f"start tensor_data {project}/{run_id}/{name}")
    run_path, _ = await get_run_path(project, run_id, current_user)
    reader = BoardReader(run_path)
    entries = await _call_in_thread(
        reader.get_tensor_data, name, include_payload=include_data
    )

    for entry in entries:
        payload = entry.pop("payload", None)
        if include_data and payload is not None:
            entry["encoding"] = "npy_base64"
            entry["data_base64"] = base64.b64encode(payload).decode("ascii")

    return {"experiment_id": run_id, "tensor_name": name, "entries": entries}


def should_include_metric(metric_name: str) -> bool:
    """Filter out params/ and gradients/ metrics."""
    return not metric_name.startswith("params/") and not metric_name.startswith(
        "gradients/"
    )


@router.post("/projects/{project}/runs/batch/summary")
async def batch_get_run_summaries(
    project: str,
    batch_request: BatchSummaryRequest = Body(...),
    current_user: User | None = Depends(get_optional_user),
):
    """Batch fetch summaries for multiple runs.

    Args:
        project: Project name
        batch_request: List of run IDs to fetch

    Returns:
        dict: Map of run_id -> summary data (with params/gradients filtered)
    """
    logger_api.info(
        f"start batch_run_summaries {project} ({len(batch_request.run_ids)} runs)"
    )

    async def fetch_one_summary(run_id: str) -> tuple[str, dict | None]:
        try:
            run_path, _ = await get_run_path(project, run_id, current_user)
            reader = BoardReader(run_path)
            summary = reader.get_summary()

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
            filtered_summary["available_tensors"] = [
                m
                for m in summary.get("available_tensors", [])
                if should_include_metric(m)
            ]
            filtered_summary["available_kernel_density"] = [
                m
                for m in summary.get("available_kernel_density", [])
                if should_include_metric(m)
            ]

            # Return in same format as single summary endpoint
            folder_run_id, annotation = split_run_dir_name(run_path.name)
            metadata = dict(filtered_summary.get("metadata") or {})
            metadata.setdefault("run_id", folder_run_id)
            metadata["annotation"] = annotation
            filtered_summary["metadata"] = metadata

            return run_id, {
                "experiment_id": folder_run_id,
                "project": project,
                "run_id": folder_run_id,
                "experiment_info": {
                    "id": folder_run_id,
                    "name": metadata.get("name", folder_run_id),
                    "description": f"Config: {metadata.get('config', {})}",
                    "status": "completed",
                    "total_steps": filtered_summary["metrics_count"],
                    "duration": "N/A",
                    "created_at": metadata.get("created_at", ""),
                    "annotation": annotation,
                },
                "total_steps": filtered_summary["metrics_count"],
                "available_data": {
                    "scalars": filtered_summary["available_metrics"],
                    "media": filtered_summary["available_media"],
                    "tables": filtered_summary["available_tables"],
                    "histograms": filtered_summary["available_histograms"],
                    "tensors": filtered_summary.get("available_tensors", []),
                    "kernel_density": filtered_summary.get(
                        "available_kernel_density", []
                    ),
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
    current_user: User | None = Depends(get_optional_user),
):
    """Batch fetch scalar data for multiple runs and metrics.

    Args:
        project: Project name
        body: List of run IDs and metrics to fetch

    Returns:
        dict: Nested map of run_id -> metric -> data
    """
    logger_api.info(
        f"start batch_scalar_data {project} "
        f"({len(body.metrics)} metrics x {len(body.run_ids)} runs)"
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
            run_path, _ = await get_run_path(project, run_id, current_user)
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
