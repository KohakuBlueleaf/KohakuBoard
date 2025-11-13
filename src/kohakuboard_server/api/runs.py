"""Run data access API endpoints

Unified API for accessing run data (scalars, media, tables, histograms).
Works in both local and remote modes with project-based organization.
"""

import asyncio
import base64
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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


AXIS_LABELS = {
    "step": "Training Step",
    "global_step": "Global Step",
    "relative_walltime": "Relative Time (s)",
}


def _extract_axis_value(entry: dict[str, Any], axis_type: str) -> float | None:
    if axis_type == "step":
        return entry.get("step")
    if axis_type == "global_step":
        return entry.get("global_step")
    if axis_type == "relative_walltime":
        wall = entry.get("relative_walltime")
        if wall is not None:
            return wall
    return entry.get("step") or entry.get("global_step")


def _resample_counts(
    bins: list[float], counts: list[float], target_edges: np.ndarray
) -> np.ndarray:
    bins_arr = np.asarray(bins, dtype=np.float64)
    counts_arr = np.asarray(counts, dtype=np.float64)

    if bins_arr.size != counts_arr.size + 1:
        raise ValueError("bins/counts length mismatch")

    centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])
    widths = np.diff(bins_arr)
    safe_widths = np.clip(widths, 1e-9, None)
    densities = counts_arr / safe_widths

    target_centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    interp_density = np.interp(
        target_centers,
        centers,
        densities,
        left=0.0,
        right=0.0,
    )

    target_widths = np.diff(target_edges)
    return interp_density * target_widths


def _build_histogram_surface_payload(
    entries: list[dict[str, Any]],
    axis_type: str,
    normalize_mode: str,
    downsample: int,
    target_bins: int | None,
) -> dict[str, Any]:
    usable_entries = [
        entry for entry in entries if entry.get("bins") and entry.get("counts")
    ]

    if not usable_entries:
        raise ValueError("Histogram surface view requires precomputed bins and counts")

    requested_bins = (
        target_bins if target_bins is not None else len(usable_entries[0]["counts"])
    )
    requested_bins = max(8, min(int(requested_bins), 1024))

    canonical_min = min(float(entry["bins"][0]) for entry in usable_entries)
    canonical_max = max(float(entry["bins"][-1]) for entry in usable_entries)
    if canonical_min >= canonical_max:
        canonical_max = canonical_min + 1.0

    target_edges = np.linspace(
        canonical_min,
        canonical_max,
        requested_bins + 1,
        dtype=np.float32,
    )
    bin_centers = ((target_edges[:-1] + target_edges[1:]) / 2.0).tolist()

    stride = 1
    if downsample == -1:
        target_rows = 180
        stride = max(1, math.ceil(len(usable_entries) / target_rows))
    elif downsample and downsample > 1:
        stride = downsample

    sampled_entries = usable_entries[::stride]
    if not sampled_entries:
        sampled_entries = [usable_entries[-1]]
        stride = len(usable_entries)

    matrix_rows: list[np.ndarray] = []
    axis_values: list[float] = []
    steps: list[int | None] = []
    global_steps: list[int | None] = []
    axis_fallback_used = False

    for idx, entry in enumerate(sampled_entries):
        try:
            row = _resample_counts(entry["bins"], entry["counts"], target_edges)
        except ValueError as exc:
            logger_api.warning(
                f"Skipping malformed histogram entry for surface view: {exc}"
            )
            continue

        axis_value = _extract_axis_value(entry, axis_type)
        if axis_value is None:
            fallback_value = entry.get("step") or entry.get("global_step")
            if fallback_value is None:
                fallback_value = idx
            axis_value = fallback_value
            axis_fallback_used = True

        matrix_rows.append(row.astype(np.float32))
        axis_values.append(float(axis_value))
        steps.append(entry.get("step"))
        global_steps.append(entry.get("global_step"))

    if not matrix_rows:
        raise ValueError("No valid histogram entries after resampling")

    raw_matrix = np.vstack(matrix_rows)
    normalized_matrix = raw_matrix.copy()

    internal_mode = (normalize_mode or "none").replace("-", "_")
    if internal_mode == "per_step":
        row_max = np.max(normalized_matrix, axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        normalized_matrix = normalized_matrix / row_max
    elif internal_mode == "global":
        global_max = float(np.max(normalized_matrix))
        if global_max > 0:
            normalized_matrix = normalized_matrix / global_max
    else:
        internal_mode = "none"

    if internal_mode == "none":
        matrix_data = raw_matrix
    else:
        matrix_data = normalized_matrix

    resolved_axis_type = (
        "step" if axis_fallback_used and axis_type != "step" else axis_type
    )

    return {
        "axis_values": axis_values,
        "axis_type": resolved_axis_type,
        "axis_requested": axis_type,
        "axis_label": AXIS_LABELS.get(resolved_axis_type, "Training Step"),
        "axis_stats": {
            "min": float(min(axis_values)),
            "max": float(max(axis_values)),
            "count": len(axis_values),
        },
        "steps": steps,
        "global_steps": global_steps,
        "bin_edges": target_edges.tolist(),
        "bin_centers": bin_centers,
        "bin_range": {
            "min": float(target_edges[0]),
            "max": float(target_edges[-1]),
        },
        "matrix": matrix_data.tolist(),
        "raw_max_value": float(np.max(raw_matrix)),
        "downsample_rate": stride,
        "original_entries": len(usable_entries),
        "sampled_entries": len(matrix_rows),
        "normalize": normalize_mode,
    }


@router.get("/projects/{project}/runs/{run_id}/histograms/{name:path}/surface")
async def get_histogram_surface_data(
    project: str,
    run_id: str,
    name: str,
    axis: str = Query(
        "global_step",
        description="Axis for surface plot ('step', 'global_step', 'relative_walltime')",
    ),
    normalize: str = Query(
        "none",
        description="Normalization mode: 'per-step', 'global', or 'none'",
    ),
    downsample: int = Query(
        -1,
        ge=-1,
        description="Downsample stride (-1 = auto target ~180 entries)",
    ),
    bins: int | None = Query(
        None,
        ge=8,
        le=1024,
        description="Target number of bins for the surface grid",
    ),
    limit: int | None = Query(None, description="Maximum number of entries"),
    range_min: float | None = Query(
        None, description="Override minimum value for KDE resampling"
    ),
    range_max: float | None = Query(
        None, description="Override maximum value for KDE resampling"
    ),
    current_user: User | None = Depends(get_optional_user),
):
    """Get histogram surface-ready data for Plotly 3D rendering."""

    logger_api.info(f"start hist_surface {project}/{run_id}/{name}")

    if range_min is not None and range_max is not None and range_min >= range_max:
        raise HTTPException(400, detail={"error": "range_min must be < range_max"})

    allowed_axes = {"step", "global_step", "relative_walltime"}
    if axis not in allowed_axes:
        raise HTTPException(
            400, detail={"error": f"axis must be one of {sorted(allowed_axes)}"}
        )

    allowed_normalize = {"per-step", "global", "none"}
    if normalize not in allowed_normalize:
        raise HTTPException(
            400,
            detail={"error": f"normalize must be one of {sorted(allowed_normalize)}"},
        )

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

    if not data:
        raise HTTPException(404, detail={"error": "Histogram not found"})

    try:
        payload = _build_histogram_surface_payload(
            data,
            axis_type=axis,
            normalize_mode=normalize,
            downsample=downsample,
            target_bins=bins,
        )
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)})

    return {
        "experiment_id": run_id,
        "histogram_name": name,
        **payload,
    }


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
