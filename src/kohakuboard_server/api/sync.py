"""Sync API endpoints for uploading boards to remote server"""

import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import orjson
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile

from kohakuboard_server.api.sync_models import (
    LogSyncRequest,
    LogSyncResponse,
    MediaUploadResponse,
    SyncRange,
)
from kohakuboard_server.auth import get_current_user
from kohakuboard.storage.hybrid import HybridStorage
from kohakuboard.storage.sqlite_kv import SQLiteKVStorage
from kohakuboard_server.config import cfg
from kohakuboard_server.db import Board, User
from kohakuboard_server.db_operations import get_organization, get_user_organization
from kohakuboard_server.logger import logger_api

router = APIRouter()


@router.post("/projects/{project_name}/sync")
async def sync_run(
    project_name: str,
    metadata: str = Form(...),
    duckdb_file: UploadFile = File(...),
    media_files: list[UploadFile] = File(default=[]),
    current_user: User = Depends(get_current_user),
):
    """Sync run to remote server (remote mode only)

    Uploads DuckDB file and media files to create or update a run.

    Args:
        project_name: Project name
        metadata: JSON string with run metadata
        duckdb_file: board.duckdb file
        media_files: List of media files
        current_user: Authenticated user

    Returns:
        dict: Sync result with run_id, URL, status

    Raises:
        HTTPException: 400 if mode is local, 401 if not authenticated
    """
    if cfg.app.mode != "remote":
        raise HTTPException(
            400,
            detail={"error": "Sync only available in remote mode"},
        )

    logger_api.info(
        f"Syncing run to project {project_name} for user {current_user.username}"
    )

    # Parse metadata
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(
            400,
            detail={"error": f"Invalid JSON metadata: {str(e)}"},
        )

    run_id = meta.get("run_id") or meta.get("board_id")
    if not run_id:
        raise HTTPException(
            400,
            detail={"error": "Missing run_id in metadata"},
        )

    name = meta.get("name", run_id)
    private = meta.get("private", True)
    config = meta.get("config", {})

    logger_api.info(f"Run ID: {run_id}, Name: {name}, Private: {private}")

    # Determine owner (support org/project format)
    owner = current_user
    if "/" in project_name:
        # Format: {org_name}/{project}
        org_name, actual_project = project_name.split("/", 1)
        org = get_organization(org_name)
        if not org:
            raise HTTPException(
                404, detail={"error": f"Organization '{org_name}' not found"}
            )

        # Check if user is member of org
        membership = get_user_organization(current_user, org)
        if not membership or membership.role not in ["member", "admin", "super-admin"]:
            raise HTTPException(
                403,
                detail={
                    "error": f"You don't have permission to sync to organization '{org_name}'"
                },
            )

        owner = org
        project_name = actual_project

    # Check if board exists
    existing = Board.get_or_none(
        (Board.owner == owner)
        & (Board.project_name == project_name)
        & (Board.run_id == run_id)
    )

    if existing:
        # Update existing
        board = existing
        logger_api.info(f"Updating existing board: {board.id}")
    else:
        # Create new
        storage_path = f"users/{owner.username}/{project_name}/{run_id}"
        board = Board.create(
            run_id=run_id,
            name=name,
            project_name=project_name,
            owner=owner,
            private=private,
            config=json.dumps(config),
            storage_path=storage_path,
            backend="duckdb",
        )
        logger_api.info(f"Created new board: {board.id} (owner: {owner.username})")

    # Save files to filesystem
    base_dir = Path(cfg.app.board_data_dir)
    run_dir = base_dir / board.storage_path
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save DuckDB file
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    duckdb_path = data_dir / "board.duckdb"

    logger_api.info(f"Saving DuckDB file to: {duckdb_path}")
    content = await duckdb_file.read()
    async with aiofiles.open(duckdb_path, "wb") as f:
        await f.write(content)
    total_size = len(content)

    # Save media files to SQLite KV
    media_kv_path = run_dir / "media" / "blobs.db"
    media_kv = SQLiteKVStorage(media_kv_path, readonly=False, logger=logger_api)

    logger_api.info(f"Saving {len(media_files)} media files to SQLite KV")
    try:
        for media_file in media_files:
            key = media_file.filename  # Key is {media_hash}.{format}
            logger_api.debug(f"Saving media to SQLite KV: {media_file.filename}")
            content = await media_file.read()
            media_kv.put(key, content)
            total_size += len(content)
    finally:
        media_kv.close()

    # Save metadata.json
    metadata_path = run_dir / "metadata.json"
    async with aiofiles.open(metadata_path, "w") as f:
        await f.write(json.dumps(meta, indent=2))

    # Update Board record
    board.total_size_bytes = total_size
    board.last_synced_at = datetime.now(timezone.utc)
    board.updated_at = datetime.now(timezone.utc)
    board.save()

    logger_api.info(f"Sync completed: {run_id} ({total_size} bytes)")

    return {
        "run_id": board.run_id,
        "project": project_name,
        "url": f"/projects/{project_name}/runs/{run_id}",
        "status": "synced",
        "total_size": total_size,
    }


# ============================================================================
# New Incremental Sync Endpoints (v0.3.0+)
# ============================================================================


def _get_or_create_board(
    project_name: str, run_id: str, current_user: User
) -> tuple[Board, Path]:
    """Get existing board or create new one

    Args:
        project_name: Project name (may include org prefix)
        run_id: Run ID
        current_user: Authenticated user

    Returns:
        Tuple of (Board, storage_path)
    """
    # Determine owner (support org/project format)
    owner = current_user
    if "/" in project_name:
        # Format: {org_name}/{project}
        org_name, actual_project = project_name.split("/", 1)
        org = get_organization(org_name)
        if not org:
            raise HTTPException(
                404, detail={"error": f"Organization '{org_name}' not found"}
            )

        # Check if user is member of org
        membership = get_user_organization(current_user, org)
        if not membership or membership.role not in ["member", "admin", "super-admin"]:
            raise HTTPException(
                403,
                detail={
                    "error": f"You don't have permission to sync to organization '{org_name}'"
                },
            )

        owner = org
        project_name = actual_project

    # Check if board exists
    existing = Board.get_or_none(
        (Board.owner == owner)
        & (Board.project_name == project_name)
        & (Board.run_id == run_id)
    )

    if existing:
        board = existing
    else:
        # Create new board
        storage_path = f"users/{owner.username}/{project_name}/{run_id}"
        board = Board.create(
            run_id=run_id,
            name=run_id,  # Will be updated when metadata syncs
            project_name=project_name,
            owner=owner,
            private=True,  # Default to private
            config="{}",
            storage_path=storage_path,
            backend="hybrid",  # Use hybrid backend for new boards
        )
        logger_api.info(f"Created new board: {board.id} (owner: {owner.username})")

    # Get storage path
    base_dir = Path(cfg.app.board_data_dir)
    board_dir = base_dir / board.storage_path

    # Ensure directories exist
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "data").mkdir(exist_ok=True)
    (board_dir / "media").mkdir(exist_ok=True)

    return board, board_dir


@router.post("/projects/{project_name}/runs/{run_id}/log")
async def sync_logs_incremental(
    project_name: str,
    run_id: str,
    request: LogSyncRequest = Body(...),
    current_user: User = Depends(get_current_user),
) -> LogSyncResponse:
    """Incremental log sync endpoint (v0.3.0+)

    Receives batched logs and writes to hybrid storage.
    Returns list of missing media files that need upload.

    Args:
        project_name: Project name
        run_id: Run ID
        request: Log sync request with steps, scalars, media, tables, histograms
        current_user: Authenticated user

    Returns:
        LogSyncResponse with status and missing media hashes

    Raises:
        HTTPException: 400/401/403/404 on error
    """
    if cfg.app.mode != "remote":
        raise HTTPException(
            400,
            detail={"error": "Incremental sync only available in remote mode"},
        )

    logger_api.info(
        f"Incremental sync: {project_name}/{run_id} "
        f"(steps {request.sync_range.start_step}-{request.sync_range.end_step})"
    )

    # Get or create board
    board, board_dir = _get_or_create_board(project_name, run_id, current_user)

    # Write data using hybrid storage
    try:
        storage = HybridStorage(board_dir / "data", logger=logger_api)

        # Group scalars by step for batch writing
        scalars_by_step = {}
        for metric_name, points in request.scalars.items():
            for point in points:
                if point.step not in scalars_by_step:
                    scalars_by_step[point.step] = {
                        "global_step": None,
                        "timestamp_ms": None,
                        "metrics": {},
                    }
                scalars_by_step[point.step]["metrics"][metric_name] = point.value

        # Merge step info from request.steps into scalars_by_step
        for step_data in request.steps:
            if step_data.step in scalars_by_step:
                scalars_by_step[step_data.step]["global_step"] = step_data.global_step
                scalars_by_step[step_data.step]["timestamp_ms"] = step_data.timestamp
            else:
                # Step with no scalars, still need to record it
                scalars_by_step[step_data.step] = {
                    "global_step": step_data.global_step,
                    "timestamp_ms": step_data.timestamp,
                    "metrics": {},
                }

        # Write scalars with step info
        for step, data in scalars_by_step.items():
            # Convert timestamp_ms to datetime for append_metrics
            if data["timestamp_ms"]:
                timestamp_obj = datetime.fromtimestamp(
                    data["timestamp_ms"] / 1000.0, tz=timezone.utc
                )
            else:
                timestamp_obj = datetime.now(timezone.utc)

            # Only call append_metrics if there are actual metrics
            if data["metrics"]:
                storage.append_metrics(
                    step=step,
                    global_step=data["global_step"],
                    metrics=data["metrics"],
                    timestamp=timestamp_obj,
                )
            else:
                # Just record step info without metrics
                timestamp_ms = (
                    data["timestamp_ms"]
                    if data["timestamp_ms"]
                    else int(timestamp_obj.timestamp() * 1000)
                )
                storage.metadata_storage.append_step_info(
                    step=step,
                    global_step=data["global_step"],
                    timestamp=timestamp_ms,
                )

        # Write media metadata (convert to media_list format expected by SQLite)
        for media_data in request.media:
            media_list = [
                {
                    "media_hash": media_data.media_hash,
                    "format": media_data.format,
                    "type": media_data.type,
                    "size_bytes": media_data.size_bytes or 0,
                    "width": media_data.width,
                    "height": media_data.height,
                }
            ]
            storage.append_media(
                step=media_data.step,
                global_step=media_data.global_step,
                name=media_data.name,
                media_list=media_list,
                caption=media_data.caption,
            )

        # Write tables
        for table_data in request.tables:
            table_dict = {
                "columns": table_data.columns,
                "column_types": table_data.column_types,
                "rows": table_data.rows,
            }
            storage.append_table(
                step=table_data.step,
                global_step=table_data.global_step,
                name=table_data.name,
                table_data=table_dict,
            )

        # Write histograms
        for hist_data in request.histograms:
            storage.append_histogram(
                step=hist_data.step,
                global_step=hist_data.global_step,
                name=hist_data.name,
                bins=hist_data.bins,
                counts=hist_data.counts,
                precision=hist_data.precision,
            )

        # Flush all buffers to disk
        storage.flush_all()

        # Save metadata.json if provided
        if request.metadata:
            metadata_path = board_dir / "metadata.json"
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(json.dumps(request.metadata, indent=2))
            logger_api.debug(f"Saved metadata.json to {metadata_path}")

            # Update board name from metadata if available
            if "name" in request.metadata:
                board.name = request.metadata["name"]
            if "config" in request.metadata:
                board.config = json.dumps(request.metadata["config"])

        # Append log lines if provided
        if request.log_lines:
            log_dir = board_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "output.log"
            async with aiofiles.open(log_file, "a") as f:
                for line in request.log_lines:
                    await f.write(line + "\n")
            logger_api.debug(
                f"Appended {len(request.log_lines)} log lines to {log_file}"
            )

        # Check which media files we don't have in SQLite KV
        media_kv_path = board_dir / "media" / "blobs.db"
        missing_media = []

        # If KV database doesn't exist yet, all media is missing
        if not media_kv_path.exists():
            missing_media = [media_data.media_hash for media_data in request.media]
        else:
            # KV database exists, check which media we have
            media_kv = SQLiteKVStorage(media_kv_path, readonly=True, logger=logger_api)
            try:
                for media_data in request.media:
                    key = f"{media_data.media_hash}.{media_data.format}"
                    if not media_kv.exists(key):
                        missing_media.append(media_data.media_hash)
            finally:
                media_kv.close()

        # Update board metadata
        board.last_synced_at = datetime.now(timezone.utc)
        board.updated_at = datetime.now(timezone.utc)
        board.save()

        logger_api.info(
            f"Incremental sync completed: {len(request.steps)} steps, "
            f"{len(request.scalars)} metrics, "
            f"{len(request.media)} media, "
            f"{len(missing_media)} missing media files"
        )

        return LogSyncResponse(
            status="synced",
            synced_range=request.sync_range,
            missing_media=missing_media,
        )

    except Exception as e:
        logger_api.error(f"Incremental sync failed: {e}")
        raise HTTPException(
            500,
            detail={"error": f"Sync failed: {str(e)}"},
        )


@router.post("/projects/{project_name}/runs/{run_id}/media")
async def upload_media_files(
    project_name: str,
    run_id: str,
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
) -> MediaUploadResponse:
    """Upload media files by hash (v0.3.0+)

    Receives media files named as {media_hash}.{format} and saves them
    to the board's media directory. Skips files that already exist.

    Args:
        project_name: Project name
        run_id: Run ID
        files: List of uploaded files (must be named {hash}.{ext})
        current_user: Authenticated user

    Returns:
        MediaUploadResponse with upload statistics

    Raises:
        HTTPException: 400/401/403/404 on error
    """
    if cfg.app.mode != "remote":
        raise HTTPException(
            400,
            detail={"error": "Media upload only available in remote mode"},
        )

    logger_api.info(f"Media upload: {project_name}/{run_id} ({len(files)} files)")

    # Get board
    board, board_dir = _get_or_create_board(project_name, run_id, current_user)

    # Initialize SQLite KV storage for media
    media_kv_path = board_dir / "media" / "blobs.db"
    media_kv = SQLiteKVStorage(media_kv_path, readonly=False, logger=logger_api)

    uploaded_hashes = []
    skipped_count = 0

    try:
        for file in files:
            # Validate filename format
            if not file.filename or "." not in file.filename:
                logger_api.warning(f"Invalid media filename: {file.filename}")
                continue

            # Extract hash from filename
            media_hash = file.filename.rsplit(".", 1)[0]
            key = file.filename  # Key is {media_hash}.{format}

            # Check if already exists in SQLite KV
            if media_kv.exists(key):
                logger_api.debug(f"Media already exists in SQLite KV: {file.filename}")
                skipped_count += 1
                continue

            # Read file content and store in SQLite KV
            content = await file.read()
            media_kv.put(key, content)

            uploaded_hashes.append(media_hash)
            logger_api.debug(
                f"Uploaded media to SQLite KV: {file.filename} ({len(content)} bytes)"
            )
    finally:
        # Close SQLite KV connection
        media_kv.close()

    # Update board
    board.updated_at = datetime.now(timezone.utc)
    board.save()

    logger_api.info(
        f"Media upload completed: {len(uploaded_hashes)} uploaded, "
        f"{skipped_count} skipped"
    )

    return MediaUploadResponse(
        status="uploaded",
        uploaded_count=len(uploaded_hashes),
        uploaded_hashes=uploaded_hashes,
        skipped_count=skipped_count,
    )
