"""Sync API endpoints for uploading boards to remote server"""

import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import orjson
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile

from kohakuboard.api.sync_models import (
    LogSyncRequest,
    LogSyncResponse,
    MediaUploadResponse,
    SyncRange,
)
from kohakuboard.auth import get_current_user
from kohakuboard.client.storage.hybrid import HybridStorage
from kohakuboard.config import cfg
from kohakuboard.db import Board, User
from kohakuboard.db_operations import get_organization, get_user_organization
from kohakuboard.logger import logger_api

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

    # Save media files
    media_dir = run_dir / "media"
    media_dir.mkdir(exist_ok=True)

    logger_api.info(f"Saving {len(media_files)} media files")
    for media_file in media_files:
        media_path = media_dir / media_file.filename
        logger_api.debug(f"Saving media file: {media_file.filename}")
        content = await media_file.read()
        async with aiofiles.open(media_path, "wb") as f:
            await f.write(content)
        total_size += len(content)

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
        storage = HybridStorage(board_dir / "data")

        # Write steps
        for step_data in request.steps:
            storage.write_step(
                step=step_data.step,
                global_step=step_data.global_step,
                timestamp=step_data.timestamp,
            )

        # Write scalars (to Lance)
        for metric_name, points in request.scalars.items():
            for point in points:
                storage.write_metric(
                    step=point.step,
                    metric=metric_name,
                    value=point.value,
                )

        # Write media metadata (to SQLite)
        for media_data in request.media:
            storage.write_media_metadata(
                media_hash=media_data.media_hash,
                format=media_data.format,
                step=media_data.step,
                global_step=media_data.global_step,
                name=media_data.name,
                caption=media_data.caption,
                media_type=media_data.type,
                size_bytes=media_data.size_bytes,
                width=media_data.width,
                height=media_data.height,
            )

        # Write tables (to SQLite)
        for table_data in request.tables:
            storage.write_table(
                step=table_data.step,
                global_step=table_data.global_step,
                name=table_data.name,
                columns=table_data.columns,
                column_types=table_data.column_types,
                rows=table_data.rows,
            )

        # Write histograms (to Lance)
        for hist_data in request.histograms:
            storage.write_histogram(
                step=hist_data.step,
                global_step=hist_data.global_step,
                name=hist_data.name,
                bins=hist_data.bins,
                counts=hist_data.counts,
                precision=hist_data.precision,
            )

        # Flush to disk
        storage.flush()

        # Check which media files we don't have
        media_dir = board_dir / "media"
        missing_media = []
        for media_data in request.media:
            filename = f"{media_data.media_hash}.{media_data.format}"
            if not (media_dir / filename).exists():
                missing_media.append(media_data.media_hash)

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

    media_dir = board_dir / "media"
    media_dir.mkdir(exist_ok=True)

    uploaded_hashes = []
    skipped_count = 0

    for file in files:
        # Validate filename format
        if not file.filename or "." not in file.filename:
            logger_api.warning(f"Invalid media filename: {file.filename}")
            continue

        # Extract hash from filename
        media_hash = file.filename.rsplit(".", 1)[0]

        # Check if already exists
        filepath = media_dir / file.filename
        if filepath.exists():
            logger_api.debug(f"Media file already exists: {file.filename}")
            skipped_count += 1
            continue

        # Save file
        content = await file.read()
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(content)

        uploaded_hashes.append(media_hash)
        logger_api.debug(f"Uploaded media: {file.filename} ({len(content)} bytes)")

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
