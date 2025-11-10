"""Sync API endpoints for uploading boards to remote server"""

import asyncio
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import orjson
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from kohakuvault import KVault

from kohakuboard.storage.hybrid import HybridStorage
from kohakuboard_server.api.sync_models import (
    LogSyncRequest,
    LogSyncResponse,
    MediaUploadResponse,
    SyncRange,
)
from kohakuboard_server.auth import get_current_user
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
    logger_api.info(f"start sync_run {project_name}")

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

    board, run_dir, created = await asyncio.to_thread(
        _get_or_create_board, project_name, run_id, current_user
    )
    project_name = board.project_name

    duckdb_bytes = await duckdb_file.read()
    media_payloads = []
    for media_file in media_files:
        media_payloads.append((media_file.filename, await media_file.read()))

    total_size = await asyncio.to_thread(
        _write_full_sync_run,
        board,
        run_dir,
        duckdb_bytes,
        media_payloads,
        meta,
        name,
        private,
        config,
        created,
    )

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
) -> tuple[Board, Path, bool]:
    """Get existing board or create new one

    Args:
        project_name: Project name (may include org prefix)
        run_id: Run ID
        current_user: Authenticated user

    Returns:
        Tuple of (Board, storage_path, created_flag)
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

    created = False
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
        )
        logger_api.info(f"Created new board: {board.id} (owner: {owner.username})")
        created = True

    # Get storage path
    base_dir = Path(cfg.app.board_data_dir)
    board_dir = base_dir / board.storage_path

    # Ensure directories exist
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "data").mkdir(exist_ok=True)
    (board_dir / "media").mkdir(exist_ok=True)

    return board, board_dir, created


def _write_full_sync_run(
    board: Board,
    run_dir: Path,
    duckdb_bytes: bytes,
    media_payloads: list[tuple[str | None, bytes]],
    metadata: dict,
    name: str,
    private: bool,
    config: dict,
    created: bool,
) -> int:
    """Blocking helper to write full sync payload to disk."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = data_dir / "board.duckdb"

    logger_api.info(f"Saving DuckDB file to: {duckdb_path}")
    with open(duckdb_path, "wb") as f:
        f.write(duckdb_bytes)
    total_size = len(duckdb_bytes)

    media_dir = run_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media_kv_path = media_dir / "blobs.db"
    media_kv = KVault(str(media_kv_path))

    logger_api.info(f"Saving {len(media_payloads)} media files to KVault")
    try:
        with media_kv.cache(64 * 1024 * 1024):
            for filename, content in media_payloads:
                if not filename:
                    continue
                key = filename
                logger_api.debug(f"Saving media to KVault: {filename}")
                media_kv[key] = content
                total_size += len(content)
    finally:
        media_kv.close()

    metadata_to_save = dict(metadata)
    metadata_to_save.pop("run_id", None)
    metadata_to_save["board_id"] = run_dir.name
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_to_save, f, indent=2)

    if created:
        board.name = name
        board.private = private
        board.config = json.dumps(config)

    board.total_size_bytes = total_size
    board.last_synced_at = datetime.now(timezone.utc)
    board.updated_at = datetime.now(timezone.utc)
    board.save()

    return total_size


def _process_incremental_sync(
    board: Board, board_dir: Path, request: LogSyncRequest
) -> tuple[dict[str, str], list[str]]:
    """Blocking helper to persist incremental sync payload."""
    storage = HybridStorage(board_dir / "data", logger=logger_api)

    scalars_by_step: dict[int, dict] = {}
    for metric_name, points in request.scalars.items():
        for point in points:
            if point.step not in scalars_by_step:
                scalars_by_step[point.step] = {
                    "global_step": None,
                    "timestamp_ms": None,
                    "metrics": {},
                }
            scalars_by_step[point.step]["metrics"][metric_name] = point.value

    for step_data in request.steps:
        container = scalars_by_step.setdefault(
            step_data.step,
            {
                "global_step": None,
                "timestamp_ms": None,
                "metrics": {},
            },
        )
        container["global_step"] = step_data.global_step
        container["timestamp_ms"] = step_data.timestamp

    for step, data in scalars_by_step.items():
        metrics = data["metrics"]
        if not metrics and data["timestamp_ms"] is None:
            continue
        timestamp_ms = data["timestamp_ms"]
        timestamp = (
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            if timestamp_ms
            else None
        )
        storage.append_metrics(
            step=step,
            global_step=data["global_step"],
            metrics=metrics,
            timestamp=timestamp or datetime.now(timezone.utc),
        )

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

    for hist_data in request.histograms:
        storage.append_histogram(
            step=hist_data.step,
            global_step=hist_data.global_step,
            name=hist_data.name,
            bins=hist_data.bins,
            counts=hist_data.counts,
            precision=hist_data.precision,
        )

    for kde_data in request.kernel_density:
        try:
            payload_bytes = base64.b64decode(kde_data.payload)
        except Exception as exc:
            logger_api.warning(
                f"Failed to decode kernel density payload {kde_data.name}: {exc}"
            )
            continue

        storage.append_kernel_density(
            step=kde_data.step,
            global_step=kde_data.global_step,
            name=kde_data.name,
            payload=payload_bytes,
            kde_meta=kde_data.kde_meta or {},
        )

    for tensor_data in request.tensors:
        try:
            payload_bytes = base64.b64decode(tensor_data.payload)
        except Exception as exc:
            logger_api.warning(
                f"Failed to decode tensor payload {tensor_data.name}: {exc}"
            )
            continue

        storage.append_tensor(
            step=tensor_data.step,
            global_step=tensor_data.global_step,
            name=tensor_data.name,
            payload=payload_bytes,
            tensor_meta=tensor_data.tensor_meta or {},
        )

    storage.flush_all()

    if request.metadata:
        metadata_to_save = dict(request.metadata)
        metadata_to_save.pop("run_id", None)
        metadata_to_save["board_id"] = board_dir.name
        metadata_path = board_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_to_save, f, indent=2)
        logger_api.debug(f"Saved metadata.json to {metadata_path}")

        if "name" in metadata_to_save:
            board.name = metadata_to_save["name"]
        if "config" in metadata_to_save:
            board.config = json.dumps(metadata_to_save["config"])

    log_dir = board_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    server_log_hashes = {}
    for log_name in [
        "output.log",
        "board.log",
        "writer.log",
        "sync_worker.log",
        "storage.log",
    ]:
        log_file = log_dir / log_name
        if log_file.exists():
            try:
                content = log_file.read_bytes()
                server_log_hashes[log_name] = hashlib.sha256(content).hexdigest()
            except Exception:
                pass

    media_kv_path = board_dir / "media" / "blobs.db"
    missing_media: list[str] = []
    if not media_kv_path.exists():
        missing_media = [media_data.media_hash for media_data in request.media]
    else:
        media_kv = KVault(str(media_kv_path))
        try:
            for media_data in request.media:
                key = f"{media_data.media_hash}.{media_data.format}"
                if key not in media_kv:
                    missing_media.append(media_data.media_hash)
        finally:
            media_kv.close()

    board.last_synced_at = datetime.now(timezone.utc)
    board.updated_at = datetime.now(timezone.utc)
    board.save()

    return server_log_hashes, missing_media


def _store_media_files(
    board: Board, board_dir: Path, payloads: list[tuple[str | None, bytes]]
) -> tuple[list[str], int]:
    """Blocking helper that persists media blobs to KVault."""
    media_kv_path = board_dir / "media" / "blobs.db"
    media_kv = KVault(str(media_kv_path))

    uploaded_hashes: list[str] = []
    skipped_count = 0

    try:
        with media_kv.cache(64 * 1024 * 1024):
            for filename, content in payloads:
                if not filename or "." not in filename:
                    logger_api.warning(f"Invalid media filename: {filename}")
                    continue

                key = filename
                media_hash = filename.rsplit(".", 1)[0]

                if key in media_kv:
                    logger_api.debug(f"Media already exists in KVault: {filename}")
                    skipped_count += 1
                    continue

                media_kv[key] = content
                uploaded_hashes.append(media_hash)
                logger_api.debug(
                    f"Uploaded media to KVault: {filename} ({len(content)} bytes)"
                )
    finally:
        media_kv.close()

    board.updated_at = datetime.now(timezone.utc)
    board.save()

    return uploaded_hashes, skipped_count


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
    logger_api.info(
        f"start incremental_sync {project_name}/{run_id} "
        f"(steps {request.sync_range.start_step}-{request.sync_range.end_step})"
    )

    if cfg.app.mode != "remote":
        raise HTTPException(
            400,
            detail={"error": "Incremental sync only available in remote mode"},
        )

    logger_api.info(
        f"Incremental sync: {project_name}/{run_id} "
        f"(steps {request.sync_range.start_step}-{request.sync_range.end_step})"
    )

    # Get or create board without blocking
    board, board_dir, _ = await asyncio.to_thread(
        _get_or_create_board, project_name, run_id, current_user
    )

    try:
        server_log_hashes, missing_media = await asyncio.to_thread(
            _process_incremental_sync, board, board_dir, request
        )

        logger_api.info(
            f"Incremental sync completed: {len(request.steps)} steps, "
            f"{len(request.scalars)} metrics, "
            f"{len(request.media)} media, "
            f"{len(request.histograms)} histograms, "
            f"{len(request.kernel_density)} kernel density entries, "
            f"{len(request.tensors)} tensors, "
            f"{len(missing_media)} missing media files"
        )

        return LogSyncResponse(
            status="synced",
            synced_range=request.sync_range,
            missing_media=missing_media,
            log_hashes=server_log_hashes,
        )

    except Exception as e:
        logger_api.error(f"Incremental sync failed: {e}")
        raise HTTPException(
            500,
            detail={"error": f"Sync failed: {str(e)}"},
        )


@router.put("/projects/{project_name}/runs/{run_id}/logs/{log_filename}")
async def upload_log_file(
    project_name: str,
    run_id: str,
    log_filename: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Upload a log file (binary endpoint)

    Args:
        project_name: Project name (may include org/)
        run_id: Run ID
        log_filename: Log file name (output.log, board.log, etc.)
        request: Raw request with binary body
        current_user: Authenticated user

    Returns:
        Success status with hash
    """
    logger_api.info(f"start upload_log_file {project_name}/{run_id}/{log_filename}")

    # Validate log filename (security)
    allowed_logs = [
        "output.log",
        "board.log",
        "writer.log",
        "sync_worker.log",
        "storage.log",
    ]
    if log_filename not in allowed_logs:
        raise HTTPException(
            400,
            detail={"error": f"Invalid log filename: {log_filename}"},
        )

    logger_api.info(f"Log upload: {project_name}/{run_id}/{log_filename}")

    # Get or create board
    board, board_dir, _ = await asyncio.to_thread(
        _get_or_create_board, project_name, run_id, current_user
    )

    # Read binary body
    content_bytes = await request.body()

    # Write to log file (binary mode)
    log_dir = board_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / log_filename

    async with aiofiles.open(log_file, "wb") as f:
        await f.write(content_bytes)

    # Calculate hash for verification
    file_hash = hashlib.sha256(content_bytes).hexdigest()

    logger_api.debug(
        f"Uploaded log file: {log_filename} ({len(content_bytes)} bytes, hash: {file_hash[:8]}...)"
    )

    return {
        "status": "success",
        "filename": log_filename,
        "size": len(content_bytes),
        "hash": file_hash,
    }


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
    logger_api.info(f"start upload_media {project_name}/{run_id} ({len(files)} files)")

    if cfg.app.mode != "remote":
        raise HTTPException(
            400,
            detail={"error": "Media upload only available in remote mode"},
        )

    logger_api.info(f"Media upload: {project_name}/{run_id} ({len(files)} files)")

    # Get board without blocking
    board, board_dir, _ = await asyncio.to_thread(
        _get_or_create_board, project_name, run_id, current_user
    )

    payloads: list[tuple[str | None, bytes]] = []
    for file in files:
        payloads.append((file.filename, await file.read()))

    uploaded_hashes, skipped_count = await asyncio.to_thread(
        _store_media_files, board, board_dir, payloads
    )

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
