"""Sync API endpoints for uploading boards to remote server"""

import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from kohakuboard.auth import get_current_user
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
