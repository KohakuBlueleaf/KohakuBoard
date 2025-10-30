"""Project management API endpoints"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from peewee import fn

from kohakuboard.api.utils.board_reader import list_boards
from kohakuboard.auth import get_optional_user
from kohakuboard.config import cfg
from kohakuboard.db import Board, User
from kohakuboard.logger import logger_api
from kohakuboard.utils.datetime_utils import safe_isoformat

router = APIRouter()


def fetchProjectRuns(project_name: str, current_user: User | None):
    """Helper to fetch project runs based on mode.

    Args:
        project_name: Project name
        current_user: Current user (optional)

    Returns:
        dict with project info and runs list
    """
    if cfg.app.mode == "local":
        # Local mode: project must be "local"
        if project_name != "local":
            raise HTTPException(404, detail={"error": "Project not found"})

        # List all runs in base_dir
        base_dir = Path(cfg.app.board_data_dir)
        boards = list_boards(base_dir)

        return {
            "project": "local",
            "runs": [
                {
                    "run_id": board["board_id"],
                    "name": board["name"],
                    "created_at": board["created_at"],
                    "updated_at": board.get("updated_at"),
                    "config": board.get("config", {}),
                }
                for board in boards
            ],
        }
    else:
        # Remote mode: require authentication
        if not current_user:
            raise HTTPException(401, detail={"error": "Authentication required"})

        # Query runs from DB for current user
        runs_query = (
            Board.select()
            .where((Board.owner == current_user) & (Board.project_name == project_name))
            .order_by(Board.created_at.desc())
        )

        runs = []
        for run in runs_query:
            # Get actual updated_at from latest step in board storage
            updated_at = safe_isoformat(run.updated_at)
            try:
                board_path = Path(cfg.app.board_data_dir) / run.storage_path
                if board_path.exists():
                    from kohakuboard.api.utils.board_reader import BoardReader

                    reader = BoardReader(board_path)
                    latest_step = reader.get_latest_step()
                    if latest_step and latest_step.get("timestamp"):
                        updated_at = latest_step["timestamp"]
            except Exception as e:
                logger_api.debug(f"Failed to get latest step for {run.run_id}: {e}")

            runs.append(
                {
                    "run_id": run.run_id,
                    "name": run.name,
                    "private": run.private,
                    "created_at": safe_isoformat(run.created_at),
                    "updated_at": updated_at,
                    "last_synced_at": safe_isoformat(run.last_synced_at),
                    "total_size": run.total_size_bytes,
                    "config": json.loads(run.config) if run.config else {},
                }
            )

        return {
            "project": project_name,
            "owner": current_user.username,
            "runs": runs,
        }


@router.get("/projects")
async def list_projects(current_user: User | None = Depends(get_optional_user)):
    """List projects accessible to current user

    Local mode: Returns single "local" project
    Remote mode: Returns user's projects (authenticated) or empty list (anonymous)

    Returns:
        dict: {"projects": [...]}
    """
    if cfg.app.mode == "local":
        # Local mode: single "local" project
        base_dir = Path(cfg.app.board_data_dir)
        run_count = len(
            [
                d
                for d in base_dir.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()
            ]
        )

        return {
            "projects": [
                {
                    "name": "local",
                    "display_name": "Local Boards",
                    "run_count": run_count,
                    "created_at": None,
                    "updated_at": None,
                }
            ]
        }

    else:  # remote mode
        if not current_user:
            # Anonymous: no projects
            return {"projects": []}

        # Query user's projects from DB
        query = (
            Board.select(
                Board.project_name,
                fn.COUNT(Board.id).alias("run_count"),
                fn.MIN(Board.created_at).alias("created_at"),
                fn.MAX(Board.updated_at).alias("updated_at"),
            )
            .where(Board.owner == current_user)
            .group_by(Board.project_name)
            .order_by(Board.project_name)
        )

        projects = []
        for project in query:
            projects.append(
                {
                    "name": project.project_name,
                    "display_name": project.project_name.replace("-", " ").title(),
                    "run_count": project.run_count,
                    "created_at": safe_isoformat(project.created_at),
                    "updated_at": safe_isoformat(project.updated_at),
                }
            )

        return {"projects": projects}


@router.get("/projects/{project_name}/runs")
async def list_runs(
    project_name: str,
    current_user: User | None = Depends(get_optional_user),
):
    """List runs within a project

    Args:
        project_name: Project name
        current_user: Current user (optional)

    Returns:
        dict: {"project": ..., "runs": [...], "owner": ...}
    """
    logger_api.info(f"Listing runs for project: {project_name}")
    return fetchProjectRuns(project_name, current_user)
