"""Project management API endpoints"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from peewee import fn

from kohakuboard.utils.board_reader import (
    BoardReader,
    DEFAULT_LOCAL_PROJECT,
    list_boards,
)
from kohakuboard_server.auth import get_optional_user
from kohakuboard_server.config import cfg
from kohakuboard_server.db import Board, User
from kohakuboard_server.logger import logger_api
from kohakuboard.utils.datetime_utils import safe_isoformat

router = APIRouter()


def _group_boards_by_project(base_dir: Path):
    boards = list_boards(base_dir)
    projects = defaultdict(list)
    for board in boards:
        project = board.get("project") or DEFAULT_LOCAL_PROJECT
        projects[project].append(board)
    return projects


def _fetch_project_runs_sync(project_name: str, current_user: User | None):
    if cfg.app.mode == "local":
        base_dir = Path(cfg.app.board_data_dir)
        projects = _group_boards_by_project(base_dir)

        if project_name not in projects:
            raise HTTPException(404, detail={"error": "Project not found"})

        runs = []
        for board in sorted(
            projects[project_name],
            key=lambda b: b.get("created_at") or "",
            reverse=True,
        ):
            runs.append(
                {
                    "run_id": board["board_id"],
                    "name": board["name"],
                    "created_at": board["created_at"],
                    "updated_at": board.get("updated_at"),
                    "config": board.get("config", {}),
                    "project": project_name,
                }
            )

        return {"project": project_name, "runs": runs}

    # Remote mode: require authentication
    if not current_user:
        raise HTTPException(401, detail={"error": "Authentication required"})

    base_dir = Path(cfg.app.board_data_dir)

    runs_query = (
        Board.select()
        .where((Board.owner == current_user) & (Board.project_name == project_name))
        .order_by(Board.created_at.desc())
    )

    runs = []
    for run in runs_query:
        updated_at = safe_isoformat(run.updated_at)
        try:
            board_path = base_dir / run.storage_path
            if board_path.exists():
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


async def fetchProjectRuns(project_name: str, current_user: User | None):
    """Helper to fetch project runs based on mode.

    Args:
        project_name: Project name
        current_user: Current user (optional)

    Returns:
        dict with project info and runs list
    """
    return await asyncio.to_thread(_fetch_project_runs_sync, project_name, current_user)


@router.get("/projects")
async def list_projects(current_user: User | None = Depends(get_optional_user)):
    """List projects accessible to current user

    Local mode: Returns single "local" project
    Remote mode: Returns user's projects (authenticated) or empty list (anonymous)

    Returns:
        dict: {"projects": [...]}
    """
    if cfg.app.mode == "local":
        base_dir = Path(cfg.app.board_data_dir)
        grouped = await asyncio.to_thread(_group_boards_by_project, base_dir)

        project_list = []
        for name, boards in grouped.items():
            created_times = [b.get("created_at") for b in boards if b.get("created_at")]
            updated_times = [b.get("updated_at") for b in boards if b.get("updated_at")]

            display_name = "Default Project" if name == DEFAULT_LOCAL_PROJECT else name
            project_list.append(
                {
                    "name": name,
                    "display_name": display_name,
                    "run_count": len(boards),
                    "created_at": min(created_times) if created_times else None,
                    "updated_at": max(updated_times) if updated_times else None,
                }
            )

        if not project_list:
            project_list.append(
                {
                    "name": DEFAULT_LOCAL_PROJECT,
                    "display_name": "Default Project",
                    "run_count": 0,
                    "created_at": None,
                    "updated_at": None,
                }
            )

        return {"projects": sorted(project_list, key=lambda p: p["name"])}

    else:  # remote mode
        if not current_user:
            # Anonymous: no projects
            return {"projects": []}

        def _list_remote_projects(user: User):
            query = (
                Board.select(
                    Board.project_name,
                    fn.COUNT(Board.id).alias("run_count"),
                    fn.MIN(Board.created_at).alias("created_at"),
                    fn.MAX(Board.updated_at).alias("updated_at"),
                )
                .where(Board.owner == user)
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
            return projects

        projects = await asyncio.to_thread(_list_remote_projects, current_user)
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
    return await fetchProjectRuns(project_name, current_user)
