"""Project management API endpoints (Local Mode)"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from kohakuboard.utils.board_reader import list_boards
from kohakuboard.config import cfg
from kohakuboard.logger import logger_api

router = APIRouter()


def fetchProjectRuns(project_name: str):
    """Fetch project runs in local mode.

    Args:
        project_name: Project name (must be "local")

    Returns:
        dict with project info and runs list
    """
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


@router.get("/projects")
async def list_projects():
    """List projects in local mode

    Returns single "local" project.

    Returns:
        dict: {"projects": [...]}
    """
    # Local mode: single "local" project
    base_dir = Path(cfg.app.board_data_dir)
    run_count = len(
        [d for d in base_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
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


@router.get("/projects/{project_name}/runs")
async def list_runs(project_name: str):
    """List runs within a project in local mode

    Args:
        project_name: Project name (must be "local")

    Returns:
        dict: {"project": ..., "runs": [...]}
    """
    logger_api.info(f"Listing runs for project: {project_name}")
    return fetchProjectRuns(project_name)
