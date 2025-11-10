"""Project management API endpoints (Local Mode)"""

from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from kohakuboard.utils.board_reader import DEFAULT_LOCAL_PROJECT, list_boards
from kohakuboard.config import cfg
from kohakuboard.logger import logger_api


router = APIRouter()


def _group_boards_by_project(base_dir: Path):
    boards = list_boards(base_dir)
    projects = defaultdict(list)
    for board in boards:
        project = board.get("project") or DEFAULT_LOCAL_PROJECT
        projects[project].append(board)
    return projects


def fetch_project_runs(project_name: str):
    """Fetch project runs in local mode."""
    base_dir = Path(cfg.app.board_data_dir)
    projects = _group_boards_by_project(base_dir)

    if project_name not in projects:
        raise HTTPException(404, detail={"error": "Project not found"})

    runs = []
    for board in sorted(
        projects[project_name], key=lambda b: b.get("created_at") or "", reverse=True
    ):
        run_id = board.get("run_id") or board["board_id"]
        annotation = board.get("annotation")
        if annotation is None and "_" in board["board_id"]:
            _, annotation = board["board_id"].split("_", 1)
        runs.append(
            {
                "run_id": run_id,
                "annotation": annotation,
                "name": board["name"],
                "created_at": board["created_at"],
                "updated_at": board.get("updated_at"),
                "config": board.get("config", {}),
                "project": project_name,
            }
        )

    return {
        "project": project_name,
        "runs": runs,
    }


@router.get("/projects")
async def list_projects():
    """List projects in local mode"""
    base_dir = Path(cfg.app.board_data_dir)
    projects = _group_boards_by_project(base_dir)

    project_list = []
    for name, boards in projects.items():
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

    # If no runs found, still expose legacy "local" project for UX parity
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


@router.get("/projects/{project_name}/runs")
async def list_runs(project_name: str):
    """List runs within a project in local mode"""
    logger_api.info(f"Listing runs for project: {project_name}")
    return fetch_project_runs(project_name)
